"""
PointMassNavigate — DiffPhysDrone v1.0-style navigation task for a simplified
point-mass drone, implemented in JAX.

The drone is modelled as a 3-D point mass with:
  • Action low-pass filter (control delay)
  • Body-frame quadratic + linear aerodynamic drag
  • Ornstein–Uhlenbeck wind disturbance
  • Attitude tracked via thrust direction + velocity-prediction blend

Task: fly from a random start to a random goal while avoiding obstacles.
The policy receives velocity commands (tracks a target speed toward the goal),
matching DiffPhysDrone's velocity-tracking training objective.

State (dict of JAX arrays — vmappable pytree):
    p               (3,)          position (world frame)
    v               (3,)          velocity (world frame)
    a               (3,)          filtered action / acceleration command
    dg              (3,)          Ornstein–Uhlenbeck disturbance
    R               (3,3)         rotation matrix, columns = [fwd, left, up]
    target_pos      (3,)          goal position
    max_speed       ()            episode max speed (m/s)
    pitch_ctl_delay ()            control-delay stiffness
    yaw_ctl_delay   ()            yaw-dynamics stiffness
    drag_2          (2,)          [quadratic, linear] drag coefficients
    drone_radius    ()            effective collision radius (m)
    thr_est_error   ()            thrust-estimation error multiplier
    scene           (scene_dim,)  flat obstacle array (from SceneConfig.sample)

Observation: (depth_img [12×16 float32], obs_vec [10-D float32])
    obs_vec = [local_v(3), target_v_local(3), up_world(3), safety_margin(1)]

Raw policy output (6-D, body frame):
    [:3] → a_pred (acceleration command) — rotated to world by caller
    [3:] → v_pred (velocity prediction)  — rotated to world by caller
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "depth_render"))

import jax
import jax.numpy as jnp

from .dynamics import dynamics_step, update_attitude
from .scene    import SceneConfig

from renderer   import render_depth, apply_sensor_noise
from primitives import (
    point_sphere_dist, point_aabb_dist,
    point_capsule_dist, point_plane_dist,
)


# ---------------------------------------------------------------------------
# Rotation-matrix → quaternion  [w, x, y, z]
# Shepperd method — handles all quadrants via jnp.where
# ---------------------------------------------------------------------------

def _rotmat_to_quat(R):
    """(3,3) rotation matrix → unit quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    # Case 1: trace > 0
    s1 = 0.5 / jnp.sqrt(jnp.maximum(trace + 1.0, 1e-8))
    q1 = jnp.array([0.25 / s1,
                    (R[2, 1] - R[1, 2]) * s1,
                    (R[0, 2] - R[2, 0]) * s1,
                    (R[1, 0] - R[0, 1]) * s1])

    # Case 2: R[0,0] largest diagonal
    s2 = 2.0 * jnp.sqrt(jnp.maximum(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-8))
    q2 = jnp.array([(R[2, 1] - R[1, 2]) / s2,
                    0.25 * s2,
                    (R[0, 1] + R[1, 0]) / s2,
                    (R[0, 2] + R[2, 0]) / s2])

    # Case 3: R[1,1] largest diagonal
    s3 = 2.0 * jnp.sqrt(jnp.maximum(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-8))
    q3 = jnp.array([(R[0, 2] - R[2, 0]) / s3,
                    (R[0, 1] + R[1, 0]) / s3,
                    0.25 * s3,
                    (R[1, 2] + R[2, 1]) / s3])

    # Case 4: R[2,2] largest diagonal
    s4 = 2.0 * jnp.sqrt(jnp.maximum(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-8))
    q4 = jnp.array([(R[1, 0] - R[0, 1]) / s4,
                    (R[0, 2] + R[2, 0]) / s4,
                    (R[1, 2] + R[2, 1]) / s4,
                    0.25 * s4])

    q = jnp.where(trace > 0, q1,
        jnp.where(R[0, 0] > R[1, 1],
            jnp.where(R[0, 0] > R[2, 2], q2, q4),
            jnp.where(R[1, 1] > R[2, 2], q3, q4)))

    return q / (jnp.linalg.norm(q) + 1e-8)


# ---------------------------------------------------------------------------
# Smooth-L1 loss
# ---------------------------------------------------------------------------

def _smooth_l1(x, beta=1.0):
    return jnp.where(jnp.abs(x) < beta,
                     0.5 * x ** 2 / beta,
                     jnp.abs(x) - 0.5 * beta)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PointMassNavigate:
    """
    DiffPhysDrone-style point-mass navigation environment.

    Construct once; use reset() + step() inside jax.vmap + jax.lax.scan.
    """

    # --- Dimensions ----------------------------------------------------------
    obs_dim: int   = 10      # obs_vec only (depth handled separately)
    act_dim: int   = 6       # 3-D a_pred + 3-D v_pred (both body-frame raw)
    dt:      float = 1.0 / 15.0

    # --- Depth camera ---------------------------------------------------------
    cam_width:         int   = 64
    cam_height:        int   = 48
    cam_pool:          int   = 4      # max-pool factor → (12, 16) input to CNN
    cam_angle_deg:     float = 20.0   # downward pitch from drone forward axis
    cam_min_range:     float = 0.2
    cam_max_range:     float = 24.0
    cam_quantization_m: float = 0.001
    fov_x_deg:         float = 79.5   # ≈ 2·atan(0.82), matching DiffPhysDrone

    # --- Episode randomisation ------------------------------------------------
    max_speed_min:         float = 0.75
    max_speed_max:         float = 2.5
    pitch_ctl_delay_mean:  float = 12.0
    pitch_ctl_delay_std:   float = 1.2
    yaw_ctl_delay_mean:    float = 6.0
    yaw_ctl_delay_std:     float = 0.6
    drone_radius_min:      float = 0.10
    drone_radius_max:      float = 0.15
    thr_est_error_std:     float = 0.01
    drag_quad_min:         float = 0.30
    drag_quad_max:         float = 0.45

    # --- Gradient decay -------------------------------------------------------
    grad_decay: float = 0.4

    # --- Loss coefficients (DiffPhysDrone single_agent.args) -----------------
    coef_v:             float = 1.0
    coef_v_pred:        float = 2.0
    coef_collide:       float = 7.5
    coef_obj_avoidance: float = 3.0
    coef_d_acc:         float = 0.01
    coef_d_jerk:        float = 0.001

    # --- Initial state & target spawn ----------------------------------------
    init_pos_std:  float = 0.2
    init_vel_std:  float = 0.2
    target_x_min:  float = 8.0
    target_x_max:  float = 10.0
    target_yz_rng: float = 3.0

    # -------------------------------------------------------------------------

    def __init__(self, scene=None, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter '{k}'")
            setattr(self, k, v)

        if scene is None:
            self.scene_cfg = SceneConfig()
        elif isinstance(scene, dict):
            self.scene_cfg = SceneConfig(**scene)
        else:
            self.scene_cfg = scene

        # Pre-compute constants used inside jitted functions
        self._gd_factor = float(self.grad_decay ** self.dt)
        theta = float(self.cam_angle_deg * 3.14159265 / 180.0)
        import math
        self._cam_cos = math.cos(theta)
        self._cam_sin = math.sin(theta)

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(self, key):
        """
        Sample a random episode.

        Returns:
            obs   : (depth_img [12,16], obs_vec [10])
            state : dict of JAX arrays (see module docstring)
            info  : {}
        """
        keys = jax.random.split(key, 11)
        ki = iter(keys)

        # Episode-level random parameters
        max_speed    = jax.random.uniform(next(ki), minval=self.max_speed_min,    maxval=self.max_speed_max)
        drone_radius = jax.random.uniform(next(ki), minval=self.drone_radius_min, maxval=self.drone_radius_max)
        pitch_ctl_delay = (self.pitch_ctl_delay_mean
                           + self.pitch_ctl_delay_std * jax.random.normal(next(ki)))
        yaw_ctl_delay   = (self.yaw_ctl_delay_mean
                           + self.yaw_ctl_delay_std * jax.random.normal(next(ki)))
        drag_quad    = jax.random.uniform(next(ki), minval=self.drag_quad_min, maxval=self.drag_quad_max)
        drag_2       = jnp.array([drag_quad, 0.0])
        thr_est_error = 1.0 + self.thr_est_error_std * jax.random.normal(next(ki))

        # Initial drone state
        p_xy = self.init_pos_std * jax.random.normal(next(ki), shape=(2,))
        p    = jnp.array([p_xy[0], p_xy[1], -1.5])
        v  = self.init_vel_std * jax.random.normal(next(ki), shape=(3,))
        a  = jnp.zeros(3)
        dg = jnp.zeros(3)

        # Target position
        tx = jax.random.uniform(next(ki), minval=self.target_x_min, maxval=self.target_x_max)
        ty = jax.random.uniform(next(ki), minval=-self.target_yz_rng, maxval=self.target_yz_rng)
        target_pos = jnp.array([tx, ty, -1.5])

        # Initial attitude: up = world-z, forward = toward target (horizontal)
        to_tgt     = target_pos - p
        to_tgt_h   = to_tgt.at[2].set(0.0)
        fwd        = to_tgt_h / (jnp.linalg.norm(to_tgt_h) + 1e-6)
        up         = jnp.array([0.0, 0.0, 1.0])
        fwd        = fwd - jnp.dot(fwd, up) * up
        fwd        = fwd / (jnp.linalg.norm(fwd) + 1e-6)
        left       = jnp.cross(up, fwd)
        R          = jnp.stack([fwd, left, up], axis=1)

        # Obstacles
        scene = self.scene_cfg.sample(next(ki))

        state = {
            "p":               p,
            "v":               v,
            "a":               a,
            "dg":              dg,
            "R":               R,
            "target_pos":      target_pos,
            "max_speed":       max_speed,
            "pitch_ctl_delay": pitch_ctl_delay,
            "yaw_ctl_delay":   yaw_ctl_delay,
            "drag_2":          drag_2,
            "drone_radius":    drone_radius,
            "thr_est_error":   thr_est_error,
            "scene":           scene,
        }
        return self.get_obs(state), state, {}

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def step(self, state, a_pred, v_pred, base_key, step_idx):
        """
        One simulation step.

        Args:
            state     : environment state dict
            a_pred    : (3,) acceleration command, world frame
            v_pred    : (3,) velocity prediction, world frame
            base_key  : PRNGKey for this episode's noise stream
            step_idx  : int / 0-D array — current step index for fold_in

        Returns:
            new_state : updated state dict
            step_data : dict with quantities needed for loss computation
        """
        step_key = jax.random.fold_in(base_key, step_idx)

        a_cmd_world = a_pred * state["thr_est_error"]

        p_next, v_next, a_filt, dg_next = dynamics_step(
            state["p"], state["v"], state["a"], state["dg"], state["R"],
            a_cmd_world, step_key,
            state["pitch_ctl_delay"], state["drag_2"], 1.0,
            self._gd_factor, self.dt,
        )

        R_next = update_attitude(
            state["R"], a_filt, v_pred,
            state["yaw_ctl_delay"], self.dt,
        )

        new_state = {**state, "p": p_next, "v": v_next, "a": a_filt,
                     "dg": dg_next, "R": R_next}

        # Target velocity: fly toward goal, capped at max_speed
        to_target = state["target_pos"] - p_next
        dist_to_target = jnp.sqrt(jnp.dot(to_target, to_target) + 1e-8)
        target_v_world = (jnp.minimum(dist_to_target, state["max_speed"])
                          * to_target / dist_to_target)

        obs_dist = self._get_nearest_obstacle_dist(new_state)

        step_data = {
            "v":              v_next,
            "target_v_world": target_v_world,
            "dist":           obs_dist,
            "a_cmd":          a_filt,
            "v_pred":         v_pred,
            "drone_radius":   state["drone_radius"],
        }
        return new_state, step_data

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------

    def get_obs(self, state):
        """
        Returns (depth_img [12,16], obs_vec [10]).

        obs_vec = [local_v(3), target_v_local(3), up_world(3), margin(1)]
        """
        depth_img = jax.lax.stop_gradient(self._get_processed_depth(state))

        R = state["R"]
        local_v = R.T @ state["v"]

        to_target      = state["target_pos"] - state["p"]
        dist_to_target = jnp.sqrt(jnp.dot(to_target, to_target) + 1e-8)
        target_v_world = (jnp.minimum(dist_to_target, state["max_speed"])
                          * to_target / dist_to_target)
        target_v_local = R.T @ target_v_world

        up_world = R[:, 2]

        dist   = self._get_nearest_obstacle_dist(state)
        margin = jnp.array([dist - state["drone_radius"]])

        obs_vec = jnp.concatenate([local_v, target_v_local, up_world, margin])
        return depth_img, obs_vec

    # -----------------------------------------------------------------------
    # Depth rendering
    # -----------------------------------------------------------------------

    def _get_camera_quat(self, R):
        """Apply downward pitch and convert to quaternion for the renderer."""
        c, s = self._cam_cos, self._cam_sin
        cam_fwd  =  c * R[:, 0] - s * R[:, 2]
        cam_left =  R[:, 1]
        cam_up   =  s * R[:, 0] + c * R[:, 2]
        R_cam    = jnp.stack([cam_fwd, cam_left, cam_up], axis=1)
        return _rotmat_to_quat(R_cam)

    def _get_processed_depth(self, state):
        raw   = self._get_depth(state)
        normd = 3.0 / jnp.clip(raw, 0.3, self.cam_max_range) - 0.6
        # 4×4 max-pool: (48, 64) → (12, 16)
        return jax.lax.reduce_window(
            normd, -jnp.inf, jax.lax.max,
            window_dimensions=(4, 4), window_strides=(4, 4), padding="VALID",
        )

    def _get_depth(self, state):
        arrays = self.scene_cfg.unpack(state["scene"])
        quat   = self._get_camera_quat(state["R"])
        return apply_sensor_noise(
            render_depth(
                position         = state["p"],
                quaternion       = quat,
                fov_deg          = self.fov_x_deg,
                width            = self.cam_width,
                height           = self.cam_height,
                sphere_centers   = arrays["sphere_centers"],
                sphere_radii     = arrays["sphere_radii"],
                box_centers      = arrays["box_centers"],
                box_half_extents = arrays["box_half_extents"],
                capsule_centers  = arrays["capsule_centers"],
                capsule_axes     = arrays["capsule_axes"],
                capsule_hh       = arrays["capsule_hh"],
                capsule_radii    = arrays["capsule_radii"],
            ),
            min_range       = self.cam_min_range,
            max_range       = self.cam_max_range,
            quantization_m  = self.cam_quantization_m,
        )

    # -----------------------------------------------------------------------
    # Collision detection
    # -----------------------------------------------------------------------

    def _get_nearest_obstacle_dist(self, state):
        """Signed distance to the nearest obstacle surface (negative = inside)."""
        pos    = state["p"]
        arrays = self.scene_cfg.unpack(state["scene"])
        dist   = jnp.inf

        # Ground plane at z = -1 (normal pointing up)
        dist = jnp.minimum(dist,
            point_plane_dist(pos,
                jnp.array([0.0, 0.0, -1.0]),
                jnp.array([0.0, 0.0,  1.0])))

        if arrays["sphere_centers"].shape[0] > 0:
            ds = jax.vmap(lambda c, r: point_sphere_dist(pos, c, r))(
                arrays["sphere_centers"], arrays["sphere_radii"])
            dist = jnp.minimum(dist, jnp.min(ds))

        if arrays["box_centers"].shape[0] > 0:
            ds = jax.vmap(lambda c, he: point_aabb_dist(pos, c, he))(
                arrays["box_centers"], arrays["box_half_extents"])
            dist = jnp.minimum(dist, jnp.min(ds))

        if arrays["capsule_centers"].shape[0] > 0:
            ds = jax.vmap(lambda c, ax, hh, r: point_capsule_dist(pos, c, ax, hh, r))(
                arrays["capsule_centers"], arrays["capsule_axes"],
                arrays["capsule_hh"], arrays["capsule_radii"])
            dist = jnp.minimum(dist, jnp.min(ds))

        return dist

    # -----------------------------------------------------------------------
    # Trajectory loss  (called by APG after full rollout scan)
    # -----------------------------------------------------------------------

    def compute_loss(self, traj):
        """
        Multi-component DiffPhysDrone-style loss from trajectory data.

        traj: dict of (batch, horizon, ...) arrays stacked by vmap+scan.

        Returns: (total_loss, mean_return)  — mean_return = -total_loss
        """
        v            = traj["v"]              # (B, T, 3)
        target_v     = traj["target_v_world"] # (B, T, 3)
        dist         = traj["dist"]           # (B, T)
        a_cmd        = traj["a_cmd"]          # (B, T, 3)
        v_pred       = traj["v_pred"]         # (B, T, 3)
        drone_radius = traj["drone_radius"]   # (B, T)

        # -- Velocity tracking: smooth-L1 on mean trajectory velocity ---------
        v_avg   = v.mean(axis=1)        # (B, 3)
        tv_avg  = target_v.mean(axis=1) # (B, 3)
        diff_v  = v_avg - tv_avg
        loss_v  = jnp.mean(_smooth_l1(jnp.sqrt(jnp.sum(diff_v ** 2, axis=-1) + 1e-8)))

        # -- Velocity prediction: MSE against actual (stop-gradient) ----------
        loss_v_pred = jnp.mean((v_pred - jax.lax.stop_gradient(v)) ** 2)

        # -- Collision barriers ------------------------------------------------
        # Rate of approach from consecutive distance samples
        dist_diff = jnp.diff(dist, axis=1)                    # (B, T-1)
        v_to_pt   = jax.lax.stop_gradient(jnp.clip(-dist_diff / self.dt, 1.0, None)) # (B, T-1)
        clearance = dist[:, 1:] - drone_radius[:, 1:]          # (B, T-1)

        # Softplus: exponential penalty when drone body is inside obstacle
        loss_collide = jnp.mean(jax.nn.softplus(-32.0 * clearance) * v_to_pt)

        # Quadratic barrier: penalises approaching within 1 m of collision
        loss_obj  = jnp.mean(v_to_pt * jax.nn.relu(1.0 - clearance) ** 2)

        # -- Control regularisation -------------------------------------------
        loss_d_acc  = jnp.mean(a_cmd ** 2)
        jerk        = jnp.diff(a_cmd, axis=1) / self.dt       # (B, T-1, 3)
        loss_d_jerk = jnp.mean(jerk ** 2)

        total = (
              self.coef_v             * loss_v
            + self.coef_v_pred        * loss_v_pred
            + self.coef_collide       * loss_collide
            + self.coef_obj_avoidance * loss_obj
            + self.coef_d_acc         * loss_d_acc
            + self.coef_d_jerk        * loss_d_jerk
        )
        return total, -total
