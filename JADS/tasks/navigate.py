import os, sys, yaml
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "depth_render"))

import jax
import jax.numpy as jnp
from ..drone_physics.dynamics import forward_euler, semi_implicit_euler, rk4, _gdecay
from ..drone_physics.morphology import morphology, PROP_DIAMETER
from ..drone_physics.quat_math import euler_to_quat, quat_to_euler, quat_to_rotmat
from ..scene.scene import SceneConfig

from JADS.depth_render.renderer import render_depth, apply_sensor_noise  # depth_render/renderer.py
from JADS.depth_render.primitives import (                               # depth_render/primitives.py
    point_sphere_dist, point_aabb_dist, point_obb_dist,
    point_capsule_dist, point_plane_dist,
)


class Navigate:
    """
    Multicopter navigation task: fly from a random start to a random target
    while avoiding randomly generated obstacles.

    Scene geometry (target + obstacles) is sampled fresh every reset() and
    stored in the state array so jax.vmap gives each parallel rollout its own
    independent environment.

    State layout (flat float32 array of length state_dim):
        [0:19]         drone state  (pos, vel, quat, omega, W)
        [19:22]        target pos
        [22:]          scene array  (SceneConfig.sample — see scene.py)
                         static mode:     full obstacle geometry  (scene_dim floats)
                         procedural mode: single float32 episode seed  (1 float)

    Observation: plain drone obs (pos, vel, euler, omega, W) — 18 dims.
    Depth-image obs will be added once the renderer is integrated.

    Scene config can be passed as:
        Navigate(scene=SceneConfig(...))
        Navigate(scene={"obstacle_density": 0.2, ...})   ← dict from YAML
        Navigate()                                        ← SceneConfig defaults
    """

    # ---- Drone --------------------------------------------------------
    obs_dim: int = 18
    act_dim: int = 6
    dt: float = 0.02 # 1.0/15.0

    # ---- Gradient decay -----------------------------------------------
    grad_decay: float = 0.4

    # ---- Collision loss -----------------------------------------------
    b1: float = 1.0
    b2: float = 32.0
    motor_collision_radius: float = PROP_DIAMETER / 2  # ~0.038 m sphere per motor

    # ---- Loss weights -------------------------------------------------
    xy_weight:         float = 1.0
    z_weight:          float = 2.0
    vel_weight:        float = 0.1
    rate_weight:       float = 0.01
    collision_weight:  float = 7.5
    obj_weight:        float = 3.0
    heading_weight:    float = 1.0

    # ---- Depth camera ------------------------------------------------------
    cam_fov_deg:       float = 90.0
    cam_width:         int   = 64
    cam_height:        int   = 48
    cam_pool:          int   = 4      # max-pool factor → (12, 16) input to CNN
    cam_min_range:     float = 0.2
    cam_max_range:     float = 8.0
    cam_quantization_m: float = 0.001
    cam_hz:            float = 50.0   # depth-camera update rate; defaults to 1/dt (no frame skip)

    # ---- Morphology --------------------------------------------------------
    l_min:     float = 0.06;        l_max:     float = 0.14;        l_default:     float = 0.10
    # l_min:     float = 0.06;        l_max:     float = 0.15;        l_default:     float = 0.10
    psi_min:   float = -jnp.pi/9;   psi_max:   float = jnp.pi/9;    psi_default:   float = 0.0
    # theta_min: float = -jnp.pi / 6; theta_max: float = jnp.pi / 6;  theta_default: float = 0.0
    theta_min: float = -jnp.pi / 4; theta_max: float = jnp.pi / 4;  theta_default: float = 0.0
    phi_min:   float = -jnp.pi / 2; phi_max:   float = jnp.pi / 2;  phi_default:   float = 0.0
    alpha_min: float = -jnp.pi/2;         alpha_max: float = jnp.pi / 2;  alpha_default: float = 0.0
    # alpha_min: float = 0.0;         alpha_max: float = jnp.pi / 2;  alpha_default: float = 0.0
    alternating_phi: bool = False
    train_morphology:  bool = False
    morph_init_scale:  float = 0.0  # std of Normal noise added to raw pre-sigmoid values; 0 = deterministic
    integrator: str = "rk4"

    # ---- Drone initial state bounds -----------------------------------------
    init_x_min:     float = -1.0;  init_x_max:     float = 1.0
    init_y_min:     float = -1.0;  init_y_max:     float = 1.0
    init_z_min:     float = -1.0;  init_z_max:     float = 1.0
    init_vx_min:    float = -0.5;  init_vx_max:    float = 0.5
    init_vy_min:    float = -0.5;  init_vy_max:    float = 0.5
    init_vz_min:    float = -0.5;  init_vz_max:    float = 0.5
    init_roll_min:  float = -jnp.pi/6; init_roll_max:  float = jnp.pi/6
    init_pitch_min: float = -jnp.pi/6; init_pitch_max: float = jnp.pi/6
    init_yaw_min:   float = -jnp.pi/6; init_yaw_max:   float = jnp.pi/6
    init_wx_min:    float = -0.3;  init_wx_max:    float = 0.3
    init_wy_min:    float = -0.3;  init_wy_max:    float = 0.3
    init_wz_min:    float = -0.3;  init_wz_max:    float = 0.3
    init_W_min:     float = -1.0;  init_W_max:     float = 1.0

    # ---- Target spawn bounds -----------------------------------------------
    target_x_min: float = 3.0;  target_x_max: float = 5.0
    target_y_min: float = -1.0; target_y_max: float = 1.0
    target_z_min: float = -1.0; target_z_max: float = 1.0

    # -----------------------------------------------------------------------

    def __init__(self, scene=None, depth_camera=None, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter '{k}'")
            setattr(self, k, v)

        if scene is None:
            self.scene_cfg = SceneConfig()
        elif isinstance(scene, dict):
            self.scene_cfg = SceneConfig(**scene)
        elif isinstance(scene, str):
            with open(scene) as f:
                self.scene_cfg = SceneConfig(**yaml.safe_load(f))
        else:
            self.scene_cfg = scene

        if depth_camera is not None:
            dc = depth_camera
            self.cam_fov_deg        = float(dc.get("fov_deg",        self.cam_fov_deg))
            self.cam_width          = int(  dc.get("width",          self.cam_width))
            self.cam_height         = int(  dc.get("height",         self.cam_height))
            self.cam_min_range      = float(dc.get("min_range",      self.cam_min_range))
            self.cam_max_range      = float(dc.get("max_range",      self.cam_max_range))
            self.cam_quantization_m = float(dc.get("quantization_m", self.cam_quantization_m))
            self.cam_hz             = float(dc.get("cam_hz",         self.cam_hz))

        if self.scene_cfg.procedural:
            self.scene_cfg.cell_size = self.cam_max_range

        self.state_dim = 22 + self.scene_cfg.scene_dim
        self._gd_factor = float(self.grad_decay ** self.dt)

        # Depth camera runs slower than the physics/policy loop (dt): render a
        # fresh frame every `frame_skip` steps and hold it (zero-order hold)
        # on the steps in between.
        self.frame_skip = max(1, round(1.0 / (self.dt * self.cam_hz)))

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(self, key: jax.Array) -> tuple:
        """
        Sample a random episode: drone start state, target position, and scene.

        Returns:
            obs:   (obs_dim,)   — plain drone obs
            state: (state_dim,) — drone + target + scene array
            info:  {}
        """
        key, drone_key, target_key, scene_key = jax.random.split(key, 4)

        # ---- Drone initial state -------------------------------------------
        keys = jax.random.split(drone_key, 13)
        k = 0
        x     = jax.random.uniform(keys[k], minval=self.init_x_min,     maxval=self.init_x_max);     k+=1
        y     = jax.random.uniform(keys[k], minval=self.init_y_min,     maxval=self.init_y_max);     k+=1
        z     = jax.random.uniform(keys[k], minval=self.init_z_min,     maxval=self.init_z_max);     k+=1
        vx    = jax.random.uniform(keys[k], minval=self.init_vx_min,    maxval=self.init_vx_max);    k+=1
        vy    = jax.random.uniform(keys[k], minval=self.init_vy_min,    maxval=self.init_vy_max);    k+=1
        vz    = jax.random.uniform(keys[k], minval=self.init_vz_min,    maxval=self.init_vz_max);    k+=1
        roll  = jax.random.uniform(keys[k], minval=self.init_roll_min,  maxval=self.init_roll_max);  k+=1
        pitch = jax.random.uniform(keys[k], minval=self.init_pitch_min, maxval=self.init_pitch_max); k+=1
        yaw   = jax.random.uniform(keys[k], minval=self.init_yaw_min,   maxval=self.init_yaw_max);   k+=1
        wx    = jax.random.uniform(keys[k], minval=self.init_wx_min,    maxval=self.init_wx_max);    k+=1
        wy    = jax.random.uniform(keys[k], minval=self.init_wy_min,    maxval=self.init_wy_max);    k+=1
        wz    = jax.random.uniform(keys[k], minval=self.init_wz_min,    maxval=self.init_wz_max);    k+=1
        W     = jax.random.uniform(keys[k], shape=(6,), minval=self.init_W_min, maxval=self.init_W_max)

        quat       = euler_to_quat(roll, pitch, yaw)
        drone_state = jnp.concatenate([
            jnp.array([x, y, z]),
            jnp.array([vx, vy, vz]),
            quat,
            jnp.array([wx, wy, wz]),
            W,
        ])  # (19,)

        # ---- Target --------------------------------------------------------
        tkeys = jax.random.split(target_key, 3)
        target = jnp.array([
            jax.random.uniform(tkeys[0], minval=self.target_x_min, maxval=self.target_x_max),
            jax.random.uniform(tkeys[1], minval=self.target_y_min, maxval=self.target_y_max),
            jax.random.uniform(tkeys[2], minval=self.target_z_min, maxval=self.target_z_max),
        ])

        # ---- Scene (obstacles) --------------------------------------------
        scene_array = self.scene_cfg.sample(scene_key)  # (scene_dim,)

        state = jnp.concatenate([drone_state, target, scene_array])
        return self._get_obs(state), state, {}

    # -----------------------------------------------------------------------
    # Observation & scene extraction
    # -----------------------------------------------------------------------

    def _get_obs(self, state: jnp.ndarray, arrays: dict = None,
                 step_idx: jnp.ndarray = None, prev_depth: jnp.ndarray = None) -> jnp.ndarray:
        """
        Args:
            step_idx, prev_depth: pass both together to enable asynchronous
                depth rendering — a fresh frame is only rendered when
                `(step_idx + 1) % self.frame_skip == 0`; on the steps in
                between, `prev_depth` (the caller's previous depth_map) is
                held and returned unchanged (zero-order hold). Leave both
                None (default) to always render, e.g. on reset().
        """
        rel_pos = state[0:3] - state[19:22]
        euler = jax.lax.stop_gradient(quat_to_euler(state[6:10]))
        drone_states = jnp.concatenate([rel_pos, state[3:6], euler, state[10:13], state[13:19]])

        if step_idx is not None and self.frame_skip > 1:
            should_render = (step_idx + 1) % self.frame_skip == 0
            depth_map = jax.lax.cond(
                should_render,
                lambda: jax.lax.stop_gradient(self._get_processed_depth(state, arrays=arrays)),
                lambda: prev_depth,
            )
        else:
            depth_map = jax.lax.stop_gradient(self._get_processed_depth(state, arrays=arrays))
        return (depth_map, drone_states)
    
    def _unpack_scene(self, state: jnp.ndarray) -> dict:
        """Extract obstacle geometry from state, handling both scene modes."""
        if self.scene_cfg.procedural:
            (sphere_centers, sphere_radii,
             box_centers, box_half_extents,
             cap_centers, cap_axes, cap_hh, cap_radii,
             obb_centers, obb_quats, obb_he) = self.scene_cfg.get_local_obstacles(
                state[0:3], state[22]
            )
            return jax.lax.stop_gradient({
                "sphere_centers":    sphere_centers,
                "sphere_radii":      sphere_radii,
                "box_centers":       box_centers,
                "box_half_extents":  box_half_extents,
                "cylinder_centers":  cap_centers,
                "cylinder_axes":     cap_axes,
                "cylinder_hh":       cap_hh,
                "cylinder_radii":    cap_radii,
                "obb_centers":       obb_centers,
                "obb_quats":         obb_quats,
                "obb_half_extents":  obb_he,
            })
        return self.scene_cfg.unpack(state[22:])

    def _get_depth(self, state: jnp.ndarray, arrays: dict = None) -> jnp.ndarray:
        """
        Render a depth image from the drone's perspective.

        Args:
            state:  (state_dim,) — full environment state vector.
            arrays: pre-unpacked scene geometry (see _unpack_scene). Pass this
                    in when the caller already unpacked the scene for `state`,
                    to avoid re-sampling the procedural obstacle neighbourhood.

        Returns:
            (cam_height, cam_width) float32 — depth in metres.
            0 = closer than cam_min_range.
            cam_max_range = no-hit or saturated.
        """
        arrays = arrays if arrays is not None else self._unpack_scene(state)
        return apply_sensor_noise(
            render_depth(
                position         = state[0:3],
                quaternion       = state[6:10],
                fov_deg          = self.cam_fov_deg,
                width            = self.cam_width,
                height           = self.cam_height,
                sphere_centers   = arrays["sphere_centers"],
                sphere_radii     = arrays["sphere_radii"],
                box_centers      = arrays["box_centers"],
                box_half_extents = arrays["box_half_extents"],
                cylinder_centers = arrays["cylinder_centers"],
                cylinder_axes    = arrays["cylinder_axes"],
                cylinder_hh      = arrays["cylinder_hh"],
                cylinder_radii   = arrays["cylinder_radii"],
                obb_centers      = arrays["obb_centers"],
                obb_quaternions  = arrays["obb_quats"],
                obb_half_extents = arrays["obb_half_extents"],
            ),
            min_range      = self.cam_min_range,
            max_range      = self.cam_max_range,
            quantization_m = self.cam_quantization_m,
        )

    def get_vis_depth(self, state: jnp.ndarray, width: int = 320, height: int = 240) -> jnp.ndarray:
        """Render depth at arbitrary resolution for visualization (no pooling, no normalization)."""
        arrays = self._unpack_scene(state)
        return apply_sensor_noise(
            render_depth(
                position         = state[0:3],
                quaternion       = state[6:10],
                fov_deg          = self.cam_fov_deg,
                width            = width,
                height           = height,
                sphere_centers   = arrays["sphere_centers"],
                sphere_radii     = arrays["sphere_radii"],
                box_centers      = arrays["box_centers"],
                box_half_extents = arrays["box_half_extents"],
                cylinder_centers = arrays["cylinder_centers"],
                cylinder_axes    = arrays["cylinder_axes"],
                cylinder_hh      = arrays["cylinder_hh"],
                cylinder_radii   = arrays["cylinder_radii"],
                obb_centers      = arrays["obb_centers"],
                obb_quaternions  = arrays["obb_quats"],
                obb_half_extents = arrays["obb_half_extents"],
            ),
            min_range      = self.cam_min_range,
            max_range      = self.cam_max_range,
            quantization_m = self.cam_quantization_m,
        )

    def _get_processed_depth(self, state, arrays: dict = None):
        raw   = self._get_depth(state, arrays=arrays)
        normd = 3.0 / jnp.clip(raw, 0.3, self.cam_max_range) - 0.6
        # 4×4 max-pool: (48, 64) → (12, 16)
        return jax.lax.reduce_window(
            normd, -jnp.inf, jax.lax.max,
            window_dimensions=(4, 4), window_strides=(4, 4), padding="VALID",
        )
    
    def _get_nearest_obstacle_dist(self, state, motor_positions_world=None, arrays: dict = None):
        """Signed distance to the nearest obstacle surface (negative = inside).

        If motor_positions_world (6, 3) is provided, the effective distance is
        min(body_center_dist, min_i(motor_dist_i - motor_collision_radius)).

        `arrays` (pre-unpacked scene geometry) can be passed in to avoid
        re-sampling the procedural obstacle neighbourhood when the caller
        already unpacked the scene for `state`.
        """
        pos    = state[0:3]
        arrays = arrays if arrays is not None else self._unpack_scene(state)

        def _point_dist(pt):
            d = point_plane_dist(pt, jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, -1.0]))
            if arrays["sphere_centers"].shape[0] > 0:
                ds = jax.vmap(lambda c, r: point_sphere_dist(pt, c, r))(
                    arrays["sphere_centers"], arrays["sphere_radii"])
                d = jnp.minimum(d, jnp.min(ds))
            if arrays["box_centers"].shape[0] > 0:
                ds = jax.vmap(lambda c, he: point_aabb_dist(pt, c, he))(
                    arrays["box_centers"], arrays["box_half_extents"])
                d = jnp.minimum(d, jnp.min(ds))
            if arrays["cylinder_centers"].shape[0] > 0:
                ds = jax.vmap(lambda c, ax, hh, r: point_capsule_dist(pt, c, ax, hh, r))(
                    arrays["cylinder_centers"], arrays["cylinder_axes"],
                    arrays["cylinder_hh"], arrays["cylinder_radii"])
                d = jnp.minimum(d, jnp.min(ds))
            if arrays["obb_centers"].shape[0] > 0:
                ds = jax.vmap(lambda c, q, he: point_obb_dist(pt, c, q, he))(
                    arrays["obb_centers"], arrays["obb_quats"], arrays["obb_half_extents"])
                d = jnp.minimum(d, jnp.min(ds))
            return d

        dist = _point_dist(pos)

        if motor_positions_world is not None:
            motor_dists = jax.vmap(_point_dist)(motor_positions_world) - self.motor_collision_radius
            dist = jnp.minimum(dist, jnp.min(motor_dists))

        return dist

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def compute_morphology(self, morph_params: dict = None):
        """
        Compute (Bf, Bm, m, J, J_inv, motor_pos_body) from morph_params.

        Pure function of morph_params (or, when untrained, the env's fixed
        default angles) — constant across an entire rollout. Callers running
        a multi-step rollout (e.g. lax.scan over step()) should call this
        once and pass the result to every step() via `morph_matrices` instead
        of letting step() recompute it every call.
        """
        if self.train_morphology and morph_params is not None:
            l = self.get_l(morph_params)
            psi = self.get_psi(morph_params)
            theta = self.get_theta(morph_params)
            phi = self.get_phi(morph_params)
            alpha = self.get_alpha(morph_params)
        else:
            l     = jnp.full(3, self.l_default)
            psi   = jnp.full(3, self.psi_default)
            theta = jnp.full(3, self.theta_default)
            if self.alternating_phi:
                phi = jnp.array([self.phi_default, -self.phi_default, self.phi_default])
            else:
                phi = jnp.full(3, self.phi_default)
            alpha  = jnp.full(3, self.alpha_default)

        return morphology(l, psi, theta, phi, alpha)

    def step(self, state: jnp.ndarray, action: jnp.ndarray, morph_params: dict = None,
             morph_matrices: tuple = None,
             step_idx: jnp.ndarray = None, prev_depth: jnp.ndarray = None) -> tuple:
        """
        step_idx, prev_depth: see `_get_obs` — pass both to hold the depth
        image across steps and only re-render every `frame_skip` steps.
        """
        # ---- Morphology ----------------------------------------------------
        if morph_matrices is None:
            morph_matrices = self.compute_morphology(morph_params)
        Bf, Bm, m, J, J_inv, motor_pos_body = morph_matrices
        U = jnp.clip(action, -1.0, 1.0)

        integrators = {"euler": forward_euler, "semi_implicit_euler": semi_implicit_euler, "rk4": rk4}
        next_drone   = integrators[self.integrator](_gdecay(state[0:19], self._gd_factor), U, Bf, Bm, m, J, J_inv, self.dt)

        # Renormalize quaternion, clip velocities
        quat_norm = jnp.maximum(jnp.linalg.norm(next_drone[6:10]), 1e-8)
        next_drone = next_drone.at[6:10].set(next_drone[6:10] / quat_norm)
        next_drone = next_drone.at[3:6].set(jnp.clip(next_drone[3:6],   -20.0, 20.0))
        next_drone = next_drone.at[10:13].set(jnp.clip(next_drone[10:13], -20.0, 20.0))

        # Target + scene are frozen; append unchanged
        next_state = jnp.concatenate([next_drone, state[19:]])

        R_sg            = jax.lax.stop_gradient(quat_to_rotmat(next_drone[6:10]))
        motor_pos_world = next_drone[0:3] + motor_pos_body @ R_sg.T  # (6, 3) world frame

        arrays = self._unpack_scene(next_state)
        dist = self._get_nearest_obstacle_dist(next_state, motor_pos_world, arrays=arrays)
        step_data = {
            "pos":        next_state[0:3],
            "vel":        next_state[3:6],
            "quat":       next_state[6:10],
            "target_pos": next_state[19:22],
            "omega":      next_state[10:13],
            "dist":       dist,
        }
        obs = self._get_obs(next_state, arrays=arrays, step_idx=step_idx, prev_depth=prev_depth)
        return next_state, obs, step_data

    def compute_loss(self, traj):
        """
        Trajectory loss from a full rollout.

        traj: dict of (batch, horizon, ...) arrays stacked by vmap+scan.

        Returns: (total_loss, mean_return) — mean_return = -total_loss
        """
        pos    = traj["pos"]        # (B, T, 3)
        vel    = traj["vel"]        # (B, T, 3)
        quat   = traj["quat"]       # (B, T, 4) — [w, x, y, z]
        target = traj["target_pos"] # (B, T, 3)
        omega  = traj["omega"]      # (B, T, 3)
        dist   = traj["dist"]       # (B, T)

        diff = pos - target
        loss_xy   = jnp.mean(jnp.sum(diff[..., :2] ** 2, axis=-1))
        loss_z    = jnp.mean(diff[..., 2] ** 2)
        loss_vel  = jnp.mean(jnp.sum(vel   ** 2, axis=-1))
        loss_rate = jnp.mean(jnp.sum(omega ** 2, axis=-1))

        # Heading alignment: body x-axis (forward) should point toward target
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        fwd = jnp.stack([
            1.0 - 2.0 * (y**2 + z**2),
            2.0 * (x*y + z*w),
            2.0 * (x*z - y*w),
        ], axis=-1)  # (B, T, 3) — body x-axis in world frame

        to_target = target - pos
        to_target = to_target / jnp.sqrt(
            jnp.maximum(jnp.sum(to_target**2, axis=-1, keepdims=True), 1e-8)
        )

        cos_sim = jnp.sum(fwd * to_target, axis=-1)  # (B, T)
        loss_heading = jnp.mean((1.0 - cos_sim) ** 2)

        dist_diff = jnp.diff(dist, axis=1)                                           # (B, T-1)
        v_to_pt   = jax.lax.stop_gradient(jnp.clip(-dist_diff / self.dt, 1.0, None)) # (B, T-1)
        loss_collision = jnp.mean(
            (self.b1 * jax.nn.softplus(self.b2 * (-dist[:, 1:]))) * v_to_pt
        )
        loss_obj = jnp.mean(
            (jax.nn.relu(1.0 - dist[:, 1:]) ** 2) * v_to_pt
        )

        total = (self.xy_weight*loss_xy + self.z_weight*loss_z
                 + self.vel_weight*loss_vel + self.rate_weight*loss_rate
                 + self.collision_weight*loss_collision + self.obj_weight*loss_obj
                 + self.heading_weight*loss_heading)
        return total, -total



    # -----------------------------------------------------------------------
    # Morphology
    # -----------------------------------------------------------------------

    def _phi_to_raw(self, phi_val: float) -> float:
        """Inverse sigmoid: map phi value to raw logit parameter."""
        normalized = (phi_val - self.phi_min) / (self.phi_max - self.phi_min)
        normalized = jnp.clip(normalized, 1e-6, 1.0 - 1e-6)
        return jnp.log(normalized / (1.0 - normalized))

    def init_morph(self, key=None) -> dict:
        if self.alternating_phi:
            phi_raw = jnp.array([
                self._phi_to_raw( self.phi_default),
                self._phi_to_raw(-self.phi_default),
                self._phi_to_raw( self.phi_default),
            ])
        else:
            phi_raw = jnp.zeros(3)
        params = {"l_raw": jnp.zeros(3), "psi_raw": jnp.zeros(3), "theta_raw": jnp.zeros(3), "phi_raw": phi_raw, "alpha_raw": jnp.zeros(3)}
        if key is not None and self.morph_init_scale > 0.0:
            keys = jax.random.split(key, 5)
            params = {
                "l_raw":     params["l_raw"]     + jax.random.normal(keys[0], (3,)) * self.morph_init_scale,
                "psi_raw":   params["psi_raw"]   + jax.random.normal(keys[1], (3,)) * self.morph_init_scale,
                "theta_raw": params["theta_raw"] + jax.random.normal(keys[2], (3,)) * self.morph_init_scale,
                "phi_raw":   params["phi_raw"]   + jax.random.normal(keys[3], (3,)) * self.morph_init_scale,
                "alpha_raw": params["alpha_raw"] + jax.random.normal(keys[4], (3,)) * self.morph_init_scale,
            }
        return params

    def get_l(self, morph_params: dict) -> jnp.ndarray:
        return self.l_min + (self.l_max - self.l_min) * jax.nn.sigmoid(morph_params["l_raw"])
    
    def get_psi(self, morph_params: dict) -> jnp.ndarray:
        return self.psi_min + (self.psi_max - self.psi_min) * jax.nn.sigmoid(morph_params["psi_raw"])

    def get_theta(self, morph_params: dict) -> jnp.ndarray:
        return self.theta_min + (self.theta_max - self.theta_min) * jax.nn.sigmoid(morph_params["theta_raw"])

    def get_phi(self, morph_params: dict) -> jnp.ndarray:
        return self.phi_min + (self.phi_max - self.phi_min) * jax.nn.sigmoid(morph_params["phi_raw"])
    
    def get_alpha(self, morph_params: dict) -> jnp.ndarray:
        return self.alpha_min + (self.alpha_max - self.alpha_min) * jax.nn.sigmoid(morph_params["alpha_raw"])

    def get_morph_info(self, morph_params: dict) -> dict:
        l     = self.get_l(morph_params)
        psi   = self.get_psi(morph_params)
        theta = self.get_theta(morph_params)
        phi   = self.get_phi(morph_params)
        alpha = self.get_alpha(morph_params)
        return {
            "l1": float(l[0]),         "l2": float(l[1]),         "l3": float(l[2]),
            "psi1": float(psi[0]),     "psi2": float(psi[1]),     "psi3": float(psi[2]),
            "theta1": float(theta[0]), "theta2": float(theta[1]), "theta3": float(theta[2]),
            "phi1": float(phi[0]),     "phi2": float(phi[1]),     "phi3": float(phi[2]),
            "alpha1": float(alpha[0]),     "alpha2": float(alpha[1]),     "alpha3": float(alpha[2]),
        }
