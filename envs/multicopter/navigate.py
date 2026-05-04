import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "depth_render"))

import jax
import jax.numpy as jnp
from .dynamics import forward_euler, semi_implicit_euler, rk4
from .morphology import morphology
from .quat_math import euler_to_quat, quat_to_euler
from .scene import SceneConfig

from renderer import render_depth, apply_sensor_noise  # depth_render/renderer.py
from primitives import (                               # depth_render/primitives.py
    point_sphere_dist, point_aabb_dist,
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
        [22:]          scene array  (from SceneConfig.sample — see scene.py)

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
    dt: float = 0.05

    # ---- Collision loss -----------------------------------------------
    b1: float = 1.0
    b2: float = 1.0

    # ---- Depth camera ------------------------------------------------------
    cam_fov_deg:       float = 90.0
    cam_width:         int   = 64
    cam_height:        int   = 48
    cam_min_range:     float = 0.2
    cam_max_range:     float = 8.0
    cam_quantization_m: float = 0.001

    # ---- Morphology --------------------------------------------------------
    l_min: float = 0.10;  l_max: float = 0.20;  l_default: float = 0.15
    phi_min:   float = -jnp.pi / 6; phi_max:   float =  jnp.pi / 6; phi_default:   float = 0.0
    alpha_min: float = -jnp.pi / 4; alpha_max: float =  jnp.pi / 4; alpha_default: float = 0.0
    alternating_alpha: bool = True
    train_morphology:  bool = False
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

        self.state_dim = 22 + self.scene_cfg.scene_dim

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

    def _get_obs(self, state: jnp.ndarray) -> jnp.ndarray:
        rel_pos = state[0:3] - state[19:22]
        euler = quat_to_euler(state[6:10])
        drone_states = jnp.concatenate([rel_pos, state[3:6], euler, state[10:13], state[13:19]])
        depth_map = jax.lax.stop_gradient(self._get_processed_depth(state))
        return (depth_map, drone_states)
    
    def _get_depth(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Render a depth image from the drone's perspective.

        Args:
            state: (state_dim,) — full environment state vector.

        Returns:
            (cam_height, cam_width) float32 — depth in metres.
            0 = closer than cam_min_range.
            cam_max_range = no-hit or saturated.
        """
        arrays = self.scene_cfg.unpack(state[22:])
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
                box_quaternions  = arrays["box_quaternions"],
                box_half_extents = arrays["box_half_extents"],
                capsule_centers  = arrays["capsule_centers"],
                capsule_axes     = arrays["capsule_axes"],
                capsule_hh       = arrays["capsule_hh"],
                capsule_radii    = arrays["capsule_radii"],
            ),
            min_range      = self.cam_min_range,
            max_range      = self.cam_max_range,
            quantization_m = self.cam_quantization_m,
        )
    
    # def _get_processed_depth(self, state: jnp.ndarray) -> jnp.ndarray:
    #     raw_depth = self.get_depth(state)
    #     clamped = jnp.clip(raw_depth, self.cam_min_range, self.cam_max_range)
    #     inverted_depth = 1.0 / clamped
    #     min_inv = 1.0 / self.cam_max_range
    #     max_inv = 1.0 / self.cam_min_range
    #     normalized_depth = (inverted_depth - min_inv) / (max_inv - min_inv)
    #     return jax.lax.reduce_window(
    #         normalized_depth,
    #         init_value=-jnp.inf,
    #         computation=jax.lax.max,
    #         window_dimensions=(2, 2),
    #         window_strides=(2, 2),
    #         padding="VALID",
    #     )
    
    def _get_processed_depth(self, state):
        raw   = self._get_depth(state)
        normd = 3.0 / jnp.clip(raw, 0.3, self.cam_max_range) - 0.6
        # 4×4 max-pool: (48, 64) → (12, 16)
        return jax.lax.reduce_window(
            normd, -jnp.inf, jax.lax.max,
            window_dimensions=(4, 4), window_strides=(4, 4), padding="VALID",
        )
    
    def _get_nearest_obstacle_dist(self, state):
        """Signed distance to the nearest obstacle surface (negative = inside)."""
        pos    = state[0:3]
        arrays = self.scene_cfg.unpack(state[22:])
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
    # Step
    # -----------------------------------------------------------------------

    def step(self, state: jnp.ndarray, action: jnp.ndarray, morph_params: dict = None) -> tuple:
        # ---- Morphology ----------------------------------------------------
        if self.train_morphology and morph_params is not None:
            l     = self.l_min     + (self.l_max     - self.l_min)     * jax.nn.sigmoid(morph_params["l_raw"])
            phi   = self.phi_min   + (self.phi_max   - self.phi_min)   * jax.nn.sigmoid(morph_params["phi_raw"])
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * jax.nn.sigmoid(morph_params["alpha_raw"])
        else:
            l   = jnp.full(3, self.l_default)
            phi = jnp.full(3, self.phi_default)
            alpha = (
                jnp.array([self.alpha_default, -self.alpha_default, self.alpha_default])
                if self.alternating_alpha else jnp.full(3, self.alpha_default)
            )

        Bf, Bm, m, J, J_inv = morphology(l, phi, alpha)
        U = jnp.clip(action, -1.0, 1.0)

        integrators = {"euler": forward_euler, "semi_implicit_euler": semi_implicit_euler, "rk4": rk4}
        next_drone   = integrators[self.integrator](state[0:19], U, Bf, Bm, m, J, J_inv, self.dt)

        # Renormalize quaternion, clip velocities
        quat_norm = jnp.maximum(jnp.linalg.norm(next_drone[6:10]), 1e-8)
        next_drone = next_drone.at[6:10].set(next_drone[6:10] / quat_norm)
        next_drone = next_drone.at[3:6].set(jnp.clip(next_drone[3:6],   -20.0, 20.0))
        next_drone = next_drone.at[10:13].set(jnp.clip(next_drone[10:13], -20.0, 20.0))

        # Target + scene are frozen; append unchanged
        next_state = jnp.concatenate([next_drone, state[19:]])

        # ---- Reward/Loss ------------------------------------------
        position_loss = jnp.linalg.norm(next_drone[0:3] - state[19:22])
        rate_loss = jnp.linalg.norm(next_drone[10:13])
        dist = self._get_nearest_obstacle_dist(next_state)
        collision_loss = jnp.maximum(1.0 - dist, 0.0)**2 + self.b1 * jax.nn.softplus(self.b2 * (-dist))
        reward = -position_loss - rate_loss - collision_loss

        return next_state, self._get_obs(next_state), reward, jnp.bool_(False), jnp.bool_(False), {}
    
    def compute_loss(self, traj):
        v            = traj["v"]              # (B, T, 3)
        target_v     = traj["target_v_world"] # (B, T, 3)
        dist         = traj["dist"]           # (B, T)
        a_cmd        = traj["a_cmd"]          # (B, T, 3)
        v_pred       = traj["v_pred"]         # (B, T, 3)
        drone_radius = traj["drone_radius"]   # (B, T)

        # -- Position loss -----------------------------------------------------

        # -- Rate loss ---------------------------------------------------------

        # -- Collision losses --------------------------------------------------
        # Rate of approach from consecutive distance samples
        dist_diff = jnp.diff(dist, axis=1)                    # (B, T-1)
        v_to_pt   = jnp.clip(-dist_diff / self.dt, 1.0, None) # (B, T-1)
        clearance = dist[:, 1:] - drone_radius[:, 1:]          # (B, T-1)

        # Softplus: exponential penalty when drone body is inside obstacle
        loss_collide = jnp.mean(jax.nn.softplus(-32.0 * clearance) * v_to_pt)

        # Quadratic barrier: penalises approaching within 1 m of collision
        loss_obj  = jnp.mean(v_to_pt * jax.nn.relu(1.0 - clearance) ** 2)
    

    # -----------------------------------------------------------------------
    # Morphology (kept for compatibility)
    # -----------------------------------------------------------------------

    def _alpha_to_raw(self, alpha_val: float) -> float:
        normalized = (alpha_val - self.alpha_min) / (self.alpha_max - self.alpha_min)
        normalized = jnp.clip(normalized, 1e-6, 1.0 - 1e-6)
        return jnp.log(normalized / (1.0 - normalized))

    def init_morph(self) -> dict:
        alpha_raw = (
            jnp.array([self._alpha_to_raw(self.alpha_default),
                        self._alpha_to_raw(-self.alpha_default),
                        self._alpha_to_raw(self.alpha_default)])
            if self.alternating_alpha else jnp.zeros(3)
        )
        return {"l_raw": jnp.zeros(3), "phi_raw": jnp.zeros(3), "alpha_raw": alpha_raw}

    def get_l(self, mp):     return self.l_min     + (self.l_max     - self.l_min)     * jax.nn.sigmoid(mp["l_raw"])
    def get_phi(self, mp):   return self.phi_min   + (self.phi_max   - self.phi_min)   * jax.nn.sigmoid(mp["phi_raw"])
    def get_alpha(self, mp): return self.alpha_min + (self.alpha_max - self.alpha_min) * jax.nn.sigmoid(mp["alpha_raw"])

    def get_morph_info(self, mp) -> dict:
        l, phi, alpha = self.get_l(mp), self.get_phi(mp), self.get_alpha(mp)
        return {f"l{i+1}": float(l[i]) for i in range(3)} \
             | {f"phi{i+1}": float(phi[i]) for i in range(3)} \
             | {f"alpha{i+1}": float(alpha[i]) for i in range(3)}
