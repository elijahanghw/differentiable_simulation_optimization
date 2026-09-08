import os, sys, yaml
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "depth_render"))

import jax
import jax.numpy as jnp
from ..drone_physics.dynamics_real import forward_euler, semi_implicit_euler, rk4, _gdecay
from ..drone_physics import randomization
from ..drone_physics.quat_math import euler_to_quat, quat_to_euler, quat_to_rotmat
from ..scene.scene import SceneConfig

from JADS.depth_render.renderer import render_depth, apply_sensor_noise  # depth_render/renderer.py
from JADS.depth_render.tof import zone_reduce                            # depth_render/tof.py
from JADS.depth_render.primitives import (                               # depth_render/primitives.py
    point_sphere_dist, point_aabb_dist, point_obb_dist,
    point_capsule_dist, point_plane_dist,
)


class NavigateReal:
    """
    Navigate's task and loss, flown on the identified drone model.

    State layout (flat float32 array of length state_dim), with nd =
    drone_state_dim = 13 + n_motors:
        [0:nd]         drone state  (pos, vel, quat, omega, W)
        [nd:nd+3]      target pos
        [nd+3:]        scene array  (SceneConfig.sample — see scene.py)
                         static mode:     full obstacle geometry  (scene_dim floats)
                         procedural mode: single float32 episode seed  (1 float)

    Observation: (depth_map, [target-relative pos, vel, euler, omega, W]).

    Scene config can be passed as:
        NavigateReal(scene=SceneConfig(...))
        NavigateReal(scene={"obstacle_density": 0.2, ...})   ← dict from YAML
        NavigateReal()                                        ← SceneConfig defaults
    """

    # ---- Drone --------------------------------------------------------
    # drone state: [pos(3), vel(3), quat(4), omega(3), W(n)] = 13 + n
    # obs:         [pos - target(3), vel(3), euler(3), omega(3), W(n)] = 12 + n
    #
    # n is the motor count, read off the drone yaml in __init__ (one moment
    # coefficient per motor), so a quad and a hex both fly on this code. The
    # values below are the four-motor defaults, overwritten once `drone` is
    # parsed — read them from the instance, never from the class.
    obs_dim: int = 16
    critic_obs_dim: int = 20   # privileged critic input — see critic_obs()
    act_dim: int = 4
    drone_state_dim: int = 17  # 13 + act_dim; target sits right after it
    dt: float = 0.01

    # Drone parameters: the parsed configs/drone/*.yaml dict (train.py resolves
    # the path ref). `dr` overrides its domain_randomization.dr when set.
    drone: dict = None
    dr:    float = None

    # ---- Gradient decay -----------------------------------------------
    grad_decay: float = 0.4

    # ---- Collision loss -----------------------------------------------
    b1: float = 1.0
    b2: float = 32.0
    # None → taken from the drone yaml's `geometry` block: a prop_diameter/2
    # sphere per motor, and a body_radius shell at the CoM.
    motor_collision_radius: float = None
    body_collision_radius:  float = None

    # ---- Maximum Velocity ---------------------------------------------    
    max_velocity:      float = 2.0

    # ---- Loss weights -------------------------------------------------
    xy_weight:         float = 1.0
    z_weight:          float = 2.0
    vel_weight:        float = 0.1
    max_vel_weight:    float = 5.0
    rate_weight:       float = 0.01
    collision_weight:  float = 7.5
    obj_weight:        float = 3.0
    heading_weight:    float = 1.0
    smooth_action_weight: float = 0.0

    # ---- Depth camera ------------------------------------------------------
    cam_fov_deg:       float = 90.0
    cam_width:         int   = 64
    cam_height:        int   = 48
    cam_pool:          int   = 4      # max-pool factor → (12, 16) input to CNN
    cam_min_range:     float = 0.2
    cam_max_range:     float = 8.0
    cam_quantization_m: float = 0.001
    cam_hz:            float = 50.0   # depth-camera update rate; defaults to 1/dt (no frame skip)

    # ---- ToF sensor (VL53L8CX-class multizone array) -----------------------
    # Selected with `depth_camera: {type: tof, ...}`; see depth_render/tof.py.
    # The zone grid replaces the depth image as the policy's visual input —
    # each zone reports the nearest surface in its cone, found by tracing
    # tof_supersample² sub-rays. cam_width/cam_height/cam_pool are *derived*
    # from the zone grid in __init__, never configured, so everything reading
    # them (depth_shape, the HITL export) stays consistent.
    sensor_type:      str = "depth"   # "depth" | "tof"
    tof_zones_h:      int = 8
    tof_zones_w:      int = 8
    tof_supersample:  int = 4         # sub-rays per zone edge (S² per zone)
    tof_zone_agg:     str = "min"     # "min" (nearest reflector) | "mean"

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

        if self.drone is None:
            raise ValueError("NavigateReal needs a `drone` config (see configs/drone/)")
        self.nominal_params = randomization.load(self.drone["drone_params"])
        # Physical layout: the collision spheres below, and the visualiser.
        # Never the dynamics — those run off the identified coefficients alone.
        self.geometry = self.drone.get("geometry")
        if self.dr is None:
            self.dr = float(self.drone.get("domain_randomization", {}).get("dr", 0.0))

        # The motor count is a property of the airframe, so it is derived from
        # the drone config rather than configured separately — every state slice
        # below is cut relative to drone_state_dim. A config that states one of
        # these explicitly must agree with the yaml it points at.
        n_motors = int(self.nominal_params.k_p.shape[0])
        for name, derived in (("act_dim",         n_motors),
                              ("obs_dim",         12 + n_motors),
                              ("drone_state_dim", 13 + n_motors),
                              ("critic_obs_dim",  16 + n_motors)):
            if name in kwargs and int(kwargs[name]) != derived:
                raise ValueError(
                    f"drone params describe {n_motors} motors, which implies "
                    f"{name}={derived}, but the config sets {name}={kwargs[name]}")
            setattr(self, name, derived)

        # Motor spheres, in the body frame. Fixed — there is no morphology to
        # derive them from, so unlike Navigate they are not a step() argument.
        if self.geometry is None:
            raise ValueError(
                "NavigateReal needs a `geometry` block in the drone config — the "
                "collision spheres are built from motor_positions")
        self.motor_pos_body = jnp.asarray(
            self.geometry["motor_positions"], dtype=jnp.float32)   # (n, 3) FRD
        if self.motor_pos_body.shape[0] != n_motors:
            raise ValueError(
                f"geometry.motor_positions has {self.motor_pos_body.shape[0]} rows "
                f"but drone params describe {n_motors} motors")
        if self.motor_collision_radius is None:
            self.motor_collision_radius = float(self.geometry.get("prop_diameter", 0.0)) / 2.0
        if self.body_collision_radius is None:
            self.body_collision_radius = float(self.geometry.get("body_radius", 0.0))

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
            self.sensor_type        = str(  dc.get("type",           self.sensor_type)).lower()
            self.cam_fov_deg        = float(dc.get("fov_deg",        self.cam_fov_deg))
            self.cam_min_range      = float(dc.get("min_range",      self.cam_min_range))
            self.cam_max_range      = float(dc.get("max_range",      self.cam_max_range))
            self.cam_quantization_m = float(dc.get("quantization_m", self.cam_quantization_m))
            self.cam_hz             = float(dc.get("cam_hz",         self.cam_hz))

            if self.sensor_type == "depth":
                self.cam_width      = int(  dc.get("width",          self.cam_width))
                self.cam_height     = int(  dc.get("height",         self.cam_height))
                self.cam_pool       = int(  dc.get("pool",           self.cam_pool))
            elif self.sensor_type == "tof":
                zones = int(dc.get("zones", self.tof_zones_h))
                self.tof_zones_h     = int(dc.get("zones_h", zones))
                self.tof_zones_w     = int(dc.get("zones_w", zones))
                self.tof_supersample = int(dc.get("supersample", self.tof_supersample))
                self.tof_zone_agg    = str(dc.get("zone_agg", self.tof_zone_agg)).lower()
                # The ray grid is the zone grid, supersampled.
                self.cam_height = self.tof_zones_h * self.tof_supersample
                self.cam_width  = self.tof_zones_w * self.tof_supersample
                self.cam_pool   = self.tof_supersample
            else:
                raise ValueError(
                    f"depth_camera.type '{self.sensor_type}' — expected 'depth' or 'tof'")

        if self.scene_cfg.procedural:
            self.scene_cfg.cell_size = self.cam_max_range

        self.state_dim = self.drone_state_dim + 3 + self.scene_cfg.scene_dim
        self._gd_factor = float(self.grad_decay ** self.dt)

        if self.cam_height % self.cam_pool or self.cam_width % self.cam_pool:
            raise ValueError(
                f"camera {self.cam_height}×{self.cam_width} is not divisible by "
                f"pool {self.cam_pool}")

        # Depth camera runs slower than the physics/policy loop (dt): render a
        # fresh frame every `frame_skip` steps and hold it (zero-order hold)
        # on the steps in between.
        self.frame_skip = max(1, round(1.0 / (self.dt * self.cam_hz)))

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(self, key: jax.Array) -> tuple:
        """
        Sample a random episode: drone start state, target, scene, and the
        airframe this episode flies (held for its duration — see
        JADS/drone_physics/randomization.py).

        Returns:
            obs:   (depth_map, (obs_dim,))
            state: (state_dim,) — drone + target + scene array
            info:  {"params": DroneParams}
        """
        key, drone_key, target_key, scene_key, param_key = jax.random.split(key, 5)

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
        W     = jax.random.uniform(keys[k], shape=(self.act_dim,),
                                   minval=self.init_W_min, maxval=self.init_W_max)

        quat       = euler_to_quat(roll, pitch, yaw)
        drone_state = jnp.concatenate([
            jnp.array([x, y, z]),
            jnp.array([vx, vy, vz]),
            quat,
            jnp.array([wx, wy, wz]),
            W,
        ])  # (drone_state_dim,)

        # ---- Target --------------------------------------------------------
        tkeys = jax.random.split(target_key, 3)
        target = jnp.array([
            jax.random.uniform(tkeys[0], minval=self.target_x_min, maxval=self.target_x_max),
            jax.random.uniform(tkeys[1], minval=self.target_y_min, maxval=self.target_y_max),
            jax.random.uniform(tkeys[2], minval=self.target_z_min, maxval=self.target_z_max),
        ])

        # ---- Scene (obstacles) --------------------------------------------
        scene_array = self.scene_cfg.sample(scene_key)  # (scene_dim,)

        state  = jnp.concatenate([drone_state, target, scene_array])
        params = randomization.randomize(self.nominal_params, param_key, self.dr)
        return self._get_obs(state), state, {"params": params}

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
        nd = self.drone_state_dim
        rel_pos = state[0:3] - state[nd:nd + 3]
        euler = jax.lax.stop_gradient(quat_to_euler(state[6:10]))
        drone_states = jnp.concatenate([rel_pos, state[3:6], euler, state[10:13], state[13:nd]])

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
        scene_start = self.drone_state_dim + 3   # drone block, then the target
        if self.scene_cfg.procedural:
            (sphere_centers, sphere_radii,
             box_centers, box_half_extents,
             cap_centers, cap_axes, cap_hh, cap_radii,
             obb_centers, obb_quats, obb_he) = self.scene_cfg.get_local_obstacles(
                state[0:3], state[scene_start]
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
        return self.scene_cfg.unpack(state[scene_start:])

    def _get_depth(self, state: jnp.ndarray, arrays: dict = None) -> jnp.ndarray:
        """
        Render a depth image from the drone's perspective.

        Args:
            state:  (state_dim,) — full environment state vector.
            arrays: pre-unpacked scene geometry (see _unpack_scene). Pass this
                    in when the caller already unpacked the scene for `state`,
                    to avoid re-sampling the procedural obstacle neighbourhood.

        Returns:
            float32 — distance in metres, (cam_height, cam_width) for a depth
            camera and (tof_zones_h, tof_zones_w) for a ToF sensor, where the
            tof_supersample² sub-rays traced through each zone have already
            been reduced to one distance.
            0 = closer than cam_min_range.
            cam_max_range = no-hit or saturated.
        """
        arrays = arrays if arrays is not None else self._unpack_scene(state)
        depth = render_depth(
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
        )
        if self.sensor_type == "tof":
            depth = zone_reduce(
                depth, self.tof_supersample, self.cam_max_range, self.tof_zone_agg,
            )
        return apply_sensor_noise(
            depth,
            min_range      = self.cam_min_range,
            max_range      = self.cam_max_range,
            quantization_m = self.cam_quantization_m,
        )

    def get_vis_depth(self, state: jnp.ndarray, width: int = 320, height: int = 240) -> jnp.ndarray:
        """Render depth at arbitrary resolution for visualization (no pooling, no normalization).

        A ToF sensor has no image behind its zone grid, so `width`/`height` are
        ignored there and the (tof_zones_h, tof_zones_w) frame the policy
        actually receives is returned instead.
        """
        if self.sensor_type == "tof":
            return self._get_depth(state)
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

    @property
    def depth_shape(self) -> tuple:
        """Shape of the visual observation the policy sees — the CNN input.

        Depth camera: the pooled image, e.g. (48, 64) / 4 → (12, 16).
        ToF sensor:   the zone grid, e.g. (8, 8) — cam_* are the supersampled
                      ray grid and cam_pool is the supersample factor, so the
                      same division lands on the zones.
        """
        return (self.cam_height // self.cam_pool, self.cam_width // self.cam_pool)

    def _get_processed_depth(self, state, arrays: dict = None):
        raw   = self._get_depth(state, arrays=arrays)
        normd = 3.0 / jnp.clip(raw, 0.3, self.cam_max_range) - 0.6
        if self.sensor_type == "tof":
            return normd    # _get_depth already reduced the sub-rays to zones
        # max-pool by cam_pool: (48, 64) → (12, 16) at the default 4
        pool = self.cam_pool
        return jax.lax.reduce_window(
            normd, -jnp.inf, jax.lax.max,
            window_dimensions=(pool, pool), window_strides=(pool, pool), padding="VALID",
        )
    
    def _get_nearest_obstacle_dist(self, state, motor_positions_world=None, arrays: dict = None):
        """Signed distance from the airframe shell to the nearest obstacle
        surface (negative = penetrating).

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

    def step(self, state: jnp.ndarray, action: jnp.ndarray, params=None,
             morph_matrices=None,
             step_idx: jnp.ndarray = None, prev_depth: jnp.ndarray = None) -> tuple:
        """
        params: this episode's airframe, threaded from the rollout carry. Direct
            callers (eval / replay scripts) get the nominal set.
        morph_matrices: accepted and ignored — the airframe is fixed, so there is
            nothing to derive. Present only because bptt.py passes it to every env.
        step_idx, prev_depth: see `_get_obs` — pass both to hold the depth
            image across steps and only re-render every `frame_skip` steps.
        """
        if params is None:
            params = self.nominal_params

        nd = self.drone_state_dim
        U = jnp.clip(action, -1.0, 1.0)  # command ∈ [-1, 1], matching W state range

        integrators = {"euler": forward_euler, "semi_implicit_euler": semi_implicit_euler, "rk4": rk4}
        next_drone   = integrators[self.integrator](_gdecay(state[0:nd], self._gd_factor), U, params, self.dt)

        # Renormalize quaternion, clip velocities
        quat_norm = jnp.maximum(jnp.linalg.norm(next_drone[6:10]), 1e-8)
        next_drone = next_drone.at[6:10].set(next_drone[6:10] / quat_norm)
        next_drone = next_drone.at[3:6].set(jnp.clip(next_drone[3:6],   -20.0, 20.0))
        next_drone = next_drone.at[10:13].set(jnp.clip(next_drone[10:13], -20.0, 20.0))

        # Target + scene are frozen; append unchanged
        next_state = jnp.concatenate([next_drone, state[nd:]])

        R_sg            = jax.lax.stop_gradient(quat_to_rotmat(next_drone[6:10]))
        motor_pos_world = next_drone[0:3] + self.motor_pos_body @ R_sg.T  # (n_motors, 3) world frame

        arrays = self._unpack_scene(next_state)
        dist = self._get_nearest_obstacle_dist(next_state, motor_pos_world, arrays=arrays)
        step_data = {
            "pos":        next_state[0:3],
            "vel":        next_state[3:6],
            "quat":       next_state[6:10],
            "target_pos": next_state[nd:nd + 3],
            "omega":      next_state[10:13],
            "action":     U,
            "dist":       dist,
            "crashed":    jax.lax.stop_gradient(dist < 0.0),
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
        act    = traj["action"]     # (B, T, act_dim)
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

        # Action smoothness: MSE between consecutive commands. Matters more here
        # than on the morphology drone — dynamics_real feeds the *command rate*
        # straight into the moments via k_rd·(W_c - W)/tau. Like the collision
        # terms this is a diff over the window, so the first step of each TBPTT
        # window is unpenalised.
        loss_smooth_action = jnp.mean(jnp.sum(jnp.diff(act, axis=1) ** 2, axis=-1))

        # Max velocity loss: Cap maximum velocity for safety
        loss_max_vel = jnp.mean(self.b1 * jax.nn.softplus(self.b2 * (vel - self.max_velocity)))

        total = (self.xy_weight*loss_xy + self.z_weight*loss_z
                 + self.vel_weight*loss_vel + self.max_vel_weight*loss_max_vel 
                 + self.rate_weight*loss_rate
                 + self.collision_weight*loss_collision + self.obj_weight*loss_obj
                 + self.heading_weight*loss_heading
                 + self.smooth_action_weight*loss_smooth_action)
        return total, -total



    # -----------------------------------------------------------------------
    # Per-step reward (RL algorithms — see algos/ppo.py)
    # -----------------------------------------------------------------------

    def initial_dist(self, state: jnp.ndarray, morph_matrices: tuple = None) -> jnp.ndarray:
        """Signed obstacle distance for `state`, mirroring what step() reports.

        Used to seed `prev_dist` for step_reward() right after reset(), where no
        previous step exists. `morph_matrices` is ignored — the motor spheres
        come from the fixed layout, so this agrees with step() unconditionally.
        """
        R_sg            = jax.lax.stop_gradient(quat_to_rotmat(state[6:10]))
        motor_pos_world = state[0:3] + self.motor_pos_body @ R_sg.T
        return self._get_nearest_obstacle_dist(state, motor_pos_world)

    def dist_to_target(self, state: jnp.ndarray) -> jnp.ndarray:
        """Euclidean distance from the drone to the target.

        Doubles as the potential for progress shaping in PPO (algos/ppo.py,
        `progress_weight`) and as a training diagnostic.
        """
        nd = self.drone_state_dim
        return jnp.linalg.norm(state[0:3] - state[nd:nd + 3])

    def critic_obs(self, state: jnp.ndarray, dist: jnp.ndarray) -> jnp.ndarray:
        """Privileged observation for an asymmetric (feed-forward) critic.

        The critic is discarded at deployment, so it may read state the actor
        never sees. Handing it the true target-relative position and the signed
        obstacle distance makes the value problem essentially fully observed,
        which is what removes the need for a recurrent critic: there is no
        history left to infer. `dist` is not recomputed here — pass the value
        step() already produced (or initial_dist() right after reset).

        One thing stays hidden that Navigate's critic does see everything of:
        the episode's randomized airframe is not in the state array, so the
        critic cannot tell which drone it is flying. That residual variance is
        the price of keeping the signature shared with the other envs.

        Layout (critic_obs_dim = 16 + n_motors, i.e. 20 for a quad):
            [0:3]   absolute position   (altitude matters — ground plane at z=0)
            [3:6]   target-relative position
            [6:9]   velocity
            [9:12]  attitude (euler)
            [12:15] body rates
            [15:15+n] motor speeds
            [15+n]  signed distance to the nearest obstacle  (privileged)
        """
        nd = self.drone_state_dim
        return jnp.concatenate([
            state[0:3],
            state[0:3] - state[nd:nd + 3],
            state[3:6],
            quat_to_euler(state[6:10]),
            state[10:13],
            state[13:nd],
            jnp.atleast_1d(dist),
        ])

    def step_reward(self, step_data: dict, prev_dist: jnp.ndarray) -> jnp.ndarray:
        """Reward for one step: the negated per-step summand of compute_loss.

        compute_loss() is a weighted mean over (batch, horizon) of per-step
        penalties, so maximising sum_t step_reward is — up to the discount and
        the 1/T factor — the same objective BPTT minimises. Keeping the two in
        exact correspondence is what makes a PPO-vs-(T)BPTT benchmark on this
        task a comparison of *optimisers* rather than of objectives.

        Args:
            step_data: one unbatched step dict from step().
            prev_dist: `dist` from the previous step. The collision terms are
                weighted by the approach speed -(dist_t - dist_{t-1})/dt, so the
                caller must thread it across steps (seed it with initial_dist()).

        Returns:
            scalar reward (negative — this is a cost-shaped task).
        """
        pos    = step_data["pos"]
        vel    = step_data["vel"]
        quat   = step_data["quat"]
        omega  = step_data["omega"]
        target = step_data["target_pos"]
        dist   = step_data["dist"]

        diff = pos - target
        loss_xy   = jnp.sum(diff[:2] ** 2)
        loss_z    = diff[2] ** 2
        loss_vel  = jnp.sum(vel ** 2)
        loss_rate = jnp.sum(omega ** 2)

        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        fwd = jnp.stack([
            1.0 - 2.0 * (y**2 + z**2),
            2.0 * (x*y + z*w),
            2.0 * (x*z - y*w),
        ])
        to_target = target - pos
        to_target = to_target / jnp.sqrt(jnp.maximum(jnp.sum(to_target ** 2), 1e-8))
        loss_heading = (1.0 - jnp.sum(fwd * to_target)) ** 2

        v_to_pt = jnp.clip(-(dist - prev_dist) / self.dt, 1.0, None)
        loss_collision = self.b1 * jax.nn.softplus(self.b2 * (-dist)) * v_to_pt
        loss_obj       = jax.nn.relu(0.5 - dist) ** 2 * v_to_pt

        total = (self.xy_weight*loss_xy + self.z_weight*loss_z
                 + self.vel_weight*loss_vel + self.rate_weight*loss_rate
                 + self.collision_weight*loss_collision + self.obj_weight*loss_obj
                 + self.heading_weight*loss_heading)
        return -total