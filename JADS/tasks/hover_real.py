import jax
import jax.numpy as jnp
from ..drone_physics.dynamics_real import forward_euler, semi_implicit_euler, rk4
from ..drone_physics import randomization
from ..drone_physics.quat_math import euler_to_quat, quat_to_euler, quat_mul
from ..depth_render.primitives import point_plane_dist


class HoverReal:
    # state: [pos(3), vel(3), quat(4), omega(3), W(n)] = 13 + n
    # obs:   [pos - target(3), vel(3), euler(3), omega(3), W(n)] = 12 + n
    #
    # n is the motor count, read off the drone yaml in __init__ (one moment
    # coefficient per motor), so a quad and a hex both fly on this code. The
    # values below are the four-motor defaults, overwritten once `drone` is
    # parsed — read them from the instance, never from the class.
    obs_dim: int = 16
    act_dim: int = 4
    drone_state_dim: int = 17  # 13 + act_dim
    dt: float = 0.01

    integrator: str = "rk4"

    b1: float = 1.0
    b2: float = 32.0

    # ---- Maximum Velocity ---------------------------------------------    
    max_velocity:      float = 2.0
    max_vel_weight:    float = 5.0

    # Drone parameters: the parsed configs/drone/*.yaml dict (train.py resolves
    # the path ref). `dr` overrides its domain_randomization.dr when set.
    drone: dict = None
    dr:    float = None

    # Initial state bounds [min, max] per DOF
    init_x_min:     float = -3.0;  init_x_max:     float = -2.0
    init_y_min:     float = -1.0;  init_y_max:     float = 1.0
    init_z_min:     float = -2.0;  init_z_max:     float = -1.0
    init_vx_min:    float = -0.5;  init_vx_max:    float = 0.5
    init_vy_min:    float = -0.5;  init_vy_max:    float = 0.5
    init_vz_min:    float = -0.5;  init_vz_max:    float = 0.5
    init_roll_min:  float = -jnp.pi / 4;  init_roll_max:  float = jnp.pi / 4
    init_pitch_min: float = -jnp.pi / 4;  init_pitch_max: float = jnp.pi / 4
    init_yaw_min:   float = -jnp.pi / 9;  init_yaw_max:   float = jnp.pi / 9
    init_wx_min:    float = -0.3;  init_wx_max:    float = 0.3
    init_wy_min:    float = -0.3;  init_wy_max:    float = 0.3
    init_wz_min:    float = -0.3;  init_wz_max:    float = 0.3
    init_W_min:     float = -1.0;   init_W_max:     float = 1.0
    target_x:       float = 3.0
    target_y:       float = -0.0
    target_z:       float = -1.5

    # ---- Crash bounds -------------------------------------------------------
    # There is no scene here, so what counts as a "crash" is the ground plane
    # plus a divergence radius standing in for navigate's arena geometry.
    ground_z:       float = 0.0    # ground plane height (NED: +z is down)
    max_dist:       float = 10.0   # give up once this far from the target

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter '{k}'")
            setattr(self, k, v)

        if self.drone is None:
            raise ValueError("HoverReal needs a `drone` config (see configs/drone/)")
        self.nominal_params = randomization.load(self.drone["drone_params"])
        # Physical layout, used by the visualiser only — never by the dynamics.
        self.geometry = self.drone.get("geometry")
        if self.dr is None:
            self.dr = float(self.drone.get("domain_randomization", {}).get("dr", 0.0))

        # The motor count is a property of the airframe, so it is derived from
        # the drone config rather than configured separately. A config that
        # states one of these explicitly must agree with the yaml it points at.
        n_motors = int(self.nominal_params.k_p.shape[0])
        for name, derived in (("act_dim",         n_motors),
                              ("obs_dim",         12 + n_motors),
                              ("drone_state_dim", 13 + n_motors)):
            if name in kwargs and int(kwargs[name]) != derived:
                raise ValueError(
                    f"drone params describe {n_motors} motors, which implies "
                    f"{name}={derived}, but the config sets {name}={kwargs[name]}")
            setattr(self, name, derived)

        geom = self.geometry or {}
        if "motor_positions" in geom and len(geom["motor_positions"]) != n_motors:
            raise ValueError(
                f"geometry.motor_positions has {len(geom['motor_positions'])} rows "
                f"but drone params describe {n_motors} motors")

    def reset(self, key: jax.Array) -> tuple:
        """
        Sample a random initial state and a random set of drone parameters.

        The params are drawn here, once per episode, and returned in `info` so
        the training loop can park them in the per-element rollout carry — each
        parallel episode then flies its own airframe for that episode's
        duration. See JADS/drone_physics/randomization.py.

        Returns:
            obs:   observation vector, shape (obs_dim,)        [pos-target, vel, euler, omega, W]
            state: state array, shape (drone_state_dim,)       [pos, vel, quat, omega, W]
            info:  {"params": DroneParams}
        """
        (k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13,
         kp) = jax.random.split(key, 14)
        x     = jax.random.uniform(k1,  minval=self.init_x_min,     maxval=self.init_x_max)
        y     = jax.random.uniform(k2,  minval=self.init_y_min,     maxval=self.init_y_max)
        z     = jax.random.uniform(k3,  minval=self.init_z_min,     maxval=self.init_z_max)
        vx    = jax.random.uniform(k4,  minval=self.init_vx_min,    maxval=self.init_vx_max)
        vy    = jax.random.uniform(k5,  minval=self.init_vy_min,    maxval=self.init_vy_max)
        vz    = jax.random.uniform(k6,  minval=self.init_vz_min,    maxval=self.init_vz_max)
        roll  = jax.random.uniform(k7,  minval=self.init_roll_min,  maxval=self.init_roll_max)
        pitch = jax.random.uniform(k8,  minval=self.init_pitch_min, maxval=self.init_pitch_max)
        yaw   = jax.random.uniform(k9,  minval=self.init_yaw_min,   maxval=self.init_yaw_max)
        wx    = jax.random.uniform(k10, minval=self.init_wx_min,    maxval=self.init_wx_max)
        wy    = jax.random.uniform(k11, minval=self.init_wy_min,    maxval=self.init_wy_max)
        wz    = jax.random.uniform(k12, minval=self.init_wz_min,    maxval=self.init_wz_max)
        pos   = jnp.array([x, y, z])
        vel   = jnp.array([vx, vy, vz])
        rpy   = jnp.array([roll, pitch, yaw])
        omega = jnp.array([wx, wy, wz])
        quat  = euler_to_quat(rpy[0], rpy[1], rpy[2])
        W     = jax.random.uniform(k13, shape=(self.act_dim,), minval=self.init_W_min, maxval=self.init_W_max)
        state = jnp.concatenate([pos, vel, quat, omega, W])
        params = randomization.randomize(self.nominal_params, kp, self.dr)
        return self._get_obs(state), state, {"params": params}

    def reset_to(self,
                 pos:   jnp.ndarray = jnp.array([-3.0, 0.0, -1.5]),
                 vel:   jnp.ndarray = None,
                 rpy:   jnp.ndarray = None,
                 omega: jnp.ndarray = None) -> tuple:
        """
        Reset to a predefined state on the *nominal* params (no randomization),
        for evaluation and replay. Unspecified fields default to zero, except
        the motors, which start at hover trim rather than spinning up from below.

        Args:
            pos:   (3,) position [x, y, z]
            vel:   (3,) linear velocity
            rpy:   (3,) roll, pitch, yaw in radians
            omega: (3,) angular velocity
        Returns:
            obs, state, info
        """
        pos   = jnp.asarray(pos)   if pos   is not None else jnp.zeros(3)
        vel   = jnp.asarray(vel)   if vel   is not None else jnp.zeros(3)
        rpy   = jnp.asarray(rpy)   if rpy   is not None else jnp.zeros(3)
        omega = jnp.asarray(omega) if omega is not None else jnp.zeros(3)
        quat  = euler_to_quat(rpy[0], rpy[1], rpy[2])
        W, _  = randomization.hover_trim(self.nominal_params)   # per-motor trim
        state = jnp.concatenate([pos, vel, quat, omega, W])
        return self._get_obs(state), state, {"params": self.nominal_params}

    def _get_obs(self, state: jnp.ndarray) -> jnp.ndarray:
        pos   = state[0:3] - jnp.array([self.target_x, self.target_y, self.target_z])
        vel   = state[3:6]
        quat  = state[6:10]
        omega = state[10:13]
        W     = state[13:]
        euler = quat_to_euler(quat)
        return jnp.concatenate([pos, vel, euler, omega, W])

    def _signed_dist(self, state: jnp.ndarray) -> jnp.ndarray:
        pos    = state[0:3]
        target = jnp.array([self.target_x, self.target_y, self.target_z])
        ground = point_plane_dist(pos,
                                  jnp.array([0.0, 0.0, self.ground_z]),
                                  jnp.array([0.0, 0.0, -1.0]))
        return jnp.minimum(ground, self.max_dist - jnp.linalg.norm(pos - target))

    def step(self, state: jnp.ndarray, action: jnp.ndarray, params=None,
             morph_matrices=None) -> tuple:
        # `params` is this episode's airframe, threaded from the rollout carry.
        # Direct callers (eval / replay scripts) get the nominal set.
        if params is None:
            params = self.nominal_params

        U = jnp.clip(action, -1.0, 1.0)  # command ∈ [-1, 1], matching W state range

        integrators = {"euler": forward_euler, "semi_implicit_euler": semi_implicit_euler, "rk4": rk4}
        integrate   = integrators[self.integrator]
        next_state  = integrate(state, U, params, self.dt)

        # Renormalize quaternion and clip velocities
        quat_norm  = jnp.maximum(jnp.linalg.norm(next_state[6:10]), 1e-8)
        next_quat  = next_state[6:10] / quat_norm
        next_state = next_state.at[6:10].set(next_quat)
        next_state = next_state.at[3:6].set(jnp.clip(next_state[3:6],   -20.0, 20.0))
        next_state = next_state.at[10:13].set(jnp.clip(next_state[10:13], -20.0, 20.0))

        dist = self._signed_dist(next_state)
        step_data = {
            "pos":    next_state[0:3],
            "vel":    next_state[3:6],
            "omega":  next_state[10:13],
            "action": U,
            "dist":   dist,
            "crashed": jax.lax.stop_gradient(dist < 0.0),
        }
        return next_state, self._get_obs(next_state), step_data

    def compute_loss(self, traj) -> tuple:
        """
        Compute loss from a full batched trajectory.

        traj: dict of (batch, horizon, ...) arrays.
        Returns: (total_loss, mean_return)
        """
        pos   = traj["pos"]    # (B, T, 3)
        vel   = traj["vel"]    # (B, T, 3)
        omega = traj["omega"]  # (B, T, 3)
        U     = traj["action"] # (B, T, act_dim)
        target = jnp.array([self.target_x, self.target_y, self.target_z])

        loss_max_vel = jnp.mean(self.b1 * jax.nn.softplus(self.b2 * (vel - self.max_velocity)))

        per_step = -(
            jnp.sum((pos - target)** 2, axis=-1)
            + 0.1  * jnp.sum(vel   ** 2, axis=-1)
            + 0.01 * jnp.sum(omega ** 2, axis=-1)
            + self.max_vel_weight*loss_max_vel 
            + 0.05 * jnp.sum(((U + 1.0) / 2.0) ** 2, axis=-1)
        )  # (B, T)
        mean_return = jnp.mean(per_step)
        return -mean_return, mean_return