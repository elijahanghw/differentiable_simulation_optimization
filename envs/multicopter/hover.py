import jax
import jax.numpy as jnp
from .dynamics import forward_euler, semi_implicit_euler, rk4
from .morphology import morphology
from .quat_math import euler_to_quat, quat_to_euler, quat_mul


class Hover3d:
    # state: [pos(3), vel(3), quat(4), omega(3), W(6)] = 19
    # obs:   [pos(3), vel(3), euler(3), omega(3), W(6)] = 18
    obs_dim: int = 18
    act_dim: int = 6
    dt: float = 0.01

    # Morphology bounds
    l_min: float = 0.10
    l_max: float = 0.20
    l_default: float = 0.15   # fixed arm length when train_morphology=False

    phi_min: float = -jnp.pi / 6   # -30°
    phi_max: float =  jnp.pi / 6   #  30°
    phi_default: float = 0.0

    alpha_min: float = -jnp.pi / 4  # -45°
    alpha_max: float =  jnp.pi / 4  #  45°
    alpha_default: float = 0.0

    train_morphology: bool = True
    integrator: str = "rk4"

    # Initial state bounds [min, max] per DOF
    init_x_min:     float = -4.0;  init_x_max:     float = 0.0
    init_y_min:     float = -0.5;  init_y_max:     float = 0.5
    init_z_min:     float = -0.25;  init_z_max:     float = 0.25
    init_vx_min:    float = -0.5;  init_vx_max:    float = 0.5
    init_vy_min:    float = -0.5;  init_vy_max:    float = 0.5
    init_vz_min:    float = -0.5;  init_vz_max:    float = 0.5
    init_roll_min:  float = -jnp.pi / 3;  init_roll_max:  float = jnp.pi / 3
    init_pitch_min: float = -jnp.pi / 3;  init_pitch_max: float = jnp.pi / 3
    init_yaw_min:   float = -jnp.pi / 9;  init_yaw_max:   float = jnp.pi / 9
    init_wx_min:    float = -0.3;  init_wx_max:    float = 0.3
    init_wy_min:    float = -0.3;  init_wy_max:    float = 0.3
    init_wz_min:    float = -0.3;  init_wz_max:    float = 0.3
    init_W_min:     float = -1.0;   init_W_max:     float = 1.0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter '{k}'")
            setattr(self, k, v)

    def reset(self, key: jax.Array) -> tuple:
        """
        Sample a random initial state.

        Returns:
            obs:   observation vector, shape (obs_dim,)  [pos, vel, euler, omega]
            state: state array, shape (13,)              [pos, vel, quat, omega]
            info:  empty dict
        """
        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13 = jax.random.split(key, 13)
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
        W     = jax.random.uniform(k13, shape=(6,), minval=self.init_W_min, maxval=self.init_W_max)
        state = jnp.concatenate([pos, vel, quat, omega, W])
        return self._get_obs(state), state, {}

    def reset_to(self,
                 pos:   jnp.ndarray = jnp.array([-4.0, 0.0, 0.0]),
                 vel:   jnp.ndarray = None,
                 rpy:   jnp.ndarray = None,
                 omega: jnp.ndarray = None) -> tuple:
        """
        Reset to a predefined state. Unspecified fields default to zero.

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
        W     = -0.5 * jnp.ones(6)
        state = jnp.concatenate([pos, vel, quat, omega, W])
        return self._get_obs(state), state, {}

    def _get_obs(self, state: jnp.ndarray) -> jnp.ndarray:
        pos   = state[0:3]
        vel   = state[3:6]
        quat  = state[6:10]
        omega = state[10:13]
        W     = state[13:19]
        euler = quat_to_euler(quat)
        return jnp.concatenate([pos, vel, euler, omega, W])

    def step(self, state: jnp.ndarray, action: jnp.ndarray, morph_params: dict = None) -> tuple:
        if self.train_morphology and morph_params is not None:
            l     = self.l_min     + (self.l_max     - self.l_min)     * jax.nn.sigmoid(morph_params["l_raw"])     # (3,)
            phi   = self.phi_min   + (self.phi_max   - self.phi_min)   * jax.nn.sigmoid(morph_params["phi_raw"])   # (3,)
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * jax.nn.sigmoid(morph_params["alpha_raw"]) # (3,)
        else:
            l     = jnp.full(3, self.l_default)
            phi   = jnp.full(3, self.phi_default)
            alpha = jnp.full(3, self.alpha_default)

        Bf, Bm, m, J, J_inv = morphology(l, phi, alpha)

        U = jnp.clip(action, -1.0, 1.0)  # command ∈ [-1, 1], matching W state range

        integrators = {"euler": forward_euler, "semi_implicit_euler": semi_implicit_euler, "rk4": rk4}
        integrate   = integrators[self.integrator]
        next_state  = integrate(state, U, Bf, Bm, m, J, J_inv, self.dt)

        # Renormalize quaternion and clip velocities
        quat_norm  = jnp.maximum(jnp.linalg.norm(next_state[6:10]), 1e-8)
        next_quat  = next_state[6:10] / quat_norm
        next_state = next_state.at[6:10].set(next_quat)
        next_state = next_state.at[3:6].set(jnp.clip(next_state[3:6],   -20.0, 20.0))
        next_state = next_state.at[10:13].set(jnp.clip(next_state[10:13], -20.0, 20.0))

        pos   = next_state[0:3]
        vel   = next_state[3:6]
        euler = quat_to_euler(next_quat)
        omega = next_state[10:13]

        reward = -(
            jnp.sum(pos**2)
            + 0.1 * jnp.sum(vel**2)
            # + 0.01 * jnp.sum(euler[0:2]**2)
            + 0.01 * jnp.sum(omega**2)
            + 0.05 * jnp.sum(((U + 1.0) / 2.0) ** 2)
        )

        return next_state, self._get_obs(next_state), reward, jnp.bool_(False), jnp.bool_(False), {}

    # ------------------------------------------------------------------
    # Morphology
    # ------------------------------------------------------------------

    def init_morph(self) -> dict:
        # raw = 0 → sigmoid(0) = 0.5 → midpoint of bounds for all params
        return {"l_raw": jnp.zeros(3), "phi_raw": jnp.zeros(3), "alpha_raw": jnp.zeros(3)}

    def get_l(self, morph_params: dict) -> jnp.ndarray:
        return self.l_min + (self.l_max - self.l_min) * jax.nn.sigmoid(morph_params["l_raw"])

    def get_phi(self, morph_params: dict) -> jnp.ndarray:
        return self.phi_min + (self.phi_max - self.phi_min) * jax.nn.sigmoid(morph_params["phi_raw"])

    def get_alpha(self, morph_params: dict) -> jnp.ndarray:
        return self.alpha_min + (self.alpha_max - self.alpha_min) * jax.nn.sigmoid(morph_params["alpha_raw"])

    def get_morph_info(self, morph_params: dict) -> dict:
        l     = self.get_l(morph_params)
        phi   = self.get_phi(morph_params)
        alpha = self.get_alpha(morph_params)
        return {
            "l1": float(l[0]),       "l2": float(l[1]),       "l3": float(l[2]),
            "phi1": float(phi[0]),   "phi2": float(phi[1]),   "phi3": float(phi[2]),
            "alpha1": float(alpha[0]), "alpha2": float(alpha[1]), "alpha3": float(alpha[2]),
        }
