import jax
import jax.numpy as jnp
from .dynamics import euler, rk4, semi_implicit_euler
from .morphology import morphology

class Hover2d:
    obs_dim: int = 6
    act_dim: int = 2
    dt: float = 0.05

    # Default morphology
    l: float = 0.5

    train_morphology: bool = True
    integrator: str = "rk4"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter '{k}'")
            setattr(self, k, v)

    def reset(self, key: jax.Array) -> tuple:
        """
        Sample a random initial state.

        Returns:
            obs:   observation vector, shape (obs_dim,)  [x, z, vx, vz, theta, omega]
            state: state array, shape (6,)
            info:  empty dict
        """
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        x     = jax.random.uniform(k1, minval=-3.0,          maxval=3.0)
        z     = jax.random.uniform(k2, minval=-3.0,          maxval=3.0)
        vx    = jax.random.uniform(k3, minval=-1.0,          maxval=1.0)
        vz    = jax.random.uniform(k4, minval=-1.0,          maxval=1.0)
        theta = jax.random.uniform(k5, minval=-jnp.pi / 6,   maxval=jnp.pi / 6)
        omega = jax.random.uniform(k6, minval=-1.0,          maxval=1.0)
        state = jnp.array([x, z, vx, vz, theta, omega])
        return self._get_obs(state), state, {}

    def _get_obs(self, state: jnp.ndarray) -> jnp.ndarray:
        return state

    def step(self, state: tuple, action: jnp.ndarray, morph_params: dict = None) -> tuple:

        if self.train_morphology and morph_params is not None:
            r = 0.1 + (self.l - 0.1) * jax.nn.sigmoid(morph_params["r_raw"])
        else:
            r = self.l

        Bf, Bm, m, J = morphology(r)

        action = jnp.clip(action, -1, 1)
        U = (action + 1.0) / 2.0

        integrators = {"rk4": rk4, "euler": euler, "semi_implicit_euler": semi_implicit_euler}
        integrate = integrators[self.integrator]
        next_state = integrate(state, U, Bf, Bm, m, J, self.dt)
        next_state = next_state.at[2:4].set(jnp.clip(next_state[2:4], -20., 20.))
        next_state = next_state.at[5].set(jnp.clip(next_state[5], -20., 20.))

        reward = -(jnp.sum(next_state[0:2]**2) + 0.1*jnp.sum(next_state[2:4]**2) + 0.1*jnp.sum(U**2))

        return self._get_obs(next_state), next_state, reward, jnp.bool_(False), jnp.bool_(False), {}

    # ------------------------------------------------------------------
    # Morphology
    # ------------------------------------------------------------------

    def init_morph(self) -> dict:
        """r_raw = 0.0  →  r = l·sigmoid(0) = l/2 (midpoint of rod)."""
        return {"r_raw": jnp.array(0.0)}

    def get_r(self, morph_params: dict) -> jnp.ndarray:
        """Return the actual (constrained) force application position."""
        return 0.1 + (self.l - 0.1) * jax.nn.sigmoid(morph_params["r_raw"])

    def get_morph_info(self, morph_params: dict) -> dict:
        """Return a dict of all decoded morphological parameters for logging."""
        return {"r": float(self.get_r(morph_params))}