"""
Differentiable Pendulum environment with perpendicular tip force.

The force is applied at position r along the rod, where r is a learnable
morphology parameter.

API (gym-like)
--------------
    obs, state, info = env.reset(key)
    next_state, obs, reward, terminated, truncated, info = env.step(state, action, morph_params)
"""

import jax
import jax.numpy as jnp


class PendulumForce:
    obs_dim: int = 3
    act_dim: int = 1

    # Default physical parameters
    max_force: float = 2.0
    max_speed: float = 8.0
    dt: float = 0.05
    g: float = 9.81
    m: float = 1.0
    m2: float = 2.0
    l: float = 1.0
    train_morphology: bool = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter '{k}'")
            setattr(self, k, v)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, key: jax.Array) -> tuple:
        """
        Sample a random initial state.

        Returns:
            obs:   observation vector, shape (obs_dim,)
            state: internal state tuple (theta, theta_dot)
            info:  empty dict
        """
        key_theta, key_tdot = jax.random.split(key)
        theta     = jax.random.uniform(key_theta, minval=-jnp.pi, maxval=jnp.pi)
        theta_dot = jax.random.uniform(key_tdot,  minval=-1.0,    maxval=1.0)
        state = (theta, theta_dot)
        return self._get_obs(state), state, {}

    def step(self, state: tuple, action: jnp.ndarray, morph_params: dict = None) -> tuple:
        """
        Advance the state by one timestep.

        Args:
            state:        (theta, theta_dot)
            action:       normalised force in [-1, 1], shape (1,) or scalar
            morph_params: dict with key 'r_raw' — unconstrained force position

        Returns:
            next_state:  (new_theta, new_theta_dot)
            obs:         observation of next_state, shape (obs_dim,)
            reward:      scalar reward
            terminated:  False
            truncated:   False
            info:        empty dict
        """
        theta, theta_dot = state

        if self.train_morphology:
            r = self.l * jax.nn.sigmoid(morph_params["r_raw"])
        else:
            r = self.l

        F = jnp.squeeze(action) * self.max_force
        F = jnp.clip(F, -self.max_force, self.max_force)

        J = (self.m * self.l ** 2) / 3.0 + self.m2 * r**2

        # theta_ddot = (F * r + ((self.m * self.g * self.l) / 2 + self.m2 * self.g * r)* jnp.sin(theta)) / J
        theta_ddot = F * r / J

        # theta_ddot = (
        #     3.0 * self.g / (2.0 * self.l) * jnp.sin(theta)
        #     + 3.0 * F * r / (self.m * self.l ** 2)
        # )
        new_theta_dot = jnp.clip(
            theta_dot + theta_ddot * self.dt, -self.max_speed, self.max_speed
        )
        new_theta  = theta + new_theta_dot * self.dt

        theta_norm = ((new_theta + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi
        reward = -(theta_norm ** 2 + 0.1 * new_theta_dot ** 2 + 0.1 * F ** 2)

        next_state = (new_theta, new_theta_dot)

        return next_state, self._get_obs(next_state), reward, jnp.bool_(False), jnp.bool_(False), {}

    # ------------------------------------------------------------------
    # Morphology
    # ------------------------------------------------------------------

    def init_morph(self) -> dict:
        """r_raw = 0.0  →  r = l·sigmoid(0) = l/2 (midpoint of rod)."""
        return {"r_raw": jnp.array(0.0)}

    def get_r(self, morph_params: dict) -> jnp.ndarray:
        """Return the actual (constrained) force application position."""
        return self.l * jax.nn.sigmoid(morph_params["r_raw"])

    def get_morph_info(self, morph_params: dict) -> dict:
        """Return a dict of all decoded morphological parameters for logging."""
        return {"r": float(self.get_r(morph_params))}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_obs(self, state: tuple) -> jnp.ndarray:
        theta, theta_dot = state
        return jnp.array([jnp.cos(theta), jnp.sin(theta), theta_dot])
