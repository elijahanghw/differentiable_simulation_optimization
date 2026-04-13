"""
Differentiable Inverted Pendulum environment.

Follows the dynamics and rewards of gymnasium's Pendulum-v1.
State convention: theta=0 is the upright (goal) position.
State is a plain tuple (theta, theta_dot) — a valid JAX pytree.

API (gym-like)
--------------
    obs, state, info = env.reset(key)
    next_state, obs, reward, terminated, truncated, info = env.step(state, action)
"""

import jax
import jax.numpy as jnp


class Pendulum:
    obs_dim: int = 3
    act_dim: int = 1

    # Default physical parameters (gymnasium Pendulum-v1 values)
    max_torque: float = 2.0
    max_speed: float = 8.0
    dt: float = 0.05
    g: float = 9.81
    m: float = 1.0
    l: float = 1.0

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

    def step(self, state: tuple, action: jnp.ndarray) -> tuple:
        """
        Advance the state by one timestep.

        Args:
            state:  (theta, theta_dot)
            action: normalised torque in [-1, 1], shape (1,) or scalar

        Returns:
            next_state:  (new_theta, new_theta_dot)
            obs:         observation of next_state, shape (obs_dim,)
            reward:      scalar reward
            terminated:  False — pendulum has no terminal condition
            truncated:   False — time limit is managed by the training loop
            info:        empty dict
        """
        theta, theta_dot = state

        u = jnp.squeeze(action) * self.max_torque
        u = jnp.clip(u, -self.max_torque, self.max_torque)

        theta_norm = ((theta + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi
        reward = -(theta_norm ** 2 + 0.1 * theta_dot ** 2 + 0.001 * u ** 2)

        theta_ddot = (
            3.0 * self.g / (2.0 * self.l) * jnp.sin(theta)
            + 3.0 / (self.m * self.l ** 2) * u
        )
        new_theta_dot = jnp.clip(
            theta_dot + theta_ddot * self.dt, -self.max_speed, self.max_speed
        )
        new_theta  = theta + new_theta_dot * self.dt
        next_state = (new_theta, new_theta_dot)

        return next_state, self._get_obs(next_state), reward, jnp.bool_(False), jnp.bool_(False), {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_obs(self, state: tuple) -> jnp.ndarray:
        theta, theta_dot = state
        return jnp.array([jnp.cos(theta), jnp.sin(theta), theta_dot])
