"""
GRU-based Actor implemented with Flax.

GRUActor: obs → [MLP encoder] → [GRU cell] → [MLP head] → action mean
"""

import math
from typing import Sequence

import jax.numpy as jnp
import flax.linen as nn

_SQRT2 = math.sqrt(2.0)


def _ortho(gain: float):
    return nn.initializers.orthogonal(scale=gain)


class GRUActor(nn.Module):
    """
    obs → [MLP encoder] → [GRU cell] → [MLP head] → action mean

    Args:
        obs_dim:        Observation dimension (used to create dummy input for init).
        act_dim:        Action dimension.
        encoder_sizes:  Hidden layer widths for the MLP encoder.
        hidden_size:    GRU hidden state dimension.
        head_sizes:     Hidden layer widths for the actor MLP head.
        squash_output:  Apply tanh to output (default True).
    """
    obs_dim: int
    act_dim: int
    encoder_sizes: Sequence[int] = (64,)
    hidden_size: int = 128
    head_sizes: Sequence[int] = (64,)
    squash_output: bool = True

    @nn.compact
    def __call__(self, obs, hidden):
        x = obs
        for size in self.encoder_sizes:
            x = nn.Dense(size, kernel_init=_ortho(_SQRT2), bias_init=nn.initializers.zeros)(x)
            x = nn.relu(x)

        new_hidden, x = nn.GRUCell(self.hidden_size)(hidden, x)

        for size in self.head_sizes:
            x = nn.Dense(size, kernel_init=_ortho(_SQRT2), bias_init=nn.initializers.zeros)(x)
            x = nn.relu(x)
        action_mean = nn.Dense(self.act_dim, kernel_init=_ortho(0.01), bias_init=nn.initializers.zeros)(x)
        if self.squash_output:
            action_mean = jnp.tanh(action_mean)

        return action_mean, new_hidden

    def init_hidden(self):
        """Return a zero hidden state for one environment, shape (hidden_size,)."""
        return jnp.zeros(self.hidden_size)


