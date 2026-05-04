"""
Multi-layer perceptron implemented with Flax.

Architecture: Dense → ReLU (×n_hidden) → Dense → (tanh | linear)

Initialisation matches SB3's MlpPolicy (ActorCriticPolicy with ortho_init=True):
  - Hidden layers : orthogonal initialisation, gain = sqrt(2)  (ReLU)
  - Actor output  : orthogonal initialisation, gain = output_scale (default 0.01)
  - Biases        : zero
"""

import math
from typing import Sequence

import jax.numpy as jnp
import flax.linen as nn

_SQRT2 = math.sqrt(2.0)


def _ortho(gain: float):
    return nn.initializers.orthogonal(scale=gain)


class MLP(nn.Module):
    """Actor MLP with optional tanh output squashing.

    Args:
        hidden_sizes:   Width of each hidden layer, e.g. [64, 64].
        out_dim:        Output dimension (action dim for actor).
        squash_output:  If True (default), apply tanh to the output.
        output_scale:   Orthogonal init gain for the output layer.
                        Use 0.01 for the actor (near-zero initial means), 1.0 otherwise.
    """
    hidden_sizes: Sequence[int]
    out_dim: int
    squash_output: bool = True
    output_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_sizes:
            x = nn.Dense(size, kernel_init=_ortho(_SQRT2), bias_init=nn.initializers.zeros)(x)
            x = nn.relu(x)
        x = nn.Dense(self.out_dim,
                     kernel_init=_ortho(float(self.output_scale)),
                     bias_init=nn.initializers.zeros)(x)
        return jnp.tanh(x) if self.squash_output else x


