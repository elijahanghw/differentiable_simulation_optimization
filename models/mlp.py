"""
Multi-layer perceptron for continuous control.

Architecture: Linear → ReLU (×n_hidden) → Linear → (Tanh | linear)
Parameters are stored as a plain list of (W, b) tuples — a valid JAX pytree
that can be passed directly to jax.grad / jax.vmap.

Also provides a Critic class (same architecture but linear final layer) for
use with PPO and other actor-critic algorithms.

Initialisation matches SB3's MlpPolicy (ActorCriticPolicy with ortho_init=True):
  - Hidden layers : orthogonal initialisation, gain = sqrt(2)  (ReLU)
  - Actor output  : orthogonal initialisation, gain = output_scale (default 0.01)
  - Critic output : orthogonal initialisation, gain = 1.0
  - Biases        : zero
"""

from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp

# Type alias: a parameter tree is a list of (weight, bias) pairs
Params = List[Tuple[jnp.ndarray, jnp.ndarray]]


# ---------------------------------------------------------------------------
# Orthogonal initialisation (matches torch.nn.init.orthogonal_)
# ---------------------------------------------------------------------------

def _ortho_init(key: jax.Array, shape: Tuple[int, int], gain: float = 1.0) -> jnp.ndarray:
    """
    Orthogonal initialisation via QR decomposition, matching PyTorch's
    nn.init.orthogonal_(tensor, gain).

    For a (fan_in, fan_out) weight matrix:
      1. Draw Z ~ N(0, 1) with shape (max, min).
      2. QR-decompose Z = QR; adjust Q's sign so diag(R) > 0.
      3. Truncate / transpose to (fan_in, fan_out) and scale by gain.
    """
    rows, cols = shape
    flat_shape = (rows, cols) if rows >= cols else (cols, rows)
    z = jax.random.normal(key, flat_shape)
    q, r = jnp.linalg.qr(z)
    # Make Q uniform — flip column signs to match sign of diagonal of R
    q = q * jnp.sign(jnp.diag(r))
    if rows < cols:
        q = q.T
    return gain * q


def _init_params(
    key: jax.Array,
    layer_sizes: Sequence[int],
    output_scale: float = 1.0,
) -> Params:
    """
    Orthogonal initialisation for a stack of linear layers.

    Args:
        key:          JAX PRNGKey.
        layer_sizes:  e.g. [obs_dim, 64, 64, act_dim].
        output_scale: gain for the final layer (use 0.01 for actor, 1.0 for critic).
                      Hidden layers always use gain = sqrt(2) (optimal for ReLU).
    """
    params = []
    n_layers = len(layer_sizes) - 1
    for i, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        key, subkey = jax.random.split(key)
        gain = float(output_scale) if i == n_layers - 1 else float(jnp.sqrt(2.0))
        W = _ortho_init(subkey, (fan_in, fan_out), gain=gain)
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


class MLP:
    def __init__(
        self,
        layer_sizes: Sequence[int],
        squash_output: bool = True,
        output_scale: float = 1.0,
    ):
        """
        Args:
            layer_sizes:    full layer spec including input and output dims,
                            e.g. [obs_dim, 64, 64, act_dim].
            squash_output:  if True (default), apply tanh to the output layer.
                            Set to False for PPO (matches SB3's linear actor output).
            output_scale:   orthogonal init gain for the output layer.
                            SB3 uses 0.01 for the actor (near-zero initial means)
                            and 1.0 for the critic.  Default 1.0 preserves the
                            previous behaviour for APG.
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least input and output dims.")
        self.layer_sizes = list(layer_sizes)
        self.squash_output = squash_output
        self.output_scale = output_scale

    def init(self, key: jax.Array) -> Params:
        return _init_params(key, self.layer_sizes, output_scale=self.output_scale)

    def apply(self, params: Params, x: jnp.ndarray) -> jnp.ndarray:
        """Hidden layers use ReLU; output uses tanh or linear per squash_output."""
        for W, b in params[:-1]:
            x = jax.nn.relu(x @ W + b)
        W, b = params[-1]
        out = x @ W + b
        return jnp.tanh(out) if self.squash_output else out


class Critic:
    """MLP value function with a linear (unbounded) scalar output.

    Initialised with orthogonal gain = 1.0 on the output layer, matching SB3.
    Architecture: Linear → ReLU (×n_hidden) → Linear (no activation)
    """

    def __init__(self, layer_sizes: Sequence[int]):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least input and output dims.")
        self.layer_sizes = list(layer_sizes)

    def init(self, key: jax.Array) -> Params:
        return _init_params(key, self.layer_sizes, output_scale=1.0)

    def apply(self, params: Params, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass — hidden layers use ReLU, output is linear."""
        for W, b in params[:-1]:
            x = jax.nn.relu(x @ W + b)
        W, b = params[-1]
        return x @ W + b
