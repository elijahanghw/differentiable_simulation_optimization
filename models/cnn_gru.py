"""
CNN + GRU Actor implemented with Flax.

Architecture matches DiffPhysDrone v1.0 (model.py):
    depth → CNNEncoder → proj_dim features
    obs   → Dense      → proj_dim features
    fused = LeakyReLU(img_feat + obs_feat)
    GRU cell over fused
    action = Dense(LeakyReLU(gru_out))
    [optional tanh squash]
"""

from typing import Sequence

import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as nn


class CNNEncoder(nn.Module):
    """
    Depth image → flat feature vector.

    Input:  (H, W)      — single-channel depth image (e.g. 12×16 after 4×4 pool)
    Output: (proj_dim,) — projected feature vector
    """
    conv_features: Sequence[int]          = (32, 64, 128)
    kernel_sizes:  Sequence[Sequence[int]] = ((2, 2), (3, 3), (3, 3))
    strides:       Sequence[Sequence[int]] = ((2, 2), (1, 1), (1, 1))
    proj_dim:      int   = 192
    leaky_slope:   float = 0.05

    @nn.compact
    def __call__(self, depth):
        x = depth[..., None]    # (H, W) → (H, W, 1)
        for features, ks, st in zip(self.conv_features, self.kernel_sizes, self.strides):
            x = jnn.leaky_relu(
                nn.Conv(features, kernel_size=tuple(ks), strides=tuple(st), use_bias=False)(x),
                self.leaky_slope,
            )
        x = x.reshape(-1)
        return nn.Dense(self.proj_dim)(x)


class CNNGRUActor(nn.Module):
    act_dim:       int
    conv_features: Sequence[int]          = (32, 64, 128)
    kernel_sizes:  Sequence[Sequence[int]] = ((2, 2), (3, 3), (3, 3))
    strides:       Sequence[Sequence[int]] = ((2, 2), (1, 1), (1, 1))
    proj_dim:      int   = 192
    leaky_slope:   float = 0.05
    hidden_size:   int   = 192
    squash_output: bool  = False

    @nn.compact
    def __call__(self, depth, obs, hidden):
        img_feat  = CNNEncoder(
            conv_features=self.conv_features,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            proj_dim=self.proj_dim,
            leaky_slope=self.leaky_slope,
        )(depth)
        obs_feat  = nn.Dense(self.proj_dim, use_bias=False)(obs)
        fused     = jnn.leaky_relu(img_feat + obs_feat, self.leaky_slope)
        new_hidden, x = nn.GRUCell(self.hidden_size)(hidden, fused)
        x         = jnn.leaky_relu(x, self.leaky_slope)
        action    = nn.Dense(self.act_dim, use_bias=False)(x)
        if self.squash_output:
            action = jnp.tanh(action)
        return action, new_hidden

    def init_hidden(self):
        return jnp.zeros(self.hidden_size)
