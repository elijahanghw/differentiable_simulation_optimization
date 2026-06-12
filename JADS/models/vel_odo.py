"""
VelOdoActor — multi-head CNN-GRU for GPS-denied navigation.

A single shared backbone (2-channel depth CNN + GRU) with two output heads:
    1. action head  → motor commands  (act_dim,)
    2. vel head     → world-frame velocity estimate  (3,)

The obs_vec mirrors the ground-truth layout so policy capacity is unchanged:
    GT mode:  [rel_pos(3),    vel(3),      euler(3), omega(3), W(6)]
    Odo mode: [target_est(3), vel_prev(3), euler(3), omega(3), W(6)]

where target_est and vel_prev are maintained by the BPTT carry:
    pos_est_{t+1}    = pos_est_t + vel_est_t * dt
    target_est_{t+1} = init_rel_pos + pos_est_{t+1}
    vel_prev_{t+1}   = vel_est_t

An auxiliary supervised loss ||vel_est - vel_gt||² guides the vel head
using ground-truth velocity from the simulator.

Inputs to __call__:
    depth_t    (H, W)   — current processed depth
    depth_prev (H, W)   — previous processed depth (zeros at episode start)
    vel_prev   (3,)     — vel_est from previous step  (zeros at episode start)
    target_est (3,)     — dead-reckoned position relative to target
    imu_vec    (12,)    — concat([euler(3), omega(3), W(6)])
    hidden     (hidden_size,)

Returns:
    action     (act_dim,)
    new_hidden (hidden_size,)
    vel_est    (3,)
"""

from typing import Sequence

import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as nn


class VelOdoActor(nn.Module):
    act_dim:       int
    conv_features: Sequence[int]           = (32, 64, 128)
    kernel_sizes:  Sequence[Sequence[int]] = ((2, 2), (3, 3), (3, 3))
    strides:       Sequence[Sequence[int]] = ((2, 2), (1, 1), (1, 1))
    proj_dim:      int   = 192
    leaky_slope:   float = 0.05
    hidden_size:   int   = 192
    squash_output: bool  = False

    @nn.compact
    def __call__(self, depth_t, depth_prev, vel_prev, target_est, imu_vec, hidden):
        # ---- CNN on depth pair (2-channel) ----------------------------------
        # channel 0: current depth  → obstacle avoidance
        # channel 1: previous depth → optical-flow / velocity signal
        depth_pair = jnp.stack([depth_t, depth_prev], axis=-1)  # (H, W, 2)
        x = depth_pair
        for features, ks, st in zip(self.conv_features, self.kernel_sizes, self.strides):
            x = jnn.leaky_relu(
                nn.Conv(features, kernel_size=tuple(ks), strides=tuple(st), use_bias=False)(x),
                self.leaky_slope,
            )
        img_feat = nn.Dense(self.proj_dim)(x.reshape(-1))

        # ---- Obs projection (mirrors GT layout) -----------------------------
        obs_vec  = jnp.concatenate([target_est, vel_prev, imu_vec])  # 3+3+12 = 18
        obs_feat = nn.Dense(self.proj_dim, use_bias=False)(obs_vec)

        # ---- GRU ------------------------------------------------------------
        fused         = jnn.leaky_relu(img_feat + obs_feat, self.leaky_slope)
        new_hidden, h = nn.GRUCell(self.hidden_size)(hidden, fused)
        h             = jnn.leaky_relu(h, self.leaky_slope)

        # ---- Dual output heads ----------------------------------------------
        action  = nn.Dense(self.act_dim, use_bias=False)(h)
        vel_est = nn.Dense(3)(h)                             # no activation — can be negative
        if self.squash_output:
            action = jnp.tanh(action)

        return action, new_hidden, vel_est

    def init_hidden(self):
        return jnp.zeros(self.hidden_size)
