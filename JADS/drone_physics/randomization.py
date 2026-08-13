"""Identified drone parameters and their per-episode domain randomization."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .dynamics_real import G


class DroneParams(NamedTuple):
    """Drone parameters, in the order ``dynamics()`` unpacks them.

    The force and moment coefficients are already mass- and inertia-normalized
    (they map rad²/s² directly to m/s² and rad/s²), which is why there is no
    separate m or J here. Per-motor coefficients are (n_motors,) arrays; the
    rest are scalars. As a NamedTuple this is a pytree, so it vmaps and scans
    like any other carry field.
    """
    k_wx:    jnp.ndarray
    k_x:     jnp.ndarray
    k_wy:    jnp.ndarray
    k_y:     jnp.ndarray
    k_wz:    jnp.ndarray
    k_z:     jnp.ndarray
    k_p:     jnp.ndarray
    k_pd:    jnp.ndarray
    k_q:     jnp.ndarray
    k_qd:    jnp.ndarray
    k_r:     jnp.ndarray
    k_rd:    jnp.ndarray
    tau:     jnp.ndarray
    k:       jnp.ndarray
    w_min:   jnp.ndarray
    w_max:   jnp.ndarray
    W_MIN_N: jnp.ndarray
    W_MAX_N: jnp.ndarray


# Normalization constants, not physics: they define what w ∈ [-1,1] *means* in
# the state and the observation. Randomizing them would make the same w denote a
# different motor speed in every episode, so the policy could not interpret its
# own motor observation. Held at nominal.
_FIXED = ("W_MIN_N", "W_MAX_N")


def load(cfg: dict) -> DroneParams:
    """Build nominal params from a config `drone_params` block."""
    missing = [f for f in DroneParams._fields if f not in cfg]
    if missing:
        raise ValueError(f"drone_params missing keys: {missing}")
    return DroneParams(**{
        f: jnp.asarray(cfg[f], dtype=jnp.float32) for f in DroneParams._fields
    })


def randomize(params: DroneParams, key: jax.Array, dr: float) -> DroneParams:
    """Scale every physical parameter by an independent 1 ± dr factor.

    Draw this once per episode at reset and hold it for the episode's duration:
    domain randomization models flying a *different airframe*, not per-step
    noise. Each element of the per-motor arrays gets its own factor, so
    motor-to-motor asymmetry is randomized along with the overall gains.

    Multiplicative scaling fixes zero at zero, so coefficients the
    identification returned as exactly 0 (k_wx, k_wy, k_z for the 2" whoop)
    stay 0 — they need additive noise if you want them explored.
    """
    if dr <= 0.0:
        return params

    fields = [f for f in DroneParams._fields if f not in _FIXED]
    keys   = jax.random.split(key, len(fields))
    out    = dict(params._asdict())
    for f, kk in zip(fields, keys):
        v  = out[f]
        # min/max rather than (1-dr, 1+dr) directly: for a negative coefficient
        # v·0.7 > v·1.3, and uniform() needs its bounds ordered.
        lo = jnp.minimum(v * (1.0 - dr), v * (1.0 + dr))
        hi = jnp.maximum(v * (1.0 - dr), v * (1.0 + dr))
        if f == "k":
            # k shapes the throttle curve, it is not a gain: at k > 1 the
            # (1-k)·u term goes negative and opens a dead zone at low throttle
            # where the sqrt argument is negative (clamped to 0, so W_c pins at
            # w_min). Truncate the *interval*, not the sample — clipping the
            # sample would leave an atom of probability sitting on k = 1.
            hi = jnp.minimum(hi, 1.0)
        out[f] = jax.random.uniform(kk, jnp.shape(v), minval=lo, maxval=hi)

    # Keep the motor speed band ordered. Inactive at dr = 0.3 (w_min·1.3 is far
    # below w_max·0.7); it only matters if dr is raised a lot.
    out["w_max"] = jnp.maximum(out["w_max"], out["w_min"] + 1.0)
    return DroneParams(**out)


def hover_trim(params: DroneParams) -> tuple:
    """Per-motor state ``w`` and command ``U`` that hold a level hover.

    At equilibrium the motor lag term vanishes (d_W = 0), so trim is *linear in
    W²*: thrust cancels gravity and all three moments vanish. Solving that
    system gives per-motor speeds that are deliberately **not** equal — the
    identified moment coefficients do not sum to zero (the 2" whoop has a ~10%
    front/rear pitch asymmetry), so uniform throttle leaves a standing moment
    and the drone tips over. Useful as a reset default and as a check that a
    params set is trimmable at all (all four W² must come out positive).
    """
    n = jnp.shape(params.k_p)[0]
    A = jnp.stack([
        -params.k_wz * jnp.ones(n),   # Fz = -k_wz·ΣW², must cancel gravity
        params.k_p,                   # Mx = 0
        params.k_q,                   # My = 0
        params.k_r,                   # Mz = 0
    ])
    b  = jnp.array([-G, 0.0, 0.0, 0.0])
    W2 = jnp.linalg.lstsq(A, b, rcond=None)[0]
    W  = jnp.sqrt(jnp.maximum(W2, 0.0))
    w  = 2.0 * W / params.W_MAX_N - 1.0
    # Invert s = sqrt(k·u² + (1-k)·u) for u, taking the positive root.
    s = jnp.clip((W - params.w_min) / (params.w_max - params.w_min), 0.0, 1.0)
    k = params.k
    u = (-(1.0 - k) + jnp.sqrt((1.0 - k) ** 2 + 4.0 * k * s ** 2)) / (2.0 * k)
    return w, 2.0 * u - 1.0
