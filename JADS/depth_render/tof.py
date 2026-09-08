"""
tof.py — Multizone time-of-flight sensor (VL53L8CX-class 8×8 array).

A VL53L8CX is not a camera: it is an 8×8 grid of SPAD zones behind one lens,
each zone reporting a single distance in millimetres over a ~5.6° cone. The
optics are still a focal-plane array, so the zone grid is rectilinear — the
same pinhole projection depth_render/camera.py already generates, only very
coarse.

Sensor model:
    1. Trace the scene at `supersample`× the zone resolution over the sensor's
       square FoV — S² sub-rays per zone.
    2. Collapse each S×S block into one zone distance (`zone_reduce`). "min"
       is the default: a zone latches onto the nearest strong reflector inside
       its cone, which is also the conservative choice for obstacle avoidance.
    3. Range saturation, blind zone and mm quantization are the same
       post-processing the depth camera uses — renderer.apply_sensor_noise.

Equivalence with the depth path: the observation normalizer
`3/clip(d, 0.3, max) - 0.6` is monotonically *decreasing* in d, so max-pooling
the normalized fine grid by S is identical to min-pooling the raw fine grid by
S. A ToF frame with zone_agg="min" is therefore exactly what the existing depth
pipeline produces from a width=height=zones*S camera pooled by S — including
the C++ port in hitl/src/render.cpp, which needs no ToF-specific code, only the
camera block hitl/tools/export_scene.py now writes for it.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# VL53L8CX reference numbers (ST datasheet)
# ---------------------------------------------------------------------------

VL53L8CX_ZONES        = 8      # 8×8 mode (4×4 also exists, at 60 Hz)
VL53L8CX_FOV_DEG      = 45.0   # per side; ST quotes "65° diagonal, square FoV"
VL53L8CX_MAX_RANGE_M  = 4.0    # 400 cm in the dark; less in bright ambient
VL53L8CX_MIN_RANGE_M  = 0.1    # returns below this are unreliable
VL53L8CX_QUANT_M      = 0.001  # reported in mm as uint16
VL53L8CX_RATE_HZ      = 15.0   # max ranging frequency in 8×8 mode


def square_fov_from_diagonal(diag_deg: float) -> float:
    """
    Per-side FoV of a square sensor quoted by its diagonal FoV.

    Rectilinear optics: tan(diag/2) = sqrt(2)·tan(side/2), so ST's 65°
    diagonal is 48.5° per side. The datasheet's own "45°×45°" figure instead
    uses the linear approximation 65/sqrt(2) ≈ 46°. Both readings are in the
    same few degrees; `fov_deg` is configurable so you can pick one.
    """
    tan_side = math.tan(math.radians(diag_deg / 2.0)) / math.sqrt(2.0)
    return math.degrees(math.atan(tan_side)) * 2.0


# ---------------------------------------------------------------------------
# Zone aggregation — pure JAX, JIT/vmap-able
# ---------------------------------------------------------------------------

def zone_reduce(
    fine_depth:  jnp.ndarray,   # (Zh*S, Zw*S) metres, jnp.inf on no-hit
    supersample: int,           # S (static)
    max_range:   float,         # saturation distance (static)
    agg:         str = "min",   # "min" | "mean" (static)
) -> jnp.ndarray:               # (Zh, Zw)
    """
    Collapse a supersampled depth render into one distance per ToF zone.

    No-hit sub-rays (jnp.inf) saturate to `max_range` first, so "mean" stays
    finite and "min" behaves the same as it would on the raw grid.

    Args:
        fine_depth:  ray-traced depth at supersample× the zone resolution.
        supersample: sub-rays per zone edge; S² rays land in each zone.
        max_range:   sensor saturation distance in metres.
        agg:         "min"  — nearest reflector in the cone (default, and what
                              the real sensor's strongest-target return
                              approximates for obstacle avoidance);
                     "mean" — average over the cone, a softer signal that
                              blurs edges the way a signal-weighted centroid
                              would.

    Returns:
        (Zh, Zw) float32 — per-zone distance in metres, finite everywhere.
    """
    d = jnp.where(jnp.isfinite(fine_depth), fine_depth, max_range)
    d = jnp.minimum(d, max_range)

    if supersample == 1:
        return d

    window = (supersample, supersample)
    if agg == "min":
        return jax.lax.reduce_window(
            d, jnp.inf, jax.lax.min,
            window_dimensions=window, window_strides=window, padding="VALID",
        )
    if agg == "mean":
        summed = jax.lax.reduce_window(
            d, 0.0, jax.lax.add,
            window_dimensions=window, window_strides=window, padding="VALID",
        )
        return summed / float(supersample * supersample)
    raise ValueError(f"unknown zone_agg '{agg}' — expected 'min' or 'mean'")
