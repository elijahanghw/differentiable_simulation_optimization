"""
renderer.py — Depth renderer.

render_depth(...) → (H, W) float32 JAX array.

Pure JAX: no Python loops over scene primitives at call time.
JIT-able and vmap-able over environment batches.

Scene geometry is passed as stacked JAX arrays (one row per primitive).
Primitive counts (Ns, Nb, Nc) are static — determined by the SceneConfig at
construction time and fixed for the lifetime of the environment.

A ground plane at Z=0 (NED: normal pointing up = [0,0,-1]) is always included.

apply_sensor_noise(...) applies realistic depth-camera artifacts on top.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .camera     import generate_rays
from .primitives import ray_sphere, ray_aabb, ray_capsule, ray_infinite_plane


# ---------------------------------------------------------------------------
# Ground plane constants (NED: Z-down, so Z=0 is the ground, normal = -Z)
# ---------------------------------------------------------------------------

_GROUND_PT  = jnp.zeros(3,  dtype=jnp.float32)
_GROUND_NRM = jnp.array([0., 0., -1.], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Per-ray depth computation
# ---------------------------------------------------------------------------

def _depth_for_ray(
    o: jnp.ndarray,               # (3,) ray origin
    d: jnp.ndarray,               # (3,) unit ray direction
    sphere_centers:   jnp.ndarray,  # (Ns, 3)
    sphere_radii:     jnp.ndarray,  # (Ns,)
    box_centers:      jnp.ndarray,  # (Nb, 3)
    box_half_extents: jnp.ndarray,  # (Nb, 3)
    capsule_centers:  jnp.ndarray,  # (Nc, 3)
    capsule_axes:     jnp.ndarray,  # (Nc, 3)
    capsule_hh:       jnp.ndarray,  # (Nc,)
    capsule_radii:    jnp.ndarray,  # (Nc,)
) -> jnp.ndarray:                   # scalar float32
    """Nearest hit distance for a single ray against all scene primitives."""
    depth = jnp.array(jnp.inf, dtype=jnp.float32)

    # Ground plane (always present)
    depth = jnp.minimum(depth, ray_infinite_plane(o, d, _GROUND_PT, _GROUND_NRM))

    # Spheres — vmap over Ns primitives; Python `if` is static at trace time
    if sphere_centers.shape[0] > 0:
        ts = jax.vmap(lambda c, r: ray_sphere(o, d, c, r))(
            sphere_centers, sphere_radii
        )
        depth = jnp.minimum(depth, jnp.min(ts))

    # Axis-aligned boxes — vmap over Nb primitives
    if box_centers.shape[0] > 0:
        ts = jax.vmap(lambda c, he: ray_aabb(o - c, d, -he, he))(
            box_centers, box_half_extents
        )
        depth = jnp.minimum(depth, jnp.min(ts))

    # Capsules — vmap over Nc primitives
    if capsule_centers.shape[0] > 0:
        ts = jax.vmap(lambda c, ax, hh, r: ray_capsule(o, d, c, ax, hh, r))(
            capsule_centers, capsule_axes, capsule_hh, capsule_radii
        )
        depth = jnp.minimum(depth, jnp.min(ts))

    return depth


# ---------------------------------------------------------------------------
# Main renderer — pure JAX, JIT/vmap-able
# ---------------------------------------------------------------------------

def render_depth(
    position:         jnp.ndarray,   # (3,)
    quaternion:       jnp.ndarray,   # (4,) [qw, qx, qy, qz]
    fov_deg:          float,          # static
    width:            int,            # static
    height:           int,            # static
    sphere_centers:   jnp.ndarray,   # (Ns, 3)
    sphere_radii:     jnp.ndarray,   # (Ns,)
    box_centers:      jnp.ndarray,   # (Nb, 3)
    box_half_extents: jnp.ndarray,   # (Nb, 3)
    capsule_centers:  jnp.ndarray,   # (Nc, 3)
    capsule_axes:     jnp.ndarray,   # (Nc, 3)
    capsule_hh:       jnp.ndarray,   # (Nc,)
    capsule_radii:    jnp.ndarray,   # (Nc,)
) -> jnp.ndarray:                    # (H, W) float32
    """
    Render a depth image.

    All geometry is passed as JAX arrays — no Python scene objects.
    Safe to jax.jit and jax.vmap over environment batches.

    Args:
        position:   (3,)      drone / camera world position.
        quaternion: (4,)      drone body quaternion [qw, qx, qy, qz].
        fov_deg:    float     horizontal FOV in degrees (compile-time constant).
        width/height: int     image resolution (compile-time constants).
        sphere_*/box_*/capsule_*: stacked geometry arrays from get_scene_arrays().

    Returns:
        (H, W) float32 — ray-traced depth in metres, jnp.inf on no-hit.
    """
    rays_o, rays_d = generate_rays(position, quaternion, fov_deg, width, height)

    depth_flat = jax.vmap(
        lambda o, d: _depth_for_ray(
            o, d,
            sphere_centers, sphere_radii,
            box_centers, box_half_extents,
            capsule_centers, capsule_axes, capsule_hh, capsule_radii,
        )
    )(rays_o, rays_d)

    return depth_flat.reshape(height, width)


# ---------------------------------------------------------------------------
# Sensor noise model
# ---------------------------------------------------------------------------

def apply_sensor_noise(
    depth: jnp.ndarray,
    min_range: float = 0.2,
    max_range: float = 15.0,
    quantization_m: float = 0.001,
) -> jnp.ndarray:
    """
    Apply realistic depth-camera artifacts to an ideal depth image.

    Args:
        depth:          (H, W) float32 — ideal ray-traced depth in metres.
                        jnp.inf indicates no geometry hit.
        min_range:      Pixels closer than this → 0  (blind zone).
        max_range:      Pixels farther than this, or no-hit (inf) → max_range.
        quantization_m: Depth is rounded to multiples of this value.

    Returns:
        (H, W) float32 — depth in metres.
        max_range = no hit / out of range.  0 = closer than min_range.
    """
    d = jnp.where(jnp.isfinite(depth), depth, max_range)
    d = jnp.minimum(d, max_range)
    d = jnp.where(d < min_range, 0.0, d)

    if quantization_m > 0.0:
        d = jnp.where(
            d > 0.0,
            jnp.round(d / quantization_m) * quantization_m,
            0.0,
        )

    return d
