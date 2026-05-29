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
from .primitives import ray_sphere, ray_aabb, ray_cylinder, ray_infinite_plane


# ---------------------------------------------------------------------------
# Ground plane constants (NED: Z-down, so Z=0 is the ground, normal = -Z)
# ---------------------------------------------------------------------------

_GROUND_PT  = jnp.zeros(3,  dtype=jnp.float32)
_GROUND_NRM = jnp.array([0., 0., -1.], dtype=jnp.float32)


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
    cylinder_centers: jnp.ndarray,   # (Nc, 3)
    cylinder_axes:    jnp.ndarray,   # (Nc, 3)
    cylinder_hh:      jnp.ndarray,   # (Nc,)
    cylinder_radii:   jnp.ndarray,   # (Nc,)
) -> jnp.ndarray:                    # (H, W) float32
    """
    Render a depth image.

    All geometry is passed as JAX arrays — no Python scene objects.
    Safe to jax.jit and jax.vmap over environment batches.

    Capsules are composite: each is 1 cylinder + 2 end-cap spheres.
    The sphere array already includes end-cap spheres; cylinder_* holds
    the lateral-body geometry.

    Vmap order: outer over primitives (small), inner over rays (large).
    Each GPU thread handles one ray vs one primitive — low register pressure
    and no shared-memory blowup regardless of primitive count.

    Args:
        position:   (3,)      drone / camera world position.
        quaternion: (4,)      drone body quaternion [qw, qx, qy, qz].
        fov_deg:    float     horizontal FOV in degrees (compile-time constant).
        width/height: int     image resolution (compile-time constants).
        sphere_*/box_*/cylinder_*: stacked geometry arrays from SceneConfig.unpack().

    Returns:
        (H, W) float32 — ray-traced depth in metres, jnp.inf on no-hit.
    """
    rays_o, rays_d = generate_rays(position, quaternion, fov_deg, width, height)

    # Ground plane — one vmap over all rays
    depth = jax.vmap(
        lambda o, d: ray_infinite_plane(o, d, _GROUND_PT, _GROUND_NRM)
    )(rays_o, rays_d)

    # Spheres: (Ns, N_rays) — outer vmap over primitives, inner over rays
    if sphere_centers.shape[0] > 0:
        sphere_depths = jax.vmap(
            lambda c, r: jax.vmap(lambda o, d: ray_sphere(o, d, c, r))(rays_o, rays_d)
        )(sphere_centers, sphere_radii)
        depth = jnp.minimum(depth, jnp.min(sphere_depths, axis=0))

    # Boxes: (Nb, N_rays)
    if box_centers.shape[0] > 0:
        box_depths = jax.vmap(
            lambda c, he: jax.vmap(lambda o, d: ray_aabb(o - c, d, -he, he))(rays_o, rays_d)
        )(box_centers, box_half_extents)
        depth = jnp.minimum(depth, jnp.min(box_depths, axis=0))

    # Cylinders: (Nc, N_rays)
    if cylinder_centers.shape[0] > 0:
        cylinder_depths = jax.vmap(
            lambda c, ax, hh, r: jax.vmap(lambda o, d: ray_cylinder(o, d, c, ax, hh, r))(rays_o, rays_d)
        )(cylinder_centers, cylinder_axes, cylinder_hh, cylinder_radii)
        depth = jnp.minimum(depth, jnp.min(cylinder_depths, axis=0))

    return depth.reshape(height, width)


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
