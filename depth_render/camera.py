"""
camera.py — Pinhole depth camera.

Coordinate convention: X=forward, Y=right, Z=down (right-hand, NED-style).

Key function: generate_rays(position, quaternion, fov_deg, width, height)
  Pure JAX — JIT-able and vmap-able.

Camera class: thin config wrapper; delegates to generate_rays.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Quaternion → camera basis
# ---------------------------------------------------------------------------

def _quat_to_basis(
    q: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute camera basis from unit quaternion [qw, qx, qy, qz].

    Body-frame convention (NED: X=forward, Y=right, Z=down):
        forward = R @ [1,  0,  0]   (body +X)
        right   = R @ [0,  1,  0]   (body +Y)
        cam_up  = R @ [0,  0, -1]   (body -Z; Z-down so -Z is up)
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    R = jnp.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=jnp.float32)
    forward = R @ jnp.array([1.,  0.,  0.], dtype=jnp.float32)
    right   = R @ jnp.array([0.,  1.,  0.], dtype=jnp.float32)
    cam_up  = R @ jnp.array([0.,  0., -1.], dtype=jnp.float32)
    return forward, right, cam_up


# ---------------------------------------------------------------------------
# Standalone ray generator — pure JAX, JIT/vmap-able
# ---------------------------------------------------------------------------

def generate_rays(
    position:   jnp.ndarray,   # (3,)
    quaternion: jnp.ndarray,   # (4,) [qw, qx, qy, qz]
    fov_deg:    float,          # horizontal FOV (static)
    width:      int,            # image width in pixels (static)
    height:     int,            # image height in pixels (static)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate one ray per pixel.

    Returns:
        origins: (H*W, 3) — all equal to position
        dirs:    (H*W, 3) — unit direction per pixel
    """
    pos                    = jnp.asarray(position,   dtype=jnp.float32)
    forward, right, cam_up = _quat_to_basis(jnp.asarray(quaternion, dtype=jnp.float32))

    aspect = width / height
    tan_h  = jnp.tan(jnp.radians(fov_deg / 2.0))
    tan_v  = tan_h / aspect

    u = (jnp.arange(width,  dtype=jnp.float32) + 0.5) / width  * 2.0 - 1.0
    v = (jnp.arange(height, dtype=jnp.float32) + 0.5) / height * 2.0 - 1.0
    v = -v   # row 0 → top of image

    uu, vv = jnp.meshgrid(u, v)   # (H, W)

    dirs = (
          uu[..., None] * tan_h * right
        + vv[..., None] * tan_v * cam_up
        + forward[None, None, :]
    )                              # (H, W, 3)
    dirs    = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = jnp.broadcast_to(pos, dirs.shape)
    return origins.reshape(-1, 3), dirs.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Camera config dataclass — thin wrapper around generate_rays
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    """
    Pinhole depth camera config.

    Attributes:
        position:   (3,) world-space camera origin
        quaternion: (4,) [qw, qx, qy, qz] body orientation.
                    Looks along body +X; up is body -Z (NED).
        fov_deg:    Horizontal field of view in degrees
        width:      Image width in pixels
        height:     Image height in pixels
    """
    position:   Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float]
    fov_deg:    float = 90.0
    width:      int   = 320
    height:     int   = 240

    def generate_rays(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return generate_rays(
            jnp.array(self.position,   dtype=jnp.float32),
            jnp.array(self.quaternion, dtype=jnp.float32),
            self.fov_deg, self.width, self.height,
        )

    @property
    def fov_v_deg(self) -> float:
        import math
        tan_h = math.tan(math.radians(self.fov_deg / 2.0))
        tan_v = tan_h / (self.width / self.height)
        return math.degrees(math.atan(tan_v)) * 2.0

    def __repr__(self) -> str:
        return (
            f"Camera(pos={self.position}, fov_h={self.fov_deg}°,"
            f" fov_v={self.fov_v_deg:.1f}°, {self.width}×{self.height})"
        )
