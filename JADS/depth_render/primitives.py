"""
primitives.py — JAX ray-primitive intersection functions.

Each function takes a single ray (origin, direction) and a single primitive,
returning the hit distance t (jnp.inf on miss).  Designed for jax.vmap.

Coordinate convention: X=forward, Y=right, Z=down (NED-style, Z-down positive).
"""
import jax.numpy as jnp


def ray_sphere(
    ray_o: jnp.ndarray,
    ray_d: jnp.ndarray,
    center: jnp.ndarray,
    radius: float,
) -> jnp.ndarray:
    """
    Ray–sphere intersection (analytic).

    Args:
        ray_o:  (3,) ray origin
        ray_d:  (3,) unit ray direction
        center: (3,) sphere centre
        radius: sphere radius

    Returns:
        Scalar t ≥ 0 (distance to nearest hit), or jnp.inf on miss.
    """
    oc = ray_o - center
    b  = jnp.dot(oc, ray_d)
    c  = jnp.dot(oc, oc) - radius * radius
    disc = b * b - c

    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t_near = -b - sqrt_disc   # front surface
    t_far  = -b + sqrt_disc   # back surface (camera inside sphere)

    t = jnp.where(t_near > 1e-4, t_near,
        jnp.where(t_far  > 1e-4, t_far, jnp.inf))
    return jnp.where(disc >= 0.0, t, jnp.inf)


def ray_aabb(
    ray_o: jnp.ndarray,
    ray_d: jnp.ndarray,
    box_min: jnp.ndarray,
    box_max: jnp.ndarray,
) -> jnp.ndarray:
    """
    Ray–axis-aligned bounding box intersection (slab method).
    Used internally by ray_obb.
    """
    safe_d = jnp.where(jnp.abs(ray_d) > 1e-10, ray_d,
                       jnp.sign(ray_d + 1e-30) * 1e-10)
    inv_d = 1.0 / safe_d

    t1 = (box_min - ray_o) * inv_d
    t2 = (box_max - ray_o) * inv_d

    t_enter = jnp.max(jnp.minimum(t1, t2))
    t_exit  = jnp.min(jnp.maximum(t1, t2))

    hit = (t_exit >= t_enter) & (t_exit > 1e-4)
    t   = jnp.where(t_enter > 1e-4, t_enter, t_exit)
    return jnp.where(hit, t, jnp.inf)


def _quat_to_rot(q: jnp.ndarray) -> jnp.ndarray:
    """Convert unit quaternion [qw, qx, qy, qz] → 3×3 rotation matrix."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=jnp.float32)


def ray_obb(
    ray_o:        jnp.ndarray,
    ray_d:        jnp.ndarray,
    center:       jnp.ndarray,
    quaternion:   jnp.ndarray,
    half_extents: jnp.ndarray,
) -> jnp.ndarray:
    """
    Ray–oriented bounding box intersection.

    Transforms the ray into the box's local frame using the inverse rotation
    (R^T), then performs a standard AABB slab test.

    Args:
        ray_o:        (3,) ray origin
        ray_d:        (3,) unit ray direction
        center:       (3,) box centre
        quaternion:   (4,) unit quaternion [qw, qx, qy, qz] describing box orientation
        half_extents: (3,) half-extents along local x, y, z axes

    Returns:
        Scalar t ≥ 0, or jnp.inf on miss.
    """
    R = _quat_to_rot(quaternion)          # world ← local
    # Transform ray into box local frame (R^T = R^-1 for unit quaternions)
    local_o = R.T @ (ray_o - center)
    local_d = R.T @ ray_d
    return ray_aabb(local_o, local_d, -half_extents, half_extents)


def ray_cylinder(
    ray_o:  jnp.ndarray,
    ray_d:  jnp.ndarray,
    center: jnp.ndarray,
    axis:   jnp.ndarray,
    half_h: float,
    radius: float,
) -> jnp.ndarray:
    """
    Ray–finite open cylinder intersection (lateral surface only, no end caps).

    End caps are rendered as separate spheres at (center ± half_h * axis).
    Together they form a capsule composite.

    Args:
        ray_o:  (3,) ray origin
        ray_d:  (3,) unit ray direction
        center: (3,) cylinder midpoint
        axis:   (3,) unit axis
        half_h: half-length of the cylindrical section
        radius: cylinder radius

    Returns:
        Scalar t ≥ 0, or jnp.inf on miss.
    """
    AO = ray_o - (center - half_h * axis)

    d_along  = jnp.dot(ray_d, axis)
    ao_along = jnp.dot(AO, axis)

    a_q    = 1.0 - d_along * d_along
    b_half = jnp.dot(ray_d, AO) - d_along * ao_along
    c_q    = jnp.dot(AO, AO) - ao_along * ao_along - radius * radius
    disc   = b_half * b_half - a_q * c_q

    safe_a    = jnp.where(a_q > 1e-12, a_q, 1.0)
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    # Normalize projections so the segment check is against constant [0, 1],
    # avoiding keeping half_h live inside the closure (reduces register pressure).
    inv_len    = 0.5 / half_h
    d_along_n  = d_along  * inv_len
    ao_along_n = ao_along * inv_len

    def cyl_t(t_try):
        proj  = ao_along_n + t_try * d_along_n
        valid = (disc >= 0.0) & (a_q > 1e-12) & (t_try > 1e-4) & (proj >= 0.0) & (proj <= 1.0)
        return jnp.where(valid, t_try, jnp.inf)

    return jnp.minimum(
        cyl_t((-b_half - sqrt_disc) / safe_a),
        cyl_t((-b_half + sqrt_disc) / safe_a),
    )


def ray_infinite_plane(
    ray_o:  jnp.ndarray,
    ray_d:  jnp.ndarray,
    point:  jnp.ndarray,
    normal: jnp.ndarray,
) -> jnp.ndarray:
    """
    Ray–infinite plane intersection.

    Args:
        ray_o:  (3,) ray origin
        ray_d:  (3,) unit ray direction
        point:  (3,) any point on the plane
        normal: (3,) unit plane normal

    Returns:
        Scalar t ≥ 0, or jnp.inf on miss / parallel ray.
    """
    denom = jnp.dot(ray_d, normal)
    t = jnp.where(
        jnp.abs(denom) > 1e-6,
        jnp.dot(point - ray_o, normal) / denom,
        jnp.inf,
    )
    return jnp.where(t > 1e-4, t, jnp.inf)


# ---------------------------------------------------------------------------
# Point-to-primitive signed distance functions
# ---------------------------------------------------------------------------
# Convention: negative = point is inside the primitive (collision).
# All functions are pure JAX — JIT-able and vmap-able.

def _safe_norm(v: jnp.ndarray) -> jnp.ndarray:
    # sqrt(||v||^2 + eps^2): smooth gradient everywhere, no discontinuity at ||v||=0.
    # Biases the norm by at most 1mm — negligible for obstacle radii >= 0.1m.
    return jnp.sqrt(jnp.dot(v, v) + 1e-6)


def point_sphere_dist(
    p:      jnp.ndarray,   # (3,) query point
    center: jnp.ndarray,   # (3,) sphere centre
    radius: float,          # sphere radius
) -> jnp.ndarray:           # scalar
    """Signed distance from p to sphere surface. Negative = inside."""
    return _safe_norm(p - center) - radius


def point_aabb_dist(
    p:            jnp.ndarray,   # (3,) query point
    center:       jnp.ndarray,   # (3,) box centre
    half_extents: jnp.ndarray,   # (3,) half-extents (axis-aligned)
) -> jnp.ndarray:                # scalar
    """Signed distance from p to axis-aligned box surface. Negative = inside."""
    d = jnp.abs(p - center) - half_extents
    exterior = jnp.maximum(d, 0.0)
    return _safe_norm(exterior) + jnp.minimum(jnp.max(d), 0.0)


def point_obb_dist(
    p:            jnp.ndarray,   # (3,) query point
    center:       jnp.ndarray,   # (3,) box centre
    quaternion:   jnp.ndarray,   # (4,) [qw,qx,qy,qz] box orientation
    half_extents: jnp.ndarray,   # (3,) half-extents
) -> jnp.ndarray:                # scalar
    """
    Signed distance from p to OBB surface. Negative = inside.

    Transforms p into the box local frame then evaluates the box SDF.
    """
    R = _quat_to_rot(quaternion)
    local_p = R.T @ (p - center)
    d = jnp.abs(local_p) - half_extents
    # exterior component (0 when inside): length of positive part of d
    # interior component (≤ 0 when inside): most-positive axis when inside
    exterior = jnp.maximum(d, 0.0)
    return _safe_norm(exterior) + jnp.minimum(jnp.max(d), 0.0)


def point_capsule_dist(
    p:      jnp.ndarray,   # (3,) query point
    center: jnp.ndarray,   # (3,) capsule midpoint
    axis:   jnp.ndarray,   # (3,) unit axis
    half_h: float,          # half-length of cylindrical section
    radius: float,          # radius
) -> jnp.ndarray:           # scalar
    """
    Signed distance from p to capsule surface. Negative = inside.

    Clamps the nearest point on the axis segment, then subtracts radius.
    """
    A  = center - half_h * axis
    AB = 2.0 * half_h * axis
    t  = jnp.clip(jnp.dot(p - A, AB) / jnp.dot(AB, AB), 0.0, 1.0)
    closest = A + t * AB
    return _safe_norm(p - closest) - radius


def point_plane_dist(
    p:      jnp.ndarray,   # (3,) query point
    pt:     jnp.ndarray,   # (3,) any point on the plane
    normal: jnp.ndarray,   # (3,) unit outward normal
) -> jnp.ndarray:           # scalar
    """
    Signed distance from p to plane. Positive = on the normal side (outside).
    """
    return jnp.dot(p - pt, normal)
