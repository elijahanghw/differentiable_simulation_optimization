import jax.numpy as jnp

def quat_mul(q, r):
    """Quaternion multiplication: q * r"""
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return jnp.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_conj(q):
    """Quaternion conjugate"""
    w, x, y, z = q
    return jnp.array([w, -x, -y, -z])

def quat_rotate_point(v, q):
    """Rotate vector v by quaternion q"""
    q = q / jnp.linalg.norm(q)  # ensure unit quaternion
    q_v = jnp.concatenate([jnp.array([0.0]), v])
    return quat_mul(quat_mul(q, q_v), quat_conj(q))[1:]

def euler_to_quat(roll, pitch, yaw):
    """Convert Euler angles to quaternion."""
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return jnp.array([w, x, y, z])

def quat_to_euler(q):
    """Convert quaternion to Euler angles."""
    w, x, y, z = q

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(t0, t1)

    # Pitch (y-axis rotation) — arctan2 for finite gradients everywhere near ±90°
    t2    = +2.0 * (w * y - z * x)
    t2c   = jnp.sqrt(jnp.maximum(1.0 - t2 * t2, 1e-12))
    pitch = jnp.arctan2(t2, t2c)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(t3, t4)

    return jnp.array([roll, pitch, yaw])

def quat_to_rotmat(q):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    R = jnp.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])
    return R