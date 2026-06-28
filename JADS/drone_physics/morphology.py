import jax
import jax.numpy as jnp

# 2 inch drone
# BODY_MASS = 0.0136 + 0.043
# MOTOR_MASS = 0.0046
# ARM_DENSITY = 0.034 # kg/m
# MAX_RPM = 5000 # rad/s

# KT = 8.12e-8
# KM = 6.40e-10

# LENGTH = 0.1
# WIDTH = 0.1
# HEIGHT = 0.1

# 3 inch drone
BODY_MASS = 0.350
MOTOR_MASS = 0.015
ARM_DENSITY = 0.034 # kg/m
MAX_RPM = 3200 # rad/s

KT = 4.00e-07
KM = 4.00e-09
PROP_DIAMETER = 0.0762 # 3 inch propeller = 0.0762 m + 0.02 m tolerance
PROP_BUFFER = 0.02

LENGTH = 0.1
WIDTH = 0.1
HEIGHT = 0.05

# 5 inch drone 
# BODY_MASS = 0.300
# MOTOR_MASS = 0.0335
# ARM_DENSITY = 1500*0.005*0.01 # kg/m
# MAX_RPM = 3000 # rad/s

# KT = 1.08e-06
# KM = 1.22e-08

# LENGTH = 0.1
# WIDTH = 0.1
# HEIGHT = 0.1

BODY_INERTIA = jnp.diag(jnp.array([
    BODY_MASS * (WIDTH**2 + HEIGHT**2) / 12,
    BODY_MASS * (LENGTH**2 + HEIGHT**2) / 12,
    BODY_MASS * (LENGTH**2 + WIDTH**2) / 12,
]))

def _rodrigues(v, axis, angle):
    """Rotate vector v around unit axis by angle (Rodrigues' formula). Vectorized over leading dims."""
    c = jnp.cos(angle)[..., None]
    s = jnp.sin(angle)[..., None]
    dot = jnp.sum(axis * v, axis=-1, keepdims=True)
    return v * c + jnp.cross(axis, v) * s + axis * dot * (1.0 - c)


MOUNT_RADIUS = 0.05  # distance from body center to arm mounting point (m)


def morphology(l, psi=None, theta=None, phi=None, alpha=None, mount_radius=MOUNT_RADIUS):
    """
    mount_radius: scalar — distance from body center to arm mounting point (m).
    l:     scalar or (3,) array [l1, l2, l3] — arm lengths from the mounting point
           for positive-y arms (30°, 90°, 150°). Negative-y arms mirror: [l1,l2,l3,l3,l2,l1].
    psi:   scalar or (3,) array — Yaw angle from pre determined azimuth
    theta: scalar or (3,) array — vertical tilt angle (rad) pivoting on the mounting
           point for positive-y arms. +theta raises the arm tip in -z (upward NED).
           Mirrored symmetrically. Defaults to 0 (flat).
    phi:   scalar or (3,) array — propeller roll tilt (rad) about each arm's direction
           vector (mount→tip), for positive-y arms. Mirrored symmetrically.
           +phi tilts the thrust vector sideways. Defaults to 0.
    """
    l = jnp.atleast_1d(jnp.asarray(l, dtype=jnp.float32))
    if l.shape == (1,):
        l = jnp.broadcast_to(l, (3,))

    if psi is None:
        psi = jnp.zeros(3)
    psi = jnp.atleast_1d(jnp.asarray(psi, dtype=jnp.float32))
    if psi.shape == (1,):
        psi = jnp.broadcast_to(psi, (3,))

    if theta is None:
        theta = jnp.zeros(3)
    theta = jnp.atleast_1d(jnp.asarray(theta, dtype=jnp.float32))
    if theta.shape == (1,):
        theta = jnp.broadcast_to(theta, (3,))

    if phi is None:
        phi = jnp.zeros(3)
    phi = jnp.atleast_1d(jnp.asarray(phi, dtype=jnp.float32))
    if phi.shape == (1,):
        phi = jnp.broadcast_to(phi, (3,))

    if alpha is None:
        alpha = jnp.zeros(3)
    alpha = jnp.atleast_1d(jnp.asarray(alpha, dtype=jnp.float32))
    if alpha.shape == (1,):
        alpha = jnp.broadcast_to(alpha, (3,))

    # Mirror to full 6 arms: positive-y then negative-y
    l_full     = jnp.array([l[0],     l[1],     l[2],     l[2],     l[1],     l[0]])     # (6,)
    psi_full   = jnp.array([psi[0],     psi[1],     psi[2],     -psi[2],     -psi[1],     -psi[0]])     # (6,)
    theta_full = jnp.array([theta[0], theta[1], theta[2], theta[2], theta[1], theta[0]]) # (6,)
    phi_full   = jnp.array([phi[0],   phi[1],   phi[2],   -phi[2],  -phi[1],  -phi[0]])  # (6,)
    alpha_full = jnp.array([alpha[0], alpha[1], alpha[2], alpha[2], alpha[1], alpha[0]]) # (6,)

    azimuths = jnp.array([jnp.pi/6, jnp.pi*3/6, jnp.pi*5/6,
                           jnp.pi*7/6, jnp.pi*9/6, jnp.pi*11/6])              # (6,)
    
    arm_yaw = azimuths + psi_full

    # Mounting points: mount_radius from center in radial direction
    mount_points = mount_radius * jnp.stack(
        [jnp.cos(azimuths), jnp.sin(azimuths), jnp.zeros_like(azimuths)], axis=1
    )  # (6, 3)

    # Arm unit vectors: direction from mounting point to motor tip (already unit length)
    cp = jnp.cos(theta_full)
    sp = jnp.sin(theta_full)
    arm_unit = jnp.stack(
        [cp * jnp.cos(arm_yaw), cp * jnp.sin(arm_yaw), -sp], axis=1
    )  # (6, 3)

    # Motor positions: mounting point + l along arm direction
    propeller_positions = mount_points + l_full[:, None] * arm_unit            # (6, 3)

    # Thrust direction: start from [0,0,-1],
    thrust_base = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (6, 1))              # (6, 3)

    # 1. Pitch with the arm by theta around the tangential axis at the mount point
    # 2. Further pitch thrust by alpha (theta and alpha positive opposite direction) 
    # Final pitching angle = theta - alpha
    # Tangential axis = [-sin(az), cos(az), 0] — perpendicular to radial, in xy-plane
    tangential = jnp.stack(
        [-jnp.sin(arm_yaw), jnp.cos(arm_yaw), jnp.zeros_like(arm_yaw)], axis=1
    )  # (6, 3)
    thrust_pitched = _rodrigues(thrust_base, tangential, (theta_full - alpha_full))          # (6, 3)

    # 3. Roll by phi around the arm axis.
    propeller_orientations = _rodrigues(thrust_pitched, arm_unit, phi_full)   # (6, 3)

    propeller_rotations = jnp.array([1, -1, 1, -1, 1, -1])

    arm_masses = ARM_DENSITY * l_full                                          # (6,)
    m = BODY_MASS + 6 * MOTOR_MASS + jnp.sum(arm_masses)

    # Motor inertia: point mass at propeller_positions
    r2_motor  = jnp.sum(propeller_positions ** 2, axis=1)                     # (6,)
    out_motor = propeller_positions[:, :, None] * propeller_positions[:, None, :]  # (6, 3, 3)
    I_motor   = MOTOR_MASS * jnp.sum(
        r2_motor[:, None, None] * jnp.eye(3) - out_motor, axis=0
    )  # (3, 3)

    # Arm inertia: rod from mount_point to motor, using parallel axis theorem
    arm_cm       = mount_points + (l_full[:, None] / 2) * arm_unit            # (6, 3)
    out_arm_unit = arm_unit[:, :, None] * arm_unit[:, None, :]                # (6, 3, 3)
    # Rod inertia about its own CM: (m*l^2/12) * (I - u⊗u)
    I_rod_cm = jnp.sum(
        (arm_masses * l_full ** 2 / 12)[:, None, None] * (jnp.eye(3) - out_arm_unit), axis=0
    )  # (3, 3)
    # Parallel axis shift to body CG
    r2_arm_cm  = jnp.sum(arm_cm ** 2, axis=1)                                # (6,)
    out_arm_cm = arm_cm[:, :, None] * arm_cm[:, None, :]                     # (6, 3, 3)
    I_arm_pa   = jnp.sum(
        arm_masses[:, None, None] * (r2_arm_cm[:, None, None] * jnp.eye(3) - out_arm_cm), axis=0
    )  # (3, 3)

    J     = BODY_INERTIA + I_motor + I_rod_cm + I_arm_pa
    J_inv = jnp.linalg.inv(J)

    Bf = (KT * propeller_orientations).T * MAX_RPM * MAX_RPM  # (3, 6)

    Bm = (jnp.cross(propeller_positions, KT * propeller_orientations)
          - KM * propeller_rotations[:, None] * propeller_orientations).T * MAX_RPM * MAX_RPM  # (3, 6)

    return Bf, Bm, m, J, J_inv, propeller_positions

def _point_to_seg_dist(p, q0, q1, eps=1e-8):
    """Distance from point p to line segment q0→q1."""
    d = q1 - q0
    t = jnp.dot(p - q0, d) / (jnp.dot(d, d) + eps)
    closest = q0 + jnp.clip(t, 0.0, 1.0) * d
    diff = p - closest
    return jnp.sqrt(jnp.dot(diff, diff) + eps)


def propeller_collision_loss(propeller_positions, propeller_orientations, weight=100.0):
    """
    Differentiable capsule-sphere collision loss for all 6 propellers.

    For each pair (i, j) with i < j:
      - Propeller i is the sphere:   center at propeller_positions[i], radius PROP_DIAMETER/2
      - Propeller j is the capsule:  axis from propeller_positions[j] to tip[j], radius PROP_DIAMETER/2

    Args:
        propeller_positions:    (6, 3) motor positions in body frame
        propeller_orientations: (6, 3) unit thrust vectors per motor
        weight:                 float, loss coefficient

    Returns:
        scalar loss
    """
    r = (PROP_DIAMETER + PROP_BUFFER) / 2
    h = PROP_DIAMETER + PROP_BUFFER

    ornt = propeller_orientations / jnp.maximum(
        jnp.linalg.norm(propeller_orientations, axis=1, keepdims=True), 1e-8
    )
    tips = propeller_positions - h * ornt  # (6, 3)

    pairs = [(i, j) for i in range(6) for j in range(6) if i != j]
    loss = jnp.zeros(())
    for i, j in pairs:
        dist = _point_to_seg_dist(propeller_positions[i], propeller_positions[j], tips[j])
        penetration = jnp.maximum(0.0, 2.0 * r - dist)
        loss = loss + penetration ** 2

    return weight * loss


def propeller_collision_loss_from_params(l, psi=None, theta=None, phi=None, alpha=None, mount_radius=MOUNT_RADIUS, weight=100.0):
    """
    Compute propeller collision loss directly from morphology parameters.

    Args:
        l:            (3,) arm lengths from mounting point for positive-y arms
        psi:          (3,) arm yaw offset from azimuth (rad), defaults to 0
        theta:        (3,) vertical tilt angles (rad)
        phi:          (3,) propeller roll tilt angles (rad)
        alpha:        (3,) motor tilt angles (rad), defaults to 0
        mount_radius: scalar, distance from body center to arm mounting point (m)
        weight:       float, loss coefficient
    """
    Bf, Bm, m, J, J_inv, propeller_positions = morphology(l, psi, theta, phi, alpha, mount_radius)
    propeller_orientations = (Bf / (KT * MAX_RPM * MAX_RPM)).T               # (6, 3)
    return propeller_collision_loss(propeller_positions, propeller_orientations, weight=weight)