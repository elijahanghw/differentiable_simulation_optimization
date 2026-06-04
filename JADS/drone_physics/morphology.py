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
HEIGHT = 0.1

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


def morphology(l, phi=None, alpha=None):
    """
    l:     scalar or (3,) array [l1, l2, l3] — arm lengths for positive-y arms
           (30°, 90°, 150°). Negative-y arms mirror: [l1,l2,l3,l3,l2,l1].
    phi:   scalar or (3,) array — inclination angles (rad) for positive-y arms.
           +phi raises the arm tip in -z (upward NED). Mirrored symmetrically.
           Defaults to 0 (flat).
    alpha: scalar or (3,) array — propeller roll tilt (rad) about each arm's
           outward unit vector, for positive-y arms. Mirrored symmetrically.
           +alpha tilts the thrust vector sideways. Defaults to 0.
    """
    l = jnp.atleast_1d(jnp.asarray(l, dtype=jnp.float32))
    if l.shape == (1,):
        l = jnp.broadcast_to(l, (3,))

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
    l_full     = jnp.array([l[0],     l[1],     l[2],     l[2],     l[1],     l[0]])    # (6,)
    phi_full   = jnp.array([phi[0],   phi[1],   phi[2],   phi[2],   phi[1],   phi[0]])  # (6,)
    alpha_full = jnp.array([alpha[0], alpha[1], alpha[2], -alpha[2], -alpha[1], -alpha[0]])# (6,)

    azimuths = jnp.array([jnp.pi/6, jnp.pi*3/6, jnp.pi*5/6,
                           jnp.pi*7/6, jnp.pi*9/6, jnp.pi*11/6])             # (6,)

    # r_i = l_i * [cos(phi)*cos(az), cos(phi)*sin(az), -sin(phi)]
    cp = jnp.cos(phi_full)
    sp = jnp.sin(phi_full)
    propeller_positions = l_full[:, None] * jnp.stack(
        [cp * jnp.cos(azimuths), cp * jnp.sin(azimuths), -sp], axis=1
    )  # (6, 3)

    # Arm unit vectors (outward direction from body center)
    arm_norms = jnp.linalg.norm(propeller_positions, axis=1, keepdims=True)  # (6, 1)
    arm_unit  = propeller_positions / jnp.maximum(arm_norms, 1e-8)           # (6, 3)

    # Base thrust direction: -z in body frame
    thrust_base = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (6, 1))             # (6, 3)

    # Rotate thrust_base around arm_unit by alpha (Rodrigues)
    propeller_orientations = _rodrigues(thrust_base, arm_unit, alpha_full)   # (6, 3)

    propeller_rotations = jnp.array([1, -1, 1, -1, 1, -1])

    arm_lengths = jnp.linalg.norm(propeller_positions, axis=1)       # (6,)
    arm_masses  = ARM_DENSITY * arm_lengths                           # (6,)
    m = BODY_MASS + 6 * MOTOR_MASS + jnp.sum(arm_masses)

    # I = body + Σ_i (motor_i + arm_i), where for each prop:
    #   motor: point mass  → MOTOR_MASS * (|r|² I - r⊗r)
    #   arm:   rod from CG → (arm_mass/3) * (|r|² I - r⊗r)
    scale = MOTOR_MASS + arm_masses / 3                               # (6,)
    r2    = jnp.sum(propeller_positions ** 2, axis=1)                 # (6,)
    outer = propeller_positions[:, :, None] * propeller_positions[:, None, :]  # (6, 3, 3)

    J = BODY_INERTIA + jnp.sum(
        scale[:, None, None] * (r2[:, None, None] * jnp.eye(3) - outer),
        axis=0,
    )

    J_inv = jnp.linalg.inv(J)

    Bf = (KT * propeller_orientations).T  # (3, 6)

    Bf = Bf * MAX_RPM * MAX_RPM

    Bm = (jnp.cross(propeller_positions, KT * propeller_orientations)
          - KM * propeller_rotations[:, None] * propeller_orientations).T  # (3, 6)
    
    Bm = Bm * MAX_RPM * MAX_RPM

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


def propeller_collision_loss_from_params(l, phi, alpha, weight=100.0):
    """
    Compute propeller collision loss directly from morphology parameters.

    Args:
        l:      (3,) arm lengths for positive-y arms
        phi:    (3,) inclination angles (rad)
        alpha:  (3,) propeller tilt angles (rad)
        weight: float, loss coefficient
    """
    l_full     = jnp.array([l[0],     l[1],     l[2],     l[2],     l[1],     l[0]])
    phi_full   = jnp.array([phi[0],   phi[1],   phi[2],   phi[2],   phi[1],   phi[0]])
    alpha_full = jnp.array([alpha[0], alpha[1], alpha[2], -alpha[2], -alpha[1], -alpha[0]])

    azimuths = jnp.array([jnp.pi/6, jnp.pi*3/6, jnp.pi*5/6,
                           jnp.pi*7/6, jnp.pi*9/6, jnp.pi*11/6])

    cp = jnp.cos(phi_full)
    sp = jnp.sin(phi_full)
    propeller_positions = l_full[:, None] * jnp.stack(
        [cp * jnp.cos(azimuths), cp * jnp.sin(azimuths), -sp], axis=1
    )  # (6, 3)

    arm_norms = jnp.linalg.norm(propeller_positions, axis=1, keepdims=True)
    arm_unit  = propeller_positions / jnp.maximum(arm_norms, 1e-8)

    thrust_base = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (6, 1))
    propeller_orientations = _rodrigues(thrust_base, arm_unit, alpha_full)  # (6, 3)

    return propeller_collision_loss(propeller_positions, propeller_orientations, weight=weight)