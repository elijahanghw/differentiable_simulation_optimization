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
BODY_MASS = 0.150
MOTOR_MASS = 0.010
ARM_DENSITY = 0.034 # kg/m
MAX_RPM = 3200 # rad/s

KT = 4.00e-07
KM = 4.00e-09
PROP_DIAMETER = 0.0762  # 3 inch propeller → 3 * 0.0254 = 0.0762 m

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

    return Bf, Bm, m, J, J_inv

def _seg_to_seg_dist(p0, p1, q0, q1, eps=1e-8):
    """Minimum distance between two line segments P(s) and Q(t), s,t ∈ [0,1]."""
    d1 = p1 - p0
    d2 = q1 - q0
    r  = p0 - q0
    a  = jnp.dot(d1, d1)
    e  = jnp.dot(d2, d2)
    b  = jnp.dot(d1, d2)
    c  = jnp.dot(d1, r)
    f  = jnp.dot(d2, r)
    D  = a * e - b * b
    s  = jnp.clip((b * f - c * e) / (D + eps), 0.0, 1.0)
    t  = jnp.clip((b * s + f)     / (e + eps), 0.0, 1.0)
    s  = jnp.clip((b * t - c)     / (a + eps), 0.0, 1.0)
    diff = (p0 + s * d1) - (q0 + t * d2)
    return jnp.sqrt(jnp.dot(diff, diff) + eps)


def propeller_collision_loss(propeller_positions, propeller_orientations, weight=100.0):
    """
    Differentiable capsule-capsule collision loss for the 3 positive-y propellers.

    Each propeller is modeled as a capsule:
      - Radius:  PROP_DIAMETER / 2
      - Height:  PROP_DIAMETER  (one diameter long)
      - Axis:    downwash direction (-thrust), base at motor position

    Only checks the 3 positive-y arms (indices 0,1,2) — 3 pairs total.
    Negative-y arms are mirror-symmetric so if positive-y arms don't intersect,
    neither will their mirrors.

    Args:
        propeller_positions:    (6, 3) motor positions in body frame
        propeller_orientations: (6, 3) unit thrust vectors per motor
        weight:                 float, loss coefficient

    Returns:
        scalar loss
    """
    r = PROP_DIAMETER / 2   # capsule radius
    h = PROP_DIAMETER       # capsule height (cylinder height = diameter)

    pos  = propeller_positions[:3]    # (3, 3) positive-y motors
    ornt = propeller_orientations[:3] # (3, 3) thrust unit vectors

    ornt = ornt / jnp.maximum(jnp.linalg.norm(ornt, axis=1, keepdims=True), 1e-8)
    tips = pos - h * ornt  # (3, 3)  downwash end of each capsule

    pairs = [(0, 1), (0, 2), (1, 2)]
    loss = jnp.zeros(())
    for i, j in pairs:
        dist = _seg_to_seg_dist(pos[i], tips[i], pos[j], tips[j])
        penetration = jnp.maximum(0.0, 2.0 * r - dist)  # sum of radii = 2r
        loss = loss + penetration ** 2

    return weight * loss


def propeller_collision_loss_from_params(l, phi, alpha, weight=100.0):
    """
    Compute propeller collision loss directly from morphology parameters.
    Replicates the geometry computation from morphology() for the positive-y arms.

    Args:
        l:      (3,) arm lengths for positive-y arms
        phi:    (3,) inclination angles (rad)
        alpha:  (3,) propeller tilt angles (rad)
        weight: float, loss coefficient
    """
    azimuths_pos = jnp.array([jnp.pi/6, jnp.pi*3/6, jnp.pi*5/6])  # positive-y only

    cp = jnp.cos(phi)
    sp = jnp.sin(phi)
    propeller_positions = l[:, None] * jnp.stack(
        [cp * jnp.cos(azimuths_pos), cp * jnp.sin(azimuths_pos), -sp], axis=1
    )  # (3, 3)

    arm_norms = jnp.linalg.norm(propeller_positions, axis=1, keepdims=True)
    arm_unit  = propeller_positions / jnp.maximum(arm_norms, 1e-8)

    thrust_base = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (3, 1))
    propeller_orientations = _rodrigues(thrust_base, arm_unit, alpha)  # (3, 3)

    # Pad to (6, 3) shape expected by propeller_collision_loss (only [:3] is used)
    pos_pad  = jnp.concatenate([propeller_positions, jnp.zeros((3, 3))], axis=0)
    ornt_pad = jnp.concatenate([propeller_orientations, jnp.zeros((3, 3))], axis=0)

    return propeller_collision_loss(pos_pad, ornt_pad, weight=weight)