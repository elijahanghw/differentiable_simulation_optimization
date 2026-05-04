"""
Point-mass dynamics following the DiffPhysDrone v1.0 CUDA physics kernel,
translated to JAX.

Physics model (gravity is assumed compensated by a low-level attitude
controller — the policy commands net world-frame accelerations):

    a_cmd[t] = act_pred*(1 - exp(-delay*dt)) + a[t-1]*exp(-delay*dt)
    drag      = Σ_axis (drag2[0]*v_axis*|v_axis| + drag2[1]*v_axis) * axis_vec
                (computed in body frame via R, then projected back to world)
    a[t]      = a_cmd[t] + dg[t] - drag
    p[t+1]    = G(p[t]) + v[t]*dt + 0.5*a[t]*dt²
    v[t+1]    = G(v[t]) + 0.5*(a[t] + a[t+1])*dt
    dg[t+1]   = dg[t]*√(1-dt) + ε*0.2*√(dt)    ε ~ N(0, I)

where G(x) is the identity in the forward pass but multiplies the gradient by
grad_decay^dt in the backward pass (gradient truncation for training stability).

Attitude update (R has columns [forward, left, up] in world frame):
    up  = normalise(act_cmd)
    fwd = blend(normalise(v_pred), R[:,0]; α = exp(-yaw_delay*dt))
    fwd = orthogonalise(fwd, up) → normalise
    R_new = [fwd | cross(up,fwd) | up]
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Gradient-decay helper (identity forward, scaled backward)
# ---------------------------------------------------------------------------

@jax.custom_vjp
def _gdecay(x, factor):
    return x


def _gdecay_fwd(x, factor):
    return x, (jnp.asarray(factor),)


def _gdecay_bwd(res, g):
    (factor,) = res
    return g * factor, jnp.zeros_like(factor)


_gdecay.defvjp(_gdecay_fwd, _gdecay_bwd)


# ---------------------------------------------------------------------------
# Physics step
# ---------------------------------------------------------------------------

def dynamics_step(p, v, a, dg, R, act_pred, key,
                  pitch_ctl_delay, drag_2, z_drag_coef,
                  grad_decay_factor, dt):
    """
    One physics step.

    Args:
        p, v, a, dg : (3,) position, velocity, filtered action, OU disturbance
        R           : (3,3) rotation matrix; columns = [forward, left, up]
        act_pred    : (3,) commanded acceleration (world frame)
        key         : PRNGKey for OU noise sample
        pitch_ctl_delay : scalar — control-delay stiffness (~12)
        drag_2      : (2,) — [quadratic, linear] drag coefficients
        z_drag_coef : scalar — extra scale on the body-up drag axis (default 1.0)
        grad_decay_factor : scalar — grad_decay^dt, applied in backward pass
        dt          : scalar — timestep (s)

    Returns:
        p_next, v_next, a_cmd, dg_next  — all (3,)
    """
    # Action low-pass filter (first-order IIR)
    alpha = jnp.exp(-pitch_ctl_delay * dt)
    a_cmd = act_pred * (1.0 - alpha) + a * alpha

    # Body-frame velocity projections for anisotropic drag
    v_fwd  = jnp.dot(v, R[:, 0])
    v_left = jnp.dot(v, R[:, 1])
    v_up   = jnp.dot(v, R[:, 2])

    drag = (
        drag_2[0] * (  v_fwd  * jnp.abs(v_fwd)  * R[:, 0]
                      + v_left * jnp.abs(v_left) * R[:, 1]
                      + v_up   * jnp.abs(v_up)   * R[:, 2] * z_drag_coef)
      + drag_2[1] * (  v_fwd  * R[:, 0]
                      + v_left * R[:, 1]
                      + v_up   * R[:, 2] * z_drag_coef)
    )

    a_next = a_cmd + dg - drag

    # Trapezoidal integration; G() scales gradients by grad_decay_factor
    p_next = _gdecay(p, grad_decay_factor) + v * dt + 0.5 * a * dt ** 2
    v_next = _gdecay(v, grad_decay_factor) + 0.5 * (a + a_next) * dt

    # Ornstein–Uhlenbeck disturbance update
    noise   = jax.random.normal(key, shape=(3,))
    dg_next = dg * jnp.sqrt(1.0 - dt) + noise * 0.2 * jnp.sqrt(dt)

    return p_next, v_next, a_cmd, dg_next


# ---------------------------------------------------------------------------
# Attitude update
# ---------------------------------------------------------------------------

def update_attitude(R, act, v_pred, yaw_ctl_delay, dt):
    """
    Update body-frame rotation matrix from thrust direction + velocity blend.

    Args:
        R             : (3,3) current rotation matrix; columns [fwd, left, up]
        act           : (3,) current acceleration command (thrust proxy)
        v_pred        : (3,) predicted velocity in world frame
        yaw_ctl_delay : scalar — yaw dynamics stiffness (~6)
        dt            : scalar — timestep (s)

    Returns:
        R_new : (3,3) updated rotation matrix
    """
    yaw_alpha = jnp.exp(-yaw_ctl_delay * dt)

    # Up = thrust direction (normalised)
    up = act / jnp.sqrt(jnp.dot(act, act) + 1e-8)

    # Forward: blend v_pred direction with current forward, then orthogonalise
    v_unit = v_pred / jnp.sqrt(jnp.dot(v_pred, v_pred) + 1e-8)
    fwd    = v_unit * (1.0 - yaw_alpha) + R[:, 0] * yaw_alpha
    fwd    = fwd - jnp.dot(fwd, up) * up
    fwd    = fwd / jnp.sqrt(jnp.dot(fwd, fwd) + 1e-8)

    left = jnp.cross(up, fwd)

    return jax.lax.stop_gradient(jnp.stack([fwd, left, up], axis=1))
