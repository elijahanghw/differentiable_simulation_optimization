import jax
import jax.numpy as jnp
from .quat_math import *


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
# sqrt with a bounded derivative at the origin
# ---------------------------------------------------------------------------
# The throttle curve sqrt(k·u² + (1-k)·u) has slope → ∞ as u → 0, so a single
# saturated action (U = -1 ⇒ u = 0) poisons the whole BPTT backward pass with
# NaN. Clamping the sqrt *argument* would bias the forward motor speed, so
# instead the value stays exact and only the derivative is capped, at
# 1/(2·√_SQRT_EPS). This is purely a gradient knob — the simulated trajectory is
# identical for any value.
#
# 1e-2 caps the slope at 5, which only binds below u ≈ 4.3% throttle; hover trim
# sits at u ≈ 30%, so the operating range is untouched. Peak dW_c/du then lands
# at 1.4x its hover-trim value instead of the 10.7x that _SQRT_EPS = 1e-4 gave,
# which is what keeps the k_rd·d_W yaw feedthrough (a direct action → angular
# acceleration path, with no counterpart in the morphology model) from
# overflowing float32 across a long BPTT window.
_SQRT_EPS = 1e-2

@jax.custom_jvp
def _safe_sqrt(x):
    return jnp.sqrt(jnp.maximum(x, 0.0))

@_safe_sqrt.defjvp
def _safe_sqrt_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return _safe_sqrt(x), dx / (2.0 * jnp.sqrt(jnp.maximum(x, _SQRT_EPS)))


G = 9.81

def dynamics(world_states:jnp.array, U: jnp.array, params: jnp.array):
    (k_wx, k_x, k_wy, k_y, k_wz, k_z,
    k_p, k_pd, k_q, k_qd, k_r, k_rd,
    tau, k, w_min, w_max, W_MIN_N, W_MAX_N) = params

    pos = world_states[0:3]
    vel = world_states[3:6]
    quat = world_states[6:10]
    omega = world_states[10:13]
    w = world_states[13:] # normalized to [-1, 1]

    U = jnp.clip((U + 1.0)/2.0, 0.0, 1.0) # transform commands to [0, 1]
    W = (w + 1.0)/2.0 * W_MAX_N # transform motor angular rate to [0, W_MAX_N]

    W_c = (w_max - w_min) * _safe_sqrt(k * U ** 2 + (1 - k) * U) + w_min # Normalized commands to motor speed commands
    
    d_W = (W_c - W) / tau  # derivative of W_c, rad/s
    d_w = 2.0 * d_W / W_MAX_N       # derivative of w ∈ [-1,1], since W=(w+1)/2 → dw=2dW

    W2 = W * W    # [0,W_MAX_N^2]


    v_body = quat_rotate_point(vel, quat_conj(quat))
    vbx, vby, vbz = v_body

    Fx = k_wx * jnp.sum(W2) + k_x * vbx * jnp.sum(W)
    Fy = k_wy * jnp.sum(W2) + k_y * vby * jnp.sum(W)
    Fz = -k_wz * jnp.sum(W2) + k_z * vbz * jnp.sum(W)

    F = jnp.array([Fx, Fy, Fz])

    Mx = jnp.sum(k_p * W2) + jnp.sum(k_pd * d_W)
    My = jnp.sum(k_q * W2) + jnp.sum(k_qd * d_W)
    Mz = jnp.sum(k_r * W2) + jnp.sum(k_rd * d_W)

    M = jnp.array([Mx, My, Mz])
    
    d_pos = vel
    d_vel = quat_rotate_point(F, quat) + jnp.array([0, 0, G])

    omega_quat = jnp.concatenate([jnp.array([0.0]), omega])
    d_quat = 0.5 * quat_mul(quat, omega_quat)

    d_omega = M

    return jnp.concatenate([d_pos, d_vel, d_quat, d_omega, d_w])


# def motor_zoh(w: jnp.ndarray, U: jnp.ndarray, tau: float, dt: float) -> jnp.ndarray:
#     """Exact solution of dw/dt = (U-w)/TIME_CONSTANT for constant U over [0, dt].

#     Replaces the Euler-integrated w from the integrators when dt > TIME_CONSTANT.
#     Unconditionally stable for any dt/TIME_CONSTANT ratio.
#       ∂w_new/∂w = exp(-dt/τ)       < 1  always
#       ∂w_new/∂U = 1-exp(-dt/τ)    < 1  always
#     """
#     alpha = jnp.exp(jnp.asarray(-dt / tau))
#     return alpha * w + (1.0 - alpha) * U


def forward_euler(state, U, params, dt):
    # tau = params[12]
    next_state = state + dt * dynamics(state, U, params)
    # return next_state.at[13:19].set(motor_zoh(state[13:19], U, tau, dt))
    return next_state


def semi_implicit_euler(state, U, params, dt):
    # tau = params[12]
    d = dynamics(state, U, params)
    new_vel   = state[3:6]   + dt * d[3:6]
    new_omega = state[10:13] + dt * d[10:13]
    new_pos   = state[0:3]   + dt * new_vel
    omega_quat = jnp.concatenate([jnp.array([0.0]), new_omega])
    new_quat = state[6:10] + dt * 0.5 * quat_mul(state[6:10], omega_quat)
    new_quat = new_quat / jnp.linalg.norm(new_quat)
    # new_W    = motor_zoh(state[13:19], U, tau, dt)
    new_W = state[13:] + dt * d[13:]
    return jnp.concatenate([new_pos, new_vel, new_quat, new_omega, new_W])


def rk4(state, U, params, dt):
    # tau = params[12]
    k1 = dynamics(state,              U, params)
    k2 = dynamics(state + 0.5*dt*k1, U, params)
    k3 = dynamics(state + 0.5*dt*k2, U, params)
    k4 = dynamics(state +     dt*k3, U, params)
    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    # return next_state.at[13:19].set(motor_zoh(state[13:19], U, tau, dt))
    return next_state