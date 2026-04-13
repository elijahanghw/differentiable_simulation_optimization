import jax.numpy as jnp
from .quat_math import *

G = 9.81
LINEAR_DAMPING  = 0.1   # N·s/m
ANGULAR_DAMPING = 0.05  # N·m·s/rad

TIME_CONSTANT = 0.04

def dynamics(world_states:jnp.array, U: jnp.array, Bf: jnp.array, Bm: jnp.array, m: float, J: jnp.array, J_inv: jnp.array):
    pos = world_states[0:3]
    vel = world_states[3:6]
    quat = world_states[6:10]
    omega = world_states[10:13]
    w = world_states[13:19] # normalized to [-1, 1]

    W_c = (U + 1.0)/2.0 # transform commands to [0, 1]
    W = (w + 1.0)/2.0 # transform rpm ro [0, 1]
    
    d_W = (W_c - W) / TIME_CONSTANT  # derivative of W ∈ [0,1]
    d_w = 2.0 * d_W                  # derivative of w ∈ [-1,1], since W=(w+1)/2 → dw=2dW

    W2 = W * W    # [0,1]²; MAX_RPM² already in Bf/Bm

    F = Bf @ W2
    M = Bm @ W2
    
    d_pos = vel
    d_vel = quat_rotate_point(F/m, quat) + jnp.array([0, 0, G]) - (LINEAR_DAMPING / m) * vel

    omega_quat = jnp.concatenate([jnp.array([0.0]), omega])
    d_quat = 0.5 * quat_mul(quat, omega_quat)

    d_omega = J_inv @ (M - jnp.cross(omega, J @ omega)) - ANGULAR_DAMPING * J_inv @ omega

    return jnp.concatenate([d_pos, d_vel, d_quat, d_omega, d_w])


def forward_euler(state, U, Bf, Bm, m, J, J_inv, dt):
    return state + dt * dynamics(state, U, Bf, Bm, m, J, J_inv)


def semi_implicit_euler(state, U, Bf, Bm, m, J, J_inv, dt):
    d = dynamics(state, U, Bf, Bm, m, J, J_inv)
    new_vel   = state[3:6]   + dt * d[3:6]
    new_omega = state[10:13] + dt * d[10:13]
    new_W     = state[13:19] + dt * d[13:19]
    new_pos   = state[0:3]   + dt * new_vel
    omega_quat = jnp.concatenate([jnp.array([0.0]), new_omega])
    new_quat = state[6:10] + dt * 0.5 * quat_mul(state[6:10], omega_quat)
    new_quat = new_quat / jnp.linalg.norm(new_quat)
    return jnp.concatenate([new_pos, new_vel, new_quat, new_omega, new_W])


def rk4(state, U, Bf, Bm, m, J, J_inv, dt):
    k1 = dynamics(state,              U, Bf, Bm, m, J, J_inv)
    k2 = dynamics(state + 0.5*dt*k1, U, Bf, Bm, m, J, J_inv)
    k3 = dynamics(state + 0.5*dt*k2, U, Bf, Bm, m, J, J_inv)
    k4 = dynamics(state +     dt*k3, U, Bf, Bm, m, J, J_inv)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)