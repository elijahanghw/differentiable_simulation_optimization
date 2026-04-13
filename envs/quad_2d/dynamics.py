import jax.numpy as jnp

G = 9.81
LINEAR_DAMPING  = 0.1   # N·s/m
ANGULAR_DAMPING = 0.05  # N·m·s/rad


def dynamics(world_states:jnp.array, U: jnp.array, Bf: jnp.array, Bm: jnp.array, m: float, J: float):

    vel = world_states[2:4]
    theta = world_states[4]
    omega = world_states[5]

    F = Bf @ U
    M = Bm @ U

    R = jnp.array([[jnp.cos(theta), jnp.sin(theta)],
                   [-jnp.sin(theta), jnp.cos(theta)]])
    
    d_pos = vel
    d_vel = R@(F/m) + jnp.array([0, G]) - (LINEAR_DAMPING / m) * vel

    d_theta = omega
    d_omega = M[0] / J - (ANGULAR_DAMPING / J) * omega

    return jnp.array([d_pos[0], d_pos[1], d_vel[0], d_vel[1], d_theta, d_omega])


def euler(state, U, Bf, Bm, m, J, dt):
    return state + dt * dynamics(state, U, Bf, Bm, m, J)


def semi_implicit_euler(state, U, Bf, Bm, m, J, dt):
    d = dynamics(state, U, Bf, Bm, m, J)
    # Update velocities first
    new_vel   = state[2:4] + dt * d[2:4]
    new_omega = state[5]   + dt * d[5]
    # Update positions using new velocities
    new_pos   = state[0:2] + dt * new_vel
    new_theta = state[4]   + dt * new_omega
    return jnp.array([new_pos[0], new_pos[1], new_vel[0], new_vel[1], new_theta, new_omega])


def rk4(state, U, Bf, Bm, m, J, dt):
    k1 = dynamics(state,              U, Bf, Bm, m, J)
    k2 = dynamics(state + 0.5*dt*k1, U, Bf, Bm, m, J)
    k3 = dynamics(state + 0.5*dt*k2, U, Bf, Bm, m, J)
    k4 = dynamics(state +     dt*k3, U, Bf, Bm, m, J)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)