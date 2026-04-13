"""
Differentiable Quadcopter Racing environment.

State convention: tuple of JAX arrays — a valid JAX pytree.
    (world_state, target_gate, num_gates_passed, step_count, prev_action, params)

world_state[16]: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w1, w2, w3, w4]

Observation is expressed in the target gate's reference frame.
Action: 4 normalised motor commands in [-1, 1].

API (gym-like)
--------------
    obs, state, info = env.reset(key)
    next_state, obs, reward, terminated, truncated, info = env.step(state, action)
"""

import jax
import jax.numpy as jnp

# Physical constants
G = 9.81

# Motor speed range (rad/s)
W_MIN_N = 0.0
W_MAX_N = 5000.0

# Gate configuration for race track
GATE_RADIUS = 1.5
GATE_POS = jnp.array([
    [ GATE_RADIUS,  -GATE_RADIUS, -1.5],
    [ 0,             0,           -1.5],
    [-GATE_RADIUS,   GATE_RADIUS, -1.5],
    [ 0,           2*GATE_RADIUS, -1.5],
    [ GATE_RADIUS,   GATE_RADIUS, -1.5],
    [ 0,             0,           -1.5],
    [-GATE_RADIUS,  -GATE_RADIUS, -1.5],
    [ 0,          -2*GATE_RADIUS, -1.5]
], dtype=jnp.float32)

GATE_YAW = jnp.array([1, 2, 1, 0, -1, -2, -1, 0], dtype=jnp.float32) * jnp.pi / 2
START_POS = GATE_POS[0] + jnp.array([0, -1.0, 0], dtype=jnp.float32)

# Propeller axes (all pointing down)
AXES = jnp.array([
    [0, 0, -1],
    [0, 0, -1],
    [0, 0, -1],
    [0, 0, -1]
], dtype=jnp.float32)

# Default quadcopter parameters (can be randomized)
DEFAULT_PARAMS = jnp.array([
    5.07e-07,    # k_wz
    0.0,         # k_z
    0.0,         # k_wx
    -6.51e-05,   # k_x
    0.0,         # k_wy
    -6.32e-05,   # k_y
    # k_p1, k_p2, k_p3, k_p4 (roll moments)
    -1.62e-05, -1.63e-05, 1.60e-05, 1.58e-05,
    # k_pd1, k_pd2, k_pd3, k_pd4 (roll moment derivatives)
    0.0, 0.0, 0.0, 0.0,
    # k_q1, k_q2, k_q3, k_q4 (pitch moments)
    -8.10e-06, 1.07e-05, -9.93e-06, 9.14e-06,
    # k_qd1, k_qd2, k_qd3, k_qd4 (pitch moment derivatives)
    0.0, 0.0, 0.0, 0.0,
    # k_r1, k_r2, k_r3, k_r4 (yaw moments)
    -3.08e-06, 2.01e-06, 3.19e-06, -2.32e-06,
    # k_rd1, k_rd2, k_rd3, k_rd4 (yaw moment derivatives)
    -1.37e-03, 1.37e-03, 1.37e-03, -1.37e-03,
    0.04,     # tau (motor time constant)
    0.8,      # k (motor response curve parameter)
    497.53,   # w_min (minimum motor speed)
    5200.73,  # w_max (maximum motor speed)
], dtype=jnp.float32)


# ============================================================================
# DYNAMICS FUNCTIONS
# ============================================================================

def _rotation_matrix(phi, theta, psi):
    """Compute rotation matrix from body to world frame."""
    Rx = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(phi), -jnp.sin(phi)],
        [0, jnp.sin(phi), jnp.cos(phi)]
    ])
    Ry = jnp.array([
        [jnp.cos(theta), 0, jnp.sin(theta)],
        [0, 1, 0],
        [-jnp.sin(theta), 0, jnp.cos(theta)]
    ])
    Rz = jnp.array([
        [jnp.cos(psi), -jnp.sin(psi), 0],
        [jnp.sin(psi), jnp.cos(psi), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def _quadcopter_dynamics(world_state, action, params):
    x, y, z = world_state[0], world_state[1], world_state[2]
    vx, vy, vz = world_state[3], world_state[4], world_state[5]
    phi, theta, psi = world_state[6], world_state[7], world_state[8]
    p, q, r = world_state[9], world_state[10], world_state[11]
    w1, w2, w3, w4 = world_state[12], world_state[13], world_state[14], world_state[15]

    u1, u2, u3, u4 = action[0], action[1], action[2], action[3]

    k_wz, k_z, k_wx, k_x, k_wy, k_y = params[0:6]
    k_p1, k_p2, k_p3, k_p4 = params[6:10]
    k_pd1, k_pd2, k_pd3, k_pd4 = params[10:14]
    k_q1, k_q2, k_q3, k_q4 = params[14:18]
    k_qd1, k_qd2, k_qd3, k_qd4 = params[18:22]
    k_r1, k_r2, k_r3, k_r4 = params[22:26]
    k_rd1, k_rd2, k_rd3, k_rd4 = params[26:30]
    tau, k, w_min, w_max = params[30], params[31], params[32], params[33]

    # Convert normalised motor speeds to rad/s
    W1 = (w1 + 1) / 2 * (W_MAX_N - W_MIN_N) + W_MIN_N
    W2 = (w2 + 1) / 2 * (W_MAX_N - W_MIN_N) + W_MIN_N
    W3 = (w3 + 1) / 2 * (W_MAX_N - W_MIN_N) + W_MIN_N
    W4 = (w4 + 1) / 2 * (W_MAX_N - W_MIN_N) + W_MIN_N

    # Convert motor commands to [0, 1]
    U1 = (u1 + 1) / 2
    U2 = (u2 + 1) / 2
    U3 = (u3 + 1) / 2
    U4 = (u4 + 1) / 2

    # Steady-state motor response
    Wc1 = (w_max - w_min) * jnp.sqrt(k * U1**2 + (1 - k) * U1) + w_min
    Wc2 = (w_max - w_min) * jnp.sqrt(k * U2**2 + (1 - k) * U2) + w_min
    Wc3 = (w_max - w_min) * jnp.sqrt(k * U3**2 + (1 - k) * U3) + w_min
    Wc4 = (w_max - w_min) * jnp.sqrt(k * U4**2 + (1 - k) * U4) + w_min

    # Motor dynamics (first-order)
    d_W1 = (Wc1 - W1) / tau
    d_W2 = (Wc2 - W2) / tau
    d_W3 = (Wc3 - W3) / tau
    d_W4 = (Wc4 - W4) / tau

    d_w1 = d_W1 / (W_MAX_N - W_MIN_N) * 2
    d_w2 = d_W2 / (W_MAX_N - W_MIN_N) * 2
    d_w3 = d_W3 / (W_MAX_N - W_MIN_N) * 2
    d_w4 = d_W4 / (W_MAX_N - W_MIN_N) * 2

    R = _rotation_matrix(phi, theta, psi)

    vel_world = jnp.array([vx, vy, vz])
    vel_body = R.T @ vel_world
    vbx, vby, vbz = vel_body[0], vel_body[1], vel_body[2]
    vb = vel_body

    # vp_i: velocity component perpendicular to propeller axis i (shape (3,))
    # vp_i = vb - (vb · axis_i) * axis_i
    # NOTE: keep both terms as (3,) to avoid (3,)-(3,1) broadcasting to (3,3)
    vp0 = vb - jnp.dot(AXES[0, :], vb) * AXES[0, :]
    vp1 = vb - jnp.dot(AXES[1, :], vb) * AXES[1, :]
    vp2 = vb - jnp.dot(AXES[2, :], vb) * AXES[2, :]
    vp3 = vb - jnp.dot(AXES[3, :], vb) * AXES[3, :]

    vd0 = AXES[0, 0] * vbx + AXES[0, 1] * vby + AXES[0, 2] * vbz
    vd1 = AXES[1, 0] * vbx + AXES[1, 1] * vby + AXES[1, 2] * vbz
    vd2 = AXES[2, 0] * vbx + AXES[2, 1] * vby + AXES[2, 2] * vbz
    vd3 = AXES[3, 0] * vbx + AXES[3, 1] * vby + AXES[3, 2] * vbz

    inflow_coeff = 2 * jnp.pi / 0.127

    fx = (AXES[0, 0] * W1**2 + AXES[1, 0] * W2**2 + AXES[2, 0] * W3**2 + AXES[3, 0] * W4**2
          - inflow_coeff * (vd0 * AXES[0, 0] * W1 + vd1 * AXES[1, 0] * W2 +
                            vd2 * AXES[2, 0] * W3 + vd3 * AXES[3, 0] * W4))
    fy = (AXES[0, 1] * W1**2 + AXES[1, 1] * W2**2 + AXES[2, 1] * W3**2 + AXES[3, 1] * W4**2
          - inflow_coeff * (vd0 * AXES[0, 1] * W1 + vd1 * AXES[1, 1] * W2 +
                            vd2 * AXES[2, 1] * W3 + vd3 * AXES[3, 1] * W4))
    fz = (AXES[0, 2] * W1**2 + AXES[1, 2] * W2**2 + AXES[2, 2] * W3**2 + AXES[3, 2] * W4**2
          - inflow_coeff * (vd0 * AXES[0, 2] * W1 + vd1 * AXES[1, 2] * W2 +
                            vd2 * AXES[2, 2] * W3 + vd3 * AXES[3, 2] * W4))

    dx = vp0[0] * W1 + vp1[0] * W2 + vp2[0] * W3 + vp3[0] * W4
    dy = vp0[1] * W1 + vp1[1] * W2 + vp2[1] * W3 + vp3[1] * W4
    dz = vp0[2] * W1 + vp1[2] * W2 + vp2[2] * W3 + vp3[2] * W4

    Fx = k_wx * fx + k_x * dx
    Fy = k_wy * fy + k_y * dy
    Fz = k_wz * fz + k_z * dz

    Mx = (k_p1 * W1**2 + k_p2 * W2**2 + k_p3 * W3**2 + k_p4 * W4**2 +
          k_pd1 * d_W1 + k_pd2 * d_W2 + k_pd3 * d_W3 + k_pd4 * d_W4)
    My = (k_q1 * W1**2 + k_q2 * W2**2 + k_q3 * W3**2 + k_q4 * W4**2 +
          k_qd1 * d_W1 + k_qd2 * d_W2 + k_qd3 * d_W3 + k_qd4 * d_W4)
    Mz = (k_r1 * W1**2 + k_r2 * W2**2 + k_r3 * W3**2 + k_r4 * W4**2 +
          k_rd1 * d_W1 + k_rd2 * d_W2 + k_rd3 * d_W3 + k_rd4 * d_W4)

    d_x = vx
    d_y = vy
    d_z = vz

    forces_body = jnp.array([Fx, Fy, Fz])
    forces_world = R @ forces_body
    d_vx = forces_world[0]
    d_vy = forces_world[1]
    d_vz = G + forces_world[2]

    d_phi = p + q * jnp.sin(phi) * jnp.tan(theta) + r * jnp.cos(phi) * jnp.tan(theta)
    d_theta = q * jnp.cos(phi) - r * jnp.sin(phi)
    # cos_theta = jnp.cos(theta)
    # cos_theta_safe = jnp.where(jnp.abs(cos_theta) < 0.01, 0.01 * jnp.sign(cos_theta), cos_theta)
    # d_psi = q * jnp.sin(phi) / cos_theta_safe + r * jnp.cos(phi) / cos_theta_safe
    d_psi = q * jnp.sin(phi) / jnp.cos(theta) + r * jnp.cos(phi) / jnp.cos(theta) 

    d_p = Mx
    d_q = My
    d_r = Mz

    return jnp.array([
        d_x, d_y, d_z, d_vx, d_vy, d_vz,
        d_phi, d_theta, d_psi, d_p, d_q, d_r,
        d_w1, d_w2, d_w3, d_w4
    ])


# ============================================================================
# ENVIRONMENT CLASS
# ============================================================================

class QuadcopterRacing:
    """
    Differentiable quadcopter racing environment.

    State: tuple (world_state, target_gate, num_gates_passed, step_count,
                  prev_action, params)
    """

    def __init__(
        self,
        dt: float = 0.01,
        max_steps: int = 1200,
        gates_ahead: int = 1,
        motor_limit: float = 1.0,
        initialize_at_random_gate: bool = True,
        randomization: float = 0.3,
        vel_observations: bool = True,
        gate_pos: jnp.ndarray = GATE_POS,
        gate_yaw: jnp.ndarray = GATE_YAW,
        start_pos: jnp.ndarray = START_POS,
        **kwargs,
    ):
        self.dt = dt
        self.max_steps = max_steps  # 1200 matches SB3; must exceed training horizon
        self.gates_ahead = gates_ahead
        self.motor_limit = motor_limit
        self.initialize_at_random_gate = initialize_at_random_gate
        self.randomization = randomization
        self.vel_observations = vel_observations

        self.gate_pos = gate_pos
        self.gate_yaw = gate_yaw
        self.start_pos = start_pos
        self.num_gates = len(gate_pos)

        self.gate_pos_rel, self.gate_yaw_rel = self._compute_relative_gates()

        if self.vel_observations:
            # pos[3] + vel[3] + att[3] + rates[3] + rpms[4] + future_gates[4*gates_ahead]
            self.obs_dim = 16 + 4 * gates_ahead
        else:
            # pos[3] + att[3] + rates[3] + rpms[4] + future_gates[4*gates_ahead]
            self.obs_dim = 13 + 4 * gates_ahead

        self.act_dim = 4

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, key: jax.Array) -> tuple:
        """
        Sample a random initial state.

        Returns:
            obs:   observation vector, shape (obs_dim,)
            state: (world_state, target_gate, num_gates_passed, step_count,
                    prev_action, params)
            info:  empty dict
        """
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        params = self._randomize_params(key4)

        # Determine starting gate
        target_gate = jax.lax.cond(
            self.initialize_at_random_gate,
            lambda: jax.random.randint(key1, (), 0, self.num_gates),
            lambda: jnp.int32(0),
        )
        gate_p = self.gate_pos[target_gate]
        gate_y = self.gate_yaw[target_gate]
        start_x = jnp.where(self.initialize_at_random_gate,
                             gate_p[0] - jnp.cos(gate_y), self.start_pos[0])
        start_y = jnp.where(self.initialize_at_random_gate,
                             gate_p[1] - jnp.sin(gate_y), self.start_pos[1])
        start_z = jnp.where(self.initialize_at_random_gate,
                             gate_p[2], self.start_pos[2])

        # pos = jax.random.uniform(
        #     key2, (3,),
        #     minval=jnp.array([start_x - 0.3, start_y - 0.3, start_z - 0.3]),
        #     maxval=jnp.array([start_x + 0.3, start_y + 0.3, start_z + 0.3]),
        # )

        pos   = jnp.array([start_x, start_y, start_z])
        vel   = jax.random.uniform(key2, (3,), minval=-0.5, maxval=0.5)
        att   = jax.random.uniform(
            key3, (3,),
            minval=jnp.array([-jnp.pi / 9, -jnp.pi / 9, -jnp.pi]),
            maxval=jnp.array([ jnp.pi / 9,  jnp.pi / 9,  jnp.pi]),
        )
        rates = jax.random.uniform(key5, (3,), minval=-0.1, maxval=0.1)
        rpms  = jax.random.uniform(key6, (4,), minval=-1.0, maxval=1.0)

        world_state = jnp.concatenate([pos, vel, att, rates, rpms])

        state = (
            world_state,
            target_gate,
            jnp.int32(0),
            jnp.int32(0),
            jnp.zeros(4, dtype=jnp.float32),
            params,
        )
        return self._get_obs(state), state, {}

    def step(self, state: tuple, action: jnp.ndarray) -> tuple:
        """
        Advance the state by one timestep.

        Args:
            state:  (world_state, target_gate, num_gates_passed, step_count,
                     prev_action, params)
            action: normalised motor commands in [-1, 1], shape (4,)

        Returns:
            next_state:  updated state tuple
            obs:         observation of next_state, shape (obs_dim,)
            reward:      scalar reward (differentiable w.r.t. action)
            terminated:  True if the drone crashed
            truncated:   False — time limit is managed by the training loop
            info:        dict with 'gate_passed' boolean
        """
        world_state, target_gate, num_gates_passed, step_count, _, params = state

        action = jnp.clip(action, -1.0, 2.0 * self.motor_limit - 1.0)

        state_dot = _quadcopter_dynamics(world_state, action, params)
        new_world_state = world_state + self.dt * state_dot

        pos_old = world_state[0:3]
        pos_new = new_world_state[0:3]

        gate_pos = self.gate_pos[target_gate % self.num_gates]
        gate_yaw = self.gate_yaw[target_gate % self.num_gates]

        # Gate passing detection
        normal = jnp.array([jnp.cos(gate_yaw), jnp.sin(gate_yaw)])
        pos_old_proj = (pos_old[0] - gate_pos[0]) * normal[0] + (pos_old[1] - gate_pos[1]) * normal[1]
        pos_new_proj = (pos_new[0] - gate_pos[0]) * normal[0] + (pos_new[1] - gate_pos[1]) * normal[1]

        passed_gate_plane = (pos_old_proj < 0) & (pos_new_proj > 0)
        gate_size_small = 1.5
        gate_passed = passed_gate_plane & (jnp.abs(pos_new - gate_pos) < gate_size_small / 2).all()

        # Termination: drone crashed or clipped gate edge
        ground_collision = new_world_state[2] > 0
        out_of_bounds = (
            (jnp.abs(new_world_state[0:2]) > 5).any()
            | (new_world_state[2] < -7)
            | (jnp.abs(new_world_state[9:12]) > 1000).any()
        )
        gate_collision = passed_gate_plane & ~gate_passed
        terminated = ground_collision | out_of_bounds | gate_collision
        truncated = (step_count + 1) >= self.max_steps

        # Freeze state on crash so the scan doesn't integrate NaN/Inf forward.
        new_world_state = jnp.where(terminated, world_state, new_world_state)

        # Reward
        d2g_old         = jnp.linalg.norm(pos_old - gate_pos)
        d2g_new         = jnp.linalg.norm(pos_new - gate_pos)
        progress_reward = (d2g_old - d2g_new) * 1.0
        rate_penalty    = 0.001 * jnp.linalg.norm(new_world_state[9:12])
        crash_penalty   = ground_collision | out_of_bounds  # gate_collision ends episode but no -10
        reward          = jnp.where(crash_penalty, -10.0, progress_reward - rate_penalty)

        new_target_gate      = jnp.where(gate_passed, (target_gate + 1) % self.num_gates, target_gate)
        new_num_gates_passed = jnp.where(gate_passed, num_gates_passed + 1, num_gates_passed)

        next_state = (
            new_world_state,
            new_target_gate,
            new_num_gates_passed,
            step_count + 1,
            action,
            params,
        )
        return (
            next_state,
            self._get_obs(next_state),
            reward,
            terminated,
            truncated,
            {"gate_passed": gate_passed},
        )

    def _get_obs(self, state: tuple) -> jnp.ndarray:
        """
        Return the observation vector expressed in the target gate's frame.

        Shape: (obs_dim,)
        """
        world_state, target_gate, *_ = state

        gate_pos = self.gate_pos[target_gate % self.num_gates]
        gate_yaw = self.gate_yaw[target_gate % self.num_gates]

        pos_world = world_state[0:3]
        vel_world = world_state[3:6]

        cos_yaw = jnp.cos(gate_yaw)
        sin_yaw = jnp.sin(gate_yaw)
        R_2d = jnp.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])

        pos_gate_xy = R_2d @ (pos_world[0:2] - gate_pos[0:2])
        pos_gate = jnp.array([pos_gate_xy[0], pos_gate_xy[1], pos_world[2] - gate_pos[2]])

        vel_gate_xy = R_2d @ vel_world[0:2]
        vel_gate = jnp.array([vel_gate_xy[0], vel_gate_xy[1], vel_world[2]])

        att = world_state[6:9]
        yaw_rel = jnp.remainder(att[2] - gate_yaw + jnp.pi, 2 * jnp.pi) - jnp.pi
        att_gate = jnp.array([att[0], att[1], yaw_rel])

        rates = world_state[9:12]
        rpms  = world_state[12:16]

        if self.vel_observations:
            obs = jnp.concatenate([pos_gate, vel_gate, att_gate, rates, rpms])
        else:
            obs = jnp.concatenate([pos_gate, att_gate, rates, rpms])

        # Append future gate info
        future_gates = []
        for i in range(self.gates_ahead):
            next_gate_idx = (target_gate + i + 1) % self.num_gates
            rel_pos = self.gate_pos_rel[next_gate_idx]
            rel_yaw = self.gate_yaw_rel[next_gate_idx]
            future_gates.append(jnp.concatenate([rel_pos, rel_yaw[None]]))

        if future_gates:
            obs = jnp.concatenate([obs] + future_gates)

        return obs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_relative_gates(self):
        rel_pos_list = []
        rel_yaw_list = []
        for i in range(self.num_gates):
            prev_i = (i - 1) % self.num_gates
            rel_pos = self.gate_pos[i] - self.gate_pos[prev_i]
            yaw_prev = self.gate_yaw[prev_i]
            cos_yaw = jnp.cos(yaw_prev)
            sin_yaw = jnp.sin(yaw_prev)
            R = jnp.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
            rel_pos_xy = R @ rel_pos[0:2]
            rel_pos = jnp.array([rel_pos_xy[0], rel_pos_xy[1], rel_pos[2]])
            rel_yaw = jnp.remainder(self.gate_yaw[i] - yaw_prev + jnp.pi, 2 * jnp.pi) - jnp.pi
            rel_pos_list.append(rel_pos)
            rel_yaw_list.append(rel_yaw)
        return jnp.array(rel_pos_list), jnp.array(rel_yaw_list)

    def _randomize_params(self, key: jax.Array) -> jnp.ndarray:
        rand_max = 1 + self.randomization
        rand_min = 1 - self.randomization
        multipliers = jax.random.uniform(key, shape=DEFAULT_PARAMS.shape,
                                         minval=rand_min, maxval=rand_max)
        k_index = 31
        k_value = DEFAULT_PARAMS[k_index]
        k_upper = jnp.minimum(k_value * rand_max, 1.0)
        k_mult = jax.random.uniform(key, shape=(), minval=rand_min, maxval=k_upper / k_value)
        multipliers = multipliers.at[k_index].set(k_mult)
        return DEFAULT_PARAMS * multipliers
