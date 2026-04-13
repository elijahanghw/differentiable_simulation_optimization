"""
Run a single deterministic rollout from a saved checkpoint and visualise it
in Rerun.

Usage
-----
  python rollout.py --config configs/hover_2d.yaml --checkpoint checkpoints/hover_2d.pkl
  python rollout.py --config configs/hover_3d.yaml --checkpoint checkpoints/hover_3d.pkl
  python rollout.py --config configs/hover_2d.yaml --checkpoint checkpoints/hover_2d.pkl --steps 500
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import rerun as rr
import yaml

from envs import make_env
from envs.multicopter.hover import Hover3d
from envs.multicopter.quat_math import quat_to_rotmat
from envs.multicopter.morphology import morphology, PROP_DIAMETER
from models import make_model
from utils.checkpoint import load as load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps",      type=int,   default=200)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--output",     type=str,   default="rollout.rrd")
    p.add_argument("--prefix",     type=str,   default="drone")
    return p.parse_args()


def run_rollout(env, policy, policy_params, morph_params, key, steps):
    if hasattr(env, "reset_to"):
        obs, state, _ = env.reset_to()
    else:
        obs, state, _ = env.reset(key)
    states = [np.array(state)]

    for _ in range(steps):
        action = policy.apply(policy_params, obs)
        state, obs, _, _, _, _ = env.step(state, action, morph_params)
        states.append(np.array(state))

    return np.stack(states)


def visualise_2d(states, l, dt, prefix="drone"):

    xs     = states[:, 0]
    zs     = states[:, 1]
    thetas = states[:, 4]

    for t, (x, z, theta) in enumerate(zip(xs, zs, thetas)):
        rr.set_time("time", duration=t * dt)

        body_radius = 0.05
        center  = np.array([x, z])
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        arm_dir = np.array([cos_t, sin_t])
        left        = center - l * arm_dir
        right       = center + l * arm_dir
        left_inner  = center - body_radius * arm_dir
        right_inner = center + body_radius * arm_dir

        rr.log(f"{prefix}/arm", rr.LineStrips2D(
            [np.stack([left, left_inner]), np.stack([right_inner, right])],
            colors=[[80, 80, 220]], radii=0.01,
        ))
        rr.log(f"{prefix}/body", rr.Points2D(
            [center], colors=[[220, 80, 80]], radii=0.05,
        ))
        thrust_tip = center + np.array([sin_t * 0.15, -cos_t * 0.15])
        rr.log(f"{prefix}/thrust", rr.LineStrips2D(
            [np.stack([center, thrust_tip])], colors=[[80, 220, 80]], radii=0.008,
        ))

    rr.set_time("time", duration=len(states) * dt)
    rr.log(f"{prefix}/trajectory", rr.LineStrips2D(
        [np.stack([xs, zs], axis=1)], colors=[[180, 180, 180]], radii=0.005,
    ))

    for t in range(len(states)):
        rr.set_time("time", duration=t * dt)
        rr.log(f"{prefix}/state/x",     rr.Scalars(float(states[t, 0])))
        rr.log(f"{prefix}/state/z",     rr.Scalars(float(states[t, 1])))
        rr.log(f"{prefix}/state/vx",    rr.Scalars(float(states[t, 2])))
        rr.log(f"{prefix}/state/vz",    rr.Scalars(float(states[t, 3])))
        rr.log(f"{prefix}/state/theta", rr.Scalars(float(states[t, 4])))
        rr.log(f"{prefix}/state/omega", rr.Scalars(float(states[t, 5])))


def _flip_z(v):
    """Flip z axis: sim is z-down (+z = down), Rerun is z-up."""
    v = np.asarray(v)
    return v * np.array([1.0, 1.0, -1.0])


DRONE_COLORS = {
    "optimal": {"arm": [80,  180,  80], "body": [60,  200,  60], "prop": [40,  220,  40], "thrust": [120, 255, 120]},
    "fixed":   {"arm": [80,   80, 220], "body": [220,  80,  80], "prop": [80,  140, 220], "thrust": [ 80, 220, 220]},
    "default": {"arm": [80,   80, 220], "body": [220,  80,  80], "prop": [80,  180,  80], "thrust": [ 80, 220,  80]},
}

def _rodrigues_np(v, axis, angle):
    """Rodrigues rotation (numpy), vectorized over leading dim."""
    c   = np.cos(angle)[..., None]
    s   = np.sin(angle)[..., None]
    dot = np.sum(axis * v, axis=-1, keepdims=True)
    return v * c + np.cross(axis, v) * s + axis * dot * (1.0 - c)


def _disc_points(normal, radius, n_pts=32):
    """
    Return (n_pts, 3) points on a circle of given radius centred at the origin,
    lying in the plane whose normal is `normal`.
    """
    n   = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u   = np.cross(n, ref);  u /= np.linalg.norm(u)
    v   = np.cross(n, u)
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    return radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)  # (n_pts, 3)


def visualise_3d(states, l, dt, prefix="drone", phi=None, alpha=None):
    """
    states: (T, 19)  [pos(3), vel(3), quat(4), omega(3), W(6)]
    l:     scalar or (3,) — arm lengths for positive-y arms; mirrored to full 6.
    phi:   scalar or (3,) — arm inclination angles (rad); mirrored to full 6.
    alpha: scalar or (3,) — propeller tilt angles (rad); mirrored with negation.
    Colors are chosen by prefix: "optimal" (green), "fixed" (blue/red), or "default".
    """
    c = DRONE_COLORS.get(prefix, DRONE_COLORS["default"])

    l = np.asarray(l).flatten()
    l_full = np.array([l[0], l[1], l[2], l[2], l[1], l[0]]) if l.size == 3 else np.broadcast_to(l, (6,))

    if phi is None:
        phi = np.zeros(3)
    phi = np.asarray(phi).flatten()
    phi_full = np.array([phi[0], phi[1], phi[2], phi[2], phi[1], phi[0]]) if phi.size == 3 else np.broadcast_to(phi, (6,))

    if alpha is None:
        alpha = np.zeros(3)
    alpha = np.asarray(alpha).flatten()
    alpha_full = np.array([alpha[0], alpha[1], alpha[2], -alpha[2], -alpha[1], -alpha[0]])

    azimuths = np.array([np.pi/6, np.pi*3/6, np.pi*5/6, np.pi*7/6, np.pi*9/6, np.pi*11/6])
    cp = np.cos(phi_full)
    sp = np.sin(phi_full)
    prop_positions_body = l_full[:, None] * np.stack(
        [cp * np.cos(azimuths), cp * np.sin(azimuths), -sp], axis=1
    )  # (6, 3)

    # Thrust orientations per motor (Rodrigues around arm unit vector by alpha)
    arm_norms = np.linalg.norm(prop_positions_body, axis=1, keepdims=True)
    arm_unit  = prop_positions_body / np.maximum(arm_norms, 1e-8)           # (6, 3)
    thrust_base = np.tile(np.array([0.0, 0.0, -1.0]), (6, 1))              # (6, 3)
    thrust_body = _rodrigues_np(thrust_base, arm_unit, alpha_full)          # (6, 3)

    # Flip z: NED body → display body
    F = np.diag([1.0, 1.0, -1.0])
    prop_positions_disp = (F @ prop_positions_body.T).T   # (6, 3)
    thrust_disp_body    = (F @ thrust_body.T).T            # (6, 3)

    # Precompute disc point offsets in display body frame for each motor.
    # Normal of each disc = thrust direction; disc lies in the plane of the propeller.
    disc_offsets = np.stack([
        _disc_points(thrust_disp_body[i], radius=PROP_DIAMETER / 2)
        for i in range(6)
    ])  # (6, n_pts, 3)

    for t in range(len(states)):
        rr.set_time("time", duration=t * dt)

        pos  = states[t, 0:3]
        quat = states[t, 6:10]
        R    = np.array(quat_to_rotmat(quat))

        R_disp = F @ R @ F

        pos_d        = _flip_z(pos)
        prop_world_d = pos_d + (R_disp @ prop_positions_disp.T).T  # (6, 3)

        # Arms
        arm_strips = [np.stack([pos_d, prop_world_d[i]]) for i in range(6)]
        rr.log(f"{prefix}/arms", rr.LineStrips3D(
            arm_strips, colors=[c["arm"]], radii=0.004,
        ))

        rr.log(f"{prefix}/body", rr.Points3D(
            [pos_d], colors=[c["body"]], radii=0.04,
        ))
        rr.log(f"{prefix}/props", rr.Points3D(
            prop_world_d, colors=[c["prop"]], radii=0.01,
        ))

        # Per-motor propeller discs: rotate offsets to world display frame, close the loop
        disc_strips = [
            np.concatenate([
                prop_world_d[i] + (R_disp @ disc_offsets[i].T).T,
                prop_world_d[i] + disc_offsets[i, :1] @ R_disp.T,  # close loop
            ], axis=0)
            for i in range(6)
        ]
        rr.log(f"{prefix}/discs", rr.LineStrips3D(
            disc_strips, colors=[c["thrust"]], radii=0.003,
        ))

    # Trajectory
    rr.set_time("time", duration=len(states) * dt)
    traj = np.array([_flip_z(states[t, 0:3]) for t in range(len(states))])
    rr.log(f"{prefix}/trajectory", rr.LineStrips3D(
        [traj], colors=[c["arm"]], radii=0.003,
    ))

    # Time-series: pos, vel, euler, omega
    from envs.multicopter.quat_math import quat_to_euler as _quat_to_euler
    for t in range(len(states)):
        rr.set_time("time", duration=t * dt)
        rr.log(f"{prefix}/state/x",     rr.Scalars(float(states[t, 0])))
        rr.log(f"{prefix}/state/y",     rr.Scalars(float(states[t, 1])))
        rr.log(f"{prefix}/state/z",     rr.Scalars(float(states[t, 2])))
        rr.log(f"{prefix}/state/vx",    rr.Scalars(float(states[t, 3])))
        rr.log(f"{prefix}/state/vy",    rr.Scalars(float(states[t, 4])))
        rr.log(f"{prefix}/state/vz",    rr.Scalars(float(states[t, 5])))
        roll, pitch, yaw = _quat_to_euler(states[t, 6:10])
        rr.log(f"{prefix}/state/roll",  rr.Scalars(float(roll)))
        rr.log(f"{prefix}/state/pitch", rr.Scalars(float(pitch)))
        rr.log(f"{prefix}/state/yaw",   rr.Scalars(float(yaw)))
        rr.log(f"{prefix}/state/wx",    rr.Scalars(float(states[t, 10])))
        rr.log(f"{prefix}/state/wy",    rr.Scalars(float(states[t, 11])))
        rr.log(f"{prefix}/state/wz",    rr.Scalars(float(states[t, 12])))


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ecfg = config["env"]
    env  = make_env(ecfg["name"], **{k: v for k, v in ecfg.items() if k != "name"})

    pcfg        = config["policy"]
    layer_sizes = [env.obs_dim] + pcfg["hidden_sizes"] + [env.act_dim]
    policy      = make_model(pcfg["type"], layer_sizes=layer_sizes)

    policy_params, morph_params = load_checkpoint(args.checkpoint)

    if morph_params is not None and hasattr(env, "get_morph_info"):
        print("Morphology:")
        for k, v in env.get_morph_info(morph_params).items():
            print(f"  {k} = {v:.4f}")

    key    = jax.random.PRNGKey(args.seed)
    states = run_rollout(env, policy, policy_params, morph_params, key, args.steps)

    if morph_params is not None and hasattr(env, "get_l"):
        l     = np.array(env.get_l(morph_params))
        phi   = np.array(env.get_phi(morph_params))   if hasattr(env, "get_phi")   else None
        alpha = np.array(env.get_alpha(morph_params)) if hasattr(env, "get_alpha") else None
    elif morph_params is not None and hasattr(env, "get_r"):
        l     = float(env.get_r(morph_params))
        phi   = None
        alpha = None
    else:
        l     = env.l_default if hasattr(env, "l_default") else env.l
        phi   = np.full(3, env.phi_default)   if hasattr(env, "phi_default")   else None
        alpha = np.full(3, env.alpha_default) if hasattr(env, "alpha_default") else None

    if isinstance(env, Hover3d):
        rr.init("hover_3d_rollout")
        visualise_3d(states, l=l, dt=env.dt, prefix=args.prefix, phi=phi, alpha=alpha)
    else:
        rr.init("hover_2d_rollout")
        visualise_2d(states, l=l, dt=env.dt, prefix=args.prefix)

    rr.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
