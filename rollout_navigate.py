"""
Rollout for multicopter Navigate — renders one episode in Rerun (FRD, z-down).

Usage:
  python rollout_navigate.py \
      --config configs/navigate.yaml \
      --checkpoint checkpoints/navigate.pkl \
      [--steps 200] [--seed 0] [--output rollout_navigate.rrd]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "depth_render"))

import numpy as np
import jax
import jax.numpy as jnp
import rerun as rr
import yaml

from envs import make_env
from envs.multicopter.quat_math import quat_to_rotmat, quat_to_euler
from envs.multicopter.morphology import PROP_DIAMETER
from models import make_model
from utils.checkpoint import load as load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps",      type=int, default=200)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--output",     type=str, default="rollout_navigate.rrd")
    return p.parse_args()


def _rotmat_to_quat_np(R):
    """(3,3) numpy rotation matrix → quaternion [x, y, z, w]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-8)


def _quat_z_to_axis(axes):
    """Return (N, 4) quaternions [x, y, z, w] that rotate +Z onto each axis."""
    src = np.array([0.0, 0.0, 1.0])
    quats = []
    for ax in axes:
        ax = ax / (np.linalg.norm(ax) + 1e-8)
        d = float(np.dot(src, ax))
        if d > 1.0 - 1e-6:
            quats.append([0.0, 0.0, 0.0, 1.0])
        elif d < -1.0 + 1e-6:
            quats.append([1.0, 0.0, 0.0, 0.0])
        else:
            rot_axis = np.cross(src, ax)
            rot_axis /= np.linalg.norm(rot_axis)
            half = np.arccos(np.clip(d, -1.0, 1.0)) / 2.0
            s = np.sin(half)
            quats.append([rot_axis[0] * s, rot_axis[1] * s, rot_axis[2] * s, np.cos(half)])
    return np.array(quats, dtype=np.float32)


def log_scene(scene_cfg, scene_array):
    """Log obstacles and ground plane as static geometry (FRD coords, z=0 is ground)."""
    arrays = scene_cfg.unpack(scene_array)

    cx = (scene_cfg.arena_x_min + scene_cfg.arena_x_max) / 2
    cy = (scene_cfg.arena_y_min + scene_cfg.arena_y_max) / 2
    hx = (scene_cfg.arena_x_max - scene_cfg.arena_x_min) / 2 + 2.0
    hy = (scene_cfg.arena_y_max - scene_cfg.arena_y_min) / 2 + 2.0
    rr.log("world/ground", rr.Boxes3D(
        centers=[[cx, cy, 0.02]],
        half_sizes=[[hx, hy, 0.02]],
        colors=[[130, 130, 130, 160]],
        fill_mode="solid",
    ), static=True)

    sc = np.array(arrays["sphere_centers"])
    sr = np.array(arrays["sphere_radii"])
    if sc.shape[0] > 0:
        rr.log("world/spheres", rr.Ellipsoids3D(
            centers=sc,
            half_sizes=np.stack([sr, sr, sr], axis=1),
            colors=[[220, 100, 60, 200]],
            fill_mode="solid",
        ), static=True)

    bc  = np.array(arrays["box_centers"])
    bhe = np.array(arrays["box_half_extents"])
    if bc.shape[0] > 0:
        rr.log("world/boxes", rr.Boxes3D(
            centers=bc,
            half_sizes=bhe,
            colors=[[60, 100, 220, 200]],
            fill_mode="solid",
        ), static=True)

    cc  = np.array(arrays["capsule_centers"])
    ca  = np.array(arrays["capsule_axes"])
    chh = np.array(arrays["capsule_hh"])
    cr  = np.array(arrays["capsule_radii"])
    if cc.shape[0] > 0:
        base_translations = cc - ca * chh[:, None]
        lengths = (2.0 * chh).astype(np.float32)
        quats   = _quat_z_to_axis(ca)
        rr.log("world/capsules", rr.Capsules3D(
            lengths=lengths,
            radii=cr.astype(np.float32),
            translations=base_translations,
            quaternions=quats,
            colors=[[60, 200, 100, 200]],
            fill_mode="solid",
        ), static=True)


def _rodrigues_np(v, axis, angle):
    """Rodrigues rotation (numpy), vectorized over leading dim."""
    c   = np.cos(angle)[..., None]
    s   = np.sin(angle)[..., None]
    dot = np.sum(axis * v, axis=-1, keepdims=True)
    return v * c + np.cross(axis, v) * s + axis * dot * (1.0 - c)


def _disc_points(normal, radius, n_pts=32):
    n   = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u   = np.cross(n, ref);  u /= np.linalg.norm(u)
    v   = np.cross(n, u)
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    return radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)


def _norm_depth(raw):
    d_min, d_max = raw.min(), raw.max()
    return ((raw - d_min) / (d_max - d_min + 1e-8)).astype(np.float32)


def run_rollout(env, policy, policy_params, key, steps):
    reset_key, _ = jax.random.split(key)
    (depth_img, obs_vec), state, _ = env.reset(reset_key)

    hidden = policy.init_hidden()

    all_states = [np.array(state)]
    raw_depths = [np.array(env.get_vis_depth(state))]

    for _ in range(steps):
        action, hidden = policy.apply(
            {"params": policy_params}, depth_img, obs_vec, hidden
        )
        state, _ = env.step(state, action)
        depth_img, obs_vec = env._get_obs(state)

        all_states.append(np.array(state))
        raw_depths.append(np.array(env.get_vis_depth(state)))

    return all_states, raw_depths


def visualise(env, all_states, raw_depths, l, phi, alpha, dt):
    log_scene(env.scene_cfg, all_states[0][22:])

    rr.log("world/target", rr.Points3D(
        [all_states[0][19:22]], colors=[[255, 215, 0]], radii=0.15,
    ), static=True)

    # Precompute body-frame prop geometry (FRD, no z-flip needed)
    l = np.asarray(l).flatten()
    l_full = np.array([l[0], l[1], l[2], l[2], l[1], l[0]]) if l.size == 3 else np.broadcast_to(l, (6,))

    phi = np.asarray(phi).flatten()
    phi_full = np.array([phi[0], phi[1], phi[2], phi[2], phi[1], phi[0]]) if phi.size == 3 else np.broadcast_to(phi, (6,))

    alpha = np.asarray(alpha).flatten()
    alpha_full = np.array([alpha[0], alpha[1], alpha[2], -alpha[2], -alpha[1], -alpha[0]])

    azimuths = np.array([np.pi/6, np.pi*3/6, np.pi*5/6, np.pi*7/6, np.pi*9/6, np.pi*11/6])
    cp = np.cos(phi_full)
    sp = np.sin(phi_full)
    prop_positions_body = l_full[:, None] * np.stack(
        [cp * np.cos(azimuths), cp * np.sin(azimuths), -sp], axis=1
    )  # (6, 3) in FRD body frame

    arm_unit    = prop_positions_body / np.maximum(np.linalg.norm(prop_positions_body, axis=1, keepdims=True), 1e-8)
    thrust_body = _rodrigues_np(
        np.tile(np.array([0.0, 0.0, -1.0]), (6, 1)),  # thrust is -z (up) in FRD
        arm_unit, alpha_full,
    )
    disc_offsets = np.stack([
        _disc_points(thrust_body[i], radius=PROP_DIAMETER / 2)
        for i in range(6)
    ])  # (6, n_pts, 3)

    for t, (state, raw_depth) in enumerate(zip(all_states, raw_depths)):
        rr.set_time("time", duration=t * dt)

        pos  = state[0:3]
        quat = state[6:10]
        R    = np.array(quat_to_rotmat(quat))  # FRD body → FRD world

        prop_world = pos + (R @ prop_positions_body.T).T  # (6, 3)

        arm_strips = [np.stack([pos, prop_world[i]]) for i in range(6)]
        rr.log("drone/arms",  rr.LineStrips3D(arm_strips, colors=[[80, 80, 220]], radii=0.004))
        rr.log("drone/body",  rr.Points3D([pos], colors=[[220, 80, 80]], radii=0.04))
        rr.log("drone/props", rr.Points3D(prop_world, colors=[[80, 180, 80]], radii=0.01))

        disc_strips = [
            np.concatenate([
                prop_world[i] + (R @ disc_offsets[i].T).T,
                prop_world[i] + disc_offsets[i, :1] @ R.T,
            ], axis=0)
            for i in range(6)
        ]
        rr.log("drone/discs", rr.LineStrips3D(disc_strips, colors=[[80, 220, 80]], radii=0.003))

        rr.log("drone/depth_image", rr.Image(_norm_depth(raw_depth)))

        roll, pitch, yaw = quat_to_euler(state[6:10])
        rr.log("state/x",              rr.Scalars(float(pos[0])))
        rr.log("state/y",              rr.Scalars(float(pos[1])))
        rr.log("state/z",              rr.Scalars(float(pos[2])))
        rr.log("state/vx",             rr.Scalars(float(state[3])))
        rr.log("state/vy",             rr.Scalars(float(state[4])))
        rr.log("state/vz",             rr.Scalars(float(state[5])))
        rr.log("state/roll",           rr.Scalars(float(roll)))
        rr.log("state/pitch",          rr.Scalars(float(pitch)))
        rr.log("state/yaw",            rr.Scalars(float(yaw)))
        rr.log("state/wx",             rr.Scalars(float(state[10])))
        rr.log("state/wy",             rr.Scalars(float(state[11])))
        rr.log("state/wz",             rr.Scalars(float(state[12])))
        rr.log("state/dist_to_target", rr.Scalars(float(np.linalg.norm(pos - state[19:22]))))

    rr.set_time("time", duration=len(all_states) * dt)
    traj = np.array([s[0:3] for s in all_states])
    rr.log("world/trajectory", rr.LineStrips3D([traj], colors=[[160, 210, 255]], radii=0.012))


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ecfg = config["env"]
    dcfg = config.get("depth_camera", {})
    env  = make_env(ecfg["name"], depth_camera=dcfg, **{k: v for k, v in ecfg.items() if k != "name"})

    pcfg   = config["policy"]
    policy = make_model(pcfg["type"], pcfg, obs_dim=env.obs_dim, act_dim=env.act_dim)

    policy_params, morph_params = load_checkpoint(args.checkpoint)

    if morph_params is not None and hasattr(env, "get_morph_info"):
        print("Morphology:")
        for k, v in env.get_morph_info(morph_params).items():
            print(f"  {k} = {v:.4f}")

    if morph_params is not None and hasattr(env, "get_l"):
        l     = np.array(env.get_l(morph_params))
        phi   = np.array(env.get_phi(morph_params))
        alpha = np.array(env.get_alpha(morph_params))
    else:
        l     = np.full(3, env.l_default)
        phi   = np.full(3, env.phi_default)
        alpha = (
            np.array([env.alpha_default, -env.alpha_default, env.alpha_default])
            if env.alternating_alpha else np.full(3, env.alpha_default)
        )

    key = jax.random.PRNGKey(args.seed)
    print("Running rollout …")
    all_states, raw_depths = run_rollout(env, policy, policy_params, key, args.steps)
    print(f"  {len(all_states)} states collected")

    rr.init("navigate_rollout")
    rr.log("/", rr.ViewCoordinates.FRD, static=True)  # x-forward, y-right, z-down

    print("Logging to Rerun …")
    visualise(env, all_states, raw_depths, l, phi, alpha, env.dt)

    rr.save(args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
