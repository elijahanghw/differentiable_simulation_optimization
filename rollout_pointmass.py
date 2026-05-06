"""
Rollout for PointMassNavigate — renders one episode in Rerun (NED/FRD, z-down).

Usage:
  python rollout_pointmass.py \
      --config configs/navigate_pointmass.yaml \
      --checkpoint checkpoints/pointmass_navigate.pkl \
      [--steps 150] [--seed 0] [--output rollout_pointmass.rrd]
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
from models import make_model
from utils.checkpoint import load as load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps",      type=int, default=150)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--output",     type=str, default="rollout_pointmass.rrd")
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
            quats.append([0.0, 0.0, 0.0, 1.0])       # identity
        elif d < -1.0 + 1e-6:
            quats.append([1.0, 0.0, 0.0, 0.0])       # 180° around X
        else:
            rot_axis = np.cross(src, ax)
            rot_axis /= np.linalg.norm(rot_axis)
            half = np.arccos(np.clip(d, -1.0, 1.0)) / 2.0
            s = np.sin(half)
            quats.append([rot_axis[0] * s, rot_axis[1] * s, rot_axis[2] * s, np.cos(half)])
    return np.array(quats, dtype=np.float32)


def log_scene(scene_cfg, scene_array):
    """Log all obstacles and the ground plane as static geometry."""
    arrays = scene_cfg.unpack(scene_array)

    # Ground plane (thin slab at z = 0)
    rr.log("world/ground", rr.Boxes3D(
        centers=[[5.0, 0.0, 0.0]],
        half_sizes=[[12.0, 12.0, 0.02]],
        colors=[[130, 130, 130, 160]],
        fill_mode="solid",
    ), static=True)

    # Spheres
    sc = np.array(arrays["sphere_centers"])
    sr = np.array(arrays["sphere_radii"])
    if sc.shape[0] > 0:
        rr.log("world/spheres", rr.Ellipsoids3D(
            centers=sc,
            half_sizes=np.stack([sr, sr, sr], axis=1),
            colors=[[220, 100, 60, 200]],
            fill_mode="solid",
        ), static=True)

    # Boxes (axis-aligned)
    bc  = np.array(arrays["box_centers"])
    bhe = np.array(arrays["box_half_extents"])
    if bc.shape[0] > 0:
        rr.log("world/boxes", rr.Boxes3D(
            centers=bc,
            half_sizes=bhe,
            colors=[[60, 100, 220, 200]],
            fill_mode="solid",
        ), static=True)

    # Capsules — translate base to z=0 end, rotate +Z onto capsule axis
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


def run_rollout(env, policy, policy_params, key, steps):
    reset_key, base_key = jax.random.split(key)
    (depth_img, obs_vec), state, _ = env.reset(reset_key)

    hidden = policy.init_hidden()

    all_states = [jax.tree.map(np.array, state)]
    raw_depths = [np.array(env._get_depth(state))]

    for step_idx in range(steps):
        raw_act, hidden = policy.apply(
            {"params": policy_params}, depth_img, obs_vec, hidden
        )
        R      = state["R"]
        a_pred = R @ raw_act[:3]
        v_pred = R @ raw_act[3:]

        state, _ = env.step(state, a_pred, v_pred, base_key, step_idx)
        depth_img, obs_vec = env.get_obs(state)

        all_states.append(jax.tree.map(np.array, state))
        raw_depths.append(np.array(env._get_depth(state)))

    return all_states, raw_depths


def _norm_depth(raw):
    """Normalise raw depth (metres) to [0,1] with black=near, white=far."""
    d_min, d_max = raw.min(), raw.max()
    return ((raw - d_min) / (d_max - d_min + 1e-8)).astype(np.float32)


def visualise(env, all_states, raw_depths, dt):
    log_scene(env.scene_cfg, all_states[0]["scene"])

    # Target marker (static)
    rr.log("world/target", rr.Points3D(
        [all_states[0]["target_pos"]], colors=[[255, 215, 0]], radii=0.15,
    ), static=True)

    for t, (state, raw_depth) in enumerate(zip(all_states, raw_depths)):
        rr.set_time("time", duration=t * dt)

        R   = state["R"]
        rad = float(state["drone_radius"])
        rr.log("world/drone", rr.Ellipsoids3D(
            centers=[state["p"]],
            half_sizes=[[rad, rad, rad * 0.15]],
            quaternions=[_rotmat_to_quat_np(R)],
            colors=[[100, 180, 255]],
        ))

        cam_fwd = env._cam_cos * R[:, 0] + env._cam_sin * R[:, 2]
        rr.log("world/drone_cam_dir", rr.Arrows3D(
            origins=[state["p"]],
            vectors=[cam_fwd * rad * 4.0],
            colors=[[255, 200, 50]],
        ))

        rr.log("drone/depth_image", rr.Image(_norm_depth(raw_depth)))

        p, v = state["p"], state["v"]
        rr.log("state/x",  rr.Scalars(float(p[0])))
        rr.log("state/y",  rr.Scalars(float(p[1])))
        rr.log("state/z",  rr.Scalars(float(p[2])))
        rr.log("state/vx", rr.Scalars(float(v[0])))
        rr.log("state/vy", rr.Scalars(float(v[1])))
        rr.log("state/vz", rr.Scalars(float(v[2])))

    # Full trajectory logged once at the end
    rr.set_time("time", duration=len(all_states) * dt)
    traj = np.array([s["p"] for s in all_states])
    rr.log("world/trajectory", rr.LineStrips3D(
        [traj], colors=[[160, 210, 255]], radii=0.012,
    ))


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ecfg = config["env"]
    env  = make_env(ecfg["name"], **{k: v for k, v in ecfg.items() if k != "name"})

    pcfg   = config["policy"]
    policy = make_model("cnn_gru", pcfg, obs_dim=env.obs_dim, act_dim=env.act_dim)

    policy_params, _ = load_checkpoint(args.checkpoint)

    key = jax.random.PRNGKey(args.seed)
    print("Running rollout …")
    all_states, depth_imgs = run_rollout(env, policy, policy_params, key, args.steps)
    print(f"  {len(all_states)} states collected")

    rr.init("pointmass_navigate")
    rr.log("/", rr.ViewCoordinates.FLU, static=True)   # sim is z-up, x-forward, y-left

    print("Logging to Rerun …")
    visualise(env, all_states, depth_imgs, env.dt)

    rr.save(args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
