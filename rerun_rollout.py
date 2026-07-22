"""
Task-agnostic Rerun rollout visualizer for JADS environments.

Works for both Hover and Navigate. Navigate-specific elements (target, obstacles,
depth images, collision distance) are logged automatically when the environment
exposes them. Uses FRD coordinates throughout (matches the simulation frame).

Usage:
  python rerun_rollout.py --config configs/hover.yaml   --checkpoint checkpoints/hover.pkl
  python rerun_rollout.py --config configs/navigate.yaml --checkpoint checkpoints/navigate.pkl
"""

import argparse
import random

import jax
import numpy as np
import rerun as rr
import yaml

from JADS.tasks import make_env
from JADS.drone_physics.quat_math import quat_to_rotmat, quat_to_euler
from JADS.drone_physics.morphology import PROP_DIAMETER, MOUNT_RADIUS
from JADS.models import make_model
from JADS.utils.checkpoint import load as load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps",      type=int, default=None)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--output",     type=str, default="rollout.rrd")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Drone geometry helpers
# ---------------------------------------------------------------------------

def _rodrigues_np(v, axis, angle):
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


def _build_drone_geometry(l, psi, theta, phi, alpha, mount_radius=MOUNT_RADIUS):
    """Propeller positions, mount points, and disc offsets in FRD body frame."""
    l = np.asarray(l).flatten()
    l_full = np.array([l[0], l[1], l[2], l[2], l[1], l[0]]) if l.size == 3 else np.broadcast_to(l, (6,)).copy()

    psi = np.asarray(psi).flatten()
    psi_full = np.array([psi[0], psi[1], psi[2], -psi[2], -psi[1], -psi[0]])

    theta = np.asarray(theta).flatten()
    theta_full = np.array([theta[0], theta[1], theta[2], theta[2], theta[1], theta[0]]) if theta.size == 3 else np.broadcast_to(theta, (6,)).copy()

    phi = np.asarray(phi).flatten()
    phi_full = np.array([phi[0], phi[1], phi[2], -phi[2], -phi[1], -phi[0]])

    alpha = np.asarray(alpha).flatten()
    alpha_full = np.array([alpha[0], alpha[1], alpha[2], alpha[2], alpha[1], alpha[0]])

    azimuths = np.array([np.pi/6, np.pi*3/6, np.pi*5/6, np.pi*7/6, np.pi*9/6, np.pi*11/6])
    arm_yaw = azimuths + psi_full

    mount_points = mount_radius * np.stack(
        [np.cos(azimuths), np.sin(azimuths), np.zeros_like(azimuths)], axis=1
    )  # (6, 3)

    cp = np.cos(theta_full);  sp = np.sin(theta_full)
    arm_unit = np.stack(
        [cp * np.cos(arm_yaw), cp * np.sin(arm_yaw), -sp], axis=1
    )  # (6, 3)

    prop_pos_body = mount_points + l_full[:, None] * arm_unit  # (6, 3)

    thrust_base   = np.tile(np.array([0.0, 0.0, -1.0]), (6, 1))
    tangential    = np.stack([-np.sin(arm_yaw), np.cos(arm_yaw), np.zeros_like(arm_yaw)], axis=1)
    thrust_pitched = _rodrigues_np(thrust_base, tangential, theta_full - alpha_full)
    thrust_body    = _rodrigues_np(thrust_pitched, arm_unit, phi_full)

    disc_offsets = np.stack([_disc_points(thrust_body[i], PROP_DIAMETER / 2) for i in range(6)])

    return prop_pos_body, mount_points, disc_offsets, thrust_body  # (6,3), (6,3), (6,n_pts,3), (6,3)


def _log_drone(t, state, prop_pos_body, mount_points_body, disc_offsets, thrust_body, dt):
    rr.set_time("time", duration=t * dt)

    pos  = state[0:3]
    quat = state[6:10]
    R    = np.array(quat_to_rotmat(quat))

    prop_world  = pos + (R @ prop_pos_body.T).T    # (6, 3)
    mount_world = pos + (R @ mount_points_body.T).T  # (6, 3)

    rr.log("drone/arms", rr.LineStrips3D(
        [np.stack([mount_world[i], prop_world[i]]) for i in range(6)],
        colors=[[80, 80, 220]], radii=0.004,
    ))
    body_z_world = R[:, 2]
    rr.log("drone/body", rr.Cylinders3D(
        centers=[pos],
        lengths=[0.05],
        radii=[0.05],
        quaternions=_quat_z_to_axis([body_z_world]),
        colors=[[220, 80, 80, 200]],
        fill_mode="solid",
    ))
    rr.log("drone/props", rr.Points3D(prop_world,  colors=[[80,  180, 80]], radii=0.01))
    rr.log("drone/discs", rr.LineStrips3D([
        np.concatenate([
            prop_world[i] + (R @ disc_offsets[i].T).T,
            prop_world[i] + disc_offsets[i, :1] @ R.T,
        ], axis=0)
        for i in range(6)
    ], colors=[[80, 220, 80]], radii=0.003))
    thrust_world = (R @ thrust_body.T).T  # (6, 3) — unit thrust vectors in world frame
    rr.log("drone/thrust", rr.Arrows3D(
        origins=prop_world,
        vectors=thrust_world * 0.08,
        colors=[[255, 200, 0]],
        radii=0.004,
    ))

    roll, pitch, yaw = quat_to_euler(quat)
    for name, val in [
        ("x",  state[0]),  ("y",     state[1]),  ("z",    state[2]),
        ("vx", state[3]),  ("vy",    state[4]),  ("vz",   state[5]),
        ("roll", roll),    ("pitch", pitch),     ("yaw",  yaw),
        ("wx", state[10]), ("wy",    state[11]), ("wz",   state[12]),
    ]:
        rr.log(f"state/{name}", rr.Scalars(float(val)))


# ---------------------------------------------------------------------------
# Navigate-specific: scene geometry
# ---------------------------------------------------------------------------

def _quat_z_to_axis(axes):
    src = np.array([0.0, 0.0, 1.0])
    quats = []
    for ax in axes:
        ax = ax / (np.linalg.norm(ax) + 1e-8)
        d  = float(np.dot(src, ax))
        if d > 1.0 - 1e-6:
            quats.append([0.0, 0.0, 0.0, 1.0])
        elif d < -1.0 + 1e-6:
            quats.append([1.0, 0.0, 0.0, 0.0])
        else:
            rot_axis  = np.cross(src, ax);  rot_axis /= np.linalg.norm(rot_axis)
            half      = np.arccos(np.clip(d, -1.0, 1.0)) / 2.0
            s         = np.sin(half)
            quats.append([rot_axis[0]*s, rot_axis[1]*s, rot_axis[2]*s, np.cos(half)])
    return np.array(quats, dtype=np.float32)


def _get_all_obstacles_for_region(scene_cfg, seed_float, x_min, x_max, y_min, y_max):
    """
    Enumerate all procedural obstacles in an x/y bounding region.
    Pure Python + JAX (not jitted) — only called once for visualization.

    Key split order matches scene.py get_local_obstacles (17 keys):
      0-3:  sphere cx, cy, cz, r
      4-9:  box cx, cy, cz, hx, hy, hz
      10-16: capsule cx, cy, cz, theta, phi, hh, r
    Trees use fold_in(cell_key, 1000) → 7 keys, matching scene.py exactly.
    Tree trunks are AABB (merged into boxes); branches are OBB.

    Returns:
        box_centers      (N, 3) numpy array  — includes trunk AABBs
        box_half_extents (N, 3) numpy array
        sphere_centers   (N, 3) numpy array
        sphere_radii     (N,)   numpy array
        cap_centers      (N, 3) numpy array  — standalone capsules only
        cap_axes         (N, 3) numpy array
        cap_hh           (N,)   numpy array
        cap_radii        (N,)   numpy array
        obb_centers      (N, 3) numpy array  — branch OBBs
        obb_quats        (N, 4) numpy array  — [qx,qy,qz,qw] rerun convention
        obb_half_extents (N, 3) numpy array
    """
    import math as _math
    cell_size = scene_cfg.cell_size
    Mb = scene_cfg.boxes_per_cell
    Ms = scene_cfg.spheres_per_cell
    Mc = scene_cfg.capsules_per_cell
    Mt = scene_cfg.trees_per_cell

    ix_min = int(np.floor(x_min / cell_size))
    ix_max = int(np.floor(x_max / cell_size))
    iy_min = int(np.floor(y_min / cell_size))
    iy_max = int(np.floor(y_max / cell_size))

    base_key = jax.random.PRNGKey(int(seed_float))

    all_box_c, all_box_he = [], []
    all_sph_c, all_sph_r  = [], []
    all_cap_c, all_cap_ax, all_cap_hh, all_cap_r = [], [], [], []
    all_obb_c, all_obb_q, all_obb_he = [], [], []

    for ix in range(ix_min, ix_max + 1):
        for iy in range(iy_min, iy_max + 1):
            # Cast via int32 → uint32 so negative cell indices wrap correctly,
            # matching the jnp.int32 behaviour in scene.py's get_local_obstacles.
            cell_key = jax.random.fold_in(
                jax.random.fold_in(base_key, np.uint32(np.int32(ix))),
                np.uint32(np.int32(iy)),
            )
            keys = jax.random.split(cell_key, 17)

            x0, x1 = ix * cell_size, (ix + 1) * cell_size
            y0, y1 = iy * cell_size, (iy + 1) * cell_size

            # Spheres
            if Ms > 0:
                s_cx = jax.random.uniform(keys[0], (Ms,), minval=x0, maxval=x1)
                s_cy = jax.random.uniform(keys[1], (Ms,), minval=y0, maxval=y1)
                s_cz = jax.random.uniform(keys[2], (Ms,), minval=scene_cfg.arena_z_min, maxval=scene_cfg.arena_z_max)
                s_r  = jax.random.uniform(keys[3], (Ms,), minval=scene_cfg.sphere_r_min, maxval=scene_cfg.sphere_r_max)
                all_sph_c.append(np.stack([np.array(s_cx), np.array(s_cy), np.array(s_cz)], axis=-1))
                all_sph_r.append(np.array(s_r))

            # Boxes
            if Mb > 0:
                cx = jax.random.uniform(keys[4], (Mb,), minval=x0, maxval=x1)
                cy = jax.random.uniform(keys[5], (Mb,), minval=y0, maxval=y1)
                cz = jax.random.uniform(keys[6], (Mb,), minval=scene_cfg.arena_z_min, maxval=scene_cfg.arena_z_max)
                hx = jax.random.uniform(keys[7], (Mb,), minval=scene_cfg.box_hx_min,  maxval=scene_cfg.box_hx_max)
                hy = jax.random.uniform(keys[8], (Mb,), minval=scene_cfg.box_hy_min,  maxval=scene_cfg.box_hy_max)
                hz = jax.random.uniform(keys[9], (Mb,), minval=scene_cfg.box_hz_min,  maxval=scene_cfg.box_hz_max)
                all_box_c.append(np.stack([np.array(cx), np.array(cy), np.array(cz)], axis=-1))
                all_box_he.append(np.stack([np.array(hx), np.array(hy), np.array(hz)], axis=-1))

            # Capsules
            if Mc > 0:
                c_cx  = jax.random.uniform(keys[10], (Mc,), minval=x0, maxval=x1)
                c_cy  = jax.random.uniform(keys[11], (Mc,), minval=y0, maxval=y1)
                c_cz  = jax.random.uniform(keys[12], (Mc,), minval=scene_cfg.arena_z_min, maxval=scene_cfg.arena_z_max)
                theta = jax.random.uniform(keys[13], (Mc,), minval=0.0, maxval=_math.pi)
                phi   = jax.random.uniform(keys[14], (Mc,), minval=0.0, maxval=2 * _math.pi)
                c_hh  = jax.random.uniform(keys[15], (Mc,), minval=scene_cfg.capsule_hh_min, maxval=scene_cfg.capsule_hh_max)
                c_r   = jax.random.uniform(keys[16], (Mc,), minval=scene_cfg.capsule_r_min,  maxval=scene_cfg.capsule_r_max)
                theta_np, phi_np = np.array(theta), np.array(phi)
                ax = np.sin(theta_np) * np.cos(phi_np)
                ay = np.sin(theta_np) * np.sin(phi_np)
                az = np.cos(theta_np)
                all_cap_c.append(np.stack([np.array(c_cx), np.array(c_cy), np.array(c_cz)], axis=-1))
                all_cap_ax.append(np.stack([ax, ay, az], axis=-1))
                all_cap_hh.append(np.array(c_hh))
                all_cap_r.append(np.array(c_r))

            # Trees — separate key stream via fold_in, matching scene.py exactly
            if Mt > 0:
                tree_key = jax.random.fold_in(cell_key, 1000)
                tkeys   = jax.random.split(tree_key, 7)

                t_x         = np.array(jax.random.uniform(tkeys[0], (Mt,), minval=x0, maxval=x1))
                t_y         = np.array(jax.random.uniform(tkeys[1], (Mt,), minval=y0, maxval=y1))
                t_trunk_hh  = np.array(jax.random.uniform(tkeys[2], (Mt,), minval=scene_cfg.tree_trunk_hh_min,   maxval=scene_cfg.tree_trunk_hh_max))
                t_trunk_r   = np.array(jax.random.uniform(tkeys[3], (Mt,), minval=scene_cfg.tree_trunk_r_min,    maxval=scene_cfg.tree_trunk_r_max))
                t_branch_hh = np.array(jax.random.uniform(tkeys[4], (Mt,), minval=scene_cfg.tree_branch_hh_min,  maxval=scene_cfg.tree_branch_hh_max))
                t_branch_r  = np.array(jax.random.uniform(tkeys[5], (Mt,), minval=scene_cfg.tree_branch_r_min,   maxval=scene_cfg.tree_branch_r_max))
                phi_off     = np.array(jax.random.uniform(tkeys[6], (Mt,), minval=0.0, maxval=2*_math.pi/3))

                # Trunk: AABB merged into box arrays
                trunk_c  = np.stack([t_x, t_y, -t_trunk_hh], axis=-1)           # (Mt, 3)
                trunk_he = np.stack([t_trunk_r, t_trunk_r, t_trunk_hh], axis=-1) # (Mt, 3)
                all_box_c.append(trunk_c)
                all_box_he.append(trunk_he)

                # Branches: OBB — fixed tilt angle
                sin_t = _math.sin(scene_cfg.tree_branch_tilt)
                cos_t = _math.cos(scene_cfg.tree_branch_tilt)
                for b in range(3):
                    phis_b    = phi_off + b * 2 * _math.pi / 3
                    b_ax      = np.stack([sin_t * np.cos(phis_b),
                                          sin_t * np.sin(phis_b),
                                          np.full_like(phis_b, -cos_t)], axis=-1)  # (Mt, 3)
                    trunk_top = np.stack([t_x, t_y, -2.0 * t_trunk_hh], axis=-1)
                    b_c       = trunk_top + t_branch_hh[:, None] * b_ax   # (Mt, 3)
                    b_he      = np.stack([t_branch_r, t_branch_r, t_branch_hh], axis=-1)
                    # Quaternion: rotate [0,0,1] → branch axis (half-angle, rerun uses [qx,qy,qz,qw])
                    bz   = b_ax[:, 2]
                    norm = np.sqrt(np.maximum(2.0 * (1.0 + bz), 1e-8))
                    qw = (1.0 + bz) / norm
                    qx = -b_ax[:, 1] / norm
                    qy =  b_ax[:, 0] / norm
                    qz = np.zeros_like(bz)
                    b_q = np.stack([qx, qy, qz, qw], axis=-1)  # rerun: [qx,qy,qz,qw]
                    all_obb_c.append(b_c)
                    all_obb_q.append(b_q)
                    all_obb_he.append(b_he)

    _empty3 = np.zeros((0, 3), dtype=np.float32)
    _empty4 = np.zeros((0, 4), dtype=np.float32)
    _empty1 = np.zeros((0,),   dtype=np.float32)
    return (
        np.concatenate(all_box_c,  axis=0) if all_box_c  else _empty3,
        np.concatenate(all_box_he, axis=0) if all_box_he else _empty3,
        np.concatenate(all_sph_c,  axis=0) if all_sph_c  else _empty3,
        np.concatenate(all_sph_r,  axis=0) if all_sph_r  else _empty1,
        np.concatenate(all_cap_c,  axis=0) if all_cap_c  else _empty3,
        np.concatenate(all_cap_ax, axis=0) if all_cap_ax else _empty3,
        np.concatenate(all_cap_hh, axis=0) if all_cap_hh else _empty1,
        np.concatenate(all_cap_r,  axis=0) if all_cap_r  else _empty1,
        np.concatenate(all_obb_c,  axis=0) if all_obb_c  else _empty3,
        np.concatenate(all_obb_q,  axis=0) if all_obb_q  else _empty4,
        np.concatenate(all_obb_he, axis=0) if all_obb_he else _empty3,
    )


def _log_scene(scene_cfg, scene_array, traj_positions=None):
    if scene_cfg.procedural:
        # Derive the region to visualize from the trajectory bounding box,
        # extended by one cell on each side so the drone's full view is covered.
        buf  = scene_cfg.cell_size
        seed = float(scene_array[0])
        xs, ys = traj_positions[:, 0], traj_positions[:, 1]
        x_min, x_max = float(xs.min()) - buf, float(xs.max()) + buf
        y_min, y_max = float(ys.min()) - buf, float(ys.max()) + buf

        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        hx, hy = (x_max - x_min) / 2 + 1.0, (y_max - y_min) / 2 + 1.0
        rr.log("world/ground", rr.Boxes3D(
            centers=[[cx, cy, 0.02]], half_sizes=[[hx, hy, 0.02]],
            colors=[[130, 130, 130, 255]], fill_mode="solid",
        ), static=True)

        bc, bhe, sc, sr, cap_c, cap_ax, cap_hh, cap_r, obb_c, obb_q, obb_he = \
            _get_all_obstacles_for_region(scene_cfg, seed, x_min, x_max, y_min, y_max)
        if bc.shape[0] > 0:
            rr.log("world/boxes", rr.Boxes3D(
                centers=bc, half_sizes=bhe, colors=[[60, 100, 220, 200]], fill_mode="solid",
            ), static=True)
        if sc.shape[0] > 0:
            rr.log("world/spheres", rr.Ellipsoids3D(
                centers=sc, half_sizes=np.stack([sr, sr, sr], axis=-1),
                colors=[[220, 100, 60, 200]], fill_mode="solid",
            ), static=True)
        if cap_c.shape[0] > 0:
            rr.log("world/capsules", rr.Capsules3D(
                lengths=(cap_hh * 2).astype(np.float32),
                radii=cap_r.astype(np.float32),
                translations=(cap_c - cap_hh[:, None] * cap_ax).astype(np.float32),
                quaternions=_quat_z_to_axis(cap_ax),
                colors=[[60, 200, 100, 200]],
                fill_mode="solid",
            ), static=True)
        if obb_c.shape[0] > 0:
            rr.log("world/branches", rr.Boxes3D(
                centers=obb_c, half_sizes=obb_he, quaternions=obb_q,
                colors=[[60, 200, 100, 200]], fill_mode="solid",
            ), static=True)
        return

    arrays = scene_cfg.unpack(scene_array)

    cx = (scene_cfg.arena_x_min + scene_cfg.arena_x_max) / 2
    cy = (scene_cfg.arena_y_min + scene_cfg.arena_y_max) / 2
    hx = (scene_cfg.arena_x_max - scene_cfg.arena_x_min) / 2 + 2.0
    hy = (scene_cfg.arena_y_max - scene_cfg.arena_y_min) / 2 + 2.0
    rr.log("world/ground", rr.Boxes3D(
        centers=[[cx, cy, 0.02]], half_sizes=[[hx, hy, 0.02]],
        colors=[[130, 130, 130, 255]], fill_mode="solid",
    ), static=True)

    sc = np.array(arrays["sphere_centers"]);  sr = np.array(arrays["sphere_radii"])
    if sc.shape[0] > 0:
        rr.log("world/spheres", rr.Ellipsoids3D(
            centers=sc, half_sizes=np.stack([sr, sr, sr], axis=1),
            colors=[[220, 100, 60, 200]], fill_mode="solid",
        ), static=True)

    bc  = np.array(arrays["box_centers"]);  bhe = np.array(arrays["box_half_extents"])
    if bc.shape[0] > 0:
        rr.log("world/boxes", rr.Boxes3D(
            centers=bc, half_sizes=bhe, colors=[[60, 100, 220, 200]], fill_mode="solid",
        ), static=True)

    cc  = np.array(arrays["cylinder_centers"]);  ca  = np.array(arrays["cylinder_axes"])
    chh = np.array(arrays["cylinder_hh"]);       cr  = np.array(arrays["cylinder_radii"])
    if cc.shape[0] > 0:
        rr.log("world/capsules", rr.Capsules3D(
            lengths=(2.0 * chh).astype(np.float32), radii=cr.astype(np.float32),
            translations=cc - ca * chh[:, None], quaternions=_quat_z_to_axis(ca),
            colors=[[60, 200, 100, 200]], fill_mode="solid",
        ), static=True)
    obb_c  = np.array(arrays["obb_centers"])
    obb_q  = np.array(arrays["obb_quats"])    # scene.py stores [qw,qx,qy,qz]; rerun uses [qx,qy,qz,qw]
    obb_he = np.array(arrays["obb_half_extents"])
    if obb_c.shape[0] > 0:
        obb_q_rr = obb_q[:, [1, 2, 3, 0]]   # reorder to rerun's [qx,qy,qz,qw]
        rr.log("world/branches", rr.Boxes3D(
            centers=obb_c, half_sizes=obb_he, quaternions=obb_q_rr,
            colors=[[60, 200, 100, 200]], fill_mode="solid",
        ), static=True)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_rollout(env, policy, policy_params, morph_params, key, steps):
    """
    Runs the physics/policy loop at env.dt while the depth camera (both the
    policy's own depth obs and the visualized vis_depth) only refreshes every
    env.frame_skip steps — the frames in between hold the last rendered image,
    matching what the policy actually saw at train/inference time.
    """
    has_hidden    = hasattr(policy, "init_hidden")
    has_depth     = hasattr(policy, "conv_features")
    has_vis_depth = hasattr(env, "get_vis_depth")
    frame_skip    = getattr(env, "frame_skip", 1)

    obs, state, _ = env.reset(key)
    states     = [np.array(state)]
    last_vis_depth = np.array(env.get_vis_depth(state)) if has_vis_depth else None
    vis_depths = [last_vis_depth] if has_vis_depth else None

    hidden = policy.init_hidden() if has_hidden else None

    for t in range(steps):
        if has_hidden:
            if has_depth:
                depth_img, obs_vec = obs
                action, hidden = policy.apply({"params": policy_params}, depth_img, obs_vec, hidden)
            else:
                action, hidden = policy.apply({"params": policy_params}, obs, hidden)
        else:
            action = policy.apply({"params": policy_params}, obs)

        if has_depth:
            state, obs, _ = env.step(state, action, morph_params, step_idx=t, prev_depth=depth_img)
        else:
            state, obs, _ = env.step(state, action, morph_params)
        states.append(np.array(state))
        if has_vis_depth:
            if (t + 1) % frame_skip == 0:
                last_vis_depth = np.array(env.get_vis_depth(state))
            vis_depths.append(last_vis_depth)

    return np.stack(states), vis_depths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ecfg       = config["env"]
    env_kwargs = {k: v for k, v in ecfg.items() if k != "name"}
    if "depth_camera" in config:
        env_kwargs["depth_camera"] = config["depth_camera"]
    env    = make_env(ecfg["name"], **env_kwargs)
    pcfg   = config["policy"]
    policy = make_model(pcfg["type"], pcfg, env.obs_dim, env.act_dim)

    policy_params, morph_params = load_checkpoint(args.checkpoint)

    if morph_params is not None and hasattr(env, "get_morph_info"):
        print("Morphology:")
        for k, v in env.get_morph_info(morph_params).items():
            print(f"  {k} = {v:.4f}")

    if morph_params is not None and hasattr(env, "get_l"):
        l     = np.array(env.get_l(morph_params))
        psi   = np.array(env.get_psi(morph_params))
        theta = np.array(env.get_theta(morph_params))
        phi   = np.array(env.get_phi(morph_params))
        alpha = np.array(env.get_alpha(morph_params))
    else:
        l     = np.full(3, env.l_default)
        psi   = np.full(3, env.psi_default)
        theta = np.full(3, env.theta_default)
        phi   = (
            np.array([env.phi_default, -env.phi_default, env.phi_default])
            if env.alternating_phi else np.full(3, env.phi_default)
        )
        alpha = np.full(3, env.alpha_default)

    steps = args.steps if args.steps is not None else config["training"]["horizon"]

    seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    print(f"Seed: {seed}")
    key = jax.random.PRNGKey(seed)
    print("Running rollout…")
    states, vis_depths = run_rollout(env, policy, policy_params, morph_params, key, steps)
    print(f"  {len(states)} steps collected")

    prop_pos_body, mount_points_body, disc_offsets, thrust_body = _build_drone_geometry(l, psi, theta, phi, alpha)

    rr.init(f"{ecfg['name']}_rollout")
    rr.log("/", rr.ViewCoordinates.FRD, static=True)

    if hasattr(env, "scene_cfg"):
        _log_scene(env.scene_cfg, states[0][22:], traj_positions=states[:, 0:3])
        rr.log("world/target", rr.Points3D(
            [states[0][19:22]], colors=[[255, 215, 0]], radii=0.15,
        ), static=True)

    rr.set_time("time", duration=len(states) * env.dt)
    rr.log("world/trajectory", rr.LineStrips3D(
        [states[:, 0:3]], colors=[[160, 210, 255]], radii=0.008,
    ))

    print("Logging to Rerun…")
    for t, state in enumerate(states):
        _log_drone(t, state, prop_pos_body, mount_points_body, disc_offsets, thrust_body, env.dt)

        if vis_depths is not None:
            rr.set_time("time", duration=t * env.dt)
            rr.log("drone/depth", rr.Image(
                np.clip(vis_depths[t] / env.cam_max_range, 0.0, 1.0).astype(np.float32)
            ))

        if hasattr(env, "scene_cfg"):
            rr.set_time("time", duration=t * env.dt)
            rr.log("state/dist_to_target", rr.Scalars(
                float(np.linalg.norm(state[0:3] - state[19:22]))
            ))

    rr.save(args.output)
    print(f"Saved → {args.output}  (open with: rerun {args.output})")


if __name__ == "__main__":
    main()
