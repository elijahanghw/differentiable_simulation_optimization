#!/usr/bin/env python3
"""
live_view.py — real-time rerun view of the HITL session.

Shows, live from the mocap:
  * the virtual obstacle field the drone is flying through
  * the drone's pose, its body axes, and the camera frustum
  * the depth image the CNN is being fed, projected into that frustum
  * the flown trajectory, and timing/health scalars

It subscribes to depth_hitl's output rather than to the mocap directly: every
depth packet carries the exact camera pose it was rendered from, so the map and
the image can never drift apart, and the viewer adds no load to the 10 Hz loop.

  # terminal 1 — pose source (or the real mocap)
  python hitl/tools/fake_mocap.py --traj orbit --cx 5 --radius 3.5 --speed 1.5

  # terminal 2 — renderer, with the full-resolution frame enabled
  ./hitl/depth_hitl --scene hitl/scenes/ep00.json --raw-out --no-preview

  # terminal 3 — this viewer
  python hitl/tools/live_view.py --scene hitl/scenes/ep00.json

`--raw-out` is worth it here: without it the viewer only receives the 12×16
tensor and shows that instead of the 48×64 image.

The obstacle field is logged once at startup, so if you change episode with the
'n' hotkey in depth_hitl, restart this viewer with the matching --scene.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import select
import socket
import sys
import time
from collections import deque

import numpy as np
import rerun as rr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recv_depth import PAYLOAD_POOLED_F32, PAYLOAD_RAW_U16_MM, decode


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quat_to_rotmat(q) -> np.ndarray:
    """[qw,qx,qy,qz] → 3×3, matching JADS/drone_physics/quat_math."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def quat_z_to_axis(axes: np.ndarray) -> np.ndarray:
    """Quaternions rotating +z onto each unit axis, in rerun's [qx,qy,qz,qw]."""
    axes = np.asarray(axes, dtype=np.float64).reshape(-1, 3)
    out = np.zeros((len(axes), 4))
    for i, a in enumerate(axes):
        a = a / max(np.linalg.norm(a), 1e-12)
        if a[2] < -0.999999:                       # antiparallel to +z
            out[i] = [1.0, 0.0, 0.0, 0.0]
            continue
        w = math.sqrt(max(2.0 * (1.0 + a[2]), 1e-18))
        out[i] = [-a[1] / w, a[0] / w, 0.0, w / 2.0]
    return out


# ---------------------------------------------------------------------------
# Static world
# ---------------------------------------------------------------------------

def log_scene(doc: dict) -> None:
    """Log the obstacle field once. Colours match rerun_rollout.py."""
    spheres   = doc.get("spheres", [])
    boxes     = doc.get("boxes", [])
    cylinders = doc.get("cylinders", [])
    obbs      = doc.get("obbs", [])

    pts = np.array([p["c"] for p in spheres + boxes + cylinders] or [[0, 0, 0]], np.float64)
    lo, hi = pts.min(0) - 3.0, pts.max(0) + 3.0

    if doc.get("ground_plane", True):
        rr.log("world/ground", rr.Boxes3D(
            centers=[[(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.02]],
            half_sizes=[[(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, 0.02]],
            colors=[[130, 130, 130, 255]], fill_mode="solid",
        ), static=True)

    if spheres:
        c = np.array([s["c"] for s in spheres], np.float32)
        r = np.array([s["r"] for s in spheres], np.float32)
        rr.log("world/spheres", rr.Ellipsoids3D(
            centers=c, half_sizes=np.stack([r, r, r], axis=1),
            colors=[[220, 100, 60, 200]], fill_mode="solid",
        ), static=True)

    if boxes:
        rr.log("world/boxes", rr.Boxes3D(
            centers=np.array([b["c"] for b in boxes], np.float32),
            half_sizes=np.array([b["he"] for b in boxes], np.float32),
            colors=[[60, 100, 220, 200]], fill_mode="solid",
        ), static=True)

    if cylinders:
        c  = np.array([x["c"]  for x in cylinders], np.float32)
        ax = np.array([x["ax"] for x in cylinders], np.float32)
        hh = np.array([x["hh"] for x in cylinders], np.float32)
        r  = np.array([x["r"]  for x in cylinders], np.float32)
        rr.log("world/capsules", rr.Capsules3D(
            lengths=2.0 * hh, radii=r,
            translations=c - ax * hh[:, None],      # Capsules3D starts at the base
            quaternions=quat_z_to_axis(ax),
            colors=[[60, 200, 100, 200]], fill_mode="solid",
        ), static=True)

    if obbs:
        q = np.array([o["q"] for o in obbs], np.float32)   # scene json is [qw,qx,qy,qz]
        rr.log("world/branches", rr.Boxes3D(
            centers=np.array([o["c"] for o in obbs], np.float32),
            half_sizes=np.array([o["he"] for o in obbs], np.float32),
            quaternions=q[:, [1, 2, 3, 0]],                # rerun wants [qx,qy,qz,qw]
            colors=[[60, 200, 100, 200]], fill_mode="solid",
        ), static=True)


def log_camera_static(cam: dict, rows: int, cols: int) -> None:
    """
    The frustum. focal = (W/2)/tan(fov_h/2) on both axes, because the simulator
    builds vertical rays with tan_v = tan_h / aspect (see depth_render/camera.py),
    which is exactly a square-pixel pinhole.
    """
    tan_h = math.tan(math.radians(cam["fov_deg"]) / 2.0)
    focal = (cols / 2.0) / tan_h
    rr.log("drone/camera", rr.Pinhole(
        focal_length=float(focal), width=int(cols), height=int(rows),
        # The camera looks along body +x, image right is +y, image down is +z.
        camera_xyz=rr.ViewCoordinates.FRD,
        image_plane_distance=1.0,
    ), static=True)


# ---------------------------------------------------------------------------
# Per-frame
# ---------------------------------------------------------------------------

def log_airframe_static(motor_pos_body, body_radius: float) -> None:
    """
    The airframe is rigid in the body frame, so it is logged once and carried by
    the per-frame Transform3D on "drone". Only the transform changes at 10 Hz.
    """
    rr.log("drone/body", rr.Ellipsoids3D(
        centers=[[0.0, 0.0, 0.0]], half_sizes=[[body_radius] * 3],
        colors=[[220, 80, 80, 220]], fill_mode="solid",
    ), static=True)

    # Body axes: forward red, right green, down blue.
    rr.log("drone/axes", rr.Arrows3D(
        origins=np.zeros((3, 3)), vectors=np.eye(3) * 0.25,
        colors=[[230, 60, 60], [60, 200, 60], [80, 120, 255]], radii=0.008,
    ), static=True)

    if motor_pos_body is not None:
        rr.log("drone/arms", rr.LineStrips3D(
            [np.stack([np.zeros(3), m]) for m in motor_pos_body],
            colors=[[80, 80, 220]], radii=0.004), static=True)
        rr.log("drone/motors", rr.Points3D(
            motor_pos_body, colors=[[80, 180, 80]], radii=0.02), static=True)


def log_drone_pose(pos: np.ndarray, quat: np.ndarray) -> None:
    rr.log("drone", rr.Transform3D(
        translation=pos, rotation=rr.Quaternion(xyzw=[quat[1], quat[2], quat[3], quat[0]]),
    ))


def to_metres(frame, cam: dict) -> np.ndarray:
    """Both payload types → a metric depth image, for the frustum projection."""
    if frame.payload_type == PAYLOAD_RAW_U16_MM:
        return np.asarray(frame.data, np.float32) / 1000.0
    # Invert the training normalization: v = num/clip(d) - off  →  d = num/(v + off).
    v = np.asarray(frame.data, np.float32)
    return cam["norm_numerator"] / np.maximum(v + cam["norm_offset"], 1e-3)


def to_proximity(depth_m: np.ndarray, cam: dict) -> np.ndarray:
    """
    Depth in metres → 0..1 where *bright means close*.

    Rerun's colormaps all run dark→bright with increasing value, so handing them
    raw depth paints the far plane — most of a typical frame — in the loudest
    colour. Obstacles are the signal here, so invert it: a pixel that hit nothing
    sits at max_range and becomes 0, and the blind zone (0 m, closer than
    min_range) is pinned to 1 rather than reading as infinitely far away.
    """
    d = np.asarray(depth_m, np.float32)
    prox = 1.0 - np.clip(d / cam["max_range"], 0.0, 1.0)
    return np.where(d <= 0.0, 1.0, prox).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="the scene JSON depth_hitl is running")
    ap.add_argument("--port", type=int, default=5010, help="pooled tensor port [5010]")
    ap.add_argument("--raw-port", type=int, default=5011,
                    help="full-resolution frame port [5011]; needs depth_hitl --raw-out")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--drone", default="configs/drone/2inchwhoop.yaml",
                    help="drone config, for drawing the airframe (optional)")
    ap.add_argument("--trail", type=int, default=300, help="trajectory points to keep [300]")
    ap.add_argument("--connect", action="store_true",
                    help="attach to an already-running viewer instead of spawning one")
    ap.add_argument("--save", help="record the session to this .rrd; on its own this "
                                   "records headlessly, add --spawn to also watch live")
    ap.add_argument("--spawn", action="store_true",
                    help="force spawning a viewer even when --save is given")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after this many seconds [run until Ctrl-C]")
    ap.add_argument("--colormap", default="grayscale",
                    choices=["grayscale", "viridis", "inferno", "magma", "plasma", "turbo"],
                    help="colormap for the metric depth image in the frustum "
                         "[grayscale]; note these all run dark→bright with distance, "
                         "so see the view/proximity image for a close-is-bright view")
    args = ap.parse_args()

    with open(args.scene) as f:
        doc = json.load(f)
    cam = doc["camera"]

    # The pooled tensor's brightest possible value: a pixel inside the blind zone,
    # which clamps to norm_clip_min. Scaling by it keeps the view stable frame to
    # frame instead of auto-ranging on whatever happens to be closest.
    cnn_full_scale = cam["norm_numerator"] / cam["norm_clip_min"] - cam["norm_offset"]

    motor_pos_body, body_radius = None, 0.05
    try:
        import yaml
        with open(args.drone) as f:
            geom = (yaml.safe_load(f) or {}).get("geometry", {})
        if geom.get("motor_positions"):
            motor_pos_body = np.array(geom["motor_positions"], np.float64)
        body_radius = float(geom.get("body_radius", body_radius))
    except (OSError, ImportError):
        pass   # airframe drawing is a nicety, not a requirement

    spawn = args.spawn or not (args.connect or args.save)
    rr.init("depth_hitl_live", spawn=spawn)
    if args.connect:
        rr.connect_grpc()
    if args.save:
        rr.save(args.save)

    rr.log("/", rr.ViewCoordinates.FRD, static=True)
    log_scene(doc)
    log_camera_static(cam, cam["height"], cam["width"])
    log_airframe_static(motor_pos_body, body_radius)

    socks = {}
    for port, kind in ((args.port, "pooled"), (args.raw_port, "raw")):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((args.bind, port))
        except OSError as e:
            print(f"warning: cannot bind {kind} port {port}: {e}", file=sys.stderr)
            continue
        s.setblocking(False)
        socks[s.fileno()] = (s, kind)

    if not socks:
        print("no ports available — is another viewer already running?", file=sys.stderr)
        return 1

    print(f"viewing {args.scene}\n"
          f"listening on {', '.join(str(s.getsockname()[1]) for s, _ in socks.values())}"
          f" — Ctrl-C to stop")
    if args.raw_port not in [s.getsockname()[1] for s, _ in socks.values()]:
        print("note: run depth_hitl with --raw-out for the full-resolution image")

    trail = deque(maxlen=args.trail)
    have_raw = False
    t0 = None
    n = 0
    last_arrival = None

    started = time.monotonic()
    try:
        while not (args.duration and time.monotonic() - started >= args.duration):
            ready, _, _ = select.select([s for s, _ in socks.values()], [], [], 1.0)
            if not ready:
                print("\rwaiting for frames from depth_hitl...", end="", flush=True)
                continue

            for sock in ready:
                try:
                    packet, _ = sock.recvfrom(65535)
                    frame = decode(packet)
                except (OSError, ValueError) as e:
                    print(f"\nbad packet: {e}", file=sys.stderr)
                    continue

                if t0 is None:
                    t0 = frame.send_time_us
                rr.set_time("time", duration=(frame.send_time_us - t0) / 1e6)
                rr.set_time("frame", sequence=frame.seq)

                is_raw = frame.payload_type == PAYLOAD_RAW_U16_MM
                have_raw = have_raw or is_raw

                # The full-resolution frame wins the frustum; the pooled tensor is
                # what the CNN actually eats, so show it in its own view.
                depth_m = to_metres(frame, cam)
                if is_raw or not have_raw:
                    # Metric, so rerun back-projects it into the frustum as points.
                    rr.log("drone/camera/depth", rr.DepthImage(
                        depth_m, meter=1.0, colormap=args.colormap,
                        depth_range=(0.0, float(cam["max_range"]))))
                    # Same frame, inverted for legibility: close is bright.
                    rr.log("view/proximity", rr.Image(to_proximity(depth_m, cam)))
                if frame.payload_type == PAYLOAD_POOLED_F32:
                    # The tensor exactly as the policy receives it — already
                    # "bright is close", since v = 3/clip(d) - 0.6.
                    rr.log("cnn_input", rr.Image(
                        np.asarray(frame.data, np.float32) / cnn_full_scale))

                # Pose only needs logging once per timestamp; the pooled frame and
                # the raw frame of a given seq carry the same one.
                pos = np.asarray(frame.pos, np.float64)
                log_drone_pose(pos, np.asarray(frame.quat, np.float64))
                trail.append(pos)
                if len(trail) > 1:
                    rr.log("world/trajectory", rr.LineStrips3D(
                        [np.array(trail)], colors=[[255, 220, 60]], radii=0.01))

                rr.log("health/pose_age_ms", rr.Scalars(frame.pose_age_us / 1000.0))
                rr.log("health/render_ms",   rr.Scalars(frame.render_us / 1000.0))
                rr.log("health/stale",       rr.Scalars(float(frame.stale)))
                rr.log("health/nearest_m",   rr.Scalars(float(depth_m[depth_m > 0].min())
                                                        if (depth_m > 0).any() else 0.0))

                n += 1
                arrival = time.monotonic()
                if last_arrival is not None:
                    rr.log("health/rate_hz", rr.Scalars(1.0 / max(arrival - last_arrival, 1e-6)))
                last_arrival = arrival
                if n % 10 == 0:
                    print(f"\rseq {frame.seq}  {n} frames  "
                          f"pos [{pos[0]:6.2f} {pos[1]:6.2f} {pos[2]:6.2f}]  "
                          f"{'STALE' if frame.stale else '     '}", end="", flush=True)
    except KeyboardInterrupt:
        pass

    print(f"\n{n} frames logged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
