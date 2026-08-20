#!/usr/bin/env python3
"""
fake_mocap.py — stand in for the OptiTrack stream so the ground station can be
tested on the bench.

Sends the same 76-byte datagram relay.cpp consumes (see hitl/src/mocap.hpp),
flying a scripted trajectory through the scene.

  # fly +x across the arena at 1 m/s, 200 Hz, into a local depth_hitl
  python hitl/tools/fake_mocap.py --traj line --speed 1.0

  # orbit the obstacle field, looking at its centre
  python hitl/tools/fake_mocap.py --traj orbit --radius 3 --host 192.168.1.42

Ctrl-C to stop.
"""
from __future__ import annotations

import argparse
import math
import socket
import struct
import time

# uint32 id | uint64 t | 3f pos | 4f quat(xyzw) | 4B pad | uint64 t | 3f vel | 3f rate
PACKET = struct.Struct("<IQ3f4f4xQ3f3f")


def quat_from_rpy(roll: float, pitch: float, yaw: float):
    """[qw,qx,qy,qz] from ZYX Euler angles — matches quat_math.euler_to_quat."""
    cr, sr = math.cos(roll * 0.5),  math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5),   math.sin(yaw * 0.5)
    return (cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy)


def trajectory(kind: str, t: float, args):
    """Returns (pos, rpy, vel) in the NED training frame."""
    if kind == "hover":
        return (args.x0, args.y0, args.z0), (0.0, 0.0, args.yaw), (0.0, 0.0, 0.0)

    if kind == "line":
        x = args.x0 + args.speed * t
        return (x, args.y0, args.z0), (0.0, 0.0, args.yaw), (args.speed, 0.0, 0.0)

    if kind == "orbit":
        w = args.speed / max(args.radius, 1e-3)
        a = w * t
        x = args.cx + args.radius * math.cos(a)
        y = args.cy + args.radius * math.sin(a)
        vx = -args.radius * w * math.sin(a)
        vy =  args.radius * w * math.cos(a)
        yaw = math.atan2(args.cy - y, args.cx - x)      # look at the centre
        return (x, y, args.z0), (0.0, 0.0, yaw), (vx, vy, 0.0)

    if kind == "spin":
        return (args.x0, args.y0, args.z0), (0.0, 0.0, args.speed * t), (0.0, 0.0, 0.0)

    raise ValueError(kind)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--rate", type=float, default=200.0, help="send rate in Hz [200]")
    ap.add_argument("--rb-id", type=int, default=1, help="rigid-body streaming id [1]")
    ap.add_argument("--traj", default="line", choices=["hover", "line", "orbit", "spin"])
    ap.add_argument("--speed", type=float, default=1.0,
                    help="m/s (line, orbit) or rad/s (spin) [1.0]")
    ap.add_argument("--x0", type=float, default=0.0)
    ap.add_argument("--y0", type=float, default=0.0)
    ap.add_argument("--z0", type=float, default=-1.5, help="NED: negative is up [-1.5]")
    ap.add_argument("--yaw", type=float, default=0.0, help="radians [0]")
    ap.add_argument("--cx", type=float, default=5.0, help="orbit centre x [5]")
    ap.add_argument("--cy", type=float, default=0.0, help="orbit centre y [0]")
    ap.add_argument("--radius", type=float, default=3.0, help="orbit radius [3]")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds, 0 = forever")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    dest = (args.host, args.port)

    period = 1.0 / args.rate
    t0 = time.monotonic()
    n = 0
    print(f"sending {args.traj} to {args.host}:{args.port} at {args.rate:g} Hz "
          f"as rigid body {args.rb_id} — Ctrl-C to stop")

    try:
        while True:
            now = time.monotonic()
            t = now - t0
            if args.duration and t >= args.duration:
                break

            pos, rpy, vel = trajectory(args.traj, t, args)
            qw, qx, qy, qz = quat_from_rpy(*rpy)
            stamp = int(time.time() * 1e6)

            sock.sendto(PACKET.pack(args.rb_id, stamp,
                                    pos[0], pos[1], pos[2],
                                    qx, qy, qz, qw,
                                    stamp,
                                    vel[0], vel[1], vel[2],
                                    0.0, 0.0, 0.0), dest)
            n += 1
            if n % int(max(1, args.rate)) == 0:
                print(f"\r t={t:7.2f}s  pos [{pos[0]:6.2f} {pos[1]:6.2f} {pos[2]:6.2f}]  "
                      f"yaw {math.degrees(rpy[2]):6.1f}°  {n} packets", end="", flush=True)

            slack = (t0 + n * period) - time.monotonic()
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        pass

    print(f"\nsent {n} packets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
