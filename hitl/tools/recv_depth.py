#!/usr/bin/env python3
"""
recv_depth.py — decode the frames depth_hitl publishes.

Doubles as the reference decoder for the CNN process: `decode()` returns the
12×16 float32 tensor the policy's CNNEncoder expects, already normalized and
max-pooled, so it can be fed straight in.

With `depth_hitl --encode` the same port carries the 64-float latent instead —
the CNN having already run on the ground station — and `decode()` returns that
as a (1, 64) array. `Frame.is_features` tells the two apart.

  # watch the stream and check the timing
  python hitl/tools/recv_depth.py --port 5010

  # print the tensor, and save every frame for offline inspection
  python hitl/tools/recv_depth.py --port 5010 --show --save frames.npz

Wire format: hitl/src/netout.hpp.
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from dataclasses import dataclass

import numpy as np

HEADER = struct.Struct("<4sHHIHHHHIQQII3f4f")   # 76 bytes
assert HEADER.size == 76

PAYLOAD_POOLED_F32   = 0
PAYLOAD_RAW_U16_MM   = 1
PAYLOAD_FEATURES_F32 = 2
FLAG_STALE_POSE      = 1 << 0

SUPPORTED_VERSIONS = (1, 2)


@dataclass
class Frame:
    seq:          int
    stale:        bool
    payload_type: int
    rows:         int
    cols:         int
    send_time_us: int
    pose_time_us: int
    pose_age_us:  int
    render_us:    int
    encode_us:    int          # CNN cost; 0 unless this frame carries features
    pos:          np.ndarray   # (3,) camera position used, NED
    quat:         np.ndarray   # (4,) [qw,qx,qy,qz]
    data:         np.ndarray   # (rows, cols) tensor, or (1, dim) latent

    @property
    def is_features(self) -> bool:
        return self.payload_type == PAYLOAD_FEATURES_F32

    @property
    def features(self) -> np.ndarray:
        """The latent as a flat (dim,) array. Raises on a non-feature frame."""
        if not self.is_features:
            raise ValueError(f"frame carries payload type {self.payload_type}, not features")
        return self.data.reshape(-1)


def decode(packet: bytes) -> Frame:
    """Raises ValueError if the datagram is not a well-formed depth frame."""
    if len(packet) < HEADER.size:
        raise ValueError(f"short packet ({len(packet)} bytes)")

    (magic, version, flags, seq, payload_type, encode_us, rows, cols, payload_bytes,
     send_us, pose_us, pose_age, render_us,
     px, py, pz, qw, qx, qy, qz) = HEADER.unpack_from(packet)

    if magic != b"DPTH":
        raise ValueError(f"bad magic {magic!r}")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported version {version}")
    if version < 2:
        encode_us = 0          # the field was reserved (always zero) before v2
    body = packet[HEADER.size:]
    if len(body) != payload_bytes:
        raise ValueError(f"payload is {len(body)} bytes, header says {payload_bytes}")

    if payload_type in (PAYLOAD_POOLED_F32, PAYLOAD_FEATURES_F32):
        data = np.frombuffer(body, dtype="<f4").reshape(rows, cols)
    elif payload_type == PAYLOAD_RAW_U16_MM:
        data = np.frombuffer(body, dtype="<u2").reshape(rows, cols)
    else:
        raise ValueError(f"unknown payload type {payload_type}")

    return Frame(seq=seq, stale=bool(flags & FLAG_STALE_POSE), payload_type=payload_type,
                 rows=rows, cols=cols, send_time_us=send_us, pose_time_us=pose_us,
                 pose_age_us=pose_age, render_us=render_us, encode_us=encode_us,
                 pos=np.array([px, py, pz], np.float32),
                 quat=np.array([qw, qx, qy, qz], np.float32),
                 data=data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5010)
    ap.add_argument("--show", action="store_true", help="print the tensor each frame")
    ap.add_argument("--save", help="write every frame to this .npz on exit")
    ap.add_argument("--count", type=int, default=0, help="stop after N frames [forever]")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind, args.port))
    print(f"listening on {args.bind}:{args.port} — Ctrl-C to stop")

    frames, poses, times = [], [], []
    prev_arrival = None
    gaps, last_seq, dropped = [], None, 0

    try:
        while True:
            packet, _ = sock.recvfrom(65535)
            arrival = time.monotonic()
            try:
                f = decode(packet)
            except ValueError as e:
                print(f"bad packet: {e}", file=sys.stderr)
                continue

            if last_seq is not None and f.seq != last_seq + 1:
                dropped += f.seq - last_seq - 1
            last_seq = f.seq
            if prev_arrival is not None:
                gaps.append((arrival - prev_arrival) * 1000.0)
            prev_arrival = arrival

            frames.append(f.features if f.is_features else np.asarray(f.data))
            poses.append(np.concatenate([f.pos, f.quat]))
            times.append([f.send_time_us, f.pose_time_us, f.pose_age_us,
                          f.render_us, f.encode_us])

            if args.show:
                what = f"latent[{f.cols}]" if f.is_features else f"{f.rows}x{f.cols}"
                enc  = f"  encode {f.encode_us} µs" if f.is_features else ""
                print(f"\nseq {f.seq}  {what}  "
                      f"{'STALE' if f.stale else 'ok'}  pose {f.pose_age_us / 1000:.1f} ms old  "
                      f"render {f.render_us} µs{enc}")
                print(f"pos {f.pos}  quat {f.quat}")
                if f.payload_type == PAYLOAD_POOLED_F32:
                    print("\n".join(" ".join(f"{v:5.2f}" for v in row) for row in f.data))
                elif f.is_features:
                    v = f.features
                    print("\n".join(" ".join(f"{x:6.3f}" for x in v[i:i + 8])
                                    for i in range(0, len(v), 8)))
            elif len(frames) % 10 == 0:
                g = np.array(gaps[-50:]) if gaps else np.array([0.0])
                enc = f"  encode {f.encode_us:4d} µs" if f.is_features else ""
                print(f"\rseq {f.seq}  {1000.0 / max(g.mean(), 1e-6):5.2f} Hz  "
                      f"period {g.mean():6.2f} ± {g.std():5.2f} ms (max {g.max():6.2f})  "
                      f"render {f.render_us:5d} µs{enc}  pose {f.pose_age_us / 1000:5.1f} ms  "
                      f"{'STALE' if f.stale else '     '}  dropped {dropped}",
                      end="", flush=True)

            if args.count and len(frames) >= args.count:
                break
    except KeyboardInterrupt:
        pass

    if gaps:
        g = np.array(gaps)
        print(f"\n\n{len(frames)} frames, {dropped} dropped")
        print(f"inter-frame period: mean {g.mean():.3f} ms, std {g.std():.3f} ms, "
              f"min {g.min():.3f}, max {g.max():.3f}  →  {1000.0 / g.mean():.4f} Hz")

    if args.save and frames:
        # Name the array for what it is: a latent stack is not a depth stack, and
        # a downstream script silently treating one as the other is a bad afternoon.
        key = "features" if f.is_features else "depth"
        np.savez_compressed(args.save, **{key: np.stack(frames)},
                            pose=np.stack(poses), timing=np.array(times, np.int64))
        print(f"wrote {args.save}  ({key} {np.stack(frames).shape}; "
              f"timing columns: send_us, pose_us, pose_age_us, render_us, encode_us)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
