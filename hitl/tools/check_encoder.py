#!/usr/bin/env python3
"""
check_encoder.py — verify the latent depth_hitl publishes against the JAX encoder.

`compare_with_jax.py` checks the renderer; this checks the half after it. Run
depth_hitl with both streams up, pair the two by sequence number, and compare the
C latent against `CNNEncoder` applied to the very tensor that produced it:

    ./hitl/depth_hitl --scene hitl/scenes/room00.json --encode --pooled-out --no-preview
    python hitl/tools/check_encoder.py --n 50

The weights are identical by construction — hitl links the generated cnn_encoder.c
— so this is not a numerics test. It is a *wiring* test, and the failures it exists
to catch are the ones no amount of weight checking would find: the pooled tensor
arriving transposed, row/column swapped, stale by a frame, or scaled differently
from what the encoder was generated for. Those produce a plausible-looking latent
and a drone that flies into things.

Passing means the ground station is feeding the CNN exactly what the simulator fed
it in training, and the latent on the wire is the one the policy expects.
"""
from __future__ import annotations

import argparse
import os
import select
import socket
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "..")))

from recv_depth import PAYLOAD_FEATURES_F32, PAYLOAD_POOLED_F32, decode  # noqa: E402


def build_encoder(config_path: str, checkpoint: str):
    """The JAX CNNEncoder with the checkpoint's weights, as a callable."""
    import jax
    import jax.numpy as jnp

    from JADS.models.cnn_gru import CNNEncoder
    from JADS.utils.checkpoint import load as load_checkpoint
    from JADS.utils.config import load_config

    config = load_config(config_path)
    pcfg   = config["policy"]
    params, _ = load_checkpoint(checkpoint)
    enc_p  = params["CNNEncoder_0"]

    encoder = CNNEncoder(
        conv_features=pcfg.get("conv_features", (32, 64, 128)),
        kernel_sizes=pcfg.get("kernel_sizes", ((2, 2), (3, 3), (3, 3))),
        strides=pcfg.get("strides", ((2, 2), (1, 1), (1, 1))),
        proj_dim=pcfg.get("proj_dim", 192),
        leaky_slope=pcfg.get("leaky_slope", 0.05),
    )
    fn = jax.jit(lambda d: encoder.apply({"params": enc_p}, d))
    return lambda pooled: np.asarray(fn(jnp.asarray(pooled)))


def bind(port: int, host: str) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.setblocking(False)
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5010, help="latent stream [5010]")
    ap.add_argument("--pooled-port", type=int, default=5012, help="tensor stream [5012]")
    ap.add_argument("--n", type=int, default=50, help="frame pairs to compare [50]")
    ap.add_argument("--config", default="configs/train/navigate_real.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/navigate_real_64.pkl")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max |C - JAX| that still counts as a pass [1e-3]")
    args = ap.parse_args()

    print(f"loading {args.checkpoint} …")
    encode = build_encoder(args.config, args.checkpoint)

    lat_sock = bind(args.port, args.bind)
    pool_sock = bind(args.pooled_port, args.bind)
    print(f"listening: latent :{args.port}, pooled :{args.pooled_port} — "
          f"pairing {args.n} frames by seq")
    print("(depth_hitl needs --encode --pooled-out)")

    latents: dict[int, np.ndarray] = {}
    pooleds: dict[int, np.ndarray] = {}
    errs, rels, pairs = [], [], 0
    encode_us = []

    while pairs < args.n:
        ready, _, _ = select.select([lat_sock, pool_sock], [], [], 5.0)
        if not ready:
            print("\nno frames for 5 s — is depth_hitl running with --encode --pooled-out?",
                  file=sys.stderr)
            return 1

        for s in ready:
            try:
                f = decode(s.recvfrom(65535)[0])
            except ValueError as e:
                print(f"bad packet: {e}", file=sys.stderr)
                continue

            if f.payload_type == PAYLOAD_FEATURES_F32:
                latents[f.seq] = f.features
                encode_us.append(f.encode_us)
            elif f.payload_type == PAYLOAD_POOLED_F32:
                pooleds[f.seq] = np.asarray(f.data)
            else:
                continue

            if f.seq not in latents or f.seq not in pooleds:
                continue

            c_lat = latents.pop(f.seq)
            pooled = pooleds.pop(f.seq)
            j_lat = encode(pooled)

            err = float(np.max(np.abs(c_lat - j_lat)))
            scale = float(np.max(np.abs(j_lat)))
            errs.append(err)
            rels.append(err / max(scale, 1e-9))
            pairs += 1
            print(f"\r{pairs}/{args.n} pairs   max |Δ| {err:.3e}   "
                  f"latent span ±{scale:.3f}", end="", flush=True)

    errs, rels = np.array(errs), np.array(rels)
    print(f"\n\n{pairs} frame pairs")
    print(f"  max  |C - JAX|  {errs.max():.3e}    (relative {rels.max():.2e})")
    print(f"  mean |C - JAX|  {errs.mean():.3e}")
    if encode_us:
        e = np.array(encode_us)
        print(f"  encode cost     {e.mean() / 1000:.2f} ms mean, {e.max() / 1000:.2f} ms max")

    if errs.max() <= args.tol:
        print(f"\nPASS — the ground station feeds the CNN what training fed it, and the "
              f"latent on :{args.port} is the one the policy expects.")
        return 0
    print(f"\nFAIL — max {errs.max():.3e} exceeds --tol {args.tol:.0e}.\n"
          "  Weights cannot differ (hitl links the generated encoder), so suspect the\n"
          "  wiring: pooled tensor orientation, a frame offset between the streams, or\n"
          "  a scene camera that does not match the checkpoint.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
