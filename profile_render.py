"""
profile_render.py — JAX profiler trace for the depth renderer.

Warms up JIT compilation, then captures a trace that can be opened in
Perfetto (https://ui.perfetto.dev) to inspect CUDA kernels, occupancy,
and register counts.

Usage:
    python profile_render.py --config configs/navigate_fixed.yaml

Workflow (remote server):
    1. Run this script on the server — it prints the trace path when done.
    2. On your local machine:
           scp <user>@<server>:<trace_path> ~/trace.pb.gz
    3. Go to https://ui.perfetto.dev → "Open trace file" → select trace.pb.gz
"""

import argparse
import glob
import os

import jax
import jax.numpy as jnp

from train import load_config
from JADS.tasks import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--batch",   type=int, default=64)
    p.add_argument("--warmup",  type=int, default=10)
    p.add_argument("--steps",   type=int, default=20,  help="steps captured in trace")
    p.add_argument("--out",     default="./tmp/jax-render-trace")
    args = p.parse_args()

    cfg  = load_config(args.config)
    ecfg = cfg["env"]
    dcfg = cfg.get("depth_camera", {})
    env  = make_env(ecfg["name"], depth_camera=dcfg,
                    **{k: v for k, v in ecfg.items() if k != "name"})

    print(f"Config : {args.config}")
    print(f"Scene  : {env.scene_cfg.summary()}")
    print(f"Batch  : {args.batch}   warmup={args.warmup}   trace steps={args.steps}")

    keys = jax.random.split(jax.random.PRNGKey(0), args.batch)
    _, states, _ = jax.vmap(env.reset)(keys)

    render = jax.jit(jax.vmap(env._get_depth))

    print("\nWarming up (compiling JIT)...")
    for _ in range(args.warmup):
        jax.block_until_ready(render(states))
    print("Done. Starting trace...")

    # create_perfetto_trace=True writes a perfetto_trace.pb.gz directly in
    # args.out — this is what ui.perfetto.dev expects.
    with jax.profiler.trace(args.out, create_perfetto_trace=True):
        for _ in range(args.steps):
            jax.block_until_ready(render(states))

    matches = glob.glob(os.path.join(args.out, "**", "perfetto_trace.*"), recursive=True)
    trace_path = matches[0] if matches else args.out

    print("\nTrace written. To view it:")
    print(f"  Trace file : {trace_path}")
    print(f"\n  On your local machine:")
    print(f"    scp $SERVER:{trace_path} ~/render_trace.json.gz")
    print(f"  Then open https://ui.perfetto.dev and upload render_trace.json.gz")


if __name__ == "__main__":
    main()
