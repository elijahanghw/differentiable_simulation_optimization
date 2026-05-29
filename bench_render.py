"""
bench_render.py — Time rendering and distance gradient computation in isolation.

Usage:
    python bench_render.py --config configs/navigate_fixed.yaml
    python bench_render.py --config configs/navigate_fixed.yaml --batch 64 --n 50
"""

import argparse
import time

import jax
import jax.numpy as jnp
import yaml
from pathlib import Path

from train import load_config
from JADS.tasks import make_env


# ---------------------------------------------------------------------------

def bench(fn, *args, warmup=5, n=50):
    """Run fn(*args) n times after warmup, return mean wall-clock seconds."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / n


def make_states(env, batch, seed=0):
    """Sample a batch of states via vmap(reset)."""
    keys = jax.random.split(jax.random.PRNGKey(seed), batch)
    _, states, _ = jax.vmap(env.reset)(keys)
    return states


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--batch",  type=int, default=64)
    p.add_argument("--n",      type=int, default=50,  help="timed iterations")
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    cfg  = load_config(args.config)
    ecfg = cfg["env"]
    dcfg = cfg.get("depth_camera", {})
    env  = make_env(ecfg["name"], depth_camera=dcfg,
                    **{k: v for k, v in ecfg.items() if k != "name"})

    print(f"\nConfig : {args.config}")
    print(f"Scene  : {env.scene_cfg.summary()}")
    print(f"Batch  : {args.batch}   n={args.n}   warmup={args.warmup}")
    print(f"State  : {env.state_dim}  obs: depth {env.cam_height}x{env.cam_width} + vec {env.obs_dim}\n")

    states = make_states(env, args.batch)

    # ------------------------------------------------------------------
    # 1. Depth rendering  (forward only — already stop_gradient in obs)
    # ------------------------------------------------------------------
    render_batched = jax.jit(jax.vmap(env._get_depth))

    ms_render = bench(render_batched, states,
                      warmup=args.warmup, n=args.n) * 1e3
    print(f"[render]   _get_depth (batch={args.batch}):               {ms_render:.2f} ms/call")

    # ------------------------------------------------------------------
    # 2. Distance computation  (forward only)
    # ------------------------------------------------------------------
    def dist_fwd(states):
        return jax.vmap(env._get_nearest_obstacle_dist)(states)

    dist_fwd_jit = jax.jit(dist_fwd)
    ms_dist_fwd = bench(dist_fwd_jit, states,
                        warmup=args.warmup, n=args.n) * 1e3
    print(f"[dist fwd] _get_nearest_obstacle_dist (batch={args.batch}): {ms_dist_fwd:.2f} ms/call")

    # ------------------------------------------------------------------
    # 3. Distance gradient  (backward — this is what APG backprops through)
    # ------------------------------------------------------------------
    def dist_loss(states):
        dists = jax.vmap(env._get_nearest_obstacle_dist)(states)
        return jnp.mean(dists)

    dist_grad_jit = jax.jit(jax.grad(dist_loss))
    ms_dist_grad = bench(dist_grad_jit, states,
                         warmup=args.warmup, n=args.n) * 1e3
    print(f"[dist bwd] grad(_get_nearest_obstacle_dist) (batch={args.batch}): {ms_dist_grad:.2f} ms/call")

    # ------------------------------------------------------------------
    # 4. Full _get_obs  (render + vec obs, as called during rollout)
    # ------------------------------------------------------------------
    get_obs_batched = jax.jit(jax.vmap(env._get_obs))
    ms_obs = bench(get_obs_batched, states,
                   warmup=args.warmup, n=args.n) * 1e3
    print(f"[obs]      _get_obs (batch={args.batch}):                  {ms_obs:.2f} ms/call")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\nSummary (per call, batch={args.batch}):")
    print(f"  Rendering alone : {ms_render:.2f} ms")
    print(f"  Dist fwd alone  : {ms_dist_fwd:.2f} ms")
    print(f"  Dist bwd alone  : {ms_dist_grad:.2f} ms")
    print(f"  Full obs        : {ms_obs:.2f} ms")
    print(f"  Bwd overhead    : {ms_dist_grad - ms_dist_fwd:.2f} ms  (bwd - fwd)")


if __name__ == "__main__":
    main()
