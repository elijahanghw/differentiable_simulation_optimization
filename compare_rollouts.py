"""
Run two or more rollouts from separate checkpoints and save them to a single
RRD file so they can be viewed together in Rerun.

Each checkpoint can optionally have its own config. If --configs receives a
single file it is used for all checkpoints (original behaviour). If multiple
configs are provided there must be one per checkpoint.

Usage
-----
  # Same config for all:
  python compare_rollouts.py \
      --configs configs/hover_3d.yaml \
      --checkpoints checkpoints/a.pkl checkpoints/b.pkl \
      --prefixes fixed optimal

  # Per-checkpoint configs:
  python compare_rollouts.py \
      --configs configs/hover_3d_alternating_fixed.yaml configs/hover_3d_optimized.yaml \
      --checkpoints checkpoints/hover_3d_alternating_fixed.pkl checkpoints/hover_3d_optimized.pkl \
      --prefixes fixed optimal
"""

import argparse

import jax
import numpy as np
import rerun as rr
import yaml

from tasks import make_env
from tasks.multicopter.hover import Hover3d
from models import make_model
from utils.checkpoint import load as load_checkpoint
from rollout import run_rollout, visualise_2d, visualise_3d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--configs",      required=True, nargs="+")
    p.add_argument("--checkpoints",  required=True, nargs="+")
    p.add_argument("--prefixes",     required=True, nargs="+")
    p.add_argument("--steps",        type=int, default=500)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--output",       type=str, default="comparison.rrd")
    return p.parse_args()


def main():
    args = parse_args()
    n = len(args.checkpoints)
    assert len(args.prefixes) == n, "Number of prefixes must match checkpoints"
    assert len(args.configs) == 1 or len(args.configs) == n, \
        "Provide either one config (shared) or one per checkpoint"

    # Expand single config to all checkpoints
    config_paths = args.configs if len(args.configs) == n else args.configs * n

    configs = []
    for path in config_paths:
        with open(path) as f:
            configs.append(yaml.safe_load(f))

    is_3d = configs[0]["env"]["name"] == "hover_3d"
    rr.init("hover_3d_comparison" if is_3d else "hover_2d_comparison")

    for config, checkpoint, prefix in zip(configs, args.checkpoints, args.prefixes):
        ecfg = config["env"]
        pcfg = config["policy"]

        env = make_env(ecfg["name"], **{k: v for k, v in ecfg.items() if k != "name"})
        layer_sizes = [env.obs_dim] + pcfg["hidden_sizes"] + [env.act_dim]
        policy = make_model(pcfg["type"], layer_sizes=layer_sizes)

        policy_params, morph_params = load_checkpoint(checkpoint)

        if morph_params is not None and hasattr(env, "get_morph_info"):
            print(f"[{prefix}] Morphology:")
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
            visualise_3d(states, l=l, dt=env.dt, prefix=prefix, phi=phi, alpha=alpha)
        else:
            visualise_2d(states, l=l, dt=env.dt, prefix=prefix)

    rr.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
