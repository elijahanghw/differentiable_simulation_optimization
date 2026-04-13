"""
Run two rollouts from separate checkpoints and save them to a single RRD file
so they can be viewed together in Rerun.

Usage
-----
  python compare_rollouts.py \
      --config configs/hover_2d.yaml \
      --checkpoints checkpoints/optimal.pkl checkpoints/fixed.pkl \
      --prefixes optimal fixed \
      --output comparison.rrd
"""

import argparse

import jax
import numpy as np
import rerun as rr
import yaml

from envs import make_env
from envs.multicopter.hover import Hover3d
from models import make_model
from utils.checkpoint import load as load_checkpoint
from rollout import run_rollout, visualise_2d, visualise_3d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--checkpoints", required=True, nargs="+")
    p.add_argument("--prefixes",    required=True, nargs="+")
    p.add_argument("--steps",       type=int, default=500)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--output",      type=str, default="comparison.rrd")
    return p.parse_args()


def main():
    args = parse_args()
    assert len(args.checkpoints) == len(args.prefixes), \
        "Number of checkpoints and prefixes must match"

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ecfg = config["env"]
    pcfg = config["policy"]

    is_3d = ecfg["name"] == "hover_3d"
    rr.init("hover_3d_comparison" if is_3d else "hover_2d_comparison")

    for checkpoint, prefix in zip(args.checkpoints, args.prefixes):
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
