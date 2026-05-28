"""
Training entry point.

Loads a YAML config, applies any CLI overrides, and dispatches to the
algorithm specified by config["training"]["algo"] (default: "apg").

Supported algorithms
--------------------
  apg  — algos/apg.py   Analytic Policy Gradient (differentiable simulation)
  ppo  — algos/ppo.py   Proximal Policy Optimization

Usage
-----
  python train.py --config configs/pendulum.yaml
  python train.py --config configs/ppo_pendulum.yaml
  python train.py --config configs/pendulum.yaml --algo ppo --epochs 500
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from JADS.algos import ALGO_REGISTRY


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f)
    _resolve_scene_ref(config, path)
    return config


def _resolve_scene_ref(config: Dict[str, Any], config_path: str) -> None:
    """If env.scene is a path string, load it and replace with the parsed dict."""
    scene = config.get("env", {}).get("scene")
    if not isinstance(scene, str):
        return
    # Resolve relative to config file directory, then fall back to cwd
    candidates = [Path(config_path).parent / scene, Path(scene)]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate) as f:
                config["env"]["scene"] = yaml.safe_load(f)
            return
    raise FileNotFoundError(
        f"Scene config '{scene}' not found (tried: {[str(c) for c in candidates]})"
    )


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    for key in ("algo", "epochs", "lr", "seed", "batch_size", "horizon",
                "grad_clip", "gamma", "morph_lr"):
        val = getattr(args, key, None)
        if val is not None:
            config["training"][key] = val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a policy.")
    p.add_argument("--config",      required=True, help="Path to YAML config file.")
    p.add_argument("--algo",        type=str,   default=None, choices=list(ALGO_REGISTRY))
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--morph_lr",    type=float, default=None)
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--horizon",     type=int,   default=None)
    p.add_argument("--grad_clip",   type=float, default=None)
    p.add_argument("--gamma",       type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not Path(args.config).exists():
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    apply_overrides(config, args)

    # Randomise seed if not specified in config or via CLI
    if config["training"].get("seed") is None:
        config["training"]["seed"] = int.from_bytes(os.urandom(4), "big")
    print(f"Seed         : {config['training']['seed']}")

    algo = config["training"].get("algo", "apg").lower()
    if algo not in ALGO_REGISTRY:
        print(f"Unknown algo '{algo}'. Choose from: {list(ALGO_REGISTRY)}", file=sys.stderr)
        sys.exit(1)

    ALGO_REGISTRY[algo](config)
