"""
config.py — YAML config loading shared by the training and evaluation entry
points (train.py, rerun_rollout.py, eval_success_rate.py).

The one thing this adds over a plain yaml.safe_load is path-ref resolution:
`env.scene` and `env.drone` may be written as paths, and are replaced by the
parsed contents of the file they point at. Keeping this in one place matters
because an entry point that skips it hands the env a *string* where it expects
a dict, which fails deep inside the env constructor rather than at load time.
"""
from pathlib import Path
from typing import Any, Dict

import yaml

# env fields that may be given as a path to another YAML file.
_PATH_REFS = ("scene", "drone")


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f)
    for field in _PATH_REFS:
        _resolve_yaml_ref(config, path, field)
    return config


def _resolve_yaml_ref(config: Dict[str, Any], config_path: str, field: str) -> None:
    """If env.<field> is a path string, load it and replace with the parsed dict."""
    ref = config.get("env", {}).get(field)
    if not isinstance(ref, str):
        return
    # Resolve relative to config file directory, then fall back to cwd
    candidates = [Path(config_path).parent / ref, Path(ref)]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate) as f:
                config["env"][field] = yaml.safe_load(f)
            return
    raise FileNotFoundError(
        f"{field.capitalize()} config '{ref}' not found "
        f"(tried: {[str(c) for c in candidates]})"
    )
