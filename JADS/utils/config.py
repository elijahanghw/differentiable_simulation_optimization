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
