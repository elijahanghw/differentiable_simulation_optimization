from .hover import Hover
from .navigate import Navigate


ENV_REGISTRY = {
    "hover": Hover,
    "navigate": Navigate,
}


def make_env(name: str, **kwargs):
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment '{name}'. Available: {list(ENV_REGISTRY)}")
    return ENV_REGISTRY[name](**kwargs)
