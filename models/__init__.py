from .mlp import MLP, Critic

MODEL_REGISTRY = {
    "mlp": MLP,
}


def make_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def make_critic(layer_sizes):
    return Critic(layer_sizes)
