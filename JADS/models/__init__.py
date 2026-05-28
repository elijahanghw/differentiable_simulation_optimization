import jax
import jax.numpy as jnp
from .mlp import MLP
from .gru import GRUActor
from .cnn_gru import CNNGRUActor


def make_model(name: str, pcfg: dict, obs_dim: int, act_dim: int, output_scale: float = 1.0):
    """
    Build an actor model from a policy config dict and env dimensions.

    MLP config keys:     hidden_sizes, squash_output
    GRU config keys:     encoder_sizes, hidden_size, head_sizes, squash_output
    CNN-GRU config keys: hidden_size, head_sizes, squash_output
    """
    squash_output = pcfg.get("squash_output", True)

    if name == "mlp":
        return MLP(
            hidden_sizes=pcfg["hidden_sizes"],
            out_dim=act_dim,
            squash_output=squash_output,
            output_scale=output_scale,
        )

    if name == "gru":
        return GRUActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            encoder_sizes=pcfg.get("encoder_sizes", (64,)),
            hidden_size=pcfg.get("hidden_size", 128),
            head_sizes=pcfg.get("head_sizes", (64,)),
            squash_output=squash_output,
        )

    if name == "cnn_gru":
        return CNNGRUActor(
            act_dim=act_dim,
            conv_features=pcfg.get("conv_features", (32, 64, 128)),
            kernel_sizes=pcfg.get("kernel_sizes", ((2, 2), (3, 3), (3, 3))),
            strides=pcfg.get("strides", ((2, 2), (1, 1), (1, 1))),
            proj_dim=pcfg.get("proj_dim", 192),
            leaky_slope=pcfg.get("leaky_slope", 0.05),
            hidden_size=pcfg.get("hidden_size", 128),
            squash_output=squash_output,
        )

    raise ValueError(f"Unknown model '{name}'. Available: mlp, gru, cnn_gru")


def init_params(model, key: jax.Array, obs_dim: int, depth_shape: tuple = None) -> dict:
    """
    Initialize model parameters, returning just the params dict.
    Pass depth_shape for CNNGRUActor, e.g. (24, 32) after pooling.
    """
    dummy_obs = jnp.zeros(obs_dim)
    if depth_shape is not None:
        dummy_depth = jnp.zeros(depth_shape)
        return model.init(key, dummy_depth, dummy_obs, model.init_hidden())["params"]
    if hasattr(model, "init_hidden"):
        return model.init(key, dummy_obs, model.init_hidden())["params"]
    return model.init(key, dummy_obs)["params"]


