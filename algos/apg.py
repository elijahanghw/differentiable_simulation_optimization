"""
Analytic Policy Gradient (APG).

Backpropagates gradients through differentiable simulation dynamics using
jax.value_and_grad.  Supports joint morphology optimisation when the
environment exposes init_morph().

Algorithm
---------
For each epoch:
  1. Sample batch_size initial states via env.reset().
  2. Roll out each for horizon steps via env.step() using jax.lax.scan
     (fully differentiable through dynamics).
  3. Compute the discounted return for each rollout.
  4. Loss = -mean(returns)  →  maximise expected return.
  5. Backpropagate with jax.value_and_grad and update with Adam.
"""

from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax

from envs import make_env
from envs.multicopter.morphology import propeller_collision_loss_from_params
from models import make_model
from utils.logger import Logger
from utils.checkpoint import save as save_checkpoint


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _build_loss_fn(env, policy, horizon: int, gamma: float):
    discounts = gamma ** jnp.arange(horizon)

    def single_rollout(policy_params, init_state, init_obs):
        def step(carry, discount):
            state, obs = carry
            action = policy.apply(policy_params, obs)
            next_state, next_obs, reward, terminated, truncated, info = env.step(state, action)
            return (next_state, next_obs), reward * discount

        (_, _), weighted_rewards = jax.lax.scan(step, (init_state, init_obs), discounts)
        return jnp.sum(weighted_rewards)

    batch_rollout = jax.vmap(single_rollout, in_axes=(None, 0, 0))

    def loss_fn(policy_params, init_states, init_obs):
        returns = batch_rollout(policy_params, init_states, init_obs)
        mean_return = jnp.mean(returns)
        return -mean_return, mean_return

    return jax.jit(loss_fn)


def _build_loss_fn_morph(env, policy, horizon: int, gamma: float):
    discounts = gamma ** jnp.arange(horizon)

    def single_rollout(policy_params, morph_params, init_state, init_obs):
        def step(carry, discount):
            state, obs = carry
            action = policy.apply(policy_params, obs)
            next_state, next_obs, reward, terminated, truncated, info = env.step(state, action, morph_params)
            return (next_state, next_obs), reward * discount

        (_, _), weighted_rewards = jax.lax.scan(step, (init_state, init_obs), discounts)
        return jnp.sum(weighted_rewards)

    batch_rollout = jax.vmap(single_rollout, in_axes=(None, None, 0, 0))

    def loss_fn(policy_params, morph_params, init_states, init_obs):
        returns = batch_rollout(policy_params, morph_params, init_states, init_obs)
        mean_return = jnp.mean(returns)
        return -mean_return, mean_return

    return jax.jit(loss_fn)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: Dict[str, Any]) -> Any:
    tcfg       = config["training"]
    seed       = tcfg["seed"]
    epochs     = tcfg["epochs"]
    horizon    = tcfg["horizon"]
    batch_size = tcfg["batch_size"]
    lr         = tcfg["lr"]
    lr_min     = tcfg.get("lr_min", lr * 0.1)
    grad_clip  = tcfg.get("grad_clip", 1.0)
    gamma      = tcfg.get("gamma", 0.99)
    log_every  = tcfg.get("log_interval", 10)
    morph_lr            = tcfg.get("morph_lr", lr)
    morph_lr_min        = tcfg.get("morph_lr_min", morph_lr * 0.1)
    morph_epochs        = tcfg.get("morph_epochs", epochs)
    use_morph_loss  = tcfg.get("morphological_loss", False)
    morph_loss_weight    = tcfg.get("morphological_loss_weight", 100.0)

    # -- Environment --------------------------------------------------------
    ecfg      = config["env"]
    env       = make_env(ecfg["name"], **{k: v for k, v in ecfg.items() if k != "name"})
    has_morph      = hasattr(env, "init_morph") and getattr(env, "train_morphology", True)
    has_morph_info = hasattr(env, "get_morph_info")

    # -- Policy -------------------------------------------------------------
    pcfg        = config["policy"]
    layer_sizes = [env.obs_dim] + pcfg["hidden_sizes"] + [env.act_dim]
    policy      = make_model(pcfg["type"], layer_sizes=layer_sizes)

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    policy_params = policy.init(init_key)

    # -- Optimiser ----------------------------------------------------------
    policy_schedule = optax.linear_schedule(init_value=lr, end_value=lr_min, transition_steps=epochs)
    policy_optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(policy_schedule),
    )
    policy_opt_state = policy_optimizer.init(policy_params)

    if has_morph:
        morph_params    = env.init_morph()
        morph_schedule  = optax.cosine_decay_schedule(init_value=morph_lr, decay_steps=morph_epochs, alpha=morph_lr_min / morph_lr)
        morph_optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adam(morph_schedule, b1=0.5, b2=0.99),
        )
        morph_opt_state = morph_optimizer.init(morph_params)
        loss_fn = _build_loss_fn_morph(env, policy, horizon, gamma)
        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

        has_morphological_loss = (
            use_morph_loss
            and hasattr(env, "get_l")
            and hasattr(env, "get_phi")
            and hasattr(env, "get_alpha")
        )
        if has_morphological_loss:
            def _morphological_loss_fn(morph_params):
                return propeller_collision_loss_from_params(
                    env.get_l(morph_params),
                    env.get_phi(morph_params),
                    env.get_alpha(morph_params),
                    weight=morph_loss_weight,
                )
            morphological_grad_fn = jax.jit(jax.value_and_grad(_morphological_loss_fn))
    else:
        loss_fn = _build_loss_fn(env, policy, horizon, gamma)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # -- Logger -------------------------------------------------------------
    log_cfg   = config.get("logging", {})
    csv_path  = log_cfg.get("csv_path", "logs/training.csv")
    ckpt_path = log_cfg.get("checkpoint_path", "checkpoints/policy.pkl")
    fields = ["epoch", "mean_return", "loss", "grad_norm"]
    if has_morph and use_morph_loss:
        fields += ["morphological_loss"]
    if has_morph and has_morph_info:
        morph_info_keys = list(env.get_morph_info(env.init_morph()).keys())
        fields += morph_info_keys
    logger = Logger(csv_path, fields)

    # -- Info ---------------------------------------------------------------
    n_params = sum(w.size + b.size for w, b in policy_params)
    print(f"Algo         : APG")
    print(f"Devices      : {jax.devices()}")
    print(f"Env          : {ecfg['name']}  |  obs_dim={env.obs_dim}  act_dim={env.act_dim}")
    print(f"Morphology   : {'yes' if has_morph else 'no'}")
    print(f"Policy       : {pcfg['type']}  |  layers={layer_sizes}  params={n_params:,}")
    print(f"Horizon      : {horizon}  |  batch={batch_size}  epochs={epochs}")
    if has_morph:
        print(f"Optim        : Adam lr={lr}  morph_lr={morph_lr}  morph_epochs={morph_epochs}  grad_clip={grad_clip}  gamma={gamma}")
    else:
        print(f"Optim        : Adam lr={lr}  grad_clip={grad_clip}  gamma={gamma}")
    print("-" * 60)

    # -- Main loop ----------------------------------------------------------
    for epoch in range(epochs):
        key, state_key = jax.random.split(key)
        init_obs, init_states, _ = jax.vmap(env.reset)(jax.random.split(state_key, batch_size))

        if has_morph:
            (loss, mean_return), (policy_grads, morph_grads) = grad_fn(
                policy_params, morph_params, init_states, init_obs
            )
            policy_grad_norm = optax.global_norm(policy_grads)

            policy_updates, policy_opt_state = policy_optimizer.update(
                policy_grads, policy_opt_state
            )
            policy_params = optax.apply_updates(policy_params, policy_updates)

            morphological_loss = 0.0
            if epoch < morph_epochs:
                rollout_morph_grads = morph_grads
                if has_morphological_loss:
                    morphological_loss, morphological_grads = morphological_grad_fn(morph_params)
                    rollout_morph_grads = jax.tree.map(
                        lambda a, b: a + b, rollout_morph_grads, morphological_grads
                    )
                morph_updates, morph_opt_state = morph_optimizer.update(
                    rollout_morph_grads, morph_opt_state
                )
                morph_params = optax.apply_updates(morph_params, morph_updates)

            if epoch % log_every == 0:
                log_data = {
                    "epoch":       epoch,
                    "mean_return": float(mean_return),
                    "loss":        float(loss),
                    "grad_norm":   float(policy_grad_norm),
                }
                if has_morphological_loss:
                    log_data["morphological_loss"] = float(morphological_loss)
                if has_morph_info:
                    log_data.update(env.get_morph_info(morph_params))
                logger.log(log_data)
        else:
            (loss, mean_return), grads = grad_fn(policy_params, init_states, init_obs)
            grad_norm = optax.global_norm(grads)

            updates, policy_opt_state = policy_optimizer.update(grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, updates)

            if epoch % log_every == 0:
                logger.log({
                    "epoch":       epoch,
                    "mean_return": float(mean_return),
                    "loss":        float(loss),
                    "grad_norm":   float(grad_norm),
                })

    save_checkpoint(ckpt_path, policy_params, morph_params if has_morph else None)

    print("-" * 60)
    print(f"Training complete. Log saved to {csv_path}")
    if has_morph and has_morph_info:
        for k, v in env.get_morph_info(morph_params).items():
            print(f"  {k} = {v:.4f}")
    return policy_params
