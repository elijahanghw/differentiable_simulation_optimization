"""
Backpropagation Through Time (BPTT).

Backpropagates gradients through differentiable simulation dynamics using
jax.value_and_grad.  Supports joint morphology optimisation when the
environment exposes init_morph().

Algorithm
---------
For each epoch:
  1. Sample batch_size initial states via env.reset().
  2. Roll out each for horizon steps via env.step() using jax.lax.scan
     (fully differentiable through dynamics).
  3. Compute loss via env.compute_loss() on the accumulated trajectory.
  4. Backpropagate with jax.value_and_grad and update with Adam.

Environment contract
--------------------
  env.reset(key)              → (obs, state, info)
  env.step(state, action)     → (next_state, next_obs, step_data)
  env.step(state, action, mp) → (next_state, next_obs, step_data)  [morph]
  env.compute_loss(traj)      → (total_loss, mean_return)
"""

import os
import time
from typing import Any, Dict

import jax
import optax

from JADS.tasks import make_env
from JADS.drone_physics.morphology import propeller_collision_loss_from_params
from JADS.models import make_model, init_params
from JADS.utils.logger import Logger
from JADS.utils.checkpoint import save as save_checkpoint


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _build_loss_fn(env, policy, horizon: int):
    has_hidden = hasattr(policy, "init_hidden")
    has_depth  = hasattr(policy, "conv_features")

    def single_rollout(policy_params, init_state, init_obs):
        if has_hidden:
            def step(carry, _):
                state, obs, hidden = carry
                if has_depth:
                    depth_img, obs_vec = obs
                    action, new_hidden = policy.apply({"params": policy_params}, depth_img, obs_vec, hidden)
                else:
                    action, new_hidden = policy.apply({"params": policy_params}, obs, hidden)
                new_state, new_obs, step_data = env.step(state, action)
                return (new_state, new_obs, new_hidden), step_data
            init_carry = (init_state, init_obs, policy.init_hidden())
        else:
            def step(carry, _):
                state, obs = carry
                action = policy.apply({"params": policy_params}, obs)
                new_state, new_obs, step_data = env.step(state, action)
                return (new_state, new_obs), step_data
            init_carry = (init_state, init_obs)

        _, traj = jax.lax.scan(step, init_carry, None, length=horizon)
        return traj

    batch_rollout = jax.vmap(single_rollout, in_axes=(None, 0, 0))

    def loss_fn(policy_params, init_states, init_obs):
        traj = batch_rollout(policy_params, init_states, init_obs)
        total_loss, mean_return = env.compute_loss(traj)
        return total_loss, mean_return

    return loss_fn


def _build_loss_fn_morph(env, policy, horizon: int):
    has_hidden = hasattr(policy, "init_hidden")
    has_depth  = hasattr(policy, "conv_features")

    def single_rollout(policy_params, morph_params, init_state, init_obs):
        if has_hidden:
            def step(carry, _):
                state, obs, hidden = carry
                if has_depth:
                    depth_img, obs_vec = obs
                    action, new_hidden = policy.apply({"params": policy_params}, depth_img, obs_vec, hidden)
                else:
                    action, new_hidden = policy.apply({"params": policy_params}, obs, hidden)
                new_state, new_obs, step_data = env.step(state, action, morph_params)
                return (new_state, new_obs, new_hidden), step_data
            init_carry = (init_state, init_obs, policy.init_hidden())
        else:
            def step(carry, _):
                state, obs = carry
                action = policy.apply({"params": policy_params}, obs)
                new_state, new_obs, step_data = env.step(state, action, morph_params)
                return (new_state, new_obs), step_data
            init_carry = (init_state, init_obs)

        _, traj = jax.lax.scan(step, init_carry, None, length=horizon)
        return traj

    batch_rollout = jax.vmap(single_rollout, in_axes=(None, None, 0, 0))

    def loss_fn(policy_params, morph_params, init_states, init_obs):
        traj = batch_rollout(policy_params, morph_params, init_states, init_obs)
        total_loss, mean_return = env.compute_loss(traj)
        return total_loss, mean_return

    return loss_fn


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
    log_every  = tcfg.get("log_interval", 10)
    morph_lr            = tcfg.get("morph_lr", lr)
    morph_lr_min        = tcfg.get("morph_lr_min", morph_lr * 0.1)
    morph_epochs        = tcfg.get("morph_epochs", epochs)
    use_morph_loss      = tcfg.get("morphological_loss", False)
    morph_loss_weight   = tcfg.get("morphological_loss_weight", 100.0)

    # -- Environment --------------------------------------------------------
    ecfg       = config["env"]
    env_kwargs = {k: v for k, v in ecfg.items() if k != "name"}
    if "depth_camera" in config:
        env_kwargs["depth_camera"] = config["depth_camera"]
    env            = make_env(ecfg["name"], **env_kwargs)
    has_morph      = hasattr(env, "init_morph") and getattr(env, "train_morphology", True)
    has_morph_info = hasattr(env, "get_morph_info")

    # -- Policy -------------------------------------------------------------
    pcfg   = config["policy"]
    policy = make_model(pcfg["type"], pcfg, env.obs_dim, env.act_dim)

    key = jax.random.PRNGKey(seed)
    key, init_key, morph_init_key = jax.random.split(key, 3)
    depth_shape = None
    if pcfg["type"] == "cnn_gru" and hasattr(env, "cam_height"):
        pool = getattr(env, "cam_pool", 2)
        depth_shape = (env.cam_height // pool, env.cam_width // pool)
    policy_params = init_params(policy, init_key, env.obs_dim, depth_shape=depth_shape)

    # -- Optimiser ----------------------------------------------------------
    policy_schedule  = optax.linear_schedule(init_value=lr, end_value=lr_min, transition_steps=epochs)
    policy_optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(policy_schedule),
    )
    policy_opt_state = policy_optimizer.init(policy_params)

    if has_morph:
        morph_params    = env.init_morph(morph_init_key)
        morph_schedule  = optax.cosine_decay_schedule(init_value=morph_lr, decay_steps=morph_epochs, alpha=morph_lr_min / morph_lr)
        morph_optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adam(morph_schedule, b1=0.5, b2=0.99),
        )
        morph_opt_state = morph_optimizer.init(morph_params)
        loss_fn = _build_loss_fn_morph(env, policy, horizon)
        grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))

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
        loss_fn = _build_loss_fn(env, policy, horizon)
        grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

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
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(policy_params))
    print(f"Algo         : BPTT")
    print(f"Devices      : {jax.devices()}")
    print(f"Env          : {ecfg['name']}  |  obs_dim={env.obs_dim}  act_dim={env.act_dim}")
    print(f"Morphology   : {'yes' if has_morph else 'no'}")
    print(f"Policy       : {pcfg['type']}  |  params={n_params:,}")
    print(f"Horizon      : {horizon}  |  batch={batch_size}  epochs={epochs}")
    if has_morph:
        print(f"Optim        : Adam lr={lr}→{lr_min}  morph_lr={morph_lr}  morph_epochs={morph_epochs}  grad_clip={grad_clip}")
    else:
        print(f"Optim        : Adam lr={lr}→{lr_min}  grad_clip={grad_clip}")
    print("-" * 60)

    # -- Persistent XLA compilation cache ----------------------------------
    _cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".jax_cache")
    os.makedirs(os.path.abspath(_cache_dir), exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", os.path.abspath(_cache_dir))

    # -- Main loop ----------------------------------------------------------
    reset_fn = jax.jit(jax.vmap(env.reset))
    print("(JIT compiles on epoch 0 — cached to .jax_cache/ for future runs)")
    t_start = time.time()

    for epoch in range(epochs):
        key, state_key = jax.random.split(key)
        init_obs, init_states, _ = reset_fn(jax.random.split(state_key, batch_size))

        if has_morph:
            (loss, mean_return), (policy_grads, morph_grads) = grad_fn(
                policy_params, morph_params, init_states, init_obs
            )
            policy_grad_norm = optax.global_norm(policy_grads)

            policy_updates, policy_opt_state = policy_optimizer.update(policy_grads, policy_opt_state)
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

            if epoch % log_every == 0 or epoch == epochs - 1:
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
                print(f"  elapsed: {time.time() - t_start:.1f}s")
        else:
            (loss, mean_return), grads = grad_fn(policy_params, init_states, init_obs)
            grad_norm = optax.global_norm(grads)

            updates, policy_opt_state = policy_optimizer.update(grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, updates)

            if epoch % log_every == 0 or epoch == epochs - 1:
                logger.log({
                    "epoch":       epoch,
                    "mean_return": float(mean_return),
                    "loss":        float(loss),
                    "grad_norm":   float(grad_norm),
                })
                print(f"  elapsed: {time.time() - t_start:.1f}s")

    save_checkpoint(ckpt_path, policy_params, morph_params if has_morph else None)

    print("-" * 60)
    print(f"Training complete. Log saved to {csv_path}")
    if has_morph and has_morph_info:
        for k, v in env.get_morph_info(morph_params).items():
            print(f"  {k} = {v:.4f}")
    return policy_params
