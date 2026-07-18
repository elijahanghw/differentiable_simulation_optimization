"""
(Truncated) Backpropagation Through Time — (T)BPTT.

Backpropagates gradients through differentiable simulation dynamics using
jax.value_and_grad.  Supports joint morphology optimisation when the
environment exposes init_morph().

Two rollout regimes (selected by training.persistent_carry)
-----------------------------------------------------------
* Fresh-batch BPTT (default, persistent_carry=false)
    Every epoch samples a fresh batch via env.reset() and rolls it out for
    ``horizon`` steps from a zero hidden state — the classic BPTT window.

* Persistent-carry / truncated BPTT (persistent_carry=true)
    A single (state, obs, hidden, age) carry is threaded across *all* epochs.
    Each epoch rolls the carry forward ``horizon`` steps, backprops through that
    one window, then ``stop_gradient``s the final carry and feeds it back as the
    next epoch's start. Gradients are truncated to a single window, but the
    state / hidden distribution seen in training is the on-policy, arbitrarily-
    aged one — not just steps 0..horizon from a zero hidden state — so the
    policy stays in-distribution far past ``horizon`` at eval.

    Resets happen per element *inside* the scan, never a whole-window reset:
      - reset_on_crash (default true): an element whose step reports
        ``step_data["crashed"]`` is re-seeded via env.reset the same step.
      - max_episode_len (default 0 = unbounded): an element that has flown this
        many steps without crashing is force-reset, keeping the fresh-reset
        state diversity from env.reset present in training as the policy
        improves. Age is tracked in the carry (the flat-array states carry no
        time field of their own).
    A reset re-inits that element's state / obs / hidden / age together and cuts
    the gradient there — the reset select is a clean gradient boundary, so we
    never backprop across a reset.

Environment contract
--------------------
  env.reset(key)                          → (obs, state, info)
  env.step(state, action, morph_matrices) → (next_state, next_obs, step_data)
  env.compute_morphology([morph_params])  → morph_matrices
  env.compute_loss(traj)                  → (total_loss, mean_return)
  step_data may include a bool "crashed"  → enables reset_on_crash.
"""

import math
import os
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax

from JADS.tasks import make_env
from JADS.drone_physics.morphology import propeller_collision_loss_from_params
from JADS.models import make_model, init_params
from JADS.utils.logger import Logger
from JADS.utils.checkpoint import save as save_checkpoint


# ---------------------------------------------------------------------------
# Rollout carry
# ---------------------------------------------------------------------------

def _init_carry(env, policy, reset_fn, keys, has_hidden):
    """Build a fresh batched rollout carry from a batch of reset keys.

    The carry is a dict so per-element resets (below) are a single tree_map and
    the shape stays self-documenting: state, obs, an integer age (steps since
    that element last reset — also used as the depth-render step index), and,
    for recurrent policies, the hidden state.
    """
    obs, states, _ = reset_fn(keys)
    batch = keys.shape[0]
    carry = {"state": states, "obs": obs, "age": jnp.zeros(batch, dtype=jnp.int32)}
    if has_hidden:
        h0 = policy.init_hidden()
        carry["hidden"] = jnp.broadcast_to(h0, (batch,) + h0.shape)
    return carry


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def _build_loss_fn(env, policy, reset_on_crash: bool, max_episode_len: int):
    """Build ``loss_fn(policy_params, morph_matrices, carry, reset_keys)``.

    One scan-based rollout covers every combination of {morph, no-morph} ×
    {recurrent, feed-forward} × {depth, plain}. ``morph_matrices`` is supplied
    by the caller (a constant for no-morph; derived from morph_params inside the
    differentiated wrapper for morph), so this builder is morphology-agnostic.

    The rollout threads a persistent carry and returns the final carry, so the
    training loop can either discard it (fresh-batch BPTT) or feed it back
    detached (persistent-carry / truncated BPTT). Per-element in-scan resets are
    only compiled in when ``reset_on_crash`` or ``max_episode_len`` is active.
    """
    has_hidden = hasattr(policy, "init_hidden")
    has_depth  = hasattr(policy, "conv_features")
    do_reset   = reset_on_crash or max_episode_len > 0

    def single_rollout(policy_params, morph_matrices, carry0, reset_keys):
        def step(carry, rkey):
            state, obs, age = carry["state"], carry["obs"], carry["age"]
            if has_hidden:
                hidden = carry["hidden"]
                if has_depth:
                    depth_img, obs_vec = obs
                    action, new_hidden = policy.apply(
                        {"params": policy_params}, depth_img, obs_vec, hidden)
                    new_state, new_obs, step_data = env.step(
                        state, action, morph_matrices=morph_matrices,
                        step_idx=age, prev_depth=depth_img,
                    )
                else:
                    action, new_hidden = policy.apply(
                        {"params": policy_params}, obs, hidden)
                    new_state, new_obs, step_data = env.step(
                        state, action, morph_matrices=morph_matrices)
            else:
                action = policy.apply({"params": policy_params}, obs)
                new_state, new_obs, step_data = env.step(
                    state, action, morph_matrices=morph_matrices)

            new_age = age + 1
            if do_reset:
                # One per-element reset mask over both triggers. Both are clean
                # gradient cuts: "crashed" is stop_gradient'd and age is a
                # discrete counter, so the select below never backprops a reset.
                should_reset = jnp.zeros((), dtype=bool)
                if reset_on_crash and "crashed" in step_data:
                    should_reset = should_reset | step_data["crashed"]
                if max_episode_len > 0:
                    should_reset = should_reset | (new_age >= max_episode_len)

                r_obs, r_state, _ = env.reset(rkey)
                sel = lambda a, b: jnp.where(should_reset, a, b)
                new_state = jax.tree_util.tree_map(sel, r_state, new_state)
                new_obs   = jax.tree_util.tree_map(sel, r_obs, new_obs)
                new_age   = jnp.where(should_reset, jnp.zeros_like(new_age), new_age)
                if has_hidden:
                    # Reset hidden TOGETHER with state — never fly a fresh
                    # position on a stale memory.
                    new_hidden = jnp.where(should_reset, policy.init_hidden(), new_hidden)

            new_carry = {"state": new_state, "obs": new_obs, "age": new_age}
            if has_hidden:
                new_carry["hidden"] = new_hidden
            return new_carry, step_data

        return jax.lax.scan(step, carry0, reset_keys)

    batch_rollout = jax.vmap(single_rollout, in_axes=(None, None, 0, 0))

    def loss_fn(policy_params, morph_matrices, carry, reset_keys):
        final_carry, traj = batch_rollout(policy_params, morph_matrices, carry, reset_keys)
        total_loss, mean_return = env.compute_loss(traj)
        aux = {"mean_return": mean_return, "final_carry": final_carry}
        if "crashed" in traj:
            aux["crash_frac"] = jnp.mean(traj["crashed"])
        return total_loss, aux

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

    # -- Truncated BPTT (persistent carry) ----------------------------------
    # persistent_carry off  → classic fresh-batch BPTT (resample every epoch).
    # persistent_carry on   → thread a detached (state, obs, hidden, age) carry
    #   across epochs, resetting elements in-scan on crash / age cap. The reset
    #   knobs only take effect while the carry persists.
    persistent_carry = tcfg.get("persistent_carry", False)
    reset_on_crash   = persistent_carry and tcfg.get("reset_on_crash", True)
    max_episode_len  = tcfg.get("max_episode_len", 0) if persistent_carry else 0

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
    has_hidden = hasattr(policy, "init_hidden")

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
        loss_fn = _build_loss_fn(env, policy, reset_on_crash, max_episode_len)

        # Differentiate w.r.t. both policy and morphology. morph_matrices is
        # derived from morph_params *inside* the differentiated function so its
        # gradient flows; it is constant across the rollout, computed once.
        def _rollout_loss(policy_params, morph_params, carry, reset_keys):
            morph_matrices = env.compute_morphology(morph_params)
            return loss_fn(policy_params, morph_matrices, carry, reset_keys)
        grad_fn = jax.jit(jax.value_and_grad(_rollout_loss, argnums=(0, 1), has_aux=True))

        has_morphological_loss = (
            use_morph_loss
            and hasattr(env, "get_l")
            and hasattr(env, "get_theta")
            and hasattr(env, "get_phi")
        )
        if has_morphological_loss:
            def _morphological_loss_fn(morph_params):
                psi   = env.get_psi(morph_params)   if hasattr(env, "get_psi")   else None
                alpha = env.get_alpha(morph_params) if hasattr(env, "get_alpha") else None
                return propeller_collision_loss_from_params(
                    env.get_l(morph_params),
                    psi=psi,
                    theta=env.get_theta(morph_params),
                    phi=env.get_phi(morph_params),
                    alpha=alpha,
                    weight=morph_loss_weight,
                )
            morphological_grad_fn = jax.jit(jax.value_and_grad(_morphological_loss_fn))
    else:
        loss_fn = _build_loss_fn(env, policy, reset_on_crash, max_episode_len)
        # No morphology to train: derived matrices are constant, computed once.
        morph_matrices = env.compute_morphology()
        def _rollout_loss(policy_params, carry, reset_keys):
            return loss_fn(policy_params, morph_matrices, carry, reset_keys)
        grad_fn = jax.jit(jax.value_and_grad(_rollout_loss, has_aux=True))

    # -- Logger -------------------------------------------------------------
    log_cfg   = config.get("logging", {})
    csv_path  = log_cfg.get("csv_path", "logs/training.csv")
    ckpt_path = log_cfg.get("checkpoint_path", "checkpoints/policy.pkl")
    fields = ["epoch", "mean_return", "loss", "grad_norm"]
    if reset_on_crash:
        fields += ["crash_frac"]
    if has_morph and use_morph_loss:
        fields += ["morphological_loss"]
    if has_morph and has_morph_info:
        morph_info_keys = list(env.get_morph_info(env.init_morph()).keys())
        fields += morph_info_keys
    logger = Logger(csv_path, fields)

    # -- Info ---------------------------------------------------------------
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(policy_params))
    print(f"Algo         : {'TBPTT (persistent-carry)' if persistent_carry else 'BPTT (fresh-batch)'}")
    print(f"Devices      : {jax.devices()}")
    print(f"Env          : {ecfg['name']}  |  obs_dim={env.obs_dim}  act_dim={env.act_dim}")
    print(f"Morphology   : {'yes' if has_morph else 'no'}")
    print(f"Policy       : {pcfg['type']}  |  params={n_params:,}")
    print(f"Horizon      : {horizon}  |  batch={batch_size}  epochs={epochs}")
    if persistent_carry:
        _cap = max_episode_len if max_episode_len > 0 else "unbounded"
        print(f"Carry        : reset_on_crash={reset_on_crash}  |  max_episode_len={_cap}")
    if has_morph:
        print(f"Optim        : Adam lr={lr}→{lr_min}  morph_lr={morph_lr}  morph_epochs={morph_epochs}  grad_clip={grad_clip}")
    else:
        print(f"Optim        : Adam lr={lr}→{lr_min}  grad_clip={grad_clip}")
    print("-" * 60)

    # -- Debug flags --------------------------------------------------------
    if tcfg.get("debug_nans", False):
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_disable_jit", True)
        print("WARNING: jax_debug_nans + jax_disable_jit enabled — training will be very slow")

    # -- Persistent XLA compilation cache ----------------------------------
    _cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".jax_cache")
    os.makedirs(os.path.abspath(_cache_dir), exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", os.path.abspath(_cache_dir))

    # -- Main loop ----------------------------------------------------------
    reset_fn = jax.jit(jax.vmap(env.reset))
    print("(JIT compiles on epoch 0 — cached to .jax_cache/ for future runs)")
    t_start = time.time()

    # Build the initial carry once. Fresh-batch BPTT rebuilds it every epoch;
    # persistent-carry (TBPTT) threads the detached final carry back in instead.
    key, ck = jax.random.split(key)
    carry = _init_carry(env, policy, reset_fn, jax.random.split(ck, batch_size), has_hidden)

    for epoch in range(epochs):
        if not persistent_carry:
            key, ck = jax.random.split(key)
            carry = _init_carry(env, policy, reset_fn, jax.random.split(ck, batch_size), has_hidden)

        # One fresh reset key per (element, step) for the in-scan crash/age
        # reset (a no-op consumed as scan xs when resets are disabled).
        key, rk = jax.random.split(key)
        reset_keys = jax.random.split(rk, batch_size * horizon).reshape(batch_size, horizon, 2)

        if has_morph:
            (loss, aux), (policy_grads, morph_grads) = grad_fn(
                policy_params, morph_params, carry, reset_keys
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
        else:
            (loss, aux), policy_grads = grad_fn(policy_params, carry, reset_keys)
            policy_grad_norm = optax.global_norm(policy_grads)

            policy_updates, policy_opt_state = policy_optimizer.update(policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_updates)

        # Truncate: detach the carry so next epoch backprops only its own
        # horizon window, never into this one.
        if persistent_carry:
            carry = jax.lax.stop_gradient(aux["final_carry"])

        if epoch % log_every == 0 or epoch == epochs - 1:
            log_data = {
                "epoch":       epoch,
                "mean_return": float(aux["mean_return"]),
                "loss":        float(loss),
                "grad_norm":   float(policy_grad_norm),
            }
            if reset_on_crash:
                log_data["crash_frac"] = float(aux.get("crash_frac", 0.0))
            if has_morph and has_morphological_loss:
                log_data["morphological_loss"] = float(morphological_loss)
            if has_morph and has_morph_info:
                log_data.update({
                    k: math.degrees(v) if k.startswith("theta") or k.startswith("phi") or k.startswith("alpha") or k.startswith("psi") else v
                    for k, v in env.get_morph_info(morph_params).items()
                })
            logger.log(log_data)
            print(f"  elapsed: {time.time() - t_start:.1f}s")

    save_checkpoint(ckpt_path, policy_params, morph_params if has_morph else None)

    print("-" * 60)
    print(f"Training complete. Log saved to {csv_path}")
    if has_morph and has_morph_info:
        for k, v in env.get_morph_info(morph_params).items():
            if k.startswith("theta") or k.startswith("phi") or k.startswith("alpha") or k.startswith("psi"):
                print(f"  {k} = {math.degrees(v):.4f} deg")
            else:
                print(f"  {k} = {v:.4f}")
    return policy_params
