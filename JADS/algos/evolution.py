"""
Evolutionary morphology optimisation.

Outer loop: an evolution strategy (CMA-ES by default, via evosax) searches over
the morphology genotype -- the raw pre-sigmoid parameter vector.
Inner loop: for each candidate morphology, a fresh policy is trained from
scratch with BPTT (differentiable-simulation gradients through the fixed
morphology), then evaluated on a held-out batch of episodes. The candidate's
fitness is its mean eval return.

Modularity
----------
Any evosax strategy exposing init/ask/tell (CMA_ES, Open_ES, PGPE,
Sep_CMA_ES, ...) can be selected via config["evolution"]["strategy"]. The
genotype is the morph_params pytree itself -- evosax 0.2 batches and searches
pytrees directly, so no manual flatten/unflatten is needed: ask() returns each
morph_params leaf with a leading `popsize` axis.

Environment contract
--------------------
  env.init_morph()                       -> morph_params pytree (genotype template)
  env.compute_morphology(morph_params)   -> derived thrust/inertia matrices
  env.reset(key)                         -> (obs, state, info)
  env.step(state, action, morph_matrices=...) -> (next_state, next_obs, step_data)
  env.compute_loss(traj)                 -> (total_loss, mean_return)
"""

import dataclasses
import math
import os
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax

import evosax.algorithms as es_algos

from JADS.tasks import make_env
from JADS.models import make_model, init_params
from JADS.utils.logger import Logger
from JADS.utils.checkpoint import save as save_checkpoint
from .bptt import _build_loss_fn_morph


# ---------------------------------------------------------------------------
# Modular strategy factory
# ---------------------------------------------------------------------------

def _make_strategy(name: str, population_size: int, solution):
    """Instantiate an evosax strategy by class name (e.g. 'CMA_ES', 'Open_ES').

    All distribution-based evosax strategies share the (population_size,
    solution) constructor and the init/ask/tell interface, so swapping the
    algorithm is a one-line config change.
    """
    try:
        cls = getattr(es_algos, name)
    except AttributeError as e:
        raise ValueError(
            f"Unknown evolution strategy '{name}'. Choose an evosax.algorithms "
            f"class, e.g. CMA_ES, Open_ES, PGPE, Sep_CMA_ES."
        ) from e
    return cls(population_size=population_size, solution=solution)


# ---------------------------------------------------------------------------
# Inner-loop evaluator: train a fresh policy for a fixed morphology, then eval
# ---------------------------------------------------------------------------

def _make_evaluator(env, policy, tcfg, inner_epochs, depth_shape, vmap_population=False):
    horizon    = tcfg["horizon"]
    batch_size = tcfg["batch_size"]
    eval_batch = tcfg.get("eval_batch_size", 2 * batch_size)
    lr         = tcfg["lr"]
    lr_min     = tcfg.get("lr_min", lr * 0.1)
    grad_clip  = tcfg.get("grad_clip", 1.0)

    # loss_fn(policy_params, morph_params, init_states, init_obs) -> (loss, ret)
    loss_fn = _build_loss_fn_morph(env, policy, horizon)
    # Only the policy is differentiated; the morphology is fixed for the whole
    # inner run (the outer ES owns it).
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

    schedule  = optax.linear_schedule(init_value=lr, end_value=lr_min, transition_steps=inner_epochs)
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(schedule))
    reset_vmap = jax.vmap(env.reset)

    def evaluate(morph_params, train_key, eval_key):
        """Train a fresh policy for `morph_params`, return (eval_return,
        policy_params, final_train_return). `eval_key` is shared across a
        generation's candidates (common random numbers) for fair comparison."""
        init_key, scan_key = jax.random.split(train_key)
        policy_params = init_params(policy, init_key, env.obs_dim, depth_shape=depth_shape)
        opt_state     = optimizer.init(policy_params)
        epoch_keys    = jax.random.split(scan_key, inner_epochs)

        def epoch_step(carry, ek):
            pp, os = carry
            obs, states, _ = reset_vmap(jax.random.split(ek, batch_size))
            (loss, ret), grads = grad_fn(pp, morph_params, states, obs)
            updates, os = optimizer.update(grads, os)
            pp = optax.apply_updates(pp, updates)
            return (pp, os), ret

        (policy_params, _), train_rets = jax.lax.scan(
            epoch_step, (policy_params, opt_state), epoch_keys
        )

        # Held-out evaluation on fresh episodes.
        eobs, estates, _ = reset_vmap(jax.random.split(eval_key, eval_batch))
        _, eval_ret = loss_fn(policy_params, morph_params, estates, eobs)
        return eval_ret, policy_params, train_rets[-1]

    if vmap_population:
        # Train + evaluate the whole population in parallel. morph_params and
        # train_key are batched over the population axis; eval_key is shared
        # (common random numbers across candidates). Returns are batched:
        # eval_ret (popsize,), policy_params leaves (popsize, ...).
        evaluate = jax.vmap(evaluate, in_axes=(0, 0, None))
    return jax.jit(evaluate)


# ---------------------------------------------------------------------------
# Training loop (outer: ES over morphology, inner: BPTT over policy)
# ---------------------------------------------------------------------------

def train(config: Dict[str, Any]) -> Any:
    tcfg  = config["training"]
    ecfg  = config["env"]
    pcfg  = config["policy"]
    evcfg = config.get("evolution", {})

    seed          = tcfg["seed"]
    strategy_name = evcfg.get("strategy", "CMA_ES")
    popsize       = evcfg.get("popsize", 12)
    generations   = evcfg.get("generations", 30)
    sigma0        = evcfg.get("sigma0", 0.5)
    inner_epochs  = evcfg.get("inner_epochs", tcfg.get("epochs", 200))
    vmap_population = evcfg.get("vmap_population", False)
    log_every     = tcfg.get("log_interval", 1)

    # -- Environment --------------------------------------------------------
    env_kwargs = {k: v for k, v in ecfg.items() if k != "name"}
    if "depth_camera" in config:
        env_kwargs["depth_camera"] = config["depth_camera"]
    env = make_env(ecfg["name"], **env_kwargs)
    if not (hasattr(env, "init_morph") and getattr(env, "train_morphology", False)):
        raise ValueError(
            "The 'evolution' algo requires an env with train_morphology=true "
            "and an init_morph() method."
        )
    has_morph_info = hasattr(env, "get_morph_info")

    # -- Policy -------------------------------------------------------------
    policy = make_model(pcfg["type"], pcfg, env.obs_dim, env.act_dim)
    depth_shape = None
    if pcfg["type"] == "cnn_gru" and hasattr(env, "cam_height"):
        pool = getattr(env, "cam_pool", 2)
        depth_shape = (env.cam_height // pool, env.cam_width // pool)

    # -- Evolution strategy -------------------------------------------------
    solution  = env.init_morph()  # zeros dict -> genotype template & initial mean
    strategy  = _make_strategy(strategy_name, popsize, solution)
    es_params = strategy.default_params
    if hasattr(es_params, "std_init"):
        es_params = dataclasses.replace(es_params, std_init=sigma0)

    key = jax.random.PRNGKey(seed)
    key, es_key = jax.random.split(key)
    es_state = strategy.init(es_key, solution, es_params)

    evaluate = _make_evaluator(env, policy, tcfg, inner_epochs, depth_shape,
                               vmap_population=vmap_population)

    # -- Logger -------------------------------------------------------------
    log_cfg   = config.get("logging", {})
    csv_path  = log_cfg.get("csv_path", "logs/evolution.csv")
    ckpt_path = log_cfg.get("checkpoint_path", "checkpoints/evolution.pkl")
    fields = ["generation", "best_return", "gen_best_return", "gen_mean_return"]
    if has_morph_info:
        morph_info_keys = list(env.get_morph_info(solution).keys())
        fields += morph_info_keys
    logger = Logger(csv_path, fields)

    # -- Info ---------------------------------------------------------------
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(init_params(policy, key, env.obs_dim, depth_shape=depth_shape)))
    n_genes  = sum(x.size for x in jax.tree_util.tree_leaves(solution))
    print(f"Algo         : Evolution ({strategy_name})")
    print(f"Devices      : {jax.devices()}")
    print(f"Env          : {ecfg['name']}  |  obs_dim={env.obs_dim}  act_dim={env.act_dim}")
    print(f"Genotype     : {n_genes} raw morphology params  |  sigma0={sigma0}")
    print(f"Policy       : {pcfg['type']}  |  params={n_params:,}")
    print(f"Outer        : popsize={popsize}  generations={generations}  "
          f"eval={'vmap over population' if vmap_population else 'sequential'}")
    print(f"Inner        : BPTT epochs={inner_epochs}  horizon={tcfg['horizon']}  batch={tcfg['batch_size']}")
    print("-" * 60)

    # -- Persistent XLA compilation cache ----------------------------------
    _cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".jax_cache")
    os.makedirs(os.path.abspath(_cache_dir), exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", os.path.abspath(_cache_dir))

    # -- Main loop ----------------------------------------------------------
    print("(JIT compiles the inner training on the first candidate — cached to .jax_cache/)")
    t_start = time.time()

    best_return = -math.inf
    best_policy = None
    best_morph  = None

    for gen in range(generations):
        key, ask_key, tell_key, eval_key, train_key = jax.random.split(key, 5)
        population, es_state = strategy.ask(ask_key, es_state, es_params)

        # Shared eval episodes across candidates (common random numbers);
        # distinct train/init key per candidate.
        train_keys = jax.random.split(train_key, popsize)

        if vmap_population:
            # One fused call trains + evals every candidate in parallel.
            returns, policy_batched, _ = evaluate(population, train_keys, eval_key)

            def policy_at(i, _pb=policy_batched):
                return jax.tree_util.tree_map(lambda x: x[i], _pb)
        else:
            rets, policies = [], []
            for i in range(popsize):
                cand = jax.tree_util.tree_map(lambda x: x[i], population)
                eval_ret, policy_params, _ = evaluate(cand, train_keys[i], eval_key)
                rets.append(eval_ret)
                policies.append(policy_params)
            returns = jnp.stack(rets)

            def policy_at(i, _ps=policies):
                return _ps[i]

        # evosax minimises fitness -> negate the (maximised) return.
        es_state, _ = strategy.tell(tell_key, population, -returns, es_state, es_params)

        returns_host   = jax.device_get(returns)             # (popsize,)
        gen_best_i      = int(returns_host.argmax())
        gen_best_return = float(returns_host[gen_best_i])
        gen_mean_return = float(returns_host.mean())
        gen_best_morph  = jax.tree_util.tree_map(lambda x: x[gen_best_i], population)

        if gen_best_return > best_return:
            best_return = gen_best_return
            best_policy = jax.device_get(policy_at(gen_best_i))
            best_morph  = jax.device_get(gen_best_morph)

        if gen % log_every == 0 or gen == generations - 1:
            log_data = {
                "generation":      gen,
                "best_return":     float(best_return),
                "gen_best_return": float(gen_best_return),
                "gen_mean_return": gen_mean_return,
            }
            if has_morph_info:
                log_data.update({
                    k: math.degrees(v) if k[:-1] in ("psi", "theta", "phi", "alpha") else v
                    for k, v in env.get_morph_info(gen_best_morph).items()
                })
            logger.log(log_data)
            print(f"  gen {gen:3d}  best={best_return:8.3f}  "
                  f"gen_best={gen_best_return:8.3f}  gen_mean={gen_mean_return:8.3f}  "
                  f"elapsed={time.time() - t_start:.1f}s")

    save_checkpoint(ckpt_path, best_policy, best_morph)

    print("-" * 60)
    print(f"Evolution complete. Best eval return = {float(best_return):.4f}")
    print(f"Log saved to {csv_path}")
    if has_morph_info and best_morph is not None:
        for k, v in env.get_morph_info(best_morph).items():
            if k[:-1] in ("psi", "theta", "phi", "alpha"):
                print(f"  {k} = {math.degrees(v):.4f} deg")
            else:
                print(f"  {k} = {v:.4f}")
    return best_policy
