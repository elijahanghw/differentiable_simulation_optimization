"""
Recurrent PPO (PPO-RNN) — model-free baseline for benchmarking against (T)BPTT.

Same environment, same policy architecture, same objective as algos/bptt.py; the
only thing that changes is how the gradient is obtained. BPTT differentiates
*through* the simulator; PPO estimates the policy gradient from sampled returns
and never touches the simulator's derivatives. Wall-clock and sample cost of the
two are therefore directly comparable — the logged ``mean_return`` is the same
quantity in both algorithms (see Navigate.step_reward).

Architecture
------------
The actor is built by ``models.make_model`` from the *same* ``policy:`` config
block BPTT uses, so the network being compared is identical. PPO adds:
  * a critic — a second network of the same type (``critic:`` config block,
    defaults to a copy of ``policy:``) with a 1-d output and its own hidden
    state, kept separate from the actor rather than sharing a trunk;
  * a state-independent ``log_std`` for the Gaussian exploration noise.
Actions are sampled unsquashed and clipped by the env, as in the reference impl.

Rollout regime
--------------
A persistent (state, obs, hidden, age, prev_dist) carry is threaded across all
updates, exactly like TBPTT's persistent_carry: each update rolls it forward
``horizon`` steps, does its PPO passes, and continues from where it stopped.
Resets:
  * reset_on_crash (default **false**): when true, ``step_data["crashed"]``
    terminates the episode; when false a crashed element keeps flying (as under
    TBPTT) and is re-seeded at the window boundary. See the warning below.
  * max_episode_len (default 0 = unbounded): truncates the episode (bootstrapped,
    since the episode did not really end).
An in-scan ``env.reset`` would cost a full depth render *every* step under vmap
(``cond`` lowers to ``select``), so instead a pool of ``reset_pool`` pre-sampled
fresh episodes is drawn once per update and an element that terminates swaps in
its next pool slot — a plain ``where``, no render. An element that terminates
more often than ``reset_pool`` times within one window reuses its last slot.

Why reset_on_crash defaults to false
------------------------------------
This task's reward is negative everywhere (it is a negated cost), so under
gamma=0.995 a healthy state is worth roughly -18,000. Textbook PPO bootstraps a
*terminal* state with 0 — which here is worth +18,000 relative to flying on, so
crashing becomes overwhelmingly the best action available and the agent learns
to do it deliberately. Measured: over 400 updates with terminal-zero bootstrap
the crash rate tripled (0.007 -> 0.023) while the return fell monotonically
(-92 -> -110) with clip_frac collapsing to 0.004, i.e. the policy confidently
converged on crashing. Two ways out, both supported:
  * reset_on_crash: false (default) — no termination at all, exactly TBPTT's
    treatment, so the exploit does not exist. Crashed elements are re-seeded at
    the window boundary.
  * reset_on_crash: true with crash_bootstrap: "value" (the default when
    termination is on) — the terminal state bootstraps V(s_t) instead of 0,
    making termination value-neutral; the crash is then paid for by its (large)
    collision reward rather than rewarded by the absence of a future.
    crash_bootstrap: "zero" restores textbook behaviour, and the pathology.
None of this arises in BPTT, which never terminates an episode.

Comparing the logged mean_return to BPTT's
-------------------------------------------
Both are the same per-step quantity (Navigate.step_reward is the negated
summand of Navigate.compute_loss — verified to agree to ~1e-4), but they are
only *numerically* comparable when the two runs crash at similar rates. Under
the default (reset_on_crash: false) both algorithms handle a crash identically
and the two numbers can be plotted on one axis. Switch termination on and they
diverge: PPO's episode ends the moment the drone touches something, whereas
TBPTT lets a crashed element keep flying to the end of the window, where the
collision term keeps growing as it sinks deeper into the geometry — so a
crash-heavy TBPTT epoch reports a far worse return than PPO for the same quality
of policy. In that case compare the trained checkpoints with
eval_success_rate.py instead, which is indifferent to either convention.

Environment contract (superset of the BPTT one)
-----------------------------------------------
  env.reset(key)                          → (obs, state, info)
  env.step(state, action, morph_matrices) → (next_state, next_obs, step_data)
  env.compute_morphology([morph_params])  → morph_matrices      (morphology envs)
  env.step_reward(step_data, prev_dist)   → scalar reward       (PPO only)
  env.initial_dist(state, morph_matrices) → prev_dist seed      (PPO only)
  info may include "params"               → per-episode domain randomization
  step_data may include a bool "crashed"  → enables reset_on_crash.

Morphology is *not* optimised here — the morphology gradient in BPTT comes from
differentiating the dynamics, which PPO cannot do. Point ``morph_checkpoint`` at
a BPTT checkpoint to train the policy on an already-optimised airframe.
"""

import math
import os
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax

from JADS.tasks import make_env
from JADS.models import make_model, init_params
from JADS.utils.logger import Logger
from JADS.utils.checkpoint import save as save_checkpoint, load as load_checkpoint

_LOG2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Diagonal Gaussian policy
# ---------------------------------------------------------------------------

def _clamp_log_std(log_std, min_log_std):
    """Apply the exploration floor. Clamping at the use site (rather than the
    stored parameter) keeps the optimiser state untouched and lets the raw
    parameter recover if the floor is later lowered."""
    return log_std if min_log_std is None else jnp.maximum(log_std, min_log_std)


def _log_prob(action, mean, log_std):
    """Log-density of a diagonal Gaussian, summed over the action dimension."""
    z = (action - mean) * jnp.exp(-log_std)
    return -0.5 * jnp.sum(z ** 2 + 2.0 * log_std + _LOG2PI, axis=-1)


def _entropy(log_std, act_dim):
    """Differential entropy of a diagonal Gaussian (independent of the mean)."""
    return jnp.sum(log_std) + 0.5 * act_dim * (1.0 + _LOG2PI)


# ---------------------------------------------------------------------------
# Running return statistics (reward normalisation)
# ---------------------------------------------------------------------------

def _rms_init():
    return {"mean": jnp.array(0.0), "var": jnp.array(1.0), "count": jnp.array(1e-4)}


def _rms_update(rms, x):
    """Parallel (Chan et al.) variance update from a batch of samples."""
    batch_mean, batch_var = jnp.mean(x), jnp.var(x)
    batch_count = jnp.array(float(x.size))
    delta = batch_mean - rms["mean"]
    total = rms["count"] + batch_count
    m2 = (rms["var"] * rms["count"] + batch_var * batch_count
          + delta ** 2 * rms["count"] * batch_count / total)
    return {"mean": rms["mean"] + delta * batch_count / total,
            "var":  m2 / total,
            "count": total}


# ---------------------------------------------------------------------------
# Pytree helpers
# ---------------------------------------------------------------------------

def _where_batch(mask, new, old):
    """Element-wise select over a batched pytree; `mask` is (batch,)."""
    def sel(a, b):
        m = mask.reshape((mask.shape[0],) + (1,) * (b.ndim - 1))
        return jnp.where(m, a, b)
    return jax.tree_util.tree_map(sel, new, old)


def _pool_take(pool, slot):
    """Gather each element's `slot`-th entry from a (batch, n_slots, ...) pool."""
    def take(x):
        idx = slot.reshape((slot.shape[0],) + (1,) * (x.ndim - 1))
        return jnp.take_along_axis(x, idx, axis=1)[:, 0]
    return jax.tree_util.tree_map(take, pool)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: Dict[str, Any]) -> Any:
    tcfg       = config["training"]
    seed       = tcfg["seed"]
    updates    = tcfg["epochs"]          # one "epoch" == one PPO update
    horizon    = tcfg["horizon"]         # steps collected per env per update
    batch_size = tcfg["batch_size"]      # number of parallel envs
    lr         = tcfg["lr"]
    lr_min     = tcfg.get("lr_min", lr * 0.1)
    grad_clip  = tcfg.get("grad_clip", 0.5)
    log_every  = tcfg.get("log_interval", 10)

    num_minibatches = tcfg.get("num_minibatches", 4)
    update_epochs   = tcfg.get("update_epochs", 4)
    gamma           = tcfg.get("gamma", 0.99)
    gae_lambda      = tcfg.get("gae_lambda", 0.95)
    clip_eps        = tcfg.get("clip_eps", 0.2)
    vf_coef         = tcfg.get("vf_coef", 0.5)
    ent_coef        = tcfg.get("ent_coef", 0.0)
    clip_value_loss = tcfg.get("clip_value_loss", True)
    init_log_std    = tcfg.get("init_log_std", -1.0)
    learn_std       = tcfg.get("learn_std", True)
    # Floor on the exploration std. With learn_std and no entropy bonus the
    # surrogate always prefers a sharper policy, so std decays monotonically
    # until the policy is deterministic and stops discovering anything — and,
    # because the policy gradient scales as (a-mu)/std^2, the gradient norm
    # blows up on the way down until grad_clip is all that is left. A floor
    # costs nothing when exploration is healthy and prevents that collapse.
    min_log_std     = tcfg.get("min_log_std", None)
    anneal_lr       = tcfg.get("anneal_lr", True)
    normalize_rew   = tcfg.get("normalize_reward", True)
    reward_clip     = tcfg.get("reward_clip", 10.0)
    reward_scale    = tcfg.get("reward_scale", 1.0)
    smooth_action   = tcfg.get("smooth_action_loss", 0.0)
    # Potential-based progress shaping (Ng et al. 1999) with potential
    # Phi(s) = -k*dist_to_target(s), i.e. F = gamma*Phi(s') - Phi(s)
    #                                      = k*(d - gamma*d').
    # Because it is potential-based it provably leaves the optimal policy
    # unchanged, so the benchmark's objective is not altered — it only supplies
    # a dense directional signal in place of "the huge distance^2 penalty got
    # slightly smaller", which is what leaves PPO stuck hovering at spawn.
    # mean_return is always logged on the UNSHAPED reward, so it stays
    # comparable to BPTT's. 0 disables.
    progress_weight = tcfg.get("progress_weight", 0.0)
    trunc_bootstrap = tcfg.get("truncation_bootstrap", True)
    reset_on_crash  = tcfg.get("reset_on_crash", False)
    crash_bootstrap = tcfg.get("crash_bootstrap", "value")
    max_episode_len = tcfg.get("max_episode_len", 0)
    n_slots         = max(1, tcfg.get("reset_pool", 4))

    if crash_bootstrap not in ("zero", "value"):
        raise ValueError(
            f"crash_bootstrap must be 'zero' or 'value', got '{crash_bootstrap}'")
    if batch_size % num_minibatches != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by num_minibatches "
            f"({num_minibatches}) — recurrent PPO splits minibatches along the "
            f"env dimension so each sequence stays intact in time."
        )

    # -- Environment --------------------------------------------------------
    ecfg       = config["env"]
    env_kwargs = {k: v for k, v in ecfg.items() if k != "name"}
    if "depth_camera" in config:
        env_kwargs["depth_camera"] = config["depth_camera"]
    env = make_env(ecfg["name"], **env_kwargs)

    if not (hasattr(env, "step_reward") and hasattr(env, "initial_dist")):
        raise ValueError(
            f"Env '{ecfg['name']}' exposes no per-step reward. PPO needs "
            f"step_reward(step_data, prev_dist) and initial_dist(state, "
            f"morph_matrices); see JADS/tasks/navigate.py."
        )

    # -- Morphology (fixed — PPO has no gradient through the dynamics) ------
    morph_ckpt   = tcfg.get("morph_checkpoint")
    morph_params = None
    if morph_ckpt is not None:
        _, morph_params = load_checkpoint(morph_ckpt)
        print(f"Morphology loaded from {morph_ckpt} (frozen)")
    elif getattr(env, "train_morphology", False):
        print("WARNING: env.train_morphology=true, but PPO cannot optimise "
              "morphology (no gradient through the simulator). Falling back to "
              "the env's default airframe; set training.morph_checkpoint to a "
              "BPTT checkpoint to benchmark on an optimised one.")
    # A *_real env has a fixed identified airframe and no morphology at all.
    morph_matrices = (env.compute_morphology(morph_params)
                      if hasattr(env, "compute_morphology") else None)

    # -- Actor / critic -----------------------------------------------------
    # Actor is built from the same `policy:` block BPTT uses, so the compared
    # network is identical. The critic defaults to the same architecture with a
    # scalar output and its own recurrent state.
    pcfg = config["policy"]
    ccfg = dict(config.get("critic", pcfg))
    ccfg.setdefault("type", pcfg["type"])
    ccfg.setdefault("hidden_sizes", (256, 256))
    ccfg["squash_output"] = False

    actor = make_model(pcfg["type"], pcfg, env.obs_dim, env.act_dim)
    has_depth = hasattr(actor, "conv_features")

    # A feed-forward critic cannot consume the actor's (depth image, vector)
    # observation — there is no CNN-without-GRU model — so it reads the env's
    # privileged state instead. That is the point of it: privileged state makes
    # the value problem fully observed, so there is no history left for a
    # recurrent critic to infer, and the value re-run stops being sequential.
    critic_recurrent = ccfg["type"] in ("gru", "cnn_gru")
    critic_privileged = not critic_recurrent
    if critic_privileged and not hasattr(env, "critic_obs"):
        raise ValueError(
            f"critic.type='{ccfg['type']}' is feed-forward, which requires the "
            f"env to expose critic_obs(state, dist) and critic_obs_dim; "
            f"'{ecfg['name']}' does not. Use a recurrent critic (gru / cnn_gru) "
            f"or add critic_obs to the env — see JADS/tasks/navigate.py."
        )
    critic_in_dim = env.critic_obs_dim if critic_privileged else env.obs_dim
    critic = make_model(ccfg["type"], ccfg, critic_in_dim, 1)
    if not hasattr(actor, "init_hidden"):
        raise ValueError(
            f"policy.type='{pcfg['type']}' is not recurrent. Recurrent PPO "
            f"expects a policy with init_hidden() (gru / cnn_gru)."
        )

    key = jax.random.PRNGKey(seed)
    key, actor_key, critic_key = jax.random.split(key, 3)
    depth_shape = None
    if has_depth and hasattr(env, "depth_shape"):
        depth_shape = env.depth_shape
    params = {
        "actor":   init_params(actor,  actor_key, env.obs_dim, depth_shape=depth_shape),
        "critic":  init_params(critic, critic_key, critic_in_dim,
                               depth_shape=None if critic_privileged else depth_shape),
        "log_std": jnp.full((env.act_dim,), float(init_log_std)),
    }

    # -- Batched forward passes --------------------------------------------
    def _net_step(model, model_params, obs, hidden, is_depth, is_recurrent):
        """One batched step: (batch, ...) → (out, hidden).

        A feed-forward net ignores and passes through `hidden`, so callers do not
        have to branch.
        """
        if not is_recurrent:
            out = jax.vmap(model.apply, in_axes=(None, 0))({"params": model_params}, obs)
            return out, hidden
        if is_depth:
            depth, vec = obs
            return jax.vmap(model.apply, in_axes=(None, 0, 0, 0))(
                {"params": model_params}, depth, vec, hidden)
        return jax.vmap(model.apply, in_axes=(None, 0, 0))(
            {"params": model_params}, obs, hidden)

    def _net_seq(model, model_params, obs_seq, reset_seq, hidden0, is_depth, is_recurrent):
        """Re-run a net over a (time, batch, ...) observation sequence.

        Recurrent: a sequential scan in which `reset_seq[t]` zeroes the hidden
        state at steps that begin a fresh episode, reproducing exactly the hidden
        states the rollout sampled its actions from (the rollout zeroes the
        hidden at the end of the terminating step, which is the same thing one
        step later).

        Feed-forward: no hidden state to thread, so the whole (time, batch) block
        collapses into a single batched forward — no scan, no sequential
        dependency. This is most of why a feed-forward critic is so much cheaper
        than a recurrent one: the value re-run happens update_epochs x
        num_minibatches times per update, and each one stops being serial.
        """
        if not is_recurrent:
            steps, batch = obs_seq.shape[0], obs_seq.shape[1]
            flat = obs_seq.reshape((steps * batch,) + obs_seq.shape[2:])
            out = jax.vmap(model.apply, in_axes=(None, 0))({"params": model_params}, flat)
            return out.reshape((steps, batch) + out.shape[1:])

        def step(hidden, x):
            obs_t, reset_t = x
            hidden = jnp.where(reset_t[:, None], 0.0, hidden)
            out, hidden = _net_step(model, model_params, obs_t, hidden, is_depth, True)
            return hidden, out
        _, outs = jax.lax.scan(step, hidden0, (obs_seq, reset_seq))
        return outs

    def _actor_step(p, obs, hidden):
        return _net_step(actor, p, obs, hidden, has_depth, True)

    def _critic_step(p, obs, hidden):
        return _net_step(critic, p, obs, hidden, has_depth and critic_recurrent,
                         critic_recurrent)

    # -- Optimiser ----------------------------------------------------------
    opt_steps = updates * update_epochs * num_minibatches
    schedule = (optax.linear_schedule(init_value=lr, end_value=lr_min,
                                      transition_steps=opt_steps)
                if anneal_lr else lr)
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(schedule, eps=1e-5),
    )
    opt_state = optimizer.init(params)

    # -- Vmapped env --------------------------------------------------------
    reset_fn = jax.vmap(env.reset)

    # Per-episode domain randomization: a *_real env returns that episode's
    # airframe in reset()'s info, and it has to ride in the carry so each
    # parallel episode keeps its own draw until it resets — exactly what
    # bptt.py's carry does. Probed once here with a throwaway reset.
    has_env_params = "params" in env.reset(jax.random.PRNGKey(0))[2]
    p_axis = 0 if has_env_params else None   # None keeps the signature uniform

    def _params_kw(p):
        return {"params": p} if has_env_params else {}

    if has_depth:
        step_fn = jax.vmap(
            lambda s, a, p, si, pd: env.step(s, a, morph_matrices=morph_matrices,
                                             step_idx=si, prev_depth=pd,
                                             **_params_kw(p)),
            in_axes=(0, 0, p_axis, None, 0))
    else:
        step_fn = jax.vmap(
            lambda s, a, p: env.step(s, a, morph_matrices=morph_matrices,
                                     **_params_kw(p)),
            in_axes=(0, 0, p_axis))
    reward_fn = jax.vmap(env.step_reward)
    dist_fn   = jax.vmap(lambda s: env.initial_dist(s, morph_matrices))
    cobs_fn   = jax.vmap(env.critic_obs) if critic_privileged else None

    def _critic_input(state, dist, obs):
        """What the critic sees: privileged state, or the actor's observation.

        `dist` is the signed obstacle distance belonging to `state` — the carry's
        `prev_dist` field always holds exactly that, since it is set from the
        step that produced the state (or from initial_dist after a reset).
        """
        return cobs_fn(state, dist) if critic_privileged else obs

    def _critic_hidden0(batch):
        if not critic_recurrent:
            return jnp.zeros((batch, 0))   # keeps the carry pytree shape static
        h = critic.init_hidden()
        return jnp.broadcast_to(h, (batch,) + h.shape)

    dtgt_fn = jax.vmap(env.dist_to_target)

    def _fresh(keys):
        """Sample a batch of fresh episodes: obs, state and the prev_dist seed."""
        obs, states, info = reset_fn(keys)
        fresh = {"state": states, "obs": obs, "dist": dist_fn(states),
                 "d_target": dtgt_fn(states)}
        if has_env_params:
            fresh["params"] = info["params"]
        return fresh

    def _init_carry(keys):
        fresh = _fresh(keys)
        carry = {
            "state":     fresh["state"],
            "obs":       fresh["obs"],
            "prev_dist": fresh["dist"],
            "prev_dtgt": fresh["d_target"],
            "age":       jnp.zeros(batch_size, dtype=jnp.int32),
            "h_a":       jnp.broadcast_to(actor.init_hidden(), (batch_size,) + actor.init_hidden().shape),
            "h_c":       _critic_hidden0(batch_size),
            "last_done": jnp.zeros(batch_size, dtype=bool),
            "slot":      jnp.zeros(batch_size, dtype=jnp.int32),
        }
        if has_env_params:
            carry["params"] = fresh["params"]
        return carry

    # -- Rollout ------------------------------------------------------------
    def _rollout(params, carry, pool, key):
        """Collect `horizon` steps from the persistent carry."""
        def env_step(carry, step_idx):
            obs  = carry["obs"]
            cobs = _critic_input(carry["state"], carry["prev_dist"], obs)
            mean,  h_a = _actor_step(params["actor"],  obs,  carry["h_a"])
            value, h_c = _critic_step(params["critic"], cobs, carry["h_c"])
            value = jnp.squeeze(value, axis=-1)

            rng, sub = jax.random.split(carry["rng"])
            log_std = _clamp_log_std(params["log_std"], min_log_std)
            action = mean + jnp.exp(log_std) * jax.random.normal(sub, mean.shape)
            log_prob = _log_prob(action, mean, log_std)

            # This element's airframe, constant for the episode (None when the
            # env has no per-episode params — then step_fn ignores it).
            ep_params = carry.get("params")
            if has_depth:
                depth, _ = obs
                new_state, new_obs, sd = step_fn(carry["state"], action, ep_params,
                                                 step_idx, depth)
            else:
                new_state, new_obs, sd = step_fn(carry["state"], action, ep_params)
            reward = reward_fn(sd, carry["prev_dist"]) * reward_scale

            # Diagnostics + shaping potential. d_target is what distinguishes
            # "hovering at spawn" from "flying to the target" in the logs.
            d_target = jnp.linalg.norm(sd["pos"] - sd["target_pos"], axis=-1)
            speed    = jnp.linalg.norm(sd["vel"], axis=-1)
            shaping  = progress_weight * (carry["prev_dtgt"] - gamma * d_target)

            age     = carry["age"] + 1
            crashed = sd.get("crashed", jnp.zeros(batch_size, dtype=bool))
            # terminated: the episode genuinely ended → no value bootstrap.
            # truncated:  the episode was cut short  → bootstrap still valid.
            terminated = crashed if reset_on_crash else jnp.zeros(batch_size, dtype=bool)
            truncated = ((age >= max_episode_len) & ~terminated if max_episode_len > 0
                         else jnp.zeros(batch_size, dtype=bool))
            done = terminated | truncated

            # Swap terminated elements onto a pre-sampled fresh episode (a
            # select, not an env.reset — see the module docstring).
            picked   = _pool_take(pool, carry["slot"])
            new_state = _where_batch(done, picked["state"], new_state)
            new_obs   = _where_batch(done, picked["obs"],   new_obs)
            new_carry = {
                "state":     new_state,
                "obs":       new_obs,
                "prev_dist": jnp.where(done, picked["dist"], sd["dist"]),
                "prev_dtgt": jnp.where(done, picked["d_target"], d_target),
                "age":       jnp.where(done, 0, age),
                "h_a":       jnp.where(done[:, None], 0.0, h_a),
                "h_c":       jnp.where(done[:, None], 0.0, h_c),
                "last_done": done,
                "slot":      jnp.minimum(carry["slot"] + done, n_slots - 1),
                "rng":       rng,
            }
            if has_env_params:
                # A new episode means a new airframe, drawn with the pooled reset.
                new_carry["params"] = _where_batch(done, picked["params"], carry["params"])
            transition = {
                "obs":        obs,
                "action":     action,
                "log_prob":   log_prob,
                "value":      value,
                "reward":     reward,
                "shaping":    shaping,
                "d_target":   d_target,
                "speed":      speed,
                "terminated": terminated,
                "truncated":  truncated,
                "done":       done,
                "last_done":  carry["last_done"],
                "crashed":    crashed,
            }
            # Only stored when it differs from `obs` — a recurrent critic reuses
            # the actor's observation, and duplicating a depth image per step
            # would double the rollout's memory for nothing.
            if critic_privileged:
                transition["cobs"] = cobs
            return new_carry, transition

        carry = dict(carry, rng=key)
        carry, traj = jax.lax.scan(env_step, carry, jnp.arange(horizon))
        carry.pop("rng")
        return carry, traj

    # -- Advantages ---------------------------------------------------------
    def _gae(traj, last_value):
        """Truncated GAE over the window.

        `terminated` masks the *bootstrap* (there is no next state), `done` masks
        the advantage recursion (the next step belongs to another episode) — the
        two are distinct and using `done` for both silently discards the value of
        every truncated episode's tail.
        """
        def body(carry, x):
            gae, next_value = carry
            reward, value, terminated, truncated, done = x
            bootstrap = next_value * (1.0 - terminated)
            if crash_bootstrap == "value":
                bootstrap = jnp.where(terminated, value, bootstrap)
            if trunc_bootstrap:
                # The stored next obs belongs to the *reset* episode, so its
                # value is meaningless here; V(s_t) is the cheapest unbiased-ish
                # stand-in (delta collapses to reward + (gamma-1)*V).
                bootstrap = jnp.where(truncated, value, bootstrap)
            delta = reward + gamma * bootstrap - value
            gae = delta + gamma * gae_lambda * (1.0 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            body,
            (jnp.zeros_like(last_value), last_value),
            (traj["reward"], traj["value"], traj["terminated"].astype(jnp.float32),
             traj["truncated"].astype(jnp.float32), traj["done"].astype(jnp.float32)),
            reverse=True, unroll=16,
        )
        return advantages, advantages + traj["value"]

    # -- PPO loss -----------------------------------------------------------
    def _loss_fn(params, batch, hidden0):
        mean = _net_seq(actor, params["actor"], batch["obs"],
                        batch["last_done"], hidden0["a"], has_depth, True)
        value = jnp.squeeze(
            _net_seq(critic, params["critic"],
                     batch["cobs"] if critic_privileged else batch["obs"],
                     batch["last_done"], hidden0.get("c"),
                     has_depth and critic_recurrent, critic_recurrent), axis=-1)
        log_std  = _clamp_log_std(params["log_std"], min_log_std)
        log_prob = _log_prob(batch["action"], mean, log_std)

        # Value loss (clipped, as in the reference implementation)
        target = batch["target"]
        value_losses = jnp.square(value - target)
        if clip_value_loss:
            value_clipped = batch["value"] + jnp.clip(
                value - batch["value"], -clip_eps, clip_eps)
            value_losses = jnp.maximum(value_losses, jnp.square(value_clipped - target))
        value_loss = 0.5 * value_losses.mean()

        # Clipped surrogate objective
        gae = batch["advantage"]
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        ratio = jnp.exp(log_prob - batch["log_prob"])
        actor_loss = -jnp.minimum(
            ratio * gae,
            jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae,
        ).mean()

        if smooth_action > 0.0:
            actor_loss = actor_loss + smooth_action * jnp.mean(
                jnp.square(mean[:-1] - jax.lax.stop_gradient(mean[1:])))

        entropy = _entropy(log_std, env.act_dim)
        total = actor_loss + vf_coef * value_loss - ent_coef * entropy

        log_ratio = log_prob - batch["log_prob"]
        metrics = {
            "policy_loss": actor_loss,
            "value_loss":  value_loss,
            "entropy":     entropy,
            "approx_kl":   jnp.mean(jnp.exp(log_ratio) - 1.0 - log_ratio),
            "clip_frac":   jnp.mean(jnp.abs(ratio - 1.0) > clip_eps),
        }
        return total, metrics

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

    def _update_minibatch(state, minibatch):
        params, opt_state = state
        batch, hidden0 = minibatch
        (loss, metrics), grads = grad_fn(params, batch, hidden0)
        if not learn_std:
            grads = dict(grads, log_std=jnp.zeros_like(grads["log_std"]))
        updates_, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates_)
        metrics = dict(metrics, loss=loss, grad_norm=optax.global_norm(grads))
        return (params, opt_state), metrics

    def _update_epoch(state, key):
        params, opt_state, batch, hidden0 = state
        # Shuffle along the env dimension only — time must stay contiguous for
        # the recurrent re-run.
        perm = jax.random.permutation(key, batch_size)
        batch   = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=1), batch)
        hidden0 = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), hidden0)

        mb_batch = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(
                x.reshape((x.shape[0], num_minibatches, -1) + x.shape[2:]), 0, 1),
            batch)
        mb_hidden = jax.tree_util.tree_map(
            lambda x: x.reshape((num_minibatches, -1) + x.shape[1:]), hidden0)

        (params, opt_state), metrics = jax.lax.scan(
            _update_minibatch, (params, opt_state), (mb_batch, mb_hidden))
        return (params, opt_state, batch, hidden0), metrics

    # -- One full PPO iteration --------------------------------------------
    @jax.jit
    def update(params, opt_state, carry, rms, key):
        key, pool_key, roll_key = jax.random.split(key, 3)

        # Pre-sample the fresh episodes an element may swap onto this window.
        pool = _fresh(jax.random.split(pool_key, batch_size * n_slots))
        pool = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size, n_slots) + x.shape[1:]), pool)

        # The pool is fresh every update, so every element starts from slot 0.
        carry = dict(carry, slot=jnp.zeros(batch_size, dtype=jnp.int32))

        # "c" is absent for a feed-forward critic — there is no hidden state to
        # slice into minibatches, and a zero-width array cannot be reshaped.
        hidden0 = {"a": carry["h_a"]}
        if critic_recurrent:
            hidden0["c"] = carry["h_c"]
        carry, traj = _rollout(params, carry, pool, roll_key)

        # Bootstrap value of the state the window stopped in. Only used where
        # the last step was neither terminated nor truncated, so the fact that
        # the carry may already hold a reset observation is harmless.
        last_value, _ = _critic_step(
            params["critic"],
            _critic_input(carry["state"], carry["prev_dist"], carry["obs"]),
            carry["h_c"])
        last_value = jnp.squeeze(last_value, axis=-1)

        if not reset_on_crash:
            # TBPTT semantics: a crashed element is *not* terminated, it keeps
            # flying so its (growing) collision penalty stays in the objective —
            # and is re-seeded here at the window boundary, exactly where
            # bptt.py's _apply_epoch_resets does it. Done after last_value so
            # the bootstrap still refers to the state the window really ended in.
            crashed_any = jnp.any(traj["crashed"], axis=0)
            picked = _pool_take(pool, carry["slot"])
            carry = dict(
                carry,
                state     = _where_batch(crashed_any, picked["state"], carry["state"]),
                obs       = _where_batch(crashed_any, picked["obs"],   carry["obs"]),
                prev_dist = jnp.where(crashed_any, picked["dist"], carry["prev_dist"]),
                age       = jnp.where(crashed_any, 0, carry["age"]),
                h_a       = jnp.where(crashed_any[:, None], 0.0, carry["h_a"]),
                h_c       = jnp.where(crashed_any[:, None], 0.0, carry["h_c"]),
                last_done = carry["last_done"] | crashed_any,
                slot      = jnp.minimum(carry["slot"] + crashed_any, n_slots - 1),
            )
            if has_env_params:
                carry["params"] = _where_batch(
                    crashed_any, picked["params"], carry["params"])

        # raw_reward is the untouched task objective — logged as mean_return so
        # it stays comparable to BPTT. Training uses the shaped reward, which is
        # potential-based and therefore has the same optimal policy.
        raw_reward = traj["reward"]
        train_reward = raw_reward + traj["shaping"] if progress_weight != 0.0 else raw_reward
        if normalize_rew:
            # VecNormalize-style: divide by the running std of the discounted
            # return, so the critic sees O(1) targets no matter how the task's
            # loss weights are scaled. Normalises the *training* reward — the
            # shaping term has to survive into the advantages to do anything.
            def ret_step(ret, x):
                reward, done = x
                ret = ret * gamma + reward
                return jnp.where(done, 0.0, ret), ret
            new_ret, returns = jax.lax.scan(
                ret_step, rms["ret"], (train_reward, traj["done"]))
            stats = _rms_update(rms["stats"], returns.reshape(-1))
            rms = {"ret": new_ret, "stats": stats}
            train_reward = jnp.clip(train_reward / jnp.sqrt(stats["var"] + 1e-8),
                                    -reward_clip, reward_clip)
        traj = dict(traj, reward=train_reward)

        advantages, targets = _gae(traj, last_value)
        batch = {
            "obs":       traj["obs"],
            "action":    traj["action"],
            "log_prob":  traj["log_prob"],
            "value":     traj["value"],
            "last_done": traj["last_done"],
            "advantage": advantages,
            "target":    targets,
        }
        if critic_privileged:
            batch["cobs"] = traj["cobs"]

        key, *epoch_keys = jax.random.split(key, update_epochs + 1)
        (params, opt_state, _, _), metrics = jax.lax.scan(
            _update_epoch, (params, opt_state, batch, hidden0),
            jnp.stack(epoch_keys))

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        target_var = jnp.var(targets)
        metrics.update({
            "mean_return": jnp.mean(raw_reward) / reward_scale,
            "crash_frac":  jnp.mean(traj["crashed"]),
            "reset_frac":  jnp.mean(traj["done"]),
            "std":         jnp.mean(jnp.exp(_clamp_log_std(params["log_std"], min_log_std))),
            "explained_var": 1.0 - jnp.var(targets - traj["value"]) / (target_var + 1e-8),
            # What the policy actually does: mean distance to target, closest
            # approach within the window, and mean speed. A policy hovering at
            # spawn and one flying to the target report near-identical
            # mean_return early on, but these separate them immediately.
            "d_target":     jnp.mean(traj["d_target"]),
            "min_d_target": jnp.mean(jnp.min(traj["d_target"], axis=0)),
            "speed":        jnp.mean(traj["speed"]),
        })
        return params, opt_state, carry, rms, key, metrics

    # -- Logger -------------------------------------------------------------
    log_cfg   = config.get("logging", {})
    csv_path  = log_cfg.get("csv_path", "logs/training.csv")
    ckpt_path = log_cfg.get("checkpoint_path", "checkpoints/policy.pkl")
    fields = ["epoch", "mean_return", "d_target", "min_d_target", "speed",
              "loss", "grad_norm", "env_steps", "elapsed",
              "sps", "crash_frac", "reset_frac", "policy_loss", "value_loss",
              "entropy", "approx_kl", "clip_frac", "explained_var", "std"]
    logger = Logger(csv_path, fields)

    # -- Info ---------------------------------------------------------------
    n_actor  = sum(x.size for x in jax.tree_util.tree_leaves(params["actor"]))
    n_critic = sum(x.size for x in jax.tree_util.tree_leaves(params["critic"]))
    steps_per_update = horizon * batch_size
    print(f"Algo         : Recurrent PPO")
    print(f"Devices      : {jax.devices()}")
    print(f"Env          : {ecfg['name']}  |  obs_dim={env.obs_dim}  act_dim={env.act_dim}")
    print(f"Morphology   : fixed ({'checkpoint' if morph_ckpt else 'env default'})")
    print(f"Policy       : {pcfg['type']}  |  actor={n_actor:,}  critic={n_critic:,}")
    print(f"Critic       : {ccfg['type']}  |  "
          f"{'privileged state, feed-forward' if critic_privileged else 'recurrent, actor obs'}"
          f"  |  in_dim={critic_in_dim}")
    print(f"Horizon      : {horizon}  |  batch={batch_size}  updates={updates}"
          f"  ({steps_per_update:,} env steps/update, {steps_per_update * updates:,} total)")
    print(f"PPO          : epochs={update_epochs}  minibatches={num_minibatches}"
          f"  clip={clip_eps}  gamma={gamma}  lam={gae_lambda}"
          f"  vf={vf_coef}  ent={ent_coef}")
    _cap = max_episode_len if max_episode_len > 0 else "unbounded"
    print(f"Episodes     : reset_on_crash={reset_on_crash}  |  max_episode_len={_cap}"
          f"  |  reset_pool={n_slots}")
    print(f"Optim        : Adam lr={lr}"
          f"{f'→{lr_min}' if anneal_lr else ''}  grad_clip={grad_clip}"
          f"  |  reward_norm={normalize_rew}  init_std={math.exp(init_log_std):.3f}")
    print("-" * 60)

    if tcfg.get("debug_nans", False):
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_disable_jit", True)
        print("WARNING: jax_debug_nans + jax_disable_jit enabled — training will be very slow")

    # -- Persistent XLA compilation cache ----------------------------------
    _cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".jax_cache")
    os.makedirs(os.path.abspath(_cache_dir), exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", os.path.abspath(_cache_dir))

    # -- Main loop ----------------------------------------------------------
    print("(JIT compiles on update 0 — cached to .jax_cache/ for future runs)")
    t_start = time.time()

    key, ck = jax.random.split(key)
    carry = _init_carry(jax.random.split(ck, batch_size))
    rms = {"ret": jnp.zeros(batch_size), "stats": _rms_init()}

    t_mark, steps_mark = t_start, 0
    for epoch in range(updates):
        params, opt_state, carry, rms, key, metrics = update(
            params, opt_state, carry, rms, key)

        if epoch % log_every == 0 or epoch == updates - 1:
            elapsed = time.time() - t_start
            env_steps = (epoch + 1) * steps_per_update
            # `elapsed` is cumulative and includes the epoch-0 JIT compile;
            # `sps` is the throughput over the last logging interval only.
            now = time.time()
            sps = (env_steps - steps_mark) / max(now - t_mark, 1e-6)
            t_mark, steps_mark = now, env_steps
            logger.log({
                "epoch":         epoch,
                "mean_return":   float(metrics["mean_return"]),
                "d_target":      float(metrics["d_target"]),
                "min_d_target":  float(metrics["min_d_target"]),
                "speed":         float(metrics["speed"]),
                "loss":          float(metrics["loss"]),
                "grad_norm":     float(metrics["grad_norm"]),
                "env_steps":     env_steps,
                "elapsed":       round(elapsed, 1),
                "sps":           round(sps),
                "crash_frac":    float(metrics["crash_frac"]),
                "reset_frac":    float(metrics["reset_frac"]),
                "policy_loss":   float(metrics["policy_loss"]),
                "value_loss":    float(metrics["value_loss"]),
                "entropy":       float(metrics["entropy"]),
                "approx_kl":     float(metrics["approx_kl"]),
                "clip_frac":     float(metrics["clip_frac"]),
                "explained_var": float(metrics["explained_var"]),
                "std":           float(metrics["std"]),
            })

    # Saved in the BPTT checkpoint format — actor params only, so
    # eval_success_rate.py / rerun_rollout.py load a PPO policy unchanged
    # (they run the deterministic mean action, which is what the actor emits).
    save_checkpoint(ckpt_path, params["actor"], morph_params)

    print("-" * 60)
    print(f"Training complete in {time.time() - t_start:.1f}s. Log saved to {csv_path}")
    return params["actor"]
