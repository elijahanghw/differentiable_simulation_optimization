"""
Proximal Policy Optimisation (PPO).

Follows the SB3 implementation exactly:
  - Log probs computed on raw (pre-clip) Gaussian samples; actions are
    clipped separately before being passed to the environment.
  - GAE uses (1 - done) masks to zero out bootstrap and lambda terms at
    episode boundaries.
  - Per-minibatch advantage normalisation.
  - No value clipping (clip_range_vf=None, the SB3 default).
  - Auto-reset: when an episode ends the environment is immediately reset
    and the horizon continues into the new episode (gymnax / ppo_alt style).

Algorithm
---------
For each epoch:
  1. Sample num_env initial states and collect horizon-step rollouts.
     Actions are drawn from N(mean, std) where mean = actor(obs) and
     log_std is a learnable parameter.  Episodes auto-reset mid-horizon.
  2. Compute GAE advantages and value-function targets with done masking.
  3. For ppo_epochs passes over the data (shuffled, n_minibatches minibatches):
       a. ratio  = exp(new_log_prob - old_log_prob)
       b. L_clip = -E[min(ratio * A, clip(ratio, 1±ε) * A)]
       c. L_vf   = E[(V(s) - R)²]
       d. L_ent  = -E[entropy(π)]
       e. total  = L_clip + vf_coef * L_vf - ent_coef * L_ent
       f. Update actor + log_std + critic jointly with Adam.
"""

from collections import deque
from typing import Any, Dict

import numpy as np
import jax
import jax.numpy as jnp
import optax

from envs import make_env
from models import make_model, make_critic
from utils.logger import Logger


# ---------------------------------------------------------------------------
# Gaussian helpers
# ---------------------------------------------------------------------------

def _gaussian_log_prob(action: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
    """Sum of diagonal Gaussian log-probs over action dimensions → scalar."""
    return jnp.sum(
        -0.5 * ((action - mean) / jnp.exp(log_std)) ** 2
        - log_std
        - 0.5 * jnp.log(2 * jnp.pi),
        axis=-1,
    )


def _gaussian_entropy(log_std: jnp.ndarray) -> jnp.ndarray:
    """Entropy of a diagonal Gaussian (sum over action dims)."""
    return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)


# ---------------------------------------------------------------------------
# Episodic return helper
# ---------------------------------------------------------------------------

def _collect_episode_returns(rewards, dones) -> list:
    """
    Extract the undiscounted return of every complete episode from a rollout,
    matching SB3's VecMonitor episode tracking.

    Args:
        rewards: (horizon, batch) JAX array or numpy array
        dones:   (horizon, batch) — 1.0 when an episode ended at that step

    Returns:
        List of scalar floats, one per complete episode across all envs.
    """
    rewards_np = np.asarray(rewards)          # (H, B)
    dones_np   = np.asarray(dones)            # (H, B)
    H, B       = rewards_np.shape

    # Build episode index per step: increments the step after each done
    shifted    = np.concatenate([np.zeros((1, B)), dones_np[:-1]], axis=0)
    ep_idx     = np.cumsum(shifted, axis=0).astype(int)   # (H, B)

    # Number of complete episodes per env (include a done on the last step)
    n_complete = ep_idx[-1] + dones_np[-1].astype(int)    # (B,)

    # Scatter-add rewards into (max_eps, B) buckets
    max_eps    = H + 1
    ep_returns = np.zeros((max_eps, B))
    b_idx      = np.broadcast_to(np.arange(B), (H, B))
    np.add.at(ep_returns, (ep_idx, b_idx), rewards_np)

    results = []
    for b in range(B):
        for e in range(int(n_complete[b])):
            results.append(float(ep_returns[e, b]))
    return results


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def _build_collect_fn(env, actor, critic, horizon: int):
    """
    Returns a JIT-compiled function:
        collect_fn(actor_params, log_std, critic_params, init_states, init_obs, keys)
            -> (obs, actions, rewards, values, log_probs, dones), final_values

    All trajectory arrays have shape (horizon, batch, ...).
    final_values has shape (batch,).

    Auto-reset: when an episode ends (terminated or truncated) the environment
    is immediately reset and the new obs/state is used as the carry for the
    next step, matching gymnax / ppo_alt behaviour.
    """
    def collect_single(actor_params, log_std, critic_params, init_state, init_obs, key):
        def step_fn(carry, _):
            state, obs, key = carry
            key, action_key, reset_key = jax.random.split(key, 3)
            mean       = actor.apply(actor_params, obs)
            noise      = jax.random.normal(action_key, mean.shape)
            raw_action = mean + jnp.exp(log_std) * noise
            # Log prob on the raw sample — consistent with SB3 DiagGaussianDistribution
            log_prob   = _gaussian_log_prob(raw_action, mean, log_std)
            value      = critic.apply(critic_params, obs)[0]
            # Clip only for the environment; buffer stores raw_action
            action_env = jnp.clip(raw_action, -1.0, 1.0)
            next_state, next_obs, reward, terminated, truncated, info = env.step(state, action_env)
            done = jnp.logical_or(terminated, truncated)
            # Auto-reset: if done (terminated OR truncated), replace carry with a fresh episode
            reset_obs, reset_state, _ = env.reset(reset_key)
            carry_state = jax.tree_util.tree_map(
                lambda r, n: jnp.where(done, r, n), reset_state, next_state
            )
            carry_obs = jnp.where(done, reset_obs, next_obs)
            # Store terminated (not done) for GAE masking: truncation should NOT zero out
            # the value bootstrap; only true termination (crash) should.
            # done is stored separately for episode-return tracking.
            return (carry_state, carry_obs, key), (obs, raw_action, reward, value, log_prob, done, terminated)

        (final_state, final_obs, _), traj = jax.lax.scan(
            step_fn, (init_state, init_obs, key), None, length=horizon
        )
        final_value = critic.apply(critic_params, final_obs)[0]
        return traj, final_value, final_state, final_obs

    def collect_batch(actor_params, log_std, critic_params, init_states, init_obs, keys):
        # vmap gives (batch, horizon, ...) → transpose to (horizon, batch, ...)
        trajs, final_values, final_states, final_obs_batch = jax.vmap(
            collect_single, in_axes=(None, None, None, 0, 0, 0)
        )(actor_params, log_std, critic_params, init_states, init_obs, keys)
        obs, actions, rewards, values, log_probs, dones, terminateds = trajs
        return (
            jnp.swapaxes(obs,         0, 1),
            jnp.swapaxes(actions,     0, 1),
            jnp.swapaxes(rewards,     0, 1),
            jnp.swapaxes(values,      0, 1),
            jnp.swapaxes(log_probs,   0, 1),
            jnp.swapaxes(dones,       0, 1),
            jnp.swapaxes(terminateds, 0, 1),
        ), final_values, final_states, final_obs_batch

    return jax.jit(collect_batch)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def _compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    final_values: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple:
    """
    Generalised Advantage Estimation via a reverse scan (SB3-consistent).

    done[t] = 1 if the episode terminated at step t.
    The (1 - done[t]) mask zeros out the bootstrap value and the propagated
    advantage at episode boundaries, matching SB3's next_non_terminal logic.

    Args:
        rewards:      (horizon, batch)
        values:       (horizon, batch)
        dones:        (horizon, batch)  — terminated flags from env.step
        final_values: (batch,)          — V(s_T), bootstrap from last state

    Returns:
        advantages: (horizon, batch)
        returns:    (horizon, batch)
    """
    def gae_step(carry, inp):
        gae_next, value_next = carry
        reward, value_t, done = inp
        next_non_terminal = 1.0 - done
        delta = reward + gamma * value_next * next_non_terminal - value_t
        gae_t = delta + gamma * gae_lambda * next_non_terminal * gae_next
        return (gae_t, value_t), (gae_t, gae_t + value_t)

    init = (jnp.zeros_like(final_values), final_values)
    _, (advantages, returns) = jax.lax.scan(
        gae_step, init, (rewards, values, dones), reverse=True
    )
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------

def _build_ppo_loss_fn(actor, critic, clip_eps: float, vf_coef: float, ent_coef: float):
    """
    Returns the PPO loss function:
        loss_fn(params, batch) -> (total_loss, (policy_loss, value_loss, entropy))

    params = {"actor": ..., "log_std": ..., "critic": ...}
    batch  = (obs, actions, old_log_probs, advantages, returns)
    """
    def loss_fn(params, batch):
        actor_params  = params["actor"]
        log_std       = params["log_std"]
        critic_params = params["critic"]
        obs, actions, old_log_probs, advantages, returns = batch

        # Normalise advantages within the mini-batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy
        new_means     = actor.apply(actor_params, obs)
        new_log_probs = _gaussian_log_prob(actions, new_means, log_std)
        ratio         = jnp.exp(new_log_probs - old_log_probs)
        clipped       = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        policy_loss   = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))

        # Value function
        values     = critic.apply(critic_params, obs)[:, 0]
        value_loss = jnp.mean((values - returns) ** 2)

        # Entropy bonus
        entropy    = jnp.mean(_gaussian_entropy(log_std))

        total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (policy_loss, value_loss, entropy)

    return loss_fn


# ---------------------------------------------------------------------------
# Fully JIT-compiled update (nested lax.scan over epochs × minibatches)
# ---------------------------------------------------------------------------

def _build_update_fn(actor, critic, optimizer, clip_eps, vf_coef, ent_coef,
                     ppo_epochs: int, n_minibatches: int):
    """
    Returns a JIT-compiled function that runs all PPO update epochs:

        update_fn(params, opt_state, key, flat_data)
            -> (params, opt_state, key, (policy_loss, value_loss, entropy, grad_norm))

    flat_data = (obs, actions, log_probs, advantages, returns), each (N, ...)
    Loss arrays each have shape (ppo_epochs, n_minibatches).

    Structure mirrors ppo_alt: inner lax.scan over minibatches, outer
    lax.scan over epochs — the entire update is one compiled XLA program.
    """
    loss_fn = _build_ppo_loss_fn(actor, critic, clip_eps, vf_coef, ent_coef)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def _update_minibatch(carry, minibatch):
        params, opt_state = carry
        (_, (pl, vl, ent)), grads = grad_fn(params, minibatch)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (pl, vl, ent, grad_norm)

    def run_updates(params, opt_state, key, flat_data):
        N             = flat_data[0].shape[0]
        minibatch_size = N // n_minibatches

        def _update_epoch(carry, _):
            params, opt_state, key = carry
            key, perm_key = jax.random.split(key)
            perm        = jax.random.permutation(perm_key, N)
            shuffled    = jax.tree_util.tree_map(lambda x: x[perm], flat_data)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((n_minibatches, minibatch_size) + x.shape[1:]),
                shuffled,
            )
            (params, opt_state), aux = jax.lax.scan(
                _update_minibatch, (params, opt_state), minibatches
            )
            return (params, opt_state, key), aux

        (params, opt_state, key), all_aux = jax.lax.scan(
            _update_epoch, (params, opt_state, key), None, length=ppo_epochs
        )
        return params, opt_state, key, all_aux

    return jax.jit(run_updates)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: Dict[str, Any]) -> Any:
    tcfg          = config["training"]
    seed          = tcfg["seed"]
    epochs        = tcfg["epochs"]
    horizon       = tcfg["horizon"]
    num_env       = tcfg["num_env"]
    lr            = tcfg["lr"]
    grad_clip     = tcfg.get("grad_clip", 0.5)
    gamma         = tcfg.get("gamma", 0.99)
    gae_lambda    = tcfg.get("gae_lambda", 0.95)
    ppo_epochs    = tcfg.get("ppo_epochs", 10)
    n_minibatches = tcfg.get("n_minibatches", 4)
    clip_eps      = tcfg.get("clip_eps", 0.2)
    vf_coef       = tcfg.get("vf_coef", 0.5)
    ent_coef      = tcfg.get("ent_coef", 0.01)
    log_std_init  = tcfg.get("log_std_init", -0.5)
    log_every     = tcfg.get("log_interval", 10)

    # -- Environment --------------------------------------------------------
    ecfg = config["env"]
    env  = make_env(ecfg["name"], **{k: v for k, v in ecfg.items() if k != "name"})

    # -- Actor --------------------------------------------------------------
    pcfg        = config["policy"]
    actor_sizes   = [env.obs_dim] + pcfg["hidden_sizes"] + [env.act_dim]
    squash_output = pcfg.get("squash_output", True)
    # output_scale=0.01 matches SB3's MlpPolicy orthogonal init with gain=0.01,
    # giving near-zero initial action means for training stability.
    actor         = make_model(pcfg["type"], layer_sizes=actor_sizes,
                               squash_output=squash_output, output_scale=0.01)

    # -- Critic -------------------------------------------------------------
    critic_hidden = pcfg.get("critic_hidden_sizes", pcfg["hidden_sizes"])
    critic_sizes  = [env.obs_dim] + critic_hidden + [1]
    critic        = make_critic(critic_sizes)

    key = jax.random.PRNGKey(seed)
    key, actor_key, critic_key = jax.random.split(key, 3)

    actor_params  = actor.init(actor_key)
    critic_params = critic.init(critic_key)
    log_std       = jnp.full((env.act_dim,), log_std_init)

    params    = {"actor": actor_params, "log_std": log_std, "critic": critic_params}
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr, eps=1e-5))
    opt_state = optimizer.init(params)

    # -- Build JIT-compiled functions ---------------------------------------
    collect_fn = _build_collect_fn(env, actor, critic, horizon)
    update_fn  = _build_update_fn(
        actor, critic, optimizer, clip_eps, vf_coef, ent_coef, ppo_epochs, n_minibatches
    )

    # -- Logger -------------------------------------------------------------
    log_cfg  = config.get("logging", {})
    csv_path = log_cfg.get("csv_path", "logs/training.csv")
    logger   = Logger(csv_path, ["epoch", "mean_return", "policy_loss", "value_loss", "entropy", "grad_norm"])

    # -- Info ---------------------------------------------------------------
    n_actor_params  = sum(w.size + b.size for w, b in actor_params)
    n_critic_params = sum(w.size + b.size for w, b in critic_params)

    print(f"Algo         : PPO")
    print(f"Devices      : {jax.devices()}")
    print(f"Env          : {ecfg['name']}  |  obs_dim={env.obs_dim}  act_dim={env.act_dim}")
    print(f"Actor        : {pcfg['type']}  |  layers={actor_sizes}  params={n_actor_params:,}")
    print(f"Critic       : mlp  |  layers={critic_sizes}  params={n_critic_params:,}")
    print(f"log_std init : {log_std_init}  →  std ≈ {float(jnp.exp(log_std_init)):.3f}")
    print(f"Horizon      : {horizon}  |  batch={num_env}  epochs={epochs}")
    print(f"PPO          : update_epochs={ppo_epochs}  n_minibatches={n_minibatches}  clip={clip_eps}")
    print(f"Optim        : Adam lr={lr}  grad_clip={grad_clip}  gamma={gamma}  λ={gae_lambda}")
    print("-" * 60)

    # -- Episodic return buffer (matches SB3's ep_info_buffer) -------------
    ep_return_buffer = deque(maxlen=100)

    # -- Initial env states (persisted across epochs, like SB3) ------------
    key, init_key = jax.random.split(key)
    current_obs, current_states, _ = jax.vmap(env.reset)(
        jax.random.split(init_key, num_env)
    )

    # -- Main loop ----------------------------------------------------------
    for epoch in range(epochs):
        key, collect_key = jax.random.split(key)
        collect_keys = jax.random.split(collect_key, num_env)

        (obs, actions, rewards, values, log_probs, dones, terminateds), final_values, \
            current_states, current_obs = collect_fn(
            params["actor"], params["log_std"], params["critic"],
            current_states, current_obs, collect_keys,
        )

        # Use terminated (not done) for GAE: truncation should bootstrap, not zero out value
        advantages, returns = _compute_gae(rewards, values, terminateds, final_values, gamma, gae_lambda)
        # Flatten (horizon, batch, ...) → (horizon*batch, ...)
        N         = horizon * num_env
        flat_data = (
            obs.reshape(N, env.obs_dim),
            actions.reshape(N, env.act_dim),
            log_probs.reshape(N),
            advantages.reshape(N),
            returns.reshape(N),
        )

        # All PPO epochs × minibatches in one compiled call
        key, update_key = jax.random.split(key)
        params, opt_state, _, (pl, vl, ent, gn) = update_fn(
            params, opt_state, update_key, flat_data
        )

        ep_return_buffer.extend(_collect_episode_returns(rewards, dones))

        if epoch % log_every == 0:
            if ep_return_buffer:
                mean_return = float(np.mean(ep_return_buffer))
            else:
                mean_return = float(jnp.mean(jnp.sum(rewards, axis=0)))
            logger.log({
                "epoch":       epoch,
                "mean_return": mean_return,
                "policy_loss": float(jnp.mean(pl)),
                "value_loss":  float(jnp.mean(vl)),
                "entropy":     float(jnp.mean(ent)),
                "grad_norm":   float(jnp.mean(gn)),
            })

    print("-" * 60)
    print(f"Training complete. Log saved to {csv_path}")
    return params
