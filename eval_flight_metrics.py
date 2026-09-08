"""
Flight metrics for a trained checkpoint: max speed, mean speed, time to
target, crash rate — evaluated over held-out episodes at the deployed sensor
model (mm quantization on, no gradients).

Usage:
  python eval_flight_metrics.py --config configs/train/navigate_real_tof_pitch0.yaml \
      --checkpoint checkpoints/navigate_real_tof_pitch0.pkl [--episodes 128] [--steps 1000]
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from JADS.tasks import make_env
from JADS.models import make_model
from JADS.utils.checkpoint import load as load_checkpoint
from JADS.utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--steps",    type=int, default=1000, help="matches max_episode_len [1000]")
    p.add_argument("--success-radius", type=float, default=1.0, help="metres [1.0]")
    p.add_argument("--seed", type=int, default=999_999, help="held-out eval seed base")
    return p.parse_args()


def rollout(env, policy, policy_params, keys, n_steps):
    """Batched rollout, returning the full (B, T, ...) trajectory."""
    obs0, state0, info0 = jax.vmap(env.reset)(keys)
    hidden0 = jnp.broadcast_to(policy.init_hidden(), (keys.shape[0],) + policy.init_hidden().shape)

    def single(state0, obs0, hidden0, params0):
        def step(carry, step_idx):
            state, obs, hidden = carry
            depth_img, obs_vec = obs
            action, new_hidden = policy.apply({"params": policy_params}, depth_img, obs_vec, hidden)
            new_state, new_obs, step_data = env.step(
                state, action, params=params0, step_idx=step_idx, prev_depth=depth_img,
            )
            return (new_state, new_obs, new_hidden), step_data
        _, traj = jax.lax.scan(step, (state0, obs0, hidden0), jnp.arange(n_steps))
        return traj

    return jax.vmap(single)(state0, obs0, hidden0, info0["params"])


def summarize(traj, success_radius, dt):
    dist_to_target = np.asarray(jnp.linalg.norm(traj["pos"] - traj["target_pos"], axis=-1))  # (B,T)
    crashed = np.asarray(traj["crashed"])                                                     # (B,T)
    speed   = np.asarray(jnp.linalg.norm(traj["vel"][..., :2], axis=-1))                       # (B,T)

    crashed_by_now = np.maximum.accumulate(crashed, axis=1)
    reached = (dist_to_target < success_radius) & ~crashed_by_now

    tta = np.full(traj["pos"].shape[0], np.nan)
    for i in range(len(tta)):
        hit = np.where(reached[i])[0]
        if len(hit):
            tta[i] = hit[0] * dt

    return {
        "max_speed":  float(speed.max()),
        "avg_speed":  float(speed.mean()),
        "time_to_target_s": float(np.nanmean(tta)) if not np.all(np.isnan(tta)) else float("nan"),
        "crash_frac": float(crashed.any(axis=1).mean()),
        "success_frac": float(reached.any(axis=1).mean()),
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    ecfg = config["env"]
    env = make_env(ecfg["name"], depth_camera=config["depth_camera"],
                   **{k: v for k, v in ecfg.items() if k != "name"})

    pcfg = config["policy"]
    policy = make_model(pcfg["type"], pcfg, env.obs_dim, env.act_dim)
    policy_params, _ = load_checkpoint(args.checkpoint)

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    eval_fn = jax.jit(lambda pp, k: rollout(env, policy, pp, k, args.steps))
    traj = eval_fn(policy_params, keys)
    stats = summarize(traj, args.success_radius, env.dt)

    print(f"pitch_deg={getattr(env, 'cam_pitch_deg', 0.0):+.1f}  "
          f"({args.episodes} episodes x {args.steps} steps, success radius {args.success_radius} m)")
    for k, v in stats.items():
        print(f"  {k:16s} {v:.3f}")
    return stats


if __name__ == "__main__":
    main()
