"""
Compute success rate of a trained Navigate policy over N episodes.

An episode is SUCCESS if the drone reaches the target (within --success-radius m)
before hitting any obstacle (nearest obstacle dist <= 0).
An episode is FAILURE if it collides or exhausts --max-steps without success.

Usage:
  python eval_success_rate.py \
      --config configs/navigate_morph.yaml \
      --checkpoint checkpoints/navigate.pkl \
      [--episodes 1000] [--batch-size 100] [--max-steps 250] \
      [--success-radius 0.1] [--seed 0]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "depth_render"))

import jax
import jax.numpy as jnp
import yaml

from envs import make_env
from models import make_model
from utils.checkpoint import load as load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",         required=True)
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--episodes",       type=int,   default=1000)
    p.add_argument("--batch-size",     type=int,   default=100)
    p.add_argument("--max-steps",      type=int,   default=250)
    p.add_argument("--success-radius", type=float, default=0.1,
                   help="Distance threshold (m) for reaching the target")
    p.add_argument("--seed",           type=int,   default=0)
    return p.parse_args()


def build_eval_fn(env, policy, max_steps, success_radius):
    """
    Returns a JIT-compiled, vmapped function:
        eval_fn(policy_params, morph_params, init_states) -> bool[batch]
    where True means the episode was a success.
    """
    has_depth  = hasattr(policy, "conv_features")
    has_hidden = hasattr(policy, "init_hidden")

    def single_rollout(policy_params, morph_params, init_state):
        init_obs = env._get_obs(init_state)

        if has_depth and has_hidden:
            depth0, obs_vec0 = init_obs
            init_carry = (init_state, depth0, obs_vec0, policy.init_hidden(),
                          jnp.bool_(False), jnp.bool_(False))
        elif has_hidden:
            init_carry = (init_state, init_obs, policy.init_hidden(),
                          jnp.bool_(False), jnp.bool_(False))
        elif has_depth:
            depth0, obs_vec0 = init_obs
            init_carry = (init_state, depth0, obs_vec0,
                          jnp.bool_(False), jnp.bool_(False))
        else:
            init_carry = (init_state, init_obs,
                          jnp.bool_(False), jnp.bool_(False))

        def step(carry, _):
            # Unpack carry
            if has_depth and has_hidden:
                state, depth_img, obs_vec, hidden, failed, success = carry
                action, new_hidden = policy.apply(
                    {"params": policy_params}, depth_img, obs_vec, hidden
                )
            elif has_hidden:
                state, obs_vec, hidden, failed, success = carry
                action, new_hidden = policy.apply(
                    {"params": policy_params}, obs_vec, hidden
                )
            elif has_depth:
                state, depth_img, obs_vec, failed, success = carry
                action = policy.apply(
                    {"params": policy_params}, depth_img, obs_vec
                )
            else:
                state, obs_vec, failed, success = carry
                action = policy.apply({"params": policy_params}, obs_vec)

            new_state, step_data = env.step(state, action, morph_params)
            new_obs = env._get_obs(new_state)

            # Termination: skip update if already done
            done = failed | success

            dist   = step_data["dist"]
            pos    = step_data["pos"]
            target = step_data["target_pos"]

            collided = ~done & (dist <= 0.0)
            reached  = ~done & ~collided & (
                jnp.linalg.norm(pos - target) < success_radius
            )

            new_failed  = failed  | collided
            new_success = success | reached

            # Repack carry
            if has_depth and has_hidden:
                new_depth, new_obs_vec = new_obs
                new_carry = (new_state, new_depth, new_obs_vec, new_hidden,
                             new_failed, new_success)
            elif has_hidden:
                new_carry = (new_state, new_obs, new_hidden,
                             new_failed, new_success)
            elif has_depth:
                new_depth, new_obs_vec = new_obs
                new_carry = (new_state, new_depth, new_obs_vec,
                             new_failed, new_success)
            else:
                new_carry = (new_state, new_obs, new_failed, new_success)

            speed  = jnp.linalg.norm(step_data["vel"])
            # active = this step is before termination (done was False at step start)
            active = ~done
            return new_carry, (speed, active)

        final_carry, (speeds, active_mask) = jax.lax.scan(
            step, init_carry, None, length=max_steps
        )
        success_flag = final_carry[-1]
        max_speed    = jnp.max(speeds)
        active_float = active_mask.astype(jnp.float32)
        avg_speed    = jnp.sum(speeds * active_float) / jnp.maximum(jnp.sum(active_float), 1.0)
        return success_flag, max_speed, avg_speed

    return jax.jit(jax.vmap(single_rollout, in_axes=(None, None, 0)))


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ecfg = config["env"]
    dcfg = config.get("depth_camera", {})
    env  = make_env(ecfg["name"], depth_camera=dcfg,
                    **{k: v for k, v in ecfg.items() if k != "name"})

    pcfg   = config["policy"]
    policy = make_model(pcfg["type"], pcfg, obs_dim=env.obs_dim, act_dim=env.act_dim)

    policy_params, morph_params = load_checkpoint(args.checkpoint)

    reset_fn = jax.jit(jax.vmap(env.reset))
    eval_fn  = build_eval_fn(env, policy, args.max_steps, args.success_radius)

    key       = jax.random.PRNGKey(args.seed)
    n_ep      = args.episodes
    batch     = args.batch_size
    n_batches = (n_ep + batch - 1) // batch

    print(f"Policy       : {pcfg['type']}")
    print(f"Episodes     : {n_ep}  |  batch={batch}  max_steps={args.max_steps}")
    print(f"Success def  : dist_to_target < {args.success_radius} m, no collision")
    print("-" * 50)

    total_success    = 0
    total_ran        = 0
    total_max_speed  = 0.0
    total_avg_speed  = 0.0
    all_max_speeds   = []
    all_avg_speeds   = []

    for b in range(n_batches):
        this_batch = min(batch, n_ep - b * batch)
        key, *reset_keys = jax.random.split(key, this_batch + 1)
        reset_keys = jnp.stack(reset_keys)

        _, init_states, _ = reset_fn(reset_keys)
        successes, max_speeds, avg_speeds = eval_fn(policy_params, morph_params, init_states)

        max_speeds_b = max_speeds[:this_batch]
        avg_speeds_b = avg_speeds[:this_batch]

        n_suc = int(jnp.sum(successes))
        total_success   += n_suc
        total_ran       += this_batch
        total_max_speed += float(jnp.sum(max_speeds_b))
        total_avg_speed += float(jnp.sum(avg_speeds_b))
        all_max_speeds.append(max_speeds_b)
        all_avg_speeds.append(avg_speeds_b)

        rate      = total_success / total_ran
        avg_max_v = total_max_speed / total_ran
        avg_v     = total_avg_speed / total_ran
        print(f"  Batch {b+1:3d}/{n_batches}: {n_suc:3d}/{this_batch}"
              f"  |  running {total_success}/{total_ran} = {rate:.1%}"
              f"  |  avg speed = {avg_v:.2f} m/s"
              f"  |  avg max speed = {avg_max_v:.2f} m/s")

    all_max_speeds = jnp.concatenate(all_max_speeds)
    all_avg_speeds = jnp.concatenate(all_avg_speeds)
    std_max_speed  = float(jnp.std(all_max_speeds))
    std_avg_speed  = float(jnp.std(all_avg_speeds))

    print("-" * 50)
    print(f"Success rate     : {total_success}/{total_ran} = {total_success/total_ran:.2%}")
    print(f"Avg speed        : {total_avg_speed/total_ran:.2f} m/s  ± {std_avg_speed:.2f}  (active steps only)")
    print(f"Avg max speed    : {total_max_speed/total_ran:.2f} m/s  ± {std_max_speed:.2f}")


if __name__ == "__main__":
    main()
