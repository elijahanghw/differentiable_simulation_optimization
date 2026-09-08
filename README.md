# JADS: Just Another Differentiable/Drone Simulator

A JAX quadcopter/multicopter simulator that is differentiable end to end — dynamics,
depth rendering and loss — so a vision-based flight policy can be trained by
backpropagating *through* the simulator rather than by sampling returns from it.

Everything is `jit`/`vmap`-able: a batch of independent episodes (each with its own
obstacle field, target, and randomized airframe) rolls out as one traced program, and
`jax.value_and_grad` differentiates the whole rollout back to the policy weights.

Two things sit on top of that core:

* **Morphology co-design** — the `hover` / `navigate` tasks build the airframe (arm
  lengths, motor tilt angles) from differentiable parameters, so the same gradient that
  trains the policy also optimizes the *shape of the drone*.
* **Sim-to-real** — the `hover_real` / `navigate_real` tasks drop morphology and fly an
  identified drone model instead, with domain randomization over the identified
  coefficients. Trained policies are exported to C for indiflight and can be flown
  hardware-in-the-loop against the simulator's own depth renderer.

Coordinates are **NED / FRD** throughout: x forward, y right, z down, ground plane at
`z = 0`, so flying altitude is negative.

## Quick start

```sh
# train (algorithm comes from the config; --algo/--epochs/--lr/... override it)
python train.py --config configs/train/navigate_real.yaml

# visualize a checkpoint in rerun
python rerun_rollout.py --config configs/train/navigate_real.yaml \
    --checkpoint checkpoints/navigate_obstacles_real_64.pkl --output rollout.rrd

# success rate over many episodes
python eval_success_rate.py --config configs/train/navigate_real.yaml \
    --checkpoint checkpoints/navigate_obstacles_real_64.pkl --episodes 1000
```

## Tasks

Selected by `env.name` in the config ([JADS/tasks/](JADS/tasks/)).

| name | drone model | observation | what it does |
|---|---|---|---|
| `hover` | morphology (hex, differentiable geometry) | state only | fly to a fixed target; co-designs the airframe |
| `navigate` | morphology (hex) | depth image + state | fly to a random target through random obstacles; co-designs the airframe |
| `hover_real` | identified model + domain randomization | state only | hover on a real airframe, any motor count |
| `navigate_real` | identified model + domain randomization | depth image + state | the deployed task — depth-based navigation on a real airframe |

The `*_real` tasks derive their motor count from the drone YAML, so the same code flies a
quad or a hex; `act_dim` / `obs_dim` / `drone_state_dim` are computed, not configured, and
a config that states them inconsistently is rejected at construction.

## Algorithms

`training.algo` in the config ([JADS/algos/](JADS/algos/)):

* **`bptt`** (default) — (truncated) backpropagation through time through the
  differentiable dynamics. With `persistent_carry: true` one `(state, obs, hidden, age)`
  carry is threaded across epochs, so training sees the on-policy state distribution far
  past `horizon` while gradients stay truncated to one window. Elements reset at the
  window boundary on crash (`reset_on_crash`) or at `max_episode_len`.
* **`ppo`** — recurrent PPO on the same env, policy and objective, as a model-free
  baseline. The logged `mean_return` is the same quantity in both, so wall-clock and
  sample cost are directly comparable.

## Layout

```
train.py                  training entry point — loads YAML, dispatches to the algo
rerun_rollout.py          roll out a checkpoint and log it to a rerun .rrd
eval_success_rate.py      success rate over N episodes

JADS/
  tasks/                  the four environments: reset / step / compute_loss
  algos/                  bptt.py (differentiable sim), ppo.py (baseline)
  models/                 mlp, gru, cnn_gru (depth → CNN → GRU → motor commands)
  drone_physics/          dynamics.py (morphology), dynamics_real.py (identified),
                          morphology.py, randomization.py, quat_math.py
  depth_render/           ray-traced depth camera — renderer, primitives, camera
  scene/                  obstacle sampling: static and infinite procedural modes
  utils/                  config loading, checkpointing, CSV logging

configs/
  train/                  one YAML per experiment (env + policy + training + camera)
  drone/                  identified airframes (coefficients, limits, geometry)
  scene/                  arena bounds and obstacle counts/sizes

hitl/                     hardware-in-the-loop depth renderer (C++) — see hitl/README.md
indiflight_module/        notebooks exporting a checkpoint to C for the flight controller
```

## Configs

A run is one YAML under [configs/train/](configs/train/) with four blocks — `env:`,
`policy:`, `training:`, `logging:` (plus `depth_camera:` for the vision tasks). The
`env.drone` and `env.scene` keys are *path references*, resolved relative to the config
file and replaced by the parsed contents of the file they point at, so airframes and
arenas are shared across experiments instead of copy-pasted.

Scenes come in two modes ([JADS/scene/scene.py](JADS/scene/scene.py)): **static**, where a
fixed set of spheres/boxes/capsules/windows/trees is sampled per episode and stored in the
state array, and **procedural**, where an infinite world is generated on the fly from a
per-episode seed and the drone's current cell.

## Deployment

Trained `*_real` policies leave the simulator through
[indiflight_module/](indiflight_module/):

* `Generate_C_code_hover.ipynb` — the hover GRU as a single C module.
* `Generate_C_code_navigate_split.ipynb` — the navigate policy **split** in two: the CNN
  encoder (runs off-board / on the companion computer) and the GRU + head (runs on the
  flight controller). Only the 64-float latent crosses between them.

[hitl/](hitl/) is a C++ port of the JAX depth pipeline that renders what the real drone
*would* see from its live mocap pose against a virtual obstacle field, so the perception
half of the policy can be flown in an empty room. It links the *generated* encoder, so a
HITL session exercises the exact translation unit that flies. See
[hitl/README.md](hitl/README.md).

---