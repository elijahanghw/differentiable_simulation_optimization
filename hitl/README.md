# `depth_hitl` — hardware-in-the-loop depth rendering

Renders the depth image a real drone *would* see, from its live mocap pose, against a
virtual obstacle field — so the perception half of the navigate policy can be flown in
an empty room.

```
  OptiTrack ──udp──▶ depth_hitl ──udp──▶ your CNN ──▶ latent ──▶ drone
   (pose, NED)      (ground station)     (12×16 tensor)
                     CPU, 10 Hz
```

The renderer is a port of the JAX pipeline the policy trained against
([JADS/depth_render/](../JADS/depth_render/)), so the tensor on the wire is the same
thing `NavigateReal._get_processed_depth` produces in simulation: ray-traced depth →
sensor model → `3/clip(d, 0.3, max) − 0.6` → 4×4 max-pool.

No GPU, no external dependencies — C++17, POSIX sockets, libm. A 64×48 frame against
25 primitives takes **≈0.8 ms** on one core.

## Build

```sh
make -C hitl            # → hitl/depth_hitl
make -C hitl native     # tuned for this CPU; don't copy the binary elsewhere
```

## Quick start

```sh
# 1. bake ten episodes from the training scene config
python hitl/tools/export_scene.py --config configs/train/navigate_real.yaml \
    --seeds 0-9 --out-dir hitl/scenes

# 2. run the renderer (defaults: pose in on :5005, depth out to 127.0.0.1:5010)
./hitl/depth_hitl --scene hitl/scenes/ep00.json --scene-dir hitl/scenes

# 3. in another shell, consume the frames
python hitl/tools/recv_depth.py --port 5010
```

No mocap on the bench? Fly a scripted trajectory instead:

```sh
python hitl/tools/fake_mocap.py --traj orbit --cx 5 --radius 3.5 --speed 1.5
```

Hotkeys while running: `q` quit · `r` reload the current scene · `n` next scene in
`--scene-dir` · `p` toggle preview · `s` save the raw frame as a 16-bit PGM.

## Live 3D view

[tools/live_view.py](tools/live_view.py) opens a rerun window showing the obstacle
field, the drone's live pose and body axes, the camera frustum with the depth image
projected into it, the flown trajectory, and timing scalars.

```sh
./hitl/depth_hitl --scene hitl/scenes/ep00.json --raw-out --no-preview   # terminal 2
python hitl/tools/live_view.py --scene hitl/scenes/ep00.json             # terminal 3
```

It subscribes to the depth stream rather than to the mocap, so the pose it draws is
byte-for-byte the pose each frame was rendered from — the map and the image cannot
drift apart, and the viewer adds no load to the 10 Hz loop. `--raw-out` on the renderer
is what gives it the 48×64 image; without it the viewer falls back to displaying the
12×16 tensor.

| entity | what it is |
|---|---|
| `world/{ground,spheres,boxes,capsules,branches}` | the baked obstacle field, logged once |
| `drone/{body,arms,motors,axes}` | the airframe, static in the body frame |
| `drone/camera` | pinhole frustum, `focal = (W/2)/tan(fov/2)`, FRD |
| `drone/camera/depth` | the rendered depth image, in the frustum |
| `cnn_input` | the 12×16 tensor as it reaches the policy |
| `world/trajectory` | flown path |
| `health/*` | rate, render cost, pose age, stale flag, nearest obstacle |

`--save session.rrd` records instead of spawning a window (add `--spawn` for both),
`--connect` attaches to a viewer that is already open, and `--duration N` stops cleanly
after N seconds. The obstacle field is logged once at startup, so after changing episode
with the `n` hotkey, restart the viewer with the matching `--scene`.

## Pose input

The datagram is the one [relay.cpp](../tmp/relay.cpp) consumes — `uint32` streaming id,
then the sender's `pose_t` and `pose_der_t` structs, little-endian, 76 bytes:

| offset | type | field |
|---|---|---|
| 0 | `uint32` | rigid-body streaming id |
| 4 | `uint64` | pose timestamp, µs (camera mid-exposure) |
| 12 | `float×3` | position x, y, z — **NED, metres** |
| 24 | `float×4` | quaternion qx, qy, qz, qw — body→world |
| 40 | — | 4 bytes of `pose_t` tail padding |
| 44 | `uint64` | derivative timestamp, µs |
| 52 | `float×3` | velocity, NED m/s |
| 64 | `float×3` | body rates, rad/s |

The 72-byte packed variant (sender built without the padding) is accepted too. Only
position and quaternion are used; velocity and rates are parsed but not needed for
rendering.

> **Port.** `relay.cpp` listens on **5005**, not 5000 — that is the default here. Override
> with `--in-port`. The pose stream has to actually reach the ground station: if
> unified-mocap-client unicasts only to the drone's companion computer, point it at a
> broadcast address or run a second instance.

Not sure what is on the wire? `./hitl/depth_hitl --sniff 3 --in-port 5005` hexdumps the
next three packets and decodes them under both layouts.

If several rigid bodies are streamed, pin the drone's with `--rb-id N`.

## Depth output

One datagram per frame to `--out-host:--out-port`, 76-byte header then payload; the
layout is documented in [src/netout.hpp](src/netout.hpp) and decoded by
[tools/recv_depth.py](tools/recv_depth.py), whose `decode()` returns the 12×16 float32
array ready for `CNNEncoder`.

The header carries `seq`, the pose timestamp and age, the render cost, a stale flag, and
the camera pose the frame was rendered from — enough to detect dropped frames and to
re-render any frame offline.

`--raw-out` additionally publishes the full-resolution 48×64 depth image in millimetres
(`uint16`) on `--raw-port`, for logging and visual checks. It is not what the CNN eats.

## Coordinate conventions

Everything is NED: **x forward, y right, z down**, ground plane at `z = 0`, so flying
height is negative. The quaternion is `[qw, qx, qy, qz]`, body→world. The camera looks
along body **+x**, image up is body **−z** — identical to
[JADS/depth_render/camera.py](../JADS/depth_render/camera.py).

By default the camera sits exactly at the mocap rigid-body origin, aligned with the body
axes, which is what the simulator assumes. If the real camera is mounted forward of, or
tilted relative to, the rigid-body origin:

```sh
./hitl/depth_hitl --scene ... --cam-offset 0.03,0,-0.01 --cam-rpy 0,15,0
```

`--cam-offset` is in body FRD metres, `--cam-rpy` in degrees (ZYX, matching
`quat_math.euler_to_quat`). Calibrate by parking the drone at a known spot and checking
the preview against the real geometry.

## Scenes

One JSON file per episode, baked by
[tools/export_scene.py](tools/export_scene.py) from the same `SceneConfig` used in
training — `jax.random` cannot be reproduced in C++, so the obstacles are sampled in
Python and written out as explicit geometry. Each file also carries the camera block
(FOV, resolution, ranges, quantization, pool factor) from the training config, so the
renderer cannot silently drift from what the policy was trained with. CLI flags
(`--fov`, `--max-range`, …) override it when you need to experiment.

Procedural (infinite-world) scene configs need a region to bake over:

```sh
python hitl/tools/export_scene.py --config configs/train/navigate_morph_trees.yaml \
    --seed 3 --bake-region -2 12 -6 6 --out hitl/scenes/trees03.json
```

Obstacles are static for the lifetime of a scene file, which is what an episode needs.
Press `n` to move to the next episode without restarting.

## Timing

The loop uses absolute `CLOCK_MONOTONIC` deadlines, so the period never accumulates
drift. Measured over 150 frames on an idle machine:

```
inter-frame period: mean 99.978 ms, std 0.282 ms, min 96.730, max 100.218 → 10.0022 Hz
```

A frame that overruns its slot skips the missed slots rather than firing a catch-up
burst; the count is shown in the status line. `--rt` requests `SCHED_FIFO` and
`mlockall` (needs privileges — it warns and continues without them). Rendering takes
under 1 ms of the 100 ms budget, so the preview and the socket work are free.

The pose used is the newest packet available at the instant the frame is rendered — the
socket is drained to its tail every tick, so the depth image is never built from a
backlogged pose. If nothing has arrived within `--pose-timeout-ms` (200 ms default), the
frame is still rendered from the last known pose but flagged `FLAG_STALE_POSE`; **the
consumer should check that flag before acting on a frame.**

## Verifying against the simulator

```sh
python hitl/tools/compare_with_jax.py --scene hitl/scenes/ep00.json --n 200
```

Renders the same random poses three ways — the JAX pipeline in float64 (exact
reference), the JAX pipeline in float32 (what training actually runs), and the C++
binary — and reports each against the reference. Current result over 48 poses:

```
  c++      vs truth   raw worst |Δ| 1.000 mm;   pixels off >1cm: 0/147456
  jax f32  vs truth   raw worst |Δ| 1244.000 mm; pixels off >1cm: 8/147456
```

The C++ renderer is exact to within one 1 mm quantization step (a depth landing on a
rounding boundary can go either way). The float32 JAX row is worse, and that is a
property of the simulator, not of this port:

> Under `vmap`, the 3-vector dot products in `ray_cylinder` lower to batched DOT
> operations, which XLA runs on tensor cores in **TF32** (10-bit mantissa, ~1e-3
> relative) by default on Ampere+ GPUs. `ray_cylinder`'s discriminant `b² − a·c` cancels
> to ~5e-5 relative for rays grazing a capsule, so a TF32-level perturbation flips the
> hit/miss decision and puts metre-scale speckle on ~0.1% of pixels of capsule
> silhouettes. Re-running with `JAX_DEFAULT_MATMUL_PRECISION=highest` makes the row match
> the reference exactly, which confirms the cause. Spheres, boxes, OBBs and the ground
> plane are unaffected — their coefficients don't cancel. The C++ renderer never issues a
> DOT op, so it renders the capsules correctly.

To validate the whole chain rather than the renderer alone, capture live frames and
re-render their embedded poses:

```sh
python hitl/tools/recv_depth.py --port 5010 --count 100 --save live.npz
```

## Files

| | |
|---|---|
| [src/render.cpp](src/render.cpp) | ray tracing, sensor model, normalization, max-pool |
| [src/scene.cpp](src/scene.cpp) | scene JSON loader |
| [src/mocap.cpp](src/mocap.cpp) | pose datagram receiver + `--sniff` |
| [src/netout.cpp](src/netout.cpp) | depth frame publisher |
| [src/preview.cpp](src/preview.cpp) | terminal preview, PGM dump |
| [src/main.cpp](src/main.cpp) | CLI, real-time loop, hotkeys |
| [tools/export_scene.py](tools/export_scene.py) | bake episodes from `SceneConfig` |
| [tools/compare_with_jax.py](tools/compare_with_jax.py) | parity check against the simulator |
| [tools/fake_mocap.py](tools/fake_mocap.py) | scripted pose source for bench testing |
| [tools/recv_depth.py](tools/recv_depth.py) | frame decoder / reference consumer |
| [tools/live_view.py](tools/live_view.py) | live rerun map, pose, frustum and depth image |
