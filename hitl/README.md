# `depth_hitl` — hardware-in-the-loop depth rendering

Renders the depth image a real drone *would* see, from its live mocap pose, against a
virtual obstacle field — so the perception half of the navigate policy can be flown in
an empty room.

```
  OptiTrack ──udp──▶ depth_hitl ──udp──▶ your CNN ──▶ latent ──▶ drone
   (pose, NED)      (ground station)     (12×16 tensor)
                     CPU, 10 Hz

  with --encode:
  OptiTrack ──udp──▶ depth_hitl ─────────udp────────▶ latent ──▶ drone
   (pose, NED)     render + CNN, 10 Hz    (64 floats)
```

The renderer is a port of the JAX pipeline the policy trained against
([JADS/depth_render/](../JADS/depth_render/)), so the tensor on the wire is the same
thing `NavigateReal._get_processed_depth` produces in simulation: ray-traced depth →
sensor model → `3/clip(d, 0.3, max) − 0.6` → 4×4 max-pool.

With `--encode` the CNN runs here too and the wire carries the 64-float latent instead —
see [Encoder](#encoder-sending-the-latent-instead-of-the-pixels).

No GPU, no external dependencies — C++17, POSIX sockets, libm. A 64×48 frame against
25 primitives takes **≈0.8 ms** on one core; the CNN on top of it costs ≈6 ms.

## Build

```sh
make -C hitl            # → hitl/depth_hitl
make -C hitl native     # tuned for this CPU; don't copy the binary elsewhere
```

The build looks for the generated CNN encoder at
`../indiflight_module/c_code_navigate/onboard` and links it in when it is there, which is
what enables `--encode`. It says which way it went:

```
note: CNN encoder linked in from ../indiflight_module/c_code_navigate/onboard — --encode available
note: no CNN encoder at ../indiflight_module/c_code_navigate/onboard — built without --encode support
```

Generate it by running
[indiflight_module/Generate_C_code_navigate_split.ipynb](../indiflight_module/Generate_C_code_navigate_split.ipynb),
or point elsewhere with `make -C hitl ENCODER_DIR=/path/to/onboard`. Everything else
builds and runs identically either way.

## Dataflow

Every port below is a default and every one is overridable. The ground station is the only
thing that talks to the mocap; the drone gets the pose and the latent from it.

```
                    ┌──────────────── ground station ────────────────┐
 mocap client       │                                               │
  (OptiTrack) ──────┼──▶ :5005  depth_hitl                          │
   200 Hz           │            │                                  │
                    │            ├─ --pose-forward ──────────────────┼──▶ drone :5005
                    │            │   verbatim, 200 Hz                │     (companion)
                    │            │                                  │
                    │            ├─ render 12×16 ─▶ CNN ─▶ latent    │
                    │            │   0.5 ms         6 ms   64 floats │
                    │            │                                  │
                    │            ├─ --out-host/--out-port ───────────┼──▶ drone :5010
                    │            │   10 Hz, 256 B + 76 B header      │
                    │            │                                  │
                    │            ├─ --pooled-out  :5012  12×16 tensor (debug / live_view)
                    │            └─ --raw-out     :5011  48×64 mm    (debug / live_view)
                    └───────────────────────────────────────────────┘
```

```sh
./hitl/depth_hitl --scene hitl/scenes/room00.json \
    --encode \
    --out-host 10.0.0.7 --out-port 5010 \
    --pose-forward 10.0.0.7:5005 \
    --sync-pose                      # worth adding below ~50 Hz mocap
```

| stream | port | rate | contents |
|---|---|---|---|
| pose in | 5005 | mocap rate | `--in-port`, the OptiTrack datagram |
| pose forward | 5005 | mocap rate | `--pose-forward`, byte-identical to what came in |
| main out | 5010 | `cam_hz` | `--out-host:--out-port` — 64-float latent with `--encode`, else the 12×16 tensor |
| raw out | 5011 | `cam_hz` | `--raw-out`, 48×64 uint16 mm |
| pooled out | 5012 | `cam_hz` | `--pooled-out`, the 12×16 tensor as a side channel |

Note the two rates: the pose relay runs at whatever the mocap client sends, while
everything rendered runs at the scene's `cam_hz` (10 Hz), because that is the rate the
policy was trained to expect its depth at. The latent packet header does also carry the
camera pose, but at 10 Hz and after the `--cam-offset`/`--cam-rpy` mount transform — it is
there so a frame can be re-rendered offline, not as a pose source for the controller.

## Quick start

```sh
# 1. bake ten episodes sized to your flight room (see hitl/configs/cyberzoo.yaml)
python hitl/tools/export_scene.py --hitl hitl/configs/cyberzoo.yaml

# 2. run the renderer (defaults: pose in on :5005, depth out to 127.0.0.1:5010)
./hitl/depth_hitl --scene hitl/scenes/cyberzoo00.json --scene-dir hitl/scenes

# 3. in another shell, consume the frames
python hitl/tools/recv_depth.py --port 5010
```

No mocap on the bench? Fly a scripted trajectory instead:

```sh
python hitl/tools/fake_mocap.py --traj orbit --cx 5 --radius 3.5 --speed 1.5
```

Hotkeys while running: `q` quit · `r` reload the current scene · `n` next scene in
`--scene-dir` · `p` toggle preview · `s` save the raw frame as a 16-bit PGM.

## Runbook — flying the split policy

Everything below is run from the repository root.

### Once, and again after every retrain

```sh
# 1. emit both halves of the split policy from the checkpoint
#    (notebook: indiflight_module/Generate_C_code_navigate_split.ipynb)
#      -> indiflight_module/c_code_navigate/onboard/  the CNN, for here
#      -> indiflight_module/c_code_navigate/fc/       the GRU, for the FC

# 2. bake the obstacle field to your flight room
python hitl/tools/export_scene.py --hitl hitl/configs/cyberzoo.yaml

# 3. build the renderer — picks up the encoder from step 1 automatically
make -C hitl
```

Step 3 takes about **0.7 s** after a new encoder: `make` sees the regenerated
`cnn_encoder.c`, recompiles that one object and relinks. There is no separate "install the
weights" step, and no way for the binary and the weights to fall out of sync.

Step 2 is only needed again if the retrain changed the arena or the camera block. It is
deterministic — the same seeds give byte-identical scenes — so re-running it is always
safe. If the camera resolution changed and the scenes did not, `--encode` refuses to start
rather than feed the CNN a wrongly-shaped tensor.

Then flash indiflight with `indiflight_module/c_code_navigate/fc/`. **Both halves come
from the same run of the notebook** — nothing checks this at runtime, and mismatched halves
produce a plausible-looking latent rather than an error.

### Every session

```sh
# terminal 1 — pose source. Real mocap: point unified-mocap-client at this
#              machine on :5005. Bench: fly a scripted trajectory instead.
python hitl/tools/fake_mocap.py --traj orbit --cx 2 --radius 2.0 --speed 1.5 --rate 20

# terminal 2 — render + encode, latent to the drone, pose relayed on to it
./hitl/depth_hitl \
    --scene hitl/scenes/cyberzoo00.json --scene-dir hitl/scenes \
    --encode \
    --out-host 10.0.0.7 --out-port 5010 \
    --pose-forward 10.0.0.7:5005 \
    --sync-pose
```

Drop `--pose-forward` if your mocap client already reaches the drone directly, and
`--sync-pose` above roughly 50 Hz mocap. For a loopback bench test, leave `--out-host` at
its `127.0.0.1` default.

### Watching it

```sh
# add --pooled-out --raw-out to depth_hitl, then:
python hitl/tools/live_view.py --scene hitl/scenes/cyberzoo00.json   # 3D map + latent
python hitl/tools/recv_depth.py --port 5010 --show                   # the 64 floats
```

### Before the first flight

```sh
# renderer matches the simulator
python hitl/tools/compare_with_jax.py --scene hitl/scenes/cyberzoo00.json --n 200

# the CNN is being fed correctly (needs --encode --pooled-out running)
python hitl/tools/check_encoder.py --n 50
```

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
| `drone/camera/depth` | the rendered depth image in metres, back-projected into the frustum |
| `view/proximity` | the same frame inverted — **close is bright**, empty space is dark |
| `cnn_input` | the 12×16 tensor as it reaches the policy |
| `cnn_latent` | with `--encode`: the 64 features, as a bar chart |
| `world/trajectory` | flown path |
| `health/*` | rate, render cost, encode cost, pose age, stale flag, nearest obstacle |

With `depth_hitl --encode` the main port carries the latent rather than pixels, so run it
`--encode --pooled-out --raw-out` to keep every panel populated; the viewer binds the
pooled side channel (5012) by default and needs no extra flags.

Every rerun colormap runs dark→bright with increasing value, so a metric depth image
paints the far plane — typically two thirds of a frame — in the loudest colour. That is
why `drone/camera/depth` stays metric (rerun needs true metres to back-project it into
the frustum) but is drawn in `grayscale` by default, while `view/proximity` carries the
inverted copy where obstacles are the bright thing. `--colormap` changes the former.

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

### Relaying the pose on to the drone

The companion computer needs the pose too, and if the mocap client can only unicast to one
destination, the ground station can pass it along:

```sh
./hitl/depth_hitl --scene ... --pose-forward 10.0.0.7        # port 5005 by default
./hitl/depth_hitl --scene ... --pose-forward 10.0.0.7:5005   # explicit
```

Each accepted datagram is re-sent **unmodified** — same 72/76-byte layout, same
timestamps, same rigid-body id — so whatever consumes the mocap stream onboard needs no
changes. Only packets that pass the size, `--rb-id` and finiteness checks are relayed, so
the drone sees the same filtered stream the renderer does.

Forwarding happens as each packet arrives, not once per rendered frame. That distinction
matters: `drain()` normally runs only at the 10 Hz tick, so relaying from there would hold
every packet for up to a render period and then deliver the backlog in a burst — the
average rate would be right and nothing would be lost, but a 100 Hz controller cannot fly
on a pose that arrives in 100 ms clumps. With forwarding enabled the loop instead waits on
the socket and relays on arrival, handing only the last 2 ms to `clock_nanosleep` so the
render cadence keeps its sub-millisecond precision. Measured with a 200 Hz source:

```
                   mean gap   median    p95      max
  at the tick      5.000 ms   0.005    0.007   99.455    ← bursts, unusable
  on arrival       5.000 ms   5.003    7.104   10.951    ← a stream
```

The residual ~11 ms worst case is a packet landing during the render-plus-encode window;
it is relayed at the next drain. The render loop is unaffected — 9.9935 Hz, 0.41 ms std,
against 0.28 ms with forwarding off.

Forwarding to the port this process listens on, on this machine, is refused rather than
allowed to become a feedback loop. `fwd N → host:port` in the status line counts what has
gone out, and turns red on send errors.

If your mocap client *can* address both machines — broadcast, multicast, or two unicast
destinations — prefer that. It is one less hop, one less process in the pose path, and the
drone keeps its pose stream when the ground station is restarted mid-session.

Not sure what is on the wire? `./hitl/depth_hitl --sniff 3 --in-port 5005` hexdumps the
next three packets and decodes them under both layouts.

If several rigid bodies are streamed, pin the drone's with `--rb-id N`.

## Depth output

`--out-host` takes any address, so the stream can go straight to the drone's companion
computer rather than to loopback:

```sh
./hitl/depth_hitl --scene ... --encode --out-host 10.0.0.7 --out-port 5010
```

`127.0.0.1` is only the default because the common bench case is a consumer on the same
machine. Broadcast addresses work too (`SO_BROADCAST` is set).

One datagram per frame to `--out-host:--out-port`, 76-byte header then payload; the
layout is documented in [src/netout.hpp](src/netout.hpp) and decoded by
[tools/recv_depth.py](tools/recv_depth.py), whose `decode()` returns the 12×16 float32
array ready for `CNNEncoder`.

The header carries `seq`, the pose timestamp and age, the render cost, a stale flag, and
the camera pose the frame was rendered from — enough to detect dropped frames and to
re-render any frame offline.

`--raw-out` additionally publishes the full-resolution 48×64 depth image in millimetres
(`uint16`) on `--raw-port`, for logging and visual checks. It is not what the CNN eats.

## Encoder: sending the latent instead of the pixels

The deployed policy is split: the CNN runs off-board and only its 64-float output crosses
to the flight controller, where the GRU turns it into motor commands
([Generate_C_code_navigate_split.ipynb](../indiflight_module/Generate_C_code_navigate_split.ipynb)).
`--encode` makes the HITL rig match that shape — it runs the CNN on the ground station and
publishes the latent, so what goes over the air is what will go over the air in the real
system:

```sh
./hitl/depth_hitl --scene hitl/scenes/room00.json --encode
python hitl/tools/recv_depth.py --port 5010 --show      # 64 floats per frame
```

`--out-port` then carries payload type 2: `rows=1`, `cols=64`, float32.
`Frame.is_features` and `Frame.features` in [recv_depth.py](tools/recv_depth.py) pick it
apart, and `Frame.encode_us` reports what the CNN cost.

**This is not a reimplementation.** The Makefile compiles the *generated* `cnn_encoder.c`
straight into `depth_hitl` — the same translation unit the companion computer flies. So a
HITL session exercises the encoder that will actually be deployed, and the only thing left
that can differ between bench and flight is the depth image feeding it. That is also why
`--encode` refuses a scene whose camera pools to something other than the shape the
encoder was generated for, on startup and on the `n` hotkey, rather than feeding the CNN a
tensor of the wrong size.

`cnn_preprocess()` from the generated encoder is deliberately *not* called: `render_frame`
has already applied the sensor model, the normalization and the max-pool, and doing it
twice would square the normalization. Only `cnn_forward()` runs here.

### Keeping the tensor visible

`--pooled-out` publishes the 12×16 tensor on `--pooled-port` (5012) alongside the latent,
carrying the same `seq` and pose. That keeps [live_view.py](tools/live_view.py)'s
`cnn_input` panel alive once the main port has stopped carrying pixels, and it is what
lets the latent be checked against the tensor that produced it:

```sh
./hitl/depth_hitl --scene hitl/scenes/room00.json --encode --pooled-out --no-preview
python hitl/tools/check_encoder.py --n 50
```

[check_encoder.py](tools/check_encoder.py) pairs the two streams by `seq` and compares the
C latent against `CNNEncoder` applied to that exact tensor:

```
50 frame pairs
  max  |C - JAX|  6.706e-06    (relative 2.59e-05)
  encode cost     5.19 ms mean, 7.42 ms max

PASS — the ground station feeds the CNN what training fed it, and the latent on :5010 is
the one the policy expects.
```

The weights cannot disagree — the same object file produced both halves of the comparison
— so this is a **wiring** test, not a numerics one. What it catches is the pooled tensor
arriving transposed, row/column swapped, off by a frame, or scaled differently from what
the encoder was generated for. Each of those yields a plausible-looking latent and a drone
that flies into things, and none would be caught by checking weights.

`--print-features` prints the latent under the terminal preview, which is enough to notice
an encoder that has gone constant or saturated.

### Cost

The CNN is 4.8 M MAC per frame against the renderer's 0.8 ms, and takes **≈6 ms** on one
core — 6% of the 100 ms budget at 10 Hz, so the loop rate is unaffected. It is reported
per-frame in the status line, in the packet header (`encode_us`), and as
`health/encode_ms` in the live view.

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

### Fitting the obstacle field to your room

The training arena is not your flight volume, so [configs/room.yaml](configs/room.yaml)
states only the differences and inherits the rest:

```yaml
base: ../../configs/train/navigate_real.yaml

scene:
  arena_x_min: -4.0
  arena_x_max:  4.0
  arena_y_min: -4.0
  arena_y_max:  4.0
  arena_z_min: -4.0     # NED — this is a 4 m ceiling
  arena_z_max:  0.0
  n_spheres:  3
  n_boxes:    3
  n_capsules: 3

export:
  seeds:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  out_dir: hitl/scenes
  prefix:  room
```

```sh
python hitl/tools/export_scene.py --hitl hitl/configs/room.yaml
```

Any key of `SceneConfig` may go under `scene:` — spawn bounds, counts, or the size
ranges (`sphere_r_min`, `box_hx_max`, `capsule_hh_min`, …) — and an unknown key is
rejected rather than silently ignored. The training config is never touched, so the
simulator keeps training on its own arena.

The exporter prints the obstacle density either side of the change, because that is the
number the policy is sensitive to:

```
arena    training   144.0 m³ /  18 obstacles = 0.125 per m³
         this run   256.0 m³ /   9 obstacles = 0.035 per m³   (0.28× as dense)
```

Sparser than training is the safe direction for a first flight. Substantially denser
means tighter clearances than the policy ever saw, and it says so.

Keep obstacles inside the volume the mocap actually covers — the drone can only fly up
to obstacles that exist where it can go — and remember `arena_z_max: 0.0` lets them sit
on the floor, which is usually what you want since the ground plane is part of the
trained scene.

### Format

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

The pose used is the newest packet available at the instant the frame is rendered. The
socket is read as packets arrive rather than only at the tick, so the depth image is never
built from a backlogged pose and `pose_age_us` is the pose's real age. If nothing has
arrived within `--pose-timeout-ms` (200 ms default), the frame is still rendered from the
last known pose but flagged `FLAG_STALE_POSE`; **the consumer should check that flag before
acting on a frame.**

### Low-rate mocap, and `--sync-pose`

The render cadence and the mocap stream are independent clocks, so the age of the pose a
frame is rendered from is set by their relative phase: constant through a flight, but
anywhere between zero and one full mocap period, and not something you get to choose. At
100 Hz mocap the worst case is 10 ms and nobody cares. **At 20 Hz it is up to 50 ms**,
which at the trained 2 m/s is 10 cm of position error baked into the depth image — in the
unsafe direction, since obstacles appear further away than they are.

`--sync-pose` steers the render deadline until the pose age sits at `--sync-target-ms`
(5 ms default). Rendering slightly earlier lands the deadline earlier against the pose grid
and lowers the age one-for-one, so the correction is a slew rather than a jump: capped at
5 ms per frame, which keeps the period within 5% of the trained rate while it acquires and
takes about a second to absorb a 50 ms error. The target is a few ms rather than zero so
mocap jitter cannot push the render just ahead of an arrival and cost a whole period.

Measured against a 20 Hz source, three runs each at a different starting phase:

```
                render period        pose age at render
  free-running  9.99 Hz              36.3 / 36.4 / 37.6 ms   ← whatever the phase is
  --sync-pose   9.99 Hz               5.0 /  5.0 /  5.0 ms   ← what you asked for
```

The rate is unchanged either way; only the phase moves.

> A 20 Hz mocap is a perfectly good **EKF correction** rate — the flight controller fuses
> it with the IMU to produce state at the full control rate. It is not a 100 Hz pose
> source, and nothing here turns it into one. What `--sync-pose` fixes is narrower: which
> pose the *depth image* is rendered from.

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
| [src/encoder.cpp](src/encoder.cpp) | wrapper around the generated CNN (`--encode`) |
| [src/scene.cpp](src/scene.cpp) | scene JSON loader |
| [src/mocap.cpp](src/mocap.cpp) | pose datagram receiver + `--sniff` |
| [src/netout.cpp](src/netout.cpp) | depth / latent frame publisher |
| [src/preview.cpp](src/preview.cpp) | terminal preview, PGM dump |
| [src/main.cpp](src/main.cpp) | CLI, real-time loop, hotkeys |
| [configs/room.yaml](configs/room.yaml) | your flight room: arena bounds and obstacle counts |
| [tools/export_scene.py](tools/export_scene.py) | bake episodes from `SceneConfig` |
| [tools/compare_with_jax.py](tools/compare_with_jax.py) | renderer parity check against the simulator |
| [tools/check_encoder.py](tools/check_encoder.py) | latent parity check — is the CNN fed correctly |
| [tools/fake_mocap.py](tools/fake_mocap.py) | scripted pose source for bench testing |
| [tools/recv_depth.py](tools/recv_depth.py) | frame decoder / reference consumer |
| [tools/live_view.py](tools/live_view.py) | live rerun map, pose, frustum and depth image |
