#!/usr/bin/env python3
"""
compare_with_jax.py — verify the C++ renderer against the training pipeline.

Renders the same poses against the same baked scene three ways:

  truth    JADS/depth_render in float64  — exact reference
  jax f32  JADS/depth_render exactly as training runs it
  c++      the compiled depth_hitl binary

  python hitl/tools/compare_with_jax.py --scene hitl/scenes/ep00.json --n 200

Pass/fail is on `c++ vs truth` — that is what this port controls.

`jax f32 vs truth` is reported for information, and on a GPU with scenes that
contain capsules it is expected to be the worse of the two. Under vmap the
3-vector dot products in ray_cylinder lower to batched DOT ops, which XLA runs
on tensor cores in TF32 (~1e-3 relative) by default; the discriminant
b² - a·c cancels to ~5e-5 relative for rays grazing a capsule, so the hit/miss
decision flips and the simulator's own depth images pick up metre-scale speckle
on ~0.1% of pixels. Running the comparison with
JAX_DEFAULT_MATMUL_PRECISION=highest makes that row match the reference, which
confirms the cause. The C++ renderer never issues a DOT op, so it is unaffected.

Run from the repository root, after `make -C hitl`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax

jax.config.update("jax_enable_x64", True)   # must precede any jnp work

import jax.numpy as jnp
import numpy as np

from JADS.depth_render.renderer import render_depth, apply_sensor_noise


def load_scene(path: str, dtype=jnp.float32):
    with open(path) as f:
        doc = json.load(f)

    def stack(items, key, dim):
        if not items:
            return jnp.zeros((0, dim) if dim > 1 else (0,), dtype)
        arr = np.array([it[key] for it in items], dtype=np.float64)
        return jnp.asarray(arr.reshape(len(items), dim) if dim > 1 else arr.reshape(-1), dtype)

    spheres   = doc.get("spheres", [])
    boxes     = doc.get("boxes", [])
    cylinders = doc.get("cylinders", [])
    obbs      = doc.get("obbs", [])

    geom = dict(
        sphere_centers   = stack(spheres, "c", 3),
        sphere_radii     = stack(spheres, "r", 1),
        box_centers      = stack(boxes, "c", 3),
        box_half_extents = stack(boxes, "he", 3),
        cylinder_centers = stack(cylinders, "c", 3),
        cylinder_axes    = stack(cylinders, "ax", 3),
        cylinder_hh      = stack(cylinders, "hh", 1),
        cylinder_radii   = stack(cylinders, "r", 1),
        obb_centers      = stack(obbs, "c", 3),
        obb_quaternions  = stack(obbs, "q", 4),
        obb_half_extents = stack(obbs, "he", 3),
    )
    return doc, geom


def jax_pipeline(geom, cam, pos, quat, dtype):
    """render → sensor model → normalize → max-pool, as NavigateReal does."""
    raw = apply_sensor_noise(
        render_depth(
            position=jnp.asarray(pos, dtype),
            quaternion=jnp.asarray(quat, dtype),
            fov_deg=cam["fov_deg"], width=cam["width"], height=cam["height"],
            **geom,
        ),
        min_range=cam["min_range"],
        max_range=cam["max_range"],
        quantization_m=cam["quantization_m"],
    )
    normd = cam["norm_numerator"] / jnp.clip(raw, cam["norm_clip_min"], cam["max_range"]) \
            - cam["norm_offset"]
    p = cam["pool"]
    pooled = jax.lax.reduce_window(
        normd, -jnp.inf, jax.lax.max,
        window_dimensions=(p, p), window_strides=(p, p), padding="VALID",
    )
    return np.asarray(raw, np.float64), np.asarray(pooled, np.float64)


def random_poses(n: int, seed: int, doc) -> np.ndarray:
    """Poses spread over the region the obstacles occupy, with free attitude."""
    rng = np.random.default_rng(seed)
    centres = np.array([s["c"] for s in doc.get("spheres", [])] +
                       [b["c"] for b in doc.get("boxes", [])] +
                       [c["c"] for c in doc.get("cylinders", [])], dtype=np.float64)
    if len(centres) == 0:
        lo, hi = np.array([-2., -2., -2.]), np.array([8., 2., -0.2])
    else:
        lo = centres.min(0) - 2.0
        hi = centres.max(0) + 2.0
        lo[2], hi[2] = min(lo[2], -2.5), -0.2      # stay above the ground plane

    pos  = rng.uniform(lo, hi, size=(n, 3))
    quat = rng.normal(size=(n, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    quat[quat[:, 0] < 0] *= -1.0
    return np.hstack([pos, quat]).astype(np.float32)


def run_cpp(binary: str, scene: str, poses: np.ndarray, n_raw: int, n_pooled: int):
    with tempfile.TemporaryDirectory() as tmp:
        pose_file = os.path.join(tmp, "poses.txt")
        dump_file = os.path.join(tmp, "frames.bin")
        with open(pose_file, "w") as f:
            for p in poses:
                f.write(" ".join(f"{v:.9g}" for v in p) + "\n")

        res = subprocess.run([binary, "--scene", scene, "--poses", pose_file,
                              "--dump", dump_file], capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr, file=sys.stderr)
            raise SystemExit(f"FAIL: {binary} exited {res.returncode}")
        blob = np.fromfile(dump_file, dtype=np.float32)

    per_frame = n_raw + n_pooled
    if blob.size != per_frame * len(poses):
        raise SystemExit(f"FAIL: dump has {blob.size} floats, expected {per_frame * len(poses)}")
    blob = blob.reshape(len(poses), per_frame).astype(np.float64)
    return blob[:, :n_raw], blob[:, n_raw:]


class Stats:
    """Accumulates one renderer's disagreement with the reference."""

    def __init__(self, label):
        self.label = label
        self.worst_raw = 0.0
        self.worst_raw_pose = -1
        self.worst_pooled = 0.0
        self.worst_pooled_pose = -1
        self.px_off = 0
        self.px_total = 0
        self.cells_off = 0
        self.cells_total = 0
        self.frames_off = 0

    def add(self, i, raw, pooled, ref_raw, ref_pooled):
        d_raw = np.abs(raw - ref_raw)
        d_pool = np.abs(pooled - ref_pooled)
        self.px_total += d_raw.size
        self.cells_total += d_pool.size
        self.px_off += int((d_raw > 0.01).sum())
        self.cells_off += int((d_pool > 0.01).sum())
        self.frames_off += int(d_pool.max() > 0.01)
        if d_raw.max() > self.worst_raw:
            self.worst_raw, self.worst_raw_pose = float(d_raw.max()), i
        if d_pool.max() > self.worst_pooled:
            self.worst_pooled, self.worst_pooled_pose = float(d_pool.max()), i

    def report(self):
        print(f"  {self.label}")
        print(f"    raw    worst |Δ| {self.worst_raw * 1000:9.3f} mm (pose {self.worst_raw_pose});"
              f"  pixels off >1cm: {self.px_off}/{self.px_total}"
              f" ({100.0 * self.px_off / max(1, self.px_total):.4f}%)")
        print(f"    pooled worst |Δ| {self.worst_pooled:9.6f}    (pose {self.worst_pooled_pose});"
              f"  cells off >0.01: {self.cells_off}/{self.cells_total}"
              f" ({100.0 * self.cells_off / max(1, self.cells_total):.4f}%)"
              f"  in {self.frames_off} frame(s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--bin", default="hitl/depth_hitl")
    ap.add_argument("--n", type=int, default=64, help="number of random poses [64]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol-raw", type=float, default=1.1e-3,
                    help="allowed C++ raw depth error in metres [1.1 mm = one "
                         "quantization step]")
    ap.add_argument("--tol-pooled", type=float, default=None,
                    help="allowed C++ error in the pooled CNN input "
                         "[default: what one quantization step is worth at the "
                         "closest depth, i.e. num/clip_min² × quant]")
    args = ap.parse_args()

    doc, geom32 = load_scene(args.scene, jnp.float32)
    _,   geom64 = load_scene(args.scene, jnp.float64)
    cam = doc["camera"]
    H, W   = cam["height"], cam["width"]
    PH, PW = H // cam["pool"], W // cam["pool"]

    # A raw depth that lands exactly on a quantization boundary may round either
    # way between the two implementations. Allow one step, and whatever that step
    # is worth after 3/clip(d) - 0.6, which is steepest at clip_min.
    tol_pooled = args.tol_pooled
    if tol_pooled is None:
        tol_pooled = cam["norm_numerator"] / cam["norm_clip_min"] ** 2 * cam["quantization_m"]

    poses = random_poses(args.n, args.seed, doc)
    c_raw, c_pooled = run_cpp(args.bin, args.scene, poses, H * W, PH * PW)

    cpp = Stats("c++      vs truth")
    jx  = Stats("jax f32  vs truth")
    for i, p in enumerate(poses):
        ref_raw, ref_pooled = jax_pipeline(geom64, cam, p[:3], p[3:], jnp.float64)
        j_raw,   j_pooled   = jax_pipeline(geom32, cam, p[:3], p[3:], jnp.float32)
        cpp.add(i, c_raw[i].reshape(H, W), c_pooled[i].reshape(PH, PW), ref_raw, ref_pooled)
        jx.add(i, j_raw, j_pooled, ref_raw, ref_pooled)

    print(f"scene   {args.scene}  ({H}x{W} → {PH}x{PW}, {len(poses)} poses, "
          f"backend {jax.devices()[0].platform})")
    print(f"tol     raw {args.tol_raw * 1000:.2f} mm, pooled {tol_pooled:.4f} "
          f"(one {cam['quantization_m'] * 1000:.0f} mm quantization step)")
    cpp.report()
    jx.report()

    ok = cpp.worst_raw <= args.tol_raw and cpp.worst_pooled <= tol_pooled
    print("PASS" if ok else "FAIL")
    if ok and jx.worst_pooled > tol_pooled:
        print("note: the float32 JAX renderer disagrees with exact maths by "
              f"{jx.worst_pooled:.3f} in the pooled tensor — see this file's docstring.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
