#!/usr/bin/env python3
"""
export_scene.py — bake one HITL episode's obstacle field to JSON.

The ground-station renderer (hitl/depth_hitl) is a C++ port of the JAX depth
pipeline, but it cannot reproduce jax.random. So the obstacles are sampled here,
with the same JADS/scene/scene.py SceneConfig the policy trained against, and
written out as explicit geometry. One file = one episode = one static scene.

  # one scene from the navigate_real training config
  python hitl/tools/export_scene.py --config configs/train/navigate_real.yaml \
      --seed 0 --out hitl/scenes/ep00.json

  # a directory of ten episodes to cycle through with the 'n' hotkey
  python hitl/tools/export_scene.py --config configs/train/navigate_real.yaml \
      --seeds 0-9 --out-dir hitl/scenes

  # a procedural (infinite-world) scene config, baked over the flight volume
  python hitl/tools/export_scene.py --config configs/train/navigate_morph_trees.yaml \
      --seed 3 --bake-region -2 12 -6 6 --out hitl/scenes/trees03.json

Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax
import numpy as np
import yaml

from JADS.scene.scene import SceneConfig
from JADS.utils.config import load_config


# ---------------------------------------------------------------------------
# Geometry → JSON
# ---------------------------------------------------------------------------

def _f(x) -> float:
    """float32 → the shortest double that round-trips it exactly."""
    return float(np.float32(x))


def _vec(v):
    return [_f(c) for c in np.asarray(v).reshape(-1)]


def geometry_to_json(d: dict) -> dict:
    """`SceneConfig.unpack`-style dict of arrays → the renderer's JSON blocks."""
    spheres, boxes, cylinders, obbs = [], [], [], []

    sc, sr = np.asarray(d["sphere_centers"]), np.asarray(d["sphere_radii"])
    for c, r in zip(sc.reshape(-1, 3), sr.reshape(-1)):
        spheres.append({"c": _vec(c), "r": _f(r)})

    bc, bhe = np.asarray(d["box_centers"]), np.asarray(d["box_half_extents"])
    for c, he in zip(bc.reshape(-1, 3), bhe.reshape(-1, 3)):
        boxes.append({"c": _vec(c), "he": _vec(he)})

    cc  = np.asarray(d["cylinder_centers"]).reshape(-1, 3)
    cax = np.asarray(d["cylinder_axes"]).reshape(-1, 3)
    chh = np.asarray(d["cylinder_hh"]).reshape(-1)
    cr  = np.asarray(d["cylinder_radii"]).reshape(-1)
    for c, ax, hh, r in zip(cc, cax, chh, cr):
        cylinders.append({"c": _vec(c), "ax": _vec(ax), "hh": _f(hh), "r": _f(r)})

    oc  = np.asarray(d["obb_centers"]).reshape(-1, 3)
    oq  = np.asarray(d["obb_quats"]).reshape(-1, 4)
    ohe = np.asarray(d["obb_half_extents"]).reshape(-1, 3)
    for c, q, he in zip(oc, oq, ohe):
        obbs.append({"c": _vec(c), "q": _vec(q), "he": _vec(he)})

    return {"spheres": spheres, "boxes": boxes, "cylinders": cylinders, "obbs": obbs}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_static(cfg: SceneConfig, seed: int) -> dict:
    arr = cfg.sample(jax.random.PRNGKey(seed))
    return {k: np.asarray(v) for k, v in cfg.unpack(arr).items()}


def sample_procedural(cfg: SceneConfig, seed: int, region) -> dict:
    """
    Bake the infinite world over a finite region.

    get_local_obstacles() returns the 3×3 cell neighbourhood around a position,
    in a fixed offset order whose index 4 is the centre cell. Walking the cells
    that cover `region` and keeping only each call's centre block reproduces the
    same obstacles the environment would generate, with no duplicates.
    """
    x_min, x_max, y_min, y_max = region
    cs = cfg.cell_size
    seed_f = np.float32(seed)

    per_cell = {
        "sphere":   cfg.spheres_per_cell + 2 * cfg.capsules_per_cell,
        "box":      cfg.boxes_per_cell + cfg.trees_per_cell,
        "cylinder": cfg.capsules_per_cell,
        "obb":      3 * cfg.trees_per_cell,
    }

    acc = {k: [] for k in ("sphere_centers", "sphere_radii", "box_centers",
                           "box_half_extents", "cylinder_centers", "cylinder_axes",
                           "cylinder_hh", "cylinder_radii", "obb_centers",
                           "obb_quats", "obb_half_extents")}

    ix0, ix1 = math.floor(x_min / cs), math.floor(x_max / cs)
    iy0, iy1 = math.floor(y_min / cs), math.floor(y_max / cs)
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            pos = np.array([(ix + 0.5) * cs, (iy + 0.5) * cs, -1.0], dtype=np.float32)
            (sph_c, sph_r, box_c, box_he, cap_c, cap_ax, cap_hh, cap_r,
             obb_c, obb_q, obb_he) = cfg.get_local_obstacles(pos, seed_f)

            def centre(arr, n):        # keep only the centre cell's block
                arr = np.asarray(arr)
                return arr[4 * n:5 * n] if n else arr[:0]

            acc["sphere_centers"].append(centre(sph_c, per_cell["sphere"]))
            acc["sphere_radii"].append(centre(sph_r, per_cell["sphere"]))
            acc["box_centers"].append(centre(box_c, per_cell["box"]))
            acc["box_half_extents"].append(centre(box_he, per_cell["box"]))
            acc["cylinder_centers"].append(centre(cap_c, per_cell["cylinder"]))
            acc["cylinder_axes"].append(centre(cap_ax, per_cell["cylinder"]))
            acc["cylinder_hh"].append(centre(cap_hh, per_cell["cylinder"]))
            acc["cylinder_radii"].append(centre(cap_r, per_cell["cylinder"]))
            acc["obb_centers"].append(centre(obb_c, per_cell["obb"]))
            acc["obb_quats"].append(centre(obb_q, per_cell["obb"]))
            acc["obb_half_extents"].append(centre(obb_he, per_cell["obb"]))

    out = {}
    for k, chunks in acc.items():
        chunks = [c for c in chunks if c.size]
        out[k] = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), np.float32)
    return out


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def camera_block(depth_camera: dict, pool: int) -> dict:
    """
    The camera the policy was trained with. `norm_*` mirror the constants
    hardcoded in NavigateReal._get_processed_depth:
        3.0 / clip(raw, 0.3, max_range) - 0.6
    """
    return {
        "fov_deg":        float(depth_camera.get("fov_deg", 90.0)),
        "width":          int(depth_camera.get("width", 64)),
        "height":         int(depth_camera.get("height", 48)),
        "min_range":      float(depth_camera.get("min_range", 0.2)),
        "max_range":      float(depth_camera.get("max_range", 8.0)),
        "quantization_m": float(depth_camera.get("quantization_m", 0.001)),
        "cam_hz":         float(depth_camera.get("cam_hz", 10.0)),
        "pool":           int(pool),
        "norm_numerator": 3.0,
        "norm_clip_min":  0.3,
        "norm_offset":    0.6,
    }


def load_hitl_config(path: str) -> dict:
    """
    Read a hitl/configs/*.yaml: pull the training config it names in `base`,
    then merge this file's `scene` / `depth_camera` blocks on top of it.

    Only the keys present here are overridden, so a HITL config stays a short
    statement of how the flight room differs from the training arena.
    """
    with open(path) as f:
        hitl = yaml.safe_load(f) or {}

    base_ref = hitl.get("base")
    if not base_ref:
        raise SystemExit(f"{path}: needs a `base:` pointing at a training config")
    base_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), base_ref))
    if not os.path.exists(base_path):
        raise SystemExit(f"{path}: base config '{base_ref}' not found at {base_path}")

    base = load_config(base_path)
    scene = dict(base["env"]["scene"])
    scene.update(hitl.get("scene") or {})
    camera = dict(base.get("depth_camera") or {})
    camera.update(hitl.get("depth_camera") or {})

    unknown = set(scene) - set(SceneConfig.__dataclass_fields__)
    if unknown:
        raise SystemExit(f"{path}: unknown scene key(s): {', '.join(sorted(unknown))}")

    return {
        "scene": scene,
        "depth_camera": camera,
        "export": hitl.get("export") or {},
        "base_path": base_path,
        "source": path,
    }


def report_density(base_scene: dict, scene: dict) -> None:
    """Obstacles per m³, before and after — the number that has to stay sane."""
    def stats(s):
        vol = ((s["arena_x_max"] - s["arena_x_min"])
               * (s["arena_y_max"] - s["arena_y_min"])
               * (s["arena_z_max"] - s["arena_z_min"]))
        n = s.get("n_spheres", 0) + s.get("n_boxes", 0) + s.get("n_capsules", 0)
        return vol, n, (n / vol if vol > 0 else 0.0)

    bv, bn, bd = stats(base_scene)
    nv, nn, nd = stats(scene)
    print(f"arena    training {bv:7.1f} m³ / {bn:3d} obstacles = {bd:.3f} per m³")
    print(f"         this run {nv:7.1f} m³ / {nn:3d} obstacles = {nd:.3f} per m³"
          f"   ({nd / bd:.2f}× as dense)" if bd > 0 else "")
    if bd > 0 and nd > 1.5 * bd:
        print("         NOTE: denser than training — the policy may not have seen "
              "clearances this tight")


def parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part[1:]:
            lo, hi = part.split("-", 1) if not part.startswith("-") else (part, part)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("scene source")
    src.add_argument("--hitl", help="HITL config (hitl/configs/*.yaml) — a training "
                                    "config plus the room-specific overrides")
    src.add_argument("--config", help="train config (configs/train/*.yaml) — supplies "
                                      "env.scene and depth_camera")
    src.add_argument("--scene", help="scene YAML (configs/scene/*.yaml), instead of --config")
    src.add_argument("--depth-camera", help="YAML/JSON file with a depth_camera block, "
                                            "when using --scene")

    ap.add_argument("--seed", type=int, help="episode seed")
    ap.add_argument("--seeds", help="seed list/range, e.g. 0-9 or 1,4,7 (needs --out-dir)")
    ap.add_argument("--out", help="output JSON path")
    ap.add_argument("--out-dir", help="output directory, one file per seed")
    ap.add_argument("--prefix", default="ep", help="filename prefix for --out-dir [ep]")
    ap.add_argument("--pool", type=int, default=4, help="max-pool factor [4]")
    ap.add_argument("--bake-region", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="procedural configs only: region to bake obstacles over")
    ap.add_argument("--no-ground", action="store_true",
                    help="drop the z=0 ground plane (it is part of the trained scene — "
                         "only use this if your flight volume has no visible floor)")
    args = ap.parse_args()

    # ---- resolve the scene + camera config ----
    export_defaults = {}
    base_scene = None
    if args.hitl:
        hitl         = load_hitl_config(args.hitl)
        scene_dict   = hitl["scene"]
        depth_camera = hitl["depth_camera"]
        export_defaults = hitl["export"]
        base_scene   = load_config(hitl["base_path"])["env"]["scene"]
        source_name  = f"{args.hitl} (base {os.path.relpath(hitl['base_path'])})"
    elif args.config:
        cfg_all      = load_config(args.config)
        scene_dict   = cfg_all["env"]["scene"]
        depth_camera = cfg_all.get("depth_camera", {})
        source_name  = args.config
    elif args.scene:
        with open(args.scene) as f:
            scene_dict = yaml.safe_load(f)
        depth_camera = {}
        if args.depth_camera:
            with open(args.depth_camera) as f:
                loaded = yaml.safe_load(f)
            depth_camera = loaded.get("depth_camera", loaded)
        source_name = args.scene
    else:
        ap.error("need --hitl, --config or --scene")

    if not isinstance(scene_dict, dict):
        ap.error("could not resolve the scene config to a dict")

    cam = camera_block(depth_camera, args.pool)
    cfg = SceneConfig(**scene_dict)
    if base_scene is not None:
        report_density(base_scene, scene_dict)
    if cfg.procedural:
        # Navigate.__init__ ties the cell size to the camera's max range.
        cfg.cell_size = cam["max_range"]
        if not args.bake_region:
            ap.error("this is a procedural scene config — pass --bake-region XMIN XMAX YMIN YMAX "
                     "to choose the volume to bake (cell size %.2f m)" % cfg.cell_size)
    print(cfg.summary())

    # ---- seeds and destinations ----
    # A HITL config's `export:` block supplies these; CLI flags still win.
    out_dir = args.out_dir or export_defaults.get("out_dir")
    prefix  = args.prefix if args.prefix != "ep" else export_defaults.get("prefix", "ep")

    if args.seeds:
        seeds = parse_seeds(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    elif export_defaults.get("seeds"):
        seeds = [int(s) for s in export_defaults["seeds"]]
    else:
        ap.error("need --seed or --seeds (or an export.seeds list in the HITL config)")

    if len(seeds) > 1 and not out_dir:
        ap.error("several seeds need --out-dir (or export.out_dir in the HITL config)")
    if not args.out and not out_dir:
        ap.error("need --out or --out-dir")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for seed in seeds:
        geom = (sample_procedural(cfg, seed, args.bake_region) if cfg.procedural
                else sample_static(cfg, seed))
        blocks = geometry_to_json(geom)

        path = (args.out if args.out and len(seeds) == 1
                else os.path.join(out_dir, f"{prefix}{seed:02d}.json"))
        name = os.path.splitext(os.path.basename(path))[0]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        doc = {
            "format": "jads-hitl-scene",
            "version": 1,
            "name": name,
            "source": {
                "config": source_name,
                "seed": seed,
                "mode": "procedural" if cfg.procedural else "static",
                "bake_region": list(args.bake_region) if args.bake_region else None,
                "cell_size": cfg.cell_size if cfg.procedural else None,
            },
            "ground_plane": not args.no_ground,
            "camera": cam,
            **blocks,
        }

        with open(path, "w") as f:
            json.dump(doc, f, indent=1)

        print(f"seed {seed:>4} → {path}   "
              f"{len(blocks['spheres'])} spheres, {len(blocks['boxes'])} boxes, "
              f"{len(blocks['cylinders'])} cylinders, {len(blocks['obbs'])} obbs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
