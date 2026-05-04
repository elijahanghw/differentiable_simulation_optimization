"""
test_scene_render.py — Visualise one random Navigate scene.

Renders three panels:
  Left   : 3-D scene overview (drone, target, all obstacles)
  Top-right  : depth image from the drone's perspective (looking toward target)
  Bot-right  : top-down 2-D map (XY plane)

Usage:
    python test_scene_render.py                        # default navigate.yaml
    python test_scene_render.py --config configs/navigate.yaml
    python test_scene_render.py --seed 7
    python test_scene_render.py --save scene.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import jax
import jax.numpy as jnp
import yaml

# Make depth_render importable (it uses bare module-level imports)
_REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
_DEPTH_DIR  = os.path.join(_REPO_ROOT, "depth_render")
sys.path.insert(0, _DEPTH_DIR)

import scene as dr_scene    # depth_render/scene.py  (Scene / Sphere / Box / Panel)

from envs.multicopter.navigate import Navigate
from envs.multicopter.scene    import SceneConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_navigate_env(config_path: str) -> Navigate:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ecfg = cfg["env"]
    kwargs = {k: v for k, v in ecfg.items() if k != "name"}
    return Navigate(**kwargs, depth_camera=cfg.get("depth_camera"))


def scene_arrays_to_dr_scene(arrays: dict) -> dr_scene.Scene:
    """Convert get_scene_arrays() output → depth_render Scene object."""
    s = dr_scene.Scene()
    Ns = arrays["sphere_centers"].shape[0]
    Nb = arrays["box_centers"].shape[0]
    Nc = arrays["capsule_centers"].shape[0]

    for i in range(Ns):
        s.add_sphere(
            center=tuple(float(x) for x in arrays["sphere_centers"][i]),
            radius=float(arrays["sphere_radii"][i]),
        )
    for i in range(Nb):
        s.add_box(
            center      =tuple(float(x) for x in arrays["box_centers"][i]),
            quaternion  =tuple(float(x) for x in arrays["box_quaternions"][i]),
            half_extents=tuple(float(x) for x in arrays["box_half_extents"][i]),
        )
    for i in range(Nc):
        s.add_capsule(
            center=tuple(float(x) for x in arrays["capsule_centers"][i]),
            axis  =tuple(float(x) for x in arrays["capsule_axes"][i]),
            half_h=float(arrays["capsule_hh"][i]),
            radius=float(arrays["capsule_radii"][i]),
        )
    return s


def depth_to_display(depth_np: np.ndarray, max_range: float = 15.0) -> np.ndarray:
    """Normalise depth to [0, 1] where 0 = blind-zone/dark, 1 = far/no-hit (bright).

    Input convention (post-noise): 0 = closer than min_range (blind zone),
    positive = metres, max_range = no-hit or saturated.
    """
    return np.clip(depth_np, 0.0, max_range) / max_range


# ---------------------------------------------------------------------------
# 3-D scene drawing helpers (same style as depth_render/demo.py)
# ---------------------------------------------------------------------------

def _sphere_rings(ax, center, radius, color, n=40):
    t  = np.linspace(0, 2*np.pi, n)
    cx, cy, cz = center
    ax.plot(cx + radius*np.cos(t), cy + radius*np.sin(t), np.full(n, cz), color=color, lw=0.8, alpha=0.7)
    ax.plot(cx + radius*np.cos(t), np.full(n, cy), cz + radius*np.sin(t), color=color, lw=0.8, alpha=0.7)
    ax.plot(np.full(n, cx), cy + radius*np.cos(t), cz + radius*np.sin(t), color=color, lw=0.8, alpha=0.7)


def _quat_to_rot(q):
    """[qw, qx, qy, qz] → 3×3 rotation matrix (numpy)."""
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])


def _box_wireframe(ax, center, quaternion, half_extents, color):
    cx, cy, cz = center
    hx, hy, hz = half_extents
    R = _quat_to_rot(quaternion)
    # 8 local-frame corners
    local = np.array([
        [-hx, -hy, -hz], [+hx, -hy, -hz],
        [+hx, +hy, -hz], [-hx, +hy, -hz],
        [-hx, -hy, +hz], [+hx, -hy, +hz],
        [+hx, +hy, +hz], [-hx, +hy, +hz],
    ])
    v = (R @ local.T).T + np.array([cx, cy, cz])
    for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        ax.plot([v[a,0],v[b,0]], [v[a,1],v[b,1]], [v[a,2],v[b,2]], color=color, lw=0.9)


def _panel_outline(ax, center, normal, tangent, half_w, half_h, color):
    c = np.array(center, float)
    n = np.array(normal,  float); n /= np.linalg.norm(n)
    t = np.array(tangent, float); t /= np.linalg.norm(t)
    b = np.cross(n, t);           b /= np.linalg.norm(b)
    corners = [c+half_w*t+half_h*b, c-half_w*t+half_h*b,
               c-half_w*t-half_h*b, c+half_w*t-half_h*b,
               c+half_w*t+half_h*b]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    zs = [p[2] for p in corners]
    ax.plot(xs, ys, zs, color=color, lw=1.2, alpha=0.85)


def _capsule_3d(ax, center, axis, half_h, radius, color, n=24):
    """Draw a capsule as rings along its axis plus end-cap circles."""
    c   = np.array(center, float)
    axs = np.array(axis,   float); axs /= np.linalg.norm(axs)
    A   = c - half_h * axs
    B   = c + half_h * axs

    # Build an orthonormal frame around the axis
    perp = np.array([1., 0., 0.]) if abs(axs[0]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(axs, perp); u /= np.linalg.norm(u)
    v = np.cross(axs, u)

    t = np.linspace(0, 2 * np.pi, n)
    ring = radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))

    # Cylinder body: two end rings + 4 side lines
    for end in (A, B):
        pts = end + ring
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=0.8, alpha=0.7)
    for j in range(0, n, n // 4):
        ax.plot([A[0]+ring[j,0], B[0]+ring[j,0]],
                [A[1]+ring[j,1], B[1]+ring[j,1]],
                [A[2]+ring[j,2], B[2]+ring[j,2]], color=color, lw=0.8, alpha=0.7)


def draw_3d_scene(ax, arrays, drone_pos, target_pos, cfg: SceneConfig):
    ax.set_facecolor("#0d0d0d")

    Ns = arrays["sphere_centers"].shape[0]
    Nb = arrays["box_centers"].shape[0]
    Nc = arrays["capsule_centers"].shape[0]

    # Ground plane at Z=0 (Z-down positive: Z=0 is the ground surface)
    gx = np.linspace(0, 10, 2)
    gy = np.linspace(0, 10, 2)
    gxx, gyy = np.meshgrid(gx, gy)
    gzz = np.zeros_like(gxx)
    ax.plot_surface(gxx, gyy, gzz, color="#dddddd", alpha=0.35, zorder=0, linewidth=0)

    for i in range(Ns):
        _sphere_rings(ax, arrays["sphere_centers"][i], float(arrays["sphere_radii"][i]), "#4fc3f7")
    for i in range(Nb):
        _box_wireframe(ax, arrays["box_centers"][i], arrays["box_quaternions"][i], arrays["box_half_extents"][i], "#ffb74d")
    for i in range(Nc):
        _capsule_3d(ax, arrays["capsule_centers"][i], arrays["capsule_axes"][i],
                    float(arrays["capsule_hh"][i]), float(arrays["capsule_radii"][i]), "#a5d6a7")

    ax.scatter(*drone_pos,  color="#ff4444", s=80, zorder=10, depthshade=False)
    ax.scatter(*target_pos, color="#00e676", s=80, marker="*", zorder=10, depthshade=False)
    ax.plot([drone_pos[0], target_pos[0]],
            [drone_pos[1], target_pos[1]],
            [drone_pos[2], target_pos[2]], "--", color="#ffffff", lw=0.8, alpha=0.4)

    ax.set_xlabel("X", color="#888", fontsize=7, labelpad=2)
    ax.set_ylabel("Y", color="#888", fontsize=7, labelpad=2)
    ax.set_zlabel("Z (↓)", color="#888", fontsize=7, labelpad=2)
    ax.invert_yaxis()
    ax.invert_zaxis()   # Z-down positive: larger Z values appear lower
    ax.tick_params(colors="#555", labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#2a2a2a")


def draw_2d_topdown(ax, arrays, drone_pos, target_pos, cfg: SceneConfig):
    """Top-down XY map."""
    ax.set_facecolor("#0d0d0d")

    Ns = arrays["sphere_centers"].shape[0]
    Nb = arrays["box_centers"].shape[0]
    Nc = arrays["capsule_centers"].shape[0]

    # Spheres → circles
    for i in range(Ns):
        c = arrays["sphere_centers"][i]
        r = float(arrays["sphere_radii"][i])
        circle = plt.Circle((c[0], c[1]), r, color="#4fc3f7", fill=True, alpha=0.3)
        ax.add_patch(circle)
        circle2 = plt.Circle((c[0], c[1]), r, color="#4fc3f7", fill=False, lw=0.8)
        ax.add_patch(circle2)

    # Boxes → rotated rectangles (XY projection of 8 OBB corners)
    for i in range(Nb):
        c  = arrays["box_centers"][i]
        q  = arrays["box_quaternions"][i]
        he = arrays["box_half_extents"][i]
        R  = _quat_to_rot(q)
        hx, hy, hz = he
        local = np.array([
            [-hx,-hy,-hz],[+hx,-hy,-hz],[+hx,+hy,-hz],[-hx,+hy,-hz],
            [-hx,-hy,+hz],[+hx,-hy,+hz],[+hx,+hy,+hz],[-hx,+hy,+hz],
        ])
        corners_3d = (R @ local.T).T + c
        xy = corners_3d[:, :2]
        # Convex hull of XY projections
        from matplotlib.patches import Polygon as MplPolygon
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(xy)
            hull_pts = xy[hull.vertices]
            poly = MplPolygon(hull_pts, closed=True, color="#ffb74d", fill=True, alpha=0.3, lw=0.8, edgecolor="#ffb74d")
        except Exception:
            poly = MplPolygon(xy, closed=True, color="#ffb74d", fill=True, alpha=0.3, lw=0.8, edgecolor="#ffb74d")
        ax.add_patch(poly)

    # Capsules → line segment (XY projection of axis) with end circles
    for i in range(Nc):
        c   = arrays["capsule_centers"][i]
        axs = arrays["capsule_axes"][i]; axs = axs / (np.linalg.norm(axs) + 1e-8)
        hh  = float(arrays["capsule_hh"][i])
        r   = float(arrays["capsule_radii"][i])
        A   = c[:2] - hh * axs[:2]
        B   = c[:2] + hh * axs[:2]
        ax.plot([A[0], B[0]], [A[1], B[1]], color="#a5d6a7", lw=2.0, solid_capstyle="round")
        ax.add_patch(plt.Circle((A[0], A[1]), r, color="#a5d6a7", fill=False, lw=0.8, alpha=0.6))
        ax.add_patch(plt.Circle((B[0], B[1]), r, color="#a5d6a7", fill=False, lw=0.8, alpha=0.6))

    ax.scatter(drone_pos[0],  drone_pos[1],  color="#ff4444", s=80, zorder=5)
    ax.scatter(target_pos[0], target_pos[1], color="#00e676", s=80, marker="*", zorder=5)
    ax.plot([drone_pos[0], target_pos[0]], [drone_pos[1], target_pos[1]],
            "--", color="#ffffff", lw=0.8, alpha=0.4)

    # Arena outline
    ax.add_patch(plt.Rectangle(
        (cfg.arena_x_min, cfg.arena_y_min),
        cfg.arena_x_max - cfg.arena_x_min,
        cfg.arena_y_max - cfg.arena_y_min,
        fill=False, edgecolor="#555", lw=0.8, linestyle=":",
    ))

    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.set_xlabel("X", color="#888", fontsize=7)
    ax.set_ylabel("Y", color="#888", fontsize=7)
    ax.tick_params(colors="#555", labelsize=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/navigate.yaml")
    parser.add_argument("--seed",   type=int, default=None)
    parser.add_argument("--save",   default="test_scene_render.png")
    args = parser.parse_args()

    # ---- Build env & sample one episode -----------------------------------
    env = load_navigate_env(args.config)
    print(env.scene_cfg.summary())

    seed = args.seed if args.seed is not None else np.random.randint(0, 2**31 - 1)
    print(f"Seed   : {seed}")
    key         = jax.random.PRNGKey(seed)
    obs, state, _ = env.reset(key)

    drone_pos  = np.array(state[0:3])
    target_pos = np.array(state[19:22])
    arrays     = {k: np.array(v) for k, v in env.get_scene_arrays(state).items()}

    print(f"Drone  : {drone_pos}")
    print(f"Target : {target_pos}")
    print(f"Dist   : {np.linalg.norm(target_pos - drone_pos):.2f} m")
    print(f"Obstacles: Ns={arrays['sphere_centers'].shape[0]}  "
          f"Nb={arrays['box_centers'].shape[0]}  "
          f"Nc={arrays['capsule_centers'].shape[0]}")

    # ---- Build depth_render Scene (for 3-D visualisation only) ------------
    dr_scene_obj = scene_arrays_to_dr_scene(arrays)
    dr_scene_obj.add_plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, -1.0), label="ground")
    print(dr_scene_obj.summary())

    # ---- Depth image via env.get_depth ------------------------------------
    print(f"\nRendering {env.cam_width}×{env.cam_height} depth image …")
    depth_np = np.array(env.get_depth(state))
    valid = depth_np > 0.0
    if valid.any():
        print(f"  valid {valid.mean()*100:.1f}% of pixels  "
              f"depth range [{depth_np[valid].min():.2f}, {depth_np[valid].max():.2f}] m")
    else:
        print("  no valid pixels (empty scene or all out of range)")

    # ---- Figure layout -------------------------------------------------------
    #
    #   ┌─────────────────────┬─────────────────┐
    #   │                     │  depth image    │
    #   │   3-D scene view    ├─────────────────┤
    #   │                     │  top-down map   │
    #   └─────────────────────┴─────────────────┘

    fig = plt.figure(figsize=(14, 7), facecolor="#111111")
    fig.suptitle(
        f"Navigate scene  •  seed={seed}  •  "
        f"{env.scene_cfg.n_spheres}S / {env.scene_cfg.n_boxes}B / {env.scene_cfg.n_capsules}C obstacles  •  "
        f"density={env.scene_cfg.obstacle_density}/m³",
        color="white", fontsize=11, y=0.98,
    )

    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.08,
                              left=0.03, right=0.97, top=0.93, bottom=0.05,
                              width_ratios=[1.4, 1])
    right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.35)

    # 3-D overview
    ax3d = fig.add_subplot(outer[0], projection="3d")
    ax3d.set_facecolor("#0d0d0d")
    draw_3d_scene(ax3d, arrays, drone_pos, target_pos, env.scene_cfg)
    # Orient 3D view to match depth camera direction:
    # look from *behind* the drone toward the target so left/right
    # match the depth image (instead of the default which looks
    # from roughly the opposite side and appears left-right mirrored).
    _dx = float(drone_pos[0] - target_pos[0])
    _dy = float(drone_pos[1] - target_pos[1])
    ax3d.view_init(elev=20, azim=float(np.degrees(np.arctan2(_dy, _dx))))
    ax3d.set_title("3-D scene overview  (Z↓ positive)", color="white", fontsize=9, pad=4)
    legend_elems = [
        mpatches.Patch(color="#4fc3f7", label="sphere"),
        mpatches.Patch(color="#ffb74d", label="box"),
        mpatches.Patch(color="#a5d6a7", label="capsule"),
        mpatches.Patch(color="#888888", label="ground (Z=0)"),
        mpatches.Patch(color="#ff4444", label="drone"),
        mpatches.Patch(color="#00e676", label="target"),
    ]
    ax3d.legend(handles=legend_elems, fontsize=7, loc="upper left",
                facecolor="#1a1a1a", edgecolor="#444", labelcolor="white", framealpha=0.8)

    # Depth image
    ax_depth = fig.add_subplot(right_gs[0])
    ax_depth.set_facecolor("#000000")
    disp = depth_to_display(depth_np, max_range=env.cam_max_range)
    im = ax_depth.imshow(disp, cmap="inferno", vmin=0, vmax=1,
                         aspect="auto", interpolation="nearest")
    ax_depth.set_title(
        f"Depth  •  FOV {env.cam_fov_deg:.0f}°  •  drone→target  •  max {env.cam_max_range:.0f} m",
        color="white", fontsize=9, pad=4,
    )
    ax_depth.tick_params(colors="#555", labelsize=6)
    cbar = fig.colorbar(im, ax=ax_depth, fraction=0.04, pad=0.02)
    n_ticks = 5
    tick_vals = np.linspace(0, 1, n_ticks)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{v * env.cam_max_range:.1f} m" for v in tick_vals])
    cbar.ax.tick_params(colors="#888", labelsize=6)
    cbar.outline.set_edgecolor("#444")

    # Top-down map
    ax2d = fig.add_subplot(right_gs[1])
    draw_2d_topdown(ax2d, arrays, drone_pos, target_pos, env.scene_cfg)
    ax2d.set_title("Top-down map (XY)", color="white", fontsize=9, pad=4)
    ax2d.set_facecolor("#0d0d0d")
    ax2d.tick_params(colors="#555", labelsize=6)
    for sp in ax2d.spines.values():
        sp.set_edgecolor("#333")

    out = args.save
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved → {out}")

    print(depth_np)


if __name__ == "__main__":
    main()
