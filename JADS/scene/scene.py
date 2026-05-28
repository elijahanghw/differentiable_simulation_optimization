"""
scene.py — Random scene generation for the Navigate environment.

SceneConfig holds all obstacle parameters and exposes two JAX-compatible
methods that navigate.py calls from reset():

    scene_array          = cfg.sample(key)        → flat float32 JAX array
    scene_dict           = cfg.unpack(scene_array) → structured arrays

The flat layout inside scene_array (Nb_t = n_boxes + 4*n_windows):
    [0           : Ns*3        ]  sphere centres   (Ns, 3)
    [Ns*3        : Ns*4        ]  sphere radii     (Ns,)
    [Ns*4        : Ns*4+Nb_t*3 ]  box centres      (Nb_t, 3)  — regular boxes then window bars
    [Ns*4+Nb_t*3 : Ns*4+Nb_t*6 ]  box half-extents (Nb_t, 3)  — AABB only, no rotation
    [Ns*4+Nb_t*6 : ...]           capsule params   (Nc, 8)
        per entry: [cx, cy, cz, ax, ay, az, half_h, r]

Window obstacles (n_windows) are solid walls facing +x (drone forward) with a rectangular
hole cut out. Each is decomposed into 4 AABBs (top/bottom/left/right slabs) at sample time.
Parameterised by opening size (window_w, window_h), border width, and wall depth.

This layout matches depth_render.scene field-for-field so the renderer can consume
unpack() output directly.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field as _field

import jax
import jax.numpy as jnp


@dataclass
class SceneConfig:
    """
    All parameters controlling random obstacle generation.

    Obstacle counts (n_spheres, n_boxes, n_capsules) are specified directly.
    JAX array shapes are static for the lifetime of a training run.
    """

    # ---- Arena (world-space box where obstacles are spawned) ---------------
    arena_x_min: float = 0.0;  arena_x_max: float = 10.0
    arena_y_min: float = 0.0; arena_y_max: float = 10.0
    arena_z_min: float = -4.0; arena_z_max: float = 0.0

    # ---- Obstacle counts ---------------------------------------------------
    n_spheres:  int = 0
    n_boxes:    int = 0
    n_capsules: int = 0
    n_windows:  int = 0   # wall-with-window obstacles (each = 4 AABBs)

    # ---- Sphere size bounds ------------------------------------------------
    sphere_r_min: float = 0.15;  sphere_r_max: float = 0.60

    # ---- Box size bounds (half-extents per axis) ----------------------------
    box_hx_min: float = 0.10;  box_hx_max: float = 0.60
    box_hy_min: float = 0.10;  box_hy_max: float = 0.60
    box_hz_min: float = 0.10;  box_hz_max: float = 0.60

    # ---- Capsule size bounds -----------------------------------------------
    capsule_r_min:  float = 0.10;  capsule_r_max:  float = 0.40
    capsule_hh_min: float = 0.30;  capsule_hh_max: float = 1.20

    # ---- Window (wall-with-hole) obstacles — fully predefined geometry ------
    # Each window is a solid wall facing +x (drone forward) with a rectangular
    # hole. Positions are fixed; no randomness.
    #
    # YAML example:
    #   n_windows: 2
    #   window_positions: [[5.0, 0.0, -1.5], [8.0, -1.0, -2.0]]
    #   window_w: 1.0      # opening width  (y axis)
    #   window_h: 1.0      # opening height (z axis)
    #   window_border: 0.5 # solid border around the opening on each side
    #   window_depth: 0.2  # wall thickness (x axis)
    window_positions: list  = _field(default_factory=list)  # [[x,y,z], …] len n_windows
    window_w:         float = 0.8
    window_h:         float = 0.8
    window_border:    float = 0.5
    window_depth:     float = 0.2

    scene_dim: int = _field(init=False, repr=True)

    def __post_init__(self):
        Ns, Nb, Nc, Nw = self.n_spheres, self.n_boxes, self.n_capsules, self.n_windows
        self.scene_dim = Ns * 4 + (Nb + 4 * Nw) * 6 + Nc * 8

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """
        Sample a random scene, returning a flat JAX array of shape (scene_dim,).

        This is a pure JAX function: fully jittable and vmappable.
        Uses one key per obstacle group (shape=(N,) draws) so the key
        count is constant regardless of N.
        """
        Ns, Nb, Nc = self.n_spheres, self.n_boxes, self.n_capsules

        # 4 keys for spheres + 6 for boxes + 7 for capsules = 17 total
        k = jax.random.split(key, 17)
        i = 0

        # ---- Spheres -------------------------------------------------------
        s_cx = jax.random.uniform(k[i], shape=(Ns,), minval=self.arena_x_min, maxval=self.arena_x_max); i+=1
        s_cy = jax.random.uniform(k[i], shape=(Ns,), minval=self.arena_y_min, maxval=self.arena_y_max); i+=1
        s_cz = jax.random.uniform(k[i], shape=(Ns,), minval=self.arena_z_min, maxval=self.arena_z_max); i+=1
        s_r  = jax.random.uniform(k[i], shape=(Ns,), minval=self.sphere_r_min, maxval=self.sphere_r_max); i+=1

        sphere_centers = jnp.stack([s_cx, s_cy, s_cz], axis=-1).reshape(-1)  # (Ns*3,)
        sphere_radii   = s_r                                                   # (Ns,)

        # ---- Boxes ---------------------------------------------------------
        b_cx = jax.random.uniform(k[i], shape=(Nb,), minval=self.arena_x_min, maxval=self.arena_x_max); i+=1
        b_cy = jax.random.uniform(k[i], shape=(Nb,), minval=self.arena_y_min, maxval=self.arena_y_max); i+=1
        b_cz = jax.random.uniform(k[i], shape=(Nb,), minval=self.arena_z_min, maxval=self.arena_z_max); i+=1

        b_hx = jax.random.uniform(k[i], shape=(Nb,), minval=self.box_hx_min,  maxval=self.box_hx_max);  i+=1
        b_hy = jax.random.uniform(k[i], shape=(Nb,), minval=self.box_hy_min,  maxval=self.box_hy_max);  i+=1
        b_hz = jax.random.uniform(k[i], shape=(Nb,), minval=self.box_hz_min,  maxval=self.box_hz_max);  i+=1

        box_centers      = jnp.stack([b_cx, b_cy, b_cz], axis=-1).reshape(-1)  # (Nb*3,)
        box_half_extents = jnp.stack([b_hx, b_hy, b_hz], axis=-1).reshape(-1)  # (Nb*3,)


        # ---- Capsules -------------------------------------------------------
        c_cx = jax.random.uniform(k[i], shape=(Nc,), minval=self.arena_x_min, maxval=self.arena_x_max); i+=1
        c_cy = jax.random.uniform(k[i], shape=(Nc,), minval=self.arena_y_min, maxval=self.arena_y_max); i+=1
        c_cz = jax.random.uniform(k[i], shape=(Nc,), minval=self.arena_z_min, maxval=self.arena_z_max); i+=1

        # Random unit axis via spherical coordinates (uniform on S²)
        theta = jax.random.uniform(k[i], shape=(Nc,), minval=0.0, maxval=math.pi); i+=1
        phi   = jax.random.uniform(k[i], shape=(Nc,), minval=0.0, maxval=2*math.pi); i+=1

        ax = jnp.sin(theta) * jnp.cos(phi)
        ay = jnp.sin(theta) * jnp.sin(phi)
        az = jnp.cos(theta)
        axes = jnp.stack([ax, ay, az], axis=-1)  # (Nc, 3), unit vectors

        c_hh = jax.random.uniform(k[i], shape=(Nc,), minval=self.capsule_hh_min, maxval=self.capsule_hh_max); i+=1
        c_r  = jax.random.uniform(k[i], shape=(Nc,), minval=self.capsule_r_min,  maxval=self.capsule_r_max); i+=1

        # Pack capsule params row-wise: (Nc, 8) → flat (Nc*8,)
        capsule_params = jnp.concatenate([
            jnp.stack([c_cx, c_cy, c_cz], axis=-1),  # (Nc, 3) centers
            axes,                                      # (Nc, 3) unit axes
            c_hh[:, None],                             # (Nc, 1) half-heights
            c_r[:, None],                              # (Nc, 1) radii
        ], axis=-1).reshape(-1)  # (Nc*8,)

        # ---- Windows (fixed positions, decomposed into 4 AABBs each) ----------
        Nw = self.n_windows
        if Nw > 0:
            w_pos = jnp.array(self.window_positions, dtype=jnp.float32)  # (Nw, 3)
            w_cx, w_cy, w_cz = w_pos[:, 0], w_pos[:, 1], w_pos[:, 2]

            iw  = jnp.full((Nw,), self.window_w)
            ih  = jnp.full((Nw,), self.window_h)
            brd = jnp.full((Nw,), self.window_border)
            d   = jnp.full((Nw,), self.window_depth)

            hd      = d * 0.5           # half depth (x)
            hbrd    = brd * 0.5         # half border (y/z)
            wall_hy = iw * 0.5 + brd    # half total wall width (y), covers full wall

            # 4 bars: top, bottom, left, right
            bars_cx = jnp.concatenate([w_cx,               w_cx,               w_cx,                    w_cx])
            bars_cy = jnp.concatenate([w_cy,               w_cy,               w_cy - iw*0.5 - hbrd,    w_cy + iw*0.5 + hbrd])
            bars_cz = jnp.concatenate([w_cz - ih*0.5 - hbrd, w_cz + ih*0.5 + hbrd, w_cz,              w_cz])
            bars_hx = jnp.concatenate([hd,                 hd,                 hd,                      hd])
            bars_hy = jnp.concatenate([wall_hy,            wall_hy,            hbrd,                    hbrd])
            bars_hz = jnp.concatenate([hbrd,               hbrd,               ih*0.5,                  ih*0.5])

            win_centers      = jnp.stack([bars_cx, bars_cy, bars_cz], axis=-1).reshape(-1)
            win_half_extents = jnp.stack([bars_hx, bars_hy, bars_hz], axis=-1).reshape(-1)
        else:
            win_centers = win_half_extents = jnp.zeros(0)

        return jnp.concatenate([
            sphere_centers,
            sphere_radii,
            box_centers,      win_centers,
            box_half_extents, win_half_extents,
            capsule_params,
        ])

    # -----------------------------------------------------------------------
    # Unpacking
    # -----------------------------------------------------------------------

    def unpack(self, scene_array: jnp.ndarray) -> dict:
        """
        Split a flat scene array into named geometry arrays.

        Compatible with depth_render.scene field names so the renderer
        can consume this output directly.

        Returns:
            sphere_centers    (Ns, 3)
            sphere_radii      (Ns,)
            box_centers       (Nb, 3)
            box_half_extents  (Nb, 3)  — axis-aligned boxes (no rotation)
            capsule_centers   (Nc, 3)
            capsule_axes      (Nc, 3)   unit axes
            capsule_hh        (Nc,)     half-heights
            capsule_radii     (Nc,)
        """
        Ns   = self.n_spheres
        Nb_t = self.n_boxes + 4 * self.n_windows
        Nc   = self.n_capsules
        i = 0

        sphere_centers   = scene_array[i : i+Ns*3].reshape(Ns, 3);      i += Ns*3
        sphere_radii     = scene_array[i : i+Ns];                        i += Ns
        box_centers      = scene_array[i : i+Nb_t*3].reshape(Nb_t, 3);  i += Nb_t*3
        box_half_extents = scene_array[i : i+Nb_t*3].reshape(Nb_t, 3);  i += Nb_t*3
        capsule_flat     = scene_array[i : i+Nc*8].reshape(Nc, 8)

        return {
            "sphere_centers":   sphere_centers,
            "sphere_radii":     sphere_radii,
            "box_centers":      box_centers,
            "box_half_extents": box_half_extents,
            "capsule_centers":  capsule_flat[:, 0:3],
            "capsule_axes":     capsule_flat[:, 3:6],
            "capsule_hh":       capsule_flat[:, 6],
            "capsule_radii":    capsule_flat[:, 7],
        }

    def summary(self) -> str:
        vol = (
            (self.arena_x_max - self.arena_x_min)
            * (self.arena_y_max - self.arena_y_min)
            * (self.arena_z_max - self.arena_z_min)
        )
        return (
            f"SceneConfig  arena={vol:.1f}m³  "
            f"Ns={self.n_spheres}  Nb={self.n_boxes}  Nc={self.n_capsules}  "
            f"scene_dim={self.scene_dim}"
        )
