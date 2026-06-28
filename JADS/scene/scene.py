"""
scene.py — Scene generation for the Navigate environment.

Two modes, selected by the YAML config:

  STATIC mode  (n_spheres / n_boxes / n_capsules / n_windows > 0)
    scene_array = cfg.sample(key)      → flat float32 array of shape (scene_dim,)
    scene_dict  = cfg.unpack(arr)      → structured arrays for the renderer

    The flat layout (Ns_t = n_spheres + 2*n_capsules, Nb_t = n_boxes + 4*n_windows):
      [0              : Ns_t*3         ]  sphere centres   (Ns_t, 3)
      [Ns_t*3         : Ns_t*4         ]  sphere radii     (Ns_t,)
      [Ns_t*4         : Ns_t*4+Nb_t*3  ]  box centres      (Nb_t, 3)
      [Ns_t*4+Nb_t*3  : Ns_t*4+Nb_t*6  ]  box half-extents (Nb_t, 3)
      [Ns_t*4+Nb_t*6  : ...]              cylinder params  (Nc, 8)

  PROCEDURAL mode  (obstacles_per_cell > 0)
    Obstacles are generated on-the-fly each step from the drone position and
    a per-episode seed.  The world is infinite: the same (seed, cell) always
    produces the same obstacles, so the scene is consistent across timesteps.

    scene_array = cfg.sample(key)      → shape (1,) — just the float32 seed
    box_c, box_he = cfg.get_local_obstacles(drone_pos, seed_float)
        → (K, 3), (K, 3)  where K = 9 × obstacles_per_cell

    Cell size is set to cam_max_range by Navigate.__init__ after construction.
    A 3×3 grid of cells centered on the drone's current cell is loaded;
    this always covers the full camera frustum regardless of drone position
    within its cell.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field as _field

import jax
import jax.numpy as jnp


@dataclass
class SceneConfig:
    """
    All parameters controlling obstacle generation.

    Set obstacles_per_cell > 0 to enable procedural infinite-world mode.
    Otherwise, set n_spheres / n_boxes / n_capsules / n_windows for the
    legacy static-scene mode.
    """

    # ---- Procedural mode ---------------------------------------------------
    obstacles_per_cell: int   = 0     # >0 → procedural mode
    cell_size:          float = 8.0   # set to cam_max_range by Navigate.__init__

    # ---- Arena (static mode spawn bounds / procedural z bounds) ------------
    arena_x_min: float = 0.0;  arena_x_max: float = 10.0
    arena_y_min: float = 0.0;  arena_y_max: float = 10.0
    arena_z_min: float = -4.0; arena_z_max: float = 0.0

    # ---- Static-mode obstacle counts ---------------------------------------
    n_spheres:  int = 0
    n_boxes:    int = 0
    n_capsules: int = 0
    n_windows:  int = 0

    # ---- Sphere size bounds ------------------------------------------------
    sphere_r_min: float = 0.15;  sphere_r_max: float = 0.60

    # ---- Box size bounds (half-extents per axis) ----------------------------
    box_hx_min: float = 0.10;  box_hx_max: float = 0.60
    box_hy_min: float = 0.10;  box_hy_max: float = 0.60
    box_hz_min: float = 0.10;  box_hz_max: float = 0.60

    # ---- Capsule size bounds -----------------------------------------------
    capsule_r_min:  float = 0.10;  capsule_r_max:  float = 0.40
    capsule_hh_min: float = 0.30;  capsule_hh_max: float = 1.20

    # ---- Window obstacles --------------------------------------------------
    window_positions: list  = _field(default_factory=list)
    window_w:         float = 0.8
    window_h:         float = 0.8
    window_border:    float = 0.5
    window_depth:     float = 0.2

    scene_dim: int = _field(init=False, repr=True)
    procedural: bool = _field(init=False, repr=True)

    def __post_init__(self):
        self.procedural = self.obstacles_per_cell > 0
        if self.procedural:
            self.scene_dim = 1  # only the seed float
        else:
            Ns, Nb, Nc, Nw = self.n_spheres, self.n_boxes, self.n_capsules, self.n_windows
            Ns_t = Ns + 2 * Nc
            self.scene_dim = Ns_t * 4 + (Nb + 4 * Nw) * 6 + Nc * 8

    # -----------------------------------------------------------------------
    # Procedural mode
    # -----------------------------------------------------------------------

    def get_local_obstacles(
        self,
        drone_pos:  jnp.ndarray,  # (3,)  world-space position
        seed_float: jnp.ndarray,  # ()    float32 episode seed
    ):
        """
        Generate K = 9 × obstacles_per_cell boxes for the 3×3 cell
        neighbourhood centred on the drone's current grid cell.

        The same (seed, cell_ix, cell_iy) always produces identical obstacles,
        so the scene is persistent across timesteps without storing positions.

        Returns:
            box_centers      (K, 3) float32
            box_half_extents (K, 3) float32
        """
        M         = self.obstacles_per_cell
        cell_size = self.cell_size

        ix_c = jnp.floor(drone_pos[0] / cell_size).astype(jnp.int32)
        iy_c = jnp.floor(drone_pos[1] / cell_size).astype(jnp.int32)

        # Cast seed back to uint32 for PRNGKey (stored as float32 int in [0, 2^23))
        base_key = jax.random.PRNGKey(jnp.uint32(jnp.int32(seed_float)))

        # 3×3 neighbourhood offsets
        dx = jnp.array([-1, -1, -1,  0,  0,  0,  1,  1,  1], dtype=jnp.int32)
        dy = jnp.array([-1,  0,  1, -1,  0,  1, -1,  0,  1], dtype=jnp.int32)
        offsets = jnp.stack([dx, dy], axis=-1)  # (9, 2)

        def sample_cell(offset):
            ix = ix_c + offset[0]
            iy = iy_c + offset[1]
            # fold_in accepts int32; negative values wrap to large uint32, still unique
            cell_key = jax.random.fold_in(jax.random.fold_in(base_key, ix), iy)
            k1, k2, k3, k4, k5, k6 = jax.random.split(cell_key, 6)

            cx = jax.random.uniform(
                k1, (M,),
                minval=ix.astype(jnp.float32) * cell_size,
                maxval=(ix + 1).astype(jnp.float32) * cell_size,
            )
            cy = jax.random.uniform(
                k2, (M,),
                minval=iy.astype(jnp.float32) * cell_size,
                maxval=(iy + 1).astype(jnp.float32) * cell_size,
            )
            cz = jax.random.uniform(k3, (M,), minval=self.arena_z_min, maxval=self.arena_z_max)
            hx = jax.random.uniform(k4, (M,), minval=self.box_hx_min,  maxval=self.box_hx_max)
            hy = jax.random.uniform(k5, (M,), minval=self.box_hy_min,  maxval=self.box_hy_max)
            hz = jax.random.uniform(k6, (M,), minval=self.box_hz_min,  maxval=self.box_hz_max)

            centers      = jnp.stack([cx, cy, cz], axis=-1)  # (M, 3)
            half_extents = jnp.stack([hx, hy, hz], axis=-1)  # (M, 3)
            return centers, half_extents

        centers, half_extents = jax.vmap(sample_cell)(offsets)  # (9, M, 3)
        return centers.reshape(9 * M, 3), half_extents.reshape(9 * M, 3)

    # -----------------------------------------------------------------------
    # Static mode — sampling
    # -----------------------------------------------------------------------

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """
        Procedural mode: returns a (1,) float32 array holding the episode seed.
        Static mode:     returns a flat (scene_dim,) float32 array of obstacle geometry.
        """
        if self.procedural:
            seed_int = jax.random.randint(key, shape=(), minval=0, maxval=2**23, dtype=jnp.int32)
            return jnp.array([seed_int], dtype=jnp.float32)

        Ns, Nb, Nc = self.n_spheres, self.n_boxes, self.n_capsules

        k = jax.random.split(key, 17)
        i = 0

        # ---- Spheres -------------------------------------------------------
        s_cx = jax.random.uniform(k[i], shape=(Ns,), minval=self.arena_x_min, maxval=self.arena_x_max); i+=1
        s_cy = jax.random.uniform(k[i], shape=(Ns,), minval=self.arena_y_min, maxval=self.arena_y_max); i+=1
        s_cz = jax.random.uniform(k[i], shape=(Ns,), minval=self.arena_z_min, maxval=self.arena_z_max); i+=1
        s_r  = jax.random.uniform(k[i], shape=(Ns,), minval=self.sphere_r_min, maxval=self.sphere_r_max); i+=1

        sphere_centers = jnp.stack([s_cx, s_cy, s_cz], axis=-1)
        sphere_radii   = s_r

        # ---- Boxes ---------------------------------------------------------
        b_cx = jax.random.uniform(k[i], shape=(Nb,), minval=self.arena_x_min, maxval=self.arena_x_max); i+=1
        b_cy = jax.random.uniform(k[i], shape=(Nb,), minval=self.arena_y_min, maxval=self.arena_y_max); i+=1
        b_cz = jax.random.uniform(k[i], shape=(Nb,), minval=self.arena_z_min, maxval=self.arena_z_max); i+=1
        b_hx = jax.random.uniform(k[i], shape=(Nb,), minval=self.box_hx_min,  maxval=self.box_hx_max);  i+=1
        b_hy = jax.random.uniform(k[i], shape=(Nb,), minval=self.box_hy_min,  maxval=self.box_hy_max);  i+=1
        b_hz = jax.random.uniform(k[i], shape=(Nb,), minval=self.box_hz_min,  maxval=self.box_hz_max);  i+=1

        box_centers      = jnp.stack([b_cx, b_cy, b_cz], axis=-1).reshape(-1)
        box_half_extents = jnp.stack([b_hx, b_hy, b_hz], axis=-1).reshape(-1)

        # ---- Capsules -------------------------------------------------------
        c_cx = jax.random.uniform(k[i], shape=(Nc,), minval=self.arena_x_min, maxval=self.arena_x_max); i+=1
        c_cy = jax.random.uniform(k[i], shape=(Nc,), minval=self.arena_y_min, maxval=self.arena_y_max); i+=1
        c_cz = jax.random.uniform(k[i], shape=(Nc,), minval=self.arena_z_min, maxval=self.arena_z_max); i+=1

        theta = jax.random.uniform(k[i], shape=(Nc,), minval=0.0, maxval=math.pi); i+=1
        phi   = jax.random.uniform(k[i], shape=(Nc,), minval=0.0, maxval=2*math.pi); i+=1

        ax = jnp.sin(theta) * jnp.cos(phi)
        ay = jnp.sin(theta) * jnp.sin(phi)
        az = jnp.cos(theta)
        axes = jnp.stack([ax, ay, az], axis=-1)

        c_hh = jax.random.uniform(k[i], shape=(Nc,), minval=self.capsule_hh_min, maxval=self.capsule_hh_max); i+=1
        c_r  = jax.random.uniform(k[i], shape=(Nc,), minval=self.capsule_r_min,  maxval=self.capsule_r_max); i+=1

        c_centers = jnp.stack([c_cx, c_cy, c_cz], axis=-1)
        cap_a = c_centers - c_hh[:, None] * axes
        cap_b = c_centers + c_hh[:, None] * axes

        sphere_centers = jnp.concatenate([sphere_centers, cap_a, cap_b], axis=0)
        sphere_radii   = jnp.concatenate([sphere_radii,   c_r,   c_r],  axis=0)

        cylinder_params = jnp.concatenate([
            c_centers, axes, c_hh[:, None], c_r[:, None],
        ], axis=-1).reshape(-1)

        # ---- Windows -------------------------------------------------------
        Nw = self.n_windows
        if Nw > 0:
            w_pos = jnp.array(self.window_positions, dtype=jnp.float32)
            w_cx, w_cy, w_cz = w_pos[:, 0], w_pos[:, 1], w_pos[:, 2]

            iw  = jnp.full((Nw,), self.window_w)
            ih  = jnp.full((Nw,), self.window_h)
            brd = jnp.full((Nw,), self.window_border)
            d   = jnp.full((Nw,), self.window_depth)

            hd   = d * 0.5
            hbrd = brd * 0.5
            wall_hy = iw * 0.5 + brd

            bars_cx = jnp.concatenate([w_cx, w_cx, w_cx, w_cx])
            bars_cy = jnp.concatenate([w_cy, w_cy, w_cy - iw*0.5 - hbrd, w_cy + iw*0.5 + hbrd])
            bars_cz = jnp.concatenate([w_cz - ih*0.5 - hbrd, w_cz + ih*0.5 + hbrd, w_cz, w_cz])
            bars_hx = jnp.concatenate([hd, hd, hd, hd])
            bars_hy = jnp.concatenate([wall_hy, wall_hy, hbrd, hbrd])
            bars_hz = jnp.concatenate([hbrd, hbrd, ih*0.5, ih*0.5])

            win_centers      = jnp.stack([bars_cx, bars_cy, bars_cz], axis=-1).reshape(-1)
            win_half_extents = jnp.stack([bars_hx, bars_hy, bars_hz], axis=-1).reshape(-1)
        else:
            win_centers = win_half_extents = jnp.zeros(0)

        return jnp.concatenate([
            sphere_centers.reshape(-1),
            sphere_radii,
            box_centers,      win_centers,
            box_half_extents, win_half_extents,
            cylinder_params,
        ])

    # -----------------------------------------------------------------------
    # Static mode — unpacking
    # -----------------------------------------------------------------------

    def unpack(self, scene_array: jnp.ndarray) -> dict:
        """
        Split a flat static-mode scene array into named geometry arrays.
        Not used in procedural mode.
        """
        Ns_t = self.n_spheres + 2 * self.n_capsules
        Nb_t = self.n_boxes + 4 * self.n_windows
        Nc   = self.n_capsules
        i = 0

        sphere_centers   = scene_array[i : i+Ns_t*3].reshape(Ns_t, 3);  i += Ns_t*3
        sphere_radii     = scene_array[i : i+Ns_t];                       i += Ns_t
        box_centers      = scene_array[i : i+Nb_t*3].reshape(Nb_t, 3);   i += Nb_t*3
        box_half_extents = scene_array[i : i+Nb_t*3].reshape(Nb_t, 3);   i += Nb_t*3
        cylinder_flat    = scene_array[i : i+Nc*8].reshape(Nc, 8)

        return {
            "sphere_centers":    sphere_centers,
            "sphere_radii":      sphere_radii,
            "box_centers":       box_centers,
            "box_half_extents":  box_half_extents,
            "cylinder_centers":  cylinder_flat[:, 0:3],
            "cylinder_axes":     cylinder_flat[:, 3:6],
            "cylinder_hh":       cylinder_flat[:, 6],
            "cylinder_radii":    cylinder_flat[:, 7],
        }

    def summary(self) -> str:
        if self.procedural:
            K = 9 * self.obstacles_per_cell
            return (
                f"SceneConfig  procedural  cell_size={self.cell_size:.1f}m  "
                f"obstacles_per_cell={self.obstacles_per_cell}  K={K}"
            )
        vol = (
            (self.arena_x_max - self.arena_x_min)
            * (self.arena_y_max - self.arena_y_min)
            * (self.arena_z_max - self.arena_z_min)
        )
        return (
            f"SceneConfig  static  arena={vol:.1f}m³  "
            f"Ns={self.n_spheres}  Nb={self.n_boxes}  Nc={self.n_capsules}  "
            f"scene_dim={self.scene_dim}"
        )
