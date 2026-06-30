"""
scene.py — Scene generation for the Navigate environment.

Two modes, selected by the YAML config:

  STATIC mode  (n_spheres / n_boxes / n_capsules / n_windows / n_trees > 0)
    scene_array = cfg.sample(key)      → flat float32 array of shape (scene_dim,)
    scene_dict  = cfg.unpack(arr)      → structured arrays for the renderer

    The flat layout (Ns_t = n_spheres + 2*(n_capsules + 4*n_trees),
                     Nb_t = n_boxes + 4*n_windows,
                     Nc_t = n_capsules + 4*n_trees):
      [0              : Ns_t*3         ]  sphere centres   (Ns_t, 3)
      [Ns_t*3         : Ns_t*4         ]  sphere radii     (Ns_t,)
      [Ns_t*4         : Ns_t*4+Nb_t*3  ]  box centres      (Nb_t, 3)
      [Ns_t*4+Nb_t*3  : Ns_t*4+Nb_t*6  ]  box half-extents (Nb_t, 3)
      [Ns_t*4+Nb_t*6  : ...]              cylinder params  (Nc_t, 8)

  PROCEDURAL mode  (boxes_per_cell / spheres_per_cell / capsules_per_cell /
                    trees_per_cell > 0)
    Obstacles are generated on-the-fly each step from the drone position and
    a per-episode seed.  The world is infinite: the same (seed, cell) always
    produces the same obstacles, so the scene is consistent across timesteps.

    scene_array = cfg.sample(key)      → shape (1,) — just the float32 seed
    (sphere_c, sphere_r,
     box_c, box_he,
     cap_c, cap_ax, cap_hh, cap_r) = cfg.get_local_obstacles(drone_pos, seed_float)

    K_s = 9 × spheres_per_cell, K_b = 9 × boxes_per_cell,
    K_c = 9 × (capsules_per_cell + 4*trees_per_cell)

    Cell size is set to cam_max_range by Navigate.__init__ after construction.
    A 3×3 grid of cells centered on the drone's current cell is loaded;
    this always covers the full camera frustum regardless of drone position
    within its cell.

  TREES (composite obstacle)
    Each tree consists of 4 capsules:
      - 1 vertical trunk rooted at z=0 (ground), growing upward (−z in NED)
      - 3 branches starting at the trunk top, fanning out 120° apart at
        tree_branch_tilt radians from the vertical, with a random per-tree
        azimuth offset so no two trees look identical.
    Trees are implemented as additional capsules appended to the capsule arrays;
    the renderer sees only capsules (+ their auto-generated end-cap spheres).
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

    Set any of boxes_per_cell / spheres_per_cell / capsules_per_cell /
    trees_per_cell > 0 to enable procedural infinite-world mode.  Otherwise,
    set n_spheres / n_boxes / n_capsules / n_windows / n_trees for static mode.
    """

    # ---- Procedural mode ---------------------------------------------------
    boxes_per_cell:    int   = 0     # boxes per cell  }
    spheres_per_cell:  int   = 0     # spheres per cell } any >0 → procedural mode
    capsules_per_cell: int   = 0     # capsules per cell}
    trees_per_cell:    int   = 0     # trees per cell   }
    cell_size:         float = 8.0   # set to cam_max_range by Navigate.__init__

    # ---- Arena (static mode spawn bounds / procedural z bounds) ------------
    arena_x_min: float = 0.0;  arena_x_max: float = 10.0
    arena_y_min: float = 0.0;  arena_y_max: float = 10.0
    arena_z_min: float = -4.0; arena_z_max: float = 0.0

    # ---- Static-mode obstacle counts ---------------------------------------
    n_spheres:  int = 0
    n_boxes:    int = 0
    n_capsules: int = 0
    n_windows:  int = 0
    n_trees:    int = 0

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

    # ---- Tree obstacles (composite: 1 trunk + 3 branches, all capsules) ----
    # Trunk: vertical capsule, base at z=0 (ground), grows upward (−z in NED).
    # Branches: start at trunk top, fan out 120° apart at tree_branch_tilt
    #           radians from the vertical, random per-tree azimuth offset.
    tree_trunk_r_min:   float = 0.05;  tree_trunk_r_max:   float = 0.12
    tree_trunk_hh_min:  float = 0.60;  tree_trunk_hh_max:  float = 1.50
    tree_branch_r_min:  float = 0.03;  tree_branch_r_max:  float = 0.06
    tree_branch_hh_min: float = 0.30;  tree_branch_hh_max: float = 0.70
    tree_branch_tilt:   float = 0.70   # radians from vertical, ~40°

    scene_dim: int = _field(init=False, repr=True)
    procedural: bool = _field(init=False, repr=True)

    def __post_init__(self):
        self.procedural = (
            self.boxes_per_cell > 0 or
            self.spheres_per_cell > 0 or
            self.capsules_per_cell > 0 or
            self.trees_per_cell > 0
        )
        if self.procedural:
            self.scene_dim = 1  # only the seed float
        else:
            Ns, Nb, Nc, Nw, Nt = (self.n_spheres, self.n_boxes,
                                    self.n_capsules, self.n_windows, self.n_trees)
            Nc_t = Nc + 4 * Nt          # standalone capsules + tree capsules
            Ns_t = Ns + 2 * Nc_t        # spheres + capsule end-caps
            self.scene_dim = Ns_t * 4 + (Nb + 4 * Nw) * 6 + Nc_t * 8

    # -----------------------------------------------------------------------
    # Procedural mode
    # -----------------------------------------------------------------------

    def get_local_obstacles(
        self,
        drone_pos:  jnp.ndarray,  # (3,)  world-space position
        seed_float: jnp.ndarray,  # ()    float32 episode seed
    ):
        """
        Generate obstacles for the 3×3 cell neighbourhood centred on the drone.

        The same (seed, cell_ix, cell_iy) always produces identical obstacles,
        so the scene is persistent across timesteps without storing positions.

        Key split order per cell (17 total for spheres/boxes/capsules):
          0-3:  sphere cx, cy, cz, r
          4-9:  box cx, cy, cz, hx, hy, hz
          10-16: capsule cx, cy, cz, theta, phi, hh, r
        Trees use a separate key derived via fold_in(cell_key, 1000) for
        backward compatibility with existing scenes (sphere/box/capsule
        randomisation is unchanged when trees_per_cell=0).

        Returns:
            sphere_centers   (9*(spheres_per_cell + 2*(capsules_per_cell + 4*trees_per_cell)), 3)
            sphere_radii     same length, float32
            box_centers      (9*boxes_per_cell, 3)
            box_half_extents (9*boxes_per_cell, 3)
            cap_centers      (9*(capsules_per_cell + 4*trees_per_cell), 3)
            cap_axes         same shape
            cap_hh           (9*(capsules_per_cell + 4*trees_per_cell),)
            cap_radii        same length
        """
        Ms        = self.spheres_per_cell
        Mb        = self.boxes_per_cell
        Mc        = self.capsules_per_cell
        Mt        = self.trees_per_cell
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
            keys = jax.random.split(cell_key, 17)

            x_min = ix.astype(jnp.float32) * cell_size
            x_max = (ix + 1).astype(jnp.float32) * cell_size
            y_min = iy.astype(jnp.float32) * cell_size
            y_max = (iy + 1).astype(jnp.float32) * cell_size

            # Spheres
            s_cx = jax.random.uniform(keys[0], (Ms,), minval=x_min, maxval=x_max)
            s_cy = jax.random.uniform(keys[1], (Ms,), minval=y_min, maxval=y_max)
            s_cz = jax.random.uniform(keys[2], (Ms,), minval=self.arena_z_min, maxval=self.arena_z_max)
            s_r  = jax.random.uniform(keys[3], (Ms,), minval=self.sphere_r_min, maxval=self.sphere_r_max)
            sphere_centers = jnp.stack([s_cx, s_cy, s_cz], axis=-1)  # (Ms, 3)
            sphere_radii   = s_r                                        # (Ms,)

            # Boxes
            cx = jax.random.uniform(keys[4], (Mb,), minval=x_min, maxval=x_max)
            cy = jax.random.uniform(keys[5], (Mb,), minval=y_min, maxval=y_max)
            cz = jax.random.uniform(keys[6], (Mb,), minval=self.arena_z_min, maxval=self.arena_z_max)
            hx = jax.random.uniform(keys[7], (Mb,), minval=self.box_hx_min,  maxval=self.box_hx_max)
            hy = jax.random.uniform(keys[8], (Mb,), minval=self.box_hy_min,  maxval=self.box_hy_max)
            hz = jax.random.uniform(keys[9], (Mb,), minval=self.box_hz_min,  maxval=self.box_hz_max)
            box_centers      = jnp.stack([cx, cy, cz], axis=-1)  # (Mb, 3)
            box_half_extents = jnp.stack([hx, hy, hz], axis=-1)  # (Mb, 3)

            # Capsules
            c_cx  = jax.random.uniform(keys[10], (Mc,), minval=x_min, maxval=x_max)
            c_cy  = jax.random.uniform(keys[11], (Mc,), minval=y_min, maxval=y_max)
            c_cz  = jax.random.uniform(keys[12], (Mc,), minval=self.arena_z_min, maxval=self.arena_z_max)
            theta = jax.random.uniform(keys[13], (Mc,), minval=0.0, maxval=math.pi)
            phi   = jax.random.uniform(keys[14], (Mc,), minval=0.0, maxval=2 * math.pi)
            c_hh  = jax.random.uniform(keys[15], (Mc,), minval=self.capsule_hh_min, maxval=self.capsule_hh_max)
            c_r   = jax.random.uniform(keys[16], (Mc,), minval=self.capsule_r_min,  maxval=self.capsule_r_max)
            cap_axes    = jnp.stack([jnp.sin(theta) * jnp.cos(phi),
                                     jnp.sin(theta) * jnp.sin(phi),
                                     jnp.cos(theta)], axis=-1)  # (Mc, 3)
            cap_centers = jnp.stack([c_cx, c_cy, c_cz], axis=-1)  # (Mc, 3)

            # Trees — separate key stream so existing capsule/sphere/box
            # randomisation is unchanged when trees_per_cell=0.
            tree_key = jax.random.fold_in(cell_key, 1000)
            tkeys   = jax.random.split(tree_key, 7)

            t_cx         = jax.random.uniform(tkeys[0], (Mt,), minval=x_min, maxval=x_max)
            t_cy         = jax.random.uniform(tkeys[1], (Mt,), minval=y_min, maxval=y_max)
            t_trunk_hh   = jax.random.uniform(tkeys[2], (Mt,), minval=self.tree_trunk_hh_min,  maxval=self.tree_trunk_hh_max)
            t_trunk_r    = jax.random.uniform(tkeys[3], (Mt,), minval=self.tree_trunk_r_min,   maxval=self.tree_trunk_r_max)
            t_branch_hh  = jax.random.uniform(tkeys[4], (Mt,), minval=self.tree_branch_hh_min, maxval=self.tree_branch_hh_max)
            t_branch_r   = jax.random.uniform(tkeys[5], (Mt,), minval=self.tree_branch_r_min,  maxval=self.tree_branch_r_max)
            phi_off      = jax.random.uniform(tkeys[6], (Mt,), minval=0.0, maxval=2*math.pi/3)

            # Trunk: vertical capsule, base at z=0, axis = (0, 0, 1) in NED
            trunk_centers = jnp.stack([t_cx, t_cy, -t_trunk_hh], axis=-1)   # (Mt, 3)
            trunk_axes    = jnp.zeros((Mt, 3)).at[:, 2].set(1.0)             # (Mt, 3)

            # Branches: 3 per tree at 120° azimuth spacing, tilted from vertical
            phis = jnp.stack(
                [phi_off, phi_off + 2*math.pi/3, phi_off + 4*math.pi/3], axis=-1
            )  # (Mt, 3)
            sin_t, cos_t = math.sin(self.tree_branch_tilt), math.cos(self.tree_branch_tilt)
            b_ax_x = sin_t * jnp.cos(phis)          # (Mt, 3)
            b_ax_y = sin_t * jnp.sin(phis)          # (Mt, 3)
            b_ax_z = jnp.full_like(b_ax_x, -cos_t)  # (Mt, 3) — upward in NED (−z)
            branch_axes = jnp.stack([b_ax_x, b_ax_y, b_ax_z], axis=-1)  # (Mt, 3, 3)

            # Branch centers: start at trunk top, extend along branch axis
            trunk_tops = jnp.stack([t_cx, t_cy, -2.0 * t_trunk_hh], axis=-1)  # (Mt, 3)
            branch_centers = (                             # (Mt, 3, 3)
                trunk_tops[:, None, :]
                + t_branch_hh[:, None, None] * branch_axes
            )

            b_centers_flat = branch_centers.reshape(3 * Mt, 3)   # (3*Mt, 3)
            b_axes_flat    = branch_axes.reshape(3 * Mt, 3)       # (3*Mt, 3)
            b_hh_flat      = jnp.repeat(t_branch_hh, 3)          # (3*Mt,)
            b_r_flat       = jnp.repeat(t_branch_r,  3)          # (3*Mt,)

            tree_cap_c  = jnp.concatenate([trunk_centers,  b_centers_flat], axis=0)  # (4*Mt, 3)
            tree_cap_ax = jnp.concatenate([trunk_axes,     b_axes_flat],    axis=0)  # (4*Mt, 3)
            tree_cap_hh = jnp.concatenate([t_trunk_hh,    b_hh_flat])                # (4*Mt,)
            tree_cap_r  = jnp.concatenate([t_trunk_r,     b_r_flat])                 # (4*Mt,)

            return (sphere_centers, sphere_radii,
                    box_centers, box_half_extents,
                    cap_centers, cap_axes, c_hh, c_r,
                    tree_cap_c, tree_cap_ax, tree_cap_hh, tree_cap_r)

        results = jax.vmap(sample_cell)(offsets)  # each element has leading dim 9
        (sphere_centers, sphere_radii,
         box_centers, box_half_extents,
         cap_centers, cap_axes, cap_hh, cap_radii,
         tree_cap_centers, tree_cap_axes, tree_cap_hh, tree_cap_radii) = results

        sph_c  = sphere_centers.reshape(9 * Ms, 3)
        sph_r  = sphere_radii.reshape(9 * Ms)
        cap_c  = cap_centers.reshape(9 * Mc, 3)
        cap_ax = cap_axes.reshape(9 * Mc, 3)
        cap_hh = cap_hh.reshape(9 * Mc)
        cap_r  = cap_radii.reshape(9 * Mc)
        tree_c  = tree_cap_centers.reshape(9 * 4 * Mt, 3)
        tree_ax = tree_cap_axes.reshape(9 * 4 * Mt, 3)
        tree_hh = tree_cap_hh.reshape(9 * 4 * Mt)
        tree_r  = tree_cap_radii.reshape(9 * 4 * Mt)

        # Merge standalone capsules and tree capsules into unified arrays
        all_cap_c  = jnp.concatenate([cap_c,  tree_c],  axis=0)
        all_cap_ax = jnp.concatenate([cap_ax, tree_ax], axis=0)
        all_cap_hh = jnp.concatenate([cap_hh, tree_hh])
        all_cap_r  = jnp.concatenate([cap_r,  tree_r])

        # Add end-cap spheres for all capsules (standalone + tree)
        if Mc > 0 or Mt > 0:
            cap_a = all_cap_c - all_cap_hh[:, None] * all_cap_ax
            cap_b = all_cap_c + all_cap_hh[:, None] * all_cap_ax
            sph_c = jnp.concatenate([sph_c, cap_a, cap_b], axis=0)
            sph_r = jnp.concatenate([sph_r, all_cap_r, all_cap_r], axis=0)

        return (
            sph_c,
            sph_r,
            box_centers.reshape(9 * Mb, 3),
            box_half_extents.reshape(9 * Mb, 3),
            all_cap_c,
            all_cap_ax,
            all_cap_hh,
            all_cap_r,
        )

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

        Ns, Nb, Nc, Nt = self.n_spheres, self.n_boxes, self.n_capsules, self.n_trees

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

        # ---- Trees ---------------------------------------------------------
        # Separate key stream (fold_in) so existing capsule/box/sphere
        # randomisation is unchanged when n_trees=0.
        if Nt > 0:
            k_tree = jax.random.split(jax.random.fold_in(key, 1000), 7)

            t_x         = jax.random.uniform(k_tree[0], (Nt,), minval=self.arena_x_min,      maxval=self.arena_x_max)
            t_y         = jax.random.uniform(k_tree[1], (Nt,), minval=self.arena_y_min,      maxval=self.arena_y_max)
            t_trunk_hh  = jax.random.uniform(k_tree[2], (Nt,), minval=self.tree_trunk_hh_min, maxval=self.tree_trunk_hh_max)
            t_trunk_r   = jax.random.uniform(k_tree[3], (Nt,), minval=self.tree_trunk_r_min,  maxval=self.tree_trunk_r_max)
            t_branch_hh = jax.random.uniform(k_tree[4], (Nt,), minval=self.tree_branch_hh_min,maxval=self.tree_branch_hh_max)
            t_branch_r  = jax.random.uniform(k_tree[5], (Nt,), minval=self.tree_branch_r_min, maxval=self.tree_branch_r_max)
            phi_off     = jax.random.uniform(k_tree[6], (Nt,), minval=0.0, maxval=2*math.pi/3)

            # Trunk: vertical capsule, base at z=0 (ground in NED)
            trunk_centers = jnp.stack([t_x, t_y, -t_trunk_hh], axis=-1)   # (Nt, 3)
            trunk_axes    = jnp.zeros((Nt, 3)).at[:, 2].set(1.0)           # (Nt, 3)

            # Branches: 3 per tree at 120° azimuth spacing
            phis = jnp.stack(
                [phi_off, phi_off + 2*math.pi/3, phi_off + 4*math.pi/3], axis=-1
            )  # (Nt, 3)
            sin_t = math.sin(self.tree_branch_tilt)
            cos_t = math.cos(self.tree_branch_tilt)
            b_ax_x = sin_t * jnp.cos(phis)          # (Nt, 3)
            b_ax_y = sin_t * jnp.sin(phis)          # (Nt, 3)
            b_ax_z = jnp.full_like(b_ax_x, -cos_t)  # (Nt, 3) — upward in NED (−z)
            branch_axes = jnp.stack([b_ax_x, b_ax_y, b_ax_z], axis=-1)  # (Nt, 3, 3)

            trunk_tops = jnp.stack([t_x, t_y, -2.0 * t_trunk_hh], axis=-1)  # (Nt, 3)
            branch_centers = (                             # (Nt, 3, 3)
                trunk_tops[:, None, :]
                + t_branch_hh[:, None, None] * branch_axes
            )

            b_centers_flat = branch_centers.reshape(3 * Nt, 3)
            b_axes_flat    = branch_axes.reshape(3 * Nt, 3)
            b_hh_flat      = jnp.repeat(t_branch_hh, 3)
            b_r_flat       = jnp.repeat(t_branch_r,  3)

            tree_c  = jnp.concatenate([trunk_centers, b_centers_flat], axis=0)  # (4*Nt, 3)
            tree_ax = jnp.concatenate([trunk_axes,    b_axes_flat],    axis=0)  # (4*Nt, 3)
            tree_hh = jnp.concatenate([t_trunk_hh,   b_hh_flat])               # (4*Nt,)
            tree_r  = jnp.concatenate([t_trunk_r,    b_r_flat])                # (4*Nt,)

            all_cap_c  = jnp.concatenate([c_centers, tree_c],  axis=0)  # (Nc+4*Nt, 3)
            all_cap_ax = jnp.concatenate([axes,      tree_ax], axis=0)
            all_cap_hh = jnp.concatenate([c_hh,     tree_hh])
            all_cap_r  = jnp.concatenate([c_r,      tree_r])
        else:
            all_cap_c, all_cap_ax, all_cap_hh, all_cap_r = c_centers, axes, c_hh, c_r

        # End-cap spheres for all capsules (standalone + tree)
        cap_a = all_cap_c - all_cap_hh[:, None] * all_cap_ax
        cap_b = all_cap_c + all_cap_hh[:, None] * all_cap_ax
        sphere_centers = jnp.concatenate([sphere_centers, cap_a, cap_b], axis=0)
        sphere_radii   = jnp.concatenate([sphere_radii,   all_cap_r, all_cap_r])

        cylinder_params = jnp.concatenate([
            all_cap_c, all_cap_ax, all_cap_hh[:, None], all_cap_r[:, None],
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
        Nc_t = self.n_capsules + 4 * self.n_trees   # total cylinders
        Ns_t = self.n_spheres + 2 * Nc_t
        Nb_t = self.n_boxes + 4 * self.n_windows
        i = 0

        sphere_centers   = scene_array[i : i+Ns_t*3].reshape(Ns_t, 3);  i += Ns_t*3
        sphere_radii     = scene_array[i : i+Ns_t];                       i += Ns_t
        box_centers      = scene_array[i : i+Nb_t*3].reshape(Nb_t, 3);   i += Nb_t*3
        box_half_extents = scene_array[i : i+Nb_t*3].reshape(Nb_t, 3);   i += Nb_t*3
        cylinder_flat    = scene_array[i : i+Nc_t*8].reshape(Nc_t, 8)

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
            return (
                f"SceneConfig  procedural  cell_size={self.cell_size:.1f}m  "
                f"boxes={self.boxes_per_cell}  spheres={self.spheres_per_cell}  "
                f"capsules={self.capsules_per_cell}  trees={self.trees_per_cell}  "
                f"K={9*(self.boxes_per_cell+self.spheres_per_cell+self.capsules_per_cell+4*self.trees_per_cell)}"
            )
        vol = (
            (self.arena_x_max - self.arena_x_min)
            * (self.arena_y_max - self.arena_y_min)
            * (self.arena_z_max - self.arena_z_min)
        )
        return (
            f"SceneConfig  static  arena={vol:.1f}m³  "
            f"Ns={self.n_spheres}  Nb={self.n_boxes}  Nc={self.n_capsules}  "
            f"Nt={self.n_trees}  scene_dim={self.scene_dim}"
        )
