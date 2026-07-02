"""
scene.py — Scene generation for the Navigate environment.

Two modes, selected by the YAML config:

  STATIC mode  (n_spheres / n_boxes / n_capsules / n_windows / n_trees > 0)
    scene_array = cfg.sample(key)      → flat float32 array of shape (scene_dim,)
    scene_dict  = cfg.unpack(arr)      → structured arrays for the renderer

    The flat layout (Ns_t = n_spheres + 2*n_capsules,
                     Nb_t = n_boxes + 4*n_windows + n_trees,
                     Nc_t = n_capsules,
                     No_t = 3*n_trees):
      [0              : Ns_t*3         ]  sphere centres    (Ns_t, 3)
      [Ns_t*3         : Ns_t*4         ]  sphere radii      (Ns_t,)
      [Ns_t*4         : Ns_t*4+Nb_t*3  ]  box centres       (Nb_t, 3)  last n_trees = trunk AABBs
      [Ns_t*4+Nb_t*3  : Ns_t*4+Nb_t*6  ]  box half-extents  (Nb_t, 3)
      [Ns_t*4+Nb_t*6  : ...+Nc_t*8     ]  cylinder params   (Nc_t, 8)  standalone capsules only
      [...+Nc_t*8     : ...]              branch OBB params  (No_t, 10) center(3)+quat(4)+he(3)

  PROCEDURAL mode  (boxes_per_cell / spheres_per_cell / capsules_per_cell /
                    trees_per_cell > 0)
    Obstacles are generated on-the-fly each step from the drone position and
    a per-episode seed.  The world is infinite: the same (seed, cell) always
    produces the same obstacles, so the scene is consistent across timesteps.

    scene_array = cfg.sample(key)      → shape (1,) — just the float32 seed
    (sphere_c, sphere_r,
     box_c, box_he,
     cap_c, cap_ax, cap_hh, cap_r) = cfg.get_local_obstacles(drone_pos, seed_float)

    K_s = 9 × spheres_per_cell, K_b = 9 × (boxes_per_cell + trees_per_cell),
    K_c = 9 × capsules_per_cell, K_ob = 9 × 3 × trees_per_cell

    Cell size is set to cam_max_range by Navigate.__init__ after construction.
    A 3×3 grid of cells centered on the drone's current cell is loaded;
    this always covers the full camera frustum regardless of drone position
    within its cell.

  TREES (composite obstacle)
    Each tree consists of 1 trunk AABB + 3 branch OBBs:
      - Trunk: vertical AABB rooted at z=0 (ground), merged into the box arrays.
      - Branches: 3 OBBs starting at the trunk top, fanning out 120° apart at
        tree_branch_tilt radians from the vertical, with a random per-tree
        azimuth offset so no two trees look identical.
    Branch quaternions are computed via the half-angle formula rotating [0,0,1]
    onto the branch axis direction.
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

    # ---- Tree obstacles (trunk AABB + 3 branch OBBs) -----------------------
    # Trunk: vertical AABB, base at z=0 (ground), grows upward (−z in NED).
    # Branches: start at trunk top, fan out 120° apart at a fixed tilt angle
    #           (tree_branch_tilt radians from vertical) with a random per-tree
    #           azimuth offset.
    tree_trunk_r_min:     float = 0.05;  tree_trunk_r_max:     float = 0.12
    tree_trunk_hh_min:    float = 0.60;  tree_trunk_hh_max:    float = 1.50
    tree_branch_r_min:    float = 0.03;  tree_branch_r_max:    float = 0.06
    tree_branch_hh_min:   float = 0.30;  tree_branch_hh_max:   float = 0.70
    tree_branch_tilt:     float = 0.75

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
            Ns_t = Ns + 2 * Nc           # spheres + standalone capsule end-caps
            Nb_t = Nb + 4 * Nw + Nt     # standalone boxes + windows + trunk AABBs
            self.scene_dim = Ns_t * 4 + Nb_t * 6 + Nc * 8 + 3 * Nt * 10

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
            sphere_centers   (9*(spheres_per_cell + 2*capsules_per_cell), 3)
            sphere_radii     same length, float32
            box_centers      (9*(boxes_per_cell + trees_per_cell), 3)   last trees_per_cell = trunk AABBs
            box_half_extents same shape
            cap_centers      (9*capsules_per_cell, 3)
            cap_axes         same shape
            cap_hh           (9*capsules_per_cell,)
            cap_radii        same length
            obb_centers      (9*3*trees_per_cell, 3)   branch OBBs
            obb_quats        (9*3*trees_per_cell, 4)   [qw,qx,qy,qz] per branch
            obb_half_extents (9*3*trees_per_cell, 3)
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

            # End-cap spheres for standalone capsules only, merged into the
            # sphere arrays here (mirrors trunk-into-box merging below).
            if Mc > 0:
                cap_a = cap_centers - c_hh[:, None] * cap_axes
                cap_b = cap_centers + c_hh[:, None] * cap_axes
                sphere_centers = jnp.concatenate([sphere_centers, cap_a, cap_b], axis=0)
                sphere_radii   = jnp.concatenate([sphere_radii,   c_r, c_r],     axis=0)

            # Trees — separate key stream so existing capsule/sphere/box
            # randomisation is unchanged when trees_per_cell=0.
            tree_key = jax.random.fold_in(cell_key, 1000)
            tkeys   = jax.random.split(tree_key, 7)

            t_cx        = jax.random.uniform(tkeys[0], (Mt,), minval=x_min, maxval=x_max)
            t_cy        = jax.random.uniform(tkeys[1], (Mt,), minval=y_min, maxval=y_max)
            t_trunk_hh  = jax.random.uniform(tkeys[2], (Mt,), minval=self.tree_trunk_hh_min,   maxval=self.tree_trunk_hh_max)
            t_trunk_r   = jax.random.uniform(tkeys[3], (Mt,), minval=self.tree_trunk_r_min,    maxval=self.tree_trunk_r_max)
            t_branch_hh = jax.random.uniform(tkeys[4], (Mt,), minval=self.tree_branch_hh_min,  maxval=self.tree_branch_hh_max)
            t_branch_r  = jax.random.uniform(tkeys[5], (Mt,), minval=self.tree_branch_r_min,   maxval=self.tree_branch_r_max)
            phi_off     = jax.random.uniform(tkeys[6], (Mt,), minval=0.0, maxval=2*math.pi/3)

            # Trunk: AABB, merged into box arrays
            trunk_centers = jnp.stack([t_cx, t_cy, -t_trunk_hh], axis=-1)           # (Mt, 3)
            trunk_he      = jnp.stack([t_trunk_r, t_trunk_r, t_trunk_hh], axis=-1)  # (Mt, 3)

            # Branches: OBB — 3 per tree at 120° azimuth spacing, fixed tilt
            phis = jnp.stack(
                [phi_off, phi_off + 2*math.pi/3, phi_off + 4*math.pi/3], axis=-1
            )  # (Mt, 3)
            sin_t = math.sin(self.tree_branch_tilt)  # scalar constant
            cos_t = math.cos(self.tree_branch_tilt)
            b_ax_x = sin_t * jnp.cos(phis)           # (Mt, 3)
            b_ax_y = sin_t * jnp.sin(phis)           # (Mt, 3)
            b_ax_z = -cos_t * jnp.ones_like(b_ax_x)  # (Mt, 3)
            branch_axes = jnp.stack([b_ax_x, b_ax_y, b_ax_z], axis=-1)  # (Mt, 3, 3)

            trunk_tops = jnp.stack([t_cx, t_cy, -2.0 * t_trunk_hh], axis=-1)  # (Mt, 3)
            branch_centers = (                             # (Mt, 3, 3)
                trunk_tops[:, None, :]
                + t_branch_hh[:, None, None] * branch_axes
            )

            b_centers_flat = branch_centers.reshape(3 * Mt, 3)   # (3*Mt, 3)
            b_axes_flat    = branch_axes.reshape(3 * Mt, 3)       # (3*Mt, 3)
            b_hh_flat      = jnp.repeat(t_branch_hh, 3)          # (3*Mt,)
            b_r_flat       = jnp.repeat(t_branch_r,  3)          # (3*Mt,)

            # Quaternion: rotate [0,0,1] → branch axis (half-angle formula)
            bz   = b_axes_flat[:, 2]
            norm = jnp.sqrt(jnp.maximum(2.0 * (1.0 + bz), 1e-8))
            branch_quats = jnp.stack([
                (1.0 + bz) / norm,           # qw
                -b_axes_flat[:, 1] / norm,   # qx
                 b_axes_flat[:, 0] / norm,   # qy
                jnp.zeros_like(bz),          # qz
            ], axis=-1)  # (3*Mt, 4)
            branch_obb_he = jnp.stack([b_r_flat, b_r_flat, b_hh_flat], axis=-1)  # (3*Mt, 3)

            all_box_c  = jnp.concatenate([box_centers,      trunk_centers], axis=0)
            all_box_he = jnp.concatenate([box_half_extents, trunk_he],      axis=0)

            return (sphere_centers, sphere_radii,
                    all_box_c, all_box_he,
                    cap_centers, cap_axes, c_hh, c_r,
                    b_centers_flat, branch_quats, branch_obb_he)

        results = jax.vmap(sample_cell)(offsets)  # each element has leading dim 9
        (sphere_centers, sphere_radii,
         box_centers, box_half_extents,
         cap_centers, cap_axes, cap_hh, cap_radii,
         obb_centers, obb_quats, obb_he) = results

        sph_c      = sphere_centers.reshape(9 * (Ms + 2 * Mc), 3)
        sph_r      = sphere_radii.reshape(9 * (Ms + 2 * Mc))
        box_c      = box_centers.reshape(9 * (Mb + Mt), 3)
        box_he     = box_half_extents.reshape(9 * (Mb + Mt), 3)
        cap_c      = cap_centers.reshape(9 * Mc, 3)
        cap_ax     = cap_axes.reshape(9 * Mc, 3)
        cap_hh_out = cap_hh.reshape(9 * Mc)
        cap_r_out  = cap_radii.reshape(9 * Mc)
        obb_c_out  = obb_centers.reshape(9 * 3 * Mt, 3)
        obb_q_out  = obb_quats.reshape(9 * 3 * Mt, 4)
        obb_he_out = obb_he.reshape(9 * 3 * Mt, 3)

        return (
            sph_c,
            sph_r,
            box_c,
            box_he,
            cap_c,
            cap_ax,
            cap_hh_out,
            cap_r_out,
            obb_c_out,
            obb_q_out,
            obb_he_out,
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

            t_x         = jax.random.uniform(k_tree[0], (Nt,), minval=self.arena_x_min,       maxval=self.arena_x_max)
            t_y         = jax.random.uniform(k_tree[1], (Nt,), minval=self.arena_y_min,       maxval=self.arena_y_max)
            t_trunk_hh  = jax.random.uniform(k_tree[2], (Nt,), minval=self.tree_trunk_hh_min,  maxval=self.tree_trunk_hh_max)
            t_trunk_r   = jax.random.uniform(k_tree[3], (Nt,), minval=self.tree_trunk_r_min,   maxval=self.tree_trunk_r_max)
            t_branch_hh = jax.random.uniform(k_tree[4], (Nt,), minval=self.tree_branch_hh_min, maxval=self.tree_branch_hh_max)
            t_branch_r  = jax.random.uniform(k_tree[5], (Nt,), minval=self.tree_branch_r_min,  maxval=self.tree_branch_r_max)
            phi_off     = jax.random.uniform(k_tree[6], (Nt,), minval=0.0, maxval=2*math.pi/3)

            # Trunk: AABB, merged into box arrays
            trunk_centers = jnp.stack([t_x, t_y, -t_trunk_hh], axis=-1)           # (Nt, 3)
            trunk_he      = jnp.stack([t_trunk_r, t_trunk_r, t_trunk_hh], axis=-1) # (Nt, 3)
            box_centers      = jnp.concatenate([box_centers,      trunk_centers.reshape(-1)])
            box_half_extents = jnp.concatenate([box_half_extents, trunk_he.reshape(-1)])

            # Branches: OBB — 3 per tree at 120° azimuth spacing, fixed tilt
            phis = jnp.stack(
                [phi_off, phi_off + 2*math.pi/3, phi_off + 4*math.pi/3], axis=-1
            )  # (Nt, 3)
            sin_t = math.sin(self.tree_branch_tilt)  # scalar constant
            cos_t = math.cos(self.tree_branch_tilt)
            b_ax_x = sin_t * jnp.cos(phis)
            b_ax_y = sin_t * jnp.sin(phis)
            b_ax_z = -cos_t * jnp.ones_like(b_ax_x)
            branch_axes = jnp.stack([b_ax_x, b_ax_y, b_ax_z], axis=-1)  # (Nt, 3, 3)

            trunk_tops = jnp.stack([t_x, t_y, -2.0 * t_trunk_hh], axis=-1)  # (Nt, 3)
            branch_centers = (
                trunk_tops[:, None, :] + t_branch_hh[:, None, None] * branch_axes
            )  # (Nt, 3, 3)

            b_centers_flat = branch_centers.reshape(3 * Nt, 3)
            b_axes_flat    = branch_axes.reshape(3 * Nt, 3)
            b_hh_flat      = jnp.repeat(t_branch_hh, 3)
            b_r_flat       = jnp.repeat(t_branch_r,  3)

            bz   = b_axes_flat[:, 2]
            norm = jnp.sqrt(jnp.maximum(2.0 * (1.0 + bz), 1e-8))
            branch_quats = jnp.stack([
                (1.0 + bz) / norm,
                -b_axes_flat[:, 1] / norm,
                 b_axes_flat[:, 0] / norm,
                jnp.zeros_like(bz),
            ], axis=-1)  # (3*Nt, 4)
            branch_obb_he = jnp.stack([b_r_flat, b_r_flat, b_hh_flat], axis=-1)  # (3*Nt, 3)
            obb_params = jnp.concatenate(
                [b_centers_flat, branch_quats, branch_obb_he], axis=-1
            ).reshape(-1)
        else:
            obb_params = jnp.zeros(0)

        # End-cap spheres for standalone capsules only
        cap_a = c_centers - c_hh[:, None] * axes
        cap_b = c_centers + c_hh[:, None] * axes
        sphere_centers = jnp.concatenate([sphere_centers, cap_a, cap_b], axis=0)
        sphere_radii   = jnp.concatenate([sphere_radii,   c_r, c_r])

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
            obb_params,
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
        Nb_t = self.n_boxes + 4 * self.n_windows + self.n_trees
        Nc   = self.n_capsules
        No   = 3 * self.n_trees
        i = 0

        sphere_centers   = scene_array[i : i+Ns_t*3].reshape(Ns_t, 3);  i += Ns_t*3
        sphere_radii     = scene_array[i : i+Ns_t];                       i += Ns_t
        box_centers      = scene_array[i : i+Nb_t*3].reshape(Nb_t, 3);   i += Nb_t*3
        box_half_extents = scene_array[i : i+Nb_t*3].reshape(Nb_t, 3);   i += Nb_t*3
        cylinder_flat    = scene_array[i : i+Nc*8].reshape(Nc, 8);        i += Nc*8
        obb_flat         = scene_array[i : i+No*10].reshape(No, 10)

        return {
            "sphere_centers":    sphere_centers,
            "sphere_radii":      sphere_radii,
            "box_centers":       box_centers,
            "box_half_extents":  box_half_extents,
            "cylinder_centers":  cylinder_flat[:, 0:3],
            "cylinder_axes":     cylinder_flat[:, 3:6],
            "cylinder_hh":       cylinder_flat[:, 6],
            "cylinder_radii":    cylinder_flat[:, 7],
            "obb_centers":       obb_flat[:, 0:3],
            "obb_quats":         obb_flat[:, 3:7],
            "obb_half_extents":  obb_flat[:, 7:10],
        }

    def summary(self) -> str:
        if self.procedural:
            return (
                f"SceneConfig  procedural  cell_size={self.cell_size:.1f}m  "
                f"boxes={self.boxes_per_cell}  spheres={self.spheres_per_cell}  "
                f"capsules={self.capsules_per_cell}  trees={self.trees_per_cell}  "
                f"K_box={9*(self.boxes_per_cell+self.trees_per_cell)}  K_cap={9*self.capsules_per_cell}  K_obb={9*3*self.trees_per_cell}"
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
