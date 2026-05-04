"""
scene.py — Random scene generation for the Navigate environment.

SceneConfig holds all obstacle parameters and exposes two JAX-compatible
methods that navigate.py calls from reset():

    scene_array          = cfg.sample(key)        → flat float32 JAX array
    scene_dict           = cfg.unpack(scene_array) → structured arrays

The flat layout inside scene_array:
    [0         : Ns*3      ]  sphere centres   (Ns, 3)
    [Ns*3      : Ns*4      ]  sphere radii     (Ns,)
    [Ns*4      : Ns*4+Nb*3 ]  box centres      (Nb, 3)
    [Ns*4+Nb*3 : Ns*4+Nb*7 ]  box quaternions  (Nb, 4)  [qw, qx, qy, qz]
    [Ns*4+Nb*7 : Ns*4+Nb*10]  box half-extents (Nb, 3)
    [Ns*4+Nb*10: Ns*4+Nb*10+Nc*8]  capsule params (Nc, 8)
        per capsule: [cx, cy, cz, ax, ay, az, half_h, r]
        where (ax,ay,az) is the unit axis of the capsule

This layout matches depth_render.scene (Sphere / Box / Capsule) field-for-field,
so the depth renderer can consume unpack() output directly.
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
    n_spheres:  int = 4
    n_boxes:    int = 4
    n_capsules: int = 4

    # ---- Sphere size bounds ------------------------------------------------
    sphere_r_min: float = 0.15;  sphere_r_max: float = 0.60

    # ---- Box size bounds (half-extents per axis) ----------------------------
    box_hx_min: float = 0.10;  box_hx_max: float = 0.60
    box_hy_min: float = 0.10;  box_hy_max: float = 0.60
    box_hz_min: float = 0.10;  box_hz_max: float = 0.60

    # ---- Capsule size bounds -----------------------------------------------
    capsule_r_min:  float = 0.10;  capsule_r_max:  float = 0.40
    capsule_hh_min: float = 0.30;  capsule_hh_max: float = 1.20

    scene_dim: int = _field(init=False, repr=True)

    def __post_init__(self):
        Ns, Nb, Nc = self.n_spheres, self.n_boxes, self.n_capsules
        self.scene_dim = Ns * 4 + Nb * 6 + Nc * 8

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
        k = jax.random.split(key, 18)
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
        c_r  = jax.random.uniform(k[i], shape=(Nc,), minval=self.capsule_r_min,  maxval=self.capsule_r_max)

        # Pack capsule params row-wise: (Nc, 8) → flat (Nc*8,)
        capsule_params = jnp.concatenate([
            jnp.stack([c_cx, c_cy, c_cz], axis=-1),  # (Nc, 3) centers
            axes,                                      # (Nc, 3) unit axes
            c_hh[:, None],                             # (Nc, 1) half-heights
            c_r[:, None],                              # (Nc, 1) radii
        ], axis=-1).reshape(-1)  # (Nc*8,)

        return jnp.concatenate([
            sphere_centers,
            sphere_radii,
            box_centers,
            box_half_extents,
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
            box_quaternions   (Nb, 4)  [qw, qx, qy, qz] unit quaternions
            box_half_extents  (Nb, 3)
            capsule_centers   (Nc, 3)
            capsule_axes      (Nc, 3)   unit axes
            capsule_hh        (Nc,)     half-heights
            capsule_radii     (Nc,)
        """
        Ns, Nb, Nc = self.n_spheres, self.n_boxes, self.n_capsules
        i = 0

        sphere_centers   = scene_array[i : i+Ns*3].reshape(Ns, 3); i += Ns*3
        sphere_radii     = scene_array[i : i+Ns];                   i += Ns
        box_centers      = scene_array[i : i+Nb*3].reshape(Nb, 3); i += Nb*3
        box_half_extents = scene_array[i : i+Nb*3].reshape(Nb, 3); i += Nb*3
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
            f"SceneConfig  density={self.obstacle_density}/m³  "
            f"arena={vol:.1f}m³  "
            f"Ns={self.n_spheres}  Nb={self.n_boxes}  Nc={self.n_capsules}  "
            f"scene_dim={self.scene_dim}"
        )
