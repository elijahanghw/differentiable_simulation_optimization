"""
scene.py — Scene definition: geometric primitives.

Supported shapes:
  Sphere  — centre + radius
  Box     — centre + half-extents (axis-aligned)
  Capsule — centre + unit axis + half-height + radius
  Plane   — infinite plane defined by a point and normal
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Primitive data classes
# ---------------------------------------------------------------------------

@dataclass
class Sphere:
    center: Tuple[float, float, float]
    radius: float
    label:  str = "sphere"


@dataclass
class Box:
    """Oriented box defined by centre, orientation quaternion [qw,qx,qy,qz], and half-extents."""
    center:       Tuple[float, float, float]
    quaternion:   Tuple[float, float, float, float]   # [qw, qx, qy, qz]
    half_extents: Tuple[float, float, float]
    label:        str = "box"


@dataclass
class Capsule:
    """
    Capsule defined by centre, unit axis, half-height of cylindrical part, and radius.

    Attributes:
        center:  (3,) world position of the capsule midpoint
        axis:    (3,) unit axis direction (normalised internally)
        half_h:  half-height of the cylindrical section
        radius:  radius of the cylinder and end-cap spheres
        label:   name for display
    """
    center: Tuple[float, float, float]
    axis:   Tuple[float, float, float]
    half_h: float
    radius: float
    label:  str = "capsule"


@dataclass
class Plane:
    """
    Infinite flat plane defined by a point on the plane and a unit normal.

    Attributes:
        point:  (3,) any point on the plane
        normal: (3,) outward-facing normal (normalised internally)
        label:  name for display
    """
    point:  Tuple[float, float, float]
    normal: Tuple[float, float, float]
    label:  str = "plane"


# ---------------------------------------------------------------------------
# Scene container
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    spheres:  List[Sphere]  = field(default_factory=list)
    boxes:    List[Box]     = field(default_factory=list)
    capsules: List[Capsule] = field(default_factory=list)
    planes:   List[Plane]   = field(default_factory=list)

    # chainable add methods
    def add_sphere(self, center, radius, label="sphere") -> "Scene":
        self.spheres.append(Sphere(center, radius, label))
        return self

    def add_box(self, center, quaternion, half_extents, label="box") -> "Scene":
        self.boxes.append(Box(center, quaternion, half_extents, label))
        return self

    def add_capsule(self, center, axis, half_h, radius, label="capsule") -> "Scene":
        self.capsules.append(Capsule(center, axis, half_h, radius, label))
        return self

    def add_plane(self, point, normal, label="plane") -> "Scene":
        self.planes.append(Plane(point, normal, label))
        return self

    def summary(self) -> str:
        lines = [f"Scene — {len(self.spheres)} sphere(s), "
                 f"{len(self.boxes)} box(es), "
                 f"{len(self.capsules)} capsule(s), "
                 f"{len(self.planes)} plane(s)"]
        for s in self.spheres:
            lines.append(f"  Sphere   '{s.label}'  centre={s.center}  r={s.radius}")
        for b in self.boxes:
            lines.append(f"  Box      '{b.label}'  centre={b.center}  q={b.quaternion}  half={b.half_extents}")
        for c in self.capsules:
            lines.append(f"  Capsule  '{c.label}'  centre={c.center}  axis={c.axis}  hh={c.half_h:.2f}  r={c.radius:.2f}")
        for pl in self.planes:
            lines.append(f"  Plane    '{pl.label}'  point={pl.point}  n={pl.normal}")
        return "\n".join(lines)
