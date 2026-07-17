"""Analytic, renderer-free image-space visibility of an object silhouette.

``occ_frac`` (see :mod:`multicam_sim.manifest`) is the legacy *point* estimator:
the fraction of a small 3D jittered neighbourhood whose sightline to the camera
centre is blocked. It stays frozen (downstream contract + byte-identical golden).

``visible_fraction`` is the *area* quantity of record for object-extent occlusion:
the fraction of the object's projected silhouette that is NOT covered by nearer
occluder silhouettes, computed in image space. It is tied to ``occ_frac`` only at
the hard endpoints and by monotonicity — NOT equal to ``1 - occ_frac`` (one is a
3D-ball ray metric, the other a 2D silhouette-area integral; they can disagree in
the partial regime, e.g. when the object is smaller than the jitter ball).

Honesty about exactness: the endpoints are exact — a fully-covering nearer
occluder gives exactly ``0.0`` and no nearer occluder gives exactly ``1.0``. The
partial values are **model-approximate**: the object and each occluder are modelled
as their bounding discs (paraxial radius ``r_px ~= r * fx / w``) and disc–disc lens
areas are combined by a clamped sum. This is deterministic and renderer-free; the
rasterizer's visible-pixel count is an empirical CROSS-CHECK of this value, never
its source. Pixels are not the contract.
"""

from __future__ import annotations

import math

import numpy as np

from .cameras import Camera
from .geometry import FloatArray
from .occluders import Box, Occluder, Sphere


def _bounding_radius(occ: Occluder) -> float | None:
    """World-space bounding-sphere radius of a static occluder, or ``None``.

    Only static solids have a silhouette here; a time-varying occluder must be
    resolved with :meth:`Occluder.at_frame` before reaching this function.
    """
    if isinstance(occ, Sphere):
        return occ.radius
    if isinstance(occ, Box):
        return float(np.linalg.norm(np.asarray(occ.half_extents, dtype=np.float64)))
    return None


def _occluder_center(occ: Occluder) -> FloatArray | None:
    if isinstance(occ, (Box, Sphere)):
        return np.asarray(occ.center, dtype=np.float64)
    return None


def _disc_overlap_area(c0: FloatArray, r0: float, c1: FloatArray, r1: float) -> float:
    """Area of the intersection of two image-space discs (closed-form lens)."""
    if r0 <= 0.0 or r1 <= 0.0:
        return 0.0
    d = float(np.linalg.norm(c0 - c1))
    if d >= r0 + r1:
        return 0.0  # disjoint
    if d <= abs(r0 - r1):
        return math.pi * min(r0, r1) ** 2  # one disc inside the other
    r0sq, r1sq = r0 * r0, r1 * r1
    a0 = math.acos((d * d + r0sq - r1sq) / (2.0 * d * r0))
    a1 = math.acos((d * d + r1sq - r0sq) / (2.0 * d * r1))
    triangle = 0.5 * math.sqrt((-d + r0 + r1) * (d + r0 - r1) * (d - r0 + r1) * (d + r0 + r1))
    return r0sq * a0 + r1sq * a1 - triangle


def silhouette_visible_fraction(
    camera: Camera,
    point3d: FloatArray,
    object_radius: float,
    occluders: list[Occluder],
) -> float:
    """Fraction of the object's silhouette visible on ``camera`` in ``[0, 1]``.

    ``occluders`` must already be the per-frame static solids (resolved via
    :meth:`Occluder.at_frame`). Only occluders strictly NEARER than the object
    (smaller camera-space depth) can cover it; coverage is the clamped sum of
    each nearer occluder disc's overlap with the object disc.

    Endpoints are exact (``0.0`` fully covered, ``1.0`` no nearer occluder);
    partial values are the bounding-disc model described in the module docstring.
    A behind-camera object returns ``0.0`` (no silhouette to see). This measures
    occluder coverage only, NOT image-bounds clipping — an in-front point with no
    nearer occluder returns ``1.0`` even if it projects outside the frame (use the
    ``in_view`` flag for bounds).
    """
    if object_radius <= 0.0:
        raise ValueError("object_radius must be > 0")
    obj_uv, obj_w = camera.project(point3d)
    if obj_w <= 0.0:
        return 0.0
    fx = camera.intrinsics.fx
    obj_r_px = object_radius * fx / obj_w
    if obj_r_px <= 0.0:
        return 1.0
    # Same token sequence as the full-containment branch of _disc_overlap_area
    # (``math.pi * min(r0, r1) ** 2``) so a fully-covering occluder yields
    # ``covered == object_area`` bit-identically -> exactly 0.0 on every arch.
    object_area = math.pi * obj_r_px**2

    covered = 0.0
    for occ in occluders:
        radius = _bounding_radius(occ)
        center = _occluder_center(occ)
        if radius is None or center is None:
            continue
        occ_uv, occ_w = camera.project(center)
        if occ_w <= 0.0 or occ_w >= obj_w:
            continue  # behind camera, or not nearer than the object
        occ_r_px = radius * fx / occ_w
        covered += _disc_overlap_area(obj_uv, obj_r_px, occ_uv, occ_r_px)

    fraction = 1.0 - min(covered, object_area) / object_area
    return float(min(1.0, max(0.0, fraction)))
