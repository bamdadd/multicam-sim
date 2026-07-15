"""Pinhole camera geometry: look-at rotation, projection, ray/segment tests.

Camera convention mirrored from multicam-occlusion@59f4906
(``src/multicam_occlusion/triangulation.py::look_at_rotation`` and
``build_ring_cameras``) so a manifest emitted here is consumed, convention for
convention, by that package's ``triangulate_dlt`` reader.

OpenCV pinhole, RDF axes:
  * camera +z = forward = (target - eye) normalised (viewing direction),
  * camera +x = right   = forward x up_world,
  * camera +y = down    = forward x right,
  * world up = +Z (Z-up).
  * R rows = [right, down, forward] maps world -> camera.
  * t = -R @ C  (world->camera translation; NOT the camera centre C).
  * P = K [R | t],  K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
  * projection: x ~ P [X; 1]; divide by w = third coordinate; w > 0 == in front.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: World up axis (Z-up), mirrored from multicam-occlusion@59f4906.
UP_WORLD: FloatArray = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def look_at_rotation(eye: FloatArray, target: FloatArray, up: FloatArray = UP_WORLD) -> FloatArray:
    """World->camera rotation ``R`` for a camera at ``eye`` looking at ``target``.

    OpenCV convention: +z forward (eye->target), +x right, +y down. Rows of ``R``
    are the camera axes in world coordinates. Mirrored from
    multicam-occlusion@59f4906.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=0)


def camera_translation(rotation: FloatArray, centre: FloatArray) -> FloatArray:
    """World->camera translation ``t = -R @ C`` from rotation and camera centre."""
    return -rotation @ centre


def camera_centre(rotation: FloatArray, translation: FloatArray) -> FloatArray:
    """Camera centre in world coordinates: ``C = -R^T @ t`` (inverse of the above)."""
    return -rotation.T @ translation


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> FloatArray:
    """Pinhole intrinsic matrix ``K``."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def projection_matrix(
    intrinsics: FloatArray, rotation: FloatArray, translation: FloatArray
) -> FloatArray:
    """Full ``3x4`` projection matrix ``P = K [R | t]``."""
    extrinsic = np.hstack([rotation, translation.reshape(3, 1)])
    result: FloatArray = intrinsics @ extrinsic
    return result


def project_point(proj_mat: FloatArray, point3d: FloatArray) -> tuple[FloatArray, float]:
    """Project a single world point through ``P``.

    Returns ``(uv, w)`` where ``uv`` is the pixel coordinate ``(u, v)`` and ``w``
    is the third homogeneous coordinate. ``w > 0`` means the point is in front of
    the camera. ``uv`` is ``x[:2] / w``; callers gate on ``w`` themselves (this
    stays total so it can report points behind the camera rather than raising).
    """
    homogeneous = np.append(point3d, 1.0)
    projected: FloatArray = proj_mat @ homogeneous
    w = float(projected[2])
    uv: FloatArray = projected[:2] / w
    return uv, w


def segment_intersects_sphere(
    a: FloatArray, b: FloatArray, centre: FloatArray, radius: float
) -> bool:
    """Does the closed segment ``a->b`` intersect the solid sphere?"""
    d = b - a
    f = a - centre
    aa = float(d @ d)
    if aa == 0.0:
        return bool(f @ f <= radius * radius)
    bb = 2.0 * float(f @ d)
    cc = float(f @ f) - radius * radius
    disc = bb * bb - 4.0 * aa * cc
    if disc < 0.0:
        return False
    sq = float(np.sqrt(disc))
    t1 = (-bb - sq) / (2.0 * aa)
    t2 = (-bb + sq) / (2.0 * aa)
    # intersection within the segment parameter range [0, 1]
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0) or (t1 < 0.0 < t2)


def segment_intersects_aabb(
    a: FloatArray, b: FloatArray, lower: FloatArray, upper: FloatArray
) -> bool:
    """Does the closed segment ``a->b`` intersect the axis-aligned box [lower, upper]?

    Slab method over the segment parameter ``t in [0, 1]``.
    """
    d = b - a
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        origin = float(a[axis])
        direction = float(d[axis])
        lo = float(lower[axis])
        hi = float(upper[axis])
        if abs(direction) < 1e-15:
            if origin < lo or origin > hi:
                return False
            continue
        inv = 1.0 / direction
        t0 = (lo - origin) * inv
        t1 = (hi - origin) * inv
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_min > t_max:
            return False
    return True
