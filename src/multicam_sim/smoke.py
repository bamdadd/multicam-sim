"""The smoke scene — the whole point of the foundational vertical.

Hand-specified, GL-free: three ring cameras, one point moving on a straight
path, one sphere positioned on CAMERA 1's sightline so it occludes that view
during a middle interval while cameras 0 and 2 keep the point. The manifest is
computed analytically (pure projection + boolean visibility); the smoke test
then triangulates the occluded frame from the OTHER TWO views through the real
multicam-occlusion DLT and recovers ground truth.

Geometry is chosen so the point stays in front of and inside all three frames
for every frame — the only reason cam 1 goes ``visible=False`` is the sphere.
"""

from __future__ import annotations

import numpy as np

from .cameras import Camera, Intrinsics
from .entities import Entity, EntityFrame
from .occluders import Sphere
from .scene import Scene

_RADIUS = 4.0
_HEIGHT = 1.5
_FOCAL = 800.0
_WIDTH = 640
_HEIGHT_PX = 480
_LOOK_AT = np.array([0.0, 0.0, 0.5], dtype=np.float64)
_NUM_FRAMES = 11
_FPS = 30.0

# Point path: a straight sweep in y at fixed x, z.
_PATH_X = 0.0
_PATH_Z = 0.5
_PATH_Y0 = -0.6
_PATH_Y1 = 0.6

# Occluder: a sphere on camera 1's sightline to the mid-path point.
_OCC_T = 0.15  # fraction from the point toward camera 1's centre
_OCC_RADIUS = 0.15


def _ring_eye(index: int) -> np.ndarray:
    angle = 2.0 * np.pi * index / 3.0
    return np.array([_RADIUS * np.cos(angle), _RADIUS * np.sin(angle), _HEIGHT], dtype=np.float64)


def _point_at(frame: int) -> np.ndarray:
    frac = frame / (_NUM_FRAMES - 1)
    y = _PATH_Y0 + frac * (_PATH_Y1 - _PATH_Y0)
    return np.array([_PATH_X, y, _PATH_Z], dtype=np.float64)


def build_smoke_scene() -> Scene:
    """Construct the deterministic smoke scene (3 cameras, 1 moving point, 1 sphere)."""
    intrinsics = Intrinsics.from_focal(_FOCAL, _WIDTH, _HEIGHT_PX)
    cameras = [Camera.look_at(i, intrinsics, _ring_eye(i), _LOOK_AT) for i in range(3)]

    frames = [
        EntityFrame(frame=f, points={"center": _point_at(f).tolist()}) for f in range(_NUM_FRAMES)
    ]
    entity = Entity(id="obj", frames=frames)

    # Place the sphere on camera 1's line of sight to the mid-path point.
    mid_point = _point_at(_NUM_FRAMES // 2)
    cam1_centre = _ring_eye(1)
    occ_centre = mid_point + _OCC_T * (cam1_centre - mid_point)
    sphere = Sphere(center=occ_centre.tolist(), radius=_OCC_RADIUS)

    return Scene(
        fps=_FPS,
        num_frames=_NUM_FRAMES,
        cameras=cameras,
        entities=[entity],
        occluders=[sphere],
    )
