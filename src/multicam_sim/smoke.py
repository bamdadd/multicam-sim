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
from .pose import PoseFrame, PoseTrajectory, Skeleton
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


# -----------------------------------------------------------------------------
# Pose smoke scene — the COCO-17 twin of build_smoke_scene.
# -----------------------------------------------------------------------------

# A standing person centred at the origin with Z-up, scaled to fit the smoke
# camera framing. The left wrist is deliberately extended left (-x) so a small
# sphere placed on camera 1's sightline blocks *only* that joint for *only*
# camera 1; every other joint stays in view and unoccluded for every camera.
_POSE_JOINTS: dict[str, list[float]] = {
    "nose": [0.0, 0.0, 1.00],
    "left_eye": [-0.04, 0.0, 1.02],
    "right_eye": [0.04, 0.0, 1.02],
    "left_ear": [-0.08, 0.0, 1.00],
    "right_ear": [0.08, 0.0, 1.00],
    "left_shoulder": [-0.18, 0.0, 0.85],
    "right_shoulder": [0.18, 0.0, 0.85],
    "left_elbow": [-0.35, 0.04, 0.82],
    "right_elbow": [0.35, -0.04, 0.82],
    "left_wrist": [-0.60, 0.08, 0.80],
    "right_wrist": [0.60, -0.08, 0.80],
    "left_hip": [-0.12, 0.0, 0.50],
    "right_hip": [0.12, 0.0, 0.50],
    "left_knee": [-0.12, 0.0, 0.25],
    "right_knee": [0.12, 0.0, 0.25],
    "left_ankle": [-0.12, 0.0, 0.00],
    "right_ankle": [0.12, 0.0, 0.00],
}

# Sphere placed on camera 1's line of sight to the left wrist.
_POSE_OCC_TARGET = "left_wrist"
_POSE_OCC_T = 0.20  # fraction from the joint toward camera 1's centre
_POSE_OCC_RADIUS = 0.10


def _coco17_pose() -> PoseTrajectory:
    skeleton = Skeleton.coco17()
    frames = [PoseFrame(frame=f, joints=_POSE_JOINTS) for f in range(_NUM_FRAMES)]
    return PoseTrajectory(id="person", skeleton=skeleton, frames=frames)


def build_pose_smoke_scene() -> Scene:
    """Construct the deterministic COCO-17 pose smoke scene.

    Three ring cameras, one static standing pose, and one small sphere on camera 1's
    sightline to the left wrist. The sphere occludes ``left_wrist`` for camera 1
    while cameras 0 and 2 keep it, demonstrating per-joint, per-camera visibility.
    """
    intrinsics = Intrinsics.from_focal(_FOCAL, _WIDTH, _HEIGHT_PX)
    cameras = [Camera.look_at(i, intrinsics, _ring_eye(i), _LOOK_AT) for i in range(3)]

    entity = _coco17_pose().to_entity()

    target = np.asarray(_POSE_JOINTS[_POSE_OCC_TARGET], dtype=np.float64)
    cam1_centre = _ring_eye(1)
    occ_centre = target + _POSE_OCC_T * (cam1_centre - target)
    sphere = Sphere(center=occ_centre.tolist(), radius=_POSE_OCC_RADIUS)

    return Scene(
        fps=_FPS,
        num_frames=_NUM_FRAMES,
        cameras=cameras,
        entities=[entity],
        occluders=[sphere],
    )
