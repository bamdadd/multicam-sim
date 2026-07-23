"""multicam-sim — typed multi-camera scene simulator emitting a triangulation-ready manifest.

Camera convention mirrored from multicam-occlusion@59f4906 (see
:mod:`multicam_sim.geometry`).
"""

from __future__ import annotations

from .cameras import Camera, Intrinsics
from .dropout import SensorDropout
from .entities import Entity, EntityFrame
from .manifest import (
    AssumedCalibration,
    CameraManifest,
    EntityManifest,
    FrameObs,
    Manifest,
    PerCamObs,
    PointObs,
    build_manifest,
    write_manifest,
)
from .mtmc import build_mtmc_scene
from .noise import CalibrationDrift, NoiseModel, PixelNoise
from .occluders import Box, Cylinder, HandKeyframe, HandOccluder, Occluder, Sphere
from .overlay import export_overlay
from .pose import (
    COCO17_EDGES,
    COCO17_JOINTS,
    MeshBackend,
    PoseFrame,
    PoseTrajectory,
    Skeleton,
)
from .scene import Scene
from .smoke import build_multi_entity_scene, build_pose_smoke_scene, build_smoke_scene
from .topology import CameraTopology, Station, TransitEdge
from .validation import validate_manifest
from .visibility import silhouette_visible_fraction

__all__ = [
    "COCO17_EDGES",
    "COCO17_JOINTS",
    "AssumedCalibration",
    "Box",
    "CalibrationDrift",
    "Cylinder",
    "Camera",
    "CameraManifest",
    "CameraTopology",
    "Entity",
    "EntityFrame",
    "EntityManifest",
    "FrameObs",
    "HandKeyframe",
    "HandOccluder",
    "Intrinsics",
    "Manifest",
    "MeshBackend",
    "NoiseModel",
    "Occluder",
    "PerCamObs",
    "PixelNoise",
    "PointObs",
    "PoseFrame",
    "PoseTrajectory",
    "Scene",
    "SensorDropout",
    "Skeleton",
    "Sphere",
    "Station",
    "TransitEdge",
    "build_manifest",
    "build_mtmc_scene",
    "silhouette_visible_fraction",
    "build_multi_entity_scene",
    "build_pose_smoke_scene",
    "build_smoke_scene",
    "export_overlay",
    "validate_manifest",
    "write_manifest",
]

__version__ = "0.1.0"
