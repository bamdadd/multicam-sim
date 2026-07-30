"""multicam-sim — typed multi-camera scene simulator emitting a triangulation-ready manifest.

Camera convention mirrored from multicam-occlusion@59f4906 (see
:mod:`multicam_sim.geometry`).
"""

from __future__ import annotations

from .annotations import (
    CocoAnnotation,
    CocoCategory,
    CocoDataset,
    CocoImage,
    YoloDataset,
    YoloLabel,
    export_coco,
    export_yolo,
    write_coco,
    write_yolo,
)
from .cameras import Camera, Intrinsics
from .dropout import SensorDropout
from .entities import Entity, EntityFrame
from .groups import (
    GroupFrameMembership,
    GroupMembership,
    build_group_formation_scene,
    compute_group_membership,
    write_group_json,
)
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
from .occluders import Box, Cylinder, HandKeyframe, HandOccluder, Occluder, PathOccluder, Sphere
from .overlay import export_overlay
from .pose import (
    COCO17_EDGES,
    COCO17_JOINTS,
    MeshBackend,
    PoseFrame,
    PoseTrajectory,
    Skeleton,
)
from .possession import (
    InteractionEvent,
    PossessionSegment,
    PossessionTimeline,
    write_possession_json,
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
    "CocoAnnotation",
    "CocoCategory",
    "CocoDataset",
    "CocoImage",
    "Entity",
    "EntityFrame",
    "EntityManifest",
    "FrameObs",
    "GroupFrameMembership",
    "GroupMembership",
    "HandKeyframe",
    "HandOccluder",
    "InteractionEvent",
    "Intrinsics",
    "Manifest",
    "MeshBackend",
    "NoiseModel",
    "Occluder",
    "PathOccluder",
    "PerCamObs",
    "PixelNoise",
    "PointObs",
    "PoseFrame",
    "PoseTrajectory",
    "PossessionSegment",
    "PossessionTimeline",
    "Scene",
    "SensorDropout",
    "Skeleton",
    "Sphere",
    "Station",
    "TransitEdge",
    "YoloDataset",
    "YoloLabel",
    "build_group_formation_scene",
    "build_manifest",
    "build_mtmc_scene",
    "silhouette_visible_fraction",
    "build_multi_entity_scene",
    "build_pose_smoke_scene",
    "build_smoke_scene",
    "compute_group_membership",
    "export_coco",
    "export_overlay",
    "export_yolo",
    "validate_manifest",
    "write_coco",
    "write_group_json",
    "write_manifest",
    "write_possession_json",
    "write_yolo",
]

__version__ = "0.1.0"
