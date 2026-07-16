"""multicam-sim — typed multi-camera scene simulator emitting a triangulation-ready manifest.

Camera convention mirrored from multicam-occlusion@59f4906 (see
:mod:`multicam_sim.geometry`).
"""

from __future__ import annotations

from .cameras import Camera, Intrinsics
from .entities import Entity, EntityFrame
from .manifest import build_manifest, write_manifest
from .mtmc import build_mtmc_scene
from .occluders import Box, Occluder, Sphere
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
from .smoke import build_smoke_scene
from .topology import CameraTopology, Station, TransitEdge
from .validation import validate_manifest

__all__ = [
    "COCO17_EDGES",
    "COCO17_JOINTS",
    "Box",
    "Camera",
    "CameraTopology",
    "Entity",
    "EntityFrame",
    "Intrinsics",
    "MeshBackend",
    "Occluder",
    "PoseFrame",
    "PoseTrajectory",
    "Scene",
    "Skeleton",
    "Sphere",
    "Station",
    "TransitEdge",
    "build_manifest",
    "build_mtmc_scene",
    "build_smoke_scene",
    "export_overlay",
    "validate_manifest",
    "write_manifest",
]

__version__ = "0.1.0"
