"""multicam-sim — typed multi-camera scene simulator emitting a triangulation-ready manifest.

Camera convention mirrored from multicam-occlusion@59f4906 (see
:mod:`multicam_sim.geometry`).
"""

from __future__ import annotations

from .cameras import Camera, Intrinsics
from .entities import Entity, EntityFrame
from .manifest import build_manifest, write_manifest
from .occluders import Box, Occluder, Sphere
from .scene import Scene
from .smoke import build_smoke_scene

__all__ = [
    "Box",
    "Camera",
    "Entity",
    "EntityFrame",
    "Intrinsics",
    "Occluder",
    "Scene",
    "Sphere",
    "build_manifest",
    "build_smoke_scene",
    "write_manifest",
]

__version__ = "0.1.0"
