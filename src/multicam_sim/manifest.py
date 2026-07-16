"""Build the manifest — the shared contract of record.

The manifest is the hand-off to a triangulation reader (multicam-occlusion). For
every named point of every entity at every frame it records the ground-truth
world coordinate and, per camera, the pixel observation plus TWO occlusion
signals kept deliberately distinct:

* ``visible`` (bool): the hard DLT contract — the point is in front of the
  camera, inside the image, and its segment to the camera centre is not blocked
  by any occluder. This is what a consumer masks on.
* ``occ_frac`` (float, optional): a continuous difficulty knob — the fraction of
  a small jittered sample around the point whose sightline is blocked. Never
  feeds the mask; it just grades how marginal an occlusion is.

Floats are emitted at full double precision (no rounding/formatting) so a
consumer that rebuilds ``P = K [R | t]`` recovers ground truth to ~machine eps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .cameras import Camera
from .geometry import FloatArray
from .occluders import Occluder
from .scene import Scene

CONVENTION = "opencv_rdf"

# Fixed, deterministic jitter offsets (unit directions) for occ_frac sampling —
# no RNG, so the difficulty knob is reproducible.
_JITTER_DIRS: FloatArray = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def _any_blocks(occluders: list[Occluder], a: FloatArray, b: FloatArray) -> bool:
    return any(occ.blocks_segment(a, b) for occ in occluders)


def occlusion_fraction(
    camera: Camera,
    point3d: FloatArray,
    occluders: list[Occluder],
    jitter: float = 0.05,
) -> float:
    """Continuous occlusion: fraction of jittered sightlines that are blocked.

    Samples the point plus six small axis offsets; returns the blocked fraction
    in ``[0, 1]``. ``0.0`` when there are no occluders. Distinct from ``visible``
    and never used as a triangulation mask.
    """
    if not occluders:
        return 0.0
    centre = camera.centre()
    samples = np.vstack([point3d[None, :], point3d[None, :] + jitter * _JITTER_DIRS])
    blocked = sum(1 for s in samples if _any_blocks(occluders, s, centre))
    return float(blocked / samples.shape[0])


def observe(camera: Camera, point3d: FloatArray, occluders: list[Occluder]) -> dict[str, Any]:
    """One camera's observation of a world point.

    Returns ``{cam, uv, in_view, visible, occ_frac}``:

    * ``in_view``  — projects IN FRONT (w > 0) AND inside the image bounds. A point
      behind the camera or off the sensor is ``in_view=False``.
    * ``visible``  — the DLT mask: ``in_view AND not occluded`` (so ``visible``
      implies ``in_view``). A point in a blind gap is ``in_view=False`` on every
      camera and therefore ``visible=False`` everywhere.

    Non-raising: an out-of-frame or behind-camera point is labelled, not an error.
    ``uv`` is sanitised to finite values (behind/at-plane projections would be
    non-finite) so the manifest is always strict JSON.
    """
    uv, w = camera.project(point3d)
    in_front = w > 0.0
    in_image = bool(in_front and camera.in_image(uv))
    in_view = bool(in_front and in_image)
    unoccluded = not _any_blocks(occluders, point3d, camera.centre())
    visible = bool(in_view and unoccluded)
    u, v = float(uv[0]), float(uv[1])
    if not (in_front and np.isfinite(u) and np.isfinite(v)):
        # behind / at the image plane: pixel is meaningless, keep JSON finite.
        u, v = 0.0, 0.0
    return {
        "cam": camera.id,
        "uv": [u, v],
        "in_view": in_view,
        "visible": visible,
        "occ_frac": occlusion_fraction(camera, point3d, occluders),
    }


def camera_entry(camera: Camera) -> dict[str, Any]:
    """Serialise a camera for the manifest: full-precision K, R, t + convention."""
    return {
        "id": camera.id,
        "K": camera.intrinsics.matrix().tolist(),
        "R": [list(row) for row in camera.R],
        "t": list(camera.t),
        "width": camera.intrinsics.width,
        "height": camera.intrinsics.height,
        "convention": CONVENTION,
    }


def build_manifest(scene: Scene) -> dict[str, Any]:
    """Compute the full manifest for ``scene`` (pure projection + boolean
    visibility; no renderer)."""
    occluders: list[Occluder] = list(scene.occluders)
    entities_out: list[dict[str, Any]] = []
    for entity in scene.entities:
        frames_out: list[dict[str, Any]] = []
        for frame in entity.frames:
            points_out: dict[str, Any] = {}
            for name, xyz in frame.points.items():
                point3d = np.asarray(xyz, dtype=np.float64)
                points_out[name] = {
                    "xyz_gt": [float(c) for c in xyz],
                    "per_cam": [observe(cam, point3d, occluders) for cam in scene.cameras],
                }
            frames_out.append({"frame": frame.frame, "points": points_out})
        entry: dict[str, Any] = {"id": entity.id, "frames": frames_out}
        if entity.edges is not None:
            entry["edges"] = [list(e) for e in entity.edges]
        entities_out.append(entry)

    manifest: dict[str, Any] = {
        "cameras": [camera_entry(cam) for cam in scene.cameras],
        "fps": scene.fps,
        "num_frames": scene.num_frames,
        "entities": entities_out,
    }
    # Optional MTMC topology (stations + directed transit edges). Emitted only
    # when the scene declares one, so single-station manifests stay unchanged.
    if scene.topology is not None:
        manifest["topology"] = scene.topology.to_manifest()
    return manifest


def write_manifest(scene: Scene, path: str | Path) -> dict[str, Any]:
    """Build the manifest and write it to ``path`` as JSON (full precision).

    ``allow_nan=False`` so a non-finite pixel fails loudly rather than emitting
    ``Infinity``/``NaN`` (invalid strict JSON) that a consumer would choke on.
    """
    manifest = build_manifest(scene)
    Path(path).write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest
