"""Render a scene manifest as one 3D world view: camera locations + trajectory.

A single static 3D figure showing where every camera sits in the world (drawn as
a small view-cone / frustum pointing along its optical axis), a ground plane, and
each entity's ground-truth 3D trajectory over the frames. This is the "see the
space" view -- unlike the per-camera tiles of ``record_multiview.py``, it draws
the whole scene once so camera coverage and overlap are directly visible.

Geometry follows ``multicam_sim.cameras.Camera``: ``R`` is the world->camera
rotation with rows ``[right, down, forward]`` (OpenCV, +z forward) and ``t`` is
the world->camera translation, so the camera centre is ``C = -R^T @ t`` and the
frustum edges are the back-projected image corners ``R^T @ (K^-1 @ [u, v, 1])``.
World up is +z; the ground plane is drawn at z = 0.

Works on any manifest with the standard schema (``cameras[*].{K,R,t,width,
height}`` and ``entities[*].frames[*].points[*].xyz_gt``); defaults to the
bundled MTMC golden fixture.

``matplotlib`` is imported lazily inside :func:`main` (it is not a package
dependency), mirroring the lazy-import pattern in ``scripts/record_multiview.py``.
Run::

    uv run --with matplotlib python scripts/view_scene_3d.py
    uv run --with matplotlib python scripts/view_scene_3d.py --manifest PATH --out PATH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MANIFEST = _ROOT / "tests" / "fixtures" / "manifest_golden" / "mtmc.json"
_DEFAULT_OUT = _ROOT / "docs" / "assets" / "scene_3d.png"

# How far in front of each camera to draw the frustum apex (world units).
_FRUSTUM_DEPTH = 1.2


def _camera_centre(
    rotation: NDArray[np.float64], translation: NDArray[np.float64]
) -> NDArray[np.float64]:
    """World-space camera centre ``C = -R^T @ t``."""
    return -rotation.T @ translation


def _frustum_corners(
    intrinsics: NDArray[np.float64],
    rotation: NDArray[np.float64],
    centre: NDArray[np.float64],
    width: float,
    height: float,
    depth: float,
) -> NDArray[np.float64]:
    """The four image-corner rays back-projected to ``depth`` in world space.

    A pixel ``[u, v, 1]`` back-projects to the world ray direction
    ``R^T @ (K^-1 @ [u, v, 1])``; scaling so the forward (+z camera) component
    equals ``depth`` puts the corner a fixed distance in front of the camera.
    """
    inv_k = np.linalg.inv(intrinsics)
    corners_px = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float64,
    )
    out = np.empty((4, 3), dtype=np.float64)
    for i, (u, v) in enumerate(corners_px):
        cam_ray = inv_k @ np.array([u, v, 1.0], dtype=np.float64)
        cam_ray = cam_ray / cam_ray[2] * depth  # forward (+z) component == depth
        out[i] = centre + rotation.T @ cam_ray
    return out


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _trajectories(manifest: dict[str, Any]) -> list[tuple[str, NDArray[np.float64]]]:
    """Per-(entity, point) ground-truth 3D polylines over the frames."""
    out: list[tuple[str, NDArray[np.float64]]] = []
    for entity in manifest.get("entities", []):
        by_point: dict[str, list[list[float]]] = {}
        for frame in entity.get("frames", []):
            for name, point in frame.get("points", {}).items():
                xyz = point.get("xyz_gt")
                if xyz is not None:
                    by_point.setdefault(name, []).append(list(xyz))
        for name, coords in by_point.items():
            label = entity.get("id", "object")
            label = label if name == "center" else f"{label}:{name}"
            out.append((str(label), np.asarray(coords, dtype=np.float64)))
    return out


def render(manifest: dict[str, Any], out_path: Path) -> Path:
    """Render the 3D scene view to ``out_path`` (PNG) and return the path."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe: no display needed
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    all_points: list[NDArray[np.float64]] = []

    # Cameras: centre marker + a wireframe frustum to the four image corners.
    for cam in manifest.get("cameras", []):
        intrinsics = np.asarray(cam["K"], dtype=np.float64)
        rotation = np.asarray(cam["R"], dtype=np.float64)
        translation = np.asarray(cam["t"], dtype=np.float64)
        centre = _camera_centre(rotation, translation)
        corners = _frustum_corners(
            intrinsics,
            rotation,
            centre,
            float(cam["width"]),
            float(cam["height"]),
            _FRUSTUM_DEPTH,
        )
        ax.scatter(*centre, color="crimson", s=40, depthshade=False)
        ax.text(*centre, f"  cam {cam['id']}", color="crimson", fontsize=8)
        for corner in corners:
            edge = np.vstack([centre, corner])
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color="crimson", linewidth=0.8)
        loop = np.vstack([corners, corners[0]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="crimson", linewidth=0.8)
        all_points.append(centre[None, :])
        all_points.append(corners)

    # Trajectories: one polyline per entity point.
    for label, coords in _trajectories(manifest):
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"trajectory: {label}",
        )
        all_points.append(coords)

    if not all_points:
        raise ValueError("manifest has no cameras or trajectories to draw")
    pts = np.vstack(all_points)

    # Ground plane at z = 0 spanning the scene's xy extent.
    pad = 1.0
    x_min, y_min = pts[:, 0].min() - pad, pts[:, 1].min() - pad
    x_max, y_max = pts[:, 0].max() + pad, pts[:, 1].max() + pad
    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, 2),
        np.linspace(y_min, y_max, 2),
    )
    ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.12, color="steelblue")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z (up)")
    ax.set_title("Scene: camera locations and object trajectory")
    ax.legend(loc="upper left", fontsize=8)

    # Equal aspect over the combined bounds so directions are not skewed.
    span = pts.max(axis=0) - pts.min(axis=0)
    span = np.where(span > 0, span, 1.0)
    ax.set_box_aspect(tuple(span))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="scene manifest JSON (default: bundled MTMC golden fixture)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="output PNG path (default: docs/assets/scene_3d.png)",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    out_path = render(manifest, args.out)
    size = out_path.stat().st_size
    print(f"wrote {out_path}  ({size} bytes)")


if __name__ == "__main__":
    main()
