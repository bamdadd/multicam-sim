"""2D keypoint/skeleton overlay exporter for a manifest.

This is a **visualisation layer** over the existing ``uv`` / ``visible`` fields —
it draws per-camera 2D projections onto blank image-sized canvases. It is NOT a
renderer of the 3D scene.

Dependencies are kept optional and imported lazily so that ``import multicam_sim``
never pulls in drawing or video libraries:

* ``pillow`` is required for both output formats (``pip install multicam-sim[overlay]``).
* ``imageio[ffmpeg]`` is required only for MP4 output
  (``pip install multicam-sim[overlay-mp4]``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Visual styling
# ---------------------------------------------------------------------------
_BG = (255, 255, 255)
_VISIBLE_POINT_FILL = (0, 200, 0)
_VISIBLE_POINT_OUTLINE = (0, 100, 0)
_OCCLUDED_POINT_OUTLINE = (255, 0, 0)
_OCCLUDED_POINT_CROSS = (255, 0, 0)
_VISIBLE_EDGE = (0, 120, 255)
_OCCLUDED_EDGE = (255, 100, 0)

_POINT_RADIUS = 4
_EDGE_WIDTH = 2
_DASH_LENGTH = 5


def export_overlay(
    manifest: dict[str, Any],
    out_dir: str | Path,
    fmt: Literal["frames", "mp4"] = "frames",
) -> None:
    """Draw per-camera 2D keypoint/skeleton overlays for ``manifest``.

    For every camera and every frame, a blank ``(height, width, 3)`` canvas is
    filled with the projected points and entity edges from the manifest.

    Args:
        manifest: A manifest dict conforming to the DESIGN.md JSON contract.
        out_dir: Directory to write output into. Created if it does not exist.
        fmt:
            * ``"frames"`` — one PNG per (camera, frame):
              ``cam{cam_id}/frame_{NNNN}.png``.
            * ``"mp4"`` — one MP4 per camera: ``cam{cam_id}.mp4``.

    Raises:
        ImportError: If the required optional extra is not installed for the
            chosen format.
        ValueError: If ``fmt`` is not ``"frames"`` or ``"mp4"``.
    """
    if fmt not in {"frames", "mp4"}:
        raise ValueError(f"fmt must be 'frames' or 'mp4', got {fmt!r}")

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "export_overlay needs the 'overlay' extra: pip install multicam-sim[overlay]"
        ) from exc

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cameras = manifest["cameras"]
    num_frames = int(manifest["num_frames"])
    fps = float(manifest["fps"])
    entities = manifest.get("entities", [])

    # Pre-compute per-camera frames as lists of PIL Images.
    cam_frames: list[tuple[dict[str, Any], list[Any]]] = []
    for cam in cameras:
        width = int(cam["width"])
        height = int(cam["height"])
        frames: list[Any] = [Image.new("RGB", (width, height), _BG) for _ in range(num_frames)]
        cam_frames.append((cam, frames))

    for entity in entities:
        edges = entity.get("edges") or []
        for frame_idx in range(num_frames):
            frame_data = next(
                (f for f in entity["frames"] if f["frame"] == frame_idx),
                None,
            )
            if frame_data is None:
                continue
            points = frame_data["points"]

            for cam, frames in cam_frames:
                draw = ImageDraw.Draw(frames[frame_idx])
                cam_id = cam["id"]

                # name -> (uv, visible)
                observations: dict[str, tuple[tuple[float, float], bool]] = {}
                for name, data in points.items():
                    per_cam = next(
                        (p for p in data["per_cam"] if p["cam"] == cam_id),
                        None,
                    )
                    if per_cam is None:
                        continue
                    u, v = float(per_cam["uv"][0]), float(per_cam["uv"][1])
                    observations[name] = ((u, v), bool(per_cam["visible"]))

                # Draw edges first so points sit on top.
                for a_name, b_name in edges:
                    a = observations.get(a_name)
                    b = observations.get(b_name)
                    if a is None or b is None:
                        continue
                    (u0, v0), vis_a = a
                    (u1, v1), vis_b = b
                    if vis_a and vis_b:
                        draw.line(
                            [(u0, v0), (u1, v1)],
                            fill=_VISIBLE_EDGE,
                            width=_EDGE_WIDTH,
                        )
                    else:
                        _draw_dashed_line(
                            draw,
                            (u0, v0),
                            (u1, v1),
                            fill=_OCCLUDED_EDGE,
                            width=_EDGE_WIDTH,
                        )

                # Draw points.
                for (u, v), visible in observations.values():
                    if visible:
                        _draw_filled_circle(
                            draw,
                            (u, v),
                            radius=_POINT_RADIUS,
                            fill=_VISIBLE_POINT_FILL,
                            outline=_VISIBLE_POINT_OUTLINE,
                        )
                    else:
                        _draw_occluded_marker(draw, (u, v), radius=_POINT_RADIUS)

    # -----------------------------------------------------------------------
    # Write output.
    # -----------------------------------------------------------------------
    if fmt == "frames":
        for cam, frames in cam_frames:
            cam_dir = out_path / f"cam{cam['id']}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx, img in enumerate(frames):
                img.save(cam_dir / f"frame_{frame_idx:04d}.png")
        return

    # fmt == "mp4"
    try:
        import imageio
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "export_overlay(fmt='mp4') needs the 'overlay-mp4' extra: "
            "pip install multicam-sim[overlay-mp4]"
        ) from exc

    for cam, frames in cam_frames:
        path = out_path / f"cam{cam['id']}.mp4"
        imageio.mimsave(str(path), [np.asarray(f) for f in frames], fps=fps)


def _draw_filled_circle(
    draw: Any,
    center: tuple[float, float],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
) -> None:
    x, y = center
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=fill, outline=outline)


def _draw_occluded_marker(
    draw: Any,
    center: tuple[float, float],
    radius: int,
) -> None:
    x, y = center
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, outline=_OCCLUDED_POINT_OUTLINE)
    draw.line(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=_OCCLUDED_POINT_CROSS,
        width=1,
    )
    draw.line(
        [(x + radius, y - radius), (x - radius, y + radius)],
        fill=_OCCLUDED_POINT_CROSS,
        width=1,
    )


def _draw_dashed_line(
    draw: Any,
    xy0: tuple[float, float],
    xy1: tuple[float, float],
    fill: tuple[int, int, int],
    width: int,
    dash_length: int = _DASH_LENGTH,
) -> None:
    x0, y0 = xy0
    x1, y1 = xy1
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    step = dash_length * 2
    start = 0.0
    while start < length:
        end = min(start + dash_length, length)
        sx = x0 + dx * start
        sy = y0 + dy * start
        ex = x0 + dx * end
        ey = y0 + dy * end
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
        start += step
