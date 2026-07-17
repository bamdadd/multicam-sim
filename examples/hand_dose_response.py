"""Hero artifact: a hand sweep traces a visible-fraction occlusion dose-response.

Deterministic, CPU-only, no GL, no Blender. Run it directly::

    python examples/hand_dose_response.py    # or: uv run python examples/hand_dose_response.py

A hand-proxy (:class:`~multicam_sim.dsl.HandSweep`) sweeps across one camera's view
of a target on a work surface — entering one side, crossing centre, exiting the
other. The manifest's analytic, renderer-free ``visible_fraction`` (opt-in via
``build_manifest(object_radius=...)``) traces the classic U-shaped occlusion
dose-response (1 -> ~0 -> 1) on the blocked camera, while the other cameras keep
seeing the target. This is the value that ties to multicam-occlusion's
dose-response consumer.

Emits, next to this file (``--out`` to change):

* ``hand_dose_response.json`` — the per-frame visible_fraction curve per camera;
* ``hand_frame_XX.png`` — the blocked camera's view at a few frames (rasterized,
  an empirical look at the same occlusion the analytic curve reports).
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

from multicam_sim import build_manifest
from multicam_sim.dsl import CameraRig, HandSweep, SceneBuilder
from multicam_sim.dsl import Path as MotionPath
from multicam_sim.dsl.raster import RasterizerBackend, RasterizerConfig
from multicam_sim.geometry import FloatArray
from multicam_sim.scene import Scene

FPS = 30.0
NUM_FRAMES = 11
BLOCKED_CAM = 0
OBJECT_RADIUS = 0.06


def write_png(path: Path, image: FloatArray) -> None:
    """Write an ``(H, W, 3)`` uint8 array to ``path`` as PNG (stdlib only)."""
    import numpy as np

    rgb = np.ascontiguousarray(image, dtype=np.uint8)
    height, width, _ = rgb.shape
    raw = bytearray()
    for row in rgb:
        raw.append(0)  # filter-type byte (0 = none) per scanline
        raw.extend(row.tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def build_scene() -> Scene:
    """A 3-camera ring watching a target, with a hand sweeping across camera 0."""
    return (
        SceneBuilder(fps=FPS, num_frames=NUM_FRAMES)
        .cameras(
            CameraRig.ring(
                n=3,
                radius=4.0,
                height=1.5,
                look_at=(0.0, 0.0, 0.5),
                focal=320.0,
                width=320,
                height_px=240,
            )
        )
        .entity("item", MotionPath.linear((0.0, 0.0, 0.5), (0.0, 0.0, 0.5)))
        .occlude_hand(
            HandSweep.sphere(0.22, span=0.55).blocks(camera=BLOCKED_CAM).during((0, NUM_FRAMES - 1))
        )
        .build()
    )


def dose_response(scene: Scene) -> dict[int, list[float | None]]:
    """Per-camera list of the item's visible_fraction across every frame."""
    manifest = build_manifest(scene, object_radius=OBJECT_RADIUS)
    item = next(e for e in manifest.entities if e.id == "item")
    curves: dict[int, list[float | None]] = {cam.id: [] for cam in scene.cameras}
    for frame in item.frames:
        for obs in frame.points["center"].per_cam:
            curves[obs.cam].append(obs.visible_fraction)
    return curves


def run(out_dir: Path) -> dict[int, list[float | None]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    curves = dose_response(scene)
    (out_dir / "hand_dose_response.json").write_text(
        json.dumps({str(cam): curve for cam, curve in curves.items()}, indent=2)
    )

    backend = RasterizerBackend(RasterizerConfig(point_radius=OBJECT_RADIUS))
    for frame in (0, (NUM_FRAMES - 1) // 2, NUM_FRAMES - 1):
        image = backend.render(scene, camera_id=BLOCKED_CAM, frame=frame)
        write_png(out_dir / f"hand_frame_{frame:02d}.png", image)
    return curves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "out", help="output directory"
    )
    args = parser.parse_args()
    curves = run(args.out)

    blocked = curves[BLOCKED_CAM]
    print(f"[hand_dose_response] blocked camera {BLOCKED_CAM} visible_fraction curve:")
    print("  frame  " + " ".join(f"{i:5d}" for i in range(NUM_FRAMES)))
    print("  vis_fr " + " ".join(f"{v:5.2f}" if v is not None else "  n/a" for v in blocked))
    lowest = min(range(NUM_FRAMES), key=lambda i: blocked[i] if blocked[i] is not None else 2.0)
    print(f"  minimum at frame {lowest} (hand centred); U-shaped dose-response")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
