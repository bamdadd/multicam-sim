"""Runnable rasterizer demo: three ring cameras -> three PNGs, zero system deps.

Deterministic, CPU-only, no GL, no Blender. Run it directly::

    python examples/rasterize_demo.py        # or: uv run python examples/rasterize_demo.py

Builds a small 3-camera ring scene (one moving point + a sphere occluder) and
renders each camera's view at one frame through
:class:`~multicam_sim.dsl.raster.RasterizerBackend` — the pure-numpy backend that
runs the same on a laptop and on a bare CI box. Each ``(H, W, 3)`` uint8 image is
written to a PNG with a tiny stdlib-only encoder (no pillow), so the demo needs
nothing beyond the package's core dependencies.

Pixels are not the manifest contract; this is only a way to *look at* a scene.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

from multicam_sim.dsl import CameraRig, Occlusion, Path as MotionPath, SceneBuilder
from multicam_sim.dsl.raster import RasterizerBackend
from multicam_sim.geometry import FloatArray
from multicam_sim.scene import Scene

FRAME = 5


def build_scene() -> Scene:
    """A 3-camera ring watching one moving point past a sphere occluder."""
    return (
        SceneBuilder(fps=30.0, num_frames=11)
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
        .entity("obj", MotionPath.linear((0.0, -0.6, 0.5), (0.0, 0.6, 0.5)))
        .occlude(Occlusion.sphere(size=0.3).blocks(camera=1).during((3, 7)))
        .build()
    )


def write_png(path: Path, image: FloatArray) -> None:
    """Write an ``(H, W, 3)`` uint8 array to ``path`` as PNG (stdlib only)."""
    rgb = np.ascontiguousarray(image, dtype=np.uint8)
    height, width, _ = rgb.shape

    # each scanline is prefixed with a filter-type byte (0 = none)
    raw = bytearray()
    for row in rgb:
        raw.append(0)
        raw.extend(row.tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def run(out_dir: Path) -> list[Path]:
    """Render every camera at ``FRAME`` and write one PNG each; return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    backend = RasterizerBackend()

    written: list[Path] = []
    for cam in scene.cameras:
        image = backend.render(scene, camera_id=cam.id, frame=FRAME)
        path = out_dir / f"cam{cam.id}_frame{FRAME}.png"
        write_png(path, image)
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "out", help="output directory"
    )
    args = parser.parse_args()
    paths = run(args.out)
    print(f"[rasterize_demo] wrote {len(paths)} PNGs to {paths[0].parent}")
    for path in paths:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
