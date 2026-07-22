"""Render the multi-view hero GIF: one moving object, 8 angles, one identity.

Deterministic and dependency-light: the pure-numpy :class:`RasterizerBackend`
renders each of 8 ring cameras every frame, the analytic manifest gives the
object's projected pixel in every view, and we overlay the SAME id + colour in
all 8 tiles so a reader can see it is one object seen from eight angles. Tiled
2x4 with Pillow into ``assets/multiview_hero.gif``.

``Pillow`` is imported lazily inside :func:`main`, never at module load, so the
package/test import path (and CI) never depends on it. Run:

    uv run --with pillow python scripts/render_multiview_hero.py
"""

from __future__ import annotations

from pathlib import Path as FsPath

import numpy as np

from multicam_sim.dsl import CameraRig, Path, SceneBuilder
from multicam_sim.dsl.raster import RasterizerBackend, RasterizerConfig
from multicam_sim.manifest import build_manifest

# --- fixed, reproducible parameters ---------------------------------------- #
N_CAMS = 8
RADIUS = 4.0
HEIGHT = 1.5
LOOK_AT = (0.0, 0.0, 0.5)
FOCAL = 500.0
RES = 256
NUM_FRAMES = 40
FPS = 30.0
ENTITY_ID = "e0"
POINT_NAME = "center"
PATH_START = (-1.2, -0.5, 0.5)
PATH_END = (1.2, 0.5, 0.9)

TILE = 128  # per-camera tile size in the grid (renders downscaled from RES)
GRID_COLS, GRID_ROWS = 4, 2  # 2x4 = 8 cameras
OBJ_RADIUS = 0.12  # world-space object radius (bolder, readable sphere)
BOX_HALF = 22  # half-size (px, at RES) of the identity box drawn on the object
OBJ_COLOR = (255, 90, 70)  # one colour == one identity across all 8 views
GIF_MS = 90

OUT = FsPath(__file__).resolve().parents[1] / "assets" / "multiview_hero.gif"


def build_scene():
    cams = CameraRig.ring(
        n=N_CAMS,
        radius=RADIUS,
        height=HEIGHT,
        look_at=LOOK_AT,
        width=RES,
        height_px=RES,
        focal=FOCAL,
    )
    return (
        SceneBuilder(fps=FPS, num_frames=NUM_FRAMES)
        .cameras(cams)
        .entity(ENTITY_ID, Path.linear(PATH_START, PATH_END))
        .build()
    )


def main() -> None:
    import time

    from PIL import Image, ImageDraw  # lazy: never imported at package/test load

    t0 = time.time()
    scene = build_scene()
    manifest = build_manifest(scene)
    backend = RasterizerBackend(RasterizerConfig(point_radius=OBJ_RADIUS))

    # per (frame, cam): the object's projected uv + whether it is in view.
    obs: dict[tuple[int, int], tuple[float, float, bool]] = {}
    entity = next(e for e in manifest.entities if e.id == ENTITY_ID)
    for fr in entity.frames:
        for pc in fr.points[POINT_NAME].per_cam:
            obs[(fr.frame, pc.cam)] = (pc.uv[0], pc.uv[1], bool(pc.in_view))

    grid_w, grid_h = GRID_COLS * TILE, GRID_ROWS * TILE
    frames: list[Image.Image] = []
    for f in range(NUM_FRAMES):
        grid = Image.new("RGB", (grid_w, grid_h), (18, 18, 22))
        for cam_id in range(N_CAMS):
            img = np.asarray(backend.render(scene, cam_id, f), dtype=np.uint8)
            tile = Image.fromarray(img, "RGB")
            draw = ImageDraw.Draw(tile)
            u, v, in_view = obs[(f, cam_id)]
            if in_view:
                draw.rectangle(
                    [u - BOX_HALF, v - BOX_HALF, u + BOX_HALF, v + BOX_HALF],
                    outline=OBJ_COLOR,
                    width=3,
                )
                draw.text((u - BOX_HALF, v - BOX_HALF - 12), ENTITY_ID, fill=OBJ_COLOR)
            draw.text((4, 4), f"cam {cam_id}", fill=(210, 210, 210))
            tile = tile.resize((TILE, TILE), Image.BILINEAR)
            col, row = cam_id % GRID_COLS, cam_id // GRID_COLS
            grid.paste(tile, (col * TILE, row * TILE))
        frames.append(grid)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_MS,
        loop=0,
        optimize=True,
    )
    size_kb = OUT.stat().st_size / 1024
    print(
        f"wrote {OUT.relative_to(FsPath.cwd())}  "
        f"{NUM_FRAMES} frames  {grid_w}x{grid_h}  {size_kb:.1f} KiB  "
        f"{time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
