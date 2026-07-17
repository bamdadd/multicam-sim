"""Render the README hero: a moving object seen across a non-overlapping camera
rig, as looping GIFs, through the same Modal + Kubric path as
``recipes/kubric_modal.py`` — extended from a single frame to a **frame range**.

The story is multicam-sim's whole reason to exist: three MTMC stations with
**disjoint** fields of view watch one object cross a corridor. It hands off
cam 0 -> cam 1 -> cam 2, and between stations there are **blind frames** where no
camera sees it at all — the non-overlapping-coverage gap the benchmark is about.
The tile borders are driven by the *analytic* ``in_view`` label (green = a camera
sees it, red = blind), so the pixels illustrate the manifest; they are not the
contract (Linux/Modal photoreal path only; see ``docs/kubric-modal.md``).

Outputs (into ``docs/assets/``):

* ``hero_cam0.gif`` / ``hero_cam1.gif`` / ``hero_cam2.gif`` — one per station;
* ``hero_grid.gif`` — all three on one synced timeline, the embedded hero.

Rendering walks the object's per-frame positions and renders each frame as a
fresh static scene (one ``exec`` per camera, three total), through a
``modal.Sandbox`` on ``kubricdockerhub/kubruntu`` (its Blender Python 3.9 is below
Modal's client floor — see ``kubric_modal.py`` for why a Sandbox, not a function).

Run (from a venv with ``multicam-sim`` + ``modal`` + ``pillow``)::

    uv venv && uv pip install -e . modal pillow && .venv/bin/modal run recipes/hero_gif.py
"""

from __future__ import annotations

import base64
import io
from pathlib import Path as FsPath

import modal
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel

from multicam_sim.dsl import CameraRig, Path, SceneBuilder
from multicam_sim.dsl.kubric_spec import KubricCameraSpec, camera_to_kubric_spec
from multicam_sim.dsl.rig import StationView
from multicam_sim.scene import Scene

# --- hero scene knobs ------------------------------------------------------------
NUM_FRAMES = 24
FPS = 24.0
TILE = 256  # render resolution per camera (square: no sensor_fit ambiguity)
STATION_FOV_DEG = 38.0
STATION_Y = -2.6
STATION_Z = 1.2
#: Disjoint MTMC stations: each looks at its own patch of the corridor, so their
#: fields of view do not overlap and the object is in at most one view at a time.
STATION_XS = (-2.0, 0.0, 2.0)
MOVER_START = (-3.2, 0.0, 0.5)
MOVER_END = (3.2, 0.0, 0.5)
MOVER_RADIUS = 0.16
MOVER_COLOR = (1.0, 0.45, 0.1)  # a warm orange that reads on black
KUBRUNTU = "kubricdockerhub/kubruntu"
OUT_DIR = FsPath(__file__).parents[1] / "docs" / "assets"

# GIF cosmetics.
GIF_TILE = 200  # tile is downscaled to this before assembly (lighter GIF)
GIF_FRAME_MS = 110  # per-frame duration; whole loop ~2.6 s
GIF_COLORS = 96  # palette size after quantisation
BORDER = 6
BANNER_H = 26
SEEN = (64, 200, 96)  # green border when the camera sees the object
BLIND = (210, 64, 64)  # red border when it does not

image = modal.Image.from_registry(KUBRUNTU, add_python=None)
app = modal.App("multicam-hero-gif")


class AnimObjectSpec(BaseModel):
    """One sphere with a per-frame world trajectory (keyframed in Blender)."""

    model_config = {"frozen": True}

    name: str
    radius: float
    color: tuple[float, float, float]
    positions: list[tuple[float, float, float]]


class KubricAnimSpec(BaseModel):
    """A fixed camera + animated objects over ``num_frames`` — the wire format."""

    model_config = {"frozen": True}

    camera: KubricCameraSpec
    objects: list[AnimObjectSpec]
    num_frames: int


# --- render body, run by the container's own Python 3.9 (kubric + bpy) -----------
# Reads ONE KubricAnimSpec JSON from argv[1] and renders the frame range. Each
# frame is a FRESH static scene with the object at that frame's position: this
# kubric build ignores ``keyframe_insert`` animation (it renders the static
# ``.position``), and ``Blender.__init__`` wipes bpy to factory-empty, so
# per-frame scenes never accumulate. Prints one ``FRAME_B64 <i> <b64>`` line each.
ANIM_RENDER_SRC = r"""
import sys, io, json, base64
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender
import imageio.v2 as imageio

spec = json.loads(sys.argv[1])
cam = spec["camera"]
n = int(spec["num_frames"])
objects = spec["objects"]

for f in range(n):
    scene = kb.Scene(resolution=(cam["width"], cam["height"]))
    scene.camera = kb.PerspectiveCamera(
        focal_length=cam["focal_length"],
        sensor_width=cam["sensor_width"],
        position=tuple(cam["position"]),
        quaternion=tuple(cam["quaternion"]),
    )
    for obj in objects:
        material = kb.PrincipledBSDFMaterial(color=kb.Color(*obj["color"]))
        scene += kb.Sphere(
            name=obj["name"],
            scale=obj["radius"],
            position=tuple(obj["positions"][f]),
            material=material,
        )
    # two lights so the sphere reads from every station, never a flat silhouette.
    scene += kb.DirectionalLight(position=(2.0, -3.0, 5.0), intensity=3.5)
    scene += kb.DirectionalLight(position=(-2.0, -3.0, 3.0), intensity=1.5)

    out = Blender(scene).render(frames=[0])
    rgb = np.asarray(out["rgba"][0], dtype=np.uint8)[..., :3]
    buf = io.BytesIO()
    imageio.imwrite(buf, rgb, format="png")
    print("FRAME_B64 %d " % f + base64.b64encode(buf.getvalue()).decode())
    sys.stdout.flush()
"""


def _build_scene() -> Scene:
    """MTMC stations with disjoint FOVs; one object crossing the corridor."""
    stations = [
        StationView(
            position=(x, STATION_Y, STATION_Z), look_at=(x, 0.0, 0.5), fov_deg=STATION_FOV_DEG
        )
        for x in STATION_XS
    ]
    return (
        SceneBuilder(fps=FPS, num_frames=NUM_FRAMES)
        .cameras(CameraRig.stations(stations, width=TILE, height_px=TILE))
        .entity("mover", Path.linear(MOVER_START, MOVER_END))
        .build()
    )


def _anim_spec_for(scene: Scene, camera_id: int) -> KubricAnimSpec:
    """Pure translation: fixed camera + the mover's per-frame positions -> spec."""
    cam_spec = camera_to_kubric_spec(scene.cameras[camera_id])
    objects: list[AnimObjectSpec] = []
    for entity in scene.entities:
        by_frame = {fr.frame: fr for fr in entity.frames}
        point_names = sorted(next(iter(by_frame.values())).points)
        for pname in point_names:
            positions = [
                tuple(float(v) for v in by_frame[f].points[pname]) for f in range(NUM_FRAMES)
            ]
            objects.append(
                AnimObjectSpec(
                    name=f"{entity.id}/{pname}",
                    radius=MOVER_RADIUS,
                    color=MOVER_COLOR,
                    positions=positions,  # type: ignore[arg-type]
                )
            )
    return KubricAnimSpec(camera=cam_spec, objects=objects, num_frames=NUM_FRAMES)


def _in_view_table(scene: Scene) -> list[list[bool]]:
    """Analytic ``in_view[camera][frame]`` — drives the GIF border colour."""
    table: list[list[bool]] = []
    for cam in scene.cameras:
        row: list[bool] = []
        for f in range(NUM_FRAMES):
            frame = next(fr for fr in scene.entities[0].frames if fr.frame == f)
            xyz = np.asarray(frame.points["center"], dtype=np.float64)
            uv, w = cam.project(xyz)
            row.append(bool(w > 0.0 and cam.in_image(uv)))
        table.append(row)
    return table


def _parse_frames(stdout: str) -> list[bytes]:
    """Collect ``FRAME_B64 <i> <b64>`` lines into an index-ordered PNG list."""
    frames: dict[int, bytes] = {}
    for line in stdout.splitlines():
        if line.startswith("FRAME_B64 "):
            _, idx, payload = line.split(" ", 2)
            frames[int(idx)] = base64.b64decode(payload)
    return [frames[i] for i in sorted(frames)]


def _decorate(png: bytes, *, label: str, seen: bool) -> Image.Image:
    """One camera tile: downscaled, with a status border and a caption."""
    img = Image.open(io.BytesIO(png)).convert("RGB").resize((GIF_TILE, GIF_TILE))
    colour = SEEN if seen else BLIND
    canvas = Image.new("RGB", (GIF_TILE + 2 * BORDER, GIF_TILE + 2 * BORDER + BANNER_H), colour)
    canvas.paste(img, (BORDER, BORDER + BANNER_H))
    draw = ImageDraw.Draw(canvas)
    status = "TRACKING" if seen else "blind"
    draw.text((BORDER + 2, 6), f"{label}   {status}", fill=(255, 255, 255))
    return canvas


def _save_gif(frames: list[Image.Image], path: FsPath) -> int:
    """Palette-quantise and save a looping GIF; returns its size in bytes."""
    quant = [f.quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT) for f in frames]
    quant[0].save(
        path,
        save_all=True,
        append_images=quant[1:],
        duration=GIF_FRAME_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return path.stat().st_size


@app.local_entrypoint()
def main() -> None:
    import time

    scene = _build_scene()
    in_view = _in_view_table(scene)
    specs = {cid: _anim_spec_for(scene, cid) for cid in range(len(scene.cameras))}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"rendering {NUM_FRAMES} frames x {len(scene.cameras)} MTMC stations on Modal ...")
    t0 = time.time()
    frames_by_cam: dict[int, list[bytes]] = {}
    with modal.enable_output():
        sb = modal.Sandbox.create(app=app, image=image, timeout=1800)
        try:
            for cid in range(len(scene.cameras)):
                proc = sb.exec("python3", "-c", ANIM_RENDER_SRC, specs[cid].model_dump_json())
                stdout = proc.stdout.read()
                stderr = proc.stderr.read()
                proc.wait()
                pngs = _parse_frames(stdout)
                if len(pngs) != NUM_FRAMES:
                    print("\n".join(stderr.splitlines()[-40:]))
                    raise SystemExit(f"cam {cid}: got {len(pngs)}/{NUM_FRAMES} frames")
                frames_by_cam[cid] = pngs
        finally:
            sb.terminate()
    wall = time.time() - t0

    # per-camera GIFs.
    sizes: dict[str, int] = {}
    decorated: dict[int, list[Image.Image]] = {}
    for cid, pngs in frames_by_cam.items():
        tiles = [
            _decorate(png, label=f"CAM {cid}", seen=in_view[cid][f]) for f, png in enumerate(pngs)
        ]
        decorated[cid] = tiles
        sizes[f"hero_cam{cid}.gif"] = _save_gif(tiles, OUT_DIR / f"hero_cam{cid}.gif")

    # synced grid GIF: the three stations side by side on one timeline, with a
    # banner that names the active camera or calls out the blind gap.
    grid_frames: list[Image.Image] = []
    tile_w = decorated[0][0].width
    tile_h = decorated[0][0].height
    ncam = len(scene.cameras)
    for f in range(NUM_FRAMES):
        row = Image.new("RGB", (tile_w * ncam, tile_h + BANNER_H), (16, 16, 16))
        for cid in range(ncam):
            row.paste(decorated[cid][f], (cid * tile_w, BANNER_H))
        draw = ImageDraw.Draw(row)
        active = [cid for cid in range(ncam) if in_view[cid][f]]
        banner = f"frame {f:02d}/{NUM_FRAMES - 1}   " + (
            f"seen by CAM {active[0]}" if active else "BLIND GAP - no camera sees the object"
        )
        draw.text((6, 7), banner, fill=(255, 220, 120) if not active else (235, 235, 235))
        grid_frames.append(row)
    sizes["hero_grid.gif"] = _save_gif(grid_frames, OUT_DIR / "hero_grid.gif")

    print(f"\nwall={wall:.1f}s   frames={NUM_FRAMES}   cameras={ncam}")
    for name, size in sizes.items():
        print(f"  {name:16s} {size / 1024:7.1f} KiB   ({OUT_DIR / name})")
