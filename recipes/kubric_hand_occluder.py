"""Photoreal cross-check of A1's analytic ``visible_fraction`` dose-response.

A moving hand-proxy (the ``HandOccluder`` shipped on main — typed keyframes,
``Occluder.at_frame(frame)``) sweeps across one camera's view of a static target.
The manifest records, per frame, the **analytic** ``visible_fraction`` of the
target (``multicam_sim.visibility.silhouette_visible_fraction`` — a bounding-disc
area model). This recipe renders the *same* scene on the Kubric/Blender photoreal
path and measures the **empirical** visible fraction from the SEGMENTATION masks:

    empirical_vf(frame) = target_visible_pixels(with hand) / target_pixels(clear)

then reports how well empirical tracks analytic. The target is static, so the
"clear" silhouette is frame-independent and rendered once; only the occluded
frames sweep.

This is the *second backend* for the hand occluder (the first being the
pure-numpy rasterizer cross-check in ``tests/test_hand_labels.py``). It follows
the ``recipes/hero_gif.py`` **static-per-frame** pattern: this kubruntu build
ignores ``keyframe_insert``, so each frame is a fresh ``kb.Scene`` with the hand
sphere at ``hand.center_at(frame)``.

Honest boundary. Linux/amd64 + Modal only, **not** M4/CI; **pixels are not the
contract** — the analytic manifest is. Endpoints (fully clear / fully covered)
should agree tightly; partial-cover frames only approximately, because the
analytic model is a bounding **disc** overlap while the render is a true
perspective **sphere** silhouette with anti-aliased, discretised edges.
Domain-neutral: a hand reaching over an item on a work surface / conveyor.

Run (venv with ``multicam-sim`` + ``modal`` + ``pillow``)::

    uv venv && uv pip install -e . modal pillow
    .venv/bin/modal run recipes/kubric_hand_occluder.py
"""

from __future__ import annotations

import base64
import io
from pathlib import Path as FsPath

import modal
import numpy as np
from PIL import Image
from pydantic import BaseModel

from multicam_sim import build_manifest
from multicam_sim.dsl import CameraRig, HandSweep, Path, SceneBuilder
from multicam_sim.dsl.kubric_spec import KubricCameraSpec, camera_to_kubric_spec
from multicam_sim.occluders import HandOccluder
from multicam_sim.scene import Scene

# --- scene knobs (tuned for a clean U-shaped dose-response with partials) -------
FPS = 24.0
NUM_FRAMES = 15
TARGET_CAM = 0
OBJECT_RADIUS = 0.15
HAND_RADIUS = 0.22
SPAN = 0.75
FOCAL = 420.0
RES = 256
TARGET_POS = (0.0, 0.0, 0.5)
TARGET_COLOR = (0.85, 0.20, 0.15)  # reddish target
HAND_COLOR = (0.55, 0.55, 0.58)  # neutral grey hand-proxy
KUBRUNTU = "kubricdockerhub/kubruntu"
OUT_DIR = FsPath(__file__).parents[1] / "docs" / "assets"

GIF_FRAME_MS = 130
GIF_COLORS = 96

image = modal.Image.from_registry(KUBRUNTU, add_python=None)
app = modal.App("multicam-hand-occluder")


class HandRenderSpec(BaseModel):
    """Wire format: a fixed camera + a static target + a per-frame hand centre."""

    model_config = {"frozen": True}

    camera: KubricCameraSpec
    object_radius: float
    hand_radius: float
    target_pos: tuple[float, float, float]
    target_color: tuple[float, float, float]
    hand_color: tuple[float, float, float]
    hand_centers: list[tuple[float, float, float]]


# --- render body, run by the container's own Python 3.9 (kubric + bpy) -----------
# Renders the clear target once, then one occluded frame per hand centre, and
# prints the target's segmentation-pixel count for each plus a base64 RGBA PNG of
# the occluded frame. kubric assigns segmentation ids by asset add-order (not the
# ``segmentation_id`` hint), so the target — always added FIRST — gets a stable id
# across scenes; we read it from the clear (target-only) render's dominant nonzero
# id and reuse it to isolate the target in the occluded frames.
RENDER_SRC = r"""
import sys, io, json, base64
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender
import imageio.v2 as imageio

spec = json.loads(sys.argv[1])
cam = spec["camera"]

def make_camera(scene):
    scene.camera = kb.PerspectiveCamera(
        focal_length=cam["focal_length"], sensor_width=cam["sensor_width"],
        position=tuple(cam["position"]), quaternion=tuple(cam["quaternion"]))

def add_target(scene):
    mat = kb.PrincipledBSDFMaterial(color=kb.Color(*spec["target_color"]))
    scene += kb.Sphere(name="target", scale=spec["object_radius"],
                       position=tuple(spec["target_pos"]), material=mat)

def add_hand(scene, centre):
    mat = kb.PrincipledBSDFMaterial(color=kb.Color(*spec["hand_color"]))
    scene += kb.Sphere(name="hand", scale=spec["hand_radius"], position=tuple(centre), material=mat)

def lights(scene):
    scene += kb.DirectionalLight(position=(2.0, -3.0, 5.0), intensity=3.5)
    scene += kb.DirectionalLight(position=(-2.0, -3.0, 3.0), intensity=1.5)

def seg2d(out):
    s = np.asarray(out["segmentation"][0])
    return s.reshape(s.shape[0], s.shape[1])

# clear target once (static -> frame-independent silhouette). The target is the
# only foreground, so its id is the single dominant nonzero segmentation value.
scene = kb.Scene(resolution=(cam["width"], cam["height"]))
make_camera(scene); add_target(scene); lights(scene)
clear_seg = seg2d(Blender(scene).render(frames=[0]))
nonzero = clear_seg[clear_seg > 0]
target_id = int(np.bincount(nonzero).argmax()) if nonzero.size else 0
clear_px = int((clear_seg == target_id).sum())
print("TARGET_ID %d" % target_id)
print("CLEAR_PX %d" % clear_px)

# occluded frames: count only the target's id (the hand, added after, has another).
for f, centre in enumerate(spec["hand_centers"]):
    scene = kb.Scene(resolution=(cam["width"], cam["height"]))
    make_camera(scene); add_target(scene); add_hand(scene, centre); lights(scene)
    out = Blender(scene).render(frames=[0])
    occ_px = int((seg2d(out) == target_id).sum())
    rgb = np.asarray(out["rgba"][0], dtype=np.uint8)[..., :3]
    buf = io.BytesIO(); imageio.imwrite(buf, rgb, format="png")
    print("OCC %d %d" % (f, occ_px))
    print("FRAME_B64 %d " % f + base64.b64encode(buf.getvalue()).decode())
    sys.stdout.flush()
"""


def _build_scene() -> Scene:
    """Static target on camera 0's sightline; a hand-proxy sweeps across it."""
    sweep = (
        HandSweep.sphere(HAND_RADIUS, span=SPAN)
        .blocks(camera=TARGET_CAM)
        .during((0, NUM_FRAMES - 1))
    )
    return (
        SceneBuilder(fps=FPS, num_frames=NUM_FRAMES)
        .cameras(
            CameraRig.ring(
                n=3,
                radius=4.0,
                height=1.5,
                look_at=(0.0, 0.0, 0.5),
                focal=FOCAL,
                width=RES,
                height_px=RES,
            )
        )
        .entity("obj", Path.linear(TARGET_POS, TARGET_POS))
        .occlude_hand(sweep)
        .build()
    )


def _analytic_vf(scene: Scene) -> list[float]:
    """Analytic ``visible_fraction`` of the target on TARGET_CAM, per frame."""
    manifest = build_manifest(scene, object_radius=OBJECT_RADIUS)
    entity = next(e for e in manifest.entities if e.id == "obj")
    out: list[float] = []
    for frame in entity.frames:
        obs = next(pc for pc in frame.points["center"].per_cam if pc.cam == TARGET_CAM)
        assert obs.visible_fraction is not None
        out.append(obs.visible_fraction)
    return out


def _spec_for(scene: Scene) -> HandRenderSpec:
    hand = scene.occluders[0]
    assert isinstance(hand, HandOccluder)
    centers = [tuple(float(c) for c in hand.center_at(f)) for f in range(NUM_FRAMES)]
    return HandRenderSpec(
        camera=camera_to_kubric_spec(scene.cameras[TARGET_CAM]),
        object_radius=OBJECT_RADIUS,
        hand_radius=HAND_RADIUS,
        target_pos=TARGET_POS,
        target_color=TARGET_COLOR,
        hand_color=HAND_COLOR,
        hand_centers=centers,  # type: ignore[arg-type]
    )


def _parse(stdout: str) -> tuple[int, dict[int, int], dict[int, bytes]]:
    clear_px = 0
    occ: dict[int, int] = {}
    frames: dict[int, bytes] = {}
    for line in stdout.splitlines():
        if line.startswith("CLEAR_PX "):
            clear_px = int(line.split()[1])
        elif line.startswith("OCC "):
            _, f, px = line.split()
            occ[int(f)] = int(px)
        elif line.startswith("FRAME_B64 "):
            _, idx, payload = line.split(" ", 2)
            frames[int(idx)] = base64.b64decode(payload)
    return clear_px, occ, frames


def _save_sweep_gif(frames: list[bytes], path: FsPath) -> int:
    imgs = [Image.open(io.BytesIO(p)).convert("RGB") for p in frames]
    quant = [im.quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT) for im in imgs]
    quant[0].save(
        path,
        save_all=True,
        append_images=quant[1:],
        duration=GIF_FRAME_MS,
        loop=0,
        optimize=True,
    )
    return path.stat().st_size


@app.local_entrypoint()
def main() -> None:
    import time

    scene = _build_scene()
    analytic = _analytic_vf(scene)
    spec = _spec_for(scene)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"rendering clear + {NUM_FRAMES} occluded frames on Modal ...")
    t0 = time.time()
    with modal.enable_output():
        sb = modal.Sandbox.create(app=app, image=image, timeout=1800)
        try:
            proc = sb.exec("python3", "-c", RENDER_SRC, spec.model_dump_json())
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            proc.wait()
        finally:
            sb.terminate()
    wall = time.time() - t0

    clear_px, occ, frame_pngs = _parse(stdout)
    if clear_px <= 0 or len(occ) != NUM_FRAMES:
        print("\n".join(stderr.splitlines()[-40:]))
        raise SystemExit(f"render failed: clear_px={clear_px} occ_frames={len(occ)}")

    empirical = [min(1.0, occ[f] / clear_px) for f in range(NUM_FRAMES)]
    diffs = [abs(empirical[f] - analytic[f]) for f in range(NUM_FRAMES)]

    endpoint_idx = [f for f in range(NUM_FRAMES) if analytic[f] >= 0.99 or analytic[f] <= 0.01]
    partial_idx = [f for f in range(NUM_FRAMES) if 0.01 < analytic[f] < 0.99]
    end_diffs = [diffs[f] for f in endpoint_idx]
    part_diffs = [diffs[f] for f in partial_idx]

    print(f"\nclear target silhouette = {clear_px} px   wall={wall:.1f}s")
    print("frame  analytic  empirical  |diff|")
    for f in range(NUM_FRAMES):
        tag = "end" if f in endpoint_idx else "part"
        print(f"  {f:2d}    {analytic[f]:.3f}     {empirical[f]:.3f}     {diffs[f]:.3f}  {tag}")

    print(
        f"\nAGREEMENT (empirical seg vs analytic visible_fraction):\n"
        f"  overall : mean |diff| {np.mean(diffs):.3f}, max {np.max(diffs):.3f}  (n={NUM_FRAMES})\n"
        f"  endpoints: mean |diff| {np.mean(end_diffs):.3f}, max {np.max(end_diffs):.3f}"
        f"  (n={len(end_diffs)})\n"
        f"  partials : mean |diff| {np.mean(part_diffs):.3f}, max {np.max(part_diffs):.3f}"
        f"  (n={len(part_diffs)})"
    )

    gif = OUT_DIR / "hand_occluder_sweep.gif"
    size = _save_sweep_gif([frame_pngs[f] for f in range(NUM_FRAMES)], gif)
    print(f"\nsweep GIF: {size / 1024:.1f} KiB  ({gif})")
