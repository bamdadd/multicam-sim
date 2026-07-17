"""Render the FIRST real photoreal frames from a ``KubricSceneSpec`` on Modal.

This is the **Linux photoreal path** for multicam-sim's Kubric backend. As with
every renderer in this package, *pixels are not the contract* — this never feeds
the manifest and is never run in CI (see ``docs/kubric.md``). It exists to prove,
with actual Blender pixels, the one thing the local ``1e-6`` translation test
cannot: that ``kb.PerspectiveCamera`` consumes our ``(w, x, y, z)`` quaternion +
mm intrinsics with the signs we claim.

Why a Sandbox and not ``@app.function``
---------------------------------------
The maintained ``kubricdockerhub/kubruntu`` image ships Blender's bundled
**Python 3.9**, and the Modal client requires >= 3.10, so a normal Modal function
cannot inject its runtime there. Instead we drive the image through
``modal.Sandbox``: the Modal *client* runs locally (any modern Python), and we
``exec("python3", "-c", ...)`` the render inside the container using its own 3.9 +
kubric + bpy. Nothing Modal-specific runs in-container.

Geometry contract (the CORE deliverable)
-----------------------------------------
1. locally, translate a one-point scene to a typed ``KubricSceneSpec`` (pure,
   Blender-free) and serialise it to JSON;
2. in the Sandbox, rebuild the ``kb.Scene`` verbatim from that JSON, render RGBA
   **and the segmentation pass**, and report the segmentation centroid (the pixel
   where the sphere actually landed);
3. locally, project the same 3D point analytically with ``Camera.project``
   (``P = K[R|t]``, our source of truth) and print the **reprojection error** in
   pixels between the analytic ``uv`` and the rendered centroid.

Run (from a venv that has ``multicam-sim`` + ``modal`` installed)::

    uv venv && uv pip install -e . modal && .venv/bin/modal run recipes/kubric_modal.py

Writes the sample frame to ``recipes/out/kubric_frame.png``.
"""

from __future__ import annotations

import base64
from pathlib import Path as FsPath

import modal
import numpy as np
from pydantic import BaseModel

from multicam_sim.dsl import CameraRig, Path, SceneBuilder
from multicam_sim.dsl.kubric_backend import KubricBackend
from multicam_sim.scene import Scene


class RenderResult(BaseModel):
    """What the in-container render reports back through the ``RESULT_JSON`` line.

    ``centroid_uv`` is the ``(u, v)`` pixel centroid of the sphere's segmentation
    mask — the location the point *actually landed* in the Blender render — or
    ``None`` if nothing rendered. ``shape`` is ``(height, width)`` and ``n_mask``
    is the number of foreground pixels the centroid averaged over.
    """

    model_config = {"frozen": True}

    centroid_uv: tuple[float, float] | None
    shape: tuple[int, int]
    n_mask: int


KUBRUNTU = "kubricdockerhub/kubruntu"
#: A known off-axis, off-centre world point — no optical-axis symmetry can hide a
#: sign error. Same point family as the local ``1e-6`` round-trip test.
TARGET_POINT: tuple[float, float, float] = (0.7, -0.4, 0.9)
CAMERA_ID = 0
FRAME = 5
OUT_DIR = FsPath(__file__).parent / "out"

image = modal.Image.from_registry(KUBRUNTU, add_python=None)
app = modal.App("multicam-kubric-modal")

# --- render body, executed by the container's own Python 3.9 (kubric + bpy) ------
# Reads the spec JSON from $SPEC_JSON, renders one frame, and prints two marker
# lines to stdout: ``RESULT_JSON {...}`` (centroid + shape) and ``PNG_B64 {...}``.
RENDER_SRC = r"""
import os, io, json, base64
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender

spec = json.loads(os.environ["SPEC_JSON"])
cam = spec["camera"]

scene = kb.Scene(resolution=(cam["width"], cam["height"]))
scene.camera = kb.PerspectiveCamera(
    focal_length=cam["focal_length"],
    sensor_width=cam["sensor_width"],
    position=tuple(cam["position"]),
    quaternion=tuple(cam["quaternion"]),
)
for obj in spec["objects"]:
    material = kb.PrincipledBSDFMaterial(color=kb.Color(*obj["color"]))
    scene += kb.Sphere(
        name=obj["name"],
        scale=obj["radius"],
        position=tuple(obj["position"]),
        material=material,
    )
scene += kb.DirectionalLight(position=tuple(cam["position"]), intensity=3.0)

renderer = Blender(scene)
out = renderer.render(frames=[0])

rgba = np.asarray(out["rgba"][0], dtype=np.uint8)          # (H, W, 4), row 0 = top
seg = np.asarray(out["segmentation"][0]).reshape(rgba.shape[0], rgba.shape[1])
rows, cols = np.nonzero(seg > 0)                            # foreground = the sphere
if rows.size:
    centroid_uv = [float(cols.mean()), float(rows.mean())]  # (u=col, v=row)
else:
    centroid_uv = None

buf = io.BytesIO()
try:
    import imageio.v2 as imageio
    imageio.imwrite(buf, rgba[..., :3], format="png")
except Exception:
    kb.write_png(rgba, "/tmp/frame.png")
    buf.write(open("/tmp/frame.png", "rb").read())

result = {
    "centroid_uv": centroid_uv,
    "shape": [int(rgba.shape[0]), int(rgba.shape[1])],
    "n_mask": int(rows.size),
}
print("RESULT_JSON " + json.dumps(result))
print("PNG_B64 " + base64.b64encode(buf.getvalue()).decode())
"""


def _build_scene() -> Scene:
    """A one-camera, one-point scene: a single sphere pinned at ``TARGET_POINT``.

    A constant (start == end) linear path keeps the point identical at every
    frame, so the analytic projection and the rendered centroid refer to exactly
    the same world coordinate. Square resolution removes Blender's ``sensor_fit``
    axis choice as a confound (see ``docs/kubric-modal.md``).
    """
    return (
        SceneBuilder(fps=30.0, num_frames=11)
        .cameras(
            CameraRig.ring(
                n=3,
                radius=4.0,
                height=1.5,
                look_at=(0.0, 0.0, 0.5),
                focal=512.0,
                width=512,
                height_px=512,
            )
        )
        .entity("target", Path.linear(TARGET_POINT, TARGET_POINT))
        .build()
    )


def _parse_markers(stdout: str) -> tuple[RenderResult | None, bytes]:
    """Pull the ``RESULT_JSON`` and ``PNG_B64`` lines out of the Blender log."""
    result: RenderResult | None = None
    png = b""
    for line in stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            result = RenderResult.model_validate_json(line[len("RESULT_JSON ") :])
        elif line.startswith("PNG_B64 "):
            png = base64.b64decode(line[len("PNG_B64 ") :])
    return result, png


@app.local_entrypoint()
def main() -> None:
    import time

    scene = _build_scene()
    camera = scene.cameras[CAMERA_ID]

    # (1) pure, Blender-free translation -> typed spec -> JSON wire format.
    spec = KubricBackend(point_radius=0.05).spec_for(scene, CAMERA_ID, FRAME)
    spec_json = spec.model_dump_json()

    # (3a) analytic projection of the same point (our source of truth).
    point = np.asarray(TARGET_POINT, dtype=np.float64)
    uv_analytic, w = camera.project(point)
    assert w > 0.0, "target must be in front of the camera"

    print(f"analytic uv = ({uv_analytic[0]:.3f}, {uv_analytic[1]:.3f})  (in front, w={w:.3f})")
    print(f"pulling {KUBRUNTU} and rendering in a Modal Sandbox ...")

    t0 = time.time()
    with modal.enable_output():
        sb = modal.Sandbox.create(app=app, image=image, env={"SPEC_JSON": spec_json}, timeout=1800)
        try:
            proc = sb.exec("python3", "-c", RENDER_SRC)
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            proc.wait()
        finally:
            sb.terminate()
    wall = time.time() - t0

    result, png = _parse_markers(stdout)
    if result is None or result.centroid_uv is None:
        print("---- container stderr (tail) ----")
        print("\n".join(stderr.splitlines()[-40:]))
        raise SystemExit("render produced no segmentation centroid; see stderr above")

    OUT_DIR.mkdir(exist_ok=True)
    frame_path = OUT_DIR / "kubric_frame.png"
    frame_path.write_bytes(png)

    u_r, v_r = result.centroid_uv
    err = float(np.hypot(u_r - float(uv_analytic[0]), v_r - float(uv_analytic[1])))
    print(f"rendered {result.shape}  mask={result.n_mask} px  wall={wall:.1f}s")
    print(f"rendered centroid uv = ({u_r:.3f}, {v_r:.3f})")
    print(f"analytic  projection uv = ({uv_analytic[0]:.3f}, {uv_analytic[1]:.3f})")
    print(f"==> REPROJECTION ERROR = {err:.3f} px   (frame: {frame_path})")
