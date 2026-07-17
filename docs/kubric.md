# Kubric photoreal backend

Answers [#39](https://github.com/bamdadd/multicam-sim/issues/39). A photoreal
[Kubric](https://github.com/google-research/kubric) (Blender) implementation of
`RendererBackend`, a same-Protocol open/closed swap of `PyrenderBackend`. As with
every renderer here, **pixels are not the contract**: this backend never feeds the
manifest and is never run in CI.

The design split is deliberate:

- `multicam_sim.dsl.kubric_spec` — **pure and Blender-free**. It does all the
  coordinate math and returns a typed `KubricSceneSpec` (pydantic). This is what
  the tests exercise, on a plain CPU box, with no Kubric or Blender.
- `multicam_sim.dsl.kubric_backend` — the thin adapter that feeds that spec to
  `kb.*` and renders. `kubric` is imported lazily, so importing the module (or
  constructing `KubricBackend`) never requires Blender.

## Coordinate conversion (the load-bearing part)

Our convention is OpenCV pinhole, right-down-forward (RDF), world **Z-up** (see
`DESIGN.md`). Kubric/Blender's *world* is also right-handed and Z-up, so world
points copy across unchanged. Only the **camera basis** differs.

### Extrinsics — camera pose

| | our OpenCV RDF camera | Blender/Kubric camera |
| --- | --- | --- |
| looks along | **+Z** (forward) | **-Z** |
| +X | right | right |
| +Y | **down** | **up** |

Our `R` maps world→camera with rows `[right, down, forward]`, and the centre is
`C = -Rᵀ @ t`. The camera-to-world rotation Blender wants therefore has columns
`[right, -down, -forward]`:

```
R_c2w = Rᵀ @ diag(1, -1, -1)
position   = C = camera.centre()
quaternion = quat(R_c2w)          # Kubric order (w, x, y, z)
```

`diag(1, -1, -1)` has determinant **+1** here (because `[right, down, forward]`
is right-handed: `right × down = forward`), so `R_c2w` is a proper rotation and
converts cleanly to a unit quaternion. This is the identical OpenCV→OpenGL pose
flip the **already-tested** pyrender backend uses
(`render.py`: `pose[:3,:3] = cam.rotation().T @ flip`, `pose[:3,3] = cam.centre()`),
so the two independent backends agree by construction — that shared flip is our
correctness anchor.

Kubric's quaternion order is `(w, x, y, z)` (verified from
`kubric/core/objects.py`: *"a (W, X, Y, Z) quaternion for describing the
rotation"*), not scipy's `(x, y, z, w)`. `kubric_spec` emits `(w, x, y, z)`.

### Intrinsics — `fx/fy/cx/cy` → `focal_length` (mm)

`kb.PerspectiveCamera` is parameterised by `focal_length` and `sensor_width`
(mm). From `kubric/core/cameras.py`:

```
sensor_height = sensor_width * res_y / res_x
f_x[px] = focal_length / sensor_width  * width
f_y[px] = focal_length / sensor_height * height   ==  f_x[px]
p_x, p_y = width/2, height/2     # principal point pinned at centre
```

Two consequences, **guarded** in `camera_to_kubric_spec` (it raises `ValueError`
otherwise):

1. **`fx == fy` (square pixels).** `f_y` reduces to `f_x`, so Kubric cannot
   represent non-square pixels. `Intrinsics.from_focal` / `from_fov` (without
   `fov_y_deg`) always satisfy this; a custom rig with `fx != fy` is rejected.
2. **Centred principal point** (`cx == W/2`, `cy == H/2`). `shift_x`/`shift_y`
   exist but Kubric's own source says they are *"currently not supported"*, so an
   off-centre principal point is rejected rather than rendered wrong.

The forward map is then just:

```
focal_length = fx * sensor_width / width
```

`sensor_width` is a **free constant** — it cancels in the `f_x` round-trip, so any
positive value gives identical pixels. We keep Kubric's default `36.0` mm.

## What the tests verify (no Blender)

`tests/test_kubric_backend.py` runs on a plain box and asserts the *translation*,
never the renderer:

- **Projection round-trip within `1e-6`.** A known **off-axis, off-centre** world
  point is projected two ways and compared: (a) our analytic `P = K[R|t]` via
  `Camera.project`, and (b) *through Kubric's own parameterisation* — rebuild
  `fx = focal_length/sensor_width·width`, turn the `(w, x, y, z)` quaternion back
  into `R_c2w`, decompose `point − position` into (right, up, back), and apply the
  Blender sign flips `u = cx − fx·a/c`, `v = cy + fy·b/c`. Going *through mm +
  quaternion* (not a verbatim `K`) verifies the two pieces checkable on a CPU: the
  mm formula inverts (`fx = focal_length/sensor_width·width`, from Kubric's source)
  and the `diag(1,-1,-1)` flip is sign-consistent with the projection. *(Unverified
  without Blender: that Kubric consumes the `(w, x, y, z)` quaternion as
  camera→world and projects with these signs — see the boundary section below.)*
- **Object mapping.** Each entity named point present at the frame becomes one
  sphere at its GT xyz with a stable per-entity colour; a 1-point object and a
  multi-joint (pose-shaped) entity lower identically (one sphere per point).
- **Guards.** `fx != fy` and off-centre principal points raise.
- **Isolation.** Importing the backend / building a spec does not import `kubric`.

## Actually rendering — the docker recipe (needs Blender)

Kubric **cannot `pip install` cleanly**: it needs Blender's bundled Python and
native libraries. The `pip install multicam-sim[kubric]` extra only pulls the
Python package; it will not render on its own. Use the maintained image:

```bash
docker run --rm -v "$PWD:/work" -w /work kubricdockerhub/kubruntu \
    python3 - <<'PY'
from multicam_sim.dsl import CameraRig, Path, SceneBuilder
from multicam_sim.dsl.kubric_backend import KubricBackend

scene = (
    SceneBuilder(fps=30.0, num_frames=11)
    .cameras(CameraRig.ring(n=3, radius=4.0, height=1.5,
                            look_at=(0.0, 0.0, 0.5),
                            focal=800.0, width=640, height_px=480))
    .entity("obj", Path.linear((0.0, -0.6, 0.5), (0.0, 0.6, 0.5)))
    .build()
)
rgb = KubricBackend(point_radius=0.1).render(scene, camera_id=0, frame=5)
print("rendered", rgb.shape, rgb.dtype)   # (480, 640, 3) uint8
PY
```

(Install `multicam-sim` into the image first, e.g. mount the repo and
`pip install -e .`.)

## GT cross-check against the analytic manifest

Kubric renders far more than RGB — its passes include **object coordinates,
segmentation, and depth** ("z" in camera space). Because the camera we build
projects **identically** to `P = K[R|t]` (that is exactly what the `1e-6`
round-trip test proves), the Kubric ground truth lines up with the manifest with
no fudge factor:

- **Positions / segmentation.** Project a manifest point's `xyz_gt` with
  `Camera.project` and read the Kubric segmentation/object-coordinate pass at that
  pixel — same object, same `uv`. A mismatch is a real bug in the scene, not a
  convention drift.
- **Depth ↔ `visible`.** The Kubric depth pass is camera-space `z`, the same
  quantity `Camera.project` returns as `w`. Where a nearer surface sits in front
  of a manifest point, the rendered depth is smaller than the analytic range —
  independently reproducing the analytic `visible=False`, from pixels, exactly as
  `docs/renderer-eval.md` demonstrates for the pyrender depth backend.

This stays **additive**: per `DESIGN.md` "Three per-camera fields, kept distinct",
`in_view`/`visible`/`occ_frac` remain analytic and GL-free. `scene_to_kubric_spec`
does **not** consult them — it renders every point present at the frame — so the
Kubric image is an independent second opinion, never an input to the mask.

## Honest boundary — what still needs Blender

Everything in `kubric_spec` (the coordinate math, guards, object mapping) is
verified locally. The one thing that **cannot** be tested without the docker
image is that `kb.PerspectiveCamera` / `kb.Sphere` / the Blender renderer
*consume* these spec fields as expected and emit a frame — i.e. an actual pixel
render. The spec's field names and the intrinsic/quaternion model were taken from
Kubric's source (`kubric/core/cameras.py`, `kubric/core/objects.py`), but the live
render path is exercisable only inside `kubricdockerhub/kubruntu`.
