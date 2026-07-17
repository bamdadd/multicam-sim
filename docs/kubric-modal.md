# Kubric-on-Modal — the Linux photoreal path

`recipes/kubric_modal.py` renders the **first real Blender pixels** from a
`KubricSceneSpec`, on Modal, and reports how far those pixels land from our
analytic projection. It is the runnable companion to
[`docs/kubric.md`](kubric.md): that document translates a scene to a typed spec
and proves the translation locally to `1e-6`; this one takes the spec all the way
through Blender and measures the residual.

## Honest boundary — read this first

**This is the Linux photoreal path, not the M4/CI path. Pixels are not the
contract.** Every renderer in multicam-sim (`pyrender`, the numpy rasterizer,
Kubric) is an *optional second opinion* on the analytic manifest, never an input
to it. `scene_to_kubric_spec` does not consult `in_view`/`visible`/`occ_frac`
(see `DESIGN.md`, "Three per-camera fields, kept distinct").

Concretely, this recipe:

- runs **only on Linux/amd64**, inside `kubricdockerhub/kubruntu` (Blender + its
  bundled Python 3.9). It does not run on the M4/Apple-Silicon dev box or in CI.
  Nothing here is imported by the library or exercised by the test suite.
- is **not deterministic to the pixel**. Blender's sampler, anti-aliasing and the
  segmentation-centroid measurement all carry sub-pixel noise. The number below
  is a geometry cross-check, not a golden image.
- costs real Modal compute and pulls a multi-GB image. It is a manual,
  on-demand verification tool.

What it *does* buy: the one claim the local test explicitly cannot make (see the
"Honest boundary" in `docs/kubric.md`) — that `kb.PerspectiveCamera` actually
**consumes** our `(w, x, y, z)` quaternion and mm-derived focal length, with the
`diag(1, -1, -1)` OpenCV→Blender flip, and projects a known world point to the
pixel we predict.

## The reprojection-error number

The recipe renders the whole ring (three cameras, same target) and reports the
residual per camera. The target lands at a very different pixel in each view, yet
all three agree with the analytic projection to well under a pixel:

```
target point (world)   = (0.7, -0.4, 0.9)          # off-axis, off-centre
rig / resolution / focal = ring radius 4.0, height 1.5, look_at (0,0,0.5); 512x512, focal 512 px

cam 0: analytic (194.811, 222.609)  rendered (194.669, 222.560)  resid (-0.141, -0.049)  err 0.149 px
cam 1: analytic (211.764, 195.349)  rendered (211.668, 195.405)  resid (-0.096, +0.056)  err 0.111 px
cam 2: analytic (358.438, 206.583)  rendered (358.485, 206.584)  resid (+0.046, +0.001)  err 0.046 px
------------------------------------------------------------------------------------------
REPROJECTION ERROR: max 0.149 px, mean 0.102 px over 3 cameras
```

Wall-clock: **~33 s** for all three once the image is warm (first pull of
`kubruntu` adds several minutes). `rendered` is the segmentation-mask centroid
after the pixel-centre correction below; `resid = rendered − analytic`.

### The pixel-centre correction (0.5 px), and why it's real not a fit

The raw segmentation centroid is `cols.mean()` / `rows.mean()` over **integer
pixel indices**. Our intrinsics pin the principal point at `cx = cy = W/2`
(`Intrinsics.from_focal`), i.e. pixel index `i` has its *centre* at continuous
coordinate `i + 0.5`. So the integer-index centroid must be shifted by `+0.5` in
both axes to live in the same continuous image frame as `Camera.project`'s `uv`.

This is not a knob tuned to make one number look good: **before** the shift, all
three cameras show an *identical* systematic residual of `≈ (+0.5, +0.5)` despite
the target projecting to three different pixels — the signature of a constant
convention offset, not of noise (which scatters in sign and magnitude) and not of
silhouette bias (which would point radially from the principal point). **After**
the constant `+0.5`, the residuals collapse to `≤ 0.15 px` and their signs
scatter — the measurement floor. One offset, applied blind to all three views,
explains the whole systematic part.

### Why ~0.1 px is the floor

What's left is measurement, not convention — a sign error in the quaternion or the
axis flip would put the point tens to hundreds of pixels away (or off-image) for
this deliberately off-axis target:

- **Silhouette centroid ≠ projected centre.** The sphere (radius `0.05` at range
  `~3.35`) projects to a small ellipse whose centroid sits a hair off the
  projected centre; for this radius/range the bias is `< 0.02 px`.
- **Discretisation + anti-aliasing.** The centroid averages ~`100–180` integer-grid,
  edge-antialiased pixels; the grid alone contributes order `1/sqrt(N) ≈ 0.08 px`.
- **No Y-flip needed.** Rendered `v` matches analytic `v` directly (the error is
  not `≈ height − 2v`), confirming Kubric's RGBA/segmentation arrays are top-down
  in the same sense as our image `v`-axis.

Sub-pixel agreement across three independent views is what "the flip and the
quaternion are sign-correct in real Blender" looks like.

## How it works

Three steps, split across the machine boundary so the fragile Blender python
never touches multicam-sim or Modal:

1. **Local (any modern Python).** Build a three-camera ring, one-point scene, and
   for each camera translate it with the pure `KubricBackend.spec_for(...)` →
   `KubricSceneSpec`, serialised to JSON (`spec.model_dump_json()`). Also project
   the target with `Camera.project` — the analytic ground truth.
2. **In-container (Blender's Python 3.9).** For each camera, rebuild the
   `kb.Scene` verbatim from its JSON, render RGBA **and the segmentation pass**,
   and print the sphere's segmentation-mask centroid + a base64 PNG back over
   stdout. Each camera renders in a *fresh* `exec` (fresh `bpy` process), so
   scenes never bleed into one another.
3. **Local.** Correct the integer-index centroid to pixel centres (`+0.5`) and
   compare to the analytic `uv`; the Euclidean distance is the reprojection error.

### Why `modal.Sandbox`, not `@app.function`

`kubruntu` ships Blender's bundled **Python 3.9**; the Modal client requires
`>= 3.10`, so it cannot inject a normal Modal *function* runtime into the image
(a plain `@app.function(image=...)` crash-loops with
`This version of Modal requires at least Python 3.10`). A **Sandbox** sidesteps
this: the Modal *client* runs locally on a modern Python, and we
`sb.exec("python3", "-c", RENDER_SRC, spec_json)` the render using the container's
own interpreter. The spec crosses the boundary as an `argv` string; results come
back as marker lines on stdout. Nothing Modal-specific runs in-container.

## Running it

Needs Modal auth (`~/.modal.toml`) and a venv with `multicam-sim` + `modal`:

```bash
uv venv && uv pip install -e . modal
.venv/bin/modal run recipes/kubric_modal.py
```

The sample frame is written to `recipes/out/kubric_frame.png` (committed as
`recipes/out/kubric_frame.png`): a single half-lit sphere on black, at the
predicted pixel.

If the render never converges — a stuck pull or a headless-Blender hang — the
recipe times out at 30 min per Sandbox; the geometry contract (the number above)
is the deliverable, and the frame is garnish.
