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

```
target point (world)   = (0.7, -0.4, 0.9)
camera                 = ring cam 0 (radius 4.0, height 1.5, look_at (0,0,0.5))
resolution / focal     = 512 x 512, focal 512 px  (square: no sensor_fit ambiguity)

analytic  Camera.project  uv = (194.811, 222.609)   # P = K[R|t], our source of truth
rendered  segmentation    uv = (194.169, 222.060)   # centroid of the sphere mask
------------------------------------------------------------------
REPROJECTION ERROR         = 0.844 px                # sub-pixel
```

Wall-clock: **~14 s** once the image is warm (first pull of `kubruntu` adds
several minutes). Rendered `512x512`, sphere segmentation mask `183 px`.

### Why 0.844 px is the expected floor, not a bug

The residual is dominated by *measurement*, not by a convention error — a sign
mistake in the quaternion or the axis flip would put the point tens to hundreds of
pixels away (or off-image) for this deliberately off-axis, off-centre target:

- **Silhouette centroid ≠ projected centre.** We render a sphere of radius `0.05`
  at range `~3.35`; its projection is a small ellipse whose pixel centroid sits a
  hair off the projected centre. For this radius/range the bias is `< 0.1 px`.
- **Discretisation + anti-aliasing.** The centroid averages `183` integer-grid,
  edge-antialiased pixels; the grid alone contributes order `1/sqrt(183) ≈ 0.07 px`
  of centroid noise, plus edge effects.
- **No Y-flip needed.** The rendered `v` matches analytic `v` directly (error is
  not `≈ height − 2v`), confirming Kubric's RGBA/segmentation arrays are top-down
  in the same sense as our image `v`-axis.

Sub-pixel agreement on an off-axis point is what "the flip and the quaternion are
sign-correct in real Blender" looks like.

## How it works

Three steps, split across the machine boundary so the fragile Blender python
never touches multicam-sim or Modal:

1. **Local (any modern Python).** Build a one-camera, one-point scene, translate
   it with the pure `KubricBackend.spec_for(...)` → `KubricSceneSpec`, and
   serialise to JSON (`spec.model_dump_json()`). Also project the target with
   `Camera.project` — the analytic ground truth.
2. **In-container (Blender's Python 3.9).** Rebuild the `kb.Scene` verbatim from
   the JSON, render RGBA **and the segmentation pass**, and print the sphere's
   segmentation-mask centroid + a base64 PNG back over stdout.
3. **Local.** Compare the rendered centroid to the analytic `uv`; the Euclidean
   distance is the reprojection error.

### Why `modal.Sandbox`, not `@app.function`

`kubruntu` ships Blender's bundled **Python 3.9**; the Modal client requires
`>= 3.10`, so it cannot inject a normal Modal *function* runtime into the image
(a plain `@app.function(image=...)` crash-loops with
`This version of Modal requires at least Python 3.10`). A **Sandbox** sidesteps
this: the Modal *client* runs locally on a modern Python, and we
`sb.exec("python3", "-c", RENDER_SRC)` the render using the container's own
interpreter. The spec crosses the boundary as a `SPEC_JSON` env var; results come
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
