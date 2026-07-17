# Kubric hand-occluder — photoreal cross-check of `visible_fraction`

`recipes/kubric_hand_occluder.py` is the **photoreal confirmation** of the
analytic `visible_fraction` dose-response shipped with the moving hand occluder
(`HandOccluder` + `silhouette_visible_fraction`). It renders the *same* scene the
manifest describes on the Kubric/Blender path and measures the target's visible
fraction from the **segmentation masks**, then compares it to the analytic value.

It is the **second backend** for the hand occluder — the first being the
pure-numpy rasterizer cross-check in `tests/test_hand_labels.py`
(`test_rasterizer_pixel_count_cross_checks_analytic`), which runs on M4/CI. This
one runs the real renderer on Modal and is the Linux-only photoreal counterpart.

## The number

A grey hand-proxy sphere sweeps across camera 0's view of a static reddish target
(radius 0.15) over 15 frames, at a depth nearer the camera than the target so it
occludes. The analytic `visible_fraction` traces a U-shaped dose-response
(1 → 0 → 1). The empirical value is measured from the Kubric segmentation pass:

```
empirical_vf(frame) = target_visible_pixels(with hand) / target_pixels(clear)
```

(The target is static, so the "clear" silhouette is frame-independent and rendered
once — 732 px — then each occluded frame is rendered with the hand at
`HandOccluder.center_at(frame)`.)

```
frame  analytic  empirical  |diff|
  0-3    1.000     1.000     0.000   (clear)
  4      0.957     0.956     0.000
  5      0.535     0.533     0.002
  6      0.033     0.036     0.002
  7      0.000     0.000     0.000   (fully covered)
  8      0.033     0.036     0.002
  9      0.535     0.533     0.002
  10     0.957     0.956     0.000
  11-14  1.000     1.000     0.000   (clear)

AGREEMENT (empirical segmentation vs analytic visible_fraction):
  overall  : mean |diff| 0.001, max 0.002  (n=15)
  endpoints: mean |diff| 0.000, max 0.000  (n=9)   # clear / fully-covered — exact
  partials : mean |diff| 0.001, max 0.002  (n=6)   # partial-cover — near-exact here
```

Wall-clock: **~30 s** warm (clear + 15 occluded renders, one Modal Sandbox; first
cold pull of `kubruntu` adds minutes). Sample: `docs/assets/hand_occluder_sweep.gif`
(~42 KiB) — the reddish target swept behind the grey hand and back.

### Why the partials agree so tightly (honest note)

The task anticipated that partial-cover frames would only *approximately* match —
the analytic model is a bounding-**disc** area overlap, the render a true
perspective **sphere** silhouette. Here they agree to ≤ 0.002 even on partials.
That is **not** a general guarantee; it holds because this scene is favourable to
the disc model:

- the target sits at the image centre and subtends a small angle (≈ 732 px on
  256²), so its perspective silhouette is almost exactly a circle — the disc model
  is a paraxial approximation and this is the paraxial regime;
- the occluder A1's model resolves (`at_frame` → `Sphere`) is the *same solid* we
  render, so we are validating the disc-overlap **area math** against a real
  sphere-vs-sphere occlusion, not a different shape.

Where it would diverge: an off-axis or large-angle target (circle → ellipse under
perspective), or a non-spherical occluder (a real articulated hand mesh vs the
bounding disc). The endpoints (fully clear / fully covered) stay exact regardless,
because they are set-membership, not area. So the honest reading is: **the analytic
dose-response is confirmed by real pixels, exactly at the endpoints and to
sub-percent through the partials in the paraxial regime this manifest targets.**

## How it works

Same Modal Sandbox path as `recipes/kubric_modal.py` / `recipes/hero_gif.py`:

- **Spec local, render remote.** The pure `camera_to_kubric_spec` + the hand's
  per-frame `center_at(frame)` are serialised to a typed `HandRenderSpec` and sent
  into the container as an `argv` JSON string. The analytic `visible_fraction`
  comes from `build_manifest(scene, object_radius=...)` locally — the source of
  truth.
- **Static-per-frame render.** This kubruntu build ignores `keyframe_insert`
  animation, so each frame is a fresh `kb.Scene` with the hand sphere at its
  resolved centre (`Blender.__init__` wipes bpy — no accumulation).
- **Segmentation id.** kubric assigns segmentation ids by asset **add-order**, not
  by the `segmentation_id` hint. The target is always added *first*, so it keeps a
  stable id across the clear and occluded scenes; we read that id from the clear
  render's dominant nonzero segmentation value and reuse it to isolate the target
  in the occluded frames (the hand, added second, has a different id).

## Honest boundary

Linux/amd64 + Modal only, **not** M4/CI. **Pixels are not the contract** — the
analytic manifest is; this is an on-demand second opinion that confirms it.
`scene_to_kubric_spec` / this recipe never feed the manifest. Domain-neutral: a
hand reaching over an item on a work surface / conveyor.

## Running it

```bash
uv venv && uv pip install -e . modal pillow
.venv/bin/modal run recipes/kubric_hand_occluder.py
```
