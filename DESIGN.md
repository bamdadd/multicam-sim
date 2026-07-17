# multicam-sim — design & contract

multicam-sim is the **producer**: it builds a typed multi-camera scene and emits
a JSON **manifest** that a triangulation consumer (multicam-occlusion) reads. It
does pure analytic projection + boolean occlusion — no renderer, no GL.

This document is the **contract of record**: the camera convention and the
manifest schema. Downstream layers (DSL sugar, renderer, pose, issue injection)
build on top of this without changing it.

## Camera convention

**Mirrored from multicam-occlusion@59f4906**
(`src/multicam_occlusion/triangulation.py::look_at_rotation` / `build_ring_cameras`).
Replicated exactly so a manifest produced here is consumed convention-for-convention
by that package's `triangulate_dlt`.

OpenCV pinhole, right-down-forward (RDF) camera axes, world **Z-up**:

```
forward = (target - eye) / ||target - eye||     # camera +z, viewing direction
right   = forward x up_world,  normalised        # camera +x
down    = forward x right                        # camera +y
R rows  = [right, down, forward]                 # world -> camera rotation
t       = -R @ C                                 # world -> camera translation (NOT the centre C)
C       = -R^T @ t                               # camera centre in world (inverse)
up_world = (0, 0, 1)                             # Z-up
K = [[fx, 0, cx],                                # cx = W/2, cy = H/2
     [0, fy, cy],
     [0,  0,  1]]
P = K [R | t]                                    # 3x4 projection matrix
```

Projection of a world point `X`: `x ~ P [X; 1]`; divide by `w` = the third
coordinate. **`w > 0` means the point is in front of the camera.** Pixel is
`(x0/w, x1/w)`.

Field label in the manifest: `"convention": "opencv_rdf"`.

## Types (pydantic v2, JSON-native fields)

Stored fields are floats/lists (no numpy in the schema); numpy is internal to the
compute methods only.

- **Intrinsics**(`fx, fy, cx, cy, width, height`) — `matrix()` → `K`.
- **Camera**(`id, intrinsics, R` 3×3, `t` 3) — `projection_matrix()`, `project()`,
  `centre()`. `R`/`t` are world→camera; `t = -R@C`.
- **Occluder** (ABC) + **Box**(`center, half_extents`) / **Sphere**(`center, radius`)
  — `blocks_segment(a, b)` ray/segment-vs-solid for hard visibility.
- **Entity**(`id, edges?`, `frames`) — each frame carries **named** 3D points.
  `id` is the **stable track identifier**: the manifest keeps it byte-identical
  across every frame, so a tracking consumer reads it as the ground-truth track
  id for that entity across the whole take. `build_multi_entity_scene()`
  demonstrates this end to end with two entities whose per-camera visibility
  diverges on some frames.
- **Scene**(`fps, num_frames, cameras, entities, occluders`).

### Named-points design (forward-compatible with pose)

An entity's frame maps **point name → `[x, y, z]`**. This is deliberate so a human
pose layer slots in later **without a schema fork**:

- an **object** = 1 entity with 1 named point `"center"`;
- a **COCO-17 human** (later) = 1 entity with 17 named points + an `edges` skeleton.

## Manifest schema (the JSON contract)

```json
{
  "cameras": [
    {"id": 0, "K": [[..],[..],[..]], "R": [[..],[..],[..]], "t": [..],
     "width": 640, "height": 480, "convention": "opencv_rdf"}
  ],
  "fps": 30.0,
  "num_frames": 11,
  "entities": [
    {
      "id": "obj",
      "edges": [["a", "b"]],
      "frames": [
        {
          "frame": 0,
          "points": {
            "center": {
              "xyz_gt": [x, y, z],
              "per_cam": [
                {"cam": 0, "uv": [u, v], "in_view": true, "visible": true, "occ_frac": 0.0}
              ]
            }
          }
        }
      ]
    }
  ],
  "topology": {
    "stations": [{"id": "A", "camera_ids": [0]}, {"id": "B", "camera_ids": [1, 2]}],
    "edges": [{"src": "A", "dst": "B", "transit_time_s": 0.25}]
  }
}
```

`edges` is present only when the entity defines a skeleton. `topology` is present
only for MTMC scenes that declare one (see below).

### Three per-camera fields, kept distinct

- **`in_view`** (bool) — the point projects **in front** of the camera (`w > 0`)
  **and** inside the image bounds `[0,width) x [0,height)`. Pure framing: ignores
  occluders. A point in a blind gap is `in_view=false` on **every** camera.
- **`visible`** (bool) — the **hard DLT mask**: `in_view AND not occluded` (its
  segment to the camera centre is not blocked by any occluder). So **`visible`
  implies `in_view`**; a consumer masks triangulation on this field.
- **`occ_frac`** (float, optional) — a **continuous difficulty knob**: the
  fraction of a small deterministic jittered sample around the point whose
  sightline is blocked. It **never** feeds the mask; it only grades how marginal
  an occlusion is.

The sampler is configurable via `occ_frac_sample_count` and `occ_frac_jitter`,
threaded through `observe` / `build_manifest` / `write_manifest` as optional
keyword-only settings. `occ_frac_sample_count` is the total number of samples
(centre point + deterministic jitter offsets); `occ_frac_jitter` is the radius
of the neighbourhood in scene units. The defaults (`sample_count=7`,
`jitter=0.05`) reproduce the original manifest output byte-for-byte. Increasing
`sample_count` adds more deterministic directions (face and space diagonals of
a cube after the six axis-aligned offsets) and therefore grades a marginal
occlusion more finely. The sampling stays deterministic and RNG-free, so the
same scene and settings always produce the same `occ_frac`.

The manifest path is **non-raising**: an out-of-frame or behind-camera point is
labelled (`in_view=false`, `visible=false`), not an error. Its `uv` is sanitised
to finite values, and the manifest is written with `allow_nan=False`, so the JSON
is always strict (no `Infinity`/`NaN`).

Floats are serialized at full double precision (no rounding), so a consumer that
rebuilds `P = K [R | t]` recovers ground truth to ~machine epsilon.

### `topology` — the MTMC adjacency contract (optional, top-level)

For multi-target multi-camera scenes with **non-overlapping** stations, the
manifest carries an optional `topology`:

- **`stations`**: `[{ "id": str, "camera_ids": [int, ...] }]` — a named place and
  the cameras that share (roughly) its view. Station ids are unique.
- **`edges`**: `[{ "src": str, "dst": str, "transit_time_s": float }]` — **directed**
  adjacency: an object leaving `src`'s coverage reaches `dst`'s coverage after
  `transit_time_s` seconds. Endpoints must be declared station ids.

A consumer's MTMC / re-identification path uses this to bound how long a target
may be unseen while crossing the blind gap between adjacent stations. The
`entity.id` is the cross-camera ground-truth identity a tracker must preserve
across that gap (issue #11, stable track ids).

## Smoke (the proof)

`build_smoke_scene()` hand-specifies 3 ring cameras, 1 point moving on a straight
path, and 1 sphere on **camera 1's** sightline that occludes it during a middle
interval (frames 3–7) while cameras 0 and 2 keep the point. The test writes the
manifest, reloads the JSON, rebuilds the projection matrices from `K, R, t`, and
triangulates a cam-1-occluded frame **from the other two views only** (masking on
`visible`) through the **real** `multicam_occlusion.triangulate_dlt` — recovering
ground truth to within `1e-6`. This proves occluded-in-one-view recovery through
the actual consumer reader.

### MTMC blind-gap smoke (`build_mtmc_scene()`)

Non-overlapping counterpart. Three cameras on two stations — **A** = camera 0
alone, **B** = cameras 1 & 2 as an overlapping stereo pair (slightly different
per-camera targets/fov) — with disjoint FOVs, plus one object with a **stable
`entity.id`** sweeping A → gap → B. The manifest exhibits, in one take, all three
coverage regimes: station-A single-camera (`in_view=[T,F,F]`, correctly **not**
triangulable) → a labelled **blind gap** (`in_view` false on every camera) →
station-B stereo (`in_view=[F,T,T]`), where the **real**
`multicam_occlusion.triangulate_dlt` recovers ground truth to ~machine epsilon
(≈2e-15) from the two covering views. The scene emits a `topology` (A↔B with a
transit time). This proves the `in_view` framing signal, the labelled blind gap,
stable cross-camera identity, and topology end to end through the real reader.

## DSL grammar (`multicam_sim.dsl`)

A fluent, typed sugar layer that **compiles down to the Scene/manifest above** —
it adds no fields and changes no convention. Everything below is pydantic-typed,
validated at construction, and CPU-only; the renderer is the sole optional part.

### Camera rig — `CameraRig -> list[Camera]`

Every camera is built through `Camera.look_at` (or, for `custom`, stored
verbatim), so the RDF / Z-up / `t = -R@C` convention is never re-derived.

```
CameraRig.ring(n, radius, height, look_at, *, width, height_px, focal | fov_deg)
CameraRig.line(n, start, end, look_at, *, width, height_px, focal | fov_deg)
CameraRig.custom(extrinsics=[(R, t), ...], *, width, height_px, focal | fov_deg)
CameraRig.stations([StationView(position, look_at, focal|fov_deg, width?, height_px?), ...],
                   *, width, height_px, focal | fov_deg)   # shared defaults
```

Intrinsics take **exactly one** of `focal` (pixels) or `fov_deg` (horizontal
FOV); `focal = (width/2) / tan(fov/2)`. `ring` uses the smoke ring convention:
eye `i = (radius·cos t, radius·sin t, height)`, `t = 2π·i/n`.

**`stations` — non-overlapping / heterogeneous preset.** Unlike `ring` (one
shared target, overlapping views), each `StationView` gives its **own** eye
`position` and `look_at`, and may **override** the rig-wide intrinsics with its
own `focal`/`fov_deg` and `width`/`height_px`. One preset, two modes:

- **MTMC**: separated stations with disjoint FOVs — an object is in at most one
  station's view at a time, and the space between them is a genuine **blind gap**
  (`in_view=false` on every camera).
- **Heterogeneous fusion**: co-located-ish stations with different targets/zoom —
  e.g. camera A wide/high framing a person's volume, camera B close/zoomed framing
  items on a bench. Different entities then fall in different cameras' `in_view`
  ("human in A not B, items in B not A"), captured by the manifest's per-entity
  per-camera `in_view` with **no schema change**.

### Movement — `Path` (a time → 3D-point function)

A `Path` is a discriminated union on `kind` (parallels `OccluderUnion`).
Geometry is `u ∈ [0,1]` (`point(u)`); timing is a separate seconds axis.

```
Path.linear(a, b)                       Path.waypoints([p0, p1, ...])   (≥2)
Path.circle(center, radius, axis)       Path.bezier([c0, c1, ...])      (≥2)

combinators:  a.then(b)   p.repeat(n)   p.over(seconds)   p.at_speed(v)
compile:      path.compile_frames(fps, num_frames, name="center") -> [EntityFrame]
```

`then`/`repeat` sum wall-clock durations; `over` rescales the whole trajectory;
`at_speed` sets duration from arc length. An **untimed** path is stretched to fill
the scene duration `(num_frames-1)/fps`; a **timed** one keeps its duration and
**holds at its final point** past the end. Output is exactly the per-frame named
points the manifest already expects.

### Occlusion — declarative schedule → real geometry

```
Occlusion.sphere(size) | .box(size) | .plane(size)
         .blocks(camera=i).during((frame0, frame1))
         [.on(entity, point_name)]  [.targeting(coverage)]
```

The schedule compiles to a **real** occluder placed on camera `i`'s sightline to
the target point at the window's middle frame; `build_manifest` then computes
`visible` **geometrically**. Two deliberate consequences:

- **The window is emergent.** `.during((3,7))` places geometry aimed at that
  window; the achieved `visible=False` interval is whatever the solid produces.
  Tests assert the *actual* manifest pattern, never assume equality with the
  request. (A static global occluder cannot be exactly per-frame time-gated —
  that would need a per-frame-occluder schema change, out of scope for the
  contract. Escalate if a concrete case can't be realized.)
- **`visible` is never faked.** The hard boolean stays geometric truth. `coverage`
  is a **monotonic difficulty knob** that scales occluder size, moving the
  continuous `occ_frac` readback (quantised to eighths by the manifest sampler);
  it never touches the triangulation mask. This is the dialable dose that couples
  to multicam-occlusion's occlusion dose-response.

`plane` is a thin flat box (finite-plane approximation), so no new occluder type
enters the contract's `OccluderUnion`.

### Assembly — `SceneBuilder -> Scene`

```
Scene = (
    SceneBuilder(fps, num_frames)
    .cameras(CameraRig.ring(...))
    .entity("obj", Path.linear(a, b))
    .occlude(Occlusion.sphere(0.15).blocks(camera=1).during((3, 7)))
    .build()
)
```

The result is the ordinary `Scene`; `build_manifest(scene)` is unchanged. A DSL
scene that reproduces the smoke setup recovers ground truth for a cam-1-occluded
frame through the real `triangulate_dlt`, exactly like the hand-built smoke.

## Renderer backend (`multicam_sim.dsl.render`) — not the contract

`RendererBackend` is a `Protocol` (Scene + camera + frame → `(H,W,3)` pixels).
`PyrenderBackend` is an offscreen v1; `pyrender`/`trimesh` are an optional
`render` extra imported lazily, never at package load and never in CI — pixels
cannot break the manifest's green bar. `KubricBackend`
(`multicam_sim.dsl.kubric_backend`, the `kubric` extra) is the photoreal
open/closed swap of this Protocol: its Blender-free coordinate translation lives
in `multicam_sim.dsl.kubric_spec` and is unit-tested (the built camera projects a
point identically to `P = K[R|t]` within `1e-6`); the actual Blender render runs
only inside the `kubricdockerhub/kubruntu` image. The exact OpenCV-RDF →
Blender-camera conversion, docker recipe, and GT cross-check are in
[`docs/kubric.md`](docs/kubric.md). A **Rust core via `pyo3`** for the analytic
projection/occlusion path is a v2 concern.

## Pose manifest extension

Human pose reuses the named-points design with **zero schema fork**. It is not a
new manifest; it is a typed way to build the entities the existing builder
already understands.

- A **human** = **1 entity** with **17 named joints** (COCO-17) plus an `edges`
  skeleton (19 limbs). The joints are ordinary named points, so
  `build_manifest` labels each joint exactly like any other point.
- `PoseTrajectory.to_entity()` (`src/multicam_sim/pose.py`) lowers a skeleton +
  per-frame joints to a plain `Entity`. Nothing in the manifest schema above
  changes.

So for every joint the manifest already carries:

- **GT 3D** — `xyz_gt`, the joint's world position;
- **2D keypoint per camera** — `per_cam[i].uv`;
- **per-joint occlusion** — `per_cam[i].visible` and `per_cam[i].occ_frac`. A
  joint blocked in one view but seen in others is `visible: false` for the
  blocked camera and `true` for the rest, per the same hard-DLT contract used for
  object points.

This per-joint, per-view visibility is exactly the input a **multi-view 3D human
pose estimator** in multicam-occlusion consumes: triangulate each joint from the
cameras that still see it, and use the occluded views as held-out difficulty.

**Default skeleton:** COCO-17 (17 keypoints, the canonical 19-edge skeleton),
`Skeleton.coco17()`. **Dense body models** (SMPL / SMPL-X) are an open/closed
extension point: implement the `MeshBackend` ABC to emit a `PoseTrajectory`, and
the projection/occlusion/manifest path is unchanged. No mesh backend is
implemented in this layer.
