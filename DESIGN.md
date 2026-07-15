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
                {"cam": 0, "uv": [u, v], "visible": true, "occ_frac": 0.0}
              ]
            }
          }
        }
      ]
    }
  ]
}
```

`edges` is present only when the entity defines a skeleton.

### Two occlusion fields, kept distinct

- **`visible`** (bool) — the **hard DLT contract**. `true` iff the point is in
  front of the camera, inside the image bounds, **and** its segment to the camera
  centre is not blocked by any occluder (ray-vs-occluder geometry). A consumer
  masks triangulation on this field.
- **`occ_frac`** (float, optional) — a **continuous difficulty knob**: the
  fraction of a small deterministic jittered sample around the point whose
  sightline is blocked. It **never** feeds the triangulation mask; it only grades
  how marginal an occlusion is.

Floats are serialized at full double precision (no rounding), so a consumer that
rebuilds `P = K [R | t]` recovers ground truth to ~machine epsilon.

## Smoke (the proof)

`build_smoke_scene()` hand-specifies 3 ring cameras, 1 point moving on a straight
path, and 1 sphere on **camera 1's** sightline that occludes it during a middle
interval (frames 3–7) while cameras 0 and 2 keep the point. The test writes the
manifest, reloads the JSON, rebuilds the projection matrices from `K, R, t`, and
triangulates a cam-1-occluded frame **from the other two views only** (masking on
`visible`) through the **real** `multicam_occlusion.triangulate_dlt` — recovering
ground truth to within `1e-6`. This proves occluded-in-one-view recovery through
the actual consumer reader.

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
```

Intrinsics take **exactly one** of `focal` (pixels) or `fov_deg` (horizontal
FOV); `focal = (width/2) / tan(fov/2)`. `ring` uses the smoke ring convention:
eye `i = (radius·cos t, radius·sin t, height)`, `t = 2π·i/n`.

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
cannot break the manifest's green bar. **Kubric/Blender** is a future
open/closed swap of this Protocol; a **Rust core via `pyo3`** for the analytic
projection/occlusion path is a v2 concern.
