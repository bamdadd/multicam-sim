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
