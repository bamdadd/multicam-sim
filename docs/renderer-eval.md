# Renderer backend evaluation: Open3D vs pyrender

Answers [#10](https://github.com/bamdadd/multicam-sim/issues/10). Every number and
error below was produced by running the thing in the environment described next —
nothing is quoted from documentation. Claims that could **not** be tested here are
marked *(unverified)*.

## Environment

Debian 11 (bullseye), kernel 5.10, x86_64, 5 cores. **Headless: no `DISPLAY`, no
GPU** — the only adapter is a `VMware SVGA II` with no 3D (`lspci`; Mesa reports
`VMware: No 3D enabled (0, Success)`). This is deliberately CI-shaped. Python
3.13.1 for pyrender, 3.12.13 for Open3D (see *Python support* below), uv 0.11.23.
A GPU/EGL machine will behave differently on the headless axis — calibrate
accordingly.

## Recommendation

**Keep pyrender.** It is already the incumbent (`multicam_sim.dsl.render`), and on
every axis that matters to this repo it wins or ties:

| Axis | pyrender 0.1.45 | Open3D 0.19.0 |
| --- | --- | --- |
| **Headless render** | ✅ via OSMesa — **but** needs `libosmesa6` **and** a PyOpenGL override | ✅ via software EGL — **but** needs exactly the right env vars, else **segfault** |
| **Failure mode when misconfigured** | Python exception (catchable, greppable) | **SIGSEGV, exit 139** — uncatchable, kills the worker |
| **Depth buffer** | ✅ `color, depth = r.render(scene)` — free, always there | ✅ `render_to_depth_image(z_in_view_space=True)` |
| **Depth semantics** | float32 `(H,W)`, camera-space z, `0.0` = miss | float32 `(H,W)`, camera-space z, `inf` = miss |
| **Install** | **4.6 s**, **19 packages**, **227 MB** | **40.8 s**, **81 packages**, **1.6 GB** |
| **Wheel** | `py3-none-any` (pure Python) | `cp312-cp312-manylinux_2_31_x86_64` |
| **Python support** | any (pure Python) — 3.13 ✅ | **no cp313 wheel** — caps the project at ≤3.12 |
| **License** | MIT (`trimesh` MIT, `PyOpenGL` BSD) | MIT |

The decisive points are **blast radius** and **failure mode**, not raw capability —
both can render, and both give an equally usable depth buffer.

1. **Open3D would cap `requires-python` at 3.12.** `pyproject.toml` declares
   `>=3.11`, and this box runs 3.13. Open3D publishes no cp313 wheel, so it is
   simply uninstallable here. pyrender's wheel is `py3-none-any` and installed on
   3.13 without complaint.
2. **1.6 GB vs 227 MB.** Open3D pulls 81 packages — `pandas`, `plotly`, `dash`,
   `scikit-learn`, `werkzeug`, Jupyter widgets — for a library we would use to draw
   spheres. Its wheel also ships a **766 MB CUDA `pybind` .so** on a box with no
   GPU. That is a big cache line-item for an extra that "pixels are not the
   contract" says should stay cheap.
3. **Open3D fails by segfaulting.** Misconfigured, it does not raise — the
   interpreter dies (exit 139). A backend that can `SIGSEGV` a pytest worker is a
   bad fit for something explicitly kept off the critical path.
4. **pyrender is already wired in.** `RendererBackend` + `PyrenderBackend` exist,
   and the `render` extra already names it. Recommending it costs no migration.

Open3D's genuine advantages — a much richer 3D toolkit, and a well-lit EGL path
where a GPU exists *(unverified: no GPU here)* — do not pay for the above, given
this repo only needs "put some spheres on screen and read depth".

## What actually happened

### pyrender

```
uv pip install -e '.' 'pyrender>=0.1.45' 'trimesh>=4.0'    # 4.6s, 19 pkgs, 227MB
```

Three distinct headless failures, in order:

1. **Default (pyglet) platform** — the shipped `PyrenderBackend` path:
   ```
   pyglet.display.xlib.NoSuchDisplayException: Cannot connect to "None"
   ```
2. **`PYOPENGL_PLATFORM=egl`, before installing system libs**:
   ```
   OSError: ('EGL: cannot open shared object file: No such file or directory', 'EGL', None)
   ```
   After `apt-get install libegl1 libosmesa6 libgl1-mesa-dri` (12.9 MB download,
   11.4 MB on disk, 4.5 s), EGL loads but cannot initialise on a GPU-less box:
   ```
   OpenGL.error.GLError(baseOperation = eglInitialize, result = 0)
   ```
3. **`PYOPENGL_PLATFORM=osmesa`** — the real trap:
   ```
   ImportError: cannot import name 'OSMesaCreateContextAttribs' from 'OpenGL.osmesa'
   ```
   **pyrender's own pin breaks its own headless path.** pyrender declares
   `Requires-Dist: PyOpenGL (==3.1.0)`, and 3.1.0 has `OSMesaCreateContextExt` but
   *not* `OSMesaCreateContextAttribs`, which `pyrender/platforms/osmesa.py` imports
   (verified directly with `hasattr`).

Forcing `pyopengl==3.1.10` fixes it, and the smoke frame renders:

```
PYOPENGL_PLATFORM=osmesa -> OK shape=(480, 640, 3) dtype=uint8 nonzero=10641
```

The `==` pin cannot be widened from the extra — `uv` rejects
`pyopengl>=3.1.7` against `pyrender`'s `==3.1.0` as unsatisfiable, and
`uv lock --check` fails. It is therefore expressed as
`[tool.uv] override-dependencies = ["pyopengl>=3.1.7"]`, which locks cleanly
(pyopengl 3.1.0 → 3.1.10, no other churn). pyopengl only enters the graph via the
`render` extra, which CI does not install.

**Working recipe:** `libosmesa6` + `pyopengl>=3.1.7` + `PYOPENGL_PLATFORM=osmesa`
set **before the first pyrender import** (PyOpenGL binds its platform at import
time; setting it late fails with a misleading
`AttributeError: 'GLXPlatform' object has no attribute 'OSMesa'`).
`multicam_sim.dsl.depth.configure_headless()` encapsulates this.

### Open3D

```
uv pip install -e '.' 'open3d'    # on 3.13:
  × No solution found ... only found wheels for open3d (v0.19.0) with the
    following Python ABI tags: cp38, cp39, cp310, cp311, cp312
```

Re-run on Python 3.12: installs in **40.8 s**, **81 packages**, **1.6 GB**
site-packages (open3d alone 1.1 GB).

Out of the box, `OffscreenRenderer(640, 480)` **segfaults**:

```
[Open3D INFO] EGL headless mode enabled.
FEngine (64 bits) created at 0x7fef33da3010 (threading is enabled)
eglInitialize failed
Segmentation fault      (exit 139)
```

This is not a Python exception — `try/except` cannot catch it. With Mesa's
software EGL selected explicitly it works:

```
LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless python try_open3d.py
  EGL(1.5) / OpenGL(4.5)
  COLOR OK shape=(480, 640, 3) dtype=uint8
  DEPTH OK shape=(480, 640) dtype=float32 min=4.0236 max=4.3861
```

(`LIBGL_ALWAYS_SOFTWARE=1` alone still segfaults; `EGL_PLATFORM=surfaceless` alone
fails with `libEGL warning: DRI2: failed to create dri screen`. Both are needed,
plus `libgl1-mesa-dri`.)

## Depth-buffer access, and what it's worth

Both expose depth as a float32 `(H, W)` buffer of **camera-space z** — the
perpendicular distance to the image plane, **not** radial range. Measured with a
deliberately off-axis point (frame 0, camera 0), pyrender:

| quantity | value |
| --- | --- |
| analytic radial range | 4.16653 |
| analytic camera-space z (`Camera.project`'s `w`) | 4.12311 |
| rendered depth at the projected pixel | 4.02507 |
| z at the sphere surface (z − r·z/range) | 4.02415 ✅ |
| radial at the sphere surface (range − r) | 4.06653 ❌ |

So **`depth` ≈ the `w` that `Camera.project` already returns**, which makes an
analytic-vs-rendered cross-check a direct comparison — no conversion, no fudge
factor. `tests/test_depth.py` asserts exactly this on the smoke scene.

That relation is what makes depth useful to #10's "complement or cross-check
`visible`". Rendering the smoke scene from **camera 1** (the occluded view) at
frame 5 shows it working:

| camera | rendered depth (min) | analytic range to point | reading |
| --- | --- | --- | --- |
| 0 | 4.0233 | 4.1231 | depth ≈ point surface → unoccluded |
| 1 | **3.3548** | 4.1231 | depth **far nearer** → something is in front → occluded |

Camera 1's depth buffer independently reproduces the sphere occlusion that the
analytic `visible=False` already encodes — from pixels, with no knowledge of
`Occluder.blocks_segment`. That is a real, cheap cross-check for the geometry.

**It stays additive.** Per DESIGN.md "Three per-camera fields, kept distinct",
`in_view` / `visible` / `occ_frac` remain analytic and GL-free. Nothing in
`multicam_sim.dsl.depth` touches `occluders.py` or the manifest; depth is a second
opinion a caller may ask for, never an input to the mask.

## License

Checked from the installed wheel metadata, not from memory:

- **pyrender 0.1.45** — MIT (`Classifier: License :: OSI Approved :: MIT License`).
  `trimesh` MIT, `PyOpenGL` BSD.
- **Open3D 0.19.0** — MIT (`LICENSE.txt`: "The MIT License (MIT) / Open3D:
  www.open3d.org").

Both are compatible with this project's Apache-2.0. No differentiator.
*(Unverified: Open3D's wheel bundles native third-party components — Filament and
friends — whose licenses are not enumerated in the `dist-info`. A vendoring
review would need to look at the Open3D source tree, not the wheel.)*

## Known issue this evaluation surfaced

`PyrenderBackend.render` (`src/multicam_sim/dsl/render.py`) **cannot render
headless today**: it takes pyglet's default platform and dies with
`NoSuchDisplayException`. Reproduced on pristine `main` with the `render` extra
installed — `tests/test_render.py::test_render_produces_an_image_when_pyrender_present`
fails on this box. CI never installs the extra, so CI is unaffected and green.

This is left alone deliberately (out of scope for a research issue, and the fix is
a behaviour change to the colour path). The remedy is the same
`configure_headless()` the depth backend uses; happy to do it in a follow-up if
wanted.

## Reproducing

```bash
uv sync --group dev
uv pip install -e '.[render]'          # picks up the pyopengl override
apt-get install -y libosmesa6          # the one system dep
uv run pytest tests/test_depth.py -v   # the headless smoke, with depth
```

Without the extra the smoke skips, like the rest of the optional-dep tests.
