"""Headless offscreen depth via pyrender — **additive, never the contract**.

The manifest's three per-camera fields (``in_view`` / ``visible`` / ``occ_frac``)
stay analytic and GL-free; see DESIGN.md "Three per-camera fields, kept distinct".
Nothing in this module feeds them. A rendered depth buffer is only ever a *second
opinion* a caller may use to cross-check or grade a scene.

Why a separate module from :mod:`multicam_sim.dsl.render`:

* ``RendererBackend`` returns colour pixels; depth is a different return shape, so
  :class:`DepthBackend` is its own Protocol (open/closed, same as the colour side).
* headless GL needs a platform decision (:func:`configure_headless`) that the
  colour path deliberately does not make for you.

Depth semantics (measured, see ``docs/renderer-eval.md``): the returned buffer is
**camera-space z** in scene units — the perpendicular distance to the image plane,
*not* radial range — and ``0.0`` where the ray hit nothing. That is exactly the
``w`` that :meth:`multicam_sim.cameras.Camera.project` already returns, which is
what makes an analytic-vs-rendered cross-check a direct comparison.

Requires the ``render`` extra (``pip install multicam-sim[render]``), imported
lazily: importing this module must never drag in pyrender.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ..geometry import FloatArray
from .render import _build_pyrender_scene, _import_pyrender

if TYPE_CHECKING:
    from ..scene import Scene

#: PyOpenGL's platform switch. Must be set *before* pyrender is first imported.
_PLATFORM_ENV = "PYOPENGL_PLATFORM"


def configure_headless(platform: str = "osmesa") -> str:
    """Select a headless GL platform when there is no display, and report it.

    Idempotent and deliberately non-destructive: an explicit ``PYOPENGL_PLATFORM``
    is never overridden, and nothing is set when a ``DISPLAY`` exists (that box can
    render on its own). Returns the platform in force ("" == PyOpenGL's default).

    Must be called *before* pyrender is first imported — PyOpenGL reads this at
    import time. :meth:`PyrenderDepthBackend.render_depth` calls it for you.

    ``osmesa`` (software rasteriser, needs the ``libosmesa6`` system package) is
    the default because it needs no GPU; ``egl`` is the faster choice where a GPU
    and ``libEGL`` are present.
    """
    existing = os.environ.get(_PLATFORM_ENV)
    if existing:
        return existing
    if os.environ.get("DISPLAY"):
        return ""
    if "OpenGL" in sys.modules:
        # PyOpenGL resolved its platform at import time and would ignore us; the
        # symptom is otherwise an opaque
        # "AttributeError: 'GLXPlatform' object has no attribute 'OSMesa'".
        raise RuntimeError(
            f"PyOpenGL is already imported, so {_PLATFORM_ENV}={platform!r} would be "
            f"ignored. Call configure_headless() (or set {_PLATFORM_ENV}) before the "
            "first pyrender/OpenGL import."
        )
    os.environ[_PLATFORM_ENV] = platform
    return platform


@runtime_checkable
class DepthBackend(Protocol):
    """Anything that renders one camera's view of a scene at a frame to depth.

    The return is an ``(H, W)`` float32 buffer of camera-space z in scene units,
    ``0.0`` where nothing was hit. Backends are interchangeable; none of them
    feeds the manifest.
    """

    def render_depth(self, scene: Scene, camera_id: int, frame: int) -> FloatArray: ...


class PyrenderDepthBackend:
    """Offscreen pyrender depth (optional extra; skipped in CI when absent).

    Constructing this is cheap; :meth:`render_depth` lazily imports pyrender and
    raises a clear error if the ``render`` extra is missing.
    """

    def __init__(self, *, point_radius: float = 0.05, headless_platform: str = "osmesa"):
        self.point_radius = point_radius
        self.headless_platform = headless_platform

    def render_depth(self, scene: Scene, camera_id: int, frame: int) -> FloatArray:
        configure_headless(self.headless_platform)
        pyrender, _ = _import_pyrender("PyrenderDepthBackend")
        pr_scene, intr = _build_pyrender_scene(
            scene,
            camera_id,
            frame,
            point_radius=self.point_radius,
            bg=(0.0, 0.0, 0.0),
            add_light=False,
        )
        renderer = pyrender.OffscreenRenderer(intr.width, intr.height)
        try:
            _, depth = renderer.render(pr_scene)
        finally:
            renderer.delete()
        return np.asarray(depth, dtype=np.float32)
