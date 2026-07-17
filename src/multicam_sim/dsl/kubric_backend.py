"""Photoreal Kubric/Blender backend for :class:`RendererBackend` (issue #39).

A same-Protocol, open/closed swap of :class:`~multicam_sim.dsl.render.PyrenderBackend`
that renders through **Kubric** (Google Research's Blender wrapper). Like every
renderer in this package, **pixels are not the contract**: this backend never
feeds the manifest and is never exercised in CI.

The coordinate math lives in the pure, Blender-free
:mod:`multicam_sim.dsl.kubric_spec` (:func:`~multicam_sim.dsl.kubric_spec.scene_to_kubric_spec`)
so it can be unit-tested without a GPU or a Blender install. This module is only
the thin adapter that hands that typed spec to ``kb.*`` and returns RGB.

Kubric **cannot ``pip install`` cleanly** — it needs Blender's bundled Python and
native libraries. The supported way to actually render is the maintained docker
image::

    docker run --rm -v "$PWD:/work" -w /work kubricdockerhub/kubruntu \\
        python3 your_render_script.py

See ``docs/kubric.md`` for the exact conversion, the docker recipe, and how the
Kubric ground truth (segmentation / depth / object positions) cross-checks the
analytic manifest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..geometry import FloatArray
from .kubric_spec import (
    DEFAULT_SENSOR_WIDTH_MM,
    KubricSceneSpec,
    scene_to_kubric_spec,
)

if TYPE_CHECKING:
    from ..scene import Scene


def _import_kubric(what: str) -> Any:
    """Lazily import ``kubric`` (as ``kb``), or raise a clear install hint.

    Kubric depends on Blender's Python and native libs, so it does not
    ``pip install`` cleanly; the maintained path is the ``kubricdockerhub/kubruntu``
    docker image. ``what`` names the caller so the message points at the right
    class.
    """
    try:
        import kubric as kb
    except ImportError as exc:  # pragma: no cover - optional extra, needs Blender
        raise ImportError(
            f"{what} needs the 'kubric' extra AND Blender: `pip install "
            "multicam-sim[kubric]` only pulls the Python package, which cannot run "
            "without Blender's bundled interpreter. Render inside the maintained "
            "image instead: `docker run --rm -v \"$PWD:/work\" -w /work "
            "kubricdockerhub/kubruntu python3 <script>.py`. See docs/kubric.md."
        ) from exc
    return kb


def build_kubric_scene(spec: KubricSceneSpec) -> tuple[Any, Any]:
    """Assemble a ``kb.Scene`` + ``kb.PerspectiveCamera`` from a typed spec.

    Applies the spec verbatim: the scene resolution is the camera's pixel size,
    the camera takes the converted ``focal_length``/``sensor_width`` (mm) and the
    ``position`` + ``(w, x, y, z)`` ``quaternion`` from
    :func:`~multicam_sim.dsl.kubric_spec.camera_to_kubric_spec`, and each object
    spec becomes a coloured ``kb.Sphere``. Returns ``(scene, camera)``.
    """
    kb = _import_kubric("The Kubric backend")
    cam_spec = spec.camera

    scene = kb.Scene(resolution=(cam_spec.width, cam_spec.height))
    camera = kb.PerspectiveCamera(
        focal_length=cam_spec.focal_length,
        sensor_width=cam_spec.sensor_width,
        position=cam_spec.position,
        quaternion=cam_spec.quaternion,
    )
    scene.camera = camera

    for obj in spec.objects:
        material = kb.PrincipledBSDFMaterial(color=kb.Color(*obj.color))
        scene += kb.Sphere(
            name=obj.name,
            scale=obj.radius,
            position=obj.position,
            material=material,
        )
    # a key light so the spheres are not rendered flat-black.
    scene += kb.DirectionalLight(position=cam_spec.position, intensity=3.0)
    return scene, camera


class KubricBackend:
    """Photoreal Kubric/Blender backend (optional extra; never in CI).

    Constructing this is cheap and imports nothing heavy; :meth:`render` lazily
    imports ``kubric`` (which needs Blender) and raises a clear error otherwise.
    The coordinate translation is done first by the pure
    :func:`~multicam_sim.dsl.kubric_spec.scene_to_kubric_spec`, so it is testable
    without Blender.
    """

    def __init__(
        self,
        *,
        point_radius: float = 0.05,
        sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
    ) -> None:
        self.point_radius = point_radius
        self.sensor_width_mm = sensor_width_mm

    def spec_for(self, scene: Scene, camera_id: int, frame: int) -> KubricSceneSpec:
        """The pure, Blender-free spec for this view (exposed for cross-checks)."""
        return scene_to_kubric_spec(
            scene,
            camera_id,
            frame,
            point_radius=self.point_radius,
            sensor_width_mm=self.sensor_width_mm,
        )

    def render(self, scene: Scene, camera_id: int, frame: int) -> FloatArray:
        _import_kubric("KubricBackend")
        from kubric.renderer.blender import Blender

        spec = self.spec_for(scene, camera_id, frame)
        kb_scene, _ = build_kubric_scene(spec)
        renderer = Blender(kb_scene)
        frames = renderer.render(frames=[0])
        rgba = np.asarray(frames["rgba"][0], dtype=np.uint8)
        return np.asarray(rgba[..., :3], dtype=np.uint8)
