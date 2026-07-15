"""Renderer backend: Protocol conformance + a gated pyrender smoke.

The renderer is optional (pixels are not the contract). This test never fails CI
for its absence: the pyrender path is guarded by ``importorskip`` and the module
import itself must not drag in pyrender.
"""

from __future__ import annotations

import pytest

from multicam_sim.dsl import CameraRig, Occlusion, Path, SceneBuilder
from multicam_sim.dsl.render import PyrenderBackend, RendererBackend


def _scene():
    return (
        SceneBuilder(fps=30.0, num_frames=11)
        .cameras(
            CameraRig.ring(
                n=3,
                radius=4.0,
                height=1.5,
                look_at=(0.0, 0.0, 0.5),
                focal=800.0,
                width=64,
                height_px=48,
            )
        )
        .entity("obj", Path.linear((0.0, -0.6, 0.5), (0.0, 0.6, 0.5)))
        .occlude(Occlusion.sphere(size=0.15).blocks(camera=1).during((3, 7)))
        .build()
    )


def test_pyrender_backend_satisfies_protocol() -> None:
    # structural check only — no pyrender import needed.
    backend = PyrenderBackend()
    assert isinstance(backend, RendererBackend)


def test_importing_render_does_not_import_pyrender() -> None:
    # constructing the backend must not require the optional extra.
    import sys

    PyrenderBackend()
    assert "pyrender" not in sys.modules  # only imported inside .render()


def test_render_produces_an_image_when_pyrender_present() -> None:
    pytest.importorskip("pyrender")
    pytest.importorskip("trimesh")
    backend = PyrenderBackend(point_radius=0.1)
    img = backend.render(_scene(), camera_id=0, frame=5)
    assert img.shape == (48, 64, 3)
    assert img.dtype.kind == "u"
