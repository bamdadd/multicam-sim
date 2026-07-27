"""Renderer backend: Protocol conformance, the headless ordering guard, gated smokes.

The renderer is optional (pixels are not the contract). This test never fails CI
for its absence: the pyrender paths are guarded (``importorskip`` in-process,
``find_spec`` for the subprocess smoke) and the module import itself must not drag
in pyrender.

The headless ordering guard is deliberately *not* gated — it spies both ends of the
contract instead of rendering, so the regression it guards (issue #36: the colour
path must select a GL platform before pyrender is imported) is caught even on a box
without the extra.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from multicam_sim.dsl import CameraRig, Occlusion, Path, SceneBuilder
from multicam_sim.dsl.render import PyrenderBackend, RendererBackend

_HEADLESS_COLOUR_SMOKE = """
import json
import os

import numpy as np

from multicam_sim.dsl.render import PyrenderBackend
from multicam_sim.smoke import build_smoke_scene

# Nothing here configures the platform: render() must do it, or this dies with
# pyglet's NoSuchDisplayException.
img = PyrenderBackend(point_radius=0.1).render(build_smoke_scene(), camera_id=0, frame=5)
print(json.dumps({
    "platform": os.environ.get("PYOPENGL_PLATFORM"),
    "shape": list(img.shape),
    "dtype": str(img.dtype),
    "nonzero": int(np.count_nonzero(img)),
}))
"""


class _StopAtImport(Exception):
    """Ends the render at the moment pyrender would have been imported."""


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


def test_render_configures_headless_before_importing_pyrender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no DISPLAY and no explicit platform, .render() must have PYOPENGL_PLATFORM
    # in force by the time pyrender is imported -- PyOpenGL binds it at import time,
    # so configuring afterwards is a silent no-op. Both ends are spied, so the order
    # is asserted, not just the final value.
    from multicam_sim.dsl import depth as depth_mod
    from multicam_sim.dsl import render as render_mod

    monkeypatch.delenv("DISPLAY", raising=False)
    # setenv first: the real configure_headless writes PYOPENGL_PLATFORM into
    # os.environ, and a bare delenv(raising=False) records nothing when the variable
    # was already absent -- so the write would leak into the rest of the session.
    monkeypatch.setenv("PYOPENGL_PLATFORM", "")
    monkeypatch.delenv("PYOPENGL_PLATFORM")
    # configure_headless refuses to run once PyOpenGL is imported; a sibling smoke may
    # have pulled it in, so restore the fresh-interpreter state this contract assumes.
    monkeypatch.delitem(sys.modules, "OpenGL", raising=False)

    real_configure = depth_mod.configure_headless
    events: list[tuple[str, str | None]] = []

    def spy_configure(platform: str = "osmesa") -> str:
        chosen = real_configure(platform)
        events.append(("configure_headless", os.environ.get("PYOPENGL_PLATFORM")))
        return chosen

    def spy_import(what: str) -> tuple[object, object]:
        events.append(("import_pyrender", os.environ.get("PYOPENGL_PLATFORM")))
        raise _StopAtImport(what)

    monkeypatch.setattr(depth_mod, "configure_headless", spy_configure)
    monkeypatch.setattr(render_mod, "_import_pyrender", spy_import)

    with pytest.raises(_StopAtImport):
        PyrenderBackend().render(_scene(), camera_id=0, frame=5)

    assert [name for name, _ in events] == ["configure_headless", "import_pyrender"]
    assert events[1][1] == "osmesa"  # already in force at the import, not set after it


def test_render_produces_an_image_when_pyrender_present() -> None:
    pytest.importorskip("pyrender")
    pytest.importorskip("trimesh")
    backend = PyrenderBackend(point_radius=0.1)
    img = backend.render(_scene(), camera_id=0, frame=5)
    assert img.shape == (48, 64, 3)
    assert img.dtype.kind == "u"


@pytest.mark.skipif(
    importlib.util.find_spec("pyrender") is None or importlib.util.find_spec("trimesh") is None,
    reason="needs the 'render' extra: pip install multicam-sim[render]",
)
def test_render_headless_smoke_with_no_display() -> None:
    """One colour frame rendered with ``DISPLAY`` unset — the real proof for #36.

    Runs in a subprocess, like ``tests/test_depth.py``'s headless smoke: PyOpenGL
    binds its platform at first import, so the choice has to be made in a fresh
    interpreter (and ``find_spec`` probes availability without importing).
    """
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "PYOPENGL_PLATFORM")}
    proc = subprocess.run(
        [sys.executable, "-c", _HEADLESS_COLOUR_SMOKE],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, f"headless colour render failed:\n{proc.stderr}"
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    assert out["platform"] == "osmesa"  # render() selected it; nothing else did
    assert out["shape"] == [480, 640, 3]
    assert out["dtype"] == "uint8"
    assert out["nonzero"] > 0  # the lit scene is not a black frame
