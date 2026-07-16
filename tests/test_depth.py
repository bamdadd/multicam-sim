"""Depth backend: Protocol conformance + a gated headless pyrender smoke.

Depth is additive and optional — it never feeds the manifest. Like ``test_render``,
the pyrender path is gated so CI stays green without the ``render`` extra, and the
module import itself must not drag pyrender in.

The headless smoke runs in a **subprocess** for two reasons: PyOpenGL binds its
platform at import time (so the platform must be chosen in a fresh interpreter),
and importing pyrender in-process would pollute ``sys.modules`` for the
import-light assertions here and in ``test_render``. Availability is probed with
``find_spec`` rather than ``importorskip`` for the same reason.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest

from multicam_sim.dsl.depth import DepthBackend, PyrenderDepthBackend, configure_headless

_SMOKE = """
import json
import numpy as np
from multicam_sim.dsl.depth import PyrenderDepthBackend, configure_headless
from multicam_sim.smoke import build_smoke_scene

platform = configure_headless()  # before the first pyrender import
scene = build_smoke_scene()
depth = PyrenderDepthBackend(point_radius=0.1).render_depth(scene, camera_id=0, frame=5)

frame5 = next(f for f in scene.entities[0].frames if f.frame == 5)
point = np.asarray(next(iter(frame5.points.values())), dtype=np.float64)
_, w = scene.cameras[0].project(point)
hit = depth[depth > 0.0]
print(json.dumps({
    "platform": platform,
    "shape": list(depth.shape),
    "dtype": str(depth.dtype),
    "hit_px": int(hit.size),
    "min_depth": float(hit.min()) if hit.size else None,
    "analytic_w": float(w),
}))
"""


def test_pyrender_depth_backend_satisfies_protocol() -> None:
    # structural check only — no pyrender import needed.
    assert isinstance(PyrenderDepthBackend(), DepthBackend)


def test_importing_depth_does_not_import_pyrender() -> None:
    # constructing the backend must not require the optional extra.
    PyrenderDepthBackend()
    assert "pyrender" not in sys.modules  # only imported inside .render_depth()


def test_configure_headless_respects_an_explicit_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    assert configure_headless("osmesa") == "egl"  # never overrides an explicit choice


def test_configure_headless_defers_to_a_display(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    assert configure_headless() == ""  # a display can render on its own


def test_configure_headless_rejects_a_late_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # PyOpenGL binds its platform at import time; setting the env afterwards is a
    # silent no-op, so we surface it instead.
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setitem(sys.modules, "OpenGL", object())
    with pytest.raises(RuntimeError, match="already imported"):
        configure_headless()


@pytest.mark.skipif(
    importlib.util.find_spec("pyrender") is None or importlib.util.find_spec("trimesh") is None,
    reason="needs the 'render' extra: pip install multicam-sim[render]",
)
def test_render_depth_headless_smoke() -> None:
    """One smoke frame rendered headless, with depth — the minimal proof."""
    proc = subprocess.run(
        [sys.executable, "-c", _SMOKE], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, f"headless render failed:\n{proc.stderr}"
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    assert out["shape"] == [480, 640]
    assert out["dtype"] == "float32"
    assert out["hit_px"] > 0  # the moving point is in frame 5 of camera 0

    # The depth buffer is camera-space z, which is exactly the `w` the analytic
    # projection already returns — so they must agree at the surface of the
    # radius-0.1 sphere drawn around the point. This CROSS-CHECKS the analytic
    # geometry; it does not feed (or change) `visible`.
    assert out["min_depth"] == pytest.approx(out["analytic_w"] - 0.1, abs=0.01)
