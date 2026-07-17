"""Shared test fixtures.

The rasterizer is the suite's **default** :class:`RendererBackend`: unlike the
pyrender/Kubric backends (guarded by ``importorskip`` because they need native
libraries), it is pure numpy and always runs. Tests that want "a backend" take
the ``backend`` fixture rather than hard-coding one.
"""

from __future__ import annotations

import pytest

from multicam_sim.dsl.raster import RasterizerBackend, default_backend
from multicam_sim.dsl.render import RendererBackend


@pytest.fixture
def backend() -> RendererBackend:
    """The default backend for the suite — the zero-dependency rasterizer."""
    return default_backend()


@pytest.fixture
def rasterizer() -> RasterizerBackend:
    """The concrete rasterizer, for tests that touch its own config/methods."""
    return RasterizerBackend()
