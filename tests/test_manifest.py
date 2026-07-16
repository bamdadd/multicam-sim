"""Manifest / occ_frac sampler contract.

These tests guard the configurable ``occ_frac`` sampling added in issue #7:
defaults must reproduce the original smoke output, higher sample counts grade
marginal occlusions more finely, and the parameters are validated.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam_sim import build_smoke_scene
from multicam_sim.manifest import build_manifest, observe, occlusion_fraction

# Snapshot of camera-1 occ_frac values for the smoke scene under the original
# defaults (sample_count=7, jitter=0.05). These must stay byte-identical.
_SMOKE_CAM1_OCC_FRAC = {
    0: 0.0,
    1: 0.0,
    2: 3.0 / 7.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 6.0 / 7.0,
    8: 1.0 / 7.0,
    9: 0.0,
    10: 0.0,
}


def _cam1_occ_frac(manifest: dict, frame: int) -> float:
    return manifest["entities"][0]["frames"][frame]["points"]["center"]["per_cam"][1]["occ_frac"]


def test_occ_frac_defaults_reproduce_smoke_output() -> None:
    """The default sampler settings must match the original manifest exactly."""
    scene = build_smoke_scene()
    default_manifest = build_manifest(scene)
    explicit_manifest = build_manifest(scene, occ_frac_sample_count=7, occ_frac_jitter=0.05)
    assert default_manifest == explicit_manifest

    for frame, expected in _SMOKE_CAM1_OCC_FRAC.items():
        assert _cam1_occ_frac(default_manifest, frame) == expected


def test_higher_sample_count_grades_marginal_occlusion_more_finely() -> None:
    """A larger sample count discovers more blocked directions on a partially
    occluded point, changing ``occ_frac`` in the expected (finer) direction."""
    scene = build_smoke_scene()

    # Frame 2 is a marginal occlusion: only some sightlines to camera 1 are
    # blocked. With more samples the blocked count increases monotonically.
    counts: list[int] = []
    for sample_count in (7, 19, 27):
        manifest = build_manifest(scene, occ_frac_sample_count=sample_count)
        occ_frac = _cam1_occ_frac(manifest, 2)
        assert 0.0 < occ_frac < 1.0
        blocked = round(occ_frac * sample_count)
        assert abs(blocked - occ_frac * sample_count) < 1e-9
        counts.append(blocked)

    assert counts[0] < counts[1] < counts[2]


def test_occlusion_fraction_validation() -> None:
    """``sample_count`` must be positive and within the deterministic pool;
    ``jitter`` must be non-negative."""
    scene = build_smoke_scene()
    camera = scene.cameras[0]
    point = np.array([0.0, 0.0, 0.5], dtype=np.float64)
    occluders = list(scene.occluders)

    with pytest.raises(ValueError, match="sample_count must be positive"):
        occlusion_fraction(camera, point, occluders, sample_count=0)
    with pytest.raises(ValueError, match="sample_count must be positive"):
        occlusion_fraction(camera, point, occluders, sample_count=-1)
    with pytest.raises(ValueError, match="exceeds the deterministic sample pool"):
        occlusion_fraction(camera, point, occluders, sample_count=28)
    with pytest.raises(ValueError, match="jitter radius must be non-negative"):
        occlusion_fraction(camera, point, occluders, jitter=-0.01)


def test_observe_passes_occ_frac_settings() -> None:
    """``observe`` forwards the occ_frac knobs to the returned value."""
    scene = build_smoke_scene()
    camera = scene.cameras[0]
    point = np.array(scene.entities[0].frames[0].points["center"], dtype=np.float64)
    occluders = list(scene.occluders)

    default_obs = observe(camera, point, occluders)
    custom_obs = observe(
        camera,
        point,
        occluders,
        occ_frac_sample_count=19,
        occ_frac_jitter=0.05,
    )
    assert default_obs["occ_frac"] == occlusion_fraction(
        camera, point, occluders, sample_count=7, jitter=0.05
    )
    assert custom_obs["occ_frac"] == occlusion_fraction(
        camera, point, occluders, sample_count=19, jitter=0.05
    )


def test_no_occluders_occ_frac_is_zero() -> None:
    """Without occluders the difficulty knob is always zero."""
    scene = build_smoke_scene()
    camera = scene.cameras[0]
    point = np.array([0.0, 0.0, 0.5], dtype=np.float64)
    assert occlusion_fraction(camera, point, [], sample_count=27, jitter=0.05) == 0.0
