"""Seeded per-camera sensor dropout: byte-reproducible schedule, blank dropped
observations, and byte-identical output when off."""

from __future__ import annotations

import pytest

from multicam_sim.dropout import SensorDropout, dropped_frames
from multicam_sim.manifest import build_manifest
from multicam_sim.noise import NoiseModel, PixelNoise
from multicam_sim.smoke import build_smoke_scene

_NUM_FRAMES = 24


def _drops(config: SensorDropout, cam: int) -> tuple[int, ...]:
    return dropped_frames(config, cam, _NUM_FRAMES)


# --- schedule: reproducible, seed-sensitive, per-camera independent ------------- #


def test_schedule_is_byte_reproducible_for_a_fixed_seed() -> None:
    config = SensorDropout(seed=7, drop_prob=0.3)
    for cam in range(3):
        first = _drops(config, cam)
        assert first == _drops(config, cam)  # repeated calls identical
        assert first == dropped_frames(SensorDropout(seed=7, drop_prob=0.3), cam, _NUM_FRAMES)


def test_schedule_differs_across_seeds() -> None:
    a = SensorDropout(seed=1, drop_prob=0.3)
    b = SensorDropout(seed=2, drop_prob=0.3)
    per_cam = [(_drops(a, cam), _drops(b, cam)) for cam in range(3)]
    assert any(x != y for x, y in per_cam), "a different seed must change some schedule"


def test_schedule_is_per_camera_independent() -> None:
    config = SensorDropout(seed=7, drop_prob=0.3)
    schedules = [_drops(config, cam) for cam in range(3)]
    assert any(schedules[i] != schedules[j] for i in range(3) for j in range(i + 1, 3))


def test_schedule_is_sorted_and_in_range() -> None:
    config = SensorDropout(seed=3, drop_prob=0.5)
    for cam in range(3):
        drops = _drops(config, cam)
        assert list(drops) == sorted(drops)
        assert all(0 <= f < _NUM_FRAMES for f in drops)
        assert len(set(drops)) == len(drops)


def test_prob_one_drops_every_frame_and_zero_drops_none() -> None:
    assert _drops(SensorDropout(seed=5, drop_prob=1.0), 0) == tuple(range(_NUM_FRAMES))
    assert _drops(SensorDropout(seed=5, drop_prob=0.0), 0) == ()


def test_empirical_drop_rate_is_near_prob() -> None:
    config = SensorDropout(seed=11, drop_prob=0.25)
    total = sum(len(_drops(config, cam)) for cam in range(3))
    rate = total / (3 * _NUM_FRAMES)
    assert 0.1 < rate < 0.4  # loose: ~0.25 over a small sample


# --- config validation --------------------------------------------------------- #


@pytest.mark.parametrize("prob", [-0.1, 1.1, 2.0])
def test_rejects_out_of_range_prob(prob: float) -> None:
    with pytest.raises(ValueError, match="drop_prob"):
        SensorDropout(drop_prob=prob)


def test_is_active_flag() -> None:
    assert not SensorDropout().is_active
    assert not SensorDropout(drop_prob=0.0, seed=9).is_active
    assert SensorDropout(drop_prob=0.01).is_active


# --- manifest wiring ------------------------------------------------------------ #


def test_off_is_byte_identical_to_no_dropout() -> None:
    scene = build_smoke_scene()
    base = build_manifest(scene).to_json()
    assert build_manifest(scene, dropout=None).to_json() == base
    assert build_manifest(scene, dropout=SensorDropout(drop_prob=0.0)).to_json() == base


def test_dropped_observations_are_blank_and_scheduled() -> None:
    scene = build_smoke_scene()
    config = SensorDropout(seed=7, drop_prob=1.0)  # drop everything
    manifest = build_manifest(scene, dropout=config)

    for cam in manifest.cameras:
        assert cam.dropped_frames == list(range(scene.num_frames))

    for entity in manifest.entities:
        for frame in entity.frames:
            for point in frame.points.values():
                for obs in point.per_cam:
                    assert obs.dropped is True
                    assert obs.in_view is False
                    assert obs.visible is False
                    assert obs.occ_frac is None
                    assert obs.uv == [0.0, 0.0]


def test_partial_dropout_matches_schedule() -> None:
    scene = build_smoke_scene()
    config = SensorDropout(seed=4, drop_prob=0.4)
    manifest = build_manifest(scene, dropout=config)

    for cam in manifest.cameras:
        expected = list(dropped_frames(config, cam.id, scene.num_frames))
        recorded = cam.dropped_frames or []
        assert recorded == expected
        # every scheduled (cam, frame) is blank; unscheduled frames are untouched.
        for entity in manifest.entities:
            for frame in entity.frames:
                obs = next(o for p in frame.points.values() for o in p.per_cam if o.cam == cam.id)
                if frame.frame in expected:
                    assert obs.dropped is True and obs.visible is False
                else:
                    assert obs.dropped is None


def test_dropout_is_independent_of_pixel_noise_stream() -> None:
    """A dropped point still consumes its noise draw, so non-dropped points keep
    exactly the pixel-noise-only uv (the two seeded streams never couple)."""
    scene = build_smoke_scene()
    noise = NoiseModel(seed=2, pixel=PixelNoise(sigma_px=1.5))
    noise_only = build_manifest(scene, noise=noise)
    with_both = build_manifest(scene, noise=noise, dropout=SensorDropout(seed=9, drop_prob=0.3))

    drops = {cam.id: set(cam.dropped_frames or []) for cam in with_both.cameras}
    for ent_a, ent_b in zip(noise_only.entities, with_both.entities, strict=True):
        for fa, fb in zip(ent_a.frames, ent_b.frames, strict=True):
            for name, pa in fa.points.items():
                pb = fb.points[name]
                for oa, ob in zip(pa.per_cam, pb.per_cam, strict=True):
                    if fb.frame not in drops[ob.cam]:
                        assert ob.uv == oa.uv  # untouched by dropout
