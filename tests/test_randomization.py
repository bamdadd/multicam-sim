"""Seeded domain randomization for the Scene builder (#41).

A typed ``RandomizationSpec`` (background / light / N distractors, each knob a
closed ``(min, max)`` interval) samples deterministically under a seed; the
``SceneBuilder.randomize`` entry point applies one sample — background and
light land on the :class:`Scene`, distractors go through the existing
``.distractor`` path, and a provenance sidecar (spec + seed) rides on the
scene. Everything is additive and off by default, so an un-randomized scene is
byte-identical to before.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from multicam_sim import (
    BackgroundSpec,
    DistractorSpec,
    LightSpec,
    RandomizationSpec,
    build_manifest,
    build_smoke_scene,
)
from multicam_sim.dsl import CameraRig, SceneBuilder
from multicam_sim.dsl import Path as MotionPath
from multicam_sim.randomization import Light
from multicam_sim.scene import Scene

_GOLDEN = Path(__file__).parent / "fixtures" / "manifest_golden"


def _spec(**overrides: object) -> RandomizationSpec:
    """A spec with all three knobs active (defaults overridable)."""
    knobs: dict[str, object] = {
        "background": BackgroundSpec(),
        "light": LightSpec(),
        "distractors": DistractorSpec(count=(2, 2)),
    }
    knobs.update(overrides)
    return RandomizationSpec(**knobs)  # type: ignore[arg-type]


def _builder() -> SceneBuilder:
    return (
        SceneBuilder(fps=10.0, num_frames=10)
        .cameras(
            CameraRig.ring(
                n=2,
                radius=5.0,
                height=1.0,
                look_at=(0.0, 0.0, 0.0),
                focal=800.0,
                width=640,
                height_px=480,
            )
        )
        .entity("obj", MotionPath.linear((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)))
    )


def _same_shape(got: object, ref: object, path: str = "") -> None:
    """Assert identical JSON structure (mirrors ``test_manifest_golden``)."""
    assert type(got) is type(ref), f"type drift at {path or '<root>'}: {type(got)} != {type(ref)}"
    if isinstance(ref, dict):
        assert isinstance(got, dict)
        assert list(got) == list(ref), f"key set/order drift at {path or '<root>'}"
        for k in ref:
            _same_shape(got[k], ref[k], f"{path}.{k}")
    elif isinstance(ref, list):
        assert isinstance(got, list)
        assert len(got) == len(ref), f"length drift at {path or '<root>'}: {len(got)} != {len(ref)}"
        for i, (g, r) in enumerate(zip(got, ref, strict=True)):
            _same_shape(g, r, f"{path}[{i}]")
    elif isinstance(ref, float):
        assert isinstance(got, float)
        assert math.isclose(got, ref, rel_tol=1e-9, abs_tol=1e-12), (
            f"float drift at {path}: {got} != {ref}"
        )
    else:
        assert got == ref, f"value drift at {path}: {got!r} != {ref!r}"


def test_sample_is_byte_identical_for_same_seed() -> None:
    spec = _spec()
    first = spec.sample(42)
    second = spec.sample(42)
    assert first.model_dump_json() == second.model_dump_json()


def test_different_seeds_produce_different_samples() -> None:
    spec = _spec()
    assert spec.sample(42).model_dump_json() != spec.sample(7).model_dump_json()


def test_empty_spec_samples_to_empty_sample() -> None:
    sample = RandomizationSpec().sample(0)
    assert sample.background is None
    assert sample.light is None
    assert sample.distractor_positions == []


@pytest.mark.parametrize(
    "make",
    [
        lambda: BackgroundSpec(rgb_min=(0.5, 0.0, 0.0), rgb_max=(0.4, 1.0, 1.0)),
        lambda: BackgroundSpec(rgb_min=(0.0, 0.0, 1.2)),
        lambda: LightSpec(intensity=(4.0, 2.0)),
        lambda: LightSpec(intensity=(-1.0, 2.0)),
        lambda: LightSpec(azimuth_deg=(180.0, 90.0)),
        lambda: LightSpec(elevation_deg=(80.0, 20.0)),
        lambda: LightSpec(elevation_deg=(0.0, 120.0)),
        lambda: DistractorSpec(count=(3, 1)),
        lambda: DistractorSpec(count=(-1, 2)),
        lambda: DistractorSpec(x=(1.0, -1.0)),
    ],
)
def test_inverted_or_invalid_interval_rejected(make: object) -> None:
    with pytest.raises(ValidationError):
        make()  # type: ignore[operator]


def test_sampled_values_lie_within_the_intervals() -> None:
    spec = _spec(distractors=DistractorSpec(count=(2, 4)))
    for seed in range(10):
        sample = spec.sample(seed)
        assert sample.background is not None
        assert all(0.0 <= c <= 0.2 for c in sample.background.rgb)
        assert sample.light is not None
        assert 2.0 <= sample.light.intensity <= 4.0
        assert 0.0 <= sample.light.azimuth_deg <= 360.0
        assert 30.0 <= sample.light.elevation_deg <= 90.0
        assert 2 <= len(sample.distractor_positions) <= 4
        for x, y, z in sample.distractor_positions:
            assert -2.0 <= x <= 2.0
            assert -2.0 <= y <= 2.0
            assert 0.0 <= z <= 1.0


def test_light_direction_is_unit_and_points_at_the_scene() -> None:
    light = Light(intensity=3.0, azimuth_deg=53.25, elevation_deg=41.8)
    direction = light.direction()
    assert math.isclose(math.sqrt(sum(c * c for c in direction)), 1.0, rel_tol=1e-12)
    # a light above the horizon shines downward
    assert direction[2] < 0.0
    # azimuth 0 / elevation 0 shines along -X
    assert Light(intensity=1.0, azimuth_deg=0.0, elevation_deg=0.0).direction() == pytest.approx(
        (-1.0, 0.0, 0.0)
    )


def test_unrandomized_scene_is_byte_identical_to_before() -> None:
    """No ``randomize`` call: no new Scene field is set, two independent builds
    are byte-identical, and the golden smoke manifest is structurally unchanged
    (the new fields never reach the manifest)."""
    scene = _builder().build()
    assert scene.background is None
    assert scene.light is None
    assert scene.randomization is None

    again = _builder().build()
    assert scene.model_dump_json() == again.model_dump_json()
    assert build_manifest(scene).to_json() == build_manifest(again).to_json()

    golden = json.loads((_GOLDEN / "smoke.json").read_text())
    _same_shape(json.loads(build_manifest(build_smoke_scene()).to_json()), golden)


def test_distractor_count_goes_through_the_existing_distractor_path() -> None:
    spec = RandomizationSpec(distractors=DistractorSpec(count=(3, 3)))
    scene = _builder().randomize(spec, seed=5).build()

    ids = [entity.id for entity in scene.entities]
    assert ids == ["obj", "rand_distractor_0", "rand_distractor_1", "rand_distractor_2"]

    # distractors are static: every frame holds the same sampled position
    sample = spec.sample(5)
    for entity, position in zip(scene.entities[1:], sample.distractor_positions, strict=True):
        for entity_frame in entity.frames:
            assert entity_frame.points["center"] == pytest.approx(list(position))

    # the existing distractor path means they appear as manifest entities
    manifest = build_manifest(scene)
    assert [entity.id for entity in manifest.entities] == ids


def test_zero_distractors_is_allowed() -> None:
    spec = RandomizationSpec(distractors=DistractorSpec(count=(0, 0)))
    scene = _builder().randomize(spec, seed=5).build()
    assert [entity.id for entity in scene.entities] == ["obj"]
    assert scene.randomization is not None


def test_randomized_scene_is_deterministic_per_seed() -> None:
    spec = _spec()
    first = _builder().randomize(spec, seed=11).build()
    second = _builder().randomize(spec, seed=11).build()
    other = _builder().randomize(spec, seed=12).build()
    assert first.model_dump_json() == second.model_dump_json()
    assert first.model_dump_json() != other.model_dump_json()


def test_background_and_light_are_applied_to_the_scene() -> None:
    spec = _spec()
    scene = _builder().randomize(spec, seed=3).build()
    sample = spec.sample(3)
    assert scene.background == sample.background
    assert scene.light == sample.light


def test_provenance_sidecar_roundtrips_and_reproduces_the_scene() -> None:
    spec = _spec()
    scene = _builder().randomize(spec, seed=11).build()
    record = scene.randomization
    assert record is not None
    assert record.seed == 11
    assert record.spec == spec

    # the sidecar survives JSON round-trip on the scene
    restored = Scene.model_validate_json(scene.model_dump_json())
    assert restored.randomization == record

    # and the recorded spec + seed regenerate the exact applied sample
    resampled = record.spec.sample(record.seed)
    assert resampled.background == scene.background
    assert resampled.light == scene.light
