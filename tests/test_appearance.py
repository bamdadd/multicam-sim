"""Appearance-descriptor sidecar: a seeded, pixel-free, GT-derived re-id cue.

Covers the contract of :mod:`multicam_sim.appearance`: deterministic sampling
across seeds, unit-norm/shape invariants, edge cases, the monotonic
``separation`` dose-response knob, and — the safety property — that attaching the
sidecar to a :class:`Scene` leaves the byte-golden analytic manifest untouched
(the manifest builder ignores it).

The descriptor deliberately does NOT encode the ground-truth ``entity_id``; it
models OSNet-style confusability, and ``separation`` is the axis a re-id
evaluation sweeps to trace accuracy vs. identity separability.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from multicam_sim import build_manifest, build_smoke_scene
from multicam_sim.appearance import AppearanceTable, EntityAppearance
from multicam_sim.scene import Scene

_ENTITY_IDS = ["entity_a", "entity_b", "entity_c", "entity_d"]


def _is_unit(vec: list[float], *, tol: float = 1e-9) -> bool:
    return math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, abs_tol=tol)


def _mean_pairwise_cosine(table: AppearanceTable) -> float:
    """Mean cosine similarity over all distinct entity pairs (unit descriptors, so
    cosine == dot product)."""
    vecs = [np.asarray(e.descriptor) for e in table.entries]
    sims = [float(vecs[i] @ vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    return sum(sims) / len(sims)


def test_determinism_same_inputs_identical() -> None:
    """Same (entity_ids, dim, separation, seed) -> byte-identical descriptors."""
    a = AppearanceTable.sample(_ENTITY_IDS, dim=512, separation=0.8, seed=7)
    b = AppearanceTable.sample(_ENTITY_IDS, dim=512, separation=0.8, seed=7)
    assert a == b
    assert [e.descriptor for e in a.entries] == [e.descriptor for e in b.entries]


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_seeds_give_distinct_valid_unit_vectors(seed: int) -> None:
    """Each seed yields valid unit-norm descriptors; different seeds differ."""
    table = AppearanceTable.sample(_ENTITY_IDS, dim=256, separation=0.9, seed=seed)
    assert len(table.entries) == len(_ENTITY_IDS)
    for entry in table.entries:
        assert len(entry.descriptor) == 256
        assert _is_unit(entry.descriptor)
        assert all(math.isfinite(x) for x in entry.descriptor)

    other = AppearanceTable.sample(_ENTITY_IDS, dim=256, separation=0.9, seed=seed + 1)
    assert table.entries[0].descriptor != other.entries[0].descriptor


def test_descriptor_does_not_encode_entity_id() -> None:
    """Renaming entities (same seed/dim/separation) keeps the descriptor vectors
    identical: the vector is a function of draw order, not of the id string, so it
    cannot be inverted to the GT id."""
    a = AppearanceTable.sample(["x0", "x1", "x2"], dim=128, separation=0.7, seed=3)
    b = AppearanceTable.sample(["y0", "y1", "y2"], dim=128, separation=0.7, seed=3)
    assert [e.descriptor for e in a.entries] == [e.descriptor for e in b.entries]


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_separation_knob_monotonic(seed: int) -> None:
    """Mean pairwise inter-entity cosine similarity increases monotonically as
    separation falls 1.0 -> 0.0 (the re-id dose-response axis). Coarse steps clear
    the ~1/sqrt(dim) noise floor comfortably."""
    separations = [1.0, 0.75, 0.5, 0.25, 0.0]
    sims = [
        _mean_pairwise_cosine(AppearanceTable.sample(_ENTITY_IDS, dim=512, separation=s, seed=seed))
        for s in separations
    ]
    # separations descend, so similarities must strictly ascend.
    for lower, higher in zip(sims, sims[1:], strict=False):
        assert higher > lower, f"non-monotone at seed {seed}: {sims}"

    # Endpoints: near-orthogonal (perfectly separable) at 1.0, collapsed at 0.0.
    assert sims[0] < 0.2
    assert math.isclose(sims[-1], 1.0, abs_tol=1e-6)


def test_unit_norm_and_shape_invariants() -> None:
    """Every descriptor is unit-norm and exactly `dim` long, across separations."""
    for separation in (1.0, 0.5, 0.0):
        table = AppearanceTable.sample(_ENTITY_IDS, dim=64, separation=separation, seed=11)
        assert table.dim == 64
        for entry in table.entries:
            assert len(entry.descriptor) == 64
            assert _is_unit(entry.descriptor)


def test_empty_entity_ids() -> None:
    """No entities -> empty table, no crash (mean never computed)."""
    table = AppearanceTable.sample([], dim=512, separation=0.5, seed=1)
    assert table.entries == []
    assert table.dim == 512


def test_single_entity() -> None:
    """One entity -> its unchanged unit anchor, no mean-blend, no pairwise math."""
    table = AppearanceTable.sample(["solo"], dim=128, separation=0.3, seed=2)
    assert len(table.entries) == 1
    assert _is_unit(table.entries[0].descriptor)
    # Separation is irrelevant with one entity: any separation gives the same
    # anchor for a fixed seed.
    other = AppearanceTable.sample(["solo"], dim=128, separation=0.9, seed=2)
    assert table.entries[0].descriptor == other.entries[0].descriptor


def test_default_dim_is_512() -> None:
    table = AppearanceTable.sample(_ENTITY_IDS, separation=0.8, seed=5)
    assert table.dim == 512
    assert all(len(e.descriptor) == 512 for e in table.entries)


def test_validation_rejects_out_of_range_separation() -> None:
    with pytest.raises(ValueError):
        AppearanceTable(dim=8, separation=1.5, seed=0, entries=[])
    with pytest.raises(ValueError):
        AppearanceTable(dim=0, separation=0.5, seed=0, entries=[])
    with pytest.raises(ValueError):
        EntityAppearance(entity_id="e", descriptor=[])


def test_roundtrip_json() -> None:
    """Table survives a JSON round-trip unchanged (the sidecar payload)."""
    table = AppearanceTable.sample(_ENTITY_IDS, dim=32, separation=0.6, seed=9)
    restored = AppearanceTable.model_validate_json(table.to_json())
    assert restored == table


def test_observe_sigma_zero_is_exact_anchor() -> None:
    """sigma == 0 is a bitwise no-op: observe() returns the stored anchor verbatim."""
    table = AppearanceTable.sample(_ENTITY_IDS, dim=256, separation=0.8, seed=7)
    for entry in table.entries:
        obs = table.observe(entry.entity_id, obs_seed=99, sigma=0.0)
        # Pinned to the ACTUAL stored anchor, not a recomputed sample().
        assert obs == entry.descriptor


def test_observe_deterministic_and_seed_sensitive() -> None:
    """Same (entity_id, obs_seed, sigma) -> identical vector; different obs_seed ->
    a different but still valid unit observation of the same anchor."""
    table = AppearanceTable.sample(_ENTITY_IDS, dim=256, separation=0.8, seed=7)
    a = table.observe("entity_a", obs_seed=3, sigma=0.5)
    b = table.observe("entity_a", obs_seed=3, sigma=0.5)
    assert a == b

    c = table.observe("entity_a", obs_seed=4, sigma=0.5)
    assert c != a
    assert _is_unit(c)
    assert all(math.isfinite(x) for x in c)


def test_observe_unit_norm_finite_and_dim() -> None:
    """Every noisy observation is unit-norm, finite, and exactly `dim` long."""
    for dim in (32, 256):
        table = AppearanceTable.sample(_ENTITY_IDS, dim=dim, separation=0.7, seed=5)
        for sigma in (0.0, 0.25, 1.0, 2.0):
            for entry in table.entries:
                obs = table.observe(entry.entity_id, obs_seed=11, sigma=sigma)
                assert len(obs) == dim
                assert _is_unit(obs)
                assert all(math.isfinite(x) for x in obs)


def test_observe_noise_monotonic_drift_from_anchor() -> None:
    """Mean cosine(observation, anchor) DECREASES as sigma increases: more noise
    means observations drift further from the anchor. Averaged over >=3 seeds and
    >=3 sigma values so the assertion clears sampling jitter."""
    sigmas = [0.25, 0.5, 1.0, 2.0]
    seeds = [0, 1, 2, 3, 4]
    table = AppearanceTable.sample(_ENTITY_IDS, dim=512, separation=0.8, seed=7)
    anchors = {e.entity_id: np.asarray(e.descriptor) for e in table.entries}

    mean_cos = []
    for sigma in sigmas:
        cosines = [
            float(np.asarray(table.observe(eid, obs_seed=s, sigma=sigma)) @ anchors[eid])
            for eid in anchors
            for s in seeds
        ]
        mean_cos.append(sum(cosines) / len(cosines))

    for lower_sigma, higher_sigma in zip(mean_cos, mean_cos[1:], strict=False):
        assert higher_sigma < lower_sigma, f"non-monotone drift: {mean_cos}"


def test_observe_rejects_negative_sigma_and_unknown_id() -> None:
    table = AppearanceTable.sample(_ENTITY_IDS, dim=32, separation=0.8, seed=7)
    with pytest.raises(ValueError):
        table.observe("entity_a", obs_seed=0, sigma=-0.1)
    with pytest.raises(ValueError):
        table.observe("nope", obs_seed=0, sigma=0.5)


def _scene_with_appearance(*, attach: bool) -> Scene:
    scene = build_smoke_scene()
    if not attach:
        return scene
    ids = [e.id for e in scene.entities]
    table = AppearanceTable.sample(ids, dim=512, separation=0.7, seed=13)
    return scene.model_copy(update={"appearance": table})


def test_manifest_byte_identical_with_and_without_sidecar() -> None:
    """Attaching the appearance sidecar changes NOTHING in the analytic manifest:
    the serialized bytes with and without it are exactly equal (manifest builder
    ignores the sidecar; byte-golden default preserved)."""
    with_side = build_manifest(_scene_with_appearance(attach=True)).to_json().encode()
    without = build_manifest(_scene_with_appearance(attach=False)).to_json().encode()
    assert with_side == without


def test_manifest_excludes_appearance_sidecar() -> None:
    """The manifest never contains appearance GT, even when the scene carries it."""
    manifest_json = build_manifest(_scene_with_appearance(attach=True)).to_json()
    assert "appearance" not in manifest_json
    assert "descriptor" not in manifest_json
