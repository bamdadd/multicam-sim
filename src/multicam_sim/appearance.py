"""Appearance-descriptor sidecar: a seeded, pixel-free, GT-derived re-id cue.

This channel models the *appearance* half of cross-camera re-identification. When
geometry alone cannot disambiguate two subjects — the collinear / ambiguous case
where their epipolar or triangulated positions coincide — a fuser needs a second,
appearance-based signal to tell them apart. This module supplies that signal
*without any pixels*: each entity is assigned a deterministic unit-norm descriptor
vector, and how easily two entities can be told apart is set by a single knob.

**Honest note — what this does and does NOT encode.** The descriptor does NOT
encode, and cannot be inverted to recover, the ground-truth ``entity_id``. It is
NOT a perfect label handed to the fuser. It models the *confusability* of a real
learned appearance embedding (OSNet-style): distinct identities land at distinct
but not perfectly separated points on the unit sphere, and a downstream matcher
must still cluster them under noise. The :attr:`AppearanceTable.separation` knob
is the dose-response axis for re-id evaluation: it monotonically controls the
inter-entity cosine similarity, from ``1.0`` (near-orthogonal descriptors,
perfectly separable — the easy case) down to ``0.0`` (all descriptors collapse to
a shared mean, indistinguishable — the impossible case). Sweeping it traces how
re-id accuracy degrades as identities become harder to separate.

**Per-observation noise.** :meth:`AppearanceTable.observe` draws a noisy unit
observation around an entity's anchor — one camera's *look* at the entity — via an
isotropic Gaussian jitter of stddev ``sigma``. With a single clean anchor per
entity a separation sweep is a degenerate cliff (perfect until total collapse,
zero variance); an intra-entity ``sigma`` gives each observation real spread, so
the sweep becomes a graded dose-response. ``separation`` is the inter-entity gap,
``sigma`` the intra-entity spread; ``sigma == 0.0`` reduces exactly to the clean
anchor, so the anchor-only behaviour is the special case.

Like :mod:`multicam_sim.activity` and :mod:`multicam_sim.possession`, this is a
pure typed-model sidecar. The table rides in a JSON sidecar and attaches to
:class:`~multicam_sim.scene.Scene` via an optional field that the manifest builder
IGNORES, so the byte-golden analytic manifest is unchanged when appearance GT is
absent (and even when it is present).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

_DEFAULT_DIM = 512


class EntityAppearance(BaseModel):
    """One entity's appearance descriptor: a unit-norm vector keyed by id.

    ``descriptor`` is L2-normalized (unit length). It is a synthetic stand-in for
    a learned re-id embedding; it does not encode the ``entity_id``.
    """

    model_config = ConfigDict(frozen=True)

    entity_id: str
    descriptor: list[float]

    @field_validator("descriptor")
    @classmethod
    def _non_empty(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("descriptor must be non-empty")
        return value


class AppearanceTable(BaseModel):
    """A collection of per-entity appearance descriptors of a fixed ``dim``.

    Built deterministically from ``(entity_ids, dim, separation, seed)`` via
    :meth:`sample`. ``separation`` is the dose-response knob (see the module
    docstring): ``1.0`` gives near-orthogonal, easily separable descriptors,
    ``0.0`` collapses every descriptor onto a shared mean (indistinguishable).
    The mean pairwise inter-entity cosine similarity increases monotonically as
    ``separation`` falls from ``1.0`` to ``0.0``.
    """

    model_config = ConfigDict(frozen=True)

    dim: int = _DEFAULT_DIM
    separation: float
    seed: int
    entries: list[EntityAppearance] = []

    @field_validator("dim")
    @classmethod
    def _positive_dim(cls, value: int) -> int:
        if value < 1:
            raise ValueError("dim must be >= 1")
        return value

    @field_validator("separation")
    @classmethod
    def _separation_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("separation must be in [0.0, 1.0]")
        return value

    @classmethod
    def sample(
        cls,
        entity_ids: list[str],
        *,
        dim: int = _DEFAULT_DIM,
        separation: float,
        seed: int,
    ) -> AppearanceTable:
        """Deterministically draw one unit-norm descriptor per entity.

        Draw order is fixed: a single ``rng.standard_normal((N, dim))`` call in
        the given ``entity_ids`` order. Each raw draw is L2-normalized to a unit
        anchor, then blended toward the shared mean of all unit anchors by
        ``(1 - separation)`` and renormalized. As ``separation`` falls from
        ``1.0`` to ``0.0`` every descriptor moves toward the common mean, so the
        inter-entity cosine similarity rises monotonically. ``separation == 0.0``
        collapses all descriptors onto the shared mean (cosine ``1.0``).

        Edge cases: an empty ``entity_ids`` yields an empty table; a single
        entity yields its (unchanged, unit-norm) anchor with no mean-blend.
        """
        rng = np.random.default_rng(seed)
        n = len(entity_ids)
        if n == 0:
            return cls(dim=dim, separation=separation, seed=seed, entries=[])

        raw = rng.standard_normal((n, dim))
        # Unit anchors first, so the shared mean is not dominated by the largest
        # raw draw (keeps the monotonicity closed form clean).
        anchors = raw / np.linalg.norm(raw, axis=1, keepdims=True)

        if n == 1:
            vectors = anchors
        else:
            mean = anchors.mean(axis=0, keepdims=True)
            blended = separation * anchors + (1.0 - separation) * mean
            norms = np.linalg.norm(blended, axis=1, keepdims=True)
            # separation == 0 collapses every row exactly to `mean`; guard the
            # (astronomically unlikely) zero-mean degenerate by falling back to
            # the anchor so the descriptor stays unit-norm and finite.
            safe = norms.squeeze(axis=1) > 0.0
            vectors = np.where(norms > 0.0, blended / np.where(norms > 0.0, norms, 1.0), anchors)
            if not safe.all():
                vectors[~safe] = anchors[~safe]

        entries = [
            EntityAppearance(entity_id=entity_id, descriptor=[float(x) for x in vectors[i]])
            for i, entity_id in enumerate(entity_ids)
        ]
        return cls(dim=dim, separation=separation, seed=seed, entries=entries)

    def observe(self, entity_id: str, *, obs_seed: int, sigma: float) -> list[float]:
        """Draw one noisy, unit-norm *observation* of an entity around its anchor.

        Where :meth:`sample` fixes one clean anchor per entity, this models a
        single camera's *observation* of that entity: an isotropic-Gaussian jitter
        around the anchor, re-projected to the unit sphere::

            observed = normalize(anchor(entity_id) + sigma * N(0, I_dim))

        The noise is drawn from ``numpy.random.default_rng(obs_seed)`` with a fixed
        draw order (a single ``standard_normal(dim)`` call), so a given
        ``(entity_id, obs_seed, sigma)`` is fully deterministic; different
        ``obs_seed`` values give different valid unit observations of the *same*
        anchor.

        ``sigma`` (>= 0) is the observation-noise stddev: the intra-entity spread,
        modelling sensor / viewpoint variation across cameras. It is orthogonal to
        the table's :attr:`separation`, which is the inter-entity gap. Together they
        set the confusability of a single observation: ``separation`` sets how far
        apart two entities' anchors sit, ``sigma`` sets how far a single look drifts
        from its own anchor, and re-id gets hard once the intra-entity spread rivals
        the inter-entity gap. ``sigma == 0.0`` reduces EXACTLY to the clean anchor
        (a no-op), so the pre-noise behaviour is the special case. Like the anchor,
        an observation does NOT encode or leak the ground-truth ``entity_id``.

        Raises ``ValueError`` if ``sigma`` is negative or ``entity_id`` is unknown.
        """
        if sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        anchor = self._anchor(entity_id)
        # sigma == 0 is an exact no-op: return the stored anchor verbatim. (Adding
        # zero noise then renormalising would perturb the anchor's float-rounded
        # last bits, so short-circuit to keep the reduction bitwise-exact.)
        if sigma == 0.0:
            return list(anchor)
        rng = np.random.default_rng(obs_seed)
        vec = np.asarray(anchor, dtype=np.float64) + sigma * rng.standard_normal(self.dim)
        norm = float(np.linalg.norm(vec))
        # Guard the astronomically unlikely zero vector by falling back to the
        # anchor so the observation stays unit-norm and finite.
        if norm <= 0.0:
            return list(anchor)
        return [float(x) for x in vec / norm]

    def _anchor(self, entity_id: str) -> list[float]:
        for entry in self.entries:
            if entry.entity_id == entity_id:
                return entry.descriptor
        raise ValueError(f"unknown entity_id: {entity_id!r}")

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to a JSON string (the ``appearance.json`` sidecar payload)."""
        return self.model_dump_json(indent=indent)


def write_appearance_json(table: AppearanceTable, path: str | Path) -> dict[str, Any]:
    """Write an appearance table to ``path`` as JSON.

    Returns the dumped dict so a caller can assert on it without re-reading.
    """
    data: dict[str, Any] = table.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2))
    return data
