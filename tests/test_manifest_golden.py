"""Byte-golden safety net for the typed Manifest.

The manifest JSON is the wire contract with multicam-occlusion (its
``triangulate_dlt`` parses this). The typed :class:`Manifest.to_json` MUST keep
the exact serialization structure of the historical
``json.dumps(..., indent=2, allow_nan=False)`` output — same keys, key order,
field presence/omission, and full float precision. Float *values* are compared
with a tolerance so a last-ULP difference in platform-computed extrinsics (macOS
vs Linux BLAS) is not a spurious failure. These fixtures were captured from the
pre-refactor ``build_manifest`` for the smoke, MTMC, and assembly scenes;
regenerate them deliberately (and review the diff) only when the contract is
intentionally changed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from multicam_sim import build_manifest, build_mtmc_scene, build_smoke_scene
from multicam_sim.scene import Scene

_FIXTURES = Path(__file__).parent / "fixtures" / "manifest_golden"


def _assembly_scene() -> Scene:
    examples = Path(__file__).resolve().parents[1] / "examples"
    sys.path.insert(0, str(examples))
    import assembly_station  # noqa: PLC0415

    return assembly_station.build_scene()


def _same_shape(got: object, ref: object, path: str = "") -> None:
    """Assert identical JSON structure — dict key ORDER, list length, field
    presence, and value types. Float values are compared with a tolerance so a
    last-ULP difference in platform-computed extrinsics (macOS vs Linux BLAS) is
    not treated as a contract break; the numeric consumer tolerates it.
    """
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


@pytest.mark.parametrize(
    ("name", "scene_factory"),
    [
        ("smoke", build_smoke_scene),
        ("mtmc", build_mtmc_scene),
        ("assembly", _assembly_scene),
    ],
)
def test_manifest_json_matches_golden(name: str, scene_factory: object) -> None:
    scene = scene_factory()  # type: ignore[operator]
    got = build_manifest(scene).to_json()
    ref = (_FIXTURES / f"{name}.json").read_text()
    # Structure / key order / field presence must match the golden exactly;
    # float values may differ by a platform ULP, which the numeric consumer
    # tolerates. json.loads preserves object key order.
    _same_shape(json.loads(got), json.loads(ref))


def test_optional_fields_omitted_exactly_as_before() -> None:
    """edges present only for skeletoned entities; topology only when declared;
    occ_frac always present."""
    smoke = build_manifest(build_smoke_scene()).to_json()
    mtmc = build_manifest(build_mtmc_scene()).to_json()
    # smoke: no topology, plain object has no edges; occ_frac present
    assert '"topology"' not in smoke
    assert '"edges"' not in smoke
    assert '"occ_frac"' in smoke
    # mtmc: topology present
    assert '"topology"' in mtmc
