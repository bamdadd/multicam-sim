"""Seeded noise / calibration-drift knobs (issue #30).

Guards: all-zero knobs leave the manifest byte-identical to the noiseless
output (the assumed-calibration block is absent); a fixed seed is reproducible;
a nonzero pixel sigma perturbs the observed ``uv`` with the expected empirical
std; drift records a separate assumed calibration without touching ground truth;
the typed configs validate their inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam_sim import (
    CalibrationDrift,
    Camera,
    Intrinsics,
    NoiseModel,
    PixelNoise,
    Scene,
    build_manifest,
    build_smoke_scene,
)
from multicam_sim.geometry import project_point, projection_matrix, rotation_from_axis_angle
from multicam_sim.manifest import observe


def _cam_and_point() -> tuple[Camera, np.ndarray]:
    """A smoke-scene camera and a point that projects in front and in-frame."""
    scene = build_smoke_scene()
    cam = scene.cameras[0]
    point = np.array([0.0, 0.0, 0.5], dtype=np.float64)
    uv, w = cam.project(point)
    assert w > 0.0 and cam.in_image(uv)  # precondition: a clean, noisable pixel
    return cam, point


# --------------------------------------------------------------------------- #
# (1) byte-identity at zero knobs
# --------------------------------------------------------------------------- #


def test_zero_knobs_are_byte_identical_to_no_noise() -> None:
    """An all-zero NoiseModel reproduces the noiseless manifest byte-for-byte,
    and records no assumed-calibration block."""
    scene = build_smoke_scene()
    baseline = build_manifest(scene).to_json()
    zeroed = build_manifest(scene, noise=NoiseModel()).to_json()
    assert zeroed == baseline
    assert '"assumed"' not in zeroed


def test_camera_entries_omit_assumed_when_drift_off() -> None:
    scene = build_smoke_scene()
    manifest = build_manifest(scene, noise=NoiseModel(pixel=PixelNoise(sigma_px=1.0)))
    # pixel noise on, drift off => no assumed block anywhere.
    assert all(cam.assumed is None for cam in manifest.cameras)


# --------------------------------------------------------------------------- #
# (2) pixel noise: reproducible + expected empirical std
# --------------------------------------------------------------------------- #


def test_pixel_noise_is_reproducible_for_a_fixed_seed() -> None:
    scene = build_smoke_scene()
    noise = NoiseModel(seed=1234, pixel=PixelNoise(sigma_px=2.0))
    first = build_manifest(scene, noise=noise).to_json()
    second = build_manifest(scene, noise=noise).to_json()
    assert first == second


def test_pixel_noise_differs_across_seeds() -> None:
    scene = build_smoke_scene()
    a = build_manifest(scene, noise=NoiseModel(seed=1, pixel=PixelNoise(sigma_px=2.0))).to_json()
    b = build_manifest(scene, noise=NoiseModel(seed=2, pixel=PixelNoise(sigma_px=2.0))).to_json()
    assert a != b


def test_pixel_noise_has_expected_empirical_std() -> None:
    """A nonzero pixel sigma perturbs uv with std ~= sigma.

    We draw M = 2 * N scalar offsets (u and v over N observations) of the same
    point through one camera, sharing a single rng. The sample-std estimator of
    M iid N(0, sigma) samples has std ~= sigma / sqrt(2M); for M = 40000 that is
    ~0.35% of sigma, so a 5% relative tolerance is a >14-sigma margin and never
    flakes. The mean is also checked to be ~0 (bias-free additive noise).
    """
    cam, point = _cam_and_point()
    sigma = 2.0
    true_uv, _ = cam.project(point)
    true_u, true_v = float(true_uv[0]), float(true_uv[1])

    rng = np.random.default_rng(0)
    pixel = PixelNoise(sigma_px=sigma)
    n = 20000
    deltas: list[float] = []
    for _ in range(n):
        obs = observe(cam, point, [], pixel_noise=pixel, rng=rng)
        deltas.append(obs.uv[0] - true_u)
        deltas.append(obs.uv[1] - true_v)

    arr = np.asarray(deltas, dtype=np.float64)
    assert abs(float(arr.std())) == pytest.approx(sigma, rel=0.05)
    assert abs(float(arr.mean())) < 0.05 * sigma


def test_pixel_noise_leaves_ground_truth_and_flags_exact() -> None:
    """Only the recorded uv moves; xyz_gt, in_view and visible stay truthful."""
    scene = build_smoke_scene()
    clean = build_manifest(scene)
    noisy = build_manifest(scene, noise=NoiseModel(seed=7, pixel=PixelNoise(sigma_px=3.0)))
    for c_ent, n_ent in zip(clean.entities, noisy.entities, strict=True):
        for c_fr, n_fr in zip(c_ent.frames, n_ent.frames, strict=True):
            for name, c_pt in c_fr.points.items():
                n_pt = n_fr.points[name]
                assert n_pt.xyz_gt == c_pt.xyz_gt
                for c_obs, n_obs in zip(c_pt.per_cam, n_pt.per_cam, strict=True):
                    assert n_obs.in_view == c_obs.in_view
                    assert n_obs.visible == c_obs.visible
                    assert n_obs.occ_frac == c_obs.occ_frac


# --------------------------------------------------------------------------- #
# (3) calibration drift: records a separate assumed calibration, GT exact
# --------------------------------------------------------------------------- #


def _active_drift() -> CalibrationDrift:
    return CalibrationDrift(
        rotation_sigma_deg=0.5,
        translation_sigma=0.01,
        focal_sigma_px=3.0,
        principal_point_sigma_px=2.0,
    )


def test_drift_records_assumed_calibration_leaving_truth_exact() -> None:
    scene = build_smoke_scene()
    clean = build_manifest(scene)
    drifted = build_manifest(scene, noise=NoiseModel(seed=42, drift=_active_drift()))
    for c_cam, d_cam in zip(clean.cameras, drifted.cameras, strict=True):
        # ground-truth K,R,t unchanged.
        assert d_cam.K == c_cam.K
        assert d_cam.R == c_cam.R
        assert d_cam.t == c_cam.t
        # assumed present and genuinely different from the truth.
        assert d_cam.assumed is not None
        assert d_cam.assumed.K != c_cam.K
        assert d_cam.assumed.R != c_cam.R
        assert d_cam.assumed.t != c_cam.t


def test_drift_is_reproducible_and_seed_sensitive() -> None:
    scene = build_smoke_scene()
    drift = _active_drift()
    a = build_manifest(scene, noise=NoiseModel(seed=5, drift=drift)).to_json()
    a2 = build_manifest(scene, noise=NoiseModel(seed=5, drift=drift)).to_json()
    b = build_manifest(scene, noise=NoiseModel(seed=6, drift=drift)).to_json()
    assert a == a2
    assert a != b


def test_assumed_calibration_serialises_last_and_only_when_present() -> None:
    scene = build_smoke_scene()
    drifted = build_manifest(scene, noise=NoiseModel(seed=1, drift=_active_drift()))
    dumped = drifted.model_dump(exclude_none=True)
    cam0 = dumped["cameras"][0]
    assert list(cam0)[-1] == "assumed"  # additive: appended after convention


def test_drift_assumed_rotation_stays_orthonormal() -> None:
    """A drifted R is still a rotation (orthonormal), so the assumed camera is a
    valid pinhole calibration, not a sheared matrix."""
    scene = build_smoke_scene()
    drifted = build_manifest(scene, noise=NoiseModel(seed=3, drift=_active_drift()))
    for cam in drifted.cameras:
        assert cam.assumed is not None
        r = np.asarray(cam.assumed.R, dtype=np.float64)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)


# Fixed seed for the reprojection-magnitude band below. Any fixed seed works
# (drift is deterministic per seed); this one is pinned so the measured band is
# reproducible.
_DRIFT_BAND_SEED = 20240717


def _reprojection_errors(scene: Scene, seed: int, drift: CalibrationDrift) -> np.ndarray:
    """Per-point pixel gap between the TRUE and the RECORDED ASSUMED calibration.

    For every camera, the assumed ``K, R, t`` are read verbatim from the manifest
    (what a downstream consumer actually receives under drift) and assembled into
    ``P = K [R | t]``; each smoke-scene world point is projected through both the
    true camera and that assumed matrix, and the Euclidean pixel distance between
    the two projections is collected. The smoke scene guarantees every point
    projects in front of and inside every frame, so all samples are valid pixels.
    """
    manifest = build_manifest(scene, noise=NoiseModel(seed=seed, drift=drift))
    world_points = [
        np.asarray(xyz, dtype=np.float64)
        for entity in scene.entities
        for frame in entity.frames
        for xyz in frame.points.values()
    ]
    errors: list[float] = []
    for true_cam, cam in zip(scene.cameras, manifest.cameras, strict=True):
        assumed = cam.assumed
        assert assumed is not None
        assumed_p = projection_matrix(
            np.asarray(assumed.K, dtype=np.float64),
            np.asarray(assumed.R, dtype=np.float64),
            np.asarray(assumed.t, dtype=np.float64),
        )
        for point in world_points:
            true_uv, true_w = true_cam.project(point)
            assumed_uv, assumed_w = project_point(assumed_p, point)
            assert true_w > 0.0 and assumed_w > 0.0  # both in front of the camera
            errors.append(float(np.linalg.norm(np.asarray(true_uv) - np.asarray(assumed_uv))))
    return np.asarray(errors, dtype=np.float64)


def test_drift_reprojection_error_falls_in_expected_band() -> None:
    """Pin the MAGNITUDE of true-vs-assumed disagreement, not just its existence.

    The other drift tests pin reproducibility, that assumed != truth, and that R
    stays orthonormal — but none pins *how much* the assumed calibration disagrees
    with the truth, so a regression that silently weakens drift (a unit slip on
    ``rotation_sigma_deg``, an accidental extra normalization, a half-strength
    application) would pass them all. This projects the smoke-scene world points
    through the true camera and through the recorded assumed calibration and
    asserts the mean and max reprojection error land in a band.

    Measured at ``_DRIFT_BAND_SEED`` with ``_active_drift()``: mean ~4.14 px,
    max ~7.85 px. Reprojection error scales ~linearly with the drift sigmas, so
    halving the drift gives ~2.07 / ~3.93 px and doubling gives ~8.26 / ~15.65 px.
    The bands below (~0.7x–1.4x of the measured values) therefore reject both a
    half-strength and a doubled-strength regression, while the test is fully
    deterministic (fixed seed, exact arithmetic) so it never flakes.
    """
    scene = build_smoke_scene()
    errors = _reprojection_errors(scene, seed=_DRIFT_BAND_SEED, drift=_active_drift())
    mean_px = float(errors.mean())
    max_px = float(errors.max())
    assert 3.0 < mean_px < 5.5
    assert 5.5 < max_px < 11.0


# --------------------------------------------------------------------------- #
# geometry helper + typed validation
# --------------------------------------------------------------------------- #


def test_rotation_from_axis_angle_zero_is_identity() -> None:
    assert np.allclose(rotation_from_axis_angle(np.zeros(3)), np.eye(3))


def test_rotation_from_axis_angle_is_a_rotation() -> None:
    r = rotation_from_axis_angle(np.array([0.1, -0.2, 0.05]))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert float(np.linalg.det(r)) == pytest.approx(1.0)


def test_intrinsics_drift_only_touches_the_four_intrinsic_params() -> None:
    intr = Intrinsics.from_focal(800.0, 640, 480)
    cam = Camera.look_at(0, intr, np.array([4.0, 0.0, 1.5]), np.array([0.0, 0.0, 0.5]))
    rng = np.random.default_rng(0)
    assumed = cam.drifted(rng, _active_drift())
    assert assumed.intrinsics.width == intr.width
    assert assumed.intrinsics.height == intr.height
    assert assumed.intrinsics.fx != intr.fx


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_px": -1.0},
    ],
)
def test_pixel_noise_rejects_negative_sigma(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="sigma_px must be >= 0"):
        PixelNoise(**kwargs)


@pytest.mark.parametrize(
    "field",
    ["rotation_sigma_deg", "translation_sigma", "focal_sigma_px", "principal_point_sigma_px"],
)
def test_calibration_drift_rejects_negative_sigma(field: str) -> None:
    with pytest.raises(ValueError, match="drift sigmas must be >= 0"):
        CalibrationDrift(**{field: -1.0})


def test_is_active_flags() -> None:
    assert not NoiseModel().is_active
    assert NoiseModel(pixel=PixelNoise(sigma_px=0.1)).is_active
    assert NoiseModel(drift=CalibrationDrift(focal_sigma_px=0.1)).is_active
    assert not PixelNoise().is_active
    assert not CalibrationDrift().is_active
