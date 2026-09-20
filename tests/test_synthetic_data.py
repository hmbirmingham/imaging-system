"""Tests verifying synthetic ground truth is recoverable.

These don't re-test quantify.py's detection accuracy (that's Track 1's job,
exercised continuously by run_cycle.py) — they verify the generator itself:
determinism, schema, and that the physical quantities it claims (colony
count, area_mm2, distance-invariant px_per_mm) are internally consistent.
"""
import cv2
import numpy as np

from testing.continuous.synthetic_data import (
    FeatureScenario,
    HELD_OUT_CIRCULARITY_NOISE_STD_RANGE,
    HELD_OUT_COLONY_COUNT_RANGE,
    HELD_OUT_SET_NAMES,
    MIN_SUPPORTED_CAMERA_DISTANCE_FACTOR,
    PlateScenario,
    expected_px_per_mm,
    generate_colony_features,
    generate_held_out_plate,
    generate_held_out_scenario,
    generate_plate_image,
)
from anomaly import ML_FEATURES
from quantify import NON_CIRCULAR_THRESHOLD
import quantify as q


def test_generate_plate_image_is_deterministic():
    scenario = PlateScenario(seed=42, illumination="gradient", density="dense",
                              artifacts=("streak", "debris"))
    img1, gt1 = generate_plate_image(scenario)
    img2, gt2 = generate_plate_image(scenario)
    assert np.array_equal(img1, img2)
    assert gt1 == gt2


def test_generate_plate_image_different_seeds_differ():
    img1, _ = generate_plate_image(PlateScenario(seed=1))
    img2, _ = generate_plate_image(PlateScenario(seed=2))
    assert not np.array_equal(img1, img2)


def test_ground_truth_colony_count_matches_density_range():
    from testing.continuous.synthetic_data import DENSITY_RANGES
    for density, (lo, hi) in DENSITY_RANGES.items():
        _, gt = generate_plate_image(PlateScenario(seed=7, density=density))
        assert lo <= gt["expected_count"] <= hi
        assert gt["expected_count"] == len(gt["colonies"])


def test_ground_truth_area_recoverable_by_quantify_pipeline(tmp_path):
    """The rendered image should be detectable by the real pipeline, and the
    detected count should be close to the synthetic ground truth count."""
    scenario = PlateScenario(seed=3, density="sparse", illumination="uniform")
    img, gt = generate_plate_image(scenario)
    src = tmp_path / "plate.jpg"
    cv2.imwrite(str(src), img)

    result = q.quantify_colonies(str(src), str(tmp_path / "out.jpg"))
    # Sparse, non-touching, non-artifact scenario — detection should be exact
    # or within 1 (edge rounding on the smallest colonies).
    assert abs(result["count"] - gt["expected_count"]) <= 1


def test_camera_distance_does_not_change_expected_area():
    """Real colony sizes are fixed in mm; only their pixel footprint should
    change with simulated camera distance — this is the crux of the
    distance-invariance claim tested continuously by Track 1."""
    near_scenario = PlateScenario(seed=11, density="sparse",
                                   camera_distance_factor=MIN_SUPPORTED_CAMERA_DISTANCE_FACTOR)
    far_scenario = PlateScenario(seed=11, density="sparse", camera_distance_factor=1.4)
    _, gt_near = generate_plate_image(near_scenario)
    _, gt_far = generate_plate_image(far_scenario)

    areas_near = sorted(c["area_mm2"] for c in gt_near["colonies"])
    areas_far = sorted(c["area_mm2"] for c in gt_far["colonies"])
    assert areas_near == areas_far  # same seed -> same mm-space colonies
    # But the rendered pixel radius must actually differ between the two.
    assert gt_near["plate"]["radius_px"] != gt_far["plate"]["radius_px"]


def test_min_supported_camera_distance_factor_is_actually_detectable():
    """MIN_SUPPORTED_CAMERA_DISTANCE_FACTOR claims quantify.detect_plate_circle()
    can still find the plate at that distance — guard against the two
    constants drifting apart if REFERENCE_PLATE_RADIUS_FRACTION ever changes."""
    scenario = PlateScenario(seed=1, density="sparse",
                              camera_distance_factor=MIN_SUPPORTED_CAMERA_DISTANCE_FACTOR)
    img, _ = generate_plate_image(scenario)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    assert q.detect_plate_circle(gray) is not None


def test_expected_px_per_mm_matches_quantify_calibration(tmp_path):
    scenario = PlateScenario(seed=5, density="sparse", illumination="uniform")
    img, gt = generate_plate_image(scenario)
    src = tmp_path / "plate.jpg"
    cv2.imwrite(str(src), img)

    result = q.quantify_colonies(str(src), str(tmp_path / "out.jpg"))
    predicted = expected_px_per_mm(gt["plate"]["radius_px"])
    assert abs(result["px_per_mm"] - predicted) / predicted < 0.05


def test_invalid_scenario_parameters_raise():
    import pytest
    with pytest.raises(ValueError):
        PlateScenario(seed=1, illumination="not_a_real_illumination")
    with pytest.raises(ValueError):
        PlateScenario(seed=1, density="not_a_real_density")
    with pytest.raises(ValueError):
        PlateScenario(seed=1, artifacts=("not_a_real_artifact",))
    with pytest.raises(ValueError):
        PlateScenario(seed=1, camera_distance_factor=0)


def test_generate_colony_features_schema_and_determinism():
    scenario = FeatureScenario(n_samples=120, seed=99, anomaly_fraction=0.2, label_noise=0.1)
    df1 = generate_colony_features(scenario)
    df2 = generate_colony_features(scenario)

    assert len(df1) == 120
    for col in ML_FEATURES:
        assert col in df1.columns
    assert "is_anomaly" in df1.columns
    assert "true_is_anomaly" in df1.columns
    assert df1.equals(df2)


def test_generate_colony_features_label_noise_flips_some_labels():
    clean = FeatureScenario(n_samples=300, seed=1, anomaly_fraction=0.2, label_noise=0.0)
    noisy = FeatureScenario(n_samples=300, seed=1, anomaly_fraction=0.2, label_noise=0.2)
    df_clean = generate_colony_features(clean)
    df_noisy = generate_colony_features(noisy)

    assert (df_clean["is_anomaly"] == df_clean["true_is_anomaly"]).all()
    n_flipped = (df_noisy["is_anomaly"] != df_noisy["true_is_anomaly"]).sum()
    assert n_flipped == round(300 * 0.2)


def test_generate_colony_features_anomaly_fraction_respected():
    scenario = FeatureScenario(n_samples=500, seed=4, anomaly_fraction=0.3, label_noise=0.0)
    df = generate_colony_features(scenario)
    assert df["true_is_anomaly"].sum() == round(500 * 0.3)


# ── Blind validation: held-out parameter overrides ──────────────────────────

def test_default_plate_scenario_unaffected_by_new_override_fields():
    """Every held-out override field defaults to None/0.0 — an unrelated
    scenario (as used by the continuous harness) must render byte-identical
    to before these fields existed. Pinned against a fixed seed so any
    accidental extra rng draw (e.g. from a non-short-circuited check) would
    be caught by a changed image/ground truth rather than passing silently."""
    scenario = PlateScenario(seed=42, illumination="gradient", density="dense",
                              artifacts=("streak", "debris"))
    img, gt = generate_plate_image(scenario)
    assert gt["expected_count"] == 29
    assert gt["touching_pairs"] == 6
    assert all(c["circularity"] == 1.0 for c in gt["colonies"])


def test_colony_count_range_override_widens_density():
    lo, hi = HELD_OUT_COLONY_COUNT_RANGE
    _, gt = generate_plate_image(PlateScenario(seed=7, density="dense",
                                                colony_count_range=HELD_OUT_COLONY_COUNT_RANGE))
    assert lo <= gt["expected_count"] <= hi


def test_circularity_noise_reduces_measured_circularity():
    lo, _ = HELD_OUT_CIRCULARITY_NOISE_STD_RANGE
    _, gt_noisy = generate_plate_image(PlateScenario(seed=8, circularity_noise_std=lo))
    _, gt_clean = generate_plate_image(PlateScenario(seed=8, circularity_noise_std=0.0))
    assert all(c["circularity"] == 1.0 for c in gt_clean["colonies"])
    assert any(c["circularity"] < 1.0 for c in gt_noisy["colonies"])
    assert all(0.0 <= c["circularity"] <= 1.0 for c in gt_noisy["colonies"])


def test_is_anomaly_matches_deterministic_flags_only():
    """is_anomaly must exactly equal (touching OR below NON_CIRCULAR_THRESHOLD)
    — the two flags knowable at draw time without the plate-wide Z-score
    stats quantify.py's own relative flags depend on."""
    for seed in (100, 101, 102):
        _, gt = generate_plate_image(PlateScenario(seed=seed, density="dense",
                                                     circularity_noise_std=0.2))
        for c in gt["colonies"]:
            expected = c["touching"] or c["circularity"] < NON_CIRCULAR_THRESHOLD
            assert c["is_anomaly"] == expected
            assert ("touching_colony" in c["anomaly_type"]) == c["touching"]
            assert ("non_circular" in c["anomaly_type"]) == (c["circularity"] < NON_CIRCULAR_THRESHOLD)


def test_touching_rate_override_applies_regardless_of_density():
    """Prior behavior only ever forced touching on density='dense'; the
    override must work on any density tier."""
    _, gt = generate_plate_image(PlateScenario(seed=9, density="sparse", touching_rate=1.0))
    assert gt["touching_pairs"] >= 1


def test_hotspot_magnitude_override_changes_illumination_but_not_default():
    default_field, _ = generate_plate_image(PlateScenario(seed=10, illumination="hotspot"))
    overridden_field, _ = generate_plate_image(
        PlateScenario(seed=10, illumination="hotspot", hotspot_magnitude_pct=0.45))
    assert not np.array_equal(default_field, overridden_field)


def test_size_variance_ratio_widens_colony_radius_spread():
    _, gt = generate_plate_image(PlateScenario(seed=12, density="dense", size_variance_ratio=4.5))
    radii = [c["radius_mm"] for c in gt["colonies"]]
    assert max(radii) / min(radii) > 2.0  # well beyond COLONY_RADIUS_MM_RANGE's ~3.67x cap headroom


def test_invalid_held_out_override_parameters_raise():
    import pytest
    with pytest.raises(ValueError):
        PlateScenario(seed=1, colony_count_range=(0, 10))
    with pytest.raises(ValueError):
        PlateScenario(seed=1, colony_count_range=(20, 10))
    with pytest.raises(ValueError):
        PlateScenario(seed=1, circularity_noise_std=-0.1)
    with pytest.raises(ValueError):
        PlateScenario(seed=1, hotspot_magnitude_pct=1.5)
    with pytest.raises(ValueError):
        PlateScenario(seed=1, size_variance_ratio=1.0)
    with pytest.raises(ValueError):
        PlateScenario(seed=1, touching_rate=1.5)


def test_generate_held_out_scenario_rejects_unknown_set():
    import pytest
    with pytest.raises(ValueError):
        generate_held_out_scenario("not_a_real_set", seed=1)


def test_generate_held_out_plate_is_deterministic_and_tagged():
    for set_name in HELD_OUT_SET_NAMES:
        img1, gt1 = generate_held_out_plate(set_name, seed=555)
        img2, gt2 = generate_held_out_plate(set_name, seed=555)
        assert np.array_equal(img1, img2)
        assert gt1 == gt2
        assert gt1["held_out_set"] == set_name


def test_generate_held_out_plate_different_seeds_sample_different_stress_values():
    """Each set samples its stressed axis from the seed, so two different
    seeds in the same set should (almost always) differ in the sampled
    value, not just in colony placement."""
    _, gt1 = generate_held_out_plate("irregular_morphology", seed=1)
    _, gt2 = generate_held_out_plate("irregular_morphology", seed=2)
    assert gt1["scenario"]["circularity_noise_std"] != gt2["scenario"]["circularity_noise_std"]
