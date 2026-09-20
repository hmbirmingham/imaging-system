"""Tests for the blind evaluation harness. Matching/scoring/aggregation are
tested against synthetic fixture dicts (fast, no real quantify_colonies
call); a couple of integration tests run the real pipeline against a
generated held-out plate to confirm the fixture shapes used above actually
match what quantify_colonies() and the synthetic generator produce."""
import cv2
import pytest

from testing.blind_eval.run_blind_eval import (
    aggregate_scores,
    match_detections_to_ground_truth,
    render_results_table,
    score_plate,
)
from testing.continuous.synthetic_data import generate_held_out_plate
from quantify import quantify_colonies


def _det(cx, cy, anomaly_flags=()):
    return {"centroid": (cx, cy), "anomaly_flags": list(anomaly_flags)}


def _gt(cx, cy, radius_px=10.0, is_anomaly=False):
    return {"cx": cx, "cy": cy, "radius_px": radius_px, "is_anomaly": is_anomaly}


# ── Matching ─────────────────────────────────────────────────────────────

def test_match_perfect_alignment():
    dets = [_det(10, 10), _det(50, 50)]
    gts = [_gt(10, 10), _gt(50, 50)]
    matches, unmatched_det, unmatched_gt = match_detections_to_ground_truth(dets, gts)
    assert len(matches) == 2
    assert not unmatched_det and not unmatched_gt


def test_match_rejects_distant_pair_beyond_radius_multiplier():
    dets = [_det(100, 100)]
    gts = [_gt(10, 10, radius_px=5.0)]  # far outside 1.5x radius
    matches, unmatched_det, unmatched_gt = match_detections_to_ground_truth(dets, gts)
    assert matches == []
    assert unmatched_det == dets
    assert unmatched_gt == gts


def test_match_is_one_to_one_nearest_wins():
    """Two detections near the same ground-truth colony must not both match
    it — only the nearer one should, and the other stays unmatched."""
    gts = [_gt(10, 10, radius_px=20.0)]
    near = _det(11, 10)
    far = _det(18, 10)
    matches, unmatched_det, unmatched_gt = match_detections_to_ground_truth([near, far], gts)
    assert len(matches) == 1
    assert matches[0][0] is near
    assert unmatched_det == [far]
    assert unmatched_gt == []


# ── Per-plate scoring ────────────────────────────────────────────────────

def _quantify_result(detections, count=None):
    return {"count": count if count is not None else len(detections), "contours": detections}


def _ground_truth(colonies, expected_anomaly_count=None):
    return {"colonies": colonies, "expected_count": len(colonies),
            "expected_anomaly_count": (expected_anomaly_count if expected_anomaly_count is not None
                                        else sum(1 for c in colonies if c["is_anomaly"]))}


def test_score_plate_perfect_match():
    gts = [_gt(0, 0), _gt(50, 50)]
    dets = [_det(0, 0), _det(50, 50)]
    score = score_plate(_quantify_result(dets), _ground_truth(gts))
    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    assert score["f1"] == 1.0
    assert score["count_error_pct"] == 0.0


def test_score_plate_missed_colony_hurts_recall_not_precision():
    gts = [_gt(0, 0), _gt(50, 50)]
    dets = [_det(0, 0)]  # missed the second colony
    score = score_plate(_quantify_result(dets), _ground_truth(gts))
    assert score["precision"] == 1.0
    assert score["recall"] == 0.5
    assert score["false_negatives"] == 1


def test_score_plate_false_positive_hurts_precision_not_recall():
    gts = [_gt(0, 0)]
    dets = [_det(0, 0), _det(200, 200)]  # spurious extra detection, far away
    score = score_plate(_quantify_result(dets), _ground_truth(gts))
    assert score["recall"] == 1.0
    assert score["precision"] == 0.5
    assert score["false_positives"] == 1


def test_score_plate_count_could_match_while_matching_is_wrong():
    """A count-only comparison would call this a perfect score (1 expected,
    1 detected) — matching must catch that the single detection is nowhere
    near the single ground-truth colony."""
    gts = [_gt(0, 0)]
    dets = [_det(500, 500)]
    score = score_plate(_quantify_result(dets), _ground_truth(gts))
    assert score["expected_count"] == score["detected_count"] == 1
    assert score["precision"] == 0.0
    assert score["recall"] == 0.0
    assert score["count_error_pct"] == 0.0  # counts do match...
    assert score["f1"] == 0.0               # ...but nothing else does


def test_score_plate_anomaly_true_positive():
    gts = [_gt(0, 0, is_anomaly=True)]
    dets = [_det(0, 0, anomaly_flags=["non_circular"])]
    score = score_plate(_quantify_result(dets), _ground_truth(gts))
    assert score["anomaly_true_positives"] == 1
    assert score["anomaly_precision"] == 1.0
    assert score["anomaly_recall"] == 1.0


def test_score_plate_anomaly_false_positive_on_normal_colony():
    gts = [_gt(0, 0, is_anomaly=False)]
    dets = [_det(0, 0, anomaly_flags=["unusual_size"])]  # pipeline over-flags
    score = score_plate(_quantify_result(dets), _ground_truth(gts))
    assert score["anomaly_false_positives"] == 1
    assert score["anomaly_precision"] == 0.0
    assert score["anomaly_recall"] == 1.0  # no anomalies expected -> vacuously perfect recall


def test_score_plate_anomaly_false_negative_on_missed_colony():
    """A ground-truth colony that's anomalous AND never detected at all must
    count as an anomaly false negative, not be silently dropped because it
    never reached the matched-pairs comparison."""
    gts = [_gt(0, 0, is_anomaly=True)]
    dets = []  # colony never detected
    score = score_plate(_quantify_result(dets), _ground_truth(gts))
    assert score["anomaly_false_negatives"] == 1
    assert score["anomaly_recall"] == 0.0


def test_score_plate_anomaly_false_positive_on_spurious_detection():
    gts = [_gt(0, 0, is_anomaly=False)]
    dets = [_det(0, 0), _det(300, 300, anomaly_flags=["streak_or_artifact"])]
    score = score_plate(_quantify_result(dets), _ground_truth(gts))
    assert score["anomaly_false_positives"] == 1


def test_score_plate_empty_ground_truth_and_no_detections_is_perfect():
    score = score_plate(_quantify_result([]), _ground_truth([]))
    assert score["precision"] == score["recall"] == score["f1"] == 1.0
    assert score["count_error_pct"] == 0.0


# ── Aggregation ──────────────────────────────────────────────────────────

def test_aggregate_scores_pools_counts_not_averages_ratios():
    """Two plates: one perfect (1/1), one total miss (0/1) should pool to
    recall 0.5 overall — not the mean of per-plate recalls, which would also
    be 0.5 here by coincidence, so this uses uneven plate sizes to actually
    distinguish pooling from macro-averaging."""
    small_perfect = score_plate(_quantify_result([_det(0, 0)]), _ground_truth([_gt(0, 0)]))
    big_miss = score_plate(_quantify_result([]), _ground_truth([_gt(i, i) for i in range(9)]))
    agg = aggregate_scores([small_perfect, big_miss])
    # Pooled: tp=1, fn=9 -> recall = 1/10, not mean(1.0, 0.0) = 0.5
    assert agg["recall"] == pytest.approx(0.1)
    assert agg["mean_per_plate_f1"] == pytest.approx((1.0 + 0.0) / 2)


def test_aggregate_scores_empty_list():
    agg = aggregate_scores([])
    assert agg["n_plates"] == 0
    assert agg["f1"] is None


# ── Report rendering ─────────────────────────────────────────────────────

def test_render_results_table_includes_set_names_and_pass_fail():
    report = {
        "target_f1": 0.90,
        "by_set": {"dense_pack": aggregate_scores([
            score_plate(_quantify_result([_det(0, 0)]), _ground_truth([_gt(0, 0)]))])},
        "overall": aggregate_scores([
            score_plate(_quantify_result([_det(0, 0)]), _ground_truth([_gt(0, 0)]))]),
    }
    table = render_results_table(report)
    assert "dense_pack" in table
    assert "PASS" in table
    assert "0.90" in table


# ── Integration: real pipeline against a real generated plate ──────────────

def test_score_plate_against_real_pipeline_output(tmp_path):
    """Confirms the fixture shapes used above (quantify_result['contours']
    entries having 'centroid'/'anomaly_flags', ground truth colonies having
    'cx'/'cy'/'radius_px'/'is_anomaly') match what the real generator and
    real quantify_colonies() actually produce, end to end."""
    image, ground_truth = generate_held_out_plate("dense_pack", seed=999001)
    image_path = tmp_path / "plate.png"
    cv2.imwrite(str(image_path), image)

    result = quantify_colonies(str(image_path))
    score = score_plate(result, ground_truth)

    assert 0.0 <= score["precision"] <= 1.0
    assert 0.0 <= score["recall"] <= 1.0
    assert 0.0 <= score["f1"] <= 1.0
    assert score["expected_count"] == ground_truth["expected_count"]
    assert score["expected_anomaly_count"] == ground_truth["expected_anomaly_count"]
