"""
run_blind_eval.py — blind evaluation harness for testing/blind_eval/plates/.

For each held-out plate, quantify_colonies() is called with only the image
path. Ground truth is loaded strictly afterward, to score what the pipeline
already returned — never before or during that call. That ordering is the
"blind" in blind validation: nothing about the held-out plate's true count
or anomaly labels is available to the pipeline at evaluation time.

Matching (not raw count/anomaly-count comparison) drives every metric here,
for the reason phase_c.score_track1() already matches by nearest centroid
rather than comparing counts: a false positive and a missed real colony can
otherwise cancel out and look like a perfect score. This harness needs
precision in addition to recall, which requires each detection matched to
at most one ground-truth colony (and vice versa) — score_track1's simpler
nearest-match-per-detection isn't one-to-one, so matching here is its own,
stricter implementation rather than a reuse of that function.

Aggregation is pooled (micro-averaged): true/false positives and false
negatives are summed across every plate in a set before computing one
precision/recall/F1, rather than averaging each plate's own F1. Pooling is
what "F1 > 0.90 on the held-out set" means here — it keeps a handful of
near-empty plates from swinging the aggregate disproportionately. Per-plate
mean F1 is reported alongside for visibility into variance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from quantify import quantify_colonies

PLATES_DIR_DEFAULT = Path("testing/blind_eval/plates")
RESULTS_DIR_DEFAULT = Path("testing/blind_eval/results")

# A detection matches a ground-truth colony only within this multiple of
# that colony's own radius — mirrors phase_c.score_track1()'s 1.5x rule so a
# spurious detection near an unrelated colony (e.g. an artifact) isn't
# miscounted as a match.
MATCH_RADIUS_MULTIPLIER = 1.5

TARGET_F1 = 0.90


# ── Matching ─────────────────────────────────────────────────────────────

def match_detections_to_ground_truth(
        detections: List[Dict], gt_colonies: List[Dict],
) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
    """One-to-one nearest-centroid matching via greedy nearest-first
    assignment: every (detection, gt) pair within MATCH_RADIUS_MULTIPLIER is
    a candidate, candidates are consumed in ascending distance order, and
    once either side of a candidate is claimed it can't match again. This
    keeps two detections near the same crowded colony from both claiming it.

    Returns (matches, unmatched_detections, unmatched_gt).
    """
    candidates = []
    for di, det in enumerate(detections):
        dcx, dcy = det["centroid"]
        for gi, gt in enumerate(gt_colonies):
            dist = math.hypot(gt["cx"] - dcx, gt["cy"] - dcy)
            if dist <= gt["radius_px"] * MATCH_RADIUS_MULTIPLIER:
                candidates.append((dist, di, gi))
    candidates.sort(key=lambda c: c[0])

    matched_d, matched_g = set(), set()
    matches: List[Tuple[Dict, Dict]] = []
    for _, di, gi in candidates:
        if di in matched_d or gi in matched_g:
            continue
        matched_d.add(di)
        matched_g.add(gi)
        matches.append((detections[di], gt_colonies[gi]))

    unmatched_detections = [d for i, d in enumerate(detections) if i not in matched_d]
    unmatched_gt = [g for i, g in enumerate(gt_colonies) if i not in matched_g]
    return matches, unmatched_detections, unmatched_gt


# ── Per-plate scoring ────────────────────────────────────────────────────

def score_plate(quantify_result: Dict, ground_truth: Dict) -> Dict:
    gt_colonies = ground_truth["colonies"]
    detections = quantify_result["contours"]
    expected = ground_truth["expected_count"]
    detected = quantify_result["count"]

    matches, unmatched_det, unmatched_gt = match_detections_to_ground_truth(detections, gt_colonies)
    tp, fp, fn = len(matches), len(unmatched_det), len(unmatched_gt)

    precision = tp / detected if detected else (1.0 if expected == 0 else 0.0)
    recall = tp / expected if expected else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    count_error_pct = (abs(detected - expected) / expected * 100) if expected else (0.0 if detected == 0 else 100.0)

    # Anomaly scoring only judges matched pairs plus the two ways an
    # unmatched side can still carry an anomaly signal: an unmatched
    # detection flagged anomalous is a false positive (the pipeline invented
    # an anomalous colony that wasn't there), and an unmatched ground-truth
    # colony that WAS anomalous is a false negative (missed the colony and
    # the anomaly with it — the worst case, not merely a wrong flag).
    a_tp = a_fp = a_fn = 0
    for det, gt in matches:
        det_anomalous = bool(det["anomaly_flags"])
        gt_anomalous = gt["is_anomaly"]
        if det_anomalous and gt_anomalous:
            a_tp += 1
        elif det_anomalous and not gt_anomalous:
            a_fp += 1
        elif gt_anomalous:
            a_fn += 1
    a_fp += sum(1 for d in unmatched_det if d["anomaly_flags"])
    a_fn += sum(1 for g in unmatched_gt if g["is_anomaly"])

    expected_anomaly_count = ground_truth["expected_anomaly_count"]
    a_precision = a_tp / (a_tp + a_fp) if (a_tp + a_fp) else (1.0 if expected_anomaly_count == 0 else 0.0)
    a_recall = a_tp / expected_anomaly_count if expected_anomaly_count else 1.0
    a_f1 = (2 * a_precision * a_recall / (a_precision + a_recall)) if (a_precision + a_recall) > 0 else 0.0

    return {
        "expected_count": expected, "detected_count": detected, "count_error_pct": count_error_pct,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "expected_anomaly_count": expected_anomaly_count,
        "anomaly_true_positives": a_tp, "anomaly_false_positives": a_fp, "anomaly_false_negatives": a_fn,
        "anomaly_precision": a_precision, "anomaly_recall": a_recall, "anomaly_f1": a_f1,
    }


# ── Aggregation ──────────────────────────────────────────────────────────

def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def aggregate_scores(per_plate_scores: List[Dict]) -> Dict:
    n = len(per_plate_scores)
    if n == 0:
        return {"n_plates": 0, "precision": None, "recall": None, "f1": None,
                "anomaly_precision": None, "anomaly_recall": None, "anomaly_f1": None,
                "mean_count_error_pct": None, "mean_per_plate_f1": None}

    precision, recall, f1 = _prf(
        sum(s["true_positives"] for s in per_plate_scores),
        sum(s["false_positives"] for s in per_plate_scores),
        sum(s["false_negatives"] for s in per_plate_scores))
    a_precision, a_recall, a_f1 = _prf(
        sum(s["anomaly_true_positives"] for s in per_plate_scores),
        sum(s["anomaly_false_positives"] for s in per_plate_scores),
        sum(s["anomaly_false_negatives"] for s in per_plate_scores))

    return {
        "n_plates": n,
        "precision": precision, "recall": recall, "f1": f1,
        "anomaly_precision": a_precision, "anomaly_recall": a_recall, "anomaly_f1": a_f1,
        "mean_count_error_pct": sum(s["count_error_pct"] for s in per_plate_scores) / n,
        "mean_per_plate_f1": sum(s["f1"] for s in per_plate_scores) / n,
    }


# ── Runner ───────────────────────────────────────────────────────────────

def run_blind_eval(plates_dir: Path = PLATES_DIR_DEFAULT,
                    sets: Optional[List[str]] = None) -> Dict:
    manifest = json.loads((plates_dir / "manifest.json").read_text())
    set_names = sets or list(manifest["sets"].keys())

    per_set_scores: Dict[str, List[Dict]] = {}
    for set_name in set_names:
        scores = []
        for row in manifest["sets"][set_name]:
            # Blind call: only the image path crosses into quantify_colonies.
            result = quantify_colonies(row["image_path"])
            ground_truth = json.loads(Path(row["ground_truth_path"]).read_text())
            scores.append(score_plate(result, ground_truth))
        per_set_scores[set_name] = scores

    all_scores = [s for scores in per_set_scores.values() for s in scores]
    return {
        "by_set": {name: aggregate_scores(scores) for name, scores in per_set_scores.items()},
        "overall": aggregate_scores(all_scores),
        "target_f1": TARGET_F1,
    }


# ── Report rendering ───────────────────────────────────────────────────────

def render_results_table(report: Dict) -> str:
    lines = ["# Blind Validation Results", "",
             "_Auto-generated by testing/blind_eval/run_blind_eval.py — "
             "do not hand-edit._", "",
             f"Target: F1 > {report['target_f1']:.2f} on each held-out set.", "",
             "## By held-out set", "",
             "| set | n | precision | recall | F1 | mean per-plate F1 | count error % | "
             "anomaly precision | anomaly recall | anomaly F1 | pass |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in sorted(report["by_set"].items()):
        passed = s["f1"] is not None and s["f1"] >= report["target_f1"]
        lines.append(
            f"| `{name}` | {s['n_plates']} | {s['precision']:.3f} | {s['recall']:.3f} | "
            f"{s['f1']:.3f} | {s['mean_per_plate_f1']:.3f} | {s['mean_count_error_pct']:.2f} | "
            f"{s['anomaly_precision']:.3f} | {s['anomaly_recall']:.3f} | {s['anomaly_f1']:.3f} | "
            f"{'PASS' if passed else 'FAIL'} |")
    lines.append("")

    o = report["overall"]
    passed = o["f1"] is not None and o["f1"] >= report["target_f1"]
    lines += ["## Overall (pooled across all held-out sets)", "",
              f"- n plates: {o['n_plates']}",
              f"- Precision: {o['precision']:.3f}",
              f"- Recall: {o['recall']:.3f}",
              f"- **F1: {o['f1']:.3f}** ({'PASS' if passed else 'FAIL'} vs target {report['target_f1']:.2f})",
              f"- Mean per-plate F1: {o['mean_per_plate_f1']:.3f}",
              f"- Mean count error: {o['mean_count_error_pct']:.2f}%",
              f"- Anomaly precision / recall / F1: "
              f"{o['anomaly_precision']:.3f} / {o['anomaly_recall']:.3f} / {o['anomaly_f1']:.3f}",
              ""]
    return "\n".join(lines)


def main(plates_dir: Path = PLATES_DIR_DEFAULT, results_dir: Path = RESULTS_DIR_DEFAULT,
         sets: Optional[List[str]] = None) -> Dict:
    report = run_blind_eval(plates_dir, sets)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "blind_eval_results.json").write_text(json.dumps(report, indent=2))
    (results_dir / "blind_eval_results.md").write_text(render_results_table(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plates-dir", type=Path, default=PLATES_DIR_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--sets", nargs="+", default=None,
                         help="Subset of held-out sets to evaluate (default: all in manifest).")
    args = parser.parse_args()

    report = main(args.plates_dir, args.results_dir, args.sets)
    print(render_results_table(report))
