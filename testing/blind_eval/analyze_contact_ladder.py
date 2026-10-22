"""
analyze_contact_ladder.py — score the realistic-contact ladder sets and
break the result down the three ways the decision rule needs.

Imports run_blind_eval's matching and scoring rather than reimplementing
them, so the ladder is judged by exactly the harness that produced the
committed baseline: ground truth read only after detection, one-to-one
greedy nearest-centroid assignment, the 1.5x radius gate, pooled
micro-averaged aggregation. Nothing here changes a quantify.py default.

Beyond pooled precision/recall/F1 it reports:

1. Recall bucketed by `visible_fraction`. This is what decides between the
   rule's "method holds on what's visible" and "real weakness" branches: a
   miss at 0% visible was never recoverable, a miss at 90% visible is a
   method failure, and one pooled recall number cannot tell them apart.
2. Recall by distance-transform pair class (A to D) for isolated
   overlapping pairs, using the same closed-form condition as the lab
   diagnostics. Class A pairs have no second distance-transform peak and
   are the true ceiling for any method seeded off a distance transform.
3. Where the false positives are, split by whether the extra detection
   landed inside a ground-truth colony's disc (an oversplit) or on bare
   agar (an invented colony).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from quantify import quantify_colonies
from testing.blind_eval.run_blind_eval import (
    MATCH_RADIUS_MULTIPLIER,
    aggregate_scores,
    match_detections_to_ground_truth,
    score_plate,
)

PLATES_DIR_DEFAULT = Path("testing/blind_eval/contact_plates")
RESULTS_DIR_DEFAULT = Path("testing/blind_eval/results")

VISIBILITY_BUCKETS: Tuple[Tuple[str, float, float], ...] = (
    ("under 0.10", 0.0, 0.10),
    ("0.10 to 0.25", 0.10, 0.25),
    ("0.25 to 0.50", 0.25, 0.50),
    ("0.50 and up", 0.50, 1.0001),
)


def pair_class(r1: float, r2: float, d: float) -> str:
    """Does the union of two overlapping discs have a second distance-transform
    peak, and how deep is the saddle? Same condition as the lab diagnostics."""
    if r1 < r2:
        r1, r2 = r2, r1
    if d * d <= r1 * r1 - r2 * r2:
        return "A no second peak"
    a = (d * d + r2 * r2 - r1 * r1) / (2 * d)
    depth = r2 - math.sqrt(max(r2 * r2 - a * a, 0.0))
    if depth < 1:
        return "B saddle <1 px"
    return "C saddle 1-2 px" if depth < 2 else "D saddle >=2 px"


def _isolated_overlapping_pairs(colonies: List[Dict]) -> List[Tuple[int, int, float]]:
    """Overlapping pairs whose two members touch nothing else, so the pair is
    the whole cluster and a two-disc model is the right unit for it."""
    neighbours = defaultdict(set)
    for a in range(len(colonies)):
        for b in range(a + 1, len(colonies)):
            d = math.hypot(colonies[a]["cx"] - colonies[b]["cx"],
                           colonies[a]["cy"] - colonies[b]["cy"])
            if d < colonies[a]["radius_px"] + colonies[b]["radius_px"]:
                neighbours[a].add(b)
                neighbours[b].add(a)
    pairs = []
    for a, partners in neighbours.items():
        if len(partners) != 1:
            continue
        b = next(iter(partners))
        if a < b and len(neighbours[b]) == 1:
            d = math.hypot(colonies[a]["cx"] - colonies[b]["cx"],
                           colonies[a]["cy"] - colonies[b]["cy"])
            pairs.append((a, b, d))
    return pairs


def analyze_set(set_name: str, rows: List[Dict]) -> Dict:
    plate_scores: List[Dict] = []
    visibility = {label: {"matched": 0, "total": 0} for label, _, _ in VISIBILITY_BUCKETS}
    pair_classes: Dict[str, Dict[str, int]] = defaultdict(lambda: {"split": 0, "not split": 0})
    fp_origin = {"inside a ground-truth colony (oversplit)": 0, "on bare agar (invented)": 0}
    miss_visible_25 = miss_total = 0

    for row in rows:
        # Blind call: only the image path crosses into quantify_colonies.
        result = quantify_colonies(row["image_path"])
        ground_truth = json.loads(Path(row["ground_truth_path"]).read_text())
        plate_scores.append(score_plate(result, ground_truth))

        colonies = ground_truth["colonies"]
        matches, unmatched_det, unmatched_gt = match_detections_to_ground_truth(
            result["contours"], colonies)
        matched_ids = {id(gt) for _, gt in matches}

        for colony in colonies:
            visible = colony.get("visible_fraction", 1.0)
            for label, lo, hi in VISIBILITY_BUCKETS:
                if lo <= visible < hi:
                    visibility[label]["total"] += 1
                    if id(colony) in matched_ids:
                        visibility[label]["matched"] += 1
                    break

        for colony in unmatched_gt:
            miss_total += 1
            if colony.get("visible_fraction", 1.0) < 0.25:
                miss_visible_25 += 1

        for a, b, d in _isolated_overlapping_pairs(colonies):
            cls = pair_class(colonies[a]["radius_px"], colonies[b]["radius_px"], d)
            both = id(colonies[a]) in matched_ids and id(colonies[b]) in matched_ids
            pair_classes[cls]["split" if both else "not split"] += 1

        for det in unmatched_det:
            dcx, dcy = det["centroid"]
            inside = any(math.hypot(c["cx"] - dcx, c["cy"] - dcy) <= c["radius_px"]
                         for c in colonies)
            key = ("inside a ground-truth colony (oversplit)" if inside
                   else "on bare agar (invented)")
            fp_origin[key] += 1

    pooled = aggregate_scores(plate_scores)
    return {
        "set": set_name,
        "pooled": pooled,
        "visibility_recall": {
            label: {
                "total": v["total"], "matched": v["matched"],
                "recall": (v["matched"] / v["total"]) if v["total"] else None,
            } for label, v in visibility.items()
        },
        "pair_classes": {k: dict(v) for k, v in sorted(pair_classes.items())},
        "false_positive_origin": fp_origin,
        "misses_total": miss_total,
        "misses_under_025_visible": miss_visible_25,
        "misses_under_025_visible_pct": (
            100.0 * miss_visible_25 / miss_total) if miss_total else None,
    }


def run(plates_dir: Path, sets: Optional[List[str]] = None) -> Dict:
    manifest = json.loads((plates_dir / "manifest.json").read_text())
    names = sets or list(manifest["sets"].keys())
    return {name: analyze_set(name, manifest["sets"][name]) for name in names}


# ── Report rendering ───────────────────────────────────────────────────────

def _verdict(report: Dict, set_name: str) -> str:
    r = report.get(set_name)
    if r is None:
        return "not run"
    f1 = r["pooled"]["f1"]
    pct = r["misses_under_025_visible_pct"]
    if f1 >= 0.90:
        return "method holds"
    if pct is not None and pct >= 80.0:
        return "method holds on what's visible"
    return "real weakness"


def render(report: Dict) -> str:
    lines = ["## Results", "",
             "_Auto-generated by testing/blind_eval/analyze_contact_ladder.py._", "",
             "### Pooled scores", "",
             "| set | n | precision | recall | F1 | count error % | misses | "
             "misses under 0.25 visible |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in report.items():
        p = r["pooled"]
        pct = r["misses_under_025_visible_pct"]
        pct_s = f"{pct:.1f}%" if pct is not None else "n/a"
        lines.append(
            f"| `{name}` | {p['n_plates']} | {p['precision']:.3f} | {p['recall']:.3f} | "
            f"{p['f1']:.3f} | {p['mean_count_error_pct']:.2f} | {r['misses_total']} | "
            f"{r['misses_under_025_visible']} ({pct_s}) |")

    lines += ["", "### Recall by visible fraction", "",
              "| set | " + " | ".join(l for l, _, _ in VISIBILITY_BUCKETS) + " |",
              "|---|" + "---|" * len(VISIBILITY_BUCKETS)]
    for name, r in report.items():
        cells = []
        for label, _, _ in VISIBILITY_BUCKETS:
            v = r["visibility_recall"][label]
            cells.append(f"{v['recall']:.3f} (n={v['total']})" if v["recall"] is not None
                         else "n/a")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines += ["", "### Isolated overlapping pairs, split rate by class", "",
              "| set | A no second peak | B saddle <1px | C saddle 1-2px | D saddle >=2px |",
              "|---|---|---|---|---|"]
    for name, r in report.items():
        cells = []
        for cls in ("A no second peak", "B saddle <1 px",
                    "C saddle 1-2 px", "D saddle >=2 px"):
            c = r["pair_classes"].get(cls)
            if not c:
                cells.append("n=0")
                continue
            total = c.get("split", 0) + c.get("not split", 0)
            cells.append(f"{c.get('split', 0)}/{total} ({100 * c.get('split', 0) / total:.0f}%)")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines += ["", "### False positives by origin", "",
              "| set | oversplit (inside a real colony) | invented (on bare agar) |",
              "|---|---|---|"]
    for name, r in report.items():
        o = r["false_positive_origin"]
        lines.append(f"| `{name}` | {o['inside a ground-truth colony (oversplit)']} | "
                     f"{o['on bare agar (invented)']} |")

    lines += ["", "### Outcome under the pre-registered rule", ""]
    for name in ("contact_r3", "contact_dense_r3"):
        lines.append(f"- `{name}`: **{_verdict(report, name)}**")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plates-dir", type=Path, default=PLATES_DIR_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--sets", nargs="+", default=None)
    args = parser.parse_args()

    report = run(args.plates_dir, args.sets)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "realistic_contact_results.json").write_text(
        json.dumps(report, indent=2))
    text = render(report)
    (args.results_dir / "realistic_contact_tables.md").write_text(text)
    print(text)
