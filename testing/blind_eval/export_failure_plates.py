"""
export_failure_plates.py — export 4-frame diagnostic visualizations for the
blind-eval plates the pipeline misses colonies on, so iteration decisions on
`quantify_colonies()` parameters are grounded in what the pipeline is
actually doing, not blind parameter tuning.

For each requested held-out set, every plate is scored against ground truth
(reusing run_blind_eval's exact matching logic), and the plates with the
most missed colonies (false negatives) are selected. Four panels are
rendered per selected plate:

  01_raw_input.png        - unmodified plate image
  02_binary_threshold.png - cleaned binary mask after background subtraction
  03_watershed_missed.png - watershed-labeled regions; ground-truth colonies
                            the pipeline missed are circled in red
  04_contour_overlay.png  - final annotated output (quantify.py's own overlay)

The plate-detection + background-subtraction + watershed steps are
re-derived locally (mirroring quantify.py's geometry and `_apply_watershed`)
rather than calling quantify.py's private helpers, because the watershed
`markers` array (needed to color-label regions for frame 3) isn't part of
`_apply_watershed`'s return value and this script must not change
quantify.py's behavior — this is a diagnostic export, not a pipeline change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from quantify import (
    PLATE_INNER_RADIUS_MM,
    detect_plate_circle,
    quantify_colonies,
    _subtract_background,
)
from testing.blind_eval.run_blind_eval import (
    PLATES_DIR_DEFAULT,
    match_detections_to_ground_truth,
    score_plate,
)

OUTPUT_DIR_DEFAULT = Path("testing/blind_eval/failure_analysis")
TOP_N_DEFAULT = 3


def _plate_geometry(gray: np.ndarray) -> Dict:
    """Mirrors quantify_colonies()'s plate-detection/calibration stage
    (quantify.py lines ~401-421) using its default rim_shrink_mm/
    plate_inner_radius_mm, so the mask handed to _subtract_background here
    matches what the real pipeline run used."""
    plate = detect_plate_circle(gray)
    if plate is not None:
        cx, cy, radius = plate
        rough_ppm = radius / (PLATE_INNER_RADIUS_MM + 3.0)
        inner_radius = max(0, radius - int(3.0 * rough_ppm))
    else:
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        radius = inner_radius = min(h, w) // 2
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (cx, cy), inner_radius, 255, -1)
    return {"cx": cx, "cy": cy, "radius": radius, "inner_radius": inner_radius, "mask": mask}


def _watershed_with_markers(image: np.ndarray, cleaned: np.ndarray) -> np.ndarray:
    """Reimplements quantify._apply_watershed()'s segmentation to also
    surface the labeled `markers` array, which that function doesn't
    return but which frame 3 needs to color each watershed region."""
    from quantify import WATERSHED_FG_THRESHOLD_FRAC

    k = np.ones((3, 3), np.uint8)
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, WATERSHED_FG_THRESHOLD_FRAC * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(cleaned, k, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    return cv2.watershed(image.copy(), markers)


def _label_color_image(markers: np.ndarray) -> np.ndarray:
    vis = np.zeros((*markers.shape, 3), np.uint8)
    rng = np.random.default_rng(42)
    for lbl in np.unique(markers):
        if lbl <= 1:
            continue
        vis[markers == lbl] = rng.integers(60, 255, size=3).tolist()
    vis[markers == -1] = (255, 255, 255)  # watershed boundary lines
    return vis


def _caption(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def export_plate(image_path: str, ground_truth: Dict, unmatched_gt: List[Dict],
                  out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(str(out_dir / "01_raw_input.png"), _caption(image, f"{label} - raw input"))

    geom = _plate_geometry(gray)
    cleaned = _subtract_background(gray, geom["mask"], blur_kernel=151, diff_threshold=3)
    binary_vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_dir / "02_binary_threshold.png"),
                _caption(binary_vis, f"{label} - binary threshold"))

    markers = _watershed_with_markers(image, cleaned)
    watershed_vis = _label_color_image(markers)
    for gt in unmatched_gt:
        cv2.circle(watershed_vis, (int(gt["cx"]), int(gt["cy"])), int(gt["radius_px"]) + 4,
                    (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / "03_watershed_missed.png"),
                _caption(watershed_vis, f"{label} - watershed, missed colonies circled red"))

    quantify_colonies(image_path, output_path=str(out_dir / "04_contour_overlay.png"))


def select_worst_plates(plates_dir: Path, set_name: str, top_n: int) -> List[Dict]:
    manifest = json.loads((plates_dir / "manifest.json").read_text())
    rows = manifest["sets"][set_name]

    scored = []
    for row in rows:
        result = quantify_colonies(row["image_path"])
        ground_truth = json.loads(Path(row["ground_truth_path"]).read_text())
        score = score_plate(result, ground_truth)
        _, _, unmatched_gt = match_detections_to_ground_truth(
            result["contours"], ground_truth["colonies"])
        scored.append({"row": row, "ground_truth": ground_truth,
                        "score": score, "unmatched_gt": unmatched_gt})

    scored.sort(key=lambda s: s["score"]["false_negatives"], reverse=True)
    return scored[:top_n]


def main(iteration: int, sets: List[str], top_n: int = TOP_N_DEFAULT,
         plates_dir: Path = PLATES_DIR_DEFAULT,
         output_dir: Path = OUTPUT_DIR_DEFAULT) -> None:
    iter_dir = output_dir / f"iteration_{iteration}"
    summary_lines = [f"# Failure Analysis — Iteration {iteration}", "",
                      "Plates selected as the most false negatives (missed colonies) "
                      "per failing held-out set.", ""]

    for set_name in sets:
        worst = select_worst_plates(plates_dir, set_name, top_n)
        summary_lines.append(f"## {set_name}")
        summary_lines.append("")
        summary_lines.append("| plate | expected | detected | false negatives | recall | count error % |")
        summary_lines.append("|---|---|---|---|---|---|")
        for item in worst:
            plate_stem = Path(item["row"]["image_path"]).stem
            s = item["score"]
            summary_lines.append(
                f"| `{plate_stem}` | {s['expected_count']} | {s['detected_count']} | "
                f"{s['false_negatives']} | {s['recall']:.3f} | {s['count_error_pct']:.2f} |")
            plate_dir = iter_dir / set_name / plate_stem
            export_plate(item["row"]["image_path"], item["ground_truth"],
                         item["unmatched_gt"], plate_dir, f"{set_name}/{plate_stem}")
        summary_lines.append("")

    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "summary.md").write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--sets", nargs="+", required=True)
    parser.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    parser.add_argument("--plates-dir", type=Path, default=PLATES_DIR_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()

    main(args.iteration, args.sets, args.top_n, args.plates_dir, args.output_dir)
