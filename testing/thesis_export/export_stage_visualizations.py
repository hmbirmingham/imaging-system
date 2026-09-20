"""
export_stage_visualizations.py — publication-quality, stage-by-stage
pipeline figures for the technical memo and thesis preview.

Six canonical frames per plate, using quantify_colonies(return_intermediates=True)
for every per-stage array (background-subtracted image, binary mask,
watershed labels, contour polygons) so these figures can't drift from what
the real pipeline does:

  01_raw_input.png             - unmodified plate image
  02_illumination_corrected.png - background-subtracted image, pre-threshold
  03_binary_threshold.png      - binary mask after thresholding
  04_watershed_labeled.png     - watershed segmentation, one color per colony
  05_contour_overlay.png       - final contours, per-colony area/circularity/
                                  aspect-ratio annotated on a representative subset
  06_anomaly_flagged.png       - same contours, colored red (anomalous) / green
                                  (normal) instead of neutral

Every frame is composited onto a white 1400x1400 canvas (>= the 1200x1200
minimum) using the plate's own detected circle geometry to separate "plate"
from "margin" — not a color threshold — since colonies under backlight are
themselves dark and a naive near-black threshold would misclassify them as
margin to whiten.

Output: docs/figures/<plate_label>/ — this is the memo's own artifact
directory (unlike testing/blind_eval/failure_analysis/, which is internal
diagnostic output), so PNGs here are committed, not gitignored.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from quantify import quantify_colonies
from testing.blind_eval.export_failure_plates import label_color_image
from testing.blind_eval.run_blind_eval import score_plate

PLATES_DIR_DEFAULT = Path("testing/blind_eval/plates")
OUTPUT_DIR_DEFAULT = Path("docs/figures")

CANVAS_SIZE = 1400
MARGIN_PX = 40
MAX_ANNOTATED_COLONIES = 8

TITLE_COLOR = (20, 20, 20)
SUBTITLE_COLOR = (90, 90, 90)


def _compose_on_white(image_bgr: np.ndarray, plate_circle: Dict,
                       canvas_size: int = CANVAS_SIZE, margin_px: int = MARGIN_PX,
                       title_band_px: int = 130) -> np.ndarray:
    """Places `image_bgr` on a white canvas, keeping only pixels inside the
    plate's own detected circle (+ margin) — geometry-based, not a color
    threshold, so dark colony pixels near the rim are never mistaken for
    the image's black outer margin and whitened along with it."""
    h, w = image_bgr.shape[:2]
    cx, cy, radius = plate_circle["cx"], plate_circle["cy"], plate_circle["radius"]

    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), radius + margin_px, 255, -1)
    inv_mask = cv2.bitwise_not(mask)

    plate_only = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    white_bg = np.full_like(image_bgr, 255)
    white_part = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
    composed = cv2.add(plate_only, white_part)

    canvas = np.full((canvas_size, canvas_size, 3), 255, np.uint8)
    avail = canvas_size - title_band_px
    y0 = title_band_px + max(0, (avail - h) // 2)
    x0 = max(0, (canvas_size - w) // 2)
    canvas[y0:y0 + h, x0:x0 + w] = composed
    return canvas


def _title(canvas: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    out = canvas.copy()
    cv2.putText(out, title, (MARGIN_PX, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                TITLE_COLOR, 3, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (MARGIN_PX, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    SUBTITLE_COLOR, 1, cv2.LINE_AA)
    return out


def _to_bgr(gray_or_bgr: np.ndarray) -> np.ndarray:
    if gray_or_bgr.ndim == 2:
        return cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    return gray_or_bgr


def _contrast_stretch(gray: np.ndarray, plate_circle: Dict) -> np.ndarray:
    """The background-subtracted diff is only a few intensity units above
    zero even at colony pixels (the pipeline thresholds it at
    diff_threshold=3 out of 255) — displayed raw it reads as solid black
    with barely-visible specks. Stretches contrast using only the plate's
    own interior pixel range so the flattened background and colonies are
    both legible, without changing any value the pipeline itself acts on
    (this only touches a display copy in this export script)."""
    cx, cy, radius = plate_circle["cx"], plate_circle["cy"], plate_circle["inner_radius"]
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    inside = gray[mask > 0]
    lo, hi = float(inside.min()), float(inside.max())
    if hi <= lo:
        return gray
    stretched = np.clip((gray.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    return stretched.astype(np.uint8)


def _sampled_indices(n: int, k: int) -> List[int]:
    if n <= k:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def _fits_in_circle(tx: int, ty: int, tw: int, th: int, cx: int, cy: int, r: int) -> bool:
    """Whether the text box's 4 corners all lie within radius r of (cx, cy) —
    used to keep labels inside the area _compose_on_white actually keeps
    (radius + margin_px), which is smaller than the raw image bounds and
    would otherwise silently white out any label that overhangs it."""
    corners = [(tx - 2, ty - th - 2), (tx + tw + 2, ty - th - 2),
               (tx - 2, ty + 4), (tx + tw + 2, ty + 4)]
    return all((x - cx) ** 2 + (y - cy) ** 2 <= r ** 2 for x, y in corners)


def _place_label(pcx: int, pcy: int, tw: int, th: int, plate_circle: Dict) -> Tuple[int, int]:
    """Tries the label to the upper-right of the colony first (the common
    case), then falls back to the other three quadrants, then to whichever
    candidate overhangs the visible-area boundary least."""
    cx, cy = plate_circle["cx"], plate_circle["cy"]
    r = plate_circle["radius"] + MARGIN_PX - 6
    candidates = [(pcx + 8, pcy - 8), (pcx - tw - 8, pcy - 8),
                  (pcx + 8, pcy + th + 8), (pcx - tw - 8, pcy + th + 8)]
    for tx, ty in candidates:
        if _fits_in_circle(tx, ty, tw, th, cx, cy, r):
            return tx, ty

    def overhang(pt):
        tx, ty = pt
        corners = [(tx - 2, ty - th - 2), (tx + tw + 2, ty + 4)]
        return max(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in corners)

    return min(candidates, key=overhang)


def _draw_contour_frame(original: np.ndarray, valid_contours: List[np.ndarray],
                         contour_info: List[Dict], plate_circle: Dict,
                         colored_by_anomaly: bool, annotate_metrics: bool) -> np.ndarray:
    out = original.copy()
    sample = set(_sampled_indices(len(valid_contours), MAX_ANNOTATED_COLONIES))

    for i, (contour, info) in enumerate(zip(valid_contours, contour_info)):
        if colored_by_anomaly:
            color = (0, 0, 220) if info["anomaly_flags"] else (0, 160, 0)
        else:
            color = (200, 120, 0)
        cv2.drawContours(out, [contour], -1, color, 2)

        if annotate_metrics and i in sample:
            pcx, pcy = info["centroid"]
            label = (f"A:{info['area_mm2']:.1f} C:{info['circularity']:.2f} "
                     f"AR:{info['aspect_ratio']:.1f}")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            tx, ty = _place_label(pcx, pcy, tw, th, plate_circle)
            cv2.rectangle(out, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 4),
                          (255, 255, 255), -1)
            cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (30, 30, 30), 1, cv2.LINE_AA)
    return out


def export_plate_frames(image_path: str, out_dir: Path, plate_label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = quantify_colonies(image_path, return_intermediates=True)
    inter = result["intermediates"]
    plate_circle = result["plate_circle"]

    frames = [
        ("01_raw_input.png", inter["original"],
         "Input: Synthetic Plate (Held-Out Set)", plate_label),
        ("02_illumination_corrected.png",
         _to_bgr(_contrast_stretch(inter["illumination_corrected"], plate_circle)),
         "After Illumination Correction", ""),
        ("03_binary_threshold.png", _to_bgr(inter["cleaned"]),
         "Binary Threshold: Colonies Isolated", ""),
        ("04_watershed_labeled.png", label_color_image(inter["watershed_markers"]),
         f"Watershed Segmentation: {result['count']} colonies detected", ""),
        ("05_contour_overlay.png",
         _draw_contour_frame(inter["original"], inter["valid_contours"], result["contours"],
                              plate_circle, colored_by_anomaly=False, annotate_metrics=True),
         "Morphometric Contour Analysis", ""),
        ("06_anomaly_flagged.png",
         _draw_contour_frame(inter["original"], inter["valid_contours"], result["contours"],
                              plate_circle, colored_by_anomaly=True, annotate_metrics=True),
         f"Anomaly Detection: {result['anomaly_count']} flagged", ""),
    ]

    for filename, image, title, subtitle in frames:
        canvas = _compose_on_white(image, plate_circle)
        canvas = _title(canvas, title, subtitle)
        cv2.imwrite(str(out_dir / filename), canvas)


def load_manifest(plates_dir: Path = PLATES_DIR_DEFAULT) -> Dict:
    return json.loads((plates_dir / "manifest.json").read_text())


def _all_plates(manifest: Dict) -> List[Dict]:
    plates = []
    for set_name, rows in manifest["sets"].items():
        for i, row in enumerate(rows):
            gt = json.loads(Path(row["ground_truth_path"]).read_text())
            plates.append({
                "set": set_name, "stem": Path(row["image_path"]).stem, "row": row, "gt": gt,
                "expected_count": gt["expected_count"],
                "expected_anomaly_count": gt["expected_anomaly_count"],
                "touching_pairs": gt["touching_pairs"],
            })
    return plates


def select_representative_plates(plates_dir: Path = PLATES_DIR_DEFAULT) -> List[Dict]:
    """Picks 5 plates spanning difficulty tiers, each chosen by an explicit,
    checkable rule against the held-out plates' own ground truth (never by
    eyeballing images) — see each tier's comment for the exact rule and
    testing/blind_eval/iteration_log.md for why touching_pairs is 0 outside
    dense_pack/high_touching (only those two sets stress that axis)."""
    manifest = load_manifest(plates_dir)
    plates = _all_plates(manifest)

    # Easy: sparsest plate among the two sets that stress neither density
    # nor touching (size_variance, irregular_morphology).
    easy = min((p for p in plates if p["set"] in ("size_variance", "irregular_morphology")),
               key=lambda p: p["expected_count"])
    easy["tier"] = "easy"
    easy["tier_label"] = "Easy: Sparse, Normal Illumination"

    # Medium: high_touching plate at the LOW end of its own touching_pairs
    # range (5-21 observed) — touching stress present, but mild.
    medium = min((p for p in plates if p["set"] == "high_touching"),
                 key=lambda p: p["touching_pairs"])
    medium["tier"] = "medium"
    medium["tier_label"] = "Medium: Moderate Density, Light Touching"

    # Hard: dense_pack plate nearest the MIDDLE of its held-out density
    # range (80-120) — reserves the range's extreme for worst-case below.
    hard = min((p for p in plates if p["set"] == "dense_pack"),
               key=lambda p: abs(p["expected_count"] - 100))
    hard["tier"] = "hard"
    hard["tier_label"] = "Hard: Dense Colony Packing"

    # High-anomaly: irregular_morphology plate with the most flagged
    # ground-truth anomalies.
    high_anomaly = max((p for p in plates if p["set"] == "irregular_morphology"),
                        key=lambda p: p["expected_anomaly_count"])
    high_anomaly["tier"] = "high_anomaly"
    high_anomaly["tier_label"] = "High-Anomaly: Irregular Morphology"

    # Worst-case: not a parametric extreme but an EMPIRICAL one — the single
    # lowest-F1 plate across the two held-out sets that actually fail this
    # validation's target (dense_pack, high_touching). This is the plate
    # where the pipeline does worst, not just where the generator pushed
    # hardest, which is the more honest "worst case" for a memo to show.
    worst_pool = [p for p in plates if p["set"] in ("dense_pack", "high_touching")]
    for p in worst_pool:
        result = quantify_colonies(p["row"]["image_path"])
        p["f1"] = score_plate(result, p["gt"])["f1"]
    worst = min(worst_pool, key=lambda p: p["f1"])
    worst["tier"] = "worst_case"
    worst["tier_label"] = f"Worst Case: Lowest-F1 Held-Out Plate (F1={worst['f1']:.3f})"

    return [easy, medium, hard, high_anomaly, worst]


def main(plates_dir: Path = PLATES_DIR_DEFAULT, output_dir: Path = OUTPUT_DIR_DEFAULT) -> None:
    selected = select_representative_plates(plates_dir)
    summary_lines = ["# Stage-by-Stage Visualization Export", "",
                      "5 representative plates across difficulty tiers, each selected by an "
                      "explicit rule against ground truth (see export_stage_visualizations.py's "
                      "select_representative_plates() for the exact criteria per tier).", "",
                      "| tier | plate | expected count | anomalies | touching pairs |",
                      "|---|---|---|---|---|"]

    for p in selected:
        plate_dir_name = f"{p['tier']}_{p['set']}_{p['stem']}"
        summary_lines.append(f"| {p['tier_label']} | `{p['set']}/{p['stem']}` | "
                              f"{p['expected_count']} | {p['expected_anomaly_count']} | "
                              f"{p['touching_pairs']} |")
        export_plate_frames(p["row"]["image_path"], output_dir / plate_dir_name, p["tier_label"])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
