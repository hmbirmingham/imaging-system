"""
build_memo_figures.py — regenerate the blind-validation memo's figures from
the current pipeline.

The memo's figures are pipeline output, so they go stale whenever the
pipeline changes. This regenerates all three from scratch:

  fig1_1..fig1_6.png  the six pipeline stages on the Figure 1 plate
  fig2_f1_by_set.png  pooled F1 per held-out set against the target line
  fig3_a/b/c.png      what the worst plate reported, what it missed, and
                      the same failure on a denser plate

Panels are rendered without baked-in titles — captions belong to the memo
layout, not the image — unlike testing/thesis_export/export_stage_
visualizations.py, whose frames are standalone and carry their own.

Numbers are read from testing/blind_eval/results/blind_eval_results.json,
never hardcoded, so a figure can't disagree with the results table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantify import quantify_colonies
from testing.blind_eval.export_failure_plates import label_color_image
from testing.blind_eval.run_blind_eval import match_detections_to_ground_truth, score_plate
from testing.thesis_export.export_stage_visualizations import _contrast_stretch

PLATES_DIR = Path("testing/blind_eval/plates")
RESULTS = Path("testing/blind_eval/results/blind_eval_results.json")

# The memo walks through the run's hardest plate and brings the same plate
# back in the results. That plate changed: the contour-extraction fix lifted
# the previous worst (plate_039) from F1 0.513 to 0.863, so the current
# lowest-scoring plate — and the one the walkthrough should use — is
# plate_020. Selected empirically by run_blind_eval, not fixed by hand.
FIGURE1_PLATE = "high_touching/plate_020"
PREVIOUS_FIGURE1_PLATE = "high_touching/plate_039"
FIGURE3_DENSE_PLATE = "dense_pack/plate_021"

PANEL_PX = 760          # square panel, plate centred
INK = "#1d1d1f"
MUTED = "#6e6e73"
TEAL = "#0f766e"
ORANGE = "#c2410c"


def _crop_to_plate(image: np.ndarray, plate_circle: Dict, pad_frac: float = 0.06,
                   size: int = PANEL_PX, on_black: bool = False) -> np.ndarray:
    """Square crop centred on the detected plate, padded slightly, resized.
    Keeps every panel the same scale so the six stages line up visually.

    on_black=False also masks everything outside the plate rim to white, so
    the panel reads as a clean dish on the page rather than a circle sitting
    in the generator's black surround. The binary/watershed stages are
    inherently black-background views and keep theirs (on_black=True)."""
    cx, cy, radius = plate_circle["cx"], plate_circle["cy"], plate_circle["radius"]
    half = int(radius * (1 + pad_frac))
    h, w = image.shape[:2]
    fill = 0 if on_black else 255
    canvas = np.full((half * 2, half * 2, 3), fill, np.uint8)

    x0, y0 = cx - half, cy - half
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, cx + half), min(h, cy + half)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]

    if not on_black:
        mask = np.zeros(canvas.shape[:2], np.uint8)
        cv2.circle(mask, (half, half), radius, 255, -1)
        white = np.full_like(canvas, 255)
        canvas = np.where(mask[:, :, None] > 0, canvas, white)

    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def _draw_contours(base: np.ndarray, contours: List[np.ndarray],
                   info: List[Dict], anomaly_colour: bool) -> np.ndarray:
    out = base.copy()
    for contour, meta in zip(contours, info):
        if anomaly_colour:
            colour = (0, 0, 220) if meta["anomaly_flags"] else (0, 150, 0)
        else:
            colour = (190, 110, 0)
        cv2.drawContours(out, [contour], -1, colour, 3)
    return out


def figure1(out_dir: Path) -> Dict:
    """Six stages on the Figure 1 plate, same crop and scale throughout."""
    img_path = str(PLATES_DIR / f"{FIGURE1_PLATE}.png")
    result = quantify_colonies(img_path, return_intermediates=True)
    inter, circle = result["intermediates"], result["plate_circle"]

    stages = [
        ("fig1_1", inter["original"], False),
        ("fig1_2", cv2.cvtColor(_contrast_stretch(inter["illumination_corrected"], circle),
                                cv2.COLOR_GRAY2BGR), True),
        ("fig1_3", cv2.cvtColor(inter["cleaned"], cv2.COLOR_GRAY2BGR), True),
        ("fig1_4", label_color_image(inter["watershed_markers"]), True),
        ("fig1_5", _draw_contours(inter["original"], inter["valid_contours"],
                                  result["contours"], anomaly_colour=False), False),
        ("fig1_6", _draw_contours(inter["original"], inter["valid_contours"],
                                  result["contours"], anomaly_colour=True), False),
    ]
    for name, image, on_black in stages:
        cv2.imwrite(str(out_dir / f"{name}.png"),
                    _crop_to_plate(image, circle, on_black=on_black))

    gt = json.loads((PLATES_DIR / f"{FIGURE1_PLATE}.json").read_text())
    return {"count": result["count"], "anomalies": result["anomaly_count"],
            "expected": gt["expected_count"], **score_plate(result, gt)}


def figure2(out_dir: Path) -> None:
    """Pooled F1 per set, ordered best to worst, against the 0.90 target."""
    report = json.loads(RESULTS.read_text())
    target = report["target_f1"]
    sets = sorted(report["by_set"].items(), key=lambda kv: kv[1]["f1"])

    varies = {"poor_illumination": "varies: illumination",
              "irregular_morphology": "varies: morphology",
              "size_variance": "varies: size variance",
              "dense_pack": "varies: density",
              "high_touching": "varies: touching rate"}

    fig, ax = plt.subplots(figsize=(9.2, 3.5), dpi=200)
    for i, (name, s) in enumerate(sets):
        passing = s["f1"] >= target
        colour = TEAL if passing else ORANGE
        ax.plot([0.70, s["f1"]], [i, i], color="#d7d7db", lw=1.4, zorder=1,
                solid_capstyle="round")
        ax.scatter([s["f1"]], [i], s=90, color=colour, zorder=3)
        ax.text(s["f1"] + 0.008, i, f'{s["f1"]:.3f}', va="center", ha="left",
                fontsize=10.5, fontweight="bold", color=INK)

    ax.axvline(target, ls=(0, (4, 3)), color="#8e8e93", lw=1.3, zorder=2)
    ax.text(target, len(sets) - 0.33, f"target {target:.2f}", ha="center",
            va="bottom", fontsize=9.5, color=MUTED)

    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels([n for n, _ in sets], fontsize=11, fontfamily="monospace")
    for i, (name, _) in enumerate(sets):
        ax.text(-0.004, i - 0.42, varies.get(name, ""), transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8, color=MUTED)

    ax.text(0.845, -0.95, "BELOW TARGET", fontsize=8.5, fontweight="bold",
            color=ORANGE, ha="center")
    ax.text(0.955, -0.95, "PASS", fontsize=8.5, fontweight="bold",
            color=TEAL, ha="center")

    ax.set_xlim(0.70, 1.015)
    ax.set_ylim(-1.35, len(sets) - 0.2)
    ax.set_xlabel("Pooled F1 (50 plates per set)", fontsize=9.5, color=MUTED)
    ax.tick_params(axis="x", labelsize=9, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d7d7db")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_f1_by_set.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def _missed_overlay(img_path: str) -> Tuple[np.ndarray, Dict, Dict]:
    """Watershed regions coloured, ground-truth colonies the pipeline missed
    circled in red — the same diagnostic view used during iteration."""
    result = quantify_colonies(img_path, return_intermediates=True)
    gt = json.loads(Path(img_path.replace(".png", ".json")).read_text())
    _, _, unmatched = match_detections_to_ground_truth(result["contours"], gt["colonies"])

    vis = label_color_image(result["intermediates"]["watershed_markers"])
    for colony in unmatched:
        cv2.circle(vis, (int(colony["cx"]), int(colony["cy"])),
                   int(colony["radius_px"]) + 5, (0, 0, 255), 3)
    return vis, result, score_plate(result, gt)


def figure3(out_dir: Path) -> Dict:
    worst_path = str(PLATES_DIR / f"{FIGURE1_PLATE}.png")
    result = quantify_colonies(worst_path, return_intermediates=True)
    circle = result["plate_circle"]

    reported = _draw_contours(result["intermediates"]["original"],
                              result["intermediates"]["valid_contours"],
                              result["contours"], anomaly_colour=True)
    cv2.imwrite(str(out_dir / "fig3_a.png"), _crop_to_plate(reported, circle))

    missed, _, worst_score = _missed_overlay(worst_path)
    cv2.imwrite(str(out_dir / "fig3_b.png"),
                _crop_to_plate(missed, circle, on_black=True))

    dense_path = str(PLATES_DIR / f"{FIGURE3_DENSE_PLATE}.png")
    dense_vis, dense_result, dense_score = _missed_overlay(dense_path)
    cv2.imwrite(str(out_dir / "fig3_c.png"),
                _crop_to_plate(dense_vis, dense_result["plate_circle"], on_black=True))

    return {"worst": worst_score, "dense": dense_score,
            "worst_anomalies": result["anomaly_count"]}


def main(out_dir: Path = Path("testing/thesis_export/generated/memo")) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"figure1": figure1(out_dir)}
    figure2(out_dir)
    stats["figure3"] = figure3(out_dir)
    (out_dir / "figure_stats.json").write_text(json.dumps(stats, indent=2, default=float))
    return stats


if __name__ == "__main__":
    s = main()
    f1, f3 = s["figure1"], s["figure3"]
    print(f"Figure 1 plate ({FIGURE1_PLATE}): {f1['count']} counted / "
          f"{f1['expected']} expected, {f1['anomalies']} flagged, F1 {f1['f1']:.3f}")
    print(f"Figure 3 worst: F1 {f3['worst']['f1']:.3f} recall {f3['worst']['recall']:.3f} "
          f"precision {f3['worst']['precision']:.3f}")
    print(f"Figure 3 dense: {f3['dense']['detected_count']}/{f3['dense']['expected_count']} "
          f"detected, F1 {f3['dense']['f1']:.3f}")
