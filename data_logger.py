"""
data_logger.py — Passive training data collector.

Every time quantify_colonies() is called through the GUI, this module
appends per-colony features to a CSV file. Over time this builds the
labelled dataset needed to train the ML anomaly layer.

The CSV contains one row per colony. An 'is_anomaly' column defaults to
None (unlabelled). A separate labelling script or manual review can fill
this in to create the supervised training set.

Usage
-----
    from data_logger import DataLogger
    logger = DataLogger()
    logger.log(quantify_result, manual_count=84)  # called after each run
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


LOG_DIR  = Path("data")
LOG_FILE = LOG_DIR / "colony_features.csv"

# All columns written per row
COLUMNS = [
    # Run metadata
    "timestamp",
    "image_path",
    "manual_count",       # filled in if user provides it
    "auto_count",
    "px_per_mm",
    "plate_cx", "plate_cy", "plate_radius", "plate_inner_radius",
    # Per-colony geometry
    "colony_id",
    "area_px",
    "area_mm2",
    "circularity",
    "aspect_ratio",
    "perimeter_px",
    "equiv_radius_px",
    "centroid_x", "centroid_y",
    # Colour
    "r_mean", "g_mean", "b_mean",
    "r_std",  "g_std",  "b_std",
    # Texture / haemolysis
    "texture_contrast",
    "hemolysis_delta",
    # Anomaly labels
    "stat_flags",
    "stat_score",
    "ml_anomaly",
    "ml_score",
    "is_anomaly",         # ground truth label — None until manually confirmed
]


class DataLogger:
    """Appends colony feature rows to a persistent CSV for ML training."""

    def __init__(self, log_file: str | Path = LOG_FILE):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    def log(self,
            result: Dict,
            manual_count: Optional[int] = None) -> int:
        """
        Append per-colony rows from a quantify_colonies() result.

        Parameters
        ----------
        result       : dict returned by quantify_colonies()
        manual_count : human-verified count for this plate (optional)

        Returns
        -------
        Number of rows written.
        """
        timestamp = datetime.now().isoformat(timespec="seconds")
        pc        = result.get("plate_circle", {})
        colonies  = result.get("contours", [])

        rows = []
        for i, colony in enumerate(colonies, start=1):
            cx, cy = colony.get("centroid", (0, 0))
            row = {
                "timestamp":          timestamp,
                "image_path":         result.get("input_path", ""),
                "manual_count":       manual_count if manual_count is not None else "",
                "auto_count":         result.get("count", ""),
                "px_per_mm":          round(result.get("px_per_mm", 0), 4),
                "plate_cx":           pc.get("cx", ""),
                "plate_cy":           pc.get("cy", ""),
                "plate_radius":       pc.get("radius", ""),
                "plate_inner_radius": pc.get("inner_radius", ""),
                "colony_id":          i,
                "area_px":            round(colony.get("area_px", 0), 2),
                "area_mm2":           round(colony.get("area_mm2", 0), 4),
                "circularity":        round(colony.get("circularity", 0), 4),
                "aspect_ratio":       round(colony.get("aspect_ratio", 0), 4),
                "perimeter_px":       round(colony.get("perimeter_px", 0), 2),
                "equiv_radius_px":    round(colony.get("equiv_radius_px", 0), 2),
                "centroid_x":         cx,
                "centroid_y":         cy,
                "r_mean":             round(colony.get("r_mean", 0), 2),
                "g_mean":             round(colony.get("g_mean", 0), 2),
                "b_mean":             round(colony.get("b_mean", 0), 2),
                "r_std":              round(colony.get("r_std", 0), 2),
                "g_std":              round(colony.get("g_std", 0), 2),
                "b_std":              round(colony.get("b_std", 0), 2),
                "texture_contrast":   round(colony.get("texture_contrast", 0), 2),
                "hemolysis_delta":    round(colony.get("hemolysis_delta", 0), 2),
                "stat_flags":         "|".join(colony.get("stat_flags", [])),
                "stat_score":         colony.get("stat_score", ""),
                "ml_anomaly":         colony.get("ml_anomaly", ""),
                "ml_score":           colony.get("ml_score", ""),
                "is_anomaly":         "",  # unlabelled until reviewed
            }
            rows.append(row)

        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writerows(rows)

        return len(rows)

    def summary(self) -> Dict:
        """Return basic stats about the current dataset."""
        if not self.log_file.exists():
            return {"rows": 0, "labelled": 0, "file": str(self.log_file)}

        rows = labelled = anomalies = 0
        with open(self.log_file, newline="") as f:
            for row in csv.DictReader(f):
                rows += 1
                if row["is_anomaly"] != "":
                    labelled += 1
                    if row["is_anomaly"] == "1":
                        anomalies += 1

        return {
            "rows":             rows,
            "labelled":         labelled,
            "unlabelled":       rows - labelled,
            "labelled_anomalies": anomalies,
            "file":             str(self.log_file),
            "ready_to_train":   labelled >= 50,  # rough minimum for RF
        }


# ── CLI summary ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger = DataLogger()
    stats  = logger.summary()
    print("Dataset summary")
    print("───────────────")
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")
    if not stats["ready_to_train"]:
        needed = max(0, 50 - stats["labelled"])
        print(f"\n  Need {needed} more labelled colonies before ML training.")
