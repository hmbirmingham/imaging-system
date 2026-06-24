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
from typing import Dict, Optional, Union


LOG_DIR  = Path("data")
LOG_FILE = LOG_DIR / "colony_features.csv"

# All columns written per row
COLUMNS = [
    # Run metadata
    "timestamp",
    "image_path",
    "plate_type",         # culture medium code (BAP, MAC, …) — default "unknown"
    "profile_id",         # organism profile id — default "unknown"
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
    # Validation (filled by the in-app PhD validation workflow)
    "validation_status",  # confirmed | false_positive | added | "" (unreviewed)
    "validated_by",       # who signed off
]


class DataLogger:
    """Appends colony feature rows to a persistent CSV for ML training."""

    # Columns whose "absent" default is "unknown" rather than "" when migrating
    # an older CSV that predates them.
    _DEFAULT_UNKNOWN = {"plate_type", "profile_id"}

    def __init__(self, log_file: Union[str, Path] = LOG_FILE):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """
        Create the CSV with the current header, or migrate an existing one that
        predates newer columns (e.g. plate_type/profile_id) by rewriting it with
        the full header — existing rows get sensible defaults. Append-safe.
        """
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writeheader()
            return

        with open(self.log_file, newline="") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header == COLUMNS:
            return   # already current

        # Migrate: re-read rows under the old header, rewrite under the new one.
        with open(self.log_file, newline="") as f:
            old_rows = list(csv.DictReader(f))
        with open(self.log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for r in old_rows:
                writer.writerow({
                    c: r.get(c, "unknown" if c in self._DEFAULT_UNKNOWN else "")
                    for c in COLUMNS
                })

    def log(self,
            result: Dict,
            manual_count: Optional[int] = None,
            plate_type: str = "unknown",
            profile_id: str = "unknown") -> int:
        """
        Append per-colony rows from a quantify_colonies() result.

        Parameters
        ----------
        result       : dict returned by quantify_colonies()
        manual_count : human-verified count for this plate (optional)
        plate_type   : culture medium code (BAP, MAC, …); default "unknown"
        profile_id   : organism profile id; default "unknown"

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
                "plate_type":         plate_type or "unknown",
                "profile_id":         profile_id or "unknown",
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

    def apply_validation(self,
                         image_path: str,
                         labels: Dict[int, Dict],
                         added: Optional[list] = None,
                         manual_count: Optional[int] = None,
                         plate_type: Optional[str] = None,
                         profile_id: Optional[str] = None,
                         validated_by: str = "") -> Dict:
        """
        Fill ground-truth labels for a previously-logged image. This is what the
        PhD validation workflow writes: it turns the always-blank is_anomaly
        column into real training labels.

        Parameters
        ----------
        image_path   : the result['input_path'] used when the run was logged.
        labels       : {colony_id: {"is_anomaly": 0|1,
                                     "status": "confirmed"|"false_positive"}}
        added        : list of {"centroid_x","centroid_y","is_anomaly"} the human
                       added (colonies the detector missed) — appended as new rows
                       with status "added".
        manual_count : human-verified colony count for the plate.
        validated_by : signer name.

        Returns {"updated": n, "added": m}.
        """
        if not self.log_file.exists():
            return {"updated": 0, "added": 0}

        with open(self.log_file, newline="") as f:
            rows = list(csv.DictReader(f))

        updated = 0
        max_cid = 0
        for r in rows:
            if r.get("image_path") != image_path:
                continue
            try:
                cid = int(r.get("colony_id") or 0)
            except ValueError:
                cid = 0
            max_cid = max(max_cid, cid)
            if cid in labels:
                lab = labels[cid]
                r["is_anomaly"]        = lab.get("is_anomaly", "")
                r["validation_status"] = lab.get("status", "confirmed")
                r["validated_by"]      = validated_by
                if manual_count is not None:
                    r["manual_count"] = manual_count
                if plate_type:
                    r["plate_type"] = plate_type
                if profile_id:
                    r["profile_id"] = profile_id
                updated += 1

        new_rows = []
        for i, a in enumerate(added or [], start=1):
            blank = {c: "" for c in COLUMNS}
            blank.update({
                "timestamp":         datetime.now().isoformat(timespec="seconds"),
                "image_path":        image_path,
                "plate_type":        plate_type or "unknown",
                "profile_id":        profile_id or "unknown",
                "manual_count":      manual_count if manual_count is not None else "",
                "colony_id":         max_cid + i,
                "centroid_x":        a.get("centroid_x", ""),
                "centroid_y":        a.get("centroid_y", ""),
                "is_anomaly":        a.get("is_anomaly", ""),
                "validation_status": "added",
                "validated_by":      validated_by,
            })
            new_rows.append(blank)

        with open(self.log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows + new_rows)

        return {"updated": updated, "added": len(new_rows)}

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
