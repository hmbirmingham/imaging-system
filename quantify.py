"""
quantify.py — Backlit agar plate colony detection and feature extraction.

Pipeline
--------
1. Detect plate boundary (Hough circle).
2. Model + subtract background illumination gradient.
3. Threshold → morphological cleanup.
4. Watershed segmentation to split touching colonies.
5. Filter contours by area, circularity, and aspect ratio.
6. Extract per-colony features (geometry + colour + texture).
7. Statistical anomaly flagging across the plate.
8. Annotate and optionally save output image.

Calibration
-----------
All areas are returned in mm² using the detected plate radius and the known
physical inner radius (PLATE_INNER_RADIUS_MM). This makes measurements
distance-invariant — useful when imaging height is inconsistent.
"""

import os
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

# ── Physical constants ────────────────────────────────────────────────────────
# Standard 90-mm petri dish usable inner radius after rim exclusion (mm).
PLATE_INNER_RADIUS_MM = 40.0


# ── Plate detection ───────────────────────────────────────────────────────────

def detect_plate_circle(gray: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Detect the agar plate as the dominant circle using Hough transform.
    Returns (cx, cy, radius_px) or None.
    """
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    h, w = gray.shape
    min_r = int(min(h, w) * 0.25)
    max_r = int(min(h, w) * 0.65)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) * 0.4,
        param1=60,
        param2=40,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    best = max(circles, key=lambda c: c[2])
    return int(best[0]), int(best[1]), int(best[2])


# ── Feature helpers ───────────────────────────────────────────────────────────

def _circularity(area: float, perimeter: float) -> float:
    return (4 * math.pi * area / perimeter ** 2) if perimeter > 0 else 0.0


def _aspect_ratio(contour: np.ndarray) -> float:
    _, (w, h), _ = cv2.minAreaRect(contour)
    if h == 0:
        return 1.0
    return float(max(w, h) / min(w, h)) if min(w, h) > 0 else 1.0


def _colour_features(image_bgr: np.ndarray, mask: np.ndarray) -> Dict:
    """Mean and std of each BGR channel within the colony mask."""
    feats = {}
    for i, ch in enumerate(["b", "g", "r"]):
        vals = image_bgr[:, :, i][mask > 0].astype(float)
        feats[f"{ch}_mean"] = float(np.mean(vals)) if vals.size else 0.0
        feats[f"{ch}_std"]  = float(np.std(vals))  if vals.size else 0.0
    return feats


def _texture_contrast(gray: np.ndarray, mask: np.ndarray) -> float:
    """Simple local contrast (std of pixel intensities) as a texture proxy."""
    vals = gray[mask > 0].astype(float)
    return float(np.std(vals)) if vals.size else 0.0


def _hemolysis_zone(gray: np.ndarray, cx: int, cy: int,
                    colony_r: float, scale: float = 2.5) -> float:
    """
    Estimate hemolysis zone by measuring mean brightness in an annular ring
    around the colony. Under backlight, beta-haemolysis creates a brighter
    (clearer) halo. Returns mean brightness of the halo relative to background.
    Higher = more likely haemolysis.
    """
    h, w = gray.shape
    inner = int(colony_r)
    outer = int(colony_r * scale)

    y0 = max(0, cy - outer)
    y1 = min(h, cy + outer)
    x0 = max(0, cx - outer)
    x1 = min(w, cx + outer)

    roi = gray[y0:y1, x0:x1].astype(float)
    if roi.size == 0:
        return 0.0

    ry, rx = np.ogrid[:roi.shape[0], :roi.shape[1]]
    dist = np.sqrt((rx - (cx - x0)) ** 2 + (ry - (cy - y0)) ** 2)

    annulus = (dist >= inner) & (dist <= outer)
    bg      = dist > outer * 1.1

    halo_mean = float(np.mean(roi[annulus])) if annulus.any() else 0.0
    bg_mean   = float(np.mean(roi[bg]))      if bg.any()      else halo_mean

    return halo_mean - bg_mean  # positive = lighter halo = haemolysis candidate


# ── Statistical anomaly scoring ───────────────────────────────────────────────

def _flag_anomalies(contour_info: List[Dict],
                    z_thresh: float = 2.5) -> List[Dict]:
    """
    Flag per-colony anomalies using Z-score on key features.
    Preserves any flags already set (e.g. touching_colony from watershed).
    Adds/updates 'anomaly_flags' and 'anomaly_score' on each colony dict.
    """
    if len(contour_info) < 3:
        for c in contour_info:
            c.setdefault("anomaly_flags", [])
            c["anomaly_score"] = 0.0
        return contour_info

    def z_scores(key):
        vals = np.array([c[key] for c in contour_info])
        std  = vals.std()
        if std == 0:
            return np.zeros(len(vals))
        return np.abs((vals - vals.mean()) / std)

    area_z    = z_scores("area_mm2")
    circ_z    = z_scores("circularity")
    aspect_z  = z_scores("aspect_ratio")
    texture_z = z_scores("texture_contrast")
    # Combined colour anomaly: mean Z across RGB channels
    colour_z  = (z_scores("r_mean") + z_scores("g_mean") + z_scores("b_mean")) / 3.0

    for i, colony in enumerate(contour_info):
        # Preserve pre-existing flags (e.g. touching_colony)
        flags = list(colony.get("anomaly_flags", []))

        if area_z[i]    > z_thresh: flags.append("unusual_size")
        if circ_z[i]    > z_thresh: flags.append("unusual_shape")
        if aspect_z[i]  > z_thresh: flags.append("elongated")
        if texture_z[i] > z_thresh: flags.append("texture_anomaly")
        if colour_z[i]  > z_thresh: flags.append("abnormal_colour")
        if colony["circularity"] < 0.4:
            flags.append("non_circular")
        if colony["aspect_ratio"] > 3.0:
            flags.append("streak_or_artifact")
        if colony["hemolysis_delta"] > 15:
            flags.append("hemolysis_candidate")

        # Deduplicate, preserving order
        colony["anomaly_flags"] = list(dict.fromkeys(flags))
        colony["anomaly_score"] = round(
            (area_z[i] + circ_z[i] + aspect_z[i]) / 3.0, 3)

    return contour_info


# ── Main quantification ───────────────────────────────────────────────────────

def quantify_colonies(
    image_path: str,
    output_path: Optional[str] = None,
    # detection
    min_area_mm2: float = 0.1,
    max_area_mm2: float = 20.0,
    min_circularity: float = 0.25,
    max_aspect_ratio: float = 6.0,
    # background subtraction
    bg_blur_kernel: int = 151,
    diff_threshold: int = 3,
    # plate geometry — rim expressed in mm so it scales with any image resolution
    rim_shrink_mm: float = 3.0,
    plate_inner_radius_mm: float = PLATE_INNER_RADIUS_MM,
    # anomaly
    anomaly_z_thresh: float = 2.5,
) -> Dict:
    """
    Quantify bacterial colonies in a backlit agar plate image.

    Returns
    -------
    dict with keys:
      input_path, output_path, count, px_per_mm,
      plate_circle, contours, summary_stats, anomaly_count
    """
    if bg_blur_kernel % 2 == 0:
        raise ValueError("bg_blur_kernel must be odd.")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── 1. Plate detection & calibration ────────────────────────────────────
    plate = detect_plate_circle(gray)
    if plate is not None:
        cx, cy, radius = plate
        # Rough px_per_mm from raw radius to convert rim_shrink to pixels
        rough_ppm = radius / (plate_inner_radius_mm + rim_shrink_mm)
        rim_shrink_px = int(rim_shrink_mm * rough_ppm)
        inner_radius  = max(0, radius - rim_shrink_px)
    else:
        print("Warning: plate not detected — using full image.")
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        radius = inner_radius = min(h, w) // 2

    # Precise px_per_mm calibrated to the usable inner plate area
    px_per_mm = inner_radius / plate_inner_radius_mm if inner_radius > 0 else 1.0
    min_area_px = min_area_mm2 * px_per_mm ** 2
    max_area_px = max_area_mm2 * px_per_mm ** 2

    plate_mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(plate_mask, (cx, cy), inner_radius, 255, -1)

    # ── 2. Background subtraction ────────────────────────────────────────────
    # Heavy Gaussian blur models the slow backlight/hotspot gradient.
    # background - original → dark colonies become bright, gradient cancels out.
    bg_model    = cv2.GaussianBlur(gray, (bg_blur_kernel, bg_blur_kernel), 0)
    diff        = cv2.subtract(bg_model, gray)
    diff_masked = cv2.bitwise_and(diff, plate_mask)

    # ── 3. Threshold ─────────────────────────────────────────────────────────
    _, binary = cv2.threshold(diff_masked, diff_threshold, 255, cv2.THRESH_BINARY)

    # ── 4. Morphological cleanup ─────────────────────────────────────────────
    k       = np.ones((3, 3), np.uint8)
    opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=1)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=2)

    # ── 5. Watershed segmentation (split touching colonies) ──────────────────
    # Label connected components BEFORE watershed — used later to identify
    # colonies that were touching in the raw binary image.
    _, pre_watershed_labels = cv2.connectedComponents(cleaned)

    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg    = np.uint8(sure_fg)
    sure_bg    = cv2.dilate(cleaned, k, iterations=3)
    unknown    = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers + 1
    markers[unknown == 255] = 0

    ws_image = image.copy()
    markers  = cv2.watershed(ws_image, markers)
    watershed_mask = np.zeros_like(cleaned)
    watershed_mask[markers > 1] = 255

    # ── 6. Find & filter contours ────────────────────────────────────────────
    contours, _ = cv2.findContours(
        watershed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours: List[np.ndarray] = []
    contour_info:   List[Dict]       = []

    for contour in contours:
        area_px = cv2.contourArea(contour)
        if not (min_area_px <= area_px <= max_area_px):
            continue

        perim = cv2.arcLength(contour, True)
        circ  = _circularity(area_px, perim)
        if circ < min_circularity:
            continue

        ar = _aspect_ratio(contour)
        if ar > max_aspect_ratio:
            continue

        # Colony mask for feature extraction
        col_mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(col_mask, [contour], -1, 255, -1)

        x, y, w, h = cv2.boundingRect(contour)
        pcx = int(x + w / 2)
        pcy = int(y + h / 2)

        equiv_r = math.sqrt(area_px / math.pi)  # equivalent circle radius (px)

        colour  = _colour_features(image, col_mask)
        texture = _texture_contrast(gray, col_mask)
        hemol   = _hemolysis_zone(gray, pcx, pcy, equiv_r)

        # Blob ID from pre-watershed labelling (used to detect touching colonies)
        blob_id = int(pre_watershed_labels[
            min(pcy, pre_watershed_labels.shape[0] - 1),
            min(pcx, pre_watershed_labels.shape[1] - 1),
        ])

        valid_contours.append(contour)
        contour_info.append({
            # geometry
            "area_px":       float(area_px),
            "area_mm2":      float(area_px / px_per_mm ** 2),
            "circularity":   float(circ),
            "aspect_ratio":  float(ar),
            "perimeter_px":  float(perim),
            "equiv_radius_px": float(equiv_r),
            "centroid":      (pcx, pcy),
            "bbox":          (int(x), int(y), int(w), int(h)),
            # colour
            **colour,
            # texture
            "texture_contrast": texture,
            # haemolysis
            "hemolysis_delta": hemol,
            # internal — removed after touching detection below
            "_blob_id":      blob_id,
            # flags initialised empty; touching_colony added below if needed
            "anomaly_flags": [],
        })

    # ── 7. Touching-colony detection ─────────────────────────────────────────
    # Any blob that watershed split into 2+ colonies = those colonies were
    # originally touching. Flag them before the Z-score pass runs.
    blob_counts = Counter(c["_blob_id"] for c in contour_info)
    for c in contour_info:
        bid = c.pop("_blob_id")
        if bid > 0 and blob_counts[bid] > 1:
            c["anomaly_flags"].append("touching_colony")

    # ── 8. Statistical anomaly flagging ──────────────────────────────────────
    contour_info = _flag_anomalies(contour_info, z_thresh=anomaly_z_thresh)
    count         = len(valid_contours)
    anomaly_count = sum(1 for c in contour_info if c["anomaly_flags"])

    # ── 9. Summary statistics ─────────────────────────────────────────────────
    if contour_info:
        areas  = [c["area_mm2"]     for c in contour_info]
        circs  = [c["circularity"]  for c in contour_info]
        hemols = [c["hemolysis_delta"] for c in contour_info]
        plate_area_mm2 = math.pi * plate_inner_radius_mm ** 2
        summary = {
            "count":           count,
            "anomaly_count":   anomaly_count,
            "mean_area_mm2":   float(np.mean(areas)),
            "std_area_mm2":    float(np.std(areas)),
            "min_area_mm2":    float(min(areas)),
            "max_area_mm2":    float(max(areas)),
            "mean_circularity": float(np.mean(circs)),
            "coverage_pct":    float(sum(areas) / plate_area_mm2 * 100),
            "hemolysis_candidates": sum(1 for h in hemols if h > 15),
        }
    else:
        summary = {k: 0 for k in [
            "count", "anomaly_count", "mean_area_mm2", "std_area_mm2",
            "min_area_mm2", "max_area_mm2", "mean_circularity",
            "coverage_pct", "hemolysis_candidates"]}

    # ── 10. Annotate output image ─────────────────────────────────────────────
    annotated = original.copy()
    cv2.circle(annotated, (cx, cy), radius,       (255, 165,   0), 3)  # orange rim
    cv2.circle(annotated, (cx, cy), inner_radius, (  0, 200, 255), 2)  # cyan boundary

    for i, (contour, info) in enumerate(zip(valid_contours, contour_info), 1):
        colour = (0, 0, 255) if info["anomaly_flags"] else (0, 255, 0)
        cv2.drawContours(annotated, [contour], -1, colour, 2)
        pcx, pcy = info["centroid"]
        cv2.putText(annotated, str(i), (pcx + 4, pcy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

    label = f"Count: {count}  |  Anomalies: {anomaly_count}"
    cv2.putText(annotated, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, annotated)

    return {
        "input_path":   image_path,
        "output_path":  output_path,
        "count":        count,
        "anomaly_count": anomaly_count,
        "px_per_mm":    float(px_per_mm),
        "plate_circle": {
            "cx": cx, "cy": cy,
            "radius": radius, "inner_radius": inner_radius,
        },
        "contours":     contour_info,
        "summary_stats": summary,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    image_path  = sys.argv[1] if len(sys.argv) > 1 else "captures/test.jpg"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results/annotated.jpg"

    result = quantify_colonies(image_path=image_path, output_path=output_path)
    s = result["summary_stats"]
    print(f"Colony count       : {result['count']}")
    print(f"Anomalies flagged  : {result['anomaly_count']}")
    print(f"Mean area          : {s['mean_area_mm2']:.2f} mm²")
    print(f"Coverage           : {s['coverage_pct']:.2f} %")
    print(f"Haemolysis suspects: {s['hemolysis_candidates']}")
    print(f"px/mm              : {result['px_per_mm']:.2f}")
    if output_path:
        print(f"Annotated image    : {output_path}")
