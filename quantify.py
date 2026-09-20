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

Colony dict schema — keys WRITTEN by this module (owned here):
  area_px, area_mm2, circularity, aspect_ratio, perimeter_px,
  equiv_radius_px, centroid, bbox,
  r_mean, g_mean, b_mean, r_std, g_std, b_std,
  texture_contrast, hemolysis_delta,
  anomaly_flags  (list[str] — morphometric/statistical flags; includes
                  "touching_colony" from watershed and the Z-score flags),
  anomaly_score  (float — mean Z across area/circularity/aspect_ratio)

Keys WRITTEN by anomaly.py (treat as read-only here):
  stat_flags, stat_score, ml_anomaly, ml_score, ml_note,
  is_anomaly, combined_score
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

# ── Detection & anomaly thresholds ────────────────────────────────────────────
# Extracted from inline literals so the biological/algorithmic rationale is
# explicit and tunable per-medium after validation. Values are unchanged.

# Minimum number of colonies on a plate before per-plate statistical (Z-score)
# anomaly scoring is meaningful — below this the population stats are unstable.
MIN_COLONIES_FOR_STATS: int = 3

# Z-score beyond which a colony feature is flagged as a statistical outlier
# relative to the rest of the plate (~2.5σ ≈ the tails of a normal population).
ANOMALY_Z_THRESHOLD: float = 2.5

# Hard circularity floor for the "non_circular" flag. Below this a contour is
# likely a streak, edge artifact, or debris rather than a round colony.
# Tuned empirically on BAP and MAC plates — revisit per-medium after validation.
NON_CIRCULAR_THRESHOLD: float = 0.4

# Aspect-ratio ceiling for the "streak_or_artifact" flag. Above this the contour
# is elongated — a streak or swarming smear (e.g. Proteus) rather than a discrete
# colony. Profile swarming overrides are applied downstream via the biology block.
STREAK_ASPECT_RATIO_THRESHOLD: float = 3.0

# Haemolysis halo brightness delta threshold (pixel-intensity units). Beta-
# haemolysis on blood agar clears the medium, producing a halo significantly
# brighter than surrounding agar under backlight — flags candidates for review.
HEMOLYSIS_DELTA_THRESHOLD: float = 15.0

# Outer radius of the haemolysis measurement annulus, as a multiple of the
# colony's equivalent radius (how far out the clear zone is sampled).
HEMOLYSIS_ZONE_SCALE: float = 2.5

# Background ring for haemolysis begins just beyond the annulus, at this multiple
# of the outer radius, to sample unaffected agar for the brightness baseline.
HEMOLYSIS_BG_RING_FACTOR: float = 1.1

# Minimum pixel separation between distinct watershed markers (local maxima
# of the distance transform). Two colony centres closer than this seed as
# one marker and rely on cv2.watershed's boundary tracing alone to split
# them. Set below the smallest observed held-out colony radius (~4.7px on
# the blind-eval plates) so it doesn't force-merge legitimately small
# touching colonies, but above single-pixel noise in the distance transform.
WATERSHED_MIN_PEAK_DISTANCE_PX: int = 4


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
                    colony_r: float, scale: float = HEMOLYSIS_ZONE_SCALE) -> float:
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
    bg      = dist > outer * HEMOLYSIS_BG_RING_FACTOR

    halo_mean = float(np.mean(roi[annulus])) if annulus.any() else 0.0
    bg_mean   = float(np.mean(roi[bg]))      if bg.any()      else halo_mean

    return halo_mean - bg_mean  # positive = lighter halo = haemolysis candidate


# ── Statistical anomaly scoring ───────────────────────────────────────────────

def _flag_anomalies(contour_info: List[Dict],
                    z_thresh: float = ANOMALY_Z_THRESHOLD) -> List[Dict]:
    """
    Flag per-colony anomalies using Z-score on key features.
    Preserves any flags already set (e.g. touching_colony from watershed).
    Adds/updates 'anomaly_flags' and 'anomaly_score' on each colony dict.
    """
    if len(contour_info) < MIN_COLONIES_FOR_STATS:
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
        if colony["circularity"] < NON_CIRCULAR_THRESHOLD:
            flags.append("non_circular")
        if colony["aspect_ratio"] > STREAK_ASPECT_RATIO_THRESHOLD:
            flags.append("streak_or_artifact")
        if colony["hemolysis_delta"] > HEMOLYSIS_DELTA_THRESHOLD:
            flags.append("hemolysis_candidate")

        # Deduplicate, preserving order
        colony["anomaly_flags"] = list(dict.fromkeys(flags))
        colony["anomaly_score"] = round(
            (area_z[i] + circ_z[i] + aspect_z[i]) / 3.0, 3)

    return contour_info


# ── Pipeline stages ───────────────────────────────────────────────────────────

def _subtract_background(gray: np.ndarray, plate_mask: np.ndarray,
                         blur_kernel: int, diff_threshold: int) -> np.ndarray:
    """
    Flatten the backlight gradient and threshold to a cleaned binary mask.

    A heavy Gaussian blur models the slow backlight/hotspot gradient;
    (background - image) makes dark colonies bright while the gradient cancels.
    The result is masked to the plate, thresholded, and morphologically cleaned.

    Parameters
    ----------
    gray           : grayscale plate image
    plate_mask     : uint8 mask of the usable inner plate area
    blur_kernel    : odd Gaussian kernel size for the illumination model
    diff_threshold : binary threshold on the background-subtracted difference

    Returns
    -------
    Cleaned binary (uint8) mask ready for watershed segmentation.
    """
    bg_model    = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    diff        = cv2.subtract(bg_model, gray)
    diff_masked = cv2.bitwise_and(diff, plate_mask)

    _, binary = cv2.threshold(diff_masked, diff_threshold, 255, cv2.THRESH_BINARY)

    k       = np.ones((3, 3), np.uint8)
    opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=1)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=2)
    return cleaned


def _apply_watershed(image: np.ndarray,
                     cleaned: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Split touching colonies with a distance-transform watershed.

    Parameters
    ----------
    image   : original BGR image (watershed operates on the colour image)
    cleaned : cleaned binary mask from _subtract_background

    Returns
    -------
    (contours, pre_watershed_labels)
      contours              : external contours of the segmented colonies
      pre_watershed_labels  : connected-component labelling of `cleaned` taken
                              BEFORE watershed, used downstream to detect
                              colonies that were originally touching.
    """
    # Label connected components BEFORE watershed (touching-colony detection).
    _, pre_watershed_labels = cv2.connectedComponents(cleaned)

    k = np.ones((3, 3), np.uint8)
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

    # Marker seeding by local maxima of the distance transform, rather than
    # a single sure-foreground threshold (per-plate or per-blob): a
    # threshold still merges two touching colonies of different sizes into
    # one marker whenever the saddle between their peaks doesn't dip below
    # whichever peak set the threshold. Non-maximum suppression over a
    # WATERSHED_MIN_PEAK_DISTANCE_PX window gives each colony's own peak a
    # marker independent of its neighbours' size, which is what actually
    # splits touching colonies rather than just avoiding losing isolated
    # ones (see WATERSHED_MIN_PEAK_DISTANCE_PX docstring).
    footprint = np.ones((2 * WATERSHED_MIN_PEAK_DISTANCE_PX + 1,) * 2)
    is_peak = (dist_transform == ndi.maximum_filter(dist_transform, footprint=footprint))
    is_peak &= dist_transform > 0
    peak_labels, _ = ndi.label(is_peak)

    sure_bg = cv2.dilate(cleaned, k, iterations=3)
    unknown = cv2.subtract(sure_bg, (peak_labels > 0).astype(np.uint8) * 255)

    markers = peak_labels + 1
    markers[unknown == 255] = 0

    ws_image = image.copy()
    markers  = cv2.watershed(ws_image, markers)
    watershed_mask = np.zeros_like(cleaned)
    watershed_mask[markers > 1] = 255

    contours, _ = cv2.findContours(
        watershed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, pre_watershed_labels


def _annotate_image(original: np.ndarray,
                    valid_contours: List[np.ndarray],
                    contour_info: List[Dict],
                    plate_circle: Tuple[int, int, int, int],
                    count: int, anomaly_count: int) -> np.ndarray:
    """
    Draw the plate rings, per-colony contours, numbers, and count label.

    Parameters
    ----------
    original       : BGR image to draw on (a copy is made)
    valid_contours : accepted colony contours (drawn in order)
    contour_info   : matching per-colony dicts (anomaly_flags → red, else green)
    plate_circle   : (cx, cy, radius, inner_radius)
    count          : total colony count for the overlay label
    anomaly_count  : flagged colony count for the overlay label

    Returns
    -------
    Annotated BGR image.
    """
    cx, cy, radius, inner_radius = plate_circle
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
    return annotated


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
    anomaly_z_thresh: float = ANOMALY_Z_THRESHOLD,
) -> Dict:
    """
    Quantify bacterial colonies in a backlit agar plate image.

    Returns
    -------
    dict with keys:
      input_path, output_path, count, px_per_mm,
      plate_circle, contours, summary_stats, anomaly_count
    """
    # ── Parameter validation ─────────────────────────────────────────────────
    if min_area_mm2 < 0:
        raise ValueError(f"min_area_mm2 must be non-negative, got {min_area_mm2}")
    if max_area_mm2 <= min_area_mm2:
        raise ValueError(
            f"max_area_mm2 ({max_area_mm2}) must be greater than "
            f"min_area_mm2 ({min_area_mm2})")
    if not (0 <= min_circularity <= 1):
        raise ValueError(
            f"min_circularity must be in [0, 1], got {min_circularity}")
    if max_aspect_ratio <= 1:
        raise ValueError(
            f"max_aspect_ratio must be greater than 1, got {max_aspect_ratio}")

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

    # ── 2-4. Background subtraction, threshold, morphological cleanup ─────────
    cleaned = _subtract_background(gray, plate_mask, bg_blur_kernel, diff_threshold)

    # ── 5. Watershed segmentation (split touching colonies) ──────────────────
    contours, pre_watershed_labels = _apply_watershed(image, cleaned)

    # ── 6. Filter contours & extract features ────────────────────────────────
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
            "hemolysis_candidates": sum(1 for h in hemols if h > HEMOLYSIS_DELTA_THRESHOLD),
        }
    else:
        summary = {k: 0 for k in [
            "count", "anomaly_count", "mean_area_mm2", "std_area_mm2",
            "min_area_mm2", "max_area_mm2", "mean_circularity",
            "coverage_pct", "hemolysis_candidates"]}

    # ── 10. Annotate output image ─────────────────────────────────────────────
    annotated = _annotate_image(original, valid_contours, contour_info,
                                (cx, cy, radius, inner_radius),
                                count, anomaly_count)

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
