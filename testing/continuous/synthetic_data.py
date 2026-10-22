"""
synthetic_data.py — Controllable synthetic data generation for the continuous
testing harness (Track 1: plate images, Track 2: colony feature vectors).

Every generator takes an explicit integer seed and is fully deterministic —
the same seed always reproduces the same image/features byte-for-byte. This
is what lets Phase A (see phase_a.py) log a seed instead of the generated
array itself: reproducibility comes from re-running the generator, not from
storing output.

Extends the minimal synthetic-plate pattern already used in
tests/test_quantify.py (`_synthetic_plate`): a bright agar disc on a black
background with darker colony discs inside, since quantify.py's background
subtraction is `blurred_background - image` (a colony must be darker than
its local surroundings to survive). This module adds the axes the test
matrix needs on top of that: illumination shape, colony density (including
deliberately touching colonies), artifact injection, and simulated camera
distance — plus a matching per-colony feature-vector generator for the
anomaly-detection track.

Reuses quantify.py's own constants (PLATE_INNER_RADIUS_MM, thresholds) so
synthetic ground truth is expressed in the same physical units and against
the same rationale as the production thresholds, not an independent guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from quantify import (
    ANOMALY_Z_THRESHOLD,
    HEMOLYSIS_DELTA_THRESHOLD,
    NON_CIRCULAR_THRESHOLD,
    PLATE_INNER_RADIUS_MM,
    STREAK_ASPECT_RATIO_THRESHOLD,
)
from anomaly import ML_FEATURES

# ── Blind validation: held-out parameter ranges ────────────────────────────
# Deliberately never sampled by the continuous harness's own scenarios above
# (DENSITY_RANGES tops out at 40, touching is fixed at 25% and only on
# "dense", hotspot magnitude and colony-size spread are hardcoded literals
# below). Blind evaluation needs conditions the pipeline's own dev/tuning
# loop never saw — see testing/blind_eval/.
#
# Each entry is (low, high) for the stressed axis; a plate in that held-out
# set samples one value from this range while every other axis stays at its
# ordinary default, so a failure can be attributed to the one axis that
# changed rather than a tangle of simultaneous shifts.
HELD_OUT_COLONY_COUNT_RANGE: Tuple[int, int] = (80, 120)
HELD_OUT_CIRCULARITY_NOISE_STD_RANGE: Tuple[float, float] = (0.15, 0.25)
HELD_OUT_HOTSPOT_MAGNITUDE_PCT_RANGE: Tuple[float, float] = (0.35, 0.50)
HELD_OUT_SIZE_VARIANCE_RATIO_RANGE: Tuple[float, float] = (3.5, 5.0)
HELD_OUT_TOUCHING_RATE_RANGE: Tuple[float, float] = (0.25, 0.40)

# ── Realistic-contact rendering ────────────────────────────────────────────
# The default "overlap" contact mode places a forced-touching colony at
# (r1 + r2) * (1 - U(0.3, 0.7)) from its neighbour, i.e. centre distances of
# 30 to 70% of the radius sum. Those are deep interpenetrations, and because
# every colony is drawn with the same flat fill value, a small colony landing
# mostly inside a large one leaves no rendered evidence that it is there at
# all while ground truth still counts it. No segmentation method can recover
# those. "abut" instead places contact near r1 + r2, which is how colonies
# growing on a plate actually meet: they expand until their edges touch and
# then stop, rather than growing through one another.
CONTACT_MODES = ("overlap", "abut")
ABUT_CONTACT_DISTANCE_FRACTION_RANGE: Tuple[float, float] = (0.90, 1.00)

COLONY_SHADING_MODES = ("none", "dome")
# A real backlit colony is thickest at its centre, so it attenuates most
# there and least at its rim. "dome" scales the colony's intensity delta by
# this floor at the rim rising to 1.0 at the centre, instead of the flat
# fill the default uses. The floor is deliberately well above zero: a dome
# that fades fully to the agar level at the rim would delete the silhouette
# edge and conflate a shading test with a contrast test.
DOME_RIM_DELTA_FRACTION = 0.55

# ── Track 1: plate image generation ────────────────────────────────────────

ILLUMINATIONS = ("uniform", "gradient", "hotspot", "low_contrast")

# (min_colonies, max_colonies) per density tier.
DENSITY_RANGES: Dict[str, Tuple[int, int]] = {
    "sparse":   (4, 8),
    "moderate": (12, 20),
    "dense":    (28, 40),
}

ARTIFACT_TYPES = ("streak", "debris", "oversized_blob")

# Real-world colony size band used when generating "normal" colonies —
# comfortably inside quantify.py's default [min_area_mm2, max_area_mm2]
# filter (0.1–20.0 mm²) so they aren't accidentally excluded by area alone.
COLONY_RADIUS_MM_RANGE = (0.6, 2.2)

# Rim shrink used by quantify.py's default plate_inner_radius calibration —
# duplicated here (not imported, it's a local literal in quantify.py) so the
# synthetic plate's usable inner radius matches what quantify.py will derive
# from the same detected plate circle.
_RIM_SHRINK_MM = 3.0


# 960px matches the height of server.py's own default camera preview
# configuration (main={"size": (1280, 960)}). This matters beyond realism:
# at 480px the smallest colonies in COLONY_RADIUS_MM_RANGE render at only
# ~2-3px pixel radius, which a 3x3 morphological open/close in
# quantify._subtract_background reliably erases regardless of detection
# quality — measured recall on synthetic sparse/moderate plates jumped from
# ~0.5 at 480px to ~0.85-0.93 at 960px purely from this resolution change.
REFERENCE_IMAGE_SIZE = 960
REFERENCE_PLATE_RADIUS_FRACTION = 0.35  # plate radius as a fraction of image size

# quantify.detect_plate_circle() only searches Hough radii in
# [0.25, 0.65] x min(image_h, image_w) — a camera_distance_factor pushing
# the rendered plate below that floor makes the real pipeline unable to
# find it at all (not a synthetic-data bug: this is quantify.py's own
# documented detection window). Kept here so callers building their own
# scenarios (e.g. a test matrix) can validate a distance sweep stays within
# the range quantify.py actually supports.
MIN_SUPPORTED_CAMERA_DISTANCE_FACTOR = round(0.25 / REFERENCE_PLATE_RADIUS_FRACTION * 1.15, 2)


@dataclass
class PlateScenario:
    """One row of the Track 1 test matrix.

    The five `Optional` fields below are blind-validation overrides, each
    independently defaulted to None so every existing call site (continuous
    harness scenarios) renders byte-for-byte as before. Setting one lets a
    held-out plate push a single axis past what DENSITY_RANGES / the fixed
    hotspot magnitude / the fixed touching probability / COLONY_RADIUS_MM_RANGE
    otherwise allow, without touching those defaults for anyone else.
    """
    seed: int
    illumination: str = "uniform"
    density: str = "moderate"
    artifacts: Tuple[str, ...] = ()
    camera_distance_factor: float = 1.0   # 1.0 = reference standoff distance
    image_size: int = REFERENCE_IMAGE_SIZE
    colony_count_range: Optional[Tuple[int, int]] = None
    circularity_noise_std: float = 0.0
    hotspot_magnitude_pct: Optional[float] = None
    size_variance_ratio: Optional[float] = None
    touching_rate: Optional[float] = None

    # Realistic-contact rendering. Every one of the five defaults below
    # reproduces the prior behaviour exactly, and `uses_realistic_rendering`
    # stays False while they do, so generate_plate_image() keeps taking the
    # original integer-centre cv2.circle/fillPoly path for every existing
    # caller and renders byte-for-byte as before.
    contact_mode: str = "overlap"
    contact_edge_px: float = 0.0
    contact_edge_delta: float = 0.0
    colony_shading: str = "none"
    blur_sigma_px: float = 0.0
    sensor_noise_std: float = 0.0

    def uses_realistic_rendering(self) -> bool:
        """True when any contact/shading option needs the sub-pixel renderer.

        Blur and sensor noise are deliberately NOT in this predicate: both
        are whole-canvas post-processing applied after every colony is
        drawn, so they work the same under either renderer.
        """
        return (self.contact_mode != "overlap"
                or self.contact_edge_px > 0
                or self.colony_shading != "none")

    def __post_init__(self):
        if self.illumination not in ILLUMINATIONS:
            raise ValueError(f"Unknown illumination: {self.illumination}")
        if self.density not in DENSITY_RANGES:
            raise ValueError(f"Unknown density: {self.density}")
        for a in self.artifacts:
            if a not in ARTIFACT_TYPES:
                raise ValueError(f"Unknown artifact type: {a}")
        if self.camera_distance_factor <= 0:
            raise ValueError("camera_distance_factor must be positive")
        if self.colony_count_range is not None:
            lo, hi = self.colony_count_range
            if lo <= 0 or hi < lo:
                raise ValueError(f"Invalid colony_count_range: {self.colony_count_range}")
        if self.circularity_noise_std < 0:
            raise ValueError("circularity_noise_std must be non-negative")
        if self.hotspot_magnitude_pct is not None and not (0 < self.hotspot_magnitude_pct <= 1):
            raise ValueError("hotspot_magnitude_pct must be in (0, 1]")
        if self.size_variance_ratio is not None and self.size_variance_ratio <= 1:
            raise ValueError("size_variance_ratio must be greater than 1")
        if self.touching_rate is not None and not (0 <= self.touching_rate <= 1):
            raise ValueError("touching_rate must be in [0, 1]")
        if self.contact_mode not in CONTACT_MODES:
            raise ValueError(f"Unknown contact_mode: {self.contact_mode}")
        if self.colony_shading not in COLONY_SHADING_MODES:
            raise ValueError(f"Unknown colony_shading: {self.colony_shading}")
        if self.contact_edge_px < 0:
            raise ValueError("contact_edge_px must be non-negative")
        if self.blur_sigma_px < 0:
            raise ValueError("blur_sigma_px must be non-negative")
        if self.sensor_noise_std < 0:
            raise ValueError("sensor_noise_std must be non-negative")


def expected_px_per_mm(plate_radius_px: float,
                        rim_shrink_mm: float = _RIM_SHRINK_MM,
                        plate_inner_radius_mm: float = PLATE_INNER_RADIUS_MM) -> float:
    """
    Replicates quantify.quantify_colonies()'s own px_per_mm derivation from a
    detected plate circle radius, so synthetic ground truth and the
    production calibration agree by construction.
    """
    rough_ppm = plate_radius_px / (plate_inner_radius_mm + rim_shrink_mm)
    rim_shrink_px = int(rim_shrink_mm * rough_ppm)
    inner_radius_px = max(0, plate_radius_px - rim_shrink_px)
    return inner_radius_px / plate_inner_radius_mm if inner_radius_px > 0 else 1.0


def _illumination_field(size: int, kind: str, rng: np.random.Generator,
                         hotspot_magnitude_pct: Optional[float] = None) -> np.ndarray:
    """Return an (size, size) float32 additive brightness field, roughly zero-mean.

    `hotspot_magnitude_pct`, when given, overrides the hotspot kind's default
    70.0-intensity-unit magnitude (~35% of the standard 200 agar level) with
    `hotspot_magnitude_pct * 200` — used for held-out "poor illumination"
    plates. None preserves the exact prior constant for every other caller.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    if kind == "uniform":
        return np.zeros((size, size), np.float32)
    if kind == "gradient":
        # Diagonal illumination gradient, e.g. one edge of the plate brighter.
        grad = (xx + yy) / (2 * size)
        return (grad - grad.mean()) * 60.0
    if kind == "hotspot":
        magnitude = 70.0 if hotspot_magnitude_pct is None else hotspot_magnitude_pct * 200.0
        cx = rng.uniform(size * 0.3, size * 0.7)
        cy = rng.uniform(size * 0.3, size * 0.7)
        sigma = size * 0.25
        hotspot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        return hotspot * magnitude - hotspot.mean() * magnitude
    if kind == "low_contrast":
        # No spatial structure — contrast is instead reduced at draw time.
        return np.zeros((size, size), np.float32)
    raise ValueError(f"Unknown illumination: {kind}")


def _radius_range_for_variance_ratio(ratio: float, center_mm: float = 1.4) -> Tuple[float, float]:
    """(lo, hi) mm colony-radius range with hi/lo == ratio, geometrically
    centered on `center_mm` (COLONY_RADIUS_MM_RANGE's own rough center) so a
    held-out size-variance plate stresses spread rather than shifting the
    whole population larger or smaller."""
    half = math.sqrt(ratio)
    return (center_mm / half, center_mm * half)


def _draw_colony_shape(canvas: np.ndarray, cx: float, cy: float, radius_px: float,
                        fill_value: float, circularity_noise_std: float,
                        rng: np.random.Generator) -> float:
    """Draw one colony and return its actual circularity.

    With no noise this is a plain filled circle (circularity 1.0 by
    construction). With circularity_noise_std > 0, the boundary is instead a
    jittered polygon — irregular colony morphology, one of the held-out
    stress axes — and circularity is measured from the rendered pixels using
    the same 4*pi*area/perimeter**2 formula as quantify._circularity(), so
    ground truth reflects what the pipeline would actually measure rather
    than an analytic guess about an irregular polygon's shape.
    """
    if circularity_noise_std <= 0:
        cv2.circle(canvas, (int(cx), int(cy)), int(round(radius_px)), fill_value, -1)
        return 1.0

    n_pts = 20
    angles = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    jitter = np.clip(rng.normal(1.0, circularity_noise_std, size=n_pts), 0.4, 1.8)
    pts = np.stack([cx + radius_px * jitter * np.cos(angles),
                    cy + radius_px * jitter * np.sin(angles)], axis=1)
    cv2.fillPoly(canvas, [pts.astype(np.int32)], fill_value)

    # Measure circularity from a small local mask (not the full plate canvas)
    # — held-out generation draws up to 120 colonies on 250 plates, and a
    # full-frame findContours per colony would dominate generation time.
    pad = int(radius_px * 2.2) + 4
    local = np.zeros((pad * 2, pad * 2), np.uint8)
    local_pts = pts - [cx - pad, cy - pad]
    cv2.fillPoly(local, [local_pts.astype(np.int32)], 255)
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return (4 * math.pi * area / perimeter ** 2) if perimeter > 0 else 0.0


def _draw_colony_realistic(canvas: np.ndarray, cx: float, cy: float, radius_px: float,
                            agar_value: float, colony_delta: float,
                            shading: str, edge_px: float, edge_delta: float,
                            circularity_noise_std: float,
                            rng: np.random.Generator) -> float:
    """Draw one colony at a sub-pixel centre and return its actual circularity.

    Differs from _draw_colony_shape in the three ways the realistic-contact
    ladder needs, and is only reached when PlateScenario.
    uses_realistic_rendering() is True, so the default path is untouched:

    1. Sub-pixel centre and an anti-aliased rim, via per-pixel coverage from
       a distance field, instead of cv2.circle at an integer centre. Contact
       geometry near d = r1 + r2 turns on sub-pixel distances, so rounding
       every centre to an integer quantises exactly the variable under test.
    2. Optional dome shading, so intensity falls from centre to rim the way
       a backlit colony's own thickness makes it.
    3. Optional darker or lighter rim ring. This is the cue the default
       renderer cannot produce: with one flat fill value, two colonies that
       meet render as one indivisible region, whereas under real lighting
       their touching edges leave a visible boundary line. `edge_delta` is
       signed, positive darkens the ring and negative lightens it.

    rng is consumed identically to _draw_colony_shape (one normal(size=20)
    draw when circularity_noise_std > 0, nothing otherwise), so switching
    renderers does not shift the plate's random stream.
    """
    n_pts = 20
    jitter = None
    if circularity_noise_std > 0:
        jitter = np.clip(rng.normal(1.0, circularity_noise_std, size=n_pts), 0.4, 1.8)

    reach = radius_px * (1.8 if jitter is not None else 1.0) + max(edge_px, 0.0) + 2.0
    x0 = max(0, int(math.floor(cx - reach)))
    x1 = min(canvas.shape[1], int(math.ceil(cx + reach)) + 1)
    y0 = max(0, int(math.floor(cy - reach)))
    y1 = min(canvas.shape[0], int(math.ceil(cy + reach)) + 1)
    if x1 <= x0 or y1 <= y0:
        return 1.0

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    dist = np.hypot(dx, dy)

    # Local radius: constant for a disc, angle-modulated when the scenario
    # asks for irregular morphology. np.interp wraps the 20 sampled jitter
    # values around the circle the same way fillPoly closes the polygon.
    if jitter is not None:
        angles = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
        theta = np.mod(np.arctan2(dy, dx), 2 * math.pi)
        local_radius = radius_px * np.interp(
            theta, np.append(angles, 2 * math.pi), np.append(jitter, jitter[0]))
    else:
        local_radius = np.full_like(dist, radius_px)

    # Coverage is a one-pixel linear ramp across the rim: the anti-aliasing.
    coverage = np.clip(local_radius + 0.5 - dist, 0.0, 1.0)
    if not coverage.any():
        return 1.0

    if shading == "dome":
        # Hemisphere thickness profile, floored at the rim so the silhouette
        # edge keeps its contrast (see DOME_RIM_DELTA_FRACTION).
        norm = np.clip(dist / np.maximum(local_radius, 1e-6), 0.0, 1.0)
        profile = DOME_RIM_DELTA_FRACTION + (1.0 - DOME_RIM_DELTA_FRACTION) * np.sqrt(
            np.clip(1.0 - norm * norm, 0.0, 1.0))
    else:
        profile = np.ones_like(dist)

    delta = colony_delta * profile
    if edge_px > 0 and edge_delta != 0:
        ring = (dist >= local_radius - edge_px) & (dist <= local_radius + 0.5)
        delta = delta + ring * edge_delta

    colony_value = np.clip(agar_value - delta, 0, 255)
    patch = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = patch * (1.0 - coverage) + colony_value * coverage

    if jitter is None:
        return 1.0
    # Measure circularity from the rendered coverage the same way
    # _draw_colony_shape does, so ground truth reflects the drawn shape.
    local = (coverage >= 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return (4 * math.pi * area / perimeter ** 2) if perimeter > 0 else 0.0


def _visible_fractions(placements: List[Tuple[float, float, float]],
                        supersample: int = 2) -> List[float]:
    """Per colony, the share of its own disc not covered by any later-drawn disc.

    Colonies are drawn in sequence and a later one overwrites an earlier
    one, so "covered" means covered by a colony drawn after it. This is the
    quantity that decides whether a missed colony was ever recoverable: a
    colony with visible_fraction near zero left no pixels of its own in the
    render, and no segmentation method can find it, yet ground truth still
    counts it. Rasterised rather than computed from the two-disc lens
    formula, because the lens area does not compose for three or more
    mutually overlapping discs.

    Discs are used even for irregular (polygon) colonies, so for those sets
    this is an approximation of the drawn shape, not an exact measure.
    """
    fractions: List[float] = []
    for i, (cx, cy, radius) in enumerate(placements):
        later = [(ox, oy, orad) for ox, oy, orad in placements[i + 1:]
                 if math.hypot(cx - ox, cy - oy) < radius + orad]
        if not later:
            fractions.append(1.0)
            continue
        step = 1.0 / supersample
        span = np.arange(-radius, radius + step, step, dtype=np.float32)
        gx, gy = np.meshgrid(cx + span, cy + span)
        own = np.hypot(gx - cx, gy - cy) <= radius
        total = int(own.sum())
        if total == 0:
            fractions.append(1.0)
            continue
        covered = np.zeros_like(own)
        for ox, oy, orad in later:
            covered |= np.hypot(gx - ox, gy - oy) <= orad
        fractions.append(float((own & ~covered).sum()) / total)
    return fractions


def generate_plate_image(scenario: PlateScenario) -> Tuple[np.ndarray, Dict]:
    """
    Render a synthetic backlit plate image plus its ground truth.

    Returns
    -------
    (image_bgr, ground_truth) where ground_truth contains everything needed
    by Phase C to score a quantify_colonies() run against known-correct
    values: plate geometry, per-colony expected area_mm2/position, which
    colonies were deliberately placed touching, and which drawn shapes are
    artifacts that should NOT be counted as colonies.
    """
    rng = np.random.default_rng(scenario.seed)
    size = scenario.image_size
    cx = cy = size // 2

    plate_radius_px = int(size * REFERENCE_PLATE_RADIUS_FRACTION *
                           scenario.camera_distance_factor)
    plate_radius_px = max(24, min(plate_radius_px, size // 2 - 4))
    px_per_mm = expected_px_per_mm(plate_radius_px)
    # Leave a small margin inside the rim-shrunk usable radius for colony draw.
    inner_radius_px = plate_radius_px - int(_RIM_SHRINK_MM * px_per_mm) - 2
    inner_radius_px = max(4, inner_radius_px)

    agar_level = 150 if scenario.illumination == "low_contrast" else 200
    colony_delta = 15 if scenario.illumination == "low_contrast" else 60

    field = _illumination_field(size, scenario.illumination, rng,
                                 hotspot_magnitude_pct=scenario.hotspot_magnitude_pct)
    canvas = np.zeros((size, size), np.float32)
    canvas[:] = 0  # background outside plate stays black (matches production)
    plate_mask = np.zeros((size, size), np.uint8)
    cv2.circle(plate_mask, (cx, cy), plate_radius_px, 255, -1)
    agar = np.clip(agar_level + field, 40, 255)
    canvas = np.where(plate_mask > 0, agar, canvas)

    lo_mm, hi_mm = (_radius_range_for_variance_ratio(scenario.size_variance_ratio)
                    if scenario.size_variance_ratio is not None else COLONY_RADIUS_MM_RANGE)
    lo_n, hi_n = scenario.colony_count_range or DENSITY_RANGES[scenario.density]
    n_colonies = int(rng.integers(lo_n, hi_n + 1))
    touch_prob = (scenario.touching_rate if scenario.touching_rate is not None
                  else (0.25 if scenario.density == "dense" else 0.0))

    colonies: List[Dict] = []
    placements: List[Tuple[float, float, float]] = []  # (x, y, radius_px)

    def _random_point() -> Tuple[float, float]:
        r = inner_radius_px * math.sqrt(rng.uniform(0, 0.92))
        theta = rng.uniform(0, 2 * math.pi)
        return cx + r * math.cos(theta), cy + r * math.sin(theta)

    # Minimum gap (px) enforced between non-deliberately-touching colonies so
    # ground truth "expected_count" is actually what the pipeline should
    # detect — without this, colonies placed uniformly at random collide by
    # chance often enough (birthday-paradox-style) to silently turn
    # "sparse"/"moderate" scenarios into unintended touching scenarios.
    _SEPARATION_MARGIN_PX = 3.0
    _PLACEMENT_ATTEMPTS = 40

    def _overlaps_any(px: float, py: float, radius_px: float) -> bool:
        return any(math.hypot(px - ox, py - oy) < (radius_px + orad + _SEPARATION_MARGIN_PX)
                   for ox, oy, orad in placements)

    for i in range(n_colonies):
        radius_mm = rng.uniform(lo_mm, hi_mm)
        radius_px = max(2.0, radius_mm * px_per_mm)

        touching = False
        # Deliberately force some colonies to overlap the previous one, so
        # the watershed "touching_colony" path is actually exercised rather
        # than only ever seeing isolated colonies. Rate is touch_prob: either
        # scenario.touching_rate (held-out override) or the prior fixed
        # 25%-on-dense-only behavior when that override is unset. touch_prob
        # short-circuits the rng.uniform() draw itself (not just the branch)
        # when zero, so existing non-dense scenarios consume rng identically
        # to before this override existed — determinism for old callers
        # depends on not drawing an extra random number they never drew.
        if placements and touch_prob > 0 and rng.uniform() < touch_prob:
            ox, oy, orad = placements[-1]
            angle = rng.uniform(0, 2 * math.pi)
            # Both contact modes consume exactly one uniform() draw here, in
            # the same position in the stream, so switching modes does not
            # shift any later draw and "overlap" stays byte-identical.
            if scenario.contact_mode == "abut":
                dist = (orad + radius_px) * rng.uniform(
                    *ABUT_CONTACT_DISTANCE_FRACTION_RANGE)
            else:
                overlap_frac = rng.uniform(0.3, 0.7)
                dist = (orad + radius_px) * (1 - overlap_frac)
            px, py = ox + dist * math.cos(angle), oy + dist * math.sin(angle)
            touching = True
        else:
            px, py = _random_point()
            for _ in range(_PLACEMENT_ATTEMPTS):
                if not _overlaps_any(px, py, radius_px):
                    break
                px, py = _random_point()
            else:
                # No non-overlapping spot found after all attempts (can
                # happen on crowded plates) — place anyway and record the
                # honest ground truth: this colony ended up touching another.
                touching = _overlaps_any(px, py, radius_px)

        placements.append((px, py, radius_px))
        if scenario.uses_realistic_rendering():
            # agar_value is the nominal agar level, not the locally
            # illuminated value, matching _draw_colony_shape's own constant
            # fill so the two renderers differ only in contact rendering.
            circularity = _draw_colony_realistic(
                canvas, px, py, radius_px, float(agar_level), float(colony_delta),
                scenario.colony_shading, scenario.contact_edge_px,
                scenario.contact_edge_delta, scenario.circularity_noise_std, rng)
        else:
            circularity = _draw_colony_shape(
                canvas, px, py, radius_px, float(max(0, agar_level - colony_delta)),
                scenario.circularity_noise_std, rng)

        # is_anomaly mirrors quantify._flag_anomalies()'s deterministic,
        # non-population-relative flags only (touching_colony, non_circular)
        # — not the Z-score flags (unusual_size, elongated, ...), which are
        # relative to the rest of the plate and can't be known at draw time
        # without duplicating that plate-wide statistics machinery here.
        anomaly_type: List[str] = []
        if touching:
            anomaly_type.append("touching_colony")
        if circularity < NON_CIRCULAR_THRESHOLD:
            anomaly_type.append("non_circular")

        colonies.append({
            "cx": float(px), "cy": float(py),
            "radius_px": float(radius_px),
            "radius_mm": float(radius_mm),
            "area_mm2": float(math.pi * radius_mm ** 2),
            "touching": touching,
            "circularity": float(circularity),
            "is_anomaly": bool(anomaly_type),
            "anomaly_type": anomaly_type,
        })

    artifact_records: List[Dict] = []
    for artifact in scenario.artifacts:
        px, py = _random_point()
        if artifact == "debris":
            # Below quantify.py's default min_area_mm2 — must be filtered out.
            r_px = max(1.0, 0.05 * px_per_mm)
            cv2.circle(canvas, (int(px), int(py)), int(round(r_px)),
                       float(max(0, agar_level - colony_delta)), -1)
        elif artifact == "oversized_blob":
            # Above quantify.py's default max_area_mm2 — must be filtered out.
            r_px = 3.2 * px_per_mm
            cv2.circle(canvas, (int(px), int(py)), int(round(r_px)),
                       float(max(0, agar_level - colony_delta)), -1)
        elif artifact == "streak":
            # Thin elongated smear — should trip the streak/aspect-ratio flag
            # if it survives filtering at all.
            length = rng.uniform(20, 40)
            angle = rng.uniform(0, 180)
            axes = (int(length), max(2, int(length * 0.12)))
            cv2.ellipse(canvas, (int(px), int(py)), axes, angle, 0, 360,
                        float(max(0, agar_level - colony_delta)), -1)
        artifact_records.append({"type": artifact, "cx": float(px), "cy": float(py)})

    # Camera blur then sensor noise, in that physical order: the lens blurs
    # the scene and the sensor adds noise to the already-blurred image, so
    # noise must not itself be smoothed. Both are skipped entirely at their
    # defaults, including the rng draw, so no existing caller's stream moves.
    if scenario.blur_sigma_px > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), scenario.blur_sigma_px)
    if scenario.sensor_noise_std > 0:
        canvas = canvas + rng.normal(0.0, scenario.sensor_noise_std,
                                      size=canvas.shape).astype(np.float32)

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # Recorded for every set, including the existing ones: adding a
    # ground-truth key changes no image bytes and nothing in the blind
    # harness scores against it, but it is what separates "the method missed
    # a colony it could see" from "ground truth counted a colony that was
    # never rendered".
    for colony, visible in zip(colonies, _visible_fractions(placements)):
        colony["visible_fraction"] = visible

    ground_truth = {
        "seed": scenario.seed,
        "scenario": {
            "illumination": scenario.illumination,
            "density": scenario.density,
            "artifacts": list(scenario.artifacts),
            "camera_distance_factor": scenario.camera_distance_factor,
            "image_size": scenario.image_size,
            "colony_count_range": scenario.colony_count_range,
            "circularity_noise_std": scenario.circularity_noise_std,
            "hotspot_magnitude_pct": scenario.hotspot_magnitude_pct,
            "size_variance_ratio": scenario.size_variance_ratio,
            "touching_rate": scenario.touching_rate,
        },
        "plate": {"cx": cx, "cy": cy, "radius_px": plate_radius_px,
                   "inner_radius_px": inner_radius_px, "px_per_mm": px_per_mm},
        "colonies": colonies,
        "artifacts": artifact_records,
        "expected_count": len(colonies),
        "touching_pairs": sum(1 for c in colonies if c["touching"]),
        "expected_anomaly_count": sum(1 for c in colonies if c["is_anomaly"]),
    }
    return image_bgr, ground_truth


# ── Blind validation: held-out plate scenarios ─────────────────────────────

HELD_OUT_SET_NAMES: Tuple[str, ...] = (
    "dense_pack", "irregular_morphology", "poor_illumination",
    "size_variance", "high_touching",
)


def generate_held_out_scenario(set_name: str, seed: int) -> PlateScenario:
    """Build one PlateScenario for a held-out blind-validation plate.

    Each named set pushes exactly one axis to its HELD_OUT_*_RANGE stress
    range while every other axis stays at PlateScenario's ordinary defaults,
    so a scoring failure can be attributed to the one axis that changed.
    The stressed value is sampled from a `seed`-derived RNG, so a given
    (set_name, seed) always reproduces the same scenario — recoverable by
    re-running this function rather than needing to be stored separately.
    """
    if set_name not in HELD_OUT_SET_NAMES:
        raise ValueError(f"Unknown held-out set: {set_name}")
    rng = np.random.default_rng(seed)

    if set_name == "dense_pack":
        return PlateScenario(seed=seed, density="dense",
                              colony_count_range=HELD_OUT_COLONY_COUNT_RANGE)
    if set_name == "irregular_morphology":
        std = float(rng.uniform(*HELD_OUT_CIRCULARITY_NOISE_STD_RANGE))
        return PlateScenario(seed=seed, circularity_noise_std=std)
    if set_name == "poor_illumination":
        pct = float(rng.uniform(*HELD_OUT_HOTSPOT_MAGNITUDE_PCT_RANGE))
        return PlateScenario(seed=seed, illumination="hotspot", hotspot_magnitude_pct=pct)
    if set_name == "size_variance":
        ratio = float(rng.uniform(*HELD_OUT_SIZE_VARIANCE_RATIO_RANGE))
        return PlateScenario(seed=seed, size_variance_ratio=ratio)
    if set_name == "high_touching":
        rate = float(rng.uniform(*HELD_OUT_TOUCHING_RATE_RANGE))
        return PlateScenario(seed=seed, density="dense", touching_rate=rate)
    raise AssertionError("unreachable — set_name validated against HELD_OUT_SET_NAMES above")


def generate_held_out_plate(set_name: str, seed: int) -> Tuple[np.ndarray, Dict]:
    """Render one held-out plate. Ground truth is tagged with which axis was
    stressed, for per-set scoring in testing/blind_eval/."""
    scenario = generate_held_out_scenario(set_name, seed)
    image, ground_truth = generate_plate_image(scenario)
    ground_truth["held_out_set"] = set_name
    return image, ground_truth


# ── Realistic-contact ladder: held-out sets ────────────────────────────────
# Kept out of HELD_OUT_SET_NAMES on purpose. Appending to that tuple would
# pull these sets into the default run of generate_held_out_plates and
# run_blind_eval, which would change the pooled numbers in the committed
# blind_eval_results files. These get their own seed base and their own
# plates directory instead, and the existing harness scores them unchanged.
CONTACT_SET_NAMES: Tuple[str, ...] = (
    "contact_r0", "contact_r1", "contact_r2", "contact_r3",
    "contact_dense_r0", "contact_dense_r1", "contact_dense_r2", "contact_dense_r3",
)

# Ladder step parameters. Each is a stated choice, not a tuned one: nothing
# below was selected by looking at a score.
#
# CONTACT_EDGE_PX / DELTA: a 1.5px rim darker by 25 intensity units, which
#   is ~40% of the 60-unit colony delta. Wide enough to survive the 3x3
#   morphological open/close in quantify._subtract_background, and when two
#   colonies abut, their two rims put a ~3px darker seam at the contact.
# CONTACT_BLUR_SIGMA_PX: 0.8px, a focused camera's point spread, enough to
#   soften the rim cue without erasing it.
# CONTACT_SENSOR_NOISE_STD: 2.0 intensity units, against a 60-unit colony
#   delta, so roughly 30:1 and not the limiting factor.
CONTACT_EDGE_PX = 1.5
CONTACT_EDGE_DELTA = 25.0
CONTACT_BLUR_SIGMA_PX = 0.8
CONTACT_SENSOR_NOISE_STD = 2.0


def _contact_ladder_kwargs(step: int) -> Dict:
    """Rendering options for ladder step 0 to 3. Step 0 is the control and
    returns nothing, so it renders exactly as the existing sets do."""
    if step == 0:
        return {}
    kwargs: Dict = {"contact_mode": "abut"}
    if step >= 2:
        kwargs.update(contact_edge_px=CONTACT_EDGE_PX,
                      contact_edge_delta=CONTACT_EDGE_DELTA)
    if step >= 3:
        kwargs.update(colony_shading="dome",
                      blur_sigma_px=CONTACT_BLUR_SIGMA_PX,
                      sensor_noise_std=CONTACT_SENSOR_NOISE_STD)
    return kwargs


def generate_contact_scenario(set_name: str, seed: int) -> PlateScenario:
    """Build one PlateScenario for a realistic-contact ladder plate.

    `contact_r*` matches high_touching's colony count and touching rate, and
    `contact_dense_r*` matches dense_pack's, so within each family the only
    thing the ladder changes is how contact is placed and rendered.
    """
    if set_name not in CONTACT_SET_NAMES:
        raise ValueError(f"Unknown contact set: {set_name}")
    step = int(set_name[-1])
    ladder = _contact_ladder_kwargs(step)
    rng = np.random.default_rng(seed)

    if set_name.startswith("contact_dense_"):
        # Mirrors dense_pack: 80 to 120 colonies, dense-tier touching rate.
        return PlateScenario(seed=seed, density="dense",
                              colony_count_range=HELD_OUT_COLONY_COUNT_RANGE,
                              **ladder)
    # Mirrors high_touching: dense-tier count, elevated touching rate.
    rate = float(rng.uniform(*HELD_OUT_TOUCHING_RATE_RANGE))
    return PlateScenario(seed=seed, density="dense", touching_rate=rate, **ladder)


def generate_contact_plate(set_name: str, seed: int) -> Tuple[np.ndarray, Dict]:
    """Render one realistic-contact ladder plate plus its ground truth."""
    scenario = generate_contact_scenario(set_name, seed)
    image, ground_truth = generate_plate_image(scenario)
    ground_truth["held_out_set"] = set_name
    ground_truth["contact_step"] = int(set_name[-1])
    return image, ground_truth


# ── Track 2: colony feature-vector generation ──────────────────────────────

# Nominal px/mm used only to keep equiv_radius_px internally consistent with
# area_mm2 in the synthetic feature rows — this is a feature-space stand-in,
# not a rendered image, so the exact value doesn't need to match Track 1.
_NOMINAL_PX_PER_MM = 6.0


@dataclass
class FeatureScenario:
    n_samples: int = 300
    seed: int = 0
    anomaly_fraction: float = 0.15
    # Fraction of TRAINING labels deliberately flipped, simulating imperfect
    # manual review — mirrors how data_logger.apply_validation() labels are
    # produced by a human, not an oracle. Held-out evaluation labels stay
    # clean so Track 2 can measure accuracy against true ground truth.
    label_noise: float = 0.05


def _draw_normal_colony(rng: np.random.Generator) -> Dict:
    area_mm2 = max(0.1, rng.normal(3.0, 0.6))
    return {
        "area_mm2": area_mm2,
        "circularity": float(np.clip(rng.normal(0.85, 0.07), 0.05, 1.0)),
        "aspect_ratio": max(1.0, rng.normal(1.15, 0.15)),
        "equiv_radius_px": math.sqrt(area_mm2 * _NOMINAL_PX_PER_MM ** 2 / math.pi),
        "r_mean": rng.normal(90, 10), "g_mean": rng.normal(90, 10), "b_mean": rng.normal(90, 10),
        "r_std": max(0.1, rng.normal(8, 2)), "g_std": max(0.1, rng.normal(8, 2)),
        "b_std": max(0.1, rng.normal(8, 2)),
        "texture_contrast": max(0.0, rng.normal(12, 4)),
        "hemolysis_delta": max(0.0, rng.normal(3, 2)),
    }


def _draw_anomalous_colony(rng: np.random.Generator) -> Dict:
    """
    Drawn from shifted distributions calibrated against quantify.py's actual
    thresholds (ANOMALY_Z_THRESHOLD / NON_CIRCULAR_THRESHOLD /
    STREAK_ASPECT_RATIO_THRESHOLD / HEMOLYSIS_DELTA_THRESHOLD) so a synthetic
    "anomaly" is one the production flagging logic would plausibly also flag,
    not an arbitrary out-of-distribution point.
    """
    kind = rng.choice(["size", "shape", "streak", "hemolysis"])
    row = _draw_normal_colony(rng)
    if kind == "size":
        row["area_mm2"] = max(0.05, rng.choice([rng.normal(9.0, 1.5), rng.normal(0.3, 0.1)]))
        row["equiv_radius_px"] = math.sqrt(row["area_mm2"] * _NOMINAL_PX_PER_MM ** 2 / math.pi)
    elif kind == "shape":
        row["circularity"] = float(np.clip(rng.normal(0.3, 0.1), 0.02, NON_CIRCULAR_THRESHOLD))
    elif kind == "streak":
        row["aspect_ratio"] = rng.normal(4.5, 0.8) + STREAK_ASPECT_RATIO_THRESHOLD - 3.0
        row["texture_contrast"] = max(0.0, rng.normal(28, 6))
    elif kind == "hemolysis":
        row["hemolysis_delta"] = HEMOLYSIS_DELTA_THRESHOLD + max(0.0, rng.normal(10, 4))
    return row


def generate_colony_features(scenario: FeatureScenario) -> pd.DataFrame:
    """
    Build a synthetic per-colony feature table matching anomaly.ML_FEATURES,
    with a clean ground-truth label (`true_is_anomaly`) and a noisy label
    (`is_anomaly`) standing in for imperfect human validation.
    """
    rng = np.random.default_rng(scenario.seed)
    n_anom = int(round(scenario.n_samples * scenario.anomaly_fraction))
    n_norm = scenario.n_samples - n_anom

    rows = [dict(_draw_normal_colony(rng), true_is_anomaly=0) for _ in range(n_norm)]
    rows += [dict(_draw_anomalous_colony(rng), true_is_anomaly=1) for _ in range(n_anom)]
    rng.shuffle(rows)  # np.random.Generator.shuffle works in-place on sequences

    df = pd.DataFrame(rows)
    df["colony_id"] = np.arange(1, len(df) + 1)

    noisy = df["true_is_anomaly"].to_numpy().copy()
    n_flip = int(round(len(noisy) * scenario.label_noise))
    if n_flip:
        flip_idx = rng.choice(len(noisy), size=n_flip, replace=False)
        noisy[flip_idx] = 1 - noisy[flip_idx]
    df["is_anomaly"] = noisy

    assert set(ML_FEATURES).issubset(df.columns), "feature schema drifted from anomaly.ML_FEATURES"
    return df
