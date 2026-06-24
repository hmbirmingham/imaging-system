"""
app_settings.py — Small persistent settings store for the Plate Imaging System.

Device-local configuration the Settings page reads and writes. Persisted to
data/settings.json (device-specific, synced separately from the repo). Plate
diameter is *also* pushed into the profile system's instrument layer so it
merges silently into every resolved profile (see server.py / ProfileStore).

JPEG quality is intentionally NOT user-editable — it is fixed high.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

SETTINGS_FILE = Path("data") / "settings.json"

# Fixed (not exposed in the UI)
JPEG_QUALITY = 92

DEFAULTS: Dict = {
    "plate_diameter_mm": 90,      # merges into profile instrument block
    "anomaly_z_thresh":  2.5,     # statistical detector sensitivity (1.5–3.5)
    "default_profile_id": "unknown",
    "default_plate_type": "unknown",
    "auto_quantify":      False,  # run detection automatically after capture
}

# Validation bounds for numeric settings.
BOUNDS = {
    "plate_diameter_mm": (50, 150),
    "anomaly_z_thresh":  (1.5, 3.5),
}


def load() -> Dict:
    """Return settings merged over defaults (missing keys filled)."""
    out = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        try:
            out.update({k: v for k, v in json.loads(SETTINGS_FILE.read_text()).items()
                        if k in DEFAULTS})
        except Exception:
            pass
    return out


def save(partial: Dict) -> Dict:
    """Update only known keys (clamped to bounds), persist, and return the result."""
    cur = load()
    for k, v in partial.items():
        if k not in DEFAULTS:
            continue
        if k in BOUNDS and isinstance(v, (int, float)):
            lo, hi = BOUNDS[k]
            v = max(lo, min(hi, v))
        cur[k] = v
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(cur, indent=2))
    return cur


def plate_inner_radius_mm(diameter_mm: float) -> float:
    """Map a plate diameter to the usable inner radius used for mm² calibration.
    90 mm → 40 mm (matches quantify.PLATE_INNER_RADIUS_MM); ~5 mm rim allowance."""
    return diameter_mm / 2.0 - 5.0
