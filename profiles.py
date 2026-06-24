"""
profiles.py — Profile store for the Plate Imaging System.

Two kinds of profile, kept in separate folders for organisation:

    profiles/organisms/{profile_id}.yaml
        One per organism. Colony appearance is resolved per plate type, because
        the same organism reads differently on different media (E. coli is
        grey/non-hemolytic on BAP, a pink lactose-fermenter on MAC). Structure:
            organism-level — Gram, cell morphology, swarming, incubation window.
            biology.plates — per-plate-type appearance keyed by plate code,
                             with a `default` block for any plate not described.
        Split concern: `instrument` (you set, locked after calibration) vs.
        `biology` (Dr. Ryberg validates; starts unvalidated until signed off).
        profiles/organisms/_generic.yaml is the fallback every organism merges
        over.

    profiles/plate_types/{CODE}.yaml
        One per culture medium (BAP, MAC, CHOC, …). Holds the human name and
        medium imaging properties (background colour/opacity) that affect
        detection — instrument-side, owned by you.

Colony size uses a vocabulary (pinpoint / small / medium / large) + tolerance
(tight / normal / loose); SIZE_CENTERS_MM2 + TOLERANCE_FRAC map a resolved
per-plate size to an expected mm² band for the statistical layer.

Usage
-----
    from profiles import ProfileStore
    store = ProfileStore()
    prof  = store.get("escherichia_coli_lf")
    bio   = store.plate_biology(prof, "MAC")
    lo, hi = store.expected_area_band(prof, "MAC")
    store.save("escherichia_coli_lf", plate_type="MAC",
               plate_biology={"hemolysis": "none"}, signoff={"by": "Dr Ryberg"})
    store.plate_types()                 # {code: name}
    store.plate_profile("MAC")          # full plate-type profile
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import Dict, List, Optional

# PyYAML is optional at import time so the rest of the app still boots if the
# package is missing on the Pi; profile features then raise a clear error.
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    YAML_AVAILABLE = False


PROFILE_DIR    = Path("profiles")
ORGANISM_SUBDIR = "organisms"
PLATE_SUBDIR    = "plate_types"
GENERIC_ID      = "_generic"

# ── Plate types used in the MLS lab (built-in seed) ───────────────────────────
# code → (name, background imaging hint). The live profiles live in
# profiles/plate_types/{code}.yaml so media can be added/edited in-app.
PLATE_TYPES_SEED: Dict[str, Dict[str, str]] = {
    "BAP":     {"name": "Blood Agar (BAP)",          "background": "deep red, opaque"},
    "MAC":     {"name": "MacConkey (MAC)",           "background": "pink-red, translucent"},
    "CHOC":    {"name": "Chocolate (CHOC)",          "background": "brown, opaque"},
    "BAP_MAC": {"name": "Biplate (BAP/MAC)",         "background": "split: red / pink"},
    "XLD":     {"name": "XLD",                        "background": "red-pink, translucent"},
    "CAMPY":   {"name": "Campy Select",              "background": "grey-brown, selective"},
    "MTM":     {"name": "Modified Thayer-Martin",    "background": "brown, opaque"},
    "MH":      {"name": "Mueller Hinton",            "background": "light tan, translucent"},
    "MRSA":    {"name": "MRSA ChromAgar",            "background": "chromogenic base"},
    "CNA":     {"name": "CNA",                        "background": "red (blood-based), opaque"},
    "QUAD_XV": {"name": "Quad plate (XV factor)",    "background": "clear, quadrant"},
}

# ── Colony size vocabulary → mm² area band ────────────────────────────────────
SIZE_CENTERS_MM2 = {"pinpoint": 0.3, "small": 1.2, "medium": 4.5, "large": 12.0}
TOLERANCE_FRAC   = {"tight": 0.35, "normal": 0.60, "loose": 1.00}
SIZE_VOCAB       = list(SIZE_CENTERS_MM2.keys())
TOLERANCE_VOCAB  = list(TOLERANCE_FRAC.keys())
HEMOLYSIS_VOCAB  = ["none", "alpha", "beta", "gamma", "unknown"]
LACTOSE_VOCAB    = ["fermenter", "non_fermenter", "late", "na", "unknown"]

# Floor so "pinpoint / tight" never collapses to a degenerate band.
_MIN_AREA_MM2 = 0.05


class ProfileError(RuntimeError):
    """Raised when the profile system cannot operate (e.g. PyYAML missing)."""


def slugify(name: str) -> str:
    """Turn a free-text organism name into a stable profile_id slug."""
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return s.strip("_")


class ProfileStore:
    """Loads, merges, and persists organism + plate-type profiles."""

    def __init__(self, profile_dir: Path | str = PROFILE_DIR):
        self.dir           = Path(profile_dir)
        self.organisms_dir = self.dir / ORGANISM_SUBDIR
        self.plates_dir    = self.dir / PLATE_SUBDIR
        if YAML_AVAILABLE:
            self.organisms_dir.mkdir(parents=True, exist_ok=True)
            self.plates_dir.mkdir(parents=True, exist_ok=True)

    # ── internals ────────────────────────────────────────────────────────────
    def _require_yaml(self) -> None:
        if not YAML_AVAILABLE:
            raise ProfileError(
                "PyYAML is required for the profile system. "
                "Install it: pip install pyyaml")

    def _path(self, profile_id: str) -> Path:
        return self.organisms_dir / f"{profile_id}.yaml"

    def _read(self, profile_id: str) -> Optional[Dict]:
        p = self._path(profile_id)
        return (yaml.safe_load(p.read_text()) or {}) if p.exists() else None

    @staticmethod
    def _deep_merge(base: Dict, over: Dict) -> Dict:
        """Recursively merge `over` onto a copy of `base` (None values skipped)."""
        out = dict(base)
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = ProfileStore._deep_merge(out[k], v)
            elif v is not None:
                out[k] = v
        return out

    def _generic(self) -> Dict:
        return self._read(GENERIC_ID) or _GENERIC_DEFAULT

    # ── plate-type profiles (own folder) ─────────────────────────────────────
    def _seed_plate_types(self) -> None:
        for code, info in PLATE_TYPES_SEED.items():
            p = self.plates_dir / f"{code}.yaml"
            if not p.exists():
                p.write_text(yaml.safe_dump(
                    {"code": code, "name": info["name"],
                     "imaging": {"background": info["background"],
                                 "background_rgb": None, "notes": ""}},
                    sort_keys=False, allow_unicode=True))

    def plate_types(self) -> Dict[str, str]:
        """Live {code: name} map, seeding the plate_types/ folder if empty."""
        self._require_yaml()
        if not any(self.plates_dir.glob("*.yaml")):
            self._seed_plate_types()
        out = {}
        for p in sorted(self.plates_dir.glob("*.yaml")):
            data = yaml.safe_load(p.read_text()) or {}
            out[data.get("code", p.stem)] = data.get("name", p.stem)
        return out

    def plate_profile(self, code: str) -> Optional[Dict]:
        self._require_yaml()
        p = self.plates_dir / f"{code}.yaml"
        return (yaml.safe_load(p.read_text()) or {}) if p.exists() else None

    def add_plate_type(self, code: str, name: str,
                       imaging: Optional[Dict] = None) -> Dict:
        """Create or update a plate-type profile and persist it."""
        self._require_yaml()
        current = self.plate_profile(code) or {"code": code}
        current["name"] = name
        if imaging:
            current["imaging"] = self._deep_merge(current.get("imaging", {}), imaging)
        current.setdefault("imaging", {"background": None, "background_rgb": None, "notes": ""})
        (self.plates_dir / f"{code}.yaml").write_text(
            yaml.safe_dump(current, sort_keys=False, allow_unicode=True))
        return current

    def set_instrument_default(self, field: str, value) -> int:
        """
        Push an instrument-block value (e.g. plate_diameter_mm) into the generic
        fallback *and* every organism profile, so a global setting silently
        propagates into each merged profile. Per-organism instrument values for
        this field are intentionally overwritten — plate geometry is global, not
        species-specific. Only the named field is touched; everything else in
        each file (including validated biology) is preserved. Returns file count.
        """
        self._require_yaml()
        n = 0
        for p in self.organisms_dir.glob("*.yaml"):
            data = yaml.safe_load(p.read_text()) or {}
            data.setdefault("instrument", {})[field] = value
            p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
            n += 1
        return n

    # ── organism profiles ────────────────────────────────────────────────────
    def exists(self, profile_id: str) -> bool:
        return self._path(profile_id).exists()

    def get(self, profile_id: Optional[str]) -> Dict:
        """Species profile merged over the generic fallback; unknown → generic."""
        self._require_yaml()
        base = self._deep_merge(_GENERIC_DEFAULT, self._generic())
        if not profile_id or profile_id in ("unknown", GENERIC_ID):
            return base
        species = self._read(profile_id)
        if species is None:
            base["profile_id"] = profile_id
            base["_missing"] = True
            return base
        merged = self._deep_merge(base, species)
        merged["profile_id"] = profile_id
        return merged

    def list_profiles(self) -> List[Dict]:
        """Lightweight summary of every organism profile (excludes _generic)."""
        self._require_yaml()
        out = []
        for p in sorted(self.organisms_dir.glob("*.yaml")):
            pid = p.stem
            if pid.startswith("_"):
                continue
            prof = self.get(pid)
            bio  = prof.get("biology", {})
            val  = prof.get("validation", {})
            out.append({
                "profile_id":   pid,
                "display_name": prof.get("display_name", pid),
                "plate_types":  prof.get("plate_types", []),
                "swarming":     bool(bio.get("swarming", False)),
                "gram":         bio.get("gram"),
                "validated":    bool(val.get("validated", False)),
                "validated_by": val.get("validated_by"),
            })
        return out

    def save(self,
             profile_id: str,
             organism_biology: Optional[Dict] = None,
             plate_type: Optional[str] = None,
             plate_biology: Optional[Dict] = None,
             instrument: Optional[Dict] = None,
             signoff: Optional[Dict] = None,
             display_name: Optional[str] = None,
             plate_types: Optional[List[str]] = None) -> Dict:
        """
        Persist edits to an organism profile. Only the blocks supplied are
        touched. plate_type + plate_biology edit one medium's appearance;
        signoff ({"by": str}) flips validation.validated. Returns merged profile.
        """
        self._require_yaml()
        current = self._read(profile_id) or {"profile_id": profile_id}
        current.setdefault("biology", {})

        if display_name:
            current["display_name"] = display_name
        if plate_types is not None:
            current["plate_types"] = plate_types
        if instrument:
            current["instrument"] = self._deep_merge(
                current.get("instrument", {}), instrument)
        if organism_biology:
            current["biology"] = self._deep_merge(current["biology"], organism_biology)
        if plate_type and plate_biology:
            plates = current["biology"].setdefault("plates", {})
            plates[plate_type] = self._deep_merge(plates.get(plate_type, {}), plate_biology)
        if signoff:
            current["validation"] = {
                "validated":    True,
                "validated_by": signoff.get("by"),
                "validated_at": signoff.get("at")
                                or _dt.datetime.now().isoformat(timespec="seconds"),
            }

        current.setdefault("profile_id", profile_id)
        self._path(profile_id).write_text(
            yaml.safe_dump(current, sort_keys=False, allow_unicode=True))
        return self.get(profile_id)

    # ── per-plate resolution ─────────────────────────────────────────────────
    @staticmethod
    def plate_biology(profile: Dict, plate_type: Optional[str]) -> Dict:
        """Resolve appearance for a plate type: plates.default ⊕ plates[plate_type]."""
        plates   = profile.get("biology", {}).get("plates", {})
        default  = plates.get("default", {})
        specific = plates.get(plate_type, {}) if plate_type else {}
        return ProfileStore._deep_merge(default, specific)

    @classmethod
    def expected_area_band(cls, profile: Dict,
                           plate_type: Optional[str] = None) -> tuple[float, float]:
        """Map resolved per-plate colony_size + tolerance to an (min, max) mm² band."""
        bio    = cls.plate_biology(profile, plate_type)
        centre = SIZE_CENTERS_MM2.get(bio.get("colony_size", "medium"),
                                      SIZE_CENTERS_MM2["medium"])
        frac   = TOLERANCE_FRAC.get(bio.get("size_tolerance", "normal"),
                                    TOLERANCE_FRAC["normal"])
        return max(_MIN_AREA_MM2, centre * (1.0 - frac)), centre * (1.0 + frac)

    @staticmethod
    def is_swarming(profile: Dict) -> bool:
        return bool(profile.get("biology", {}).get("swarming", False))


# ── Profile application to a quantify/anomaly result ──────────────────────────
# Flags the CV layer raises that represent "spreading" — meaningless for a
# swarming organism (Proteus mirabilis covers the plate), so they're suppressed
# when the profile is swarming.
SWARM_SUPPRESSED_FLAGS = {"touching_colony", "stat_elongated", "streak_or_artifact"}


def apply_profile(result: Dict, profile: Dict,
                  plate_type: Optional[str] = None) -> Dict:
    """
    Mutate a quantify+anomaly result in place according to an organism profile.

    Two profile-aware adjustments:
      • Swarming suppression — organism-level safe fact, always applied. Removes
        spreading/elongation flags so a swarming lawn isn't flagged as anomalous.
      • Expected-size flagging — only when the profile is *validated* for this
        plate (so unvalidated default bands never produce spurious flags). Adds
        an 'outside_expected_size' flag to colonies outside the mm² band.

    Per-colony 'is_anomaly' and the result's anomaly_count are recomputed so the
    suppression/additions are reflected downstream. Returns an applied-summary
    dict (also stashed in result['profile']).
    """
    swarming  = ProfileStore.is_swarming(profile)
    validated = bool(profile.get("validation", {}).get("validated"))
    lo, hi    = ProfileStore.expected_area_band(profile, plate_type)

    suppressed = size_flags = 0
    for c in result.get("contours", []):
        if swarming:
            before = list(c.get("anomaly_flags", []))
            c["anomaly_flags"] = [f for f in before if f not in SWARM_SUPPRESSED_FLAGS]
            c["stat_flags"]    = [f for f in c.get("stat_flags", [])
                                  if f not in SWARM_SUPPRESSED_FLAGS]
            if len(c["anomaly_flags"]) != len(before):
                suppressed += 1
        if validated:
            area = c.get("area_mm2", 0.0)
            if (area < lo or area > hi) and \
                    "outside_expected_size" not in c.get("anomaly_flags", []):
                c.setdefault("anomaly_flags", []).append("outside_expected_size")
                size_flags += 1
        # recompute combined anomaly verdict
        c["is_anomaly"] = bool(c.get("stat_flags") or c.get("anomaly_flags")
                               or c.get("ml_anomaly"))

    n_anom = sum(1 for c in result.get("contours", []) if c.get("is_anomaly"))
    result["anomaly_count"] = n_anom
    if isinstance(result.get("anomaly_report"), dict):
        result["anomaly_report"]["anomaly_count"] = n_anom

    applied = {
        "profile_id":    profile.get("profile_id", "unknown"),
        "display_name":  profile.get("display_name"),
        "plate_type":    plate_type or "unknown",
        "swarming":      swarming,
        "validated":     validated,
        "size_band_mm2": [round(lo, 3), round(hi, 3)],
        "swarm_flags_suppressed": suppressed,
        "size_flags_added":       size_flags,
    }
    result["profile"] = applied
    return applied


# ── Generic fallback (also written to organisms/_generic.yaml by the seeder) ──
_GENERIC_DEFAULT: Dict = {
    "profile_id":   GENERIC_ID,
    "display_name": "Generic / unknown organism",
    "plate_types":  [],
    "instrument": {
        "plate_diameter_mm": 90, "px_per_mm_expected": None,
        "exposure_hint_us": 20000, "gain_hint": 1.0, "locked": False,
    },
    "biology": {
        "gram":                  "unknown",
        "cell_morphology":       "unknown",
        "swarming":              False,
        "incubation_time_h_min": 18,
        "incubation_time_h_max": 24,
        "plates": {
            "default": {
                "colony_size": "medium", "size_tolerance": "loose",
                "hemolysis": "unknown", "lactose": "na",
                "colony_color": None, "morphology": None, "notes": "",
            },
        },
    },
    "validation": {"validated": False, "validated_by": None, "validated_at": None},
}


# ── CLI: quick inspection ─────────────────────────────────────────────────────
if __name__ == "__main__":
    store = ProfileStore()
    profs = store.list_profiles()
    print(f"{len(profs)} organism profiles in {store.organisms_dir}/  "
          f"({len(store.plate_types())} plate types in {store.plates_dir}/)\n")
    for p in profs:
        flag = "swarm" if p["swarming"] else "     "
        ok   = "✓" if p["validated"] else " "
        print(f"  [{ok}] {flag}  {p['profile_id']:<34} {p['gram'] or '?':<9} "
              f"plates: {','.join(p['plate_types']) or '—'}")
