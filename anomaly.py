"""
anomaly.py — Two-layer anomaly detection for agar plate colony analysis.

Layer 1 — Statistical (always on):
    Z-score + IQR flagging on colony geometry features. Works from the
    very first run with no training data required.

Layer 2 — ML (activates once enough labelled data exists):
    Random Forest classifier trained on per-colony features.
    Produces a confidence score (0-1) for each anomaly type.

Usage
-----
    from anomaly import AnomalyDetector
    detector = AnomalyDetector()
    report   = detector.analyse(quantify_result)

Training
--------
    from anomaly import AnomalyDetector
    detector = AnomalyDetector()
    detector.train("data/features.csv")      # build ML layer
    detector.save_model("models/anomaly.pkl")
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ML — optional; gracefully degrade if scikit-learn not installed
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    warnings.warn("scikit-learn not installed — ML layer disabled. "
                  "Run: pip install scikit-learn")

# ── Feature columns used for ML ───────────────────────────────────────────────
ML_FEATURES = [
    "area_mm2",
    "circularity",
    "aspect_ratio",
    "equiv_radius_px",
    "r_mean", "g_mean", "b_mean",
    "r_std",  "g_std",  "b_std",
    "texture_contrast",
    "hemolysis_delta",
]

# ── Anomaly types ─────────────────────────────────────────────────────────────
ANOMALY_TYPES = [
    "unusual_size",
    "unusual_shape",
    "elongated",
    "non_circular",
    "streak_or_artifact",
    "hemolysis_candidate",
]


# ── Statistical detector ──────────────────────────────────────────────────────

class StatisticalDetector:
    """
    Flags colonies that deviate significantly from the plate population.
    No training required — works on every run.
    """

    def __init__(self, z_thresh: float = 2.5, iqr_k: float = 1.5):
        self.z_thresh = z_thresh
        self.iqr_k    = iqr_k

    def flag(self, colonies: List[Dict]) -> List[Dict]:
        """Add/update 'stat_flags' and 'stat_score' on each colony dict."""
        if len(colonies) < 3:
            for c in colonies:
                c.setdefault("stat_flags", [])
                c.setdefault("stat_score", 0.0)
            return colonies

        features = {
            "area_mm2":     np.array([c["area_mm2"]    for c in colonies]),
            "circularity":  np.array([c["circularity"] for c in colonies]),
            "aspect_ratio": np.array([c["aspect_ratio"] for c in colonies]),
        }

        z_scores = {}
        for key, vals in features.items():
            std = vals.std()
            z_scores[key] = np.abs((vals - vals.mean()) / std) if std > 0 \
                else np.zeros(len(vals))

        for i, colony in enumerate(colonies):
            flags = list(colony.get("anomaly_flags", []))  # keep quantify flags

            if z_scores["area_mm2"][i]    > self.z_thresh: flags.append("stat_unusual_size")
            if z_scores["circularity"][i] > self.z_thresh: flags.append("stat_unusual_shape")
            if z_scores["aspect_ratio"][i]> self.z_thresh: flags.append("stat_elongated")

            # IQR outlier on area
            q1, q3 = np.percentile(features["area_mm2"], [25, 75])
            iqr    = q3 - q1
            if colony["area_mm2"] < q1 - self.iqr_k * iqr:
                flags.append("stat_undersized")
            elif colony["area_mm2"] > q3 + self.iqr_k * iqr:
                flags.append("stat_oversized")

            # Deduplicate
            colony["stat_flags"] = list(dict.fromkeys(flags))
            colony["stat_score"] = round(
                (z_scores["area_mm2"][i] +
                 z_scores["circularity"][i] +
                 z_scores["aspect_ratio"][i]) / 3.0, 3)

        return colonies


# ── ML detector ───────────────────────────────────────────────────────────────

class MLDetector:
    """
    Random Forest anomaly classifier. Trains on labelled colony feature CSVs
    produced by data_logger.py. Falls back silently if not trained.
    """

    def __init__(self):
        self.model: Optional[object] = None
        self.isolation_forest: Optional[object] = None
        self.trained = False

    def train(self, csv_path: str, label_col: str = "is_anomaly") -> Dict:
        """
        Train on a labelled CSV. Each row = one colony.
        label_col should be 1 (anomaly) or 0 (normal).

        Returns cross-validation accuracy dict.
        """
        if not ML_AVAILABLE:
            raise RuntimeError("scikit-learn required for ML training.")

        df = pd.read_csv(csv_path)
        missing = [f for f in ML_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = df[ML_FEATURES].fillna(0).values

        # ── Supervised classifier (requires labels) ───────────────────────
        if label_col in df.columns:
            y = df[label_col].values
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=42,
                )),
            ])
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="f1")
            self.model.fit(X, y)
            supervised_result = {
                "cv_f1_mean": float(cv_scores.mean()),
                "cv_f1_std":  float(cv_scores.std()),
                "n_samples":  len(y),
                "n_anomalies": int(y.sum()),
            }
        else:
            supervised_result = {"note": "No label column — supervised model skipped"}

        # ── Unsupervised isolation forest (no labels needed) ──────────────
        self.isolation_forest = Pipeline([
            ("scaler", StandardScaler()),
            ("iso",    IsolationForest(
                n_estimators=200,
                contamination=0.05,
                random_state=42,
            )),
        ])
        self.isolation_forest.fit(X)

        self.trained = True
        return supervised_result

    def predict(self, colonies: List[Dict]) -> List[Dict]:
        """
        Add 'ml_anomaly' (bool) and 'ml_score' (0-1 confidence) to each colony.
        If not trained, marks all as untrained.
        """
        if not self.trained or not ML_AVAILABLE:
            for c in colonies:
                c["ml_anomaly"] = None
                c["ml_score"]   = None
                c["ml_note"]    = "not trained"
            return colonies

        X = np.array([[c.get(f, 0) for f in ML_FEATURES] for c in colonies])

        # Isolation forest (always available after training)
        iso_preds = self.isolation_forest.predict(X)  # -1 = anomaly, 1 = normal

        # Supervised model (if trained with labels)
        if self.model is not None:
            proba = self.model.predict_proba(X)[:, 1]
        else:
            # Map isolation forest score to 0-1
            iso_scores = self.isolation_forest["iso"].score_samples(
                self.isolation_forest["scaler"].transform(X))
            proba = 1 - (iso_scores - iso_scores.min()) / \
                        (iso_scores.max() - iso_scores.min() + 1e-9)

        for i, colony in enumerate(colonies):
            ml_flag   = bool(iso_preds[i] == -1 or proba[i] > 0.5)
            colony["ml_anomaly"] = ml_flag
            colony["ml_score"]   = round(float(proba[i]), 3)
            colony["ml_note"]    = "trained"

        return colonies

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model,
                         "iso":   self.isolation_forest,
                         "trained": self.trained}, f)
        print(f"Model saved: {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model             = data["model"]
        self.isolation_forest  = data["iso"]
        self.trained           = data["trained"]
        print(f"Model loaded: {path}")


# ── Combined detector ─────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Unified interface combining statistical and ML anomaly detection.

    Both layers run on every call. If the ML model is not yet trained,
    only statistical results are returned and ml_note = 'not trained'.
    """

    DEFAULT_MODEL_PATH = "models/anomaly.pkl"

    def __init__(self, model_path: Optional[str] = None,
                 z_thresh: float = 2.5):
        self.stat = StatisticalDetector(z_thresh=z_thresh)
        self.ml   = MLDetector()

        # Auto-load model if it exists
        mp = model_path or self.DEFAULT_MODEL_PATH
        if Path(mp).exists():
            try:
                self.ml.load(mp)
            except Exception as e:
                warnings.warn(f"Could not load model {mp}: {e}")

    def analyse(self, quantify_result: Dict) -> Dict:
        """
        Run both detection layers on a quantify_colonies() result dict.

        Returns the result dict with augmented contour entries and a
        combined anomaly report.
        """
        colonies = quantify_result.get("contours", [])

        # Layer 1 — statistical
        colonies = self.stat.flag(colonies)

        # Layer 2 — ML
        colonies = self.ml.predict(colonies)

        # Combined flag: anomalous if EITHER layer agrees
        for colony in colonies:
            stat_hit = bool(colony.get("stat_flags"))
            ml_hit   = bool(colony.get("ml_anomaly"))
            colony["is_anomaly"]     = stat_hit or ml_hit
            colony["combined_score"] = round(
                (colony.get("stat_score", 0) +
                 colony.get("ml_score", 0) or 0) / 2, 3)

        # Build report
        anomalies = [c for c in colonies if c["is_anomaly"]]
        report = {
            "total_colonies":  len(colonies),
            "anomaly_count":   len(anomalies),
            "anomaly_rate_pct": round(len(anomalies) / max(len(colonies), 1) * 100, 1),
            "ml_active":       self.ml.trained,
            "anomalies":       anomalies,
            "flag_breakdown":  _count_flags(colonies),
            "high_confidence": [c for c in anomalies
                                 if c.get("combined_score", 0) > 0.7],
        }

        quantify_result["contours"]      = colonies
        quantify_result["anomaly_report"] = report
        return quantify_result

    def train(self, csv_path: str,
              save_path: Optional[str] = None) -> Dict:
        """Train the ML layer and optionally save the model."""
        result = self.ml.train(csv_path)
        save_path = save_path or self.DEFAULT_MODEL_PATH
        self.ml.save(save_path)
        return result


def _count_flags(colonies: List[Dict]) -> Dict:
    counts: Dict[str, int] = {}
    for c in colonies:
        for flag in c.get("stat_flags", []):
            counts[flag] = counts.get(flag, 0) + 1
    return counts


# ── CLI training helper ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python anomaly.py <features.csv> [model_output.pkl]")
        sys.exit(1)

    csv_path   = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/anomaly.pkl"

    detector = AnomalyDetector()
    result   = detector.train(csv_path, save_path=model_path)
    print("Training complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")
