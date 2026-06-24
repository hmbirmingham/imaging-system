"""
ml_diagnostics.py — Accuracy diagnostics + auto-documentation for the pipeline.

Two accuracy tracks, both derived from the validation ground truth that the
in-app workflow writes into data/colony_features.csv:

  1. Counting & placement accuracy — the detector's core job. Needs NO trained
     model; it compares the auto count/positions against the human-validated
     truth (validation_status = confirmed / false_positive / added):
        confirmed       → true colony the detector found        (TP)
        false_positive  → detector found a non-colony           (FP)
        added           → real colony the detector missed       (FN)
     → count error vs manual_count, detection precision & recall.

  2. Anomaly classifier — the ML layer. Labelled-data counts, the latest
     cross-validation metrics recorded at train time, and feature importances.

On each retrain a snapshot is appended to data/ml_history.csv and a human
readable model card is written to models/anomaly_card.md.
"""

from __future__ import annotations

import csv
import datetime as _dt
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR     = Path("data")
CSV_FILE     = DATA_DIR / "colony_features.csv"
HISTORY_FILE = DATA_DIR / "ml_history.csv"
MODEL_DIR    = Path("models")
MODEL_CARD   = MODEL_DIR / "anomaly_card.md"

HISTORY_COLUMNS = ["timestamp", "n_labelled", "n_anomalies",
                   "cv_f1_mean", "cv_f1_std", "precision", "recall"]


def _rows(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── Track 1: counting & placement accuracy ────────────────────────────────────
def counting_accuracy(csv_path: Path = CSV_FILE) -> Dict:
    rows = _rows(csv_path)
    by_image: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        if r.get("validation_status"):      # only validated rows count
            by_image[r.get("image_path", "")].append(r)

    plates = []
    tp = fp = fn = 0
    for img, rs in by_image.items():
        c_conf = sum(1 for r in rs if r["validation_status"] == "confirmed")
        c_fp   = sum(1 for r in rs if r["validation_status"] == "false_positive")
        c_add  = sum(1 for r in rs if r["validation_status"] == "added")
        tp += c_conf; fp += c_fp; fn += c_add
        auto   = _num(next((r.get("auto_count") for r in rs if r.get("auto_count")), 0))
        manual = _num(next((r.get("manual_count") for r in rs if r.get("manual_count")), 0)) \
                 or (c_conf + c_add)
        plates.append({"image": Path(img).name, "auto": auto, "manual": manual,
                       "error": auto - manual})

    n = len(plates)
    abs_errs = [abs(p["error"]) for p in plates]
    within1  = sum(1 for e in abs_errs if e <= 1)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall    = tp / (tp + fn) if (tp + fn) else None
    return {
        "validated_plates":   n,
        "colonies_validated": tp + fp + fn,
        "avg_count_error":    round(sum(abs_errs) / n, 2) if n else None,
        "within_1_pct":       round(100 * within1 / n, 1) if n else None,
        "false_positives":    fp,
        "missed":             fn,
        "detection_precision": round(precision, 3) if precision is not None else None,
        "detection_recall":    round(recall, 3) if recall is not None else None,
        "plates":             plates[-30:],   # recent, for the chart
    }


# ── Track 2: anomaly classifier status ────────────────────────────────────────
def anomaly_status(csv_path: Path = CSV_FILE,
                   model_path: Path = MODEL_DIR / "anomaly.pkl",
                   min_labels: int = 50) -> Dict:
    rows = _rows(csv_path)
    labelled = [r for r in rows if r.get("is_anomaly") in ("0", "1")]
    n_lab = len(labelled)
    n_anom = sum(1 for r in labelled if r["is_anomaly"] == "1")

    latest = {}
    hist = read_history()
    if hist:
        latest = hist[-1]

    return {
        "trained":        model_path.exists(),
        "n_labelled":     n_lab,
        "n_anomalies":    n_anom,
        "min_labels":     min_labels,
        "ready_to_train": n_lab >= min_labels,
        "cv_f1":          _num(latest.get("cv_f1_mean")) if latest else None,
        "precision":      _num(latest.get("precision")) if latest else None,
        "recall":         _num(latest.get("recall")) if latest else None,
        "feature_importances": _feature_importances(model_path),
        "model_card":     str(MODEL_CARD) if MODEL_CARD.exists() else None,
    }


def _feature_importances(model_path: Path) -> Optional[List[Dict]]:
    if not model_path.exists():
        return None
    try:
        from anomaly import MLDetector, ML_FEATURES
        det = MLDetector(); det.load(str(model_path))
        clf = det.model.named_steps["clf"] if det.model else None
        if clf is None or not hasattr(clf, "feature_importances_"):
            return None
        pairs = sorted(zip(ML_FEATURES, clf.feature_importances_),
                       key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(v), 3)} for f, v in pairs]
    except Exception:
        return None


# ── History + model card ──────────────────────────────────────────────────────
def read_history() -> List[Dict]:
    return _rows(HISTORY_FILE)


def record_snapshot(train_result: Dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new = not HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp":   _dt.datetime.now().isoformat(timespec="seconds"),
            "n_labelled":  train_result.get("n_samples", ""),
            "n_anomalies": train_result.get("n_anomalies", ""),
            "cv_f1_mean":  round(_num(train_result.get("cv_f1_mean")), 4),
            "cv_f1_std":   round(_num(train_result.get("cv_f1_std")), 4),
            "precision":   round(_num(train_result.get("precision")), 4),
            "recall":      round(_num(train_result.get("recall")), 4),
        })


def write_model_card(train_result: Dict, count_acc: Optional[Dict] = None) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    ca = count_acc or counting_accuracy()
    lines = [
        "# Anomaly model card",
        "",
        f"_Auto-generated {now}_",
        "",
        "## Anomaly classifier",
        f"- Samples (labelled colonies): {train_result.get('n_samples', '?')}",
        f"- Anomalies in training set: {train_result.get('n_anomalies', '?')}",
        f"- Cross-validated F1: {round(_num(train_result.get('cv_f1_mean')), 3)} "
        f"± {round(_num(train_result.get('cv_f1_std')), 3)}",
        "",
        "## Counting & placement accuracy (from validated plates)",
        f"- Validated plates: {ca.get('validated_plates')}",
        f"- Average count error: {ca.get('avg_count_error')}",
        f"- Within ±1: {ca.get('within_1_pct')}%",
        f"- Detection precision: {ca.get('detection_precision')}",
        f"- Detection recall: {ca.get('detection_recall')}",
        "",
        "_Counting accuracy is independent of the ML model — it reflects the "
        "detector vs. human-validated ground truth._",
    ]
    MODEL_CARD.write_text("\n".join(lines))
    return MODEL_CARD


def train_and_record(csv_path: Path = CSV_FILE,
                     model_path: Path = MODEL_DIR / "anomaly.pkl") -> Dict:
    """
    Train the anomaly model, capture CV precision/recall alongside F1, append a
    history snapshot, and regenerate the model card. Raises if scikit-learn is
    unavailable or there is not enough labelled data.
    """
    from anomaly import AnomalyDetector, ML_FEATURES, ML_AVAILABLE
    if not ML_AVAILABLE:
        raise RuntimeError("scikit-learn required to train.")

    result = AnomalyDetector().train(str(csv_path), save_path=str(model_path))

    try:
        import pandas as pd
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        df = pd.read_csv(csv_path)
        df = df[df["is_anomaly"].isin([0, 1, "0", "1"])]
        X = df[ML_FEATURES].fillna(0).values
        y = df["is_anomaly"].astype(int).values
        pipe = Pipeline([("s", StandardScaler()),
                         ("c", RandomForestClassifier(n_estimators=200, max_depth=8,
                                                      class_weight="balanced",
                                                      random_state=42))])
        result["precision"] = float(cross_val_score(pipe, X, y, cv=5, scoring="precision").mean())
        result["recall"]    = float(cross_val_score(pipe, X, y, cv=5, scoring="recall").mean())
    except Exception:
        pass

    ca = counting_accuracy(csv_path)
    record_snapshot(result)
    write_model_card(result, ca)
    return {"train": result, "counting": ca}


def diagnostics() -> Dict:
    """Single payload for the Settings → ML Diagnostics dashboard."""
    return {
        "counting": counting_accuracy(),
        "anomaly":  anomaly_status(),
        "history":  read_history(),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    print(json.dumps(diagnostics(), indent=2))
