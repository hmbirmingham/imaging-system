#!/usr/bin/env python3
"""
server.py — Flask web backend for Plate Imaging System.

Serves the Apple-style web frontend at http://localhost:5000.
Launch: python3 server.py
Pi kiosk: chromium-browser --kiosk http://localhost:5000
"""

from __future__ import annotations

import io
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import (Flask, Response, jsonify, render_template,
                   request, send_from_directory, stream_with_context)
from PIL import Image

from quantify import quantify_colonies
from anomaly import AnomalyDetector
from data_logger import DataLogger
from led_pwm import get_pwm
from profiles import ProfileStore, apply_profile
import app_settings
import ml_diagnostics

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_AVAILABLE = False
_cam = None

try:
    from picamera2 import Picamera2
    _cam = Picamera2()
    try:
        cfg = _cam.create_preview_configuration(main={"size": (1280, 960)})
        _cam.configure(cfg)
    except Exception:
        cfg = _cam.create_preview_configuration(main={"size": (640, 480)})
        _cam.configure(cfg)
    _cam.start()
    CAMERA_AVAILABLE = True
    log.info("Camera initialised")
except Exception as e:
    log.warning("Camera unavailable: %s — running in demo mode", e)

# ── Hardware & pipeline ───────────────────────────────────────────────────────
_detector  = AnomalyDetector()
_logger    = DataLogger()
_pwm       = get_pwm()

# Profile store — optional. If PyYAML is missing on the Pi, the app keeps working
# (quantify just runs without profile-aware adjustments).
_profiles = ProfileStore()
try:
    _profiles.plate_types()        # touches the store; raises if PyYAML absent
    PROFILES_AVAILABLE = True
    log.info("Profiles available: %d organisms, %d plate types",
             len(_profiles.list_profiles()), len(_profiles.plate_types()))
except Exception as e:
    PROFILES_AVAILABLE = False
    log.warning("Profiles unavailable: %s — running without profile features", e)

# ── Camera settings state ─────────────────────────────────────────────────────
_cam_settings: dict = {
    "brightness":    0.0,
    "contrast":      1.0,
    "saturation":    1.0,
    "auto_exposure": True,
    "exposure_time": 20000,
    "analogue_gain": 1.0,
}

# ── Directories ───────────────────────────────────────────────────────────────
SAVE_DIR   = Path("captures")
RESULT_DIR = Path("results")
META_DIR   = Path("metadata")
THUMB_DIR  = SAVE_DIR / ".thumbs"

for d in (SAVE_DIR, RESULT_DIR, META_DIR, THUMB_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── SSE client registry ───────────────────────────────────────────────────────
_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()


def _push(event: str, data: dict) -> None:
    msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        dead = [q for q in _sse_clients if q.full()]
        for q in dead:
            _sse_clients.remove(q)
        for q in _sse_clients:
            q.put_nowait(msg)


# ── MJPEG frame buffer ────────────────────────────────────────────────────────
_latest_frame: Optional[bytes] = None
_frame_lock   = threading.Lock()

_DEMO_FRAME: Optional[bytes] = None


def _get_demo_frame() -> bytes:
    global _DEMO_FRAME
    if _DEMO_FRAME is None:
        buf = io.BytesIO()
        img = Image.new("RGB", (640, 480), (28, 28, 30))
        # Simple text via PIL would need a font — just use plain colour
        img.save(buf, "JPEG", quality=60)
        _DEMO_FRAME = buf.getvalue()
    return _DEMO_FRAME


def _capture_loop() -> None:
    global _latest_frame
    while True:
        if CAMERA_AVAILABLE and _cam:
            try:
                arr = _cam.capture_array()
                img = Image.fromarray(arr)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=70)
                with _frame_lock:
                    _latest_frame = buf.getvalue()
            except Exception as e:
                log.debug("Frame error: %s", e)
                time.sleep(0.1)
                continue
        else:
            with _frame_lock:
                _latest_frame = _get_demo_frame()
            time.sleep(0.1)
        time.sleep(0.033)   # ~30 fps ceiling


threading.Thread(target=_capture_loop, daemon=True, name="camera-loop").start()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           camera_available=CAMERA_AVAILABLE,
                           led_available=_pwm.available)


# ── Camera stream ─────────────────────────────────────────────────────────────

@app.route("/stream")
def stream():
    def generate():
        while True:
            with _frame_lock:
                frame = _latest_frame
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.033)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── SSE ───────────────────────────────────────────────────────────────────────

@app.route("/api/events")
def sse():
    q: queue.Queue = queue.Queue(maxsize=64)
    with _sse_lock:
        _sse_clients.append(q)

    def generate():
        init = {"camera": CAMERA_AVAILABLE, "led": _pwm.available,
                "led_brightness": _pwm.brightness}
        yield f"event: init\ndata: {json.dumps(init)}\n\n"
        try:
            while True:
                try:
                    yield q.get(timeout=25)
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            with _sse_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── Capture ───────────────────────────────────────────────────────────────────

@app.route("/api/capture", methods=["POST"])
def api_capture():
    if not CAMERA_AVAILABLE:
        return jsonify({"status": "error", "message": "No camera connected"}), 503

    req        = request.get_json(silent=True) or {}
    profile_id = req.get("profile_id") or "unknown"
    plate_type = req.get("plate_type") or "unknown"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = SAVE_DIR / f"plate_{ts}.jpg"

    try:
        cam_meta = {}
        if _cam:
            raw_meta = _cam.capture_file(str(filepath))
            if raw_meta:
                cam_meta = {k: raw_meta.get(k)
                            for k in ("ExposureTime", "AnalogueGain",
                                      "ColourTemperature", "Lux")
                            if raw_meta.get(k) is not None}

        meta = {
            "timestamp":  datetime.now().isoformat(),
            "filename":   filepath.name,
            "profile_id": profile_id,
            "plate_type": plate_type,
            "camera":     cam_meta,
            "settings":   dict(_cam_settings),
        }
        (META_DIR / f"{filepath.stem}.json").write_text(
            json.dumps(meta, indent=2))

        _push("capture", {"filename": filepath.name})
        return jsonify({"status": "ok", "filename": filepath.name})

    except Exception as e:
        log.error("Capture error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Quantify ──────────────────────────────────────────────────────────────────

@app.route("/api/quantify", methods=["POST"])
def api_quantify():
    data       = request.get_json(silent=True) or {}
    filename   = data.get("filename")
    profile_id = data.get("profile_id") or "unknown"
    plate_type = data.get("plate_type") or "unknown"
    if not filename:
        return jsonify({"status": "error", "message": "filename required"}), 400

    filepath = SAVE_DIR / filename
    out_path = RESULT_DIR / f"annotated_{filename}"

    if not filepath.exists():
        return jsonify({"status": "error", "message": "file not found"}), 404

    def _run():
        try:
            _push("status", {"message": "Running colony analysis…"})

            # Settings: anomaly sensitivity + plate diameter (diameter prefers the
            # resolved profile's instrument block, which the setting merges into).
            cfg     = app_settings.load()
            z_thr   = float(cfg.get("anomaly_z_thresh", 2.5))
            diam_mm = float(cfg.get("plate_diameter_mm", 90))
            if PROFILES_AVAILABLE and profile_id not in (None, "", "unknown"):
                try:
                    diam_mm = float(_profiles.get(profile_id)
                                    .get("instrument", {}).get("plate_diameter_mm", diam_mm))
                except Exception:
                    pass
            inner_mm = app_settings.plate_inner_radius_mm(diam_mm)
            _detector.stat.z_thresh = z_thr

            result = quantify_colonies(str(filepath), str(out_path),
                                       plate_inner_radius_mm=inner_mm,
                                       anomaly_z_thresh=z_thr)
            result = _detector.analyse(result)

            # Snapshot raw detector flags before profile suppression (for the
            # "show raw detection" diagnostic on the results panel).
            raw_flags = {i + 1: list(c.get("anomaly_flags", []))
                         for i, c in enumerate(result.get("contours", []))}

            # Profile-aware adjustments (swarming suppression, expected size).
            applied = None
            if PROFILES_AVAILABLE and profile_id not in (None, "", "unknown"):
                try:
                    applied = apply_profile(result, _profiles.get(profile_id), plate_type)
                except Exception as e:
                    log.warning("Profile %s not applied: %s", profile_id, e)

            _logger.log(result, plate_type=plate_type, profile_id=profile_id)

            colonies = [
                {
                    "id":            i + 1,
                    "area_mm2":      round(c.get("area_mm2", 0), 4),
                    "circularity":   round(c.get("circularity", 0), 4),
                    "aspect_ratio":  round(c.get("aspect_ratio", 0), 4),
                    "anomaly_flags": c.get("anomaly_flags", []),
                    "raw_anomaly_flags": raw_flags.get(i + 1, []),
                    "ml_anomaly":    c.get("ml_anomaly"),
                    "hemolysis_delta": round(c.get("hemolysis_delta", 0), 2),
                    "stat_score":    c.get("stat_score", 0),
                    "centroid":      list(c.get("centroid", (0, 0))),
                    "bbox":          list(c.get("bbox", (0, 0, 0, 0))),
                }
                for i, c in enumerate(result.get("contours", []))
            ]

            payload = {
                "filename":      filename,
                "timestamp":     datetime.now().isoformat(),
                "profile_id":    profile_id,
                "plate_type":    plate_type,
                "profile":       applied,
                "count":         result["count"],
                "anomaly_count": result.get("anomaly_count", 0),
                "summary_stats": result.get("summary_stats", {}),
                "anomaly_report": {
                    k: v for k, v in result.get("anomaly_report", {}).items()
                    if k not in ("anomalies",)
                },
                "colonies": colonies,
            }

            (META_DIR / f"result_{filepath.stem}.json").write_text(
                json.dumps(payload, indent=2))

            _push("quantify_done", payload)

        except Exception as e:
            log.error("Quantify error: %s", e)
            _push("error", {"message": str(e)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})


# ── Library ───────────────────────────────────────────────────────────────────

@app.route("/api/captures")
def api_captures():
    captures = []
    for p in sorted(SAVE_DIR.glob("plate_*.jpg"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        meta_p   = META_DIR / f"{p.stem}.json"
        result_p = META_DIR / f"result_{p.stem}.json"
        meta     = json.loads(meta_p.read_text())   if meta_p.exists()   else {}
        res      = json.loads(result_p.read_text())  if result_p.exists() else {}
        captures.append({
            "filename":      p.name,
            "timestamp":     meta.get("timestamp", ""),
            "count":         res.get("count"),
            "anomaly_count": res.get("anomaly_count"),
            "has_result":    result_p.exists(),
        })
    return jsonify(captures)


@app.route("/api/result/<stem>")
def api_result(stem: str):
    p = META_DIR / f"result_{stem}.json"
    if not p.exists():
        return jsonify({"status": "no_result"}), 404
    return jsonify(json.loads(p.read_text()))


# ── Profiles ──────────────────────────────────────────────────────────────────

@app.route("/api/profiles")
def api_profiles():
    if not PROFILES_AVAILABLE:
        return jsonify({"available": False, "profiles": [], "plate_types": {}})
    return jsonify({
        "available":   True,
        "profiles":    _profiles.list_profiles(),
        "plate_types": _profiles.plate_types(),
    })


@app.route("/api/profile/<profile_id>", methods=["GET", "POST"])
def api_profile(profile_id: str):
    if not PROFILES_AVAILABLE:
        return jsonify({"status": "error", "message": "profiles unavailable"}), 503
    if request.method == "GET":
        return jsonify(_profiles.get(profile_id))

    # POST — save biology edits and/or a sign-off back to the YAML profile.
    d = request.get_json(silent=True) or {}
    try:
        merged = _profiles.save(
            profile_id,
            organism_biology=d.get("organism_biology"),
            plate_type=d.get("plate_type"),
            plate_biology=d.get("plate_biology"),
            instrument=d.get("instrument"),
            signoff=d.get("signoff"),
            display_name=d.get("display_name"),
            plate_types=d.get("plate_types"),
        )
        return jsonify({"status": "ok", "profile": merged})
    except Exception as e:
        log.error("Profile save error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/validate", methods=["POST"])
def api_validate():
    """
    Write PhD validation for one capture: per-colony ground-truth labels +
    corrected count to the training CSV, and (optionally) biology edits + a
    sign-off to the organism profile. The full validation state is also saved to
    metadata/ so reopening the capture restores the overlay.
    """
    d        = request.get_json(silent=True) or {}
    filename = d.get("filename")
    if not filename:
        return jsonify({"status": "error", "message": "filename required"}), 400

    stem        = Path(filename).stem
    image_path  = str(SAVE_DIR / filename)
    profile_id  = d.get("profile_id") or "unknown"
    plate_type  = d.get("plate_type") or "unknown"
    validated_by = d.get("validated_by") or ""
    colonies    = d.get("colonies", [])    # [{id, is_anomaly, status, centroid}]
    added       = d.get("added", [])       # [{centroid_x, centroid_y, is_anomaly}]
    manual_count = d.get("manual_count")

    labels = {
        int(c["id"]): {
            "is_anomaly": 1 if c.get("is_anomaly") else 0,
            "status":     c.get("status", "confirmed"),
        }
        for c in colonies if c.get("id") is not None
    }

    csv_result = {"updated": 0, "added": 0}
    try:
        csv_result = _logger.apply_validation(
            image_path, labels, added=added, manual_count=manual_count,
            plate_type=plate_type, profile_id=profile_id, validated_by=validated_by)
    except Exception as e:
        log.error("Validation CSV writeback error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500

    # Optional biology edits + sign-off back to the profile YAML.
    if PROFILES_AVAILABLE and profile_id not in (None, "", "unknown") \
            and (d.get("organism_biology") or d.get("plate_biology") or d.get("signoff")):
        try:
            _profiles.save(
                profile_id,
                organism_biology=d.get("organism_biology"),
                plate_type=plate_type,
                plate_biology=d.get("plate_biology"),
                signoff=d.get("signoff"),
            )
        except Exception as e:
            log.warning("Profile writeback failed: %s", e)

    # Persist the validation state so the overlay can be restored.
    state = {
        "filename":     filename,
        "validated_at": datetime.now().isoformat(),
        "validated_by": validated_by,
        "profile_id":   profile_id,
        "plate_type":   plate_type,
        "manual_count": manual_count,
        "colonies":     colonies,
        "added":        added,
    }
    (META_DIR / f"validation_{stem}.json").write_text(json.dumps(state, indent=2))

    _push("validated", {"filename": filename, **csv_result})
    return jsonify({"status": "ok", **csv_result})


@app.route("/api/validation/<stem>")
def api_validation(stem: str):
    p = META_DIR / f"validation_{stem}.json"
    if not p.exists():
        return jsonify({"status": "none"}), 404
    return jsonify(json.loads(p.read_text()))


@app.route("/api/plate_types")
def api_plate_types():
    if not PROFILES_AVAILABLE:
        return jsonify({"available": False, "plate_types": {}})
    return jsonify({"available": True, "plate_types": _profiles.plate_types()})


# ── Settings ──────────────────────────────────────────────────────────────────

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        return jsonify(app_settings.load())
    d = request.get_json(silent=True) or {}
    cfg = app_settings.save(d)
    # Plate diameter propagates into the profile instrument layer.
    if "plate_diameter_mm" in d and PROFILES_AVAILABLE:
        try:
            _profiles.set_instrument_default("plate_diameter_mm", cfg["plate_diameter_mm"])
        except Exception as e:
            log.warning("Could not propagate plate diameter to profiles: %s", e)
    return jsonify({"status": "ok", "settings": cfg})


# ── ML diagnostics ────────────────────────────────────────────────────────────

@app.route("/api/diagnostics")
def api_diagnostics():
    try:
        return jsonify(ml_diagnostics.diagnostics())
    except Exception as e:
        log.error("Diagnostics error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    status = ml_diagnostics.anomaly_status()
    if not status["ready_to_train"]:
        return jsonify({"status": "error",
                        "message": f"Need {status['min_labels']} labelled colonies "
                                   f"(have {status['n_labelled']})."}), 400

    def _run():
        try:
            _push("status", {"message": "Training anomaly model…"})
            out = ml_diagnostics.train_and_record()
            _push("train_done", {
                "cv_f1":  round(out["train"].get("cv_f1_mean", 0), 3),
                "n":      out["train"].get("n_samples"),
            })
        except Exception as e:
            log.error("Training error: %s", e)
            _push("error", {"message": str(e)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})


# ── LED ───────────────────────────────────────────────────────────────────────

@app.route("/api/led", methods=["POST"])
def api_led():
    if not _pwm.available:
        return jsonify({"status": "error", "message": "LED hardware not available"}), 503
    pct = float((request.get_json(silent=True) or {}).get("brightness", 80))
    _pwm.set_brightness(pct)
    return jsonify({"status": "ok", "brightness": pct})


# ── Camera settings ───────────────────────────────────────────────────────────

@app.route("/api/camera", methods=["POST"])
def api_camera():
    if not CAMERA_AVAILABLE:
        return jsonify({"status": "error", "message": "No camera connected"}), 503
    data = request.get_json(silent=True) or {}
    _cam_settings.update({k: v for k, v in data.items() if k in _cam_settings})
    if _cam:
        try:
            ctrl: dict = {
                "Brightness": float(_cam_settings["brightness"]),
                "Contrast":   float(_cam_settings["contrast"]),
                "Saturation": float(_cam_settings["saturation"]),
            }
            if _cam_settings["auto_exposure"]:
                ctrl["AeEnable"] = True
            else:
                ctrl["AeEnable"]     = False
                ctrl["ExposureTime"] = int(_cam_settings["exposure_time"])
                ctrl["AnalogueGain"] = float(_cam_settings["analogue_gain"])
            _cam.set_controls(ctrl)
        except Exception as e:
            log.warning("Camera controls: %s", e)
    return jsonify({"status": "ok", "settings": _cam_settings})


@app.route("/api/camera/settings")
def api_camera_settings():
    return jsonify({"available": CAMERA_AVAILABLE, "settings": _cam_settings})


# ── Static file serving ───────────────────────────────────────────────────────

@app.route("/captures/<path:fn>")
def serve_capture(fn):
    return send_from_directory(SAVE_DIR, fn)


@app.route("/results/<path:fn>")
def serve_result_img(fn):
    return send_from_directory(RESULT_DIR, fn)


@app.route("/models/<path:fn>")
def serve_model_file(fn):
    return send_from_directory(Path("models"), fn)


@app.route("/api/thumbnail/<path:fn>")
def serve_thumb(fn):
    thumb = THUMB_DIR / fn
    if not thumb.exists():
        src = SAVE_DIR / fn
        if not src.exists():
            return "Not found", 404
        try:
            img = Image.open(src)
            img.thumbnail((240, 240), Image.LANCZOS)
            img.save(thumb, "JPEG", quality=72)
        except Exception as e:
            return str(e), 500
    return send_from_directory(THUMB_DIR, fn)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Plate Imaging System → http://localhost:5000")
    _pwm.set_brightness(80)
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False,
                use_reloader=False)
    finally:
        _pwm.off()
        if _cam:
            try:
                _cam.stop()
            except Exception:
                pass
