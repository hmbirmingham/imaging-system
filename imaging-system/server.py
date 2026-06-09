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
            "timestamp": datetime.now().isoformat(),
            "filename":  filepath.name,
            "camera":    cam_meta,
            "settings":  dict(_cam_settings),
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
    data     = request.get_json(silent=True) or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"status": "error", "message": "filename required"}), 400

    filepath = SAVE_DIR / filename
    out_path = RESULT_DIR / f"annotated_{filename}"

    if not filepath.exists():
        return jsonify({"status": "error", "message": "file not found"}), 404

    def _run():
        try:
            _push("status", {"message": "Running colony analysis…"})
            result   = quantify_colonies(str(filepath), str(out_path))
            result   = _detector.analyse(result)
            _logger.log(result)

            colonies = [
                {
                    "id":            i + 1,
                    "area_mm2":      round(c.get("area_mm2", 0), 4),
                    "circularity":   round(c.get("circularity", 0), 4),
                    "aspect_ratio":  round(c.get("aspect_ratio", 0), 4),
                    "anomaly_flags": c.get("anomaly_flags", []),
                    "ml_anomaly":    c.get("ml_anomaly"),
                    "hemolysis_delta": round(c.get("hemolysis_delta", 0), 2),
                    "stat_score":    c.get("stat_score", 0),
                }
                for i, c in enumerate(result.get("contours", []))
            ]

            payload = {
                "filename":      filename,
                "timestamp":     datetime.now().isoformat(),
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
