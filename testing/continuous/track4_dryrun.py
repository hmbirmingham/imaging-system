"""
track4_dryrun.py — Full end-to-end system dry run: capture -> quantify ->
SSE event -> result fetch, in Flask's existing demo-mode code path.

Hardware safety:
this module NEVER calls `/api/led` or anything touching led_pwm's real
GPIO/mmap driver — that route is simply never part of the driven flow, so
there is no path by which an automated cycle pulses the physical LEDs.

`/api/capture` itself always 503s without real camera hardware (demo mode
only covers the MJPEG /stream preview, not capture) — so this dry run
writes a synthetic plate image directly into a temporary capture directory
and exercises the same quantify/log/profile/SSE code that a real capture
would hand off to, via /api/quantify. server.py's SAVE_DIR/RESULT_DIR/
META_DIR and its DataLogger singleton are monkeypatched to temporary paths
for the duration of the call so the dry run never touches the real
captures/ library or the real data/colony_features.csv, then restored.
"""

from __future__ import annotations

import queue
import shutil
import time
from pathlib import Path
from typing import Dict

import cv2

from data_logger import DataLogger
from testing.continuous.synthetic_data import PlateScenario, generate_plate_image


def run_dry_run(tmp_root: Path, seed: int) -> Dict:
    import server  # imported lazily — pulls in Flask/picamera2-probing/etc.

    tmp_captures = tmp_root / "captures"
    tmp_results = tmp_root / "results"
    tmp_meta = tmp_root / "metadata"
    for d in (tmp_captures, tmp_results, tmp_meta):
        d.mkdir(parents=True, exist_ok=True)

    original_dirs = (server.SAVE_DIR, server.RESULT_DIR, server.META_DIR)
    original_logger = server._logger
    server.SAVE_DIR = tmp_captures
    server.RESULT_DIR = tmp_results
    server.META_DIR = tmp_meta
    server._logger = DataLogger(log_file=tmp_root / "colony_features.csv")

    my_queue: queue.Queue = queue.Queue(maxsize=64)
    with server._sse_lock:
        server._sse_clients.append(my_queue)

    try:
        img, _ = generate_plate_image(PlateScenario(seed=seed, density="sparse"))
        filename = "dryrun_plate.jpg"
        cv2.imwrite(str(tmp_captures / filename), img)

        client = server.app.test_client()
        t0 = time.perf_counter()

        resp = client.post("/api/quantify", json={
            "filename": filename, "profile_id": "unknown", "plate_type": "unknown"})
        started_ok = (resp.status_code == 200 and
                      resp.get_json(silent=True) is not None and
                      resp.get_json()["status"] == "started")

        # api_quantify runs the pipeline on a background daemon thread — poll
        # the SSE queue for the real "quantify_done" push it emits (the same
        # _push() call path a real capture uses) rather than sleeping a
        # fixed duration.
        sse_event_ok = False
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            try:
                msg = my_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if "event: quantify_done" in msg or "event: error" in msg:
                sse_event_ok = "event: quantify_done" in msg
                break

        latency_s = time.perf_counter() - t0
        stem = Path(filename).stem
        result_ok = (tmp_meta / f"result_{stem}.json").exists()

        captures_resp = client.get("/api/captures")
        captures_ok = captures_resp.status_code == 200

        return {
            "passed": bool(started_ok and sse_event_ok and result_ok and captures_ok),
            "started_ok": started_ok,
            "sse_event_ok": sse_event_ok,
            "result_ok": result_ok,
            "captures_ok": captures_ok,
            "latency_s": latency_s,
        }
    finally:
        with server._sse_lock:
            if my_queue in server._sse_clients:
                server._sse_clients.remove(my_queue)
        server.SAVE_DIR, server.RESULT_DIR, server.META_DIR = original_dirs
        server._logger = original_logger
        shutil.rmtree(tmp_root, ignore_errors=True)
