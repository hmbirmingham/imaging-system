"""
phase_b.py — Phase B (during-execution trace) capture.

Everything here is captured WHILE the pipeline stage under test runs:
per-stage wall/CPU time, a coarse memory signal, intermediate artifacts, and
every warning/exception raised.

Design note: rather than re-implementing quantify_colonies()'s internal
orchestration to get per-stage timing (which would duplicate — and risk
drifting from — the real pipeline logic), `PipelineInstrumentation`
monkeypatches the already-extracted stage functions on the `quantify`
module (detect_plate_circle, _subtract_background, _apply_watershed,
_colour_features, _texture_contrast, _hemolysis_zone, _flag_anomalies,
_annotate_image) for the duration of a single quantify_colonies() call, then
restores them. Because quantify_colonies() resolves those names from its
own module globals at call time, the patched wrappers are exactly what runs
— no fork of pipeline logic to keep in sync.
"""

from __future__ import annotations

import functools
import resource
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

import quantify

STAGE_FUNCTIONS = [
    "detect_plate_circle",
    "_subtract_background",
    "_apply_watershed",
    "_colour_features",
    "_texture_contrast",
    "_hemolysis_zone",
    "_flag_anomalies",
    "_annotate_image",
]


class StageTimer:
    """Minimal standalone wall/CPU timer for instrumenting a single call site
    outside quantify.py (used by Track 3/4 checks, which call into
    data_logger/profiles/server rather than quantify's stage functions)."""

    def __init__(self, name: str, sink: List[Dict]):
        self.name = name
        self.sink = sink

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._c0 = time.process_time()
        return self

    def __exit__(self, exc_type, exc, tb):
        wall_s = time.perf_counter() - self._t0
        cpu_s = time.process_time() - self._c0
        record = {"stage": self.name, "wall_s": wall_s, "cpu_s": cpu_s}
        if exc is not None:
            record["error"] = "".join(traceback.format_exception(exc_type, exc, tb))
        self.sink.append(record)
        return False  # never swallow exceptions


def _labels_to_visual(labels: np.ndarray) -> np.ndarray:
    """Normalize a watershed integer-label array into a viewable 8-bit image."""
    if labels.max() <= 0:
        return np.zeros(labels.shape, np.uint8)
    norm = (labels.astype(np.float32) / labels.max() * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


class PipelineInstrumentation:
    """
    Context manager: while active, every call to one of STAGE_FUNCTIONS on
    the `quantify` module is timed, RSS-sampled, and (for the first call of
    each image-producing stage in this cycle) saved to `artifact_dir`.

    Usage
    -----
        with PipelineInstrumentation(artifact_dir) as instr:
            result = quantify.quantify_colonies(image_path, output_path)
        instr.stage_timings   # {stage_name: [ {wall_s, cpu_s, peak_rss_kb}, ... ]}
        instr.log_events      # every warning/exception raised during the call
    """

    def __init__(self, artifact_dir: Optional[Path] = None):
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.stage_timings: Dict[str, List[Dict]] = defaultdict(list)
        self.log_events: List[Dict] = []
        self._originals: Dict[str, object] = {}
        self._warnings_ctx = None

    def _wrap(self, name: str, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            c0 = time.process_time()
            try:
                result = fn(*args, **kwargs)
            except Exception:
                tb = traceback.format_exc()
                self.stage_timings[name].append({
                    "wall_s": time.perf_counter() - t0,
                    "cpu_s": time.process_time() - c0,
                    "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "error": tb,
                })
                self.log_events.append({"level": "ERROR", "stage": name, "traceback": tb})
                raise
            self.stage_timings[name].append({
                "wall_s": time.perf_counter() - t0,
                "cpu_s": time.process_time() - c0,
                # NOTE: ru_maxrss is the process's peak RSS since it started,
                # not a per-call peak — sampled per stage as a coarse,
                # monotonically-increasing memory signal, not an isolated
                # per-stage measurement. Good enough to catch a stage that
                # blows up memory use on a Pi; not a substitute for a real
                # profiler.
                "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            })
            self._maybe_save_artifact(name, result)
            return result
        return wrapper

    def _maybe_save_artifact(self, name: str, result) -> None:
        if self.artifact_dir is None:
            return
        img = None
        if name == "_apply_watershed" and isinstance(result, tuple) and len(result) >= 2:
            img = _labels_to_visual(result[1])
        elif name in ("_subtract_background", "_annotate_image") and isinstance(result, np.ndarray):
            img = result
        if img is None:
            return
        path = self.artifact_dir / f"{name.lstrip('_')}.png"
        if not path.exists():  # first call of this stage in this cycle only
            cv2.imwrite(str(path), img)

    def __enter__(self):
        self._warnings_ctx = warnings.catch_warnings(record=True)
        recorded = self._warnings_ctx.__enter__()
        warnings.simplefilter("always")
        self._recorded_warnings = recorded
        for name in STAGE_FUNCTIONS:
            if hasattr(quantify, name):
                self._originals[name] = getattr(quantify, name)
                setattr(quantify, name, self._wrap(name, self._originals[name]))
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, orig in self._originals.items():
            setattr(quantify, name, orig)
        for w in self._recorded_warnings:
            self.log_events.append({
                "level": "WARNING",
                "message": str(w.message),
                "category": w.category.__name__,
            })
        self._warnings_ctx.__exit__(None, None, None)
        if exc is not None:
            self.log_events.append({
                "level": "ERROR",
                "stage": "run_cycle",
                "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
            })
        return False  # never swallow exceptions — run_cycle decides pass/fail
