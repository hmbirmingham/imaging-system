"""
run_cycle.py — Single-cycle orchestrator: wires Phase A -> Track 1 (under
Phase B instrumentation) -> Phase C together, plus Track 3 (every cycle)
and Track 4 (every cycle) and Track 2 (every Nth cycle, if
track2_ml_validation.py is present — added by feat/ml-validation).

Every cycle ends by freeing large arrays and pruning old on-disk artifacts,
because this harness may run
unattended for a long time on a memory/disk-constrained Raspberry Pi
alongside the real capture app, so nothing here is allowed to grow
unbounded.

CLI: `python -m testing.continuous.run_cycle` runs exactly one cycle and
exits 0/1 on pass/fail — this is the unit scheduler.py's both modes call.
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import cv2
import yaml

import quantify
from testing.continuous import phase_a, phase_b, phase_c, track3_integrity, track4_dryrun
from testing.continuous.synthetic_data import PlateScenario, generate_plate_image

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "testing" / "continuous" / "config" / "test_matrix.yaml"
LOGS_DIR = REPO_ROOT / "testing" / "logs"
REPORTS_DIR = REPO_ROOT / "testing" / "reports"
ARTIFACTS_DIR = REPO_ROOT / "testing" / "artifacts" / "images"
STATE_PATH = REPORTS_DIR / "aggregate_state.json"


def _load_matrix() -> Dict:
    return yaml.safe_load(MATRIX_PATH.read_text())


def _prune_dir(dir_path: Path, keep_recent: int, glob: str = "*") -> None:
    """Keep only the `keep_recent` most-recently-modified matches — everything
    here is regenerable from its logged seed, so pruning loses no information
    that matters, only disk space on the (likely SD-card) Pi."""
    if not dir_path.exists():
        return
    entries = sorted(dir_path.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in entries[keep_recent:]:
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
        else:
            stale.unlink(missing_ok=True)


def _ensure_output_dirs() -> None:
    """testing/logs/{pre,during,post}/ etc. are gitignored (raw, regenerable
    output) — a fresh checkout (e.g. GitHub Actions) has none of them on
    disk. They only ever existed locally because they'd been created once
    by hand; nothing in the code actually created them. Idempotent, cheap,
    called at the start of every cycle."""
    for sub in ("pre", "during", "post"):
        (LOGS_DIR / sub).mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def run_one_cycle() -> Dict:
    _ensure_output_dirs()
    matrix = _load_matrix()
    scenarios = matrix["track1_scenarios"]
    retention = matrix.get("artifact_retention_cycles", 5)

    state = phase_c.load_aggregate_state(STATE_PATH)
    cycle_number = state["total_cycles"] + 1
    cycle_id = f"{cycle_number:04d}"
    scenario_cfg = dict(scenarios[state["total_cycles"] % len(scenarios)])

    scenario = PlateScenario(
        seed=cycle_number,
        illumination=scenario_cfg["illumination"],
        density=scenario_cfg["density"],
        artifacts=tuple(scenario_cfg.get("artifacts") or []),
        camera_distance_factor=scenario_cfg["camera_distance_factor"],
    )

    cycle_artifact_dir = ARTIFACTS_DIR / cycle_id
    image, ground_truth = generate_plate_image(scenario)
    input_path = cycle_artifact_dir / "input.png"
    cycle_artifact_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(input_path), image)

    pre_record = phase_a.capture(scenario_cfg, scenario.seed, image.tobytes(), cycle_id)
    (LOGS_DIR / "pre" / f"cycle_{cycle_id}.json").write_text(json.dumps(pre_record, indent=2))

    passed = True
    track1_score: Optional[Dict] = None
    track3_result: Optional[Dict] = None
    track4_result: Optional[Dict] = None
    track2_result: Optional[Dict] = None

    try:
        with phase_b.PipelineInstrumentation(cycle_artifact_dir) as instr:
            quantify_result = quantify.quantify_colonies(
                str(input_path), str(cycle_artifact_dir / "annotated.jpg"))
        track1_score = phase_c.score_track1(quantify_result, ground_truth)
        passed = passed and track1_score["passed"]

        during_record = {"phase": "B", "cycle_id": cycle_id,
                          "stage_timings": dict(instr.stage_timings),
                          "log_events": instr.log_events}
        (LOGS_DIR / "during" / f"cycle_{cycle_id}.json").write_text(json.dumps(during_record, indent=2))
        phase_c.record_stage_timings(state, instr.stage_timings)
        phase_c.record_failures(state, instr.log_events, cycle_id)
        phase_c.record_track1(state, scenario_cfg, track1_score)

        # ── Track 3 — data pipeline integrity (every cycle) ─────────────────
        with tempfile.TemporaryDirectory(prefix="track3_") as tmp3:
            track3_result = track3_integrity.check_csv_and_profile_roundtrip(
                quantify_result, Path(tmp3) / "colony_features.csv")
            load_test_cfg = matrix.get("track3_load_test", {})
            if load_test_cfg and cycle_number % load_test_cfg.get("run_every_n_cycles", 10) == 0:
                throughput = track3_integrity.run_load_test(
                    load_test_cfg["batch_sizes"], Path(tmp3) / "load_test.csv")
                phase_c.record_track3(state, track3_result["passed"], throughput)
            else:
                phase_c.record_track3(state, track3_result["passed"])
        passed = passed and track3_result["passed"]

        # ── Track 4 — full system dry run (every cycle) ─────────────────────
        with tempfile.TemporaryDirectory(prefix="track4_") as tmp4:
            track4_result = track4_dryrun.run_dry_run(Path(tmp4), seed=cycle_number)
            phase_c.record_track4(state, track4_result["passed"], track4_result["latency_s"])
        passed = passed and track4_result["passed"]

        # ── Track 2 — anomaly model validation (every Nth cycle) ────────────
        track2_cfg = matrix.get("track2", {})
        run_every = track2_cfg.get("run_every_n_cycles", 5)
        if track2_cfg and cycle_number % run_every == 0:
            try:
                from testing.continuous import track2_ml_validation
            except ImportError:
                track2_ml_validation = None
            if track2_ml_validation is not None:
                track2_result = track2_ml_validation.evaluate(track2_cfg, seed=cycle_number)
                phase_c.record_track2(state, cycle_id, track2_result)

    except Exception:
        passed = False
        raise
    finally:
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        diff = phase_c.finalize_cycle(state, cycle_id, timestamp_utc, passed, track1_score)
        phase_c.save_aggregate_state(state, STATE_PATH)
        phase_c.write_aggregate_report(state, REPORTS_DIR / "aggregate_report.md")

        summary = {
            "cycle_id": cycle_id, "timestamp_utc": timestamp_utc,
            "scenario_config": scenario_cfg, "passed": passed,
            "track1": track1_score, "track3": track3_result, "track4": track4_result,
            "track2": track2_result, "diff_vs_previous": diff,
        }
        (LOGS_DIR / "post" / f"cycle_{cycle_id}.json").write_text(json.dumps(summary, indent=2, default=str))
        phase_c.write_cycle_report(cycle_id, summary, REPORTS_DIR)

        # ── Memory/disk discipline — see plan's Hardware constraints ────────
        del image
        gc.collect()
        _prune_dir(ARTIFACTS_DIR, retention, glob="*")
        _prune_dir(LOGS_DIR / "pre", max(retention, 20) * 4, glob="*.json")
        _prune_dir(LOGS_DIR / "during", max(retention, 20) * 4, glob="*.json")
        _prune_dir(LOGS_DIR / "post", max(retention, 20) * 4, glob="*.json")
        _prune_dir(REPORTS_DIR, max(retention, 20) * 8, glob="cycle_*_summary.*")

    return summary


if __name__ == "__main__":
    result = run_one_cycle()
    print(json.dumps({"cycle_id": result["cycle_id"], "passed": result["passed"]}))
    sys.exit(0 if result["passed"] else 1)
