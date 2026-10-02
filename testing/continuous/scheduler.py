#!/usr/bin/env python3
"""
scheduler.py — runs test cycles continuously or on a defined cadence.

Two operating modes:

    --mode continuous   Round-robins the test matrix indefinitely, sleeping
                         between cycles. Meant to run in the background on
                         the Pi/dev machine during active development —
                         start it with `python -m testing.continuous.scheduler
                         --mode continuous`. Self-throttles under memory
                         pressure (see below) since it may share the Pi with
                         the real capture app.

    --mode ci            Runs exactly one full pass over
                         test_matrix.yaml's track1_scenarios (one run_cycle()
                         call per scenario) and exits — this is what the
                         GitHub Action calls on every push.

Memory discipline in continuous mode: this may run unattended for a long time on a
Raspberry Pi that also runs the real capture app, so it checks its own
process RSS every cycle via resource.getrusage and backs off (a longer
sleep) once it crosses `--rss-limit-mb`, rather than free-running as fast as
it can regardless of memory pressure.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import yaml

from testing.continuous import run_cycle

DEFAULT_CYCLE_INTERVAL_S = 30.0
DEFAULT_RSS_LIMIT_MB = 512
BACKOFF_MULTIPLIER = 4.0


def _current_rss_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS — normalize by platform.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024 if sys.platform != "darwin" else raw / (1024 * 1024)


def run_continuous(interval_s: float, rss_limit_mb: float) -> None:
    print(f"scheduler: continuous mode — interval={interval_s}s, "
          f"rss_limit={rss_limit_mb}MB", flush=True)
    while True:
        try:
            summary = run_cycle.run_one_cycle()
            print(f"scheduler: cycle {summary['cycle_id']} "
                  f"{'PASS' if summary['passed'] else 'FAIL'}", flush=True)
        except Exception as e:
            print(f"scheduler: cycle raised {e!r} — continuing", flush=True)

        rss_mb = _current_rss_mb()
        sleep_s = interval_s
        if rss_mb > rss_limit_mb:
            sleep_s = interval_s * BACKOFF_MULTIPLIER
            print(f"scheduler: RSS {rss_mb:.0f}MB over limit "
                  f"{rss_limit_mb:.0f}MB — backing off to {sleep_s:.0f}s", flush=True)
        time.sleep(sleep_s)


def run_ci() -> int:
    """One full matrix pass, deterministic, exits with a real status code —
    this is the unit `.github/workflows/continuous-testing.yml` calls."""
    matrix = yaml.safe_load(run_cycle.MATRIX_PATH.read_text())
    n_scenarios = len(matrix["track1_scenarios"])
    print(f"scheduler: CI mode — running {n_scenarios} cycles (one full matrix pass)", flush=True)

    all_passed = True
    for i in range(n_scenarios):
        summary = run_cycle.run_one_cycle()
        status = "PASS" if summary["passed"] else "FAIL"
        print(f"scheduler: cycle {summary['cycle_id']} {status} "
              f"({summary['scenario_config']})", flush=True)
        all_passed = all_passed and summary["passed"]

    return 0 if all_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["continuous", "ci"], required=True)
    parser.add_argument("--interval", type=float, default=DEFAULT_CYCLE_INTERVAL_S,
                        help="Seconds between cycles in continuous mode.")
    parser.add_argument("--rss-limit-mb", type=float, default=DEFAULT_RSS_LIMIT_MB,
                        help="Back off cycle cadence above this process RSS (continuous mode only).")
    args = parser.parse_args()

    if args.mode == "continuous":
        run_continuous(args.interval, args.rss_limit_mb)
        return 0  # unreachable — continuous mode runs until killed
    return run_ci()


if __name__ == "__main__":
    sys.exit(main())
