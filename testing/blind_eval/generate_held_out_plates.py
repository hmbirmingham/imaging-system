"""
generate_held_out_plates.py — write the blind-validation held-out plate set
to disk: images the pipeline evaluates blind, plus ground-truth JSON the
pipeline never sees during that evaluation (run_blind_eval.py reads it only
afterward, to score what the pipeline returned).

Output is regenerable — deterministic from (set_name, seed) via
synthetic_data.generate_held_out_plate() — so, matching this repo's existing
testing/logs, testing/artifacts, testing/thesis_export/generated convention,
plate images and their ground truth are gitignored. What's committed is this
script (the generator) plus its seed scheme, not the 250 files it produces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from testing.continuous.synthetic_data import HELD_OUT_SET_NAMES, generate_held_out_plate

DEFAULT_OUTPUT_DIR = Path("testing/blind_eval/plates")
DEFAULT_PLATES_PER_SET = 50
# Offset from any seed range used elsewhere in the codebase, purely so a
# held-out plate's seed is recognizable as such in logs/tracebacks.
DEFAULT_SEED_BASE = 900_000


def _seed_for(seed_base: int, set_index: int, plate_index: int) -> int:
    return seed_base + set_index * 10_000 + plate_index


def generate_set(set_name: str, set_index: int, n_plates: int, seed_base: int,
                  output_dir: Path) -> list[dict]:
    set_dir = output_dir / set_name
    set_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for i in range(n_plates):
        seed = _seed_for(seed_base, set_index, i)
        image, ground_truth = generate_held_out_plate(set_name, seed)
        stem = f"plate_{i:03d}"
        cv2.imwrite(str(set_dir / f"{stem}.png"), image)
        (set_dir / f"{stem}.json").write_text(json.dumps(ground_truth, indent=2))
        manifest_rows.append({
            "held_out_set": set_name, "seed": seed,
            "image_path": str(set_dir / f"{stem}.png"),
            "ground_truth_path": str(set_dir / f"{stem}.json"),
            "expected_count": ground_truth["expected_count"],
            "expected_anomaly_count": ground_truth["expected_anomaly_count"],
        })
    return manifest_rows


def generate_all(sets: list[str], n_plates: int, seed_base: int,
                  output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed_base": seed_base, "plates_per_set": n_plates, "sets": {}}
    for set_index, set_name in enumerate(sets):
        rows = generate_set(set_name, set_index, n_plates, seed_base, output_dir)
        manifest["sets"][set_name] = rows
    manifest["total_plates"] = sum(len(rows) for rows in manifest["sets"].values())
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out", action="store_true", required=True,
                         help="Confirms intent to generate held-out (not normal-range) plates.")
    parser.add_argument("--sets", nargs="+", choices=HELD_OUT_SET_NAMES,
                         default=list(HELD_OUT_SET_NAMES),
                         help="Which held-out parameter sets to generate (default: all five).")
    parser.add_argument("--plates-per-set", type=int, default=DEFAULT_PLATES_PER_SET)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = generate_all(args.sets, args.plates_per_set, args.seed_base, args.output_dir)
    print(f"Generated {result['total_plates']} held-out plates across "
          f"{len(result['sets'])} sets under {args.output_dir}/")
