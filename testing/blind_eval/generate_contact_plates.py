"""
generate_contact_plates.py : write the realistic-contact ladder plate sets
to disk, in the same layout generate_held_out_plates.py produces so
run_blind_eval.py can score them with no change to the harness.

Why this is a separate script and a separate seed base rather than four
more entries in generate_held_out_plates.py: the existing five held-out
sets and their committed results in testing/blind_eval/results/ are the
baseline this ladder is measured against. Pulling new sets into that
script's default run would change the pooled numbers in those committed
files, which would destroy the comparison. The seed base here is disjoint
from DEFAULT_SEED_BASE's range, so no ladder plate can collide with a
held-out plate and quietly reuse a scenario the pipeline has already seen.

Output is regenerable from (set_name, seed) via
synthetic_data.generate_contact_plate(), so plate images and ground truth
are gitignored exactly as testing/blind_eval/plates/ is. What's committed
is this script plus its seed scheme.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from testing.continuous.synthetic_data import CONTACT_SET_NAMES, generate_contact_plate

DEFAULT_OUTPUT_DIR = Path("testing/blind_eval/contact_plates")
DEFAULT_PLATES_PER_SET = 50
# Disjoint from generate_held_out_plates.DEFAULT_SEED_BASE (900_000, whose
# five sets reach 940_049). Eight ladder sets from here reach 2_070_049, so
# the two ranges cannot overlap and a seed is recognisable as a ladder seed.
CONTACT_SEED_BASE = 2_000_000


def _seed_for(seed_base: int, set_index: int, plate_index: int) -> int:
    return seed_base + set_index * 10_000 + plate_index


def generate_set(set_name: str, set_index: int, n_plates: int, seed_base: int,
                  output_dir: Path) -> list[dict]:
    set_dir = output_dir / set_name
    set_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for i in range(n_plates):
        seed = _seed_for(seed_base, set_index, i)
        image, ground_truth = generate_contact_plate(set_name, seed)
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
    # set_index is the position in CONTACT_SET_NAMES, not in the requested
    # subset, so generating one set alone gives it the same seeds it gets in
    # a full run. generate_held_out_plates.py indexes by request position
    # instead, which is why its --sets flag has to be used with care.
    for set_name in sets:
        rows = generate_set(set_name, CONTACT_SET_NAMES.index(set_name),
                             n_plates, seed_base, output_dir)
        manifest["sets"][set_name] = rows
    manifest["total_plates"] = sum(len(rows) for rows in manifest["sets"].values())
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", nargs="+", choices=CONTACT_SET_NAMES,
                         default=list(CONTACT_SET_NAMES),
                         help="Which ladder sets to generate (default: all eight).")
    parser.add_argument("--plates-per-set", type=int, default=DEFAULT_PLATES_PER_SET)
    parser.add_argument("--seed-base", type=int, default=CONTACT_SEED_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = generate_all(args.sets, args.plates_per_set, args.seed_base, args.output_dir)
    print(f"Generated {result['total_plates']} contact-ladder plates across "
          f"{len(result['sets'])} sets under {args.output_dir}/")
