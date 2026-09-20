"""Tests for the held-out plate generation script. Uses small counts/output
dirs under tmp_path — the real 250-plate run is exercised manually, not in
the test suite, matching how run_cycle.py's tests avoid a real long run."""
import json

import cv2

from testing.blind_eval.generate_held_out_plates import generate_all, generate_set
from testing.continuous.synthetic_data import HELD_OUT_SET_NAMES


def test_generate_set_writes_image_and_ground_truth_per_plate(tmp_path):
    rows = generate_set("dense_pack", set_index=0, n_plates=3, seed_base=1_000, output_dir=tmp_path)
    assert len(rows) == 3
    for row in rows:
        image = cv2.imread(row["image_path"])
        assert image is not None
        gt = json.loads(open(row["ground_truth_path"]).read())
        assert gt["expected_count"] == row["expected_count"]
        assert gt["held_out_set"] == "dense_pack"


def test_generate_all_covers_every_requested_set_and_writes_manifest(tmp_path):
    manifest = generate_all(list(HELD_OUT_SET_NAMES), n_plates=2, seed_base=2_000, output_dir=tmp_path)
    assert manifest["total_plates"] == 2 * len(HELD_OUT_SET_NAMES)
    assert set(manifest["sets"].keys()) == set(HELD_OUT_SET_NAMES)

    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest


def test_generate_all_seeds_do_not_collide_across_sets(tmp_path):
    manifest = generate_all(list(HELD_OUT_SET_NAMES), n_plates=2, seed_base=3_000, output_dir=tmp_path)
    all_seeds = [row["seed"] for rows in manifest["sets"].values() for row in rows]
    assert len(all_seeds) == len(set(all_seeds))


def test_generate_all_is_deterministic(tmp_path):
    out1, out2 = tmp_path / "a", tmp_path / "b"
    m1 = generate_all(["irregular_morphology"], n_plates=2, seed_base=4_000, output_dir=out1)
    m2 = generate_all(["irregular_morphology"], n_plates=2, seed_base=4_000, output_dir=out2)
    rows1, rows2 = m1["sets"]["irregular_morphology"], m2["sets"]["irregular_morphology"]
    for r1, r2 in zip(rows1, rows2):
        assert r1["seed"] == r2["seed"]
        assert r1["expected_count"] == r2["expected_count"]
        assert r1["expected_anomaly_count"] == r2["expected_anomaly_count"]
        img1 = cv2.imread(r1["image_path"])
        img2 = cv2.imread(r2["image_path"])
        assert (img1 == img2).all()
