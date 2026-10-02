# Continuous testing harness

Pre-validation testing scope for the plate imaging pipeline: exercises
`quantify.py`, `anomaly.py`, `data_logger.py`, `profiles.py`, and `server.py`
against synthetic data continuously, since real biological validation is
out of scope for this semester. See `testing/reports/aggregate_report.md`
for the accumulated, auto-updated results.

## `continuous/synthetic_data.py`

### `generate_plate_image(PlateScenario) -> (image_bgr, ground_truth)`

Renders a synthetic backlit agar plate (bright disc on black background,
darker colony discs — matches the same convention already used in
`tests/test_quantify.py::_synthetic_plate`, since `quantify.py`'s background
subtraction is `blurred_background - image`: a colony only survives if it's
darker than its local surroundings).

Parameters, and why each one exists:

- **`seed`** — every generator call is a pure function of its seed. Nothing
  the harness produces from a scenario is ever stored long-term; Phase A
  logs the seed and scenario, and the exact image is reproduced later by
  calling the generator again. This is what keeps `testing/artifacts/` safe
  to prune aggressively on a Pi with limited SD storage.
- **`illumination`** (`uniform` / `gradient` / `hotspot` / `low_contrast`) —
  covers the lighting conditions the physical rig can plausibly produce:
  even backlighting, an off-axis brightness gradient, a diffuser hotspot,
  and reduced agar/colony contrast. `low_contrast` also shrinks the
  colony/agar brightness delta directly, since real low-contrast conditions
  reduce edge strength, not just introduce a spatial gradient.
- **`density`** (`sparse` / `moderate` / `dense`) — colony count ranges
  loosely modeled on early/mid/late culture growth. `dense` also has a 25%
  per-colony chance of being placed overlapping the previous colony, so the
  watershed "touching colony" path is actually exercised on some cycles
  rather than only ever seeing isolated colonies.
- **`artifacts`** (`streak` / `debris` / `oversized_blob`) — deliberately
  *wrong* shapes a real image could contain: a thin elongated smear, a
  sub-threshold speck, and an oversized blob. `debris` is sized below
  `quantify.MIN_AREA` defaults and `oversized_blob` above `MAX_AREA`
  defaults on purpose — ground truth expects the pipeline to filter both
  out, not count them as colonies.
- **`camera_distance_factor`** — scales only the plate's *pixel* radius;
  colony sizes are generated in mm and converted to pixels from the
  scenario's own derived `px_per_mm` (via `expected_px_per_mm`, which
  replicates `quantify_colonies()`'s own rim-shrink/calibration math). This
  is what makes the distance sweep in `test_matrix.yaml` a direct test of
  the pipeline's claimed distance-invariance, rather than a proxy for it.

Colony radii are drawn from `COLONY_RADIUS_MM_RANGE` (0.6–2.2 mm), chosen to
sit well inside `quantify_colonies()`'s default area filter (0.1–20 mm²) so
"normal" synthetic colonies aren't accidentally excluded by area alone.

### `generate_colony_features(FeatureScenario) -> DataFrame`

Produces per-colony feature rows matching `anomaly.ML_FEATURES` exactly
(so they train/evaluate against the real `MLDetector` without a schema
adapter), for Track 2 anomaly-model validation.

- **`anomaly_fraction`** — fraction of rows drawn from `_draw_anomalous_colony`
  instead of `_draw_normal_colony`. Anomalous rows are shifted using
  `quantify.py`'s own thresholds (`ANOMALY_Z_THRESHOLD`,
  `NON_CIRCULAR_THRESHOLD`, `STREAK_ASPECT_RATIO_THRESHOLD`,
  `HEMOLYSIS_DELTA_THRESHOLD`) so a synthetic "anomaly" is one the
  production flagging logic would plausibly also flag — not an arbitrary
  out-of-distribution point that would make the benchmark too easy.
- **`label_noise`** — fraction of labels flipped in the `is_anomaly` column
  only; `true_is_anomaly` stays clean. This models `data_logger.py`'s real
  labelling path (`apply_validation()` is filled in by human review, not an
  oracle) — models are trained on the noisy column and evaluated against
  the clean one, so Track 2's precision/recall/F1 numbers reflect
  robustness to imperfect manual labelling, not just curve-fitting on
  perfect synthetic labels.

## `continuous/track2_ml_validation.py` — model comparison methodology

Every `run_every_n_cycles`-th cycle (`test_matrix.yaml`'s `track2` block),
`evaluate()` generates one fresh `generate_colony_features()` batch and
runs a 70/30 train/test split (stratified on the clean `true_is_anomaly`
label, so both splits keep the same anomaly ratio regardless of how rare
anomalies are configured to be).

Three models are trained on the same split and compared:

- **Random Forest** — evaluated through the actual production
  `anomaly.MLDetector` class (writes the training split to a throwaway CSV
  and calls `MLDetector.train()`/`.predict()` exactly as `server.py` would),
  not a re-implementation. This is deliberate: Track 2 benchmarks the model
  that ships, not a lookalike that could silently drift from it if
  `anomaly.py`'s hyperparameters ever change.
- **Gradient boosting** — `sklearn.ensemble.GradientBoostingClassifier` in
  a `StandardScaler` pipeline, trained the same way.
- **Neural net** — `sklearn.neural_network.MLPClassifier` (a small
  `(32, 16)` hidden-layer network) in the same pipeline shape. Chosen over
  a deep-learning framework specifically to avoid adding a TensorFlow/
  PyTorch dependency for one comparison model — `scikit-learn` is already
  in `requirements.txt`.

All three train on the **noisy** `is_anomaly` column and are scored against
the **clean, held-out** `true_is_anomaly` column — see
`generate_colony_features()` above. This means the reported precision/
recall/F1 answers "how well does this model generalize despite imperfect
training labels", which is the actual condition `data_logger.py`'s
real-world CSV will be trained from, not "how well does this model
memorize noise-free synthetic data" (a strictly easier and less meaningful
question).

Results are **not** averaged into a single "the model is X% accurate"
number and left there — `aggregate_report.md`'s Track 2 section keeps a
rolling history (capped, per the plan's memory-discipline constraints) of
every evaluation, specifically so the thesis can show how model comparison
results evolved as the synthetic feature generator itself was refined over
time, per the original testing scope's requirement.
