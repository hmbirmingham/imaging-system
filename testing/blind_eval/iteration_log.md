# Blind Validation Iteration Log

## Baseline (Iteration 0)
- Date: 2026-09-20
- Overall F1: 0.804
- Passing: irregular_morphology (0.920), poor_illumination (0.914)
- Failing: size_variance (0.862), dense_pack (0.778), high_touching (0.725)
- Anomaly detection F1: 0.135-0.208 across all sets
- Root cause: recall failure on dense/touching plates (precision 1.000 on every
  set — the pipeline never invents colonies, it misses real ones under stress)
- Visual evidence: `testing/blind_eval/failure_analysis/iteration_0/` —
  watershed frames show clusters of touching ground-truth colonies merged
  into a single watershed region (undersegmentation); several such merged
  blobs never appear in the contour overlay at all, meaning they're filtered
  out downstream (by `min_circularity`/`max_aspect_ratio`) rather than
  miscounted as one colony
- Parameters changed: none (baseline)

## Iteration 1
- Hypothesis: doc's prescribed tuning order starts with `min_area_mm2`
  ("lower threshold to catch smaller colonies missed at high density").
  Checked this empirically against the iteration-0 failure data before
  spending a cycle on it: **zero** missed colonies in any failing set have
  `area_mm2` below the current `min_area_mm2 = 0.1` (smallest missed area
  across dense_pack/high_touching/size_variance was 1.13 mm², over 10x the
  threshold). Lowering `min_area_mm2` would have been a no-op — skipped it
  and moved to the doc's #2 candidate (watershed splitting), which the
  iteration-0 visuals directly implicated: clusters of touching
  ground-truth colonies merged into one watershed region, and several such
  merged blobs never appear in the contour overlay at all (filtered out
  downstream, not miscounted as one).
- Root cause found: `_apply_watershed()` (`quantify.py`) thresholded
  "sure foreground" against the distance transform's **global** peak across
  the entire plate (`WATERSHED_FG_THRESHOLD_FRAC * dist_transform.max()`),
  not per blob. On a plate with mixed colony sizes, one large colony sets a
  high global peak; every other blob's own peak is compared against that
  same global cutoff. Measured directly on `dense_pack/plate_038`: global
  peak distance 18.98px → threshold 7.59px, and **15 of 73 blobs (20.5%)**
  had their own peak distance entirely below that threshold — meaning they
  produced zero "sure foreground" pixels and were dropped before watershed
  ever assigned a marker, not merged or misclassified but absent outright.
  This is on top of (and largely explains) the touching-colony
  undersegmentation visible in the iteration-0 frames, since a touching
  cluster's peak is also compared against the same plate-wide max.
- Parameters changed: `_apply_watershed()` now computes the sure-foreground
  threshold **per pre-watershed connected component** (each blob's own
  distance-transform peak), instead of one threshold for the whole plate.
  `WATERSHED_FG_THRESHOLD_FRAC` itself is unchanged (0.4) — this is a
  normalization-scope fix, not a threshold-value change.
- Results (full 250-plate blind eval, `python -m testing.blind_eval.run_blind_eval`):

  | set | F1 before | F1 after | recall before | recall after | precision | status |
  |---|---|---|---|---|---|---|
  | dense_pack | 0.778 | 0.840 | 0.637 | 0.724 | 1.000 → 1.000 | still FAIL |
  | high_touching | 0.725 | 0.782 | 0.569 | 0.642 | 1.000 → 1.000 | still FAIL |
  | size_variance | 0.862 | **0.956** | 0.757 | 0.915 | 1.000 → 1.000 | **now PASS** |
  | irregular_morphology | 0.920 | 0.994 | 0.853 | 0.989 | 1.000 → 1.000 | PASS (unchanged) |
  | poor_illumination | 0.914 | **1.000** | 0.842 | 1.000 | 1.000 → 1.000 | **now PASS** |
  | **Overall (pooled)** | **0.804** | **0.872** | 0.672 | 0.774 | 1.000 → 1.000 | still FAIL vs 0.90 |

  Precision stayed at 1.000 on every set — the fix only recovers coverage
  within pixels the pipeline had already flagged as candidate foreground,
  it doesn't introduce new detections from noise, so no new false positives
  were expected or observed.
- Delta F1: overall +0.068 (0.804 → 0.872). 3 of 5 sets now pass (up from 2).
- Visual re-check (`testing/blind_eval/failure_analysis/iteration_1/`,
  dense_pack + high_touching only — the other 3 sets pass now): re-ran the
  false-negative composition check on the remaining misses. The
  touching-flagged share of what's still missed went **up**, not down —
  dense_pack 41.1% → 53.4%, high_touching 49.3% → 58.5%. Consistent with
  the fix disproportionately recovering the "isolated colony entirely
  absent from sure_fg" failure mode (now fixed), leaving a purer, more
  concentrated remainder of genuinely touching colonies that watershed
  still fails to *split* — sure_fg no longer drops these blobs, but a
  touching pair/cluster's combined blob can still yield only one connected
  sure_fg island if the saddle between the two colonies' distance-transform
  peaks doesn't dip far enough below the per-blob threshold. That's the
  next hypothesis for iteration 2.
