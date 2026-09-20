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

## Iteration 2
- Hypothesis: replace threshold-based sure-foreground (global or per-blob)
  with marker seeding by local maxima of the distance transform — one
  marker per peak within a `WATERSHED_MIN_PEAK_DISTANCE_PX` window, so a
  touching pair of *different* sizes each gets its own marker instead of
  the smaller colony's peak being swallowed by whichever threshold the
  larger one sets.
- Parameters changed: `_apply_watershed()` now seeds markers via
  non-maximum suppression over the distance transform (window size
  `2*WATERSHED_MIN_PEAK_DISTANCE_PX+1`) instead of a sure-foreground
  threshold at all — this supersedes iteration 1's per-blob threshold with
  a strictly more general technique (a local-max window doesn't care what
  blob a pixel belongs to, so it can't be swamped by a neighboring blob's
  scale the way any single threshold — global or per-blob — still can).
  `WATERSHED_FG_THRESHOLD_FRAC` is removed (no longer used anywhere) in
  favor of `WATERSHED_MIN_PEAK_DISTANCE_PX = 4`.
- `WATERSHED_MIN_PEAK_DISTANCE_PX` was swept over {2, 3, 4, 6, 8} on a
  15-plate sample of dense_pack/high_touching before picking a value:
  recall was flat within noise across the whole range (dense_pack F1
  0.847-0.853, high_touching F1 0.780-0.791, precision 1.000 throughout).
  4 was selected as a tied-best value, not because it's meaningfully better
  than its neighbors — the flatness itself is the finding (see below).
- Results (full 250-plate blind eval):

  | set | F1 (iter 1) | F1 (iter 2) | recall (iter 1) | recall (iter 2) | status |
  |---|---|---|---|---|---|
  | dense_pack | 0.840 | 0.846 | 0.724 | 0.733 | still FAIL |
  | high_touching | 0.782 | 0.787 | 0.642 | 0.649 | still FAIL |
  | size_variance | 0.956 | 0.957 | 0.915 | 0.917 | PASS |
  | irregular_morphology | 0.994 | 0.998 | 0.989 | 0.996 | PASS |
  | poor_illumination | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
  | **Overall (pooled)** | **0.872** | **0.877** | 0.774 | 0.781 | still FAIL vs 0.90 |

  Precision held at 1.000 everywhere. Delta F1: overall +0.005 — real but
  small, a fraction of iteration 1's +0.068.
- Root-cause finding (why the gain is small): before spending iteration 3 on
  the doc's #3 candidate (`bg_blur_kernel`), checked whether watershed
  itself is still the bottleneck by counting raw watershed contours before
  any area/circularity/aspect filtering, on a 15-plate sample:
  - dense_pack: 1087 raw contours vs 1443 expected colonies (filters then
    drop only 14 of those 1087 — 13 by area, 1 by circularity, 0 by aspect).
  - high_touching: 331 raw contours vs 494 expected (filters drop 8, all by
    area).
  So the shortfall is essentially 100% a segmentation problem (watershed
  not producing enough distinct regions), not a filtering problem — the
  downstream area/circularity/aspect thresholds are barely touching it, so
  further loosening them (in either direction) would not help.
  Directly inspected why: sampled the distance-transform value along the
  line between ground-truth centres for every touching-flagged pair on
  `high_touching/plate_020` (21 touching colonies). The profiles are
  monotonic in nearly every case — e.g. one pair (radii 10.5px and 7.5px,
  centre distance 5.8px vs a summed radius of 18.0px, i.e. heavy overlap)
  reads `11.0, 11.0, 11.0, 11.0, 11.0, 10.2, 9.8, ..., 7.0` walking from one
  centre to the other: a single smooth slope, no saddle/dip between the two
  colonies at all. When circles overlap this much, the merged shape's
  distance transform genuinely has only one local maximum — there is no
  second peak for any marker-seeding method (threshold-based or local-max)
  to find, regardless of window size or threshold value. That's why the
  `WATERSHED_MIN_PEAK_DISTANCE_PX` sweep was flat: the parameter isn't
  underperforming, the underlying signal it depends on doesn't exist for
  these pairs. **This is an architectural ceiling of distance-transform
  watershed for this degree of colony overlap, not a tuning gap.**
  Splitting these would need a different technique entirely — e.g. Hough
  circle fitting inside each blob, or concave-point contour analysis to
  find where two circle boundaries meet even under overlap — which is a
  new pipeline stage, not a parameter change, and out of scope for this
  iteration loop as scoped.
- `bg_blur_kernel` (doc's #3 candidate) swept over {151, 201, 251, 301} on
  the same 15-plate dense_pack/high_touching sample with the iteration-2
  watershed: **identical results at every value** (dense_pack F1 0.853,
  high_touching F1 0.791, unchanged to 3 decimal places). Ruled out for
  these two sets — neither stresses illumination (both are density/touching
  stressors, not gradient/hotspot), so the background model isn't the
  constraint here and a larger kernel has nothing to fix.
- All three doc-prescribed parameters (`min_area_mm2`, watershed
  splitting, `bg_blur_kernel`) have now been tested and exhausted for
  dense_pack/high_touching. Two iterations of watershed changes closed
  most of the gap (F1 0.804 → 0.877 overall); the remainder is a proven
  geometric limit of the current segmentation technique, not a parameter
  available in this loop's scope.

## Stopping decision — after iteration 2

Stopped after 2 of the allowed 5 iteration cycles, below the F1 > 0.90
target, by explicit decision rather than by exhausting the cycle count.

- **Final overall F1: 0.877** (target 0.90). Precision 1.000 on every set,
  every cycle — the pipeline never invented a colony; recall is the entire
  story, on both what was fixed and what remains.
- **3 of 5 held-out sets pass:** `size_variance` (0.957), `poor_illumination`
  (1.000), `irregular_morphology` (0.998).
- **2 of 5 fail:** `dense_pack` (0.846), `high_touching` (0.787).
- **Why stop before 5 cycles:** all three parameters this loop was scoped to
  tune (`min_area_mm2`, watershed splitting, `bg_blur_kernel`) were tested to
  exhaustion — one ruled out empirically before spending a cycle on it, one
  fixed twice for the bulk of the gain, one ruled out by direct sweep with
  zero effect. What's left on dense_pack/high_touching is not a parameter
  gap: sampling the distance-transform between touching ground-truth colony
  centres showed the merged shape has only one local maximum whenever two
  colonies overlap heavily — there is no second peak for any threshold or
  window-based marker method to find, watershed or otherwise, at this
  overlap level. Burning the remaining 3 cycles on further variants of the
  same three parameters would not have changed this; closing it needs a
  different segmentation technique (Hough-circle fitting inside merged
  blobs, or concave-point contour splitting), which is new pipeline
  architecture, not a parameter tune, and out of this loop's scope.
- **This gap is real and goes in the memo's limitations section as-is:**
  the pipeline is commercially comparable to published open-source counter
  accuracy (~93-97%, see Branch 1's benchmark comparison) on 3 of 5
  held-out stress conditions, including the ones stressing illumination,
  morphology irregularity, and size variance. It underperforms specifically
  on plates with dense, heavily-overlapping colonies — a known, named,
  measured limitation, not a hidden one.
