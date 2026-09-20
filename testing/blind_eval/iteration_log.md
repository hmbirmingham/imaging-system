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
