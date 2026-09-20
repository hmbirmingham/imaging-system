# Stage-by-Stage Visualization Export

5 representative plates across difficulty tiers, each selected by an explicit rule against ground truth (see export_stage_visualizations.py's select_representative_plates() for the exact criteria per tier).

| tier | plate | expected count | anomalies | touching pairs |
|---|---|---|---|---|
| Easy: Sparse, Normal Illumination | `irregular_morphology/plate_004` | 12 | 1 | 0 |
| Medium: Moderate Density, Light Touching | `high_touching/plate_011` | 29 | 5 | 5 |
| Hard: Dense Colony Packing | `dense_pack/plate_040` | 100 | 26 | 26 |
| High-Anomaly: Irregular Morphology | `irregular_morphology/plate_012` | 20 | 6 | 0 |
| Worst Case: Lowest-F1 Held-Out Plate (F1=0.513) | `high_touching/plate_039` | 29 | 16 | 16 |
