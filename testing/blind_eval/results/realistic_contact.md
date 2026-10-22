# Realistic-contact evaluation

Hynes Birmingham, Ryberg Lab. Branch `feat/realistic-contact-eval`.

Status: **pre-registered, not yet run.** The decision rule below is
committed before Phase 3 executes. Results tables are appended after.

## Why I am running this

My blind eval is synthetic-only, and the generator renders forced-touching
colonies in a way real colonies do not grow. Two specific problems:

1. **Contact depth.** A forced-touching colony is placed at a centre
   distance of `(r1 + r2) * (1 - U(0.3, 0.7))`, so 30 to 70% of the radius
   sum. Real colonies expand until their edges meet and then stop, which
   puts contact near `r1 + r2`, not at half of it.
2. **No edge cue.** Every colony is drawn with one flat fill value, so two
   colonies that meet render as a single indivisible region. Under real
   lighting their touching edges leave a visible boundary line. The
   pipeline is therefore being asked to split a shape that contains no
   evidence a split is warranted.

The consequence is measurable. Scoring each colony's `visible_fraction`,
the share of its own disc not covered by any other disc:

| set | colonies | under 10% visible | under 25% visible | exactly 0% visible |
|---|---|---|---|---|
| `dense_pack` | 4931 | 5.05% | 8.86% | 2.43% |
| `high_touching` | 1680 | 6.73% | 12.26% | 3.21% |

A colony at 0% visible contributes no silhouette area that another colony
does not already account for. No segmentation method can recover it, and
ground truth still counts it. So part of my recall gap is unreachable by
any method, and I cannot currently tell how much.

This test separates "my method fails" from "my generator is unfair."

## Method

Four-step ladder, 50 plates per step, holding colony count and touching
rate constant and changing only how contact is placed and rendered.

| step | rendering |
|---|---|
| `r0` | current deep-overlap rendering (control) |
| `r1` | `abut`: contact distance `(r1 + r2) * U(0.90, 1.00)` |
| `r2` | `abut` + 1.5px rim ring, 25 intensity units darker |
| `r3` | `abut` + rim ring + dome shading + 0.8px blur + sensor noise sd 2.0 |

Run twice: `contact_r0` to `r3` match `high_touching`'s colony count and
touching rate, `contact_dense_r0` to `r3` match `dense_pack`'s.

Seed base 2,000,000, disjoint from the held-out sets' 900,000 range. The
existing harness scores these unchanged: ground truth is loaded only after
detection, one-to-one greedy nearest-centroid matching, 1.5x radius gate,
pooled micro-averaged scoring. No `quantify.py` default was changed on
this branch.

Caveat on the ladder, stated before results: `r0` to `r1` changes two
things at once, contact placement and rasterisation, because sub-pixel
centres are only used in the new modes. `r1` to `r2` to `r3` each change
one thing.

## Pre-registered decision rule

Copied verbatim from the test plan, fixed before Phase 3 ran. The rule
applies to `contact_r3` and `contact_dense_r3`.

- **Method holds:** pooled F1 at least 0.90 on both. My read: the
  deep-overlap generator was the ceiling, and real plates are the only
  blocker.
- **Method holds on what's visible:** F1 below 0.90, but at least 80% of
  the misses are colonies with `visible_fraction` under 0.25. My read:
  same as above, and I report it with the visibility breakdown.
- **Real weakness:** F1 below 0.90, and the misses are mostly visible
  colonies. My read: a method problem. I record which part of the ladder
  (r1, r2 or r3) caused the drop. That points at the first fix to try: a
  dilution protocol, then concave-point splitting or multi-scale seeds,
  then a learned detector trained on real labelled plates.
