# Blind Validation Methodology & Benchmark Sourcing

## What this validation measures

`testing/blind_eval/run_blind_eval.py` runs `quantify_colonies()` blind
(image path only) against 250 synthetic plates generated with held-out
parameter ranges — organism density, illumination, colony size variance,
and touching-colony rate combinations never used while developing the
pipeline. Ground truth (exact colony positions, radii, anomaly labels) is
loaded only after the pipeline returns its result, to score what it
produced rather than to inform it. Metrics are pooled (micro-averaged)
across all plates in a held-out set: true/false positives and false
negatives are summed before computing one precision/recall/F1, rather than
averaging each plate's own F1, so a handful of near-empty plates can't
disproportionately swing the aggregate. See `run_blind_eval.py`'s own
docstring for the exact matching rule (nearest-centroid, one-to-one, gated
by 1.5x the ground-truth colony's radius).

**This is a synthetic-only validation.** No real biological plates are
used anywhere in this loop. That's a deliberate, stated scope limit (see
the sprint's own framing) — it demonstrates the pipeline generalizes to
morphological conditions it wasn't tuned on, not that it matches biological
reality. Biological validation is explicitly future work.

## Why the benchmark comparison doesn't use one unified accuracy column

The original plan for this comparison assumed each commercial/open-source
counter had a single citable "accuracy" percentage (~94% OpenCFU, ~96%
ColonyDoc-It, ~93-97% aCOLyte). Checking those against the primary
literature directly (not secondhand summaries) instead of taking the
placeholders as given: none of the three numbers held up.

- **OpenCFU's own validation paper** (Geissmann 2013) doesn't report a
  percentage accuracy at all — it reports median absolute colony-count
  error (3 colonies, on 10-1000-colony plates) with no statistically
  significant bias vs. manual counts. A separate, independent 2025
  re-benchmark on a harder, denser dataset found OpenCFU's mean error at
  50.31% — a very different picture from the same tool, because the test
  data was harder, not because either paper is wrong.
- **ColonyDoc-It's** only located peer-reviewed validation (Heuser et al.
  2023) explicitly describes its fully-automatic mode as low accuracy,
  with mean count error 59.7% on TSA bacteria overall and error exceeding
  140% at very low colony counts — the opposite of a clean ~96% figure.
- **aCOLyte** has no independent peer-reviewed accuracy validation that
  this search turned up — only manufacturer marketing material, which
  isn't a citable benchmark and is excluded from the comparison table
  entirely rather than presented as if it were literature-backed.

The pattern across all three: published colony-counter accuracy is not a
stable, single-number statistic. It swings by tens of percentage points
with organism, agar medium, and colony density, sometimes within the same
paper. Collapsing that into one ranked "accuracy %" per system would
manufacture a precision the underlying research doesn't have — the same
overclaiming this validation is explicitly trying to avoid for our own
results. `benchmark_comparison.py` instead reports each source's own
metric, on its own terms, with the exact conditions it was measured under.

## What is and isn't comparable

Where a literature source reports **mean count error %** (the independent
OpenCFU re-benchmark, ColonyDoc-It), that is roughly the same statistic as
this validation's own `mean_count_error_pct` — both express relative
deviation from a true/reference count — so those rows are the closest
thing to an apples-to-apples read: our pipeline's pooled 15.75% (5.31% on
the 3/5 conditions that pass this validation's F1 > 0.90 target) sits well
inside the range those two papers report for real automated counters
(18.3-59.7% for ColonyDoc-It depending on medium, 50.31% for OpenCFU on a
dense dataset). Where a source reports something else in kind (OpenCFU's
own median absolute colony-count error, or R² correlation), the numbers
are shown but not force-converted into a percentage they didn't publish.

Two further gaps that any reader comparing these numbers should weigh:
- **Domain**: our plates are synthetic renders; every literature figure
  above is from real bacterial colonies on real agar, with real-world
  failure modes (bubbles, scratches, uneven staining, edge artifacts) a
  synthetic generator doesn't necessarily reproduce.
- **Difficulty selection**: 2 of our 5 held-out conditions (dense_pack,
  high_touching) were deliberately chosen as adversarial edge cases beyond
  what a "standard plate" in the commercial literature implies; the other
  3 are closer to that baseline. Pooling all 5 is the honest number, but a
  reader should know it includes conditions chosen to be hard, not just
  conditions found to be hard.

## Literature search process

Sources were located via direct web search and verified by fetching the
primary paper itself (or, where a paywall/CAPTCHA blocked that, a
cached/mirrored full text), not by trusting a secondhand summary or
citation. Search terms and what was and wasn't found:

- OpenCFU: original validation paper located and confirmed (PLOS ONE 2013);
  a later independent re-benchmark located (PLOS ONE 2025, MCount paper)
  giving a second, harder-condition data point for the same tool.
- ColonyDoc-It: one peer-reviewed evaluation located (Microbiology
  Spectrum 2023) with full plate-count and per-medium breakdown.
- aCOLyte: multiple searches (direct product name, "validation study",
  "peer-reviewed") surfaced only manufacturer/vendor material (Synbiosis,
  Synoptics, and retailer product pages) — no independent third-party
  peer-reviewed accuracy study. Excluded rather than represented by a
  vendor claim.

Full citations are in `benchmark_comparison.py`'s `LITERATURE_BENCHMARKS`
list, next to the specific number each one supports.
