"""
benchmark_comparison.py — render our pipeline's blind-eval results against
literature-published colony-counter accuracy figures.

Every LITERATURE_BENCHMARKS entry below is a real number from a specific
peer-reviewed source, checked directly against that paper (see `citation`
and `note` per entry) rather than taken from secondhand summaries. Two
entries do not resolve to a single figure at all: this is intentional.
Published accuracy for automated colony counters is not a stable,
apples-to-apples statistic — it swings by tens of percentage points with
organism, medium, and colony density even within one paper — so this module
reports each source's own metric on its own terms instead of collapsing
everything into one ranked column, which would manufacture a precision the
underlying literature doesn't have.

aCOLyte (Synbiosis) is deliberately absent: no independent peer-reviewed
accuracy validation for it turned up in a literature search, only
manufacturer marketing material, which isn't a citable benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

RESULTS_PATH = Path("testing/blind_eval/results/blind_eval_results.json")
OUTPUT_PATH = Path("testing/blind_eval/results/benchmark_comparison.md")

LITERATURE_BENCHMARKS: List[Dict] = [
    {
        "system": "OpenCFU",
        "metric": "Median absolute colony-count error vs. manual counts",
        "value": "3 colonies (no statistically significant bias, p > 0.05)",
        "conditions": "19 plates, 10-1000 S. aureus colonies/plate, HD images. "
                       "Degrades to 8-colony median error on low-cost webcam images.",
        "citation": "Geissmann Q (2013). \"OpenCFU, a New Free and Open-Source "
                     "Software to Count Cell Colonies and Other Circular Objects.\" "
                     "PLOS ONE 8(2): e54072. doi:10.1371/journal.pone.0054072",
    },
    {
        "system": "OpenCFU (independent re-benchmark)",
        "metric": "Mean count error %",
        "value": "50.31%",
        "conditions": "960-image labeled E. coli dataset (15,847 segments) — denser "
                       "and harder than OpenCFU's own original validation set above; "
                       "measured independently by a later paper, not OpenCFU's authors.",
        "citation": "Chen, Huang, Kim, Cui, Buie (2025). \"MCount: An automated "
                     "colony counting tool for high-throughput microbiology.\" "
                     "PLOS ONE 20(3): e0311242.",
    },
    {
        "system": "ColonyDoc-It (fully automatic mode)",
        "metric": "Mean count error % vs. manual counts",
        "value": "59.7% (all bacteria, TSA, R² = 0.77); 18.3% (blood agar, R² = 0.94)",
        "conditions": "640 plates, 6 organisms (S. aureus, E. coli, P. aeruginosa, "
                       "K. pneumoniae, E. faecium, C. albicans), counts adjusted to "
                       "~1000/100/10/1 colonies/plate. Error rises sharply at the "
                       "extremes (141.7-207.4% at ~1 colony/plate).",
        "citation": "Heuser E, Becker K, Idelevich EA (2023). \"Evaluation of an "
                     "Automated System for the Counting of Microbial Colonies.\" "
                     "Microbiology Spectrum 11(4): e00673-23. "
                     "doi:10.1128/spectrum.00673-23",
    },
]

EXCLUDED_NOTE = (
    "aCOLyte (Synbiosis) is excluded: no independent peer-reviewed accuracy "
    "validation was found in this search, only manufacturer marketing material "
    "(e.g. Synoptics' own \"Automated colony counting proves accurate\" write-up), "
    "which is not a citable benchmark."
)


def _our_row(results: Dict) -> Dict:
    o = results["overall"]
    passing = {name: s for name, s in results["by_set"].items()
               if s["f1"] is not None and s["f1"] >= results["target_f1"]}
    passing_error = (sum(s["mean_count_error_pct"] * s["n_plates"] for s in passing.values())
                      / sum(s["n_plates"] for s in passing.values())) if passing else None

    conditions = (f"250 synthetic blind held-out plates, 5 stress conditions "
                  f"(dense colony packing, high touching-colony rate, extreme size "
                  f"variance, poor illumination, irregular morphology) not used "
                  f"during pipeline development.")
    if passing_error is not None:
        conditions += (f" On the 3/5 conditions that pass this validation's own "
                        f"F1 > {results['target_f1']:.2f} target (irregular_morphology, "
                        f"poor_illumination, size_variance): {passing_error:.2f}% mean "
                        f"count error, pooled.")

    return {
        "system": "Our pipeline",
        "metric": "Mean count error % (pooled) / F1 (per-colony precision-recall)",
        "value": f"{o['mean_count_error_pct']:.2f}% mean count error / F1 {o['f1']:.3f} "
                  f"(precision {o['precision']:.3f}, recall {o['recall']:.3f})",
        "conditions": conditions,
        "citation": "This validation — testing/blind_eval/run_blind_eval.py",
    }


def render() -> str:
    results = json.loads(RESULTS_PATH.read_text())
    rows = [_our_row(results)] + LITERATURE_BENCHMARKS

    lines = [
        "# Commercial / Open-Source Counter Benchmark Comparison", "",
        "_Auto-generated by testing/blind_eval/benchmark_comparison.py — "
        "do not hand-edit; see that file's docstring and "
        "testing/blind_eval/results/methodology.md for sourcing notes._", "",
        "Each row reports the metric that source actually published, not a "
        "converted or estimated single \"accuracy\" figure — see "
        "methodology.md for why a single unified column isn't used.", "",
        "| System | Metric | Value | Conditions | Source |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['system']} | {r['metric']} | {r['value']} | "
                      f"{r['conditions']} | {r['citation']} |")
    lines += ["", f"_{EXCLUDED_NOTE}_", ""]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render())
    print(render())


if __name__ == "__main__":
    main()
