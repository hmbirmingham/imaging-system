# Continuous Testing Harness — Operator's Guide

Practical answers to "how do I actually run this thing / see what it found /
get it into git". For *why* the harness is built the way it is, see
`testing/README.md`.

## Does anything auto-push?

**No.** Every cycle writes to local files only
(`testing/logs/`, `testing/artifacts/`, `testing/reports/`). Nothing commits
or pushes to git automatically — that's deliberate, matching the project's
own rule that raw data is never committed (it's regenerable from the seed
logged in each cycle's Phase A record). You decide when to commit a
snapshot (see §5) and when to push.

The one automatic piece is the **GitHub Action**: once pushed, it re-runs
one full test-matrix pass on every future push and reports results — but it
does not commit anything back into the repo.

## 1. Running one pass (CI mode)

Runs every scenario in `test_matrix.yaml` once, then exits:

```bash
python -m testing.continuous.scheduler --mode ci
```

Exit code is `0` if every cycle passed, `1` otherwise — this is exactly
what the GitHub Action calls.

## 2. Running continuously

**Foreground** (see live output, stops on Ctrl-C or when you close the
terminal):

```bash
python -m testing.continuous.scheduler --mode continuous
```

**Background**, survives closing the terminal:

```bash
nohup python -m testing.continuous.scheduler --mode continuous \
    > testing/scheduler.log 2>&1 &
echo $! > testing/scheduler.pid
```

Check on it / stop it:

```bash
ps -p $(cat testing/scheduler.pid)     # still running?
tail -f testing/scheduler.log          # watch live cycle output
kill $(cat testing/scheduler.pid)      # stop it
```

Useful flags:

```bash
--interval SECONDS      # time between cycles, default 30
--rss-limit-mb MB       # back off cadence above this process RSS, default 512
```

### Running it as a real background service on the Pi (systemd)

For something that survives a reboot, create
`/etc/systemd/system/plate-testing.service`:

```ini
[Unit]
Description=Plate imaging continuous testing harness
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/pi/imaging-system
ExecStart=/usr/bin/python3 -m testing.continuous.scheduler --mode continuous
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now plate-testing
sudo systemctl status plate-testing     # check it
journalctl -u plate-testing -f          # watch live output
sudo systemctl stop plate-testing       # stop it
```

## 3. Where the data lives

| Path | What | Committed to git? |
|---|---|---|
| `testing/logs/{pre,during,post}/cycle_XXXX.json` | Raw Phase A/B/C data per cycle | No — pruned automatically |
| `testing/artifacts/images/cycle_XXXX/` | Intermediate images (masks, watershed labels, annotated output) per cycle | No — pruned to the most recent `artifact_retention_cycles` (default 5, see `test_matrix.yaml`) |
| `testing/reports/cycle_XXXX_summary.{json,md}` | Per-cycle human/machine-readable report | No — pruned automatically |
| `testing/reports/aggregate_state.json` | Compact running statistics (Welford accumulators) — the source `aggregate_report.md` is rendered from | No |
| `testing/reports/aggregate_report.md` | Rolling, human-readable summary across every cycle ever run | **Yes** — this is the one file meant to be committed periodically |

Everything non-committed is regenerable: re-run the generator with the same
seed (logged in Phase A) to reproduce a cycle's exact input.

## 4. Looking through the data

**Read the current rolling summary** (six sections: Track 1 accuracy by
condition + distance, Track 2 model comparison, Track 3 integrity/load
test, Track 4 dry-run pass rate/latency, failure taxonomy, stage timing):

```bash
cat testing/reports/aggregate_report.md
```

**Look at one specific past cycle:**

```bash
cat testing/reports/cycle_0012_summary.md
```

**Inspect raw aggregate numbers** (if you have `jq`):

```bash
jq '.track1.by_distance_factor' testing/reports/aggregate_state.json
jq '.track2_history' testing/reports/aggregate_state.json
```

**See what a cycle actually generated** (only the most recently-run cycles
are kept on disk):

```bash
ls testing/artifacts/images/
open testing/artifacts/images/0016/annotated.jpg   # macOS
```

**Regenerate the thesis plot/table** at any point after cycles have run:

```bash
python -m testing.thesis_export.figures_and_tables
open testing/thesis_export/generated/distance_invariance.png
cat testing/thesis_export/generated/model_comparison.md
```

## 5. "Pushing the data" — committing a snapshot to git

The raw per-cycle data is never committed. What *is* meant to be committed
periodically is the `aggregate_report.md` snapshot, once you've accumulated
cycles worth recording (e.g., after letting continuous mode run overnight):

```bash
git add testing/reports/aggregate_report.md
git commit -m "docs(reports): update aggregate_report.md snapshot after N cycles"
git push origin main
```

## 6. GitHub Action (automatic on every push)

Once pushed, `.github/workflows/continuous-testing.yml` runs one full
`--mode ci` pass plus the full `pytest` suite on every push, publishes
`aggregate_report.md` to the run's job summary, and uploads it as a
downloadable artifact. Results:
`https://github.com/hmbirmingham/imaging-system/actions`

Pushing anything that touches `.github/workflows/` requires your `gh`
token to have the `workflow` OAuth scope:

```bash
gh auth refresh -h github.com -s workflow
```

## 7. Quick reference

```bash
# One pass, exits
python -m testing.continuous.scheduler --mode ci

# Run forever in the background
nohup python -m testing.continuous.scheduler --mode continuous \
    > testing/scheduler.log 2>&1 & echo $! > testing/scheduler.pid

# Check / stop the background run
ps -p $(cat testing/scheduler.pid); tail -f testing/scheduler.log
kill $(cat testing/scheduler.pid)

# Read results
cat testing/reports/aggregate_report.md

# Regenerate paper-ready figures/tables
python -m testing.thesis_export.figures_and_tables

# Commit a snapshot
git add testing/reports/aggregate_report.md && git commit -m "docs(reports): update snapshot"
```
