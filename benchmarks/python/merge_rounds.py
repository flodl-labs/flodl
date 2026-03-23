#!/usr/bin/env python3
"""Merge per-round benchmark JSON files into a single result set.

Each round file is a JSON array of BenchResult objects. For each benchmark
name, this script collects the best-run median from every round into
`run_medians_ms`, then recomputes `median_epoch_ms` as the median of those
best-run medians. This gives N i.i.d. samples (one per round) for robust
statistics.

Usage: merge_rounds.py round1.json round2.json ... > merged.json
"""

import json
import statistics
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: merge_rounds.py round1.json round2.json ...",
              file=sys.stderr)
        sys.exit(1)

    # Load all rounds
    rounds = []
    for path in sys.argv[1:]:
        with open(path) as f:
            rounds.append(json.load(f))

    # Collect per-benchmark data across rounds
    # Each round's median_epoch_ms is the best-run median for that round.
    benchmarks = {}
    for round_results in rounds:
        for r in round_results:
            name = r["name"]
            if name not in benchmarks:
                benchmarks[name] = {
                    "round_medians": [],
                    "all_run_medians": [],
                    "template": r,
                    "vram_mb_max": r.get("vram_mb"),
                    "vram_reserved_mb_max": r.get("vram_reserved_mb"),
                }
            benchmarks[name]["round_medians"].append(r["median_epoch_ms"])
            benchmarks[name]["all_run_medians"].extend(
                r.get("run_medians_ms", [r["median_epoch_ms"]])
            )
            # Track peak VRAM across rounds
            if r.get("vram_mb") is not None:
                prev = benchmarks[name]["vram_mb_max"]
                if prev is None or r["vram_mb"] > prev:
                    benchmarks[name]["vram_mb_max"] = r["vram_mb"]
            if r.get("vram_reserved_mb") is not None:
                prev = benchmarks[name]["vram_reserved_mb_max"]
                if prev is None or r["vram_reserved_mb"] > prev:
                    benchmarks[name]["vram_reserved_mb_max"] = r["vram_reserved_mb"]

    # Build merged results
    results = []
    for name, data in benchmarks.items():
        medians = data["round_medians"]
        t = data["template"]

        merged = {
            "name": name,
            "device": t["device"],
            "rounds": len(medians),
            "runs": t.get("runs", 1),
            "warmup_epochs": t.get("warmup_epochs", 3),
            "measured_epochs": t.get("measured_epochs", 20),
            "batches_per_epoch": t.get("batches_per_epoch", 100),
            "batch_size": t.get("batch_size", 128),
            "param_count": t["param_count"],
            # Best round's epoch times (lowest median)
            "epoch_times_ms": t["epoch_times_ms"],
            # Median of best-run medians across rounds
            "median_epoch_ms": statistics.median(medians),
            "mean_epoch_ms": statistics.mean(medians),
            "min_epoch_ms": min(medians),
            "max_epoch_ms": max(medians),
            # All best-run medians (one per round) — used for σ
            "run_medians_ms": medians,
            "final_loss": t["final_loss"],
            "vram_mb": data["vram_mb_max"],
            "vram_reserved_mb": data["vram_reserved_mb_max"],
            "rss_mb": t.get("rss_mb", 0),
        }
        results.append(merged)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
