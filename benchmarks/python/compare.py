#!/usr/bin/env python3
"""Compare Rust (flodl) and Python (PyTorch) benchmark results.

Usage: compare.py <rust_results.json> <python_results.json>

Each benchmark internally runs multiple passes (1 warmup + N measured).
The JSON contains the best run's median and the per-run medians for
stddev reporting.
"""

import json
import statistics
import sys


def main():
    if len(sys.argv) != 3:
        print("Usage: compare.py <rust_results.json> <python_results.json>",
              file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        rust_results = {r["name"]: r for r in json.load(f)}

    with open(sys.argv[2]) as f:
        python_results = {r["name"]: r for r in json.load(f)}

    all_names = list(dict.fromkeys(
        list(rust_results.keys()) + list(python_results.keys())
    ))

    # Determine number of runs and rounds from results
    n_runs = 1
    n_rounds = 1
    for r in list(rust_results.values()) + list(python_results.values()):
        if r.get("runs", 1) > n_runs:
            n_runs = r["runs"]
        if r.get("rounds", 1) > n_rounds:
            n_rounds = r["rounds"]

    # Header
    print()
    label = f"best-of-{n_runs}" if n_runs > 1 else "single run"
    if n_rounds > 1:
        label += f", {n_rounds} rounds"
    print(f"  [{label}, each: 3 warmup + 20 measured epochs]")
    print()
    hdr = (f"  {'benchmark':<20} {'PyTorch':>10} {'flodl':>10} {'delta':>8}"
           f"   {'Py alloc':>10} {'Rs alloc':>10} {'Py rsrvd':>10} {'Rs rsrvd':>10}")
    if n_runs > 1:
        hdr += f"   {'Py σ':>8} {'Rs σ':>8}"
    print(hdr)
    print(f"  {'-' * (92 + (20 if n_runs > 1 else 0))}")

    for name in all_names:
        py = python_results.get(name)
        rs = rust_results.get(name)

        py_ms = f"{py['median_epoch_ms']:.1f}ms" if py else "—"
        rs_ms = f"{rs['median_epoch_ms']:.1f}ms" if rs else "—"

        if py and rs:
            delta_pct = ((rs["median_epoch_ms"] - py["median_epoch_ms"])
                         / py["median_epoch_ms"] * 100)
            delta = f"{delta_pct:+.0f}%"
        else:
            delta = "—"

        py_alloc = f"{py['vram_mb']:.0f} MB" if py and py.get("vram_mb") else "—"
        rs_alloc = f"{rs['vram_mb']:.0f} MB" if rs and rs.get("vram_mb") else "—"
        py_rsrvd = f"{py['vram_reserved_mb']:.0f} MB" if py and py.get("vram_reserved_mb") else "—"
        rs_rsrvd = f"{rs['vram_reserved_mb']:.0f} MB" if rs and rs.get("vram_reserved_mb") else "—"

        line = (f"  {name:<20} {py_ms:>10} {rs_ms:>10} {delta:>8}"
                f"   {py_alloc:>10} {rs_alloc:>10} {py_rsrvd:>10} {rs_rsrvd:>10}")

        if n_runs > 1:
            py_std = "—"
            rs_std = "—"
            py_medians = py.get("run_medians_ms", []) if py else []
            rs_medians = rs.get("run_medians_ms", []) if rs else []
            if len(py_medians) > 1:
                py_std = f"±{statistics.stdev(py_medians):.1f}"
            if len(rs_medians) > 1:
                rs_std = f"±{statistics.stdev(rs_medians):.1f}"
            line += f"   {py_std:>8} {rs_std:>8}"

        print(line)

    print()


if __name__ == "__main__":
    main()
