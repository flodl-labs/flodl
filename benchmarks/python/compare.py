#!/usr/bin/env python3
"""Compare Rust (flodl) and Python (PyTorch) benchmark results.

Usage: compare.py <rust_results.json> <python_results.json>

Each benchmark internally runs multiple passes (1 warmup + N measured).
The JSON contains the best run's median and the per-run medians for
variance reporting.

Three tables are printed, one per thing measured:

- training throughput: a full step (forward, loss, backward, optimizer);
- optimizer step: the optimizer's `step()` alone;
- memory: peak allocated and reserved VRAM.

Times are per step, so rows can be read against each other. The delta is
flodl against PyTorch (negative = flodl faster) with its uncertainty, and
the verdict says whether the two sides differ by more than their noise.

Variance is scaled MAD (Median Absolute Deviation × 1.4826), which is
σ-equivalent for normal distributions but robust to outliers caused by OS
scheduling, GC pauses, or thermal transients.
"""

import json
import math
import statistics
import sys

# A difference counts as real when it exceeds this many combined σ.
NOISE_SIGMAS = 2.0
# Floor on the combined σ, as a share of the PyTorch time. With few runs
# the MAD can be zero, and a zero σ would call any difference real.
NOISE_FLOOR = 0.005

OPTIM_PREFIX = "optim_"


def scaled_mad(values):
    """Median Absolute Deviation scaled to be σ-equivalent (× 1.4826)."""
    med = statistics.median(values)
    deviations = [abs(v - med) for v in values]
    return statistics.median(deviations) * 1.4826


def step_ms(r):
    """Median time of one step, in ms."""
    return r["median_epoch_ms"] / max(r.get("batches_per_epoch", 1), 1)


def step_sigma_ms(r):
    """σ of the per-step time across runs or rounds, or None with fewer than two."""
    medians = r.get("run_medians_ms", [])
    if len(medians) < 2:
        return None
    return scaled_mad(medians) / max(r.get("batches_per_epoch", 1), 1)


def fmt_time(ms):
    return f"{ms * 1000:.1f} µs" if ms < 1.0 else f"{ms:.2f} ms"


def compare(py, rs):
    """Return (delta text, verdict) for one row with both sides present."""
    p, r = step_ms(py), step_ms(rs)
    delta = (r - p) / p * 100
    sp, sr = step_sigma_ms(py), step_sigma_ms(rs)
    if sp is None or sr is None:
        return f"{delta:+.0f}%", "—"
    sigma = max(math.hypot(sp, sr), NOISE_FLOOR * p)
    text = f"{delta:+.0f}% ±{sigma / p * 100:.0f}"
    if abs(r - p) <= NOISE_SIGMAS * sigma:
        return text, "within noise"
    return text, "flodl faster" if r < p else "PyTorch faster"


def time_table(title, names, python_results, rust_results, label, strip=""):
    if not names:
        return
    print(f"  {title}")
    print()
    print(f"  {label:<18} {'tensors':>8} {'PyTorch':>11} {'flodl':>11} {'delta':>10}   verdict")
    print(f"  {'-' * 78}")
    for name in names:
        py, rs = python_results.get(name), rust_results.get(name)
        one = rs or py
        tensors = one.get("tensor_count")
        tensors = str(tensors) if tensors else "—"
        py_t = fmt_time(step_ms(py)) if py else "—"
        rs_t = fmt_time(step_ms(rs)) if rs else "—"
        delta, verdict = compare(py, rs) if py and rs else ("—", "—")
        shown = name[len(strip):] if strip and name.startswith(strip) else name
        print(f"  {shown:<18} {tensors:>8} {py_t:>11} {rs_t:>11} {delta:>10}   {verdict}")
    print()


def memory_table(names, python_results, rust_results):
    rows = []
    for name in names:
        py, rs = python_results.get(name), rust_results.get(name)
        if not ((py and py.get("vram_mb")) or (rs and rs.get("vram_mb"))):
            continue

        def mb(r, key):
            return f"{r[key]:.0f} MB" if r and r.get(key) else "—"

        rows.append((name, mb(py, "vram_mb"), mb(rs, "vram_mb"),
                     mb(py, "vram_reserved_mb"), mb(rs, "vram_reserved_mb")))
    if not rows:
        return
    print("  Memory (peak VRAM)")
    print()
    print(f"  {'benchmark':<18} {'Py alloc':>10} {'Rs alloc':>10} {'Py rsrvd':>10} {'Rs rsrvd':>10}")
    print(f"  {'-' * 62}")
    for row in rows:
        print(f"  {row[0]:<18} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}")
    print()


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

    print()
    label = f"best-of-{n_runs}" if n_runs > 1 else "single run"
    if n_rounds > 1:
        label += f", {n_rounds} rounds"
    print(f"  [{label}, each: 3 warmup + 20 measured epochs]")
    print(f"  [time per step; delta = flodl vs PyTorch, negative is flodl faster; "
          f"a verdict needs more than {NOISE_SIGMAS:g}σ]")
    print()

    training = [n for n in all_names if not n.startswith(OPTIM_PREFIX)]
    optim = [n for n in all_names if n.startswith(OPTIM_PREFIX)]

    time_table("Training throughput (forward, loss, backward, optimizer step)",
               training, python_results, rust_results, "benchmark")
    time_table("Optimizer step alone", optim, python_results, rust_results,
               "optimizer", strip=OPTIM_PREFIX)
    memory_table(training, python_results, rust_results)


if __name__ == "__main__":
    main()
