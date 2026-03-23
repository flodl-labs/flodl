#!/usr/bin/env python3
"""Compare Rust (flodl) and Python (PyTorch) benchmark results."""

import json
import sys


def main():
    if len(sys.argv) != 3:
        print("Usage: compare.py <rust_results.json> <python_results.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        rust_results = {r["name"]: r for r in json.load(f)}

    with open(sys.argv[2]) as f:
        python_results = {r["name"]: r for r in json.load(f)}

    all_names = list(dict.fromkeys(list(rust_results.keys()) + list(python_results.keys())))

    # Header
    print()
    print(f"  {'benchmark':<20} {'PyTorch':>10} {'flodl':>10} {'delta':>8} "
          f"{'Py alloc':>10} {'Rs alloc':>10} {'Py rsrvd':>10} {'Rs rsrvd':>10}")
    print(f"  {'-' * 92}")

    for name in all_names:
        py = python_results.get(name)
        rs = rust_results.get(name)

        py_ms = f"{py['median_epoch_ms']:.1f}ms" if py else "—"
        rs_ms = f"{rs['median_epoch_ms']:.1f}ms" if rs else "—"

        if py and rs:
            delta_pct = (rs["median_epoch_ms"] - py["median_epoch_ms"]) / py["median_epoch_ms"] * 100
            delta = f"{delta_pct:+.0f}%"
        else:
            delta = "—"

        py_alloc = f"{py['vram_mb']:.0f} MB" if py and py.get("vram_mb") else "—"
        rs_alloc = f"{rs['vram_mb']:.0f} MB" if rs and rs.get("vram_mb") else "—"
        py_rsrvd = f"{py['vram_reserved_mb']:.0f} MB" if py and py.get("vram_reserved_mb") else "—"
        rs_rsrvd = f"{rs['vram_reserved_mb']:.0f} MB" if rs and rs.get("vram_reserved_mb") else "—"

        print(f"  {name:<20} {py_ms:>10} {rs_ms:>10} {delta:>8} "
              f"{py_alloc:>10} {rs_alloc:>10} {py_rsrvd:>10} {rs_rsrvd:>10}")

    print()


if __name__ == "__main__":
    main()
