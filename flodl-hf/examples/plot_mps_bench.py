#!/usr/bin/env python3
"""Turn a bert_mps_bench CSV into a markdown table and an optional SVG.

The example prints the CSV. This script does not invent timings: every
number in the table and the chart is read from that file. Axes scale to
the rows, so a new run does not need the old chart's hard-coded range.

    ./fdl flodl-hf example bert-mps-bench | tee mps-tinybert.csv
    python3 flodl-hf/examples/plot_mps_bench.py mps-tinybert.csv \\
        --svg docs/charts/mps-tinybert.svg \\
        --caption "M5 Pro · macOS 26.5.1"
"""

import argparse
import csv
import sys
from pathlib import Path


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    needed = {
        "device",
        "batch_size",
        "sequence_length",
        "median_ms_per_step",
        "examples_per_second",
    }
    if not rows or needed - set(rows[0]):
        missing = needed - set(rows[0] if rows else ())
        raise SystemExit(f"{path}: expected columns {sorted(needed)}, missing {sorted(missing)}")
    return rows


def markdown(rows: list[dict[str, str]]) -> str:
    data = {(int(r["batch_size"]), r["device"]): r for r in rows}
    batches = sorted({int(r["batch_size"]) for r in rows})
    lines = [
        "| Batch | CPU ms / step | MPS ms / step | MPS speedup | CPU examples / s | MPS examples / s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for batch in batches:
        cpu = data.get((batch, "cpu"))
        mps = data.get((batch, "mps"))
        if cpu is None or mps is None:
            raise SystemExit(f"batch {batch} is missing a cpu or mps row")
        cpu_ms = float(cpu["median_ms_per_step"])
        mps_ms = float(mps["median_ms_per_step"])
        speedup = cpu_ms / mps_ms if mps_ms else float("inf")
        lines.append(
            f"| {batch} | {cpu_ms:.3f} | {mps_ms:.3f} | {speedup:.2f}× "
            f"| {float(cpu['examples_per_second']):.1f} "
            f"| {float(mps['examples_per_second']):.1f} |"
        )
    return "\n".join(lines)


def svg(rows: list[dict[str, str]], caption: str) -> str:
    data = {(int(r["batch_size"]), r["device"]): r for r in rows}
    batches = sorted({int(r["batch_size"]) for r in rows})
    seq = rows[0]["sequence_length"]
    width, height = 960, 480
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        "<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#18212f}"
        ".title{font-size:23px;font-weight:650}.subtitle{font-size:13px;fill:#586579}"
        ".panel{font-size:16px;font-weight:600}.tick,.label{font-size:12px}"
        ".grid{stroke:#d9e0e8;stroke-width:1}.cpu{fill:#8a99ac}.mps{fill:#3975d7}</style>",
        '<text x="480" y="32" text-anchor="middle" class="title">TinyBERT training: CPU vs Apple MPS</text>',
        f'<text x="480" y="54" text-anchor="middle" class="subtitle">'
        f"Median full fine-tuning step time and throughput · sequence length {seq}</text>",
    ]
    panels = [
        ("Training step time", "median_ms_per_step", "ms / step", 80),
        ("Training throughput", "examples_per_second", "examples / second", 540),
    ]
    plot_top, plot_bottom = 110, 360
    plot_h = plot_bottom - plot_top
    for title, key, ylabel, left in panels:
        values = [float(r[key]) for r in rows]
        max_value = max(values) * 1.15 if values else 1.0
        plot_w = 360
        parts.append(
            f'<text x="{left + plot_w / 2}" y="86" text-anchor="middle" class="panel">{title}</text>'
        )
        parts.append(f'<text x="{left}" y="102" class="subtitle">{ylabel}</text>')
        for tick in range(5):
            value = max_value * tick / 4
            y = plot_bottom - plot_h * tick / 4
            parts.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>'
            )
            parts.append(
                f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.0f}</text>'
            )
        group_w = plot_w / max(len(batches), 1)
        bar_w = 27
        for i, batch in enumerate(batches):
            center = left + group_w * (i + 0.5)
            for j, device in enumerate(("cpu", "mps")):
                row = data[(batch, device)]
                value = float(row[key])
                bar_h = plot_h * value / max_value
                x = center + (j - 0.5) * (bar_w + 3) - bar_w / 2
                y = plot_bottom - bar_h
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="3" class="{device}"/>'
                )
                label = f"{value:.1f}" if key.endswith("per_second") else f"{value:.0f}"
                parts.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" class="label">{label}</text>'
                )
            parts.append(
                f'<text x="{center:.1f}" y="{plot_bottom + 23}" text-anchor="middle" class="tick">{batch}</text>'
            )
        parts.append(
            f'<text x="{left + plot_w / 2}" y="{plot_bottom + 46}" text-anchor="middle" class="subtitle">batch size</text>'
        )
    parts.extend(
        [
            '<rect x="386" y="425" width="13" height="13" rx="2" class="cpu"/><text x="405" y="436" class="tick">CPU</text>',
            '<rect x="472" y="425" width="13" height="13" rx="2" class="mps"/><text x="491" y="436" class="tick">Apple MPS</text>',
            f'<text x="480" y="466" text-anchor="middle" class="subtitle">{caption}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV printed by bert_mps_bench")
    parser.add_argument("--svg", type=Path, help="write the chart here")
    parser.add_argument(
        "--caption",
        default="measured on the machine named in the run notes",
        help="footer line on the chart (machine and OS)",
    )
    args = parser.parse_args()
    rows = load(args.csv)
    print(markdown(rows))
    if args.svg:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        args.svg.write_text(svg(rows, args.caption))
        print(f"\nwrote {args.svg}", file=sys.stderr)


if __name__ == "__main__":
    main()
