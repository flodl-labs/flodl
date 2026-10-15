#!/usr/bin/env python3
"""Render the checked-in MPS benchmark CSV as a standalone SVG chart."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/mps-tinybert.csv"
OUTPUT = ROOT / "docs/charts/mps-tinybert.svg"


def main():
    with SOURCE.open(newline="") as f:
        rows = list(csv.DictReader(f))
    data = {(int(r["batch_size"]), r["device"]): r for r in rows}
    batches = sorted({int(r["batch_size"]) for r in rows})

    width, height = 960, 480
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#18212f}.title{font-size:23px;font-weight:650}.subtitle{font-size:13px;fill:#586579}.panel{font-size:16px;font-weight:600}.tick,.label{font-size:12px}.grid{stroke:#d9e0e8;stroke-width:1}.cpu{fill:#8a99ac}.mps{fill:#3975d7}</style>',
        '<text x="480" y="32" text-anchor="middle" class="title">TinyBERT training: CPU vs Apple MPS</text>',
        '<text x="480" y="54" text-anchor="middle" class="subtitle">Median full fine-tuning step time and throughput · sequence length 128</text>',
    ]

    panels = [
        ("Training step time", "median_ms_per_step", "ms / step", 450, 80),
        ("Training throughput", "examples_per_second", "examples / second", 7000, 540),
    ]
    plot_top, plot_bottom = 110, 360
    plot_h = plot_bottom - plot_top
    for title, key, ylabel, max_value, left in panels:
        plot_w = 360
        parts.append(f'<text x="{left + plot_w / 2}" y="86" text-anchor="middle" class="panel">{title}</text>')
        parts.append(f'<text x="{left}" y="102" class="subtitle">{ylabel}</text>')
        for tick in range(5):
            value = max_value * tick / 4
            y = plot_bottom - plot_h * tick / 4
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.0f}</text>')
        group_w = plot_w / len(batches)
        bar_w = 27
        for i, batch in enumerate(batches):
            center = left + group_w * (i + 0.5)
            for j, device in enumerate(("cpu", "mps")):
                row = data[(batch, device)]
                value = float(row[key])
                bar_h = plot_h * value / max_value
                x = center + (j - 0.5) * (bar_w + 3) - bar_w / 2
                y = plot_bottom - bar_h
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="3" class="{device}"/>')
                label = f"{value:.1f}" if key.endswith("per_second") else f"{value:.0f}"
                parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" class="label">{label}</text>')
            parts.append(f'<text x="{center:.1f}" y="{plot_bottom + 23}" text-anchor="middle" class="tick">{batch}</text>')
        parts.append(f'<text x="{left + plot_w / 2}" y="{plot_bottom + 46}" text-anchor="middle" class="subtitle">batch size</text>')

    parts.extend([
        '<rect x="386" y="425" width="13" height="13" rx="2" class="cpu"/><text x="405" y="436" class="tick">CPU</text>',
        '<rect x="472" y="425" width="13" height="13" rx="2" class="mps"/><text x="491" y="436" class="tick">Apple MPS</text>',
        '<text x="480" y="466" text-anchor="middle" class="subtitle">M5 Pro · macOS 26.5.1 · 4.39M-parameter jason-zhoou/tiny-bert-sst2 · synthetic repeated-token batches</text>',
        '</svg>',
    ])
    OUTPUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
