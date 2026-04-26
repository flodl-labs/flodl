#!/usr/bin/env python3
"""Run `fdl flodl-hf export` -> `verify-export` across the full head matrix.

Iterates the 30 (family, head) cells in
`flodl-hf/tests/fixtures/head_matrix.json`, invokes the two-step
pipeline per cell, and prints a PASS/FAIL grid at the end.

Run via `fdl flodl-hf verify-matrix [-- --families ...]`. Lives outside
any docker service: each cell shells out to `fdl flodl-hf convert /
export / verify-export`, which bring up the right container themselves
(dev for export, hf-parity for convert + verify-export). That keeps
this script lightweight (host-side stdlib only) and avoids
docker-in-docker.

Filters
-------
    --families bert,albert      restrict to one or more families
    --heads    base,seqcls      restrict to one or more heads
    --keep                      keep staging dirs after each cell (default: cleanup)
    --staging-dir <path>        override staging root (default: flodl-hf/tests/.exports)

Defaults to all 30 cells. Fail-soft: continues past per-cell failures
and exits non-zero if any cell failed.

Exit code mirrors `parity_all.py`: 0 on full pass, 1 if any cell
failed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Resolve the project root the way fdl does — env var set by the fdl
# CLI on subprocess invocation, with a sane fallback for direct
# `python flodl-hf/scripts/verify_matrix.py` runs from the repo root.
import os

PROJECT_ROOT = Path(os.environ.get("FDL_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
FIXTURE = PROJECT_ROOT / "flodl-hf" / "tests" / "fixtures" / "head_matrix.json"
DEFAULT_STAGING = PROJECT_ROOT / "flodl-hf" / "tests" / ".exports"


def parse_csv(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI value to a list, dropping empties.
    Returns `None` when the flag is absent so callers can detect
    "use the default" vs "explicit empty".
    """
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts


def select_cells(
    cells: list[dict],
    families: list[str] | None,
    heads: list[str] | None,
) -> list[dict]:
    """Filter cells by family and head whitelists."""
    out = []
    for c in cells:
        if families is not None and c["family"] not in families:
            continue
        if heads is not None and c["head"] not in heads:
            continue
        out.append(c)
    return out


def run_step(label: str, cmd: list[str]) -> bool:
    """Print a banner + run the command. Inherits stdout/stderr so the
    underlying fdl/python output streams live, matching parity_all.py.
    Returns True on success.
    """
    print(f"  $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
    if rc != 0:
        print(f"!! {label}: step `{' '.join(cmd)}` exited {rc}", flush=True)
        return False
    return True


def run_cell(cell: dict, staging_root: Path, keep: bool) -> bool:
    """Run convert (if needed) -> export -> verify-export for one cell."""
    family = cell["family"]
    head = cell["head"]
    repo = cell["repo"]
    needs_convert = bool(cell.get("needs_convert", False))

    # Staging dir is repo-relative so fdl's FDL_PROJECT_ROOT path
    # resolution lines up between host and container.
    rel_dir = Path("flodl-hf") / "tests" / ".exports" / f"{family}_{head}"
    staging = PROJECT_ROOT / rel_dir
    if staging.exists():
        shutil.rmtree(staging)

    if needs_convert:
        if not run_step(
            f"{family}/{head}",
            ["fdl", "flodl-hf", "convert", repo],
        ):
            return False

    if not run_step(
        f"{family}/{head}",
        [
            "fdl",
            "flodl-hf",
            "export",
            "--hub",
            repo,
            "--out",
            str(rel_dir),
            "--force",
        ],
    ):
        return False

    if not run_step(
        f"{family}/{head}",
        ["fdl", "flodl-hf", "verify-export", str(rel_dir)],
    ):
        return False

    if not keep:
        shutil.rmtree(staging, ignore_errors=True)
    return True


def render_grid(
    families: list[str],
    heads: list[str],
    results: dict[tuple[str, str], tuple[bool, float] | None],
) -> str:
    """Format the families x heads grid showing PASS / FAIL / `-` (skipped)
    per cell. `results` keys absent from the dict mean the cell wasn't run.
    """
    fam_w = max(len(f) for f in families)
    head_w = max(8, max(len(h) for h in heads))
    lines = []
    header = " " * (fam_w + 2) + "  ".join(h.ljust(head_w) for h in heads)
    lines.append(header)
    for fam in families:
        row_cells = []
        for head in heads:
            r = results.get((fam, head))
            if r is None:
                cell_str = "-"
            else:
                ok, elapsed = r
                cell_str = f"{'PASS' if ok else 'FAIL'} {elapsed:5.1f}s"
            row_cells.append(cell_str.ljust(head_w))
        lines.append(fam.ljust(fam_w) + "  " + "  ".join(row_cells))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run `fdl flodl-hf export` then `verify-export` across the "
            "head matrix (6 families x 5 heads = 30 cells). Fail-soft, "
            "prints a status grid at end."
        ),
    )
    parser.add_argument(
        "--families",
        help="CSV of families to include (default: all). E.g. `bert,albert`.",
    )
    parser.add_argument(
        "--heads",
        help="CSV of heads to include (default: all). E.g. `base,seqcls`.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep staging dirs after each cell (default: remove on success).",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING,
        help=(
            "Staging root for export dirs (default: "
            "flodl-hf/tests/.exports). Each cell stages to "
            "<root>/<family>_<head>/."
        ),
    )
    args = parser.parse_args()

    fixture = json.loads(FIXTURE.read_text())
    all_cells = fixture["cells"]
    all_families = fixture["families"]
    all_heads = fixture["heads"]

    families_filter = parse_csv(args.families)
    heads_filter = parse_csv(args.heads)

    if families_filter is not None:
        unknown = [f for f in families_filter if f not in all_families]
        if unknown:
            parser.error(
                f"unknown family in --families: {unknown}. "
                f"Known: {all_families}"
            )
    if heads_filter is not None:
        unknown = [h for h in heads_filter if h not in all_heads]
        if unknown:
            parser.error(
                f"unknown head in --heads: {unknown}. "
                f"Known: {all_heads}"
            )

    cells = select_cells(all_cells, families_filter, heads_filter)
    if not cells:
        parser.error("no cells matched the --families / --heads filter")

    print(
        f"verify-matrix: running {len(cells)} of {len(all_cells)} cells\n"
        f"  staging: {args.staging_dir}\n"
        f"  keep: {args.keep}",
        flush=True,
    )

    results: dict[tuple[str, str], tuple[bool, float]] = {}
    t_total = time.monotonic()
    for cell in cells:
        family, head, repo = cell["family"], cell["head"], cell["repo"]
        print(f"\n=== {family}/{head}  ({repo}) ===", flush=True)
        t0 = time.monotonic()
        ok = run_cell(cell, args.staging_dir, args.keep)
        elapsed = time.monotonic() - t0
        results[(family, head)] = (ok, elapsed)

    # Render grid restricted to the subset actually run (preserving
    # canonical fixture order for both axes).
    families_in = [f for f in all_families if any(k[0] == f for k in results)]
    heads_in = [h for h in all_heads if any(k[1] == h for k in results)]

    print("\n=== matrix ===")
    print(render_grid(families_in, heads_in, results))

    failures = [(f, h) for (f, h), (ok, _) in results.items() if not ok]
    print(f"\nelapsed: {time.monotonic() - t_total:.1f}s total")
    if failures:
        print(f"\n{len(failures)} of {len(results)} cells FAILED:")
        for f, h in failures:
            print(f"  - {f}/{h}")
        return 1
    print(f"\nall {len(results)} cells passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
