#!/usr/bin/env python3
"""Regenerate every parity fixture in flodl-hf/tests/fixtures/.

Runs the 18 parity scripts in sequence (fail-soft: continues on per-fixture
failure, prints a PASS/FAIL summary at the end, exits non-zero if any failed).

Run via `fdl flodl-hf parity all` — the hf-parity container has torch +
transformers + safetensors pre-installed.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # flodl-hf/

# (label, [step, step, ...]) — labels match the per-family parity subcommands
# in flodl-hf/parity/fdl.yml. Two seqcls entries need a .bin → safetensors
# prestep because their Hub repos ship .bin-only.
MANIFEST: list[tuple[str, list[list[str]]]] = [
    ("bert", [["python", "scripts/parity_bert.py"]]),
    (
        "bert-seqcls",
        [
            ["python", "scripts/convert_bin_to_safetensors.py", "nateraw/bert-base-uncased-emotion"],
            ["python", "scripts/parity_bert_seqcls.py"],
        ],
    ),
    ("bert-tokencls", [["python", "scripts/parity_bert_tokencls.py"]]),
    ("bert-qa", [["python", "scripts/parity_bert_qa.py"]]),
    ("bert-mlm", [["python", "scripts/parity_bert_mlm.py"]]),
    ("roberta", [["python", "scripts/parity_roberta.py"]]),
    (
        "roberta-seqcls",
        [
            ["python", "scripts/convert_bin_to_safetensors.py", "cardiffnlp/twitter-roberta-base-sentiment-latest"],
            ["python", "scripts/parity_roberta_seqcls.py"],
        ],
    ),
    ("roberta-tokencls", [["python", "scripts/parity_roberta_tokencls.py"]]),
    ("roberta-qa", [["python", "scripts/parity_roberta_qa.py"]]),
    ("roberta-mlm", [["python", "scripts/parity_roberta_mlm.py"]]),
    ("distilbert", [["python", "scripts/parity_distilbert.py"]]),
    ("distilbert-seqcls", [["python", "scripts/parity_distilbert_seqcls.py"]]),
    ("distilbert-tokencls", [["python", "scripts/parity_distilbert_tokencls.py"]]),
    ("distilbert-qa", [["python", "scripts/parity_distilbert_qa.py"]]),
    ("distilbert-mlm", [["python", "scripts/parity_distilbert_mlm.py"]]),
    ("albert", [["python", "scripts/parity_albert.py"]]),
    ("albert-seqcls", [["python", "scripts/parity_albert_seqcls.py"]]),
    ("albert-tokencls", [["python", "scripts/parity_albert_tokencls.py"]]),
    ("albert-qa", [["python", "scripts/parity_albert_qa.py"]]),
    ("albert-mlm", [["python", "scripts/parity_albert_mlm.py"]]),
    ("xlm-roberta", [["python", "scripts/parity_xlm_roberta.py"]]),
    ("deberta-v2", [["python", "scripts/parity_deberta_v2.py"]]),
]


def main() -> int:
    results: list[tuple[str, bool, float]] = []
    for label, steps in MANIFEST:
        print(f"\n=== {label} ===", flush=True)
        t0 = time.monotonic()
        ok = True
        for cmd in steps:
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            if rc != 0:
                ok = False
                print(f"!! {label}: step `{' '.join(cmd)}` exited {rc}", flush=True)
                break
        results.append((label, ok, time.monotonic() - t0))

    print("\n=== summary ===")
    width = max(len(label) for label, _, _ in results)
    for label, ok, elapsed in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {label.ljust(width)}  {status}  {elapsed:6.1f}s")
    failed = [label for label, ok, _ in results if not ok]
    if failed:
        print(f"\n{len(failed)} of {len(results)} failed: {', '.join(failed)}")
        return 1
    print(f"\nall {len(results)} parity fixtures regenerated successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
