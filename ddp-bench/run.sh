#!/usr/bin/env bash
#
# ddp-bench orchestrator: build and run the DDP validation suite.
#
# Usage:
#   ./run.sh                              # all models, all modes
#   ./run.sh --model linear --mode solo-0 # single combo
#   ./run.sh --model convnet              # one model, all modes
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Rotate old report if exists
REPORT="report.txt"
if [ -f "$REPORT" ]; then
    TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
    mv "$REPORT" "report.${TIMESTAMP}.txt"
fi

echo "=== ddp-bench: building... ==="
cargo build --release --features cuda 2>&1

echo "=== ddp-bench: running... ==="
cargo run --release --features cuda -- "$@" 2>&1 | tee "$REPORT"

echo ""
echo "Report saved to $REPORT"
echo "Run artifacts in runs/"
