#!/usr/bin/env bash
# Run flodl (Rust) and PyTorch (Python) benchmarks, then compare.
#
# Usage: benchmarks/run.sh [OPTIONS] [-- BENCH_ARGS]
#
# Options:
#   --cpu              CPU-only mode (no CUDA)
#   --rounds N         Number of interleaved rounds (default: 1)
#                      Each round runs the full Rust suite then the full Python
#                      suite. Alternating distributes thermal drift, clock
#                      changes, and background noise equally across frameworks.
#   --lock-clocks F    Lock GPU clocks to F MHz before benchmarking and
#                      unlock after. Prevents boost clock variance at the
#                      cost of peak throughput. Use base clock for stability
#                      (e.g., 2407 for RTX 5060 Ti).
#   --warmup-secs S    GPU warmup duration in seconds (default: 10)
#   --tier1|--tier2    Filter benchmarks by tier
#   --bench NAME       Run a single benchmark
#
# Each benchmark internally runs 1 warmup + 3 measured runs (each with
# 3 warmup + 20 measured epochs). The best run's median is reported.
# With --rounds N, results are merged across rounds for robust statistics:
# σ is computed from N best-run medians (one per round).
#
# For publication: --rounds 10 --lock-clocks <base_freq>
# Expects to run inside the bench Docker container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

# --- Parse arguments ---
CPU_MODE=0
ROUNDS=1
LOCK_CLOCKS=""
WARMUP_SECS=10
PASS_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)
            CPU_MODE=1
            shift
            ;;
        --rounds)
            ROUNDS="${2:?--rounds requires a number}"
            shift 2
            ;;
        --lock-clocks)
            LOCK_CLOCKS="${2:?--lock-clocks requires a frequency in MHz}"
            shift 2
            ;;
        --warmup-secs)
            WARMUP_SECS="${2:?--warmup-secs requires a number}"
            shift 2
            ;;
        --)
            shift
            PASS_ARGS+=("$@")
            break
            ;;
        *)
            PASS_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$CPU_MODE" -eq 1 ]; then
    MODE="cpu"
    CARGO_FEATURES=""
    echo "=== Benchmark mode: CPU ==="
else
    MODE="cuda"
    CARGO_FEATURES="--features cuda"
    echo "=== Benchmark mode: CUDA ==="
fi
echo "    rounds: $ROUNDS"
echo "    warmup: ${WARMUP_SECS}s"
[ -n "$LOCK_CLOCKS" ] && echo "    clocks: locked at ${LOCK_CLOCKS} MHz"
echo ""

# --- Build ---
echo "=== Building flodl benchmarks (release, $MODE) ==="
cargo build --manifest-path "$SCRIPT_DIR/Cargo.toml" --release $CARGO_FEATURES 2>&1
echo ""

# --- Clock locking ---
CLOCKS_LOCKED=0
cleanup_clocks() {
    if [ "$CLOCKS_LOCKED" -eq 1 ]; then
        echo ""
        echo "=== Unlocking GPU clocks ==="
        nvidia-smi -rgc 2>/dev/null || true
        CLOCKS_LOCKED=0
    fi
}
trap cleanup_clocks EXIT

if [ -n "$LOCK_CLOCKS" ] && [ "$CPU_MODE" -eq 0 ]; then
    echo "=== Locking GPU clocks to ${LOCK_CLOCKS} MHz ==="
    nvidia-smi -lgc "$LOCK_CLOCKS","$LOCK_CLOCKS" 2>&1
    CLOCKS_LOCKED=1
    echo ""
fi

# --- GPU warmup ---
if [ "$CPU_MODE" -eq 0 ]; then
    echo "=== GPU warmup (${WARMUP_SECS}s, stabilizing clocks + thermals) ==="
    python3 -c "
import torch, time
if torch.cuda.is_available():
    d = torch.device('cuda')
    a = torch.randn(4096, 4096, device=d)
    t0 = time.time()
    while time.time() - t0 < ${WARMUP_SECS}:
        a = a @ a
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()
    print(f'  GPU warm ({time.time()-t0:.0f}s matmul burst)')
else:
    print('  no CUDA, skipping')
" 2>&1
    echo ""
fi

# --- Per-round temp directory ---
ROUND_DIR=$(mktemp -d /tmp/flodl_bench_XXXXXX)
echo "=== Round data: $ROUND_DIR ==="
echo ""

# --- Interleaved rounds ---
for ((r=1; r<=ROUNDS; r++)); do
    echo "=============================================="
    echo "=== Round $r/$ROUNDS ==="
    echo "=============================================="
    echo ""

    # Rust suite
    echo "--- flodl (Rust) round $r ---"
    "$SCRIPT_DIR/target/release/flodl-bench" --json \
        "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}" \
        > "$ROUND_DIR/flodl_r${r}.json" 2>/dev/stderr
    echo ""

    # Python suite
    echo "--- PyTorch (Python) round $r ---"
    cd "$SCRIPT_DIR/python"
    if [ "$CPU_MODE" -eq 1 ]; then
        CUDA_VISIBLE_DEVICES="" python3 run_all.py --json \
            "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}" \
            > "$ROUND_DIR/pytorch_r${r}.json" 2>/dev/stderr
    else
        python3 run_all.py --json \
            "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}" \
            > "$ROUND_DIR/pytorch_r${r}.json" 2>/dev/stderr
    fi
    echo ""
done

# --- Merge rounds ---
cd "$SCRIPT_DIR/python"

if [ "$ROUNDS" -eq 1 ]; then
    # Single round: use files directly
    cp "$ROUND_DIR/flodl_r1.json" /tmp/flodl_bench.json
    cp "$ROUND_DIR/pytorch_r1.json" /tmp/pytorch_bench.json
else
    echo "=== Merging $ROUNDS rounds ==="
    python3 merge_rounds.py "$ROUND_DIR"/flodl_r*.json > /tmp/flodl_bench.json
    python3 merge_rounds.py "$ROUND_DIR"/pytorch_r*.json > /tmp/pytorch_bench.json
    echo ""
fi

# --- Compare ---
echo "=== Comparison ($MODE, $ROUNDS round$([ "$ROUNDS" -gt 1 ] && echo 's')) ==="
python3 compare.py /tmp/flodl_bench.json /tmp/pytorch_bench.json
