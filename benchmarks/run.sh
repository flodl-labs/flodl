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
#                      When omitted (and not WSL), auto-detects the GPU's
#                      default applications clock and locks to that.
#   --warmup-secs S    GPU warmup duration in seconds (default: 10)
#   --output FILE      Write the comparison report to FILE (with metadata header)
#   --tier1|--tier2    Filter benchmarks by tier
#   --bench NAME       Run a single benchmark
#
# Each benchmark internally runs 1 warmup + 3 measured runs (each with
# 3 warmup + 20 measured epochs). The best run's median is reported.
# With --rounds N, results are merged across rounds for robust statistics:
# σ = scaled MAD (Median Absolute Deviation × 1.4826) of the N best-run
# medians — σ-equivalent for normal data, robust to OS/GC outliers.
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
OUTPUT_FILE="$WORKSPACE/benchmarks/report.txt"
[ -f "$OUTPUT_FILE" ] && mv "$OUTPUT_FILE" "${OUTPUT_FILE%.txt}.$(date +%Y-%m-%d-%H-%M-%S).txt"
PASS_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)
            CPU_MODE=1
            shift
            ;;
        --rounds)
            ROUNDS="${2:?--rounds requires a number}"
            if ! [[ "$ROUNDS" =~ ^[0-9]+$ ]]; then
                echo "ERROR: --rounds expects a numeric value, got '$ROUNDS'" >&2
                exit 1
            fi
            shift 2
            ;;
        --lock-clocks)
            LOCK_CLOCKS="${2:?--lock-clocks requires a frequency in MHz}"
            if ! [[ "$LOCK_CLOCKS" =~ ^[0-9]+$ ]]; then
                echo "ERROR: --lock-clocks expects a numeric MHz value, got '$LOCK_CLOCKS'" >&2
                exit 1
            fi
            shift 2
            ;;
        --warmup-secs)
            WARMUP_SECS="${2:?--warmup-secs requires a number}"
            if ! [[ "$WARMUP_SECS" =~ ^[0-9]+$ ]]; then
                echo "ERROR: --warmup-secs expects a numeric value, got '$WARMUP_SECS'" >&2
                exit 1
            fi
            shift 2
            ;;
        --output)
            OUTPUT_FILE="${2:?--output requires a file path}"
            # resolve relative paths against workspace root
            [[ "$OUTPUT_FILE" != /* ]] && OUTPUT_FILE="$WORKSPACE/$OUTPUT_FILE"
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

# --- Detect WSL ---
IS_WSL=0
if grep -qi 'microsoft\|wsl' /proc/version 2>/dev/null; then
    IS_WSL=1
fi

# --- Auto-detect GPU base clock (when not manually set and not WSL) ---
if [ -z "$LOCK_CLOCKS" ] && [ "$CPU_MODE" -eq 0 ] && [ "$IS_WSL" -eq 0 ] \
   && command -v nvidia-smi >/dev/null 2>&1; then
    # Try default applications clock first (works on most desktop GPUs)
    DETECTED=$(nvidia-smi --query-gpu=clocks.default_applications.graphics \
        --format=csv,noheader 2>/dev/null | grep -oP '\d+' | head -1)

    # Fallback: parse "Default Applications Clocks" from verbose output
    if [ -z "$DETECTED" ]; then
        DETECTED=$(nvidia-smi -q -d CLOCK 2>/dev/null \
            | sed -n '/Default Applications Clocks/,/^$/p' \
            | grep -i 'Graphics' | grep -oP '\d+' | head -1)
    fi

    if [ -n "$DETECTED" ]; then
        LOCK_CLOCKS="$DETECTED"
        echo "=== Auto-detected GPU base clock: ${LOCK_CLOCKS} MHz ==="
        echo ""
    fi
fi

# --- Clock locking ---
# WSL2 shares the host GPU driver. Clock control must happen on the
# Windows side (bench-publish.ps1 handles this). Never call nvidia-smi
# -lgc/-rgc from WSL — the shim may reset the host-side lock.
CLOCKS_LOCKED=0
CLOCKS_EXTERNAL=0
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
    if [ "$IS_WSL" -eq 1 ]; then
        echo "=== GPU clocks managed by host (${LOCK_CLOCKS} MHz) ==="
        CLOCKS_EXTERNAL=1
    else
        echo "=== Locking GPU clocks to ${LOCK_CLOCKS} MHz ==="
        if nvidia-smi -lgc "$LOCK_CLOCKS","$LOCK_CLOCKS" 2>&1; then
            CLOCKS_LOCKED=1
        else
            echo "WARNING: Could not lock GPU clocks. Results may show higher variance."
        fi
    fi
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

# --- Per-round data directory (in workspace so it survives container exit) ---
ROUND_DIR="$SCRIPT_DIR/rounds"
rm -rf "$ROUND_DIR"
mkdir -p "$ROUND_DIR"
echo "=== Round data: $ROUND_DIR ==="
echo ""

# --- Interleaved rounds ---
# Alternate which framework runs first each round to eliminate systematic
# thermal/cache bias. Odd rounds: Rust first. Even rounds: Python first.
run_rust_round() {
    echo "--- flodl (Rust) round $1 ---"
    "$SCRIPT_DIR/target/release/flodl-bench" --json \
        "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}" \
        > "$ROUND_DIR/flodl_r${1}.json" 2>/dev/stderr
    echo ""
}

run_python_round() {
    echo "--- PyTorch (Python) round $1 ---"
    cd "$SCRIPT_DIR/python"
    if [ "$CPU_MODE" -eq 1 ]; then
        CUDA_VISIBLE_DEVICES="" python3 run_all.py --json \
            "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}" \
            > "$ROUND_DIR/pytorch_r${1}.json" 2>/dev/stderr
    else
        python3 run_all.py --json \
            "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}" \
            > "$ROUND_DIR/pytorch_r${1}.json" 2>/dev/stderr
    fi
    echo ""
}

for ((r=1; r<=ROUNDS; r++)); do
    echo "=============================================="
    if (( r % 2 == 1 )); then
        echo "=== Round $r/$ROUNDS (Rust first) ==="
    else
        echo "=== Round $r/$ROUNDS (Python first) ==="
    fi
    echo "=============================================="
    echo ""

    if (( r % 2 == 1 )); then
        run_rust_round "$r"
        run_python_round "$r"
    else
        run_python_round "$r"
        run_rust_round "$r"
    fi
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
COMPARISON_HEADER="=== Comparison ($MODE, $ROUNDS round$([ "$ROUNDS" -gt 1 ] && echo 's' || true)) ==="

if [ -n "$OUTPUT_FILE" ]; then
    # Collect metadata for the report header.
    FLODL_VERSION=$(cargo metadata --manifest-path "$WORKSPACE/flodl/Cargo.toml" --no-deps --format-version 1 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['packages'][0]['version'])" 2>/dev/null || echo "unknown")
    PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    RUSTC_VERSION=$(rustc --version 2>/dev/null || echo "unknown")
    COMMIT_SHA=$(git -C "$WORKSPACE" rev-parse --short HEAD 2>/dev/null || echo "unknown")
    BENCH_DATE=$(date -u +"%Y-%m-%d %H:%M UTC")

    if [ "$CPU_MODE" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs)
        CUDA_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | xargs)
        CUDA_RT=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
        GPU_INFO="GPU:      $GPU_NAME | Driver $DRIVER_VERSION | CUDA $CUDA_RT"
        if [ "$CLOCKS_LOCKED" -eq 1 ]; then
            GPU_INFO="$GPU_INFO | Clocks locked at ${LOCK_CLOCKS} MHz"
        elif [ "$CLOCKS_EXTERNAL" -eq 1 ]; then
            GPU_INFO="$GPU_INFO | Clocks locked at ${LOCK_CLOCKS} MHz (host)"
        fi
    else
        GPU_INFO="GPU:      CPU-only"
    fi

    {
        echo "flodl Benchmark Report"
        echo "======================"
        echo ""
        echo "Date:     $BENCH_DATE"
        echo "Commit:   $COMMIT_SHA"
        echo "flodl:    v$FLODL_VERSION"
        echo "PyTorch:  $PYTORCH_VERSION"
        echo "Rust:     $RUSTC_VERSION"
        echo "$GPU_INFO"
        echo "Rounds:   $ROUNDS | Warmup: ${WARMUP_SECS}s"
        echo ""
        echo "$COMPARISON_HEADER"
        python3 compare.py /tmp/flodl_bench.json /tmp/pytorch_bench.json
    } > "$OUTPUT_FILE"

    echo "$COMPARISON_HEADER"
    python3 compare.py /tmp/flodl_bench.json /tmp/pytorch_bench.json
    echo ""
    echo "=== Report saved to $OUTPUT_FILE ==="
else
    echo "$COMPARISON_HEADER"
    python3 compare.py /tmp/flodl_bench.json /tmp/pytorch_bench.json
fi
