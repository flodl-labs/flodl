# flodl Benchmark Suite

Head-to-head comparison of flodl (Rust) and PyTorch (Python) on identical
model architectures. Both frameworks use the same libtorch C++ backend --
the benchmarks measure framework overhead, not kernel speed.

## Quick start

```bash
fdl bench                        # CUDA benchmarks (all tiers)
fdl bench cpu                    # CPU-only
fdl bench --tier1                # tier 1 only
fdl bench --optim                # optimizer-step benchmarks only
fdl bench --bench mlp            # single benchmark
```

Everything runs in Docker. No local Rust or Python required. Full option
list: `fdl bench --help` (auto-populated from `run.sh --fdl-schema`).

## Publication mode

```bash
fdl bench publish --lock-clocks 2407
```

The `publish` preset sets `--rounds 10 --warmup-secs 15`; override any
flag on the command line (e.g. `--rounds 20 --output …`).

This enables the full methodology designed for reproducible, publishable
results:

- **Interleaved rounds** -- alternates which framework runs first each round
  (odd: Rust first, even: Python first) to distribute thermal drift and
  background noise equally
- **GPU clock locking** -- pins the GPU to the frequency given with
  `--lock-clocks`, eliminating boost clock variance. Older drivers report a
  base clock that is used when the flag is absent; recent ones do not, and
  the run then says the clocks stay unlocked
- **Extended warmup** -- 15-second GPU matmul burst before any measurement to
  stabilize thermals
- **Multi-round statistics** -- each round's best-run median becomes one sample;
  the final reported median is the median of those samples, with standard
  deviation across rounds

### WSL2

WSL2's `nvidia-smi` is a shim that cannot lock GPU clocks, and a lock set
from Windows applies to WSL2 workloads. To publish from WSL2, lock the
clocks on the Windows host (`nvidia-smi -lgc <MHz>,<MHz>` from an admin
prompt), pass the same value as `--lock-clocks` so the report records it,
and unlock afterwards (`nvidia-smi -rgc`). The reference results are
produced on native Linux.

### Keeping a run

Every run writes to its own directory, `runs/<timestamp>/` (per-round JSON,
the merged results and the report). That directory is gitignored and the
ten most recent runs are kept, so running the suite never touches results
that were published.

```bash
fdl bench promote                 # copy the latest run into published/
fdl bench --promote <timestamp>   # or a specific one
```

The published set is named by the date and commit in its report, is never
overwritten, and is meant to be committed with the results it backs. The
step warns when the run was a single round, had unlocked GPU clocks, or
measured uncommitted sources.

## Measurement methodology

### Per-benchmark structure

Each benchmark consists of multiple **runs**. The first run is a warmup
(discarded). The remaining runs are measured.

Each run:
1. **3 warmup epochs** -- populate caches, trigger cuDNN autotuning
2. **20 measured epochs** -- timed individually with `gpu_synchronize()`
   barriers

The **best run** (lowest median across its 20 epochs) is reported. This
filters process-level noise (GC pauses, scheduling jitter) while preserving
intra-run variance.

### Multi-round aggregation

With `--rounds N`, the entire Rust+Python suite runs N times in alternating
order. For each benchmark:

- Each round contributes one best-run median
- The final median is the **median of N best-run medians**
- Standard deviation is computed across those N samples

This gives N independent measurements that are robust to thermal transients
and framework warm-up effects.

### VRAM measurement

Peak VRAM is reset **after** the warmup run to exclude cuDNN autotuning
workspace and JIT compilation buffers. Two metrics are reported:

- **alloc** -- peak active tensor bytes (`torch.cuda.max_memory_allocated()` /
  `gpu_peak_active_bytes()`)
- **reserved** -- peak allocator pool reservation
  (`torch.cuda.max_memory_reserved()` / `gpu_peak_reserved_bytes()`)

Both frameworks use the same libtorch caching allocator, so the numbers are
directly comparable.

### Statistical reporting

The comparison prints three tables, one per thing measured.

**Training throughput** and **optimizer step** share these columns:

| Column | Meaning |
|--------|---------|
| tensors | Number of parameter tensors. With the step time it says how launch-heavy a row is: many small tensors and a short step is where framework overhead shows |
| PyTorch / flodl | Median time of one step (best-run median, divided by the steps per epoch) |
| delta | flodl against PyTorch, negative = flodl faster, with its uncertainty in points |
| verdict | `flodl faster`, `PyTorch faster`, or `within noise` when the two sides differ by less than 2 combined sigma |

Sigma is the scaled MAD of the best-run medians (across runs, or across
rounds with `--rounds`), with a floor of 0.5% of the PyTorch time so a
handful of identical samples cannot make any difference look real.

**Memory** reports peak allocated and reserved VRAM for each side.

## Benchmark tiers

### Tier 1 -- Baseline patterns

Pure module + optimizer throughput. These are standard architectures that both
frameworks express naturally.

| Benchmark | Architecture | What it measures |
|-----------|-------------|------------------|
| `mlp` | 5 Linear layers (1024 -> 2048 x3 -> 1024) with GELU and LayerNorm | Matmul + activation + Adam throughput |
| `convnet` | 3x Conv2d + MaxPool + 2x Linear | Conv + pooling + mixed pipeline |
| `gru_seq` | GRUCell unrolled over 50 timesteps | Per-timestep RNN overhead |
| `transformer` | Embedding + encoder layers (attention, feed-forward, LayerNorm) + projection | Attention throughput |
| `lstm_seq` | Multi-layer LSTM + output projection | Fused recurrent kernels, comparable to `gru_seq` |
| `conv_autoenc` | Conv2d encoder + ConvTranspose2d decoder | Transposed convolution throughput |
| `resnet20` | ResNet-20 on 32x32 inputs, batch 64, SGD with momentum and weight decay | The launch-heavy case: 65 small tensors and a step of a few ms |

### Tier 2 -- Graph builder patterns

These use flodl's graph builder constructs. The PyTorch equivalents are
hand-written `nn.Module` subclasses -- the benchmark measures how much overhead
(if any) the declarative graph engine adds compared to manual Python wiring.

| Benchmark | Architecture | What it measures |
|-----------|-------------|------------------|
| `residual_tower` | 8 residual blocks with LayerNorm | `also()` residual connections |
| `gated_routing` | 3-expert soft gate with SoftmaxRouter | `gate()` mixture-of-experts |
| `iterative_refine` | Fixed 5-iteration refinement loop | `loop_body().for_n()` |
| `feedback_fixed` | 10-iteration loop (no adaptive halt) | Fixed loop overhead isolation |

### Optimizer step

The optimizer's `step()` alone, with no forward and no backward, on
ResNet-20's 65 parameter tensors and a fixed gradient. One row per
optimizer (`optim_sgd`, `optim_adam`, `optim_adamw`, `optim_rmsprop`,
`optim_adagrad`, `optim_nadam`, `optim_radam`), each with the settings a
user gets by default on both sides. The cost of a step on many small
tensors is all framework: the same elementwise updates, issued as a few
batched kernels or as a loop over tensors.

## Project structure

```
benchmarks/
  src/
    main.rs          -- CLI entry point, tier dispatch
    harness.rs       -- Timing, warmup, VRAM tracking, JSON output
    tier1/           -- Rust benchmarks (mlp, convnet, gru_seq, resnet20, ...)
    tier2/           -- Rust benchmarks (residual_tower, gated_routing, ...)
    optim/           -- Optimizer-step benchmarks
  python/
    run_all.py       -- CLI entry point (mirrors main.rs)
    harness.py       -- Python harness (mirrors harness.rs)
    compare.py       -- Comparison tables (throughput, optimizer step, memory)
    merge_rounds.py  -- Multi-round aggregation
    tier1/           -- Python benchmarks
    tier2/           -- Python benchmarks
    optim/           -- Optimizer-step benchmarks
  run.sh             -- Orchestrator: build, warmup, interleave, merge, compare
  runs/<timestamp>/  -- One directory per run: rounds, merged JSON, report (gitignored)
  published/         -- Promoted runs: the raw data behind published results (committed)
```

## Harness parity

The Rust and Python harnesses are structurally identical:

- Same run/epoch/warmup structure
- Same best-of-N selection logic
- Same VRAM reset timing (after warmup run)
- Same JSON output schema
- Same `gpu_synchronize()` barriers at measurement boundaries
- Same cuDNN benchmark mode (`torch.backends.cudnn.benchmark = True` /
  `set_cudnn_benchmark(true)`)

This ensures the comparison measures framework overhead, not harness
differences.

## Adding a benchmark

1. Create `src/tier{1,2}/my_bench.rs` -- implement a function
   `pub fn run(device: Device) -> flodl::Result<BenchResult>` that builds
   the model, generates data, and calls `harness::run_benchmark()` with
   the model's parameters
2. Create `python/tier{1,2}/my_bench.py` -- mirror the architecture exactly
3. Register in `src/tier{1,2}/mod.rs` and `python/tier{1,2}/__init__.py`
4. Add to the dispatch table in `src/main.rs` and `python/run_all.py`

The key rule: **identical architectures**. Same layer sizes, same optimizer,
same learning rate, same batch count. The only variable is the framework.
