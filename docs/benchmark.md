# Benchmark: flodl vs PyTorch

Seven models, ten interleaved rounds, idle machine. Same architectures, same
hyperparameters, same CUDA kernels — the only variable is the framework
overhead.

> This is a living document. The benchmark suite ships with flodl and runs in
> Docker — anyone can reproduce these numbers on their own hardware.

## Results (v0.1.3 — March 2026)

| Model | PyTorch | flodl | Delta | Py σ | Rs σ | Py alloc | Rs alloc | Py rsrvd | Rs rsrvd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp | 271.0 ms | 188.5 ms | **-30%** | ±10.1 | ±2.9 | 440 MB | 401 MB | 466 MB | 418 MB |
| convnet | 1189.4 ms | 1190.5 ms | +0% | ±2.7 | ±1.0 | 1102 MB | 1103 MB | 1420 MB | 1412 MB |
| gru_seq | 1015.3 ms | 949.7 ms | **-6%** | ±222.4 | ±10.8 | 523 MB | 526 MB | 566 MB | 558 MB |
| residual_tower | 371.3 ms | 278.6 ms | **-25%** | ±25.9 | ±3.6 | 400 MB | 382 MB | 452 MB | 396 MB |
| gated_routing | 222.6 ms | 196.9 ms | **-12%** | ±13.8 | ±2.6 | 126 MB | 129 MB | 148 MB | 146 MB |
| iterative_refine | 208.7 ms | 186.7 ms | **-11%** | ±27.2 | ±5.6 | 196 MB | 195 MB | 234 MB | 214 MB |
| feedback_fixed | 250.2 ms | 207.2 ms | **-17%** | ±27.3 | ±8.7 | 64 MB | 63 MB | 90 MB | 88 MB |

**flodl wins 6 of 7 benchmarks. The one tie (convnet) is compute-bound — both
frameworks saturate the GPU, proving they dispatch identical CUDA kernels.**

### Reading the table

- **Median epoch time** — median of the best run's 20 measured epochs, taken
  across 10 rounds. Lower is faster.
- **σ** — scaled MAD (Median Absolute Deviation × 1.4826) across rounds,
  σ-equivalent for normal data but robust to OS/GC outliers. Lower means
  more predictable.
- **alloc** — peak active tensor bytes (`torch.cuda.max_memory_allocated` /
  `cuda_peak_active_bytes`). This is the memory your tensors actually use.
- **rsrvd** — peak allocator reservation (`torch.cuda.max_memory_reserved` /
  `cuda_peak_reserved_bytes`). This is the memory the caching allocator holds.

### The variance story

The σ column is arguably more important than the speed column. In production,
your P99 latency is your real latency. flodl's variance is consistently 3-20x
tighter than PyTorch's:

| Model | PyTorch σ | flodl σ | Ratio |
|---|---:|---:|---:|
| mlp | ±10.1 ms | ±2.9 ms | 3.5x tighter |
| convnet | ±2.7 ms | ±1.0 ms | 2.7x tighter |
| gru_seq | ±222.4 ms | ±10.8 ms | 20.6x tighter |
| residual_tower | ±25.9 ms | ±3.6 ms | 7.2x tighter |
| gated_routing | ±13.8 ms | ±2.6 ms | 5.3x tighter |
| iterative_refine | ±27.2 ms | ±5.6 ms | 4.9x tighter |
| feedback_fixed | ±27.3 ms | ±8.7 ms | 3.1x tighter |

This comes from Rust's execution model: no garbage collector pauses, no Python
interpreter overhead, deterministic drop order, predictable memory layout. You
can't bolt this onto PyTorch — it's a property of the language runtime.

For distributed training, tight variance means synchronization barriers spend
less time waiting for the slowest worker. When every node finishes within ±3ms
instead of ±25ms, your effective throughput scales better.

## What was measured

### Tier 1 — Standard architectures

These models use standard nn modules (Linear, Conv2d, LSTM, GRU, MultiheadAttention,
ConvTranspose2d, activations, normalization). Both sides use identical PyTorch-equivalent code.

| Model | Architecture | Params | Batch | What it tests |
|---|---|---:|---:|---|
| **mlp** | 5-layer Linear+GELU+LayerNorm (1024→2048→1024) | 16.8M | 256 | Raw matmul + activation throughput |
| **convnet** | 4-layer Conv2d+BN+MaxPool → classifier head | 4.0M | 128 | Convolution pipeline, cuDNN dispatch |
| **gru_seq** | 50-step GRU → linear projection | 1.3M | 128 | Fused RNN sequence throughput |
| **transformer** | 4-layer encoder (MHA+FFN+LayerNorm+residual) | ~25M | 32 | Attention throughput, cross-entropy loss |
| **lstm_seq** | 2-layer LSTM → linear projection | ~2.5M | 128 | Fused LSTM throughput (comparable to gru_seq) |
| **conv_autoenc** | Conv2d encoder + ConvTranspose2d decoder | ~3.5M | 64 | Transposed convolution, image reconstruction |

### Tier 2 — Graph builder architectures

These models use flodl's `FlowBuilder` (residual connections, gated routing,
loops). The PyTorch equivalents use manual `forward()` implementations with
the same architecture.

| Model | Architecture | Params | Batch | What it tests |
|---|---|---:|---:|---|
| **residual_tower** | 12 residual blocks via `.also()` | 14.7M | 256 | Skip-connection overhead, deep graphs |
| **gated_routing** | 8-expert MoE via `.gate(SoftmaxRouter)` | 2.6M | 256 | Soft routing, vectorized gate combination |
| **iterative_refine** | Encoder → 8-step `.loop_body().for_n()` → decoder | 3.2M | 256 | Fixed-iteration loop overhead |
| **feedback_fixed** | Encoder → 10-step feedback loop → decoder | 0.8M | 128 | Recurrent loop with smaller model |

All models use Adam optimizer with lr=1e-3, MSE loss, and random synthetic
data. Each epoch runs 50 batches.

## Methodology

### Publication-grade protocol

The benchmark suite is designed for reproducibility and statistical rigor:

1. **Interleaved rounds** — 10 rounds alternate Rust-first/Python-first to
   distribute thermal drift and background noise equally.

2. **GPU clocks** — The RTX 5060 Ti ran at its default applications clock
   (3090 MHz). On non-WSL systems, the suite auto-detects and locks the
   GPU clock; on WSL2, clock locking requires the host-side PowerShell
   script. The tight σ values across 10 rounds confirm measurement
   stability even without explicit locking.

3. **Multi-run best-of** — each round runs 4 complete passes (1 warmup + 3
   measured). The best run's median is reported, filtering process-level noise.

4. **Per-epoch measurement** — each measured run collects 20 epoch times with
   3 warmup epochs. The median is used (not mean) to resist outliers.

5. **GPU warmup** — 15 seconds of synthetic GPU load before any measurement
   to stabilize thermal state and clocks.

6. **Idle machine** — system rebooted, non-essential services stopped, display
   off. No user processes competing for resources.

7. **Docker isolation** — both frameworks run in the same container image with
   identical CUDA toolkit, libtorch, and system libraries.

8. **Peak VRAM tracking** — VRAM measured via `cuda_reset_peak_stats()` /
   `cuda_peak_active_bytes()`, matching `torch.cuda.max_memory_allocated()`
   semantics. Not a snapshot — the true high-water mark.

### Statistical model

For N rounds, each with best-of-K runs:

- **Reported median** = median of N best-run medians (one per round)
- **Reported σ** = scaled MAD of N best-run medians (MAD × 1.4826, σ-equivalent
  for normal distributions)

This gives N independent samples (one per round) for robust statistics. Scaled
MAD resists the outlier rounds that inflate stdev on shared-driver environments
(WSL2, background OS scheduling), while remaining directly comparable to
standard deviation for well-behaved data.

### Environment

| Component | Version |
|---|---|
| flodl | 0.1.3 |
| PyTorch | 2.6.0+cu126 |
| Rust | 1.85.1 |
| CUDA | 12.6 (runtime) |
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB) |
| Driver | 572.83 |
| OS | Ubuntu 24.04 (Docker on WSL2) |
| CPU | AMD Ryzen 7 7800X3D |

## Why flodl is faster

The speed advantage comes from eliminating framework overhead, not from
different CUDA kernels. Both frameworks call the same libtorch C++ backend.
The difference is what happens between kernel launches:

### Host-side dispatch elimination

PyTorch's forward pass goes through Python → TorchScript dispatch → C++.
flodl calls libtorch directly from Rust via FFI. For models with many small
operations (MLP, residual tower, routing), the per-op dispatch overhead
dominates — hence the 25-30% improvement.

For compute-bound models (convnet), both frameworks spend >99% of time in
cuDNN kernels and the dispatch overhead is invisible — hence the 0% delta.

### Fused operations

| Operation | PyTorch | flodl |
|---|---|---|
| Adam step | N×4 kernel launches (per-parameter) | 1 fused multi-tensor kernel (`_fused_adamw_`) |
| Gradient clipping | 2N kernels (norm + scale per param) | 2 kernels total (`_foreach_norm` + `_foreach_mul_`) |
| zero_grad | N zeroing kernels | set_to_none (no kernel) |

With N = hundreds of parameters, this adds up every batch.

### Pre-computed graph routing

flodl's `Graph::build()` pre-computes a flat Vec-indexed routing table.
Forward pass dispatch is array indexing — no HashMap lookups, no dynamic
allocation. Gate combination uses vectorized stack+broadcast+sum (3 kernels
regardless of expert count).

### Memory predictability

Rust's ownership model means tensor memory is freed deterministically at scope
exit. No GC pauses, no deferred deallocation, no memory pressure spikes. This
is visible in the σ column — especially for GRU where PyTorch's ±222ms
variance likely includes GC-triggered stalls during the 50-step unrolled
sequence.

## Optimizations since v0.1.1

The first published benchmark (v0.1.1, March 18, 2026) measured 19% faster
than PyTorch on a real FBRL training workload. Since then, significant
optimization work has been done:

| Optimization | What it does | Impact |
|---|---|---|
| **Fused Adam/AdamW** | Single multi-tensor CUDA kernel for the full optimizer step | Eliminates N×4 per-parameter kernel launches |
| **Foreach ops** | 7 batched tensor operations (`foreach_zero_`, `foreach_norm`, `foreach_mul_scalar_`, etc.) | Single kernel for N parameters |
| **Fused gradient clipping** | `clip_grad_norm` via `_foreach_norm` + `_foreach_mul_` | 2 kernels instead of 2N |
| **CUDA Graphs** | Capture/replay kernel sequences, eliminating CPU dispatch overhead | Available for static-shape workloads |
| **Automatic mixed precision** | `AutocastGuard` / `autocast()` + `GradScaler` for fp16/bf16 | Up to 3x on Tensor Core GPUs |
| **Channels-last memory** | NHWC layout for Conv2d on Tensor Core GPUs | 8-35% for convolution-heavy models |
| **Async device transfer** | `pin_memory()` + `copy_(non_blocking)` + `to_device_async()` | Overlaps data transfer with computation |
| **Pre-computed graph routing** | Vec-indexed routing, cached execution buffers | Zero allocation in forward pass |
| **Loop fast-path** | Direct `forward()` call when loop body has no refs | Eliminates HashMap overhead per iteration |
| **Vectorized gate combination** | Stack+broadcast+sum for MoE routing | 3 kernels regardless of expert count |

These optimizations are automatic — users get them without changing their
training code. The benchmark suite does not enable CUDA Graphs, mixed
precision, or channels-last format to ensure a fair apples-to-apples
comparison with standard PyTorch.

## Earlier benchmark: FBRL letter model (v0.1.1)

> **Note:** This benchmark was run with flodl v0.1.1 before the optimizations
> listed above. It measures a different thing — real end-to-end training of a
> non-trivial model, including data loading, checkpointing, and a live
> dashboard. A re-run with the current version is planned.

| Metric | PyTorch 2.5.1 | flodl 0.1.1 | Delta |
|--------|--------------|-------|-------|
| Avg epoch | 49.7s | 40.3s | **-19%** |
| Total | 82m 50s | 67m 10s | **-19%** |
| GPU utilization | ~80% (spiky) | 88-92% (flat) | more stable |
| VRAM (reserved) | 2,805 MB | 2,977 MB | +6%* |

\* VRAM overhead is a fixed cost from CUDA toolkit version difference (cu126
vs cu124) and libtorch's caching allocator. Does not grow with model size.

**Model:** FBRL letter recognition — recurrent attention with GRU controller,
CNN encoder, deconv decoder, 9-component loss stack. 11,440 grayscale 128x128
images, 100 epochs on a GTX 1060 6GB.

**Key finding:** flodl ran a live training dashboard and async gzip
checkpointing — more work per epoch than PyTorch — and was still 19% faster.
The epoch time standard deviation was 0.10s (flodl) vs 0.85s (PyTorch).

Full details, raw data, and reproduction steps at
[fbrl@5c58d71](https://github.com/fab2s/fbrl/tree/5c58d71).

## Reproduce

### Synthetic benchmark suite

The benchmark suite runs entirely in Docker. No local Rust or Python
installation required.

```bash
git clone https://github.com/fab2s/floDl.git
cd floDl

# Quick single-round benchmark
make bench

# Publication benchmark (10 interleaved rounds, locked clocks)
make bench-publish

# CPU-only
make bench-cpu

# Custom configuration
make bench-publish ROUNDS=20 CLOCK=2407 OUTPUT=benchmarks/report.txt
```

Each round runs the full Rust suite then the full Python suite (alternating
order). Results are merged across rounds with median-of-medians aggregation.
The final report includes environment metadata, per-model results with σ, and
VRAM tracking.

### What you need

- Docker with NVIDIA Container Toolkit (for GPU benchmarks)
- `make`
- An NVIDIA GPU (any generation — the suite adapts)

The Docker image includes Rust, Python, PyTorch, libtorch, and all
dependencies. First build takes ~15 minutes; subsequent runs reuse the cached
image.

### Raw data

Per-round JSON files are saved in `benchmarks/rounds/` with full per-epoch
timings, VRAM measurements, and model metadata. The merged results and final
report are written to `benchmarks/report.txt`.

### Adding benchmarks

The suite is extensible. Each benchmark is a Rust module in `benchmarks/src/`
with a matching Python implementation in `benchmarks/python/`. See existing
tier1/tier2 benchmarks for the pattern. Both sides use the same harness with
identical measurement methodology.
