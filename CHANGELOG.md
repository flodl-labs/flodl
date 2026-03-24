# Changelog

All notable changes to floDl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

#### GPU Performance
- **Fused Adam/AdamW**: `_fused_adamw_` single multi-tensor CUDA kernel for the complete optimizer step across all parameters. Reduces ~4N kernel launches to 1 per parameter group. Automatic on CUDA — no API change needed. `grad_scale`/`found_inf` params exposed for GradScaler integration.
- **Foreach operations**: 7 batched tensor ops that reduce CUDA kernel launches — `foreach_add_scalar_`, `foreach_mul_scalar_`, `foreach_zero_`, `foreach_add_list_`, `foreach_norm`, `foreach_lerp_scalar_`, `foreach_sqrt_`. Used internally by fused optimizers and gradient clipping.
- **Fused gradient clipping**: `clip_grad_norm` now uses `_foreach_norm` + `_foreach_mul_` internally (2 kernels instead of 2N).
- **CUDA Graphs**: `CudaGraph` struct with capture/replay/reset for zero CPU dispatch overhead. `cuda_graph_capture()` convenience helper with warmup. `MemPoolId`, `CaptureMode` (Global/ThreadLocal/Relaxed), `cuda_graph_pool_handle()` for memory pool sharing. 2-5x speedup for models with many small kernels.
- **Autocast (AMP)**: `AutocastGuard` RAII wrapper and `autocast()` closure helper for automatic mixed-precision dispatch. Eligible ops (matmul, conv, linear) run in Float16/BFloat16 on Tensor Core GPUs. Up to 3x speedup on RTX 30xx+.
- **GradScaler**: Dynamic loss scaling for mixed-precision training. Scale, unscale with inf/nan detection, step with skip-on-inf, update with growth/backoff.

#### Tensor Operations
- **Channels-last memory format**: `Tensor::to_channels_last()` and `is_channels_last()` for NHWC layout. 8-35% speedup for Conv2d on Tensor Core GPUs.
- **Non-blocking device transfer**: `Tensor::to_device_async()` for overlapped CPU-to-GPU transfer. Pair with `pin_memory()` for maximum overlap.
- **`Tensor::copy_()`**: In-place copy with `non_blocking` parameter for async CUDA transfers. Used by CUDA Graph capture for data loading.
- **`Tensor::pin_memory()`** and `is_pinned()`: Page-locked CPU memory for fast async GPU transfers.
- **Peak VRAM tracking**: `cuda_peak_active_bytes()`, `cuda_peak_reserved_bytes()`, `cuda_reset_peak_stats()` — matches `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` / `reset_peak_memory_stats()` semantics. With `_idx` variants for multi-GPU.

#### Graph Engine
- **Pre-computed routing**: `Graph::build()` pre-computes a Vec-indexed routing table. Forward dispatch uses flat array indexing instead of HashMap lookups. Cached execution buffers reused across forward calls. Zero allocation during inference.
- **Vectorized gate combination**: Gate routing stacks all expert outputs and combines via broadcast multiply + sum (~3 kernel launches regardless of expert count, vs 3N with sequential accumulation).
- **Loop fast-path**: `for_n` loops detect at call time whether refs are needed and call `body.forward()` directly when no `.using()` is chained, skipping HashMap construction and `body_step` indirection.

#### Other
- **`MaxPool2d`** module: 2D max pooling with kernel size, stride, padding, dilation, and ceil mode.
- **`Rng`** struct: CPU-side RNG (SmallRng/Xoshiro256++) with seed, shuffle, bernoulli, range, normal.
- **`manual_seed(u64)`** / **`cuda_manual_seed_all(u64)`**: Seed libtorch RNGs for reproducibility.
- **`cuda_active_bytes()`**: Query bytes backing live tensors (matches `torch.cuda.memory_allocated()`).

### Fixed
- **VRAM monitoring**: `cuda_allocated_bytes()` now returns `reserved_bytes` from the allocator, making spill detection work.
- Removed unused `ResourceSample::vram_used_bytes` field.
- Dashboard uses `vram_alloc` as the sole VRAM metric.

### Improved
- **Benchmark suite**: Interleaved multi-round execution (`--rounds N`), GPU clock locking (`--lock-clocks FREQ`), configurable warmup (`--warmup-secs`). Peak VRAM tracking (not snapshots). Median interpolation for even-length arrays. `requires_grad=false` on benchmark inputs (matching Python). cuDNN benchmark disabled for GRU (eliminates autotuning variance). `make bench-publish` for publication-grade runs.
- **Docker**: `.dockerignore`, BuildKit cache for libtorch downloads, skip-if-exists image targets.

## [0.1.2] - 2026-03-19

### Added
- **VRAM spill detection**: New FFI function `flodl_cuda_alloc_bytes` queries libtorch's CUDA caching allocator. `cuda_allocated_bytes()` / `cuda_allocated_bytes_idx()` expose it in Rust. When allocated bytes exceed physical VRAM, the monitor shows spill in terminal output, live dashboard, CSV export, and epoch log.
- `ResourceSample::vram_allocated_bytes` field for allocator-level memory tracking.
- `vram_spill` column in CSV export.

### Fixed
- README links now use absolute GitHub URLs — fixes broken links on crates.io where relative paths don't resolve.

## [0.1.0] - 2026-03-18

### Added
- **Graph identity**: `Graph::structural_hash()` — deterministic SHA-256 hash of graph topology, module names, and parameter/buffer shapes. Any architecture change produces a different hash. `Graph::short_hash()` returns the first 8 chars. `FlowBuilder::label()` sets a human-readable name (does not affect hash).
- **Checkpoint architecture validation**: Checkpoint format v1 embeds a 32-byte structural hash. `load_checkpoint` / `load_checkpoint_file` accept an optional hash and error on architecture mismatch.
- **Dashboard metadata**: `Monitor::set_metadata(serde_json::Value)` attaches hyperparameters/config to the HTML archive. `watch()` / `watch_profiled()` capture graph label and hash. Dashboard header shows `"floDl — {label} [{hash8}]"`.
- **Parameter freezing**: `Parameter::freeze()`, `unfreeze()`, `is_frozen()` — disable/enable gradient tracking per parameter. Optimizers automatically skip frozen params (no grad). `Parameter::to_device()` now preserves frozen state.
- **Named checkpoints**: `Graph::named_parameters()` and `named_buffers()` return qualified names (`"tag/weight"` or `"node_id/running_mean"`). `save_checkpoint` / `load_checkpoint` persist both parameters and buffers (e.g., BatchNorm running stats), matching by name for partial loading. `LoadReport` reports what was loaded, skipped, and missing.
- **Optimizer parameter groups**: `Adam::with_groups()`, `SGD::with_groups()`, `AdamW::with_groups()` — builder API for per-group learning rates. `Optimizer::set_group_lr()` adjusts a single group; `set_lr()` updates all groups. Groups are persisted through `Stateful` save/load.

### Core Stack
- **Tensor**: Owned RAII tensors with Drop, ~72 operations. CPU and CUDA (feature-gated).
- **Autograd**: Reverse-mode AD backed by libtorch's native autograd engine. 37 differentiable operations with numerical gradient verification.
- **NN Modules**: Linear, Conv2d, ConvTranspose2d, LayerNorm, BatchNorm, Dropout, Embedding, GRUCell, LSTMCell.
- **Activations**: ReLU, Sigmoid, Tanh, GELU, SiLU.
- **Losses**: mse_loss, cross_entropy_loss, bce_with_logits_loss, l1_loss, smooth_l1_loss, kl_div_loss.
- **Optimizers**: SGD (with momentum), Adam, AdamW.

### Graph Builder
- Fluent API: from/through/build, split/merge, also (residual), tag/using (named refs).
- Loop constructs: for_n (fixed), while_cond (pre-condition), until_cond (post-condition).
- Routing: gate (soft, weighted), switch (hard, selected branch only).
- Map constructs: each, over, slices, with batched fast path.
- Input (auxiliary graph inputs), tag_group (auto-suffixed parallel branch names).

### Training Tools
- LR scheduling: StepDecay, CosineScheduler, WarmupScheduler (composable), PlateauScheduler.
- Mixed precision: Float16/BFloat16 dtype casting, GradScaler for loss scaling.
- Gradient clipping: clip_grad_norm, clip_grad_value.
- Checkpointing: save_checkpoint/load_checkpoint (named binary format with LoadReport, persists parameters + buffers, structural hash validation, file or io::Write).
- Weight initialization: kaiming_uniform/normal, xavier_uniform/normal.

### Training Monitor
- Human-readable ETA with adaptive formatting (hours/minutes/seconds/milliseconds).
- System resource tracking: CPU, RAM, GPU utilization (NVML), VRAM usage.
- Live web dashboard via embedded HTTP server with Server-Sent Events.
- Dashboard features: real-time training curves, resource usage charts, epoch log, graph SVG, label/hash header, metadata card.
- CSV and log file export.

### Observation & Visualization
- Tag-based metric collection: collect/flush/trend.
- Trend analysis: slope, stalled, improving, converged.
- Group trends with tag_group expansion.
- DOT/SVG graph visualization with parameter counts and node type shapes.
- Profiling: enable_profiling, profile, timing trends.
- Training curves: plot_html, export_trends, write_log.

### Infrastructure
- **CI**: GitHub Actions with CPU test matrix and CUDA build verification.
- **Docker**: CPU and CUDA Dockerfiles, docker-compose with GPU support.
- **Build**: Makefile with cpu/cuda targets (build, test, clippy, shell).

### Testing
- 368 library tests + showcase tests.
- Zero clippy warnings.
- Autograd numerical gradient checks.
- Module-level gradient checks.

### Key Design Decisions
- **Deterministic VRAM**: Rust's Drop trait replaces 5 phases of GC-based memory management.
- **No GC overhead**: No runtime.KeepAlive, no pending-free queues, no VRAM budget heuristics.
- **Variable**: `Rc<RefCell<VariableInner>>` for cheap Clone with interior mutability.
- **Module trait**: single-input forward + optional NamedInputModule for multi-input. `structural_hash()` for architecture identity.
- **Graph-as-Module**: Graph implements Module for hierarchical composition.
- **NamedInputModule on routers**: SoftmaxRouter and SigmoidRouter sum refs into input before projection.
- **Native FFI ops**: flodl_max, flodl_norm, flodl_cuda_mem_info, flodl_cuda_utilization.
