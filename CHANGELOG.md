# Changelog

All notable changes to floDl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.1] - 2026-03-29

### Added

#### PyTorch Parity — Tensor Operations
- **Math ops**: `log1p`, `expm1`, `log2`, `log10`, `tan`, `asin`, `acos`, `atan`, `erf`, `erfc`, `trunc`, `frac`, `fmod`, `fmod_tensor`, `remainder`, `remainder_tensor`, `lerp`, `lerp_tensor`, `isclose`, `addmm`, `addcmul`, `addcdiv`, `clamp_min`, `clamp_max`, `selu`, `hardswish`, `hardsigmoid`, `prelu`
- **Reductions**: `prod`, `prod_dim`, `cumsum`, `logsumexp`
- **Shape ops**: `flip`, `roll`, `diagonal`, `movedim`, `tile`, `split`, `unbind`, `contiguous`, `cat_many`, `unsqueeze_many`, `narrow_scatter`, `pad_mode` (constant/reflect/replicate/circular), `meshgrid`
- **NN tensor ops**: `conv1d`, `conv_transpose1d`, `conv3d`, `conv_transpose3d`, `avg_pool2d`, `avg_pool1d`, `max_pool1d`, `adaptive_max_pool2d`, `instance_norm`, `group_norm`, `linear` (fused), `pixel_shuffle`, `pixel_unshuffle`, `bilinear`, `embedding_bag`, `interpolate` (nearest/bilinear/bicubic/trilinear), `im2col`, `col2im`, `bce_loss`, `nll_loss`, `ctc_loss`
- **Comparison/similarity**: `maximum`, `minimum`, `atan2`, `masked_fill`, `normalize`, `cosine_similarity`

#### PyTorch Parity — Autograd
- **New differentiable ops**: `leaky_relu`, `elu`, `softplus`, `mish`, `selu`, `hardswish`, `hardsigmoid`, `prelu`, `clamp_min`, `clamp_max`, `log1p`, `expm1`, `log2`, `log10`, `atan2`, `maximum`, `minimum`, `masked_fill`, `normalize`, `cosine_similarity`, `prod`, `prod_dim`, `cumsum`, `logsumexp`, `unsqueeze_many`, `cat_many`, `stack`, `triu`, `tril`
- **NN autograd ops**: `conv1d`, `conv_transpose1d`, `conv3d`, `conv_transpose3d`, `avg_pool2d`, `avg_pool1d`, `max_pool1d`, `adaptive_max_pool2d`, `instance_norm`, `group_norm`, `pixel_shuffle`, `pixel_unshuffle`, `bilinear`, `embedding_bag`, `im2col`, `col2im`

#### PyTorch Parity — Modules
- **Convolutions**: `Conv1d` (with `Conv1dBuilder`), `Conv3d` (with `Conv3dBuilder`), `ConvTranspose1d`, `ConvTranspose3d`
- **Recurrent**: `GRU` (multi-layer sequence module), `LSTM` (multi-layer sequence module) — match `nn.GRU`/`nn.LSTM` interface with `forward_seq`, batch-first support
- **Normalization**: `GroupNorm`, `InstanceNorm`, `RMSNorm`
- **Pooling**: `AvgPool2d`, `MaxPool1d`, `AvgPool1d`, `AdaptiveMaxPool2d`, `PixelShuffle`, `PixelUnshuffle`, `Upsample`, `Unfold`, `Fold`
- **Attention**: `MultiheadAttention` — self-attention and cross-attention with optional masking
- **Bilinear**: `Bilinear` — bilinear transformation `y = x1^T A x2 + b`
- **Activations**: `LeakyReLU`, `ELU`, `Softplus`, `Mish`, `SELU`, `Hardswish`, `Hardsigmoid`, `PReLU` (learnable), `Softmax`, `LogSoftmax`, `Flatten`
- **Dropout**: `AlphaDropout` — maintains self-normalizing property for SELU networks
- **Embedding**: `EmbeddingBag` — bag-of-embeddings with sum/mean/max aggregation
- **Padding**: `ZeroPad2d`, `ReflectionPad2d` — symmetric and asymmetric padding modules

#### PyTorch Parity — Losses
- `bce_loss` (from probabilities), `nll_loss`, `ctc_loss`, `focal_loss` (class imbalance), `triplet_margin_loss`, `cosine_embedding_loss`, `hinge_embedding_loss`, `margin_ranking_loss`, `poisson_nll_loss`

#### PyTorch Parity — Optimizers
- `RMSprop` (with `RMSpropBuilder` for parameter groups)
- `Adagrad` (with `AdagradBuilder` for parameter groups)
- `RAdam` — Rectified Adam with variance-aware warmup
- `NAdam` — Nesterov-accelerated Adam

#### PyTorch Parity — LR Schedulers
- `ExponentialLR` — exponential decay (`lr = base_lr * gamma^step`)
- `MultiStepLR` — decay at specific milestones
- `OneCycleLR` — super-convergence schedule (warmup + cosine decay)
- `CyclicLR` — triangular wave between base and max LR (symmetric and asymmetric)

#### PyTorch Parity — Initialization
- `kaiming_uniform`, `kaiming_normal` now re-exported at crate root
- New: `uniform`, `normal`, `orthogonal`, `trunc_normal`, `uniform_bias`

#### Test Coverage (+165 tests, 769 total)
- **Autograd gradient verification** (55 tests): finite-difference checks for every new differentiable op — `leaky_relu`, `elu`, `softplus`, `mish`, `selu`, `hardswish`, `hardsigmoid`, `prelu`, `clamp_min`/`clamp_max`, `log1p`, `expm1`, `log2`, `log10`, `maximum`, `minimum`, `masked_fill`, `cosine_similarity`, `normalize`, `prod`, `cumsum`, `logsumexp`, `tril`, `flatten`; fused NN op gradients for all conv variants (1d/2d/3d + transpose), all pooling variants, `layer_norm`, `group_norm`, `instance_norm`, `bilinear`, `embedding_bag`, `pixel_shuffle`/`unshuffle`, `im2col`/`col2im`, `grid_sample`, `gru_cell`, `lstm_cell`; Variable API coverage (`set_grad`, `set_requires_grad`, `is_leaf`, `numel`, `zero_grad_set_to_none`, `set_data`, `to_device`)
- **Module forward/backward** (60+ tests): Conv1d (builder, groups, stride/padding, no-bias, gradient), Conv2d (builder, grouped, stride, no-bias, gradient), Conv3d, ConvTranspose1d/2d/3d (forward, gradient, stride, parameters), GroupNorm (batch-size-one, single-group, groups=channels, gradient), InstanceNorm (3D input, affine parameters, gradient), LayerNorm (3D, normalization, gradient), BatchNorm/BatchNorm2d (training, eval, running stats, rejects invalid dims, gradient), Bilinear (gradient, no-bias, rejects single input), Dropout (training, eval identity, p=0), ZeroPad2d/ReflectionPad2d (asymmetric, values, no-parameters)
- **Loss functions** (20+ tests): MSE (basic, zero loss), cross-entropy (class indices, wrong predictions, gradient), BCE/BCEWithLogits (gradient), L1, SmoothL1 (negative beta rejection), KLDiv, CTC, focal (reduces to CE at gamma=0), triplet margin (zero when far), cosine embedding (similar/dissimilar), hinge embedding (positive/negative), margin ranking (with margin), Poisson NLL (log/no-log)
- **Mixed precision** (7 tests): AutocastGuard lifecycle, autocast closure, GradScaler (defaults, scale, step finite/inf, update growth/backoff, state roundtrip), cast_parameters (basic, noop same dtype)
- **Gradient clipping** (6 tests): clip_grad_norm (scales down, no-op when small, multiple params), clip_grad_value (clamps, no-op, no-grad params)
- **Graph observation** (8 tests): collect/flush/trend pipeline, reduction modes (mean, sum, min, max, norm, scalar passthrough), rejects non-scalar, map operations (over tag, slices, batched, gradient, error cases)

## [0.2.0] - 2026-03-29

### Added

#### Graph Tree (hierarchical composition)
- **Label-path addressing**: Dot-separated paths (`"encoder.scan.hidden"`) for addressing subgraphs and tags across graph boundaries. Strict dot semantics -- dots always mean subgraph boundaries, no fuzzy resolution.
- **Tree registration**: Labeled graphs nested via `FlowBuilder` are automatically detected as child subgraphs. `tree_children()`, `child_graph()`, `subgraph()` for navigation. `is_composed()` flag on child graphs.
- **Selective freeze/thaw**: `freeze("encoder.read")`, `thaw("encoder.scan")`, `is_frozen("encoder")` -- declarative training phase control by label path.
- **Path-based parameter collection**: `parameters_at()`, `named_parameters_at()`, `named_buffers_at()` for per-subgraph optimizer groups. Target namespace used for checkpoint compatibility.
- **Subgraph checkpoint loading**: `load_subgraph_checkpoint("encoder", "encoder_v1.fdl.gz")` -- loads a checkpoint into a specific subgraph using the child's own namespace and structural hash validation.
- **Cross-boundary observation**: `tagged_at()` (null/nil semantics), `collect_at()`, `record_at()`, `trend_at()` -- read tagged outputs and metrics across graph boundaries.
- **Tree-aware flush and metrics**: `flush()` automatically recurses into labeled child subgraphs. `latest_metrics()` collects from the entire tree with dotted prefixes (`"encoder.loss"`). `Monitor::log()` sees the whole tree with zero extra code. `flush_local()` and `latest_metrics_local()` for independent per-subgraph observation cadences.
- **Internal tags**: Tags prefixed with `_` are auto-internal (hidden from parent resolution). Explicit `.internal("tag")` on FlowBuilder. Cross-boundary resolution rejects internal tags.
- **Training mode propagation**: `set_training_at("encoder", false)` for selective eval mode on subgraphs (BatchNorm running stats).
- **Verbose build output**: `.verbose(true)` on FlowBuilder prints tree structure, tag resolution, and parameter summary. `tree_summary()`, `param_summary()` methods.
- **Path validation**: `validate_path()` returns `PathKind::Subgraph` or `PathKind::Tag` for build-time wiring checks.
- **Module trait**: Added `as_graph()` method (default `None`, overridden in Graph) for subgraph detection.
- **Zero forward-path impact**: All tree metadata is build-time/query-time only. The pre-computed Vec routing in `forward_impl()` is untouched.

#### Modules
- **`GaussianBlur`**: Stateless `Module` wrapper around `gaussian_blur_2d()` for use in `FlowBuilder` graphs. Fixed sigma, no parameters. Kernel size auto-computed from sigma (`2 * ceil(3 * sigma) + 1`).

#### Checkpoint Migration
- **`migrate_checkpoint()`** / **`migrate_checkpoint_file()`**: Automatically remap parameter names from an older checkpoint to match a model's current naming. Matches by exact name first, then by shape+dtype in positional order. Handles params and buffers, supports `.gz` compression. Returns a `MigrateReport` with `unchanged`, `remapped`, `dropped`, `missing` fields and a `Display` impl for human-readable output.
- **`checkpoint_version()`**: Peek at a checkpoint file's version without loading it. Returns `1` for flodl 0.1.x, `2` for 0.2.0+.
- **`MigrateReport`**: Full accounting of a migration — `is_complete()` returns true when nothing was dropped or missing.

### Changed
- **Breaking**: Checkpoint format version bumped to v2. Checkpoints saved with 0.2.0+ write version 2; `load_checkpoint` accepts both v1 and v2 (binary layout is identical, only naming conventions differ). v1 checkpoints can be migrated with `migrate_checkpoint_file()`.
- **Breaking**: Restructuring a graph with `.label()` or renaming tags changes the parameter names that feed into `structural_hash()` — the hash algorithm is unchanged, but its inputs differ. Checkpoints saved before restructuring will fail architecture validation on load. Use `migrate_checkpoint_file()` to remap parameter names, or retrain.

## [0.1.5] - 2026-03-25

### Added
- `make docs-rs` — local docs.rs build validation via disposable Docker container (nightly Rust, `--cfg docsrs`, no libtorch). Catches docs.rs failures before publishing.

### Fixed
- Fix docs.rs build: `rand` 0.9.2 uses `feature(doc_auto_cfg)` removed in nightly 1.92+. Made `rand` an optional dependency (`rng` feature, on by default) so docs.rs can build without it.
- Fix flaky `test_clip_grad_norm` — seed RNG for deterministic weights.
- Fix rustdoc broken intra-doc links in `Tensor` (escaped shape brackets, qualified method paths).

## [0.1.4] - 2026-03-25

### Fixed
- Disable example scraping on docs.rs — examples require libtorch which the docs.rs sandbox doesn't have. The scraping failure corrupted dependency artifacts, breaking the doc build.

## [0.1.3] - 2026-03-25

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

### Changed
- **Benchmark suite**: Publication-grade methodology with interleaved multi-round execution (`--rounds N`), GPU clock locking (`--lock-clocks FREQ`), configurable warmup (`--warmup-secs`). 7 benchmarks (3 standard + 4 graph-builder). Peak VRAM tracking (not snapshots). WSL2 host-side clock management via `bench-publish.ps1`. `make bench-publish` for reproducible runs.
- **Docker**: `.dockerignore`, BuildKit cache for libtorch downloads, skip-if-exists image targets, dedicated bench image.

## [0.1.2] - 2026-03-19

### Added
- **VRAM spill detection**: New FFI function `flodl_cuda_alloc_bytes` queries libtorch's CUDA caching allocator. `cuda_allocated_bytes()` / `cuda_allocated_bytes_idx()` expose it in Rust. When allocated bytes exceed physical VRAM, the monitor shows spill in terminal output, live dashboard, CSV export, and epoch log.
- `ResourceSample::vram_allocated_bytes` field for allocator-level memory tracking.
- `vram_spill` column in CSV export.

### Fixed
- README links now use absolute GitHub URLs — fixes broken links on crates.io where relative paths don't resolve.

## [0.1.1] - 2026-03-18

### Fixed
- Replace `sha2` with `hmac-sha256` — fixes docs.rs build (sha2's asm feature doesn't compile on docs.rs).
- Widen leak test tolerance for CI parallel test jitter.

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
- 389 library tests + showcase tests.
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
