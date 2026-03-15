# Changelog

All notable changes to floDl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
- 290 library tests + 15 showcase tests.
- Zero clippy warnings.
- Autograd numerical gradient checks.
- Module-level gradient checks.

### Key Design Decisions
- **Deterministic VRAM**: Rust's Drop trait replaces goDl's entire 5-phase memory management.
- **No GC overhead**: No runtime.KeepAlive, no pending-free queues, no VRAM budget heuristics.
- **Variable**: `Rc<RefCell<VariableInner>>` for cheap Clone with interior mutability.
- **Module trait**: single-input forward + optional NamedInputModule for multi-input. `structural_hash()` for architecture identity.
- **Graph-as-Module**: Graph implements Module for hierarchical composition.
- **NamedInputModule on routers**: SoftmaxRouter and SigmoidRouter sum refs into input before projection.
- **Native FFI ops**: flodl_max, flodl_norm, flodl_cuda_mem_info, flodl_cuda_utilization.
