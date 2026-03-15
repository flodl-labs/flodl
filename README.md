<p align="center">
  <img src="docs/floDl.png" alt="floDl" width="640">
</p>

<h1 align="center">floDl</h1>

<p align="center">
A Rust-native deep learning framework built on libtorch.<br>
Same GPU kernels as PyTorch. No Python. No GIL. No GC. Just Rust.
</p>

<p align="center">
  <a href="https://github.com/fab2s/floDl/actions"><img src="https://github.com/fab2s/floDl/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/flodl"><img src="https://img.shields.io/crates/v/flodl.svg" alt="crates.io"></a>
  <a href="https://docs.rs/flodl"><img src="https://docs.rs/flodl/badge.svg" alt="docs.rs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#getting-started">Getting Started</a> &bull;
  <a href="#the-graph-builder">Graph Builder</a> &bull;
  <a href="#training-monitor">Training Monitor</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="docs/tutorials/01-tensors.md">Tutorials</a> &bull;
  <a href="docs/pytorch_migration.md">PyTorch Migration</a> &bull;
  <a href="docs/troubleshooting.md">Troubleshooting</a> &bull;
  <a href="#architecture">Architecture</a>
</p>

---

## Getting Started

Create a new project with one command:

```bash
curl -sL https://raw.githubusercontent.com/fab2s/floDl/main/init.sh | sh -s my-project
cd my-project
make build    # first build (~5 min, downloads libtorch)
make run      # train the template model
```

This generates a complete project with Dockerfiles, Makefile, and an annotated
training template. Edit `src/main.rs` to build your model.

> **New to Rust?** Read [Rust for PyTorch Users](docs/tutorials/00-rust-primer.md) — 10 patterns in 15 minutes.

## The Graph Builder

floDl's fluent graph builder lets you describe complex architectures as
readable data flow — no boilerplate, no graph construction commands.

```rust
let model = FlowBuilder::from(Linear::new(2, 16)?)
    .through(GELU)                        // activation
    .through(LayerNorm::new(16)?)         // normalization
    .also(Linear::new(16, 16)?)           // residual connection
    .through(Linear::new(16, 2)?)         // output projection
    .build()?;
```

That's a trainable model. `also` adds the residual — input flows through the
Linear *and* gets added to its output. `build()` returns a `Graph` that
implements `Module` — you can nest it inside other graphs.

Things get interesting when architectures get complex:

```rust
let g = FlowBuilder::from(encoder).tag("encoded")
    .split(modules![head_a, head_b, head_c]).merge(MergeOp::Mean)
    .loop_body(refinement_block).for_n(3).tag("refined")
    .gate(router, modules![expert_a, expert_b]).using(&["encoded"])
    .switch(selector, modules![light_path, heavy_path]).using(&["refined"])
    .through(StateAdd).using(&["memory"]).tag("memory")
    .loop_body(decoder).while_cond(halt_condition, 10)
    .through(output_head)
    .build()?;
```

Every construct — `split/merge`, `also`, `loop_body`, `gate`, `switch`, `map`,
`tag/using` — composes cleanly. Sub-graphs nest like any module. Forward
references (`using` before `tag`) carry state across calls, enabling recurrent
architectures without special-casing.

See the **[Graph Builder Tutorial](docs/tutorials/05-graph-builder.md)** and
the [full showcase](flodl/examples/showcase/) that exercises every builder
method.

## Training Monitor

Drop-in training monitor with adaptive ETA, system resource tracking, and a
live web dashboard — no external dependencies, no separate process.

```rust
use flodl::monitor::Monitor;

let mut monitor = Monitor::new(num_epochs);
monitor.serve(3000)?;  // optional: live dashboard at http://localhost:3000

for epoch in 0..num_epochs {
    let t = std::time::Instant::now();
    // ... training ...

    monitor.log(epoch, t.elapsed(), &[("loss", loss_val), ("lr", lr)]);
}
monitor.finish();
```

Terminal output adapts automatically — duration and ETA switch between hours,
minutes, seconds, and milliseconds as needed:

```
  epoch   1/100  loss=1.5264  [49ms  ETA 4.8s]
  epoch  10/100  loss=0.3817  [25ms  ETA 2.2s]  VRAM: 2.1/6.0 GB (82%)
  epoch  50/100  loss=0.0023  [24ms  ETA 1.2s]  VRAM: 2.1/6.0 GB (82%)
  epoch 100/100  loss=0.0012  [23ms]             VRAM: 2.1/6.0 GB (82%)
  training complete in 2.8s  | loss: 0.0012
```

### Live dashboard

Call `monitor.serve(port)` and open the URL in a browser. The page updates
in real time via Server-Sent Events — no polling, no WebSocket, no npm.

<p align="center">
  <img src="docs/dashboard.gif" alt="floDl live training dashboard" width="800">
</p>

The dashboard includes:

| Panel | What it shows |
|-------|--------------|
| **Header** | Epoch counter, progress bar, ETA, elapsed time |
| **Metrics chart** | All logged metrics (loss, lr, ...) as live canvas chart |
| **Resource chart** | CPU%, GPU%, RAM%, VRAM% over time |
| **Resource bars** | Current usage with values (e.g., `VRAM: 2.1/6.0 GB`) |
| **Epoch log** | Every epoch, newest first, with duration and resources |
| **Graph SVG** | Collapsible architecture diagram (via `monitor.watch(&model)`) |

Late join works — open the dashboard mid-training and it backfills all
past epochs instantly.

### Resource tracking

| Metric | Source | Availability |
|--------|--------|-------------|
| CPU % | `/proc/stat` delta | Linux |
| RAM | `/proc/meminfo` | Linux |
| GPU utilization % | NVML (dynamic `dlopen`) | NVIDIA GPU + driver |
| VRAM used/total | `cudaMemGetInfo` via FFI | CUDA builds |

Resources that aren't available are silently omitted. CPU-only builds show
CPU and RAM; CUDA builds add GPU and VRAM automatically.

### Export

```rust
monitor.save_html("training_report.html");  // self-contained dashboard archive
monitor.write_log("training.log")?;          // human-readable log
monitor.export_csv("training.csv")?;         // metrics + resources as CSV
```

`save_html` writes a complete dashboard at `finish()` — all metrics, resource
charts, and graph SVG baked into a single HTML file. Open it in any browser,
no server needed. Set it once before training and forget about it.

See the full **[Training Monitor Tutorial](docs/tutorials/09-monitor.md)**.

## Quick Start

Requirements: Docker (with NVIDIA Container Toolkit for GPU support).

**New project** (see [Getting Started](#getting-started) above):
```bash
curl -sL https://raw.githubusercontent.com/fab2s/floDl/main/init.sh | sh -s my-project
cd my-project && make run
```

**Develop floDl itself:**
```bash
git clone https://github.com/fab2s/floDl.git
cd floDl
make image      # build dev container (Rust + libtorch)
make test       # run all tests (CPU)
make cuda-test  # run all tests on CUDA (requires NVIDIA GPU)
make test-all   # CPU first, then CUDA if a GPU is available
make clippy     # lint
make shell      # interactive shell in container
```

### Train a model in 30 lines

```rust
use flodl::*;

// Build the model.
let model = FlowBuilder::from(Linear::new(2, 16)?)
    .through(GELU)
    .through(LayerNorm::new(16)?)
    .also(Linear::new(16, 16)?)
    .through(Linear::new(16, 2)?)
    .build()?;

// Set up training.
let params = model.parameters();
let mut optimizer = Adam::new(&params, 0.01);
model.set_training(true);

// Training loop.
for (input_t, target_t) in &batches {
    let input = Variable::new(input_t.clone(), true);
    let target = Variable::new(target_t.clone(), false);

    let pred = model.forward(&input)?;
    let loss = mse_loss(&pred, &target)?;

    optimizer.zero_grad();
    loss.backward()?;
    clip_grad_norm(&params, 1.0)?;
    optimizer.step()?;
}
```

## Features

### Core Stack

| Layer | What it does |
|-------|-------------|
| **Tensor** | Owned RAII tensors with `Drop`, `Clone`. CPU and CUDA. |
| **Autograd** | Reverse-mode automatic differentiation. Full backward for every op. |
| **NN Modules** | `Linear`, `Conv2d`, `ConvTranspose2d`, `LayerNorm`, `BatchNorm`/`BatchNorm2d`, `Dropout`, `Embedding`, `GRUCell`, `LSTMCell` |
| **Activations** | `Identity`, `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU` |
| **Losses** | `mse_loss`, `cross_entropy_loss`, `bce_with_logits_loss`, `l1_loss`, `smooth_l1_loss`, `kl_div_loss` |
| **Optimizers** | `SGD` (with momentum), `Adam`, `AdamW` — all support parameter groups for per-group LR |
| **LR Scheduling** | `StepDecay`, `CosineScheduler`, `WarmupScheduler` (composable), `PlateauScheduler` |
| **Mixed Precision** | `Float16`/`BFloat16` dtype casting, `GradScaler` for loss scaling |
| **Monitor** | Human-readable ETA, CPU/GPU/RAM/VRAM tracking, live web dashboard |

### Graph Builder

| Method | What it does |
|--------|-------------|
| `from(m).through(m)` | Linear chain |
| `fork(m)` | Side branch: runs module, captures output as tag, stream continues unchanged |
| `input(names)` | Auxiliary graph inputs, accessible via `using(name)` — multi-input graphs |
| `split(modules![...]).merge(op)` | Parallel branches, merged by `Add` or `Mean` |
| `also(m)` | Residual connection: `input + m(input)` |
| `tag(name)` / `using(refs)` | Named references — backward (same pass) or forward (across calls) |
| `loop_body(body).for_n(n)` | Fixed iteration with BPTT |
| `loop_body(body).while_cond(cond, max)` | Condition before body (0..max iterations) |
| `loop_body(body).until_cond(cond, max)` | Condition after body (1..max iterations) |
| `gate(router, modules![...])` | Soft routing — all experts execute, weighted combination |
| `switch(selector, modules![...])` | Hard routing — only selected branch executes |
| `map(body).each()` | Apply body to each element along dim 0 |
| `map(body).over(tag)` | Iterate over a tagged tensor |
| `map(body).slices(n)` | Decompose last dim into n slices, map, recompose |
| `.batched()` | Fast path for Map — full batch in one call |
| `tag_group(name)` | Name parallel branches: `split(...).tag_group("head")` |

### Training Tools

| Tool | What it does |
|------|-------------|
| `clip_grad_norm` | L2 norm gradient clipping |
| `clip_grad_value` | Element-wise gradient clamping |
| `save_checkpoint` / `load_checkpoint` | Named `.fdl` checkpoint with partial loading, persists parameters + buffers, structural hash validation, `LoadReport` (file path or `Write`/`Read`) |
| `Parameter::freeze` / `unfreeze` | Disable/enable gradient tracking per parameter |
| `xavier_uniform/normal` | Weight initialization (also `kaiming_*` via `nn::init`) |
| LR schedulers | `StepDecay`, `CosineScheduler`, `WarmupScheduler`, `PlateauScheduler` (composable) |
| `GradScaler` | Dynamic loss scaling for mixed precision (float16) training |
| `cast_parameters` | Cast model parameters to any dtype |

### Module Traits

Beyond the core `forward`/`parameters` methods, `Module` provides optional
methods that the graph recognizes automatically:

| Method | Default | What happens |
|--------|---------|-------------|
| `as_named_input()` | `None` | Returns `&dyn NamedInputModule` — loop and node `using()` refs arrive as a named map |
| `reset()` | no-op | Loops auto-call before iterating — clears per-forward state |
| `detach_state()` | no-op | `graph.detach_state()` propagates — breaks gradient chains on retained state |

Stateful modules just override `reset()` and/or `detach_state()` directly —
no separate trait impls needed. Modules that own child modules implement
`sub_modules()` for recursive device placement, training mode, and parameter
collection.

### Observation & Trends

Tags double as observation points — collect metrics during training, flush
to epoch history, and query trends to drive training decisions:

```rust
for epoch in 0..num_epochs {
    for (input, target) in &batches {
        let pred = graph.forward(&input)?;
        graph.collect(&["hidden"])?;                 // from graph tag

        let loss = mse_loss(&pred, &target)?;
        graph.record_scalar("loss", loss.item()?);   // external metric
    }
    graph.flush(&["hidden", "loss"]);

    if graph.trend("loss").stalled(5, 1e-4) {
        // decay learning rate
    }
}
```

| Method | What it does |
|--------|-------------|
| `g.tagged(tag)` | Access a tagged node's output after forward |
| `g.collect(tags)` / `g.flush(tags)` | Batch -> epoch metric collection |
| `g.record_scalar(tag, value)` | Inject external metrics |
| `g.trend(tag)` | Epoch-level trend: `slope`, `stalled`, `improving`, `converged` |
| `g.trends(tags)` | Group trends: `all_improving`, `any_stalled`, `mean_slope` |
| `g.end_step()` / `g.end_epoch()` | Training housekeeping |

### Visualization

```rust
println!("{}", g.dot());                       // Graphviz DOT with parameter counts
let svg = g.svg(Some("model.svg"))?;          // render to SVG

// Timing-annotated: nodes colored green->yellow->red by execution time.
g.enable_profiling();
g.forward(&input)?;
g.svg_with_profile(Some("profile.svg"))?;

// Training curves as self-contained HTML.
g.plot_html("training.html", &["loss", "head"])?;
g.export_trends("metrics.csv", &["loss"])?;
```

### Numerical Verification

Every differentiable path is verified against finite-difference gradients:
- 37 autograd op-level checks (every op + compositions)
- Module-level checks (every NN module, input + parameter gradients)
- Exact optimizer step verifications (SGD, Adam, AdamW)
- 311 library tests, zero clippy warnings — all tests run on both CPU and CUDA

## Why Rust for Deep Learning?

### The memory management problem

Python adds ~3-5 us of framework overhead to every GPU operation. For
architectures built on many small sequential operations — recurrent steps,
iterative refinement, multi-head attention — this overhead dominates.

Go solves the dispatch overhead with compiled binaries and goroutines, but
Go's garbage collector cannot manage VRAM deterministically. GPU memory lives
in libtorch's C++ allocator — invisible to Go's GC. This required goDl to
build a 5-phase memory management system: atomic refcounting, saved-tensor
lifecycle, GC callbacks, VRAM budgets, and autograd Scope. Hundreds of lines
of `runtime.KeepAlive`, `Retain()`/`Release()`, and pending-free queues.

Rust's ownership model eliminates all of this. `Tensor` owns a C++ handle.
`Drop` frees it immediately when it goes out of scope. No GC, no finalizers,
no reference counting, no VRAM budget heuristics, no KeepAlive. The entire
goDl memory management system — Phases 1 through 5 — is replaced by a single
`impl Drop for Tensor`.

### Zero-cost safety

Rust's type system catches errors at compile time that other languages defer
to runtime:

- **Ownership**: tensors are freed exactly once, exactly when no longer needed
- **Result types**: every fallible operation returns `Result<T>` — no silent
  error propagation, no nil pointer panics
- **No data races**: the borrow checker prevents concurrent mutation bugs

### Same GPU kernels

floDl binds libtorch — the same C++ library that powers PyTorch. The actual
GPU math (CUDA kernels, cuBLAS, cuDNN) is identical. floDl replaces everything
above: the dispatch path, autograd tracking, module composition, and graph
execution.

## Performance

Add this to your project's `Cargo.toml` to get optimized floDl with fast
recompilation of your own code:

```toml
# Optimize floDl in dev builds — your code stays fast to compile.
# After the first build, only your graph code recompiles.
[profile.dev.package.flodl]
opt-level = 3

[profile.dev.package.flodl-sys]
opt-level = 3

# Release: cross-crate optimization for maximum throughput.
[profile.release]
lto = "thin"
codegen-units = 1
```

| Profile | flodl | Your code | Typical rebuild |
|---------|-------|-----------|-----------------|
| `cargo build` | `-O3` (cached) | `-O0` (fast) | < 2s |
| `cargo build --release` | `-O3` + LTO | `-O3` + LTO | full link |

The GPU kernels (cuBLAS, cuDNN) run at the same speed regardless of Rust
optimization level — the profile settings affect graph dispatch, autograd
bookkeeping, and module overhead.

## Hardware Compatibility

floDl is developed and tested on an NVIDIA GTX 1060 (6 GB VRAM, Pascal
architecture). It works out of the box — no version pinning, no feature
flags, no workarounds.

This matters because PyTorch dropped Pascal support after version 2.5.1.
Training on older GPUs now requires pinning `torch==2.5.1` and hoping
nothing in your dependency tree pulls a newer version. floDl sidesteps
this entirely: it links against libtorch's stable C API, which continues
to support every CUDA architecture that the driver supports.

If your GPU runs `nvidia-smi`, floDl can train on it.

## Architecture

```
+-----------------------------------------------------------+
|  User Code / Model Definitions                            |
+-----------------------------------------------------------+
|  monitor/  ETA, resource tracking, live web dashboard     |
+-----------------------------------------------------------+
|  graph/    Fluent builder, execution, DOT/SVG             |
+-----------------------------------------------------------+
|  nn/       Modules, losses, optimizers, checkpoints       |
+-----------------------------------------------------------+
|  autograd/ Reverse-mode AD, gradient tracking             |
+-----------------------------------------------------------+
|  tensor/   Owned tensors with Drop, CPU + CUDA            |
+-----------------------------------------------------------+
|  flodl-sys   FFI bindings to libtorch C++ shim            |
+-----------------------------------------------------------+
|  libtorch / CUDA / ROCm / MPS / CPU                      |
+-----------------------------------------------------------+
```

Since floDl binds libtorch — not CUDA directly — it inherits libtorch's
backend support: NVIDIA (CUDA), AMD (ROCm), Intel (XPU), Apple Silicon (MPS),
and CPU. Switching hardware is a build flag, not a code change.

## Documentation

### Tutorials

Step-by-step guides from basics to advanced, each with code examples:

0. **[Rust for PyTorch Users](docs/tutorials/00-rust-primer.md)** — 10 Rust patterns in 15 minutes (new to Rust? start here)
1. **[Tensors](docs/tutorials/01-tensors.md)** — creation, ops, error handling, memory
2. **[Autograd](docs/tutorials/02-autograd.md)** — variables, gradients, backward pass
3. **[Modules](docs/tutorials/03-modules.md)** — Linear, Conv2d, normalization, RNN cells
4. **[Training](docs/tutorials/04-training.md)** — losses, optimizers, full training loop
5. **[Graph Builder](docs/tutorials/05-graph-builder.md)** — the fluent API from simple to complex
6. **[Advanced Graphs](docs/tutorials/06-advanced-graphs.md)** — forward refs, loops, gates, switches
7. **[Visualization](docs/tutorials/07-visualization.md)** — DOT/SVG output, reading diagrams
8. **[Utilities](docs/tutorials/08-utilities.md)** — checkpoints, clipping, freezing, initialization
9. **[Training Monitor](docs/tutorials/09-monitor.md)** — ETA, resource tracking, live web dashboard

### Design

- [Roadmap](docs/design/roadmap.md) — development plan and port status
- [Trajectory Thesis](docs/design/trajectory-thesis.md) — geometric intuition behind the project

### Examples

- [`quickstart`](flodl/examples/quickstart/) — train a model in 30 lines
- [`sine_wave`](flodl/examples/sine_wave/) — sine regression with monitor, checkpoint round-trip
- [`showcase`](flodl/examples/showcase/) — every graph builder method in one graph

## Lineage

floDl is a Rust port of [goDl](https://github.com/fab2s/goDl), a Go-native
DL framework. The port was motivated by Go's inability to manage VRAM
deterministically — Rust's ownership model solves this at the language level.
The graph builder API, module architecture, and design philosophy carry over
directly.

## License

floDl is open-sourced software licensed under the [MIT license](./LICENSE).
