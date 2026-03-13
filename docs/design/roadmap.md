# floDl Roadmap

A Rust-native deep learning framework built on libtorch, ported from
[goDl](https://github.com/fab2s/goDl).

---

## Vision

A complete DL framework in Rust that can express anything Python/PyTorch
can — with deterministic memory management, zero-cost abstractions, and
native support for architectures that Python punishes (recurrent attention,
adaptive computation, hypothesis-test loops).

---

## Guiding Principles

1. **Own every layer.** libtorch is the exception — it bridges to CUDA
   kernels — but bindings, API, autograd, and graph engine are all native
   Rust.

2. **Idiomatic Rust.** Not "PyTorch translated to Rust." Leverage Rust's
   strengths: ownership, traits, Result types, zero-cost abstractions.

3. **Deterministic memory.** Tensor memory freed by Drop — no GC, no
   finalizers, no VRAM budgets. The entire goDl memory management system
   (5 phases) replaced by `impl Drop for Tensor`.

4. **Composable primitives.** Small pieces that do one thing, compose into
   complex flows. A Linear layer is a Module. A graph is a Module. A
   trained model inside another graph is a Module.

5. **Human-readable graphs.** The fluent builder reads as data flow, not
   as graph construction commands.

---

## Architecture Overview

```
+-----------------------------------------------------------+
|  User Code / Model Definitions                            |
+-----------------------------------------------------------+
|  graph/    Fluent builder, execution, observation, viz    |  DONE
+-----------------------------------------------------------+
|  nn/       Modules, losses, optimizers, checkpoints       |  DONE
+-----------------------------------------------------------+
|  autograd/ Reverse-mode AD, gradient tracking             |  DONE
+-----------------------------------------------------------+
|  tensor/   Owned tensors with Drop, CPU + CUDA            |  DONE
+-----------------------------------------------------------+
|  flodl-sys   FFI bindings to libtorch C++ shim            |  DONE
+-----------------------------------------------------------+
|  libtorch / CUDA                                          |  External
+-----------------------------------------------------------+
```

---

## Phase 1: FFI Bindings (flodl-sys) — DONE

C++ shim (`shim.h`, `shim.cpp`) wrapping libtorch as C functions.
`build.rs` compiles via the `cc` crate, links libtorch.

~91 C functions covering:
- Tensor creation, arithmetic, activations, reductions
- Shape manipulation, slicing, joining
- Convolution, pooling, normalization
- Device management, dtype casting
- Grid sampling, layer norm

---

## Phase 2: Tensor API — DONE

`tensor.rs` — safe Rust `Tensor` type with `Drop` for deterministic
C++ handle cleanup.

- ~72 operations including conv, pool, grid_sample, dtype cast
- `Clone` shares libtorch TensorImpl (shallow copy)
- `Result<Tensor>` for all fallible operations
- CPU + CUDA support via feature gate

---

## Phase 3: Autograd Engine — DONE

`autograd/` — Reverse-mode automatic differentiation backed by libtorch.

- `Variable`: `Rc<RefCell<VariableInner>>` — cheap Clone, interior mutability
- 50+ differentiable operations delegating to libtorch's native autograd
- `no_grad` closure and `NoGradGuard` for inference
- `backward()` delegates to libtorch's C++ backward engine
- Gradient accumulation for shared parameters

---

## Phase 4: Layers & Optimizers — DONE

`nn/` — neural network building blocks.

**Module trait**: `forward(&self, &Variable) -> Result<Variable>`
+ `parameters()` + `sub_modules()` + `move_to_device()` + `set_training()`

**Layers**: Linear, Conv2d, ConvTranspose2d, LayerNorm, BatchNorm, Dropout,
Embedding, GRUCell, LSTMCell

**Activations**: ReLU, Sigmoid, Tanh, GELU, SiLU (zero-sized types)

**Losses**: mse_loss, cross_entropy_loss, bce_with_logits_loss, l1_loss,
smooth_l1_loss, kl_div_loss

**Optimizers**: SGD (momentum), Adam, AdamW

**Schedulers**: StepDecay, CosineScheduler, WarmupScheduler, PlateauScheduler

**Training tools**: clip_grad_norm, clip_grad_value, save/load parameters,
kaiming/xavier initialization, GradScaler, cast_parameters

---

## Phase 5: Graph Engine — DONE

`graph/` — composable execution graph with fluent builder API.

**Builder methods**: from, through, also, split/merge, tag/using, input,
tag_group, loop_body (for_n/while_cond/until_cond), gate, switch,
map (each/over/slices/batched)

**Graph features**:
- Graph implements Module (Graph-as-Module composition)
- Forward references (state buffers across calls)
- reset_state / detach_state for recurrent training
- Tag-based observation: collect, record, flush, trend
- Trend queries: slope, stalled, improving, converged
- TrendGroup for aggregate queries across tag groups
- Profiling: enable_profiling, profile, timing_trend
- Visualization: DOT, SVG, SVG with profile overlay
- Training curves: plot_html, export_trends, write_log
- ETA calculation from flush timing
- end_step / end_epoch housekeeping

**Routing**:
- SoftmaxRouter, SigmoidRouter (implement NamedInputModule)
- FixedSelector, ArgmaxSelector
- ThresholdHalt, LearnedHalt
- Build-time validation of NamedInputModule capability

**Primitives**: StateAdd, Reshape, Reduce (Mean, Sum, Max, Min, Norm)

---

## Phase 6: API Cleanup — DONE

- Schedulers decoupled from optimizer (pure LR calculators)
- Default `parameters()` returns empty vec
- Auto-detect `through` wiring
- `modules!` macro for `Vec<Box<dyn Module>>`
- `Result` re-exported at crate root
- `end_step()`/`end_epoch()` housekeeping
- `collect()` returns `Result<()>` — forces `collect_with()` for non-scalar
- Reduce uses native FFI ops (libtorch max, norm)
- BatchNorm safety: errors in eval with no training stats
- Build-time validation for gate/switch NamedInputModule

---

## Phase 7: Documentation — DONE

- README with graph builder showcase, features, architecture
- 8 tutorials (tensors through utilities)
- Design docs: trajectory thesis, roadmap
- CONTRIBUTING guide
- CHANGELOG

---

## Phase 8: End-to-End Training Example — NEXT

Port the fbrl letter recognition model from goDl to floDl. This proves
the framework on real data with a non-trivial architecture (attention
loops, multi-head, recurrent state).

Deliverables:
- Data loading (batched tensor datasets)
- Complete training loop with observation
- Checkpoint/resume
- Evaluation metrics

---

## Phase 9: Crates.io Publishing

- Cargo.toml metadata (license, repository, keywords, categories)
- `//!` module-level doc comments for docs.rs
- `///` doc comments on all public types and methods
- CI with GitHub Actions
- Publish `flodl-sys` and `flodl` to crates.io

---

## Phase 10: Future

- **Data loading**: Dataset trait, TensorDataset, Loader with batching/shuffle
- **Multi-threading**: rayon for parallel level execution
- **Higher-order gradients**: differentiate through backward
- **Graph serialization**: save/load graph topology
- **Attention mechanisms**: as graph primitives
- **ONNX import/export**
- **Multi-GPU**: data parallelism, model parallelism

---

## Test Coverage

228 library tests + 15 showcase tests. Zero clippy warnings.
All passing in Docker (CPU, libtorch 2.10.0).

---

## Dev Environment

- Rust stable (2024 edition)
- libtorch 2.10+ CPU (CUDA optional via feature gate)
- Docker container for reproducible builds
- WSL2 for development
- `make test` / `make clippy` / `make shell`

---

## Lineage

floDl is a Rust port of [goDl](https://github.com/fab2s/goDl). The port
was motivated by Go's inability to manage VRAM deterministically — Rust's
ownership model solves this at the language level. The graph builder API,
module architecture, and design philosophy carry over directly. The FFI
symbol prefix and C types have been renamed from `rdl_` to `flodl_`.
