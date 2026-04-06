# floDl Roadmap

A Rust-native deep learning framework built on libtorch.

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
   finalizers, no VRAM budgets. Five phases of GC-based memory management
   replaced by `impl Drop for Tensor`.

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

~92 C functions covering:
- Tensor creation, arithmetic, activations, reductions
- Shape manipulation, slicing, joining
- Convolution, pooling, normalization
- Device management, dtype casting
- Grid sampling, layer norm
- Autograd (backward, detach, detach_, grad, set_requires_grad)

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

**Training tools**: clip_grad_norm, clip_grad_value, save/load parameters
(positional and named/partial), parameter freezing (freeze/unfreeze),
kaiming/xavier initialization, GradScaler, cast_parameters, optimizer
parameter groups (per-group LR)

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

## Phase 8: End-to-End Training Example — DONE

Implement the fbrl letter recognition model in floDl. This proves
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

## Phase 10: PyTorch API Parity

Close the gaps identified in the [parity audit](../../CHANGELOG.md) so that
common architectures (transformers, modern CNNs, RL agents) can be expressed
without workarounds.

### Tier 1 — Immediate Impact (blocks common architectures)

**Tensor ops**: `masked_fill`, `masked_select`, `cumsum`, `tril`, `randperm`,
`multinomial`, `prod`/`prod(dim)`, `logsumexp(dim)`, `F.normalize`

**Activations**: `LeakyReLU`, `ELU`, `Softplus`, `Mish`

**Modules**: `MultiheadAttention` (or `F.scaled_dot_product_attention`),
`AvgPool2d`, `RMSNorm`, `nn.Softmax`/`nn.LogSoftmax`, `nn.Flatten`

*Key insight — the single biggest gap is the transformer attention path:
`masked_fill` + scaled dot-product attention unlocks the entire transformer
family.*

### Tier 2 — Strong PyTorch Parity

**Tensor ops**: `flip`/`roll`, `split`/`unbind`, `clamp_min`/`clamp_max`,
`log1p`/`expm1`, `log2`/`log10`, `randint`, `F.one_hot`,
`maximum`/`minimum`, `any`/`all`, `atan2`, `isnan`/`isinf`,
`logical_and`/`or`/`not`, `contiguous`/`is_contiguous`, `argsort`, `scatter`

**Modules**: `GroupNorm`, `Conv1d`/`ConvTranspose1d`, sequence-level
`GRU`/`LSTM`, `nn.ModuleList`/`nn.ParameterList`, `RMSprop`,
`OneCycleLR`/`ExponentialLR`/`MultiStepLR`

**Functional**: `F.interpolate`, `F.pad` (reflect/replicate),
`F.cosine_similarity`, `F.binary_cross_entropy`, `F.embedding_bag`

**Init**: `uniform_`/`normal_` (standalone), `orthogonal_`, `trunc_normal_`

**In-place ops**: `mul_(tensor)`, `div_(scalar)`/`div_(tensor)`,
`fill_`/`copy_`

**Creation**: `empty`, `full_like`/`rand_like`/`randn_like`, `bernoulli`

### Tier 3 — Completeness

**Tensor ops**: inverse trig (`tan`/`asin`/`acos`/`atan`), `erf`/`erfc`,
`fmod`/`remainder`, `lerp`, `addmm`/`addcmul`/`addcdiv`, `trunc`/`frac`,
`diagonal`/`movedim`/`tile`, `nonzero`/`unique`/`searchsorted`,
`count_nonzero`/`median`, dim-wise `norm`, multi-dim `sum`, `cumprod`,
`isclose`

**Modules**: `Conv3d`/`ConvTranspose3d`, 1D pooling, `AdaptiveMaxPool2d`,
`InstanceNorm`, `nn.Bilinear`, `AlphaDropout`, `PixelShuffle`/`Upsample`,
padding modules, `Unfold`/`Fold`

**Activations**: `SELU`, `Hardswish`, `Hardsigmoid`, `PReLU`

**Losses**: `NLLLoss`, `CosineEmbeddingLoss`/`TripletMarginLoss`,
`HingeEmbeddingLoss`/`MarginRankingLoss`, `CTCLoss`, `PoissonNLLLoss`,
`FocalLoss`

**Optimizers**: `Adagrad`/`LBFGS`/`RAdam`/`NAdam`, `CyclicLR`

**Init**: `constant_`/`zeros_`/`ones_`, `sparse_`

---

## Recently Completed (post-v0.1.2)

- **`Rng`**: CPU-side RNG (SmallRng/Xoshiro256++) for data loading and augmentation
- **`manual_seed` / `cuda_manual_seed_all`**: Full-stack reproducibility seeding
- **`cuda_active_bytes`**: Active tensor VRAM measurement (complements `cuda_allocated_bytes`)
- **`MaxPool2d`**: 2D max pooling with full FFI chain
- **Foreach ops** (7 variants): `foreach_add_scalar_`, `foreach_mul_scalar_`, `foreach_zero_`, `foreach_add_list_`, `foreach_norm`, `foreach_lerp_scalar_`, `foreach_sqrt_`
- **Fused Adam/AdamW**: `_fused_adamw_` single multi-tensor kernel on CUDA
- **Fused gradient clipping**: foreach_norm + foreach_mul (2 kernels instead of 2N)
- **Autocast / automatic mixed precision**: `AutocastGuard`, `autocast` closure, `is_autocast_enabled`
- **GradScaler**: scale, unscale, step, update with dynamic scale
- **CUDA Graphs**: capture/replay/reset, memory pools, capture modes (`CudaGraph`, `cuda_graph_capture`)
- **Peak VRAM tracking**: `cuda_peak_active_bytes`, `cuda_peak_reserved_bytes`, `cuda_reset_peak_stats`
- **Pre-computed graph routing**: Vec-indexed routes, cached exec buffers
- **Gate vectorization**: stack + broadcast multiply + sum
- **Loop fast-path**: direct forward when no refs
- **Channels-last memory format**: `to_channels_last`, `is_channels_last`
- **Non-blocking device transfer**: `to_device_async`
- **`pin_memory`**
- **`copy_` with non_blocking**

## Phase 11: Multi-GPU Training -- DONE

- **Data loading**: `DataSet` and `BatchDataSet` traits, `DataLoader` with resident/streaming/distributed modes, VRAM-aware prefetch, `Sampler` trait
- **Graph DDP**: `Ddp::setup()` one-liner. Transparent scatter/gather, AllReduce, per-replica optimizers. `Graph::step()` handles the full sync cycle.
- **DDP Builder**: `Ddp::builder()` for thread-per-GPU Local SGD. `ApplyPolicy` (Sync/Cadence/Async) x `AverageBackend` (Nccl/Cpu) for A/B testing.
- **El Che**: heterogeneous GPU cadence strategy. Slow device anchors sync, fast device fills wall time. Auto-tunes from CudaEvent timing.
- **NCCL bindings**: `NcclComms`, `NcclRankComm` (init-on-main + split), `NcclAbortHandle` for dead worker recovery.
- **CUDA primitives**: `CudaEvent`, `CudaStream`, `StreamGuard` for async GPU-CPU pipeline.
- **Auto-balancing**: EMA throughput tracking, adaptive chunk ratios, weighted gradient averaging.
- **Per-device DataLoader backends**: each GPU independently selects resident or streaming.

## Phase 12: Future

- **Multi-threading**: rayon for parallel level execution
- **Higher-order gradients**: differentiate through backward
- **Graph serialization**: save/load graph topology
- **ONNX import/export**
- **Model parallelism**: tensor/pipeline parallelism for models that exceed single-GPU VRAM
- **JEPA primitives**: EMA target encoder, latent-space prediction, masking
  strategies. JEPA (Joint Embedding Predictive Architecture) trains by
  predicting representations rather than reconstructing inputs. flodl's
  graph tree is a natural fit: the target encoder is a frozen subgraph
  updated via EMA, the predictor bridges context and target embeddings,
  and the observation system can monitor representation collapse. Exploring
  what reusable building blocks would make JEPA architectures (I-JEPA,
  V-JEPA) easy to express in the graph builder.

---

## Test Coverage

1022 library tests. Zero clippy warnings. All tests run on CPU and CUDA.
All passing in Docker (CPU, libtorch 2.10.0).

---

## Dev Environment

- Rust stable (2024 edition)
- libtorch 2.10+ CPU (CUDA optional via feature gate)
- Docker container for reproducible builds
- WSL2 for development
- `make test` / `make clippy` / `make shell`

---

## Story

The graph builder, module architecture, and design philosophy were first
explored in [an earlier Go implementation](https://github.com/fab2s/goDl).
Go's garbage collector could not manage GPU memory deterministically, which
required increasingly complex workarounds. Rust's ownership model solved
this at the language level — the architecture carried forward, the memory
management fights did not.
