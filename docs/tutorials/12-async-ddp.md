# Tutorial 12: Async DDP

Thread-per-GPU training with Local SGD. Each GPU runs its own optimizer
independently; a lightweight coordinator triggers periodic parameter
averaging. Works with any `Module`, not just `Graph`.

> **Prerequisites**: [Training](04-training.md). Familiarity with
> [Multi-GPU Training](11-multi-gpu.md) recommended but not required.

> **Time**: ~25 minutes.

## Quick start

```rust
use flodl::*;
use std::sync::Arc;

let dataset: Arc<dyn BatchDataSet> = Arc::new(MyDataset::new());

let ddp = AsyncDdp::builder(
    |dev| MyModel::on_device(dev),              // model factory
    |params| Adam::new(params, 0.001),          // optimizer factory
    |model, batch| {                            // train function
        let input = Variable::new(batch[0].clone(), false);
        let target = Variable::new(batch[1].clone(), false);
        let pred = model.forward(&input)?;
        pred.mse(&target)?.mean()
    },
)
.dataset(dataset)
.batch_size(32)
.num_epochs(10)
.run()?;                    // non-blocking: spawns threads, returns immediately

let state = ddp.join()?;   // blocks until training completes
// state.params[i] corresponds to model.parameters()[i]
// state.buffers[i] corresponds to model.buffers()[i]
```

That is the entire setup. The builder detects GPUs, spawns one thread per
GPU, creates a coordinator thread, and returns immediately. `join()` blocks
until all epochs complete and returns the averaged final parameters.

### The three closures

- **`model_factory(Device) -> Result<M>`**: Creates a model on the given
  device. Called once per GPU thread. Why a closure? `Variable` and `Buffer`
  are `Rc`-based (not Send), so each thread needs its own instance.
- **`optim_factory(&[Parameter]) -> O`**: Creates an optimizer for a
  model's parameters. Each GPU gets its own optimizer.
- **`train_fn(&M, &[Tensor]) -> Result<Variable>`**: Receives the model
  and a batch of tensors, returns the scalar loss. The worker handles
  `backward()`, `optimizer.step()`, and `zero_grad()`.

### Single-GPU fallback

With fewer than 2 CUDA devices, training runs on the main thread. No
worker threads, no coordinator, no averaging. The API is identical:
`join()` returns `TrainedState`. Develop on a laptop, deploy to a
multi-GPU server with zero code changes.

## The builder API

All configuration is done through the builder before calling `.run()`:

```rust
let ddp = AsyncDdp::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset)                      // required
    .batch_size(32)                        // required
    .num_epochs(10)                        // required
    .policy(ApplyPolicy::Cadence)          // default: Cadence
    .backend(AverageBackend::Nccl)         // default: Nccl
    .overhead_target(0.10)                 // AllReduce < 10% of compute
    .max_anchor(200)                       // gradient staleness cap
    .max_batch_diff(50)                    // max lead of fastest over slowest
    .divergence_threshold(0.05)            // Async mode: tighten at 5% divergence
    .checkpoint_every(5)                   // save every 5 averaging events
    .checkpoint_fn(|ver, model| {
        // save model checkpoint
        Ok(())
    })
    .run()?;
```

## How it works

```
GPU Thread 0:  model+Adam -> [fwd -> bwd -> step -> report timing -> repeat]
GPU Thread 1:  model+Adam -> [fwd -> bwd -> step -> report timing -> repeat]
Coordinator:   [collect timing -> trigger averaging -> monitor divergence]
```

Each GPU runs a complete training loop independently. The coordinator
collects per-batch timing from all workers (for ElChe throughput ratios)
and periodically triggers parameter averaging. Between averaging events,
each GPU trains with its own local optimizer. This is Local SGD.

Two orthogonal knobs control the behavior:
- **`ApplyPolicy`**: WHEN to average (the interval K)
- **`AverageBackend`**: HOW to average (the transport)

## Choosing a policy: when to average

### Sync (K=1)

Average after every batch. Every GPU processes one batch, then parameters
are synchronized. Equivalent to standard DDP. Best convergence guarantees,
but the fast GPU waits at every averaging point.

```rust
.policy(ApplyPolicy::Sync)
```

**Use when**: homogeneous GPUs, small models, correctness-first.

### Cadence (K=N from ElChe)

The slow GPU anchors the cadence; the fast GPU processes more batches per
window. K is determined by ElChe's adaptive algorithm.

```rust
.policy(ApplyPolicy::Cadence)  // default
```

**Use when**: heterogeneous GPUs (e.g., mixing GPU generations).
This is the recommended default for most setups.

### Async (K=adaptive)

ElChe starts conservative (K=1), then backs off as parameter divergence
stays low. If replicas drift apart, K tightens again. Maximizes GPU
utilization at the cost of some gradient staleness.

```rust
.policy(ApplyPolicy::Async)
.divergence_threshold(0.05)  // tighten at 5% relative norm difference
```

**Use when**: large models where each batch is expensive and
synchronization overhead matters. Monitor loss curves.

### Decision summary

| Scenario | Recommended policy |
|----------|-------------------|
| Same GPU model on all devices | `Sync` |
| Mixed GPU generations | `Cadence` |
| Large model, expensive batches | `Async` |
| Unsure | `Cadence` (safe default) |

## Choosing a backend: how to average

### Nccl

In-place AllReduce on GPU via DMA (NVLink or PCIe peer-to-peer). Zero
extra memory. All GPUs sync at the collective barrier.

```rust
.backend(AverageBackend::Nccl)  // default
```

### Cpu

Workers send parameter snapshots to the coordinator, which computes a
weighted average on CPU, then distributes the result back. No GPU ever
blocks on another GPU. Uses O(world_size * model_size) CPU RAM.

```rust
.backend(AverageBackend::Cpu)
```

The CPU backend operates as a non-blocking 3-phase state machine
(Idle/Collecting/Computing) that keeps the coordinator responsive even
during averaging.

### Tradeoff table

| | Nccl | Cpu |
|---|---|---|
| **Memory** | Zero extra (in-place) | O(W * M) CPU RAM |
| **Latency** | GPU-to-GPU DMA | GPU -> CPU -> average -> CPU -> GPU |
| **Blocking** | All GPUs sync at barrier | No GPU ever blocks |
| **Fault tolerance** | Abort handles unblock stuck ops | Timeout (5s) detects dead workers |

## A/B testing backends

Because `ApplyPolicy` and `AverageBackend` are orthogonal, you can run
the exact same training configuration twice, changing only the backend.
If the loss curves match, you have validated the cheaper backend for your
workload.

### Protocol

1. Fix your seed, dataset, model, and policy. Train with `Nccl`:

```rust
let ddp = AsyncDdp::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset.clone())
    .batch_size(32)
    .num_epochs(10)
    .policy(ApplyPolicy::Cadence)
    .backend(AverageBackend::Nccl)      // <-- first run
    .run()?;
let nccl_state = ddp.join()?;
```

2. Same everything, swap the backend:

```rust
let ddp = AsyncDdp::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset.clone())
    .batch_size(32)
    .num_epochs(10)
    .policy(ApplyPolicy::Cadence)
    .backend(AverageBackend::Cpu)       // <-- second run
    .run()?;
let cpu_state = ddp.join()?;
```

3. Compare final losses and wall times.

### Interpretation

- **Curves match**: CPU backend is validated. Use it when NVLink is
  unavailable, for debugging, or on machines where NCCL is problematic.
- **Curves diverge**: NCCL's tighter synchronization matters for your
  model. Stick with NCCL, or increase averaging frequency.

This is not just a debugging tool. In production you might discover that
the CPU path works fine for your model, freeing NVLink bandwidth for
other workloads or simplifying deployment on heterogeneous clusters.

## Strategy guide: the full matrix

| Policy | Backend | Use case | Throughput | Convergence |
|--------|---------|----------|------------|-------------|
| Sync | Nccl | Homogeneous GPUs, small models | Baseline | Best |
| Sync | Cpu | Debugging, validation | Lower | Best |
| Cadence | Nccl | **Heterogeneous GPUs (recommended)** | Good | Good |
| Cadence | Cpu | Heterogeneous + no NVLink | Moderate | Good |
| Async | Nccl | Large models, maximum throughput | Best | Monitor needed |
| Async | Cpu | Research, maximum independence | Good | Monitor needed |

Start with **Cadence + Nccl** for heterogeneous setups, **Sync + Nccl**
for homogeneous. Use `Cpu` backend when debugging or when NCCL is
unavailable.

## Safety guards

### max_batch_diff

Hard limit on how far any GPU can run ahead of the slowest. Workers that
exceed the limit are throttled (blocked on the control channel) until the
next averaging event.

```rust
.max_batch_diff(50)   // fast GPU can be at most 50 batches ahead
// .max_batch_diff(0) // strict lockstep (like Sync but with any policy)
```

### divergence_threshold (Async mode)

Controls how aggressively the averaging interval adapts:
- If parameter divergence > threshold: halve the interval (more frequent)
- If divergence < threshold/2: double the interval (less frequent)

```rust
.divergence_threshold(0.05)  // default: 5% relative norm difference
```

### NCCL abort handles

If a worker dies mid-collective (e.g., OOM), `AsyncDdp` calls
`ncclCommAbort` on all communicators, unblocking surviving workers instead
of letting them hang forever. Also triggered in `Drop`.

### CPU averaging timeout

If not all worker snapshots arrive within `snapshot_timeout_secs` (default
5s), the round is soft-aborted: missing ranks are logged, stale snapshots
are drained, and the coordinator retries on the next cycle.

```rust
AsyncDdpConfig::new().with_snapshot_timeout(10)  // 10 seconds
```

## Checkpointing

Save checkpoints at regular intervals during training:

```rust
let ddp = AsyncDdp::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset)
    .batch_size(32)
    .num_epochs(100)
    .checkpoint_every(5)          // every 5 averaging events
    .checkpoint_fn(|version, model| {
        model.save_checkpoint(&format!("ckpt_v{version}.fdl"))
    })
    .run()?;
```

The checkpoint function receives `(version, &model)` where `version` is
the averaging event count (multi-GPU) or epoch number (single-GPU).
Errors are logged but do not stop training.

In single-GPU mode, `checkpoint_every` counts epochs instead of averaging
events.

## The train function

The train function is called once per batch by the worker. It receives:
- `&M`: a reference to the concrete model (your specific type, not `dyn Module`)
- `&[Tensor]`: the batch tensors (on the worker's GPU device)

It must return a scalar `Variable` representing the loss:

```rust
// Simple MSE
|model, batch| {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].clone(), false);
    model.forward(&input)?.mse(&target)?.mean()
}
```

The worker handles everything after the loss is returned: `backward()`,
`optimizer.step()`, `zero_grad()`, and timing reports.

## Quick reference

### Types

| Type | Description |
|------|-------------|
| `AsyncDdp` | Orchestrator: spawns workers + coordinator |
| `AsyncDdpBuilder` | Fluent builder for configuration |
| `AsyncDdpConfig` | Configuration struct with builder methods |
| `ApplyPolicy` | Sync / Cadence / Async |
| `AverageBackend` | Nccl / Cpu |
| `TrainedState` | Final params + buffers (CPU tensors) |
| `CheckpointFn<M>` | `Arc<dyn Fn(u64, &M) -> Result<()> + Send + Sync>` |

### AsyncDdp methods

| Method | Description |
|--------|-------------|
| `AsyncDdp::builder(model_fn, optim_fn, train_fn)` | Create builder |
| `AsyncDdp::auto(...)` | Quick-start with all args |
| `AsyncDdp::auto_with(..., config)` | Quick-start with config |
| `.join()` | Block until done, return TrainedState |
| `.world_size()` | Number of GPUs |
| `.devices()` | Device list |

### AsyncDdpConfig methods

| Method | Default | Description |
|--------|---------|-------------|
| `.with_overhead_target(f64)` | 0.10 | AllReduce overhead ceiling |
| `.with_max_anchor(usize)` | 200 | Gradient staleness cap |
| `.with_anchor(usize)` | 10 | Initial anchor count |
| `.with_divergence_threshold(f64)` | 0.05 | Async mode threshold |
| `.with_max_batch_diff(usize)` | None | Max batch lead |
| `.with_checkpoint_every(usize)` | None | Checkpoint interval |
| `.with_snapshot_timeout(u64)` | 5 | CPU averaging timeout (seconds) |

---

Previous: [Multi-GPU Training](11-multi-gpu.md) |
Next: [DDP Reference](../ddp.md)
