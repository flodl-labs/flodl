# Tutorial 12: DDP Builder

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

let ddp = Ddp::builder(
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
let ddp = Ddp::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset)                      // required
    .batch_size(32)                        // required
    .num_epochs(10)                        // required
    .policy(ApplyPolicy::Cadence)          // default: Cadence
    .backend(AverageBackend::Nccl)         // default: Nccl
    .overhead_target(0.10)                 // AllReduce < 10% of compute
    .max_anchor(200)                       // gradient staleness cap
    .anchor(10)                            // initial anchor count
    .max_batch_diff(50)                    // max lead of fastest over slowest
    .divergence_threshold(0.05)            // Async mode: tighten at 5% divergence
    .progressive_dispatch(true)            // stream work in small chunks
    .checkpoint_every(5)                   // save every 5 averaging events
    .checkpoint_fn(|ver, model| {
        // save model checkpoint
        Ok(())
    })
    .epoch_fn(|epoch, worker| {
        // learning rate schedule, noise curriculum, etc.
        let lr = 0.001 * (0.95_f64).powi(epoch as i32);
        worker.set_lr(lr);
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

### Why Cadence and Async use different triggers

**Cadence** uses a **wall-time trigger**: averaging fires when the slowest
rank's accumulated wall time reaches the anchor's wall-time. This gives
predictable, stable rendezvous points for the AllReduce barrier.

**Async** uses a **batch-count trigger**: averaging fires when all ranks
complete their assigned batch counts. This is intentional: the slight
overshoot between averaging events means each replica explores a slightly
different parameter neighborhood. This diversity benefits convergence
(like implicit meta-learning).

Benchmark evidence confirms this: async with wall-time trigger gave worse
convergence than batch-count trigger, because cutting off the overshoot
eliminates the exploration diversity that makes async work.

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

## A/B testing: find the best config for your model

Every model responds differently to averaging frequency and transport
timing. You have 6 valid configurations (3 policies x 2 backends), and
the best one depends on your model, your data, and your hardware.

The recommended approach: **run 3-5 epochs with different configs, compare
loss curves, then commit to the winner for your full training run.** This
takes minutes, not hours.

### Step 1: start with El Che (Async + NCCL)

Async + NCCL is the best overall config in practice. Fast GPUs overshoot
between averaging events, creating parameter diversity that benefits
convergence. The divergence monitor auto-tunes the averaging interval.

```rust
let ddp = Ddp::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset.clone())
    .batch_size(32)
    .num_epochs(5)                          // just enough to see the trend
    .max_grad_norm(5.0)
    .policy(ApplyPolicy::Async)
    .backend(AverageBackend::Nccl)
    .run()?;
let async_state = ddp.join()?;
```

### Step 2: test Cadence (strong second)

Same code, one line changed:

```rust
    .policy(ApplyPolicy::Cadence)           // <-- the only change
```

Cadence provides more predictable sync points. If your model needs tighter
synchronization, Cadence will show it in the first few epochs.

### Step 3 (optional): strict Sync baseline

```rust
    .policy(ApplyPolicy::Sync)              // <-- strict sync
```

This tells you whether strict synchronization helps your specific model.
For most workloads, Async and Cadence match or beat Sync.

### CPU backend: known bug

The CPU averaging backend has a known convergence bug. All three CPU
policies produce near-random accuracy. Do not use `AverageBackend::Cpu`
for training. The bug is under active investigation. See the
[DDP reference](/guide/ddp) for details.

### What to compare

- **Loss at epoch N**: lower is better
- **Wall time per epoch**: Cadence should be faster than Sync on mixed hardware
- **Loss per wall-second**: the real metric. Slightly higher loss in half the time often wins.

### The full matrix

| Policy | Backend | Use case | Throughput | Convergence |
|--------|---------|----------|------------|-------------|
| Async | Nccl | **Best overall (recommended)** | Best | Best with clipping |
| Cadence | Nccl | Strong second, predictable sync | Good | Good |
| Sync | Nccl | Strict sync baseline | Baseline | Good |
| Async | Cpu | **Known bug** -- do not use | -- | Broken |
| Cadence | Cpu | **Known bug** -- do not use | -- | Broken |
| Sync | Cpu | **Known bug** -- do not use | -- | Broken |

Start with **Async + Nccl** (El Che). It's the best overall config in
practice: fast GPUs overshoot between averaging, creating parameter
diversity that benefits convergence. A/B test against **Cadence + Nccl**
(strong second, more predictable sync) or **Sync + Nccl** (strict
baseline). The **CPU backend** has a known convergence bug and should not
be used for training. See the [DDP reference](/guide/ddp) for details.
The fix is under active investigation.

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

If a worker dies mid-collective (e.g., OOM), `DdpHandle` calls
`ncclCommAbort` on all communicators, unblocking surviving workers instead
of letting them hang forever. Also triggered in `Drop`.

### CPU averaging timeout

If not all worker snapshots arrive within `snapshot_timeout_secs` (default
5s), the round is soft-aborted: missing ranks are logged, stale snapshots
are drained, and the coordinator retries on the next cycle.

```rust
DdpRunConfig::new().with_snapshot_timeout(10)  // 10 seconds
```

## Checkpointing

Save checkpoints at regular intervals during training:

```rust
let ddp = Ddp::builder(model_factory, optim_factory, train_fn)
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

## Custom metrics

Record named scalars inside the train function using `record_scalar()`.
These are aggregated per rank per epoch and available via
`DdpHandle::poll_metrics()`:

```rust
use flodl::nn::record_scalar;

let ddp = Ddp::builder(
    model_factory,
    optim_factory,
    |model, batch| {
        let input = Variable::new(batch[0].clone(), false);
        let target = Variable::new(batch[1].clone(), false);
        let pred = model.forward(&input)?;
        let loss = pred.mse(&target)?.mean()?;

        // Record custom metrics (thread-local, aggregated per epoch)
        let correct = pred.argmax(-1, false)?.eq_tensor(&target.tensor())?.sum()?;
        let accuracy = correct.item::<f64>()? / batch[0].size()[0] as f64;
        record_scalar("accuracy", accuracy);

        Ok(loss)
    },
)
.dataset(dataset)
.batch_size(32)
.num_epochs(50)
.run()?;

// Consume metrics from outside
while let Some(m) = ddp.next_metrics() {
    println!(
        "epoch {} | loss={:.4} | acc={:.3} | {:.0}ms",
        m.epoch, m.avg_loss,
        m.scalars.get("accuracy").unwrap_or(&0.0),
        m.epoch_ms,
    );
}
let state = ddp.join()?;
```

`EpochMetrics` includes per-rank breakdowns (`per_rank_loss`,
`per_rank_time_ms`, `per_rank_scalars`) alongside aggregated values.

## Monitor integration

Wire the DDP handle into a training `Monitor` for the live dashboard
and HTML archive:

```rust
let ddp = Ddp::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset)
    .batch_size(32)
    .num_epochs(100)
    .run()?;

let mut monitor = Monitor::new(100);
ddp.setup_monitor(&mut monitor);  // graph identity + architecture SVG
monitor.serve(3000)?;              // live dashboard at http://localhost:3000

while let Some(metrics) = ddp.next_metrics() {
    let elapsed = std::time::Duration::from_millis(metrics.epoch_ms as u64);
    monitor.log(metrics.epoch, elapsed, &metrics);
}

let state = ddp.join()?;
monitor.finish();
monitor.save_html("training.html");
```

The dashboard shows per-GPU tabs with VRAM, throughput, and batch share
charts automatically when 2+ GPUs are detected.

## Epoch callbacks

Use `epoch_fn` for per-epoch logic that runs inside each worker thread:

```rust
.epoch_fn(|epoch, worker| {
    // Cosine annealing
    let lr = 0.001 * 0.5 * (1.0 + (std::f64::consts::PI * epoch as f64 / 100.0).cos());
    worker.set_lr(lr);
})
```

The callback receives `(epoch: usize, worker: &mut GpuWorker<M>)` and
runs before the epoch's training begins. Available methods on `worker`:

| Method | Description |
|--------|-------------|
| `worker.set_lr(f64)` | Set learning rate on this worker's optimizer |
| `worker.current_epoch()` | Current epoch number (0-based) |
| `worker.rank()` | This worker's rank |
| `worker.device()` | This worker's CUDA device |
| `worker.model()` | Reference to the concrete model |

## Progressive dispatch

By default, Cadence and Async modes use progressive dispatch: the
coordinator sends work in small chunks rather than full epoch partitions.
This lets the system adapt to throughput changes mid-epoch.

```rust
// Explicit control (auto for Cadence/Async)
.progressive_dispatch(true)

// Sync mode: disabled by default (full partitions upfront)
.progressive_dispatch(false)
```

Progressive dispatch adds slight coordination overhead but gives better
throughput on heterogeneous hardware where speed ratios may shift during
training (thermal throttling, competing workloads).

## Quick reference

### Types

| Type | Description |
|------|-------------|
| `DdpHandle` | Orchestrator: spawns workers + coordinator |
| `DdpBuilder` | Fluent builder for configuration |
| `DdpRunConfig` | Configuration struct with builder methods |
| `ApplyPolicy` | Sync / Cadence / Async |
| `AverageBackend` | Nccl / Cpu |
| `TrainedState` | Final params + buffers (CPU tensors) |
| `EpochMetrics` | Aggregated metrics for one completed epoch |
| `GpuWorker<M>` | Per-GPU worker (available in epoch_fn callback) |
| `CheckpointFn<M>` | `Arc<dyn Fn(u64, &M) -> Result<()> + Send + Sync>` |
| `EpochFn<M>` | `Arc<dyn Fn(usize, &mut GpuWorker<M>) + Send + Sync>` |

### DdpHandle methods

| Method | Description |
|--------|-------------|
| `Ddp::builder(model_fn, optim_fn, train_fn)` | Create builder |
| `.join()` | Block until done, return TrainedState |
| `.world_size()` | Number of GPUs |
| `.devices()` | Device list |
| `.poll_metrics()` | Non-blocking: return all pending EpochMetrics |
| `.next_metrics()` | Blocking: return next EpochMetrics |
| `.setup_monitor(&mut Monitor)` | Wire graph identity + config into monitor |
| `.architecture_svg()` | Graph architecture SVG (if model is a Graph) |

### DdpRunConfig methods

| Method | Default | Description |
|--------|---------|-------------|
| `.with_overhead_target(f64)` | 0.10 | AllReduce overhead ceiling |
| `.with_max_anchor(usize)` | 200 | Gradient staleness cap |
| `.with_anchor(usize)` | 10 | Initial anchor count |
| `.with_divergence_threshold(f64)` | 0.05 | Async mode threshold |
| `.with_max_batch_diff(usize)` | None | Max batch lead |
| `.with_checkpoint_every(usize)` | None | Checkpoint interval |
| `.with_snapshot_timeout(u64)` | 5 | CPU averaging timeout (seconds) |
| `.with_partition_ratios(Vec<f64>)` | None | Fixed per-rank data splits |
| `.with_progressive_dispatch(bool)` | Auto | Stream work in small chunks |

---

Previous: [Multi-GPU Training](11-multi-gpu.md) |
Next: [Data Loading](13-data-loading.md)
