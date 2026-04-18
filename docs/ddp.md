# Distributed Data Parallel Reference

Comprehensive reference for floDl's multi-GPU training capabilities.
For progressive introductions, see [Tutorial 11: Multi-GPU Training](tutorials/11-multi-gpu.md)
and [Tutorial 12: DDP Builder](tutorials/12-async-ddp.md).

## Overview

floDl provides two approaches to multi-GPU training. Both use the same NCCL
backend and ElChe cadence strategy, but differ in how they integrate with
your model:

**Graph DDP** -- integrates with the Graph builder. One-liner setup via
`Ddp::setup()`. The training loop is identical for 1 or N GPUs. Best for
Graph-based models where you want transparent scaling.

**DDP Builder** -- works with any `Module`. Thread-per-GPU with Local SGD.
3 policies x 2 backends = 6 configs, swappable in one line for A/B testing.
Both NCCL and CPU backends are production-ready.
Best for non-Graph modules or when you need maximum configurability.

### Which one to use

```
Using the Graph builder?
  YES --> Graph DDP (Ddp::setup)
  NO  --> DDP Builder (Ddp::builder)

Need A/B testing NCCL vs CPU averaging?
  YES --> DDP Builder

Need per-GPU thread independence (different epochs, fault tolerance)?
  YES --> DDP Builder

Want the simplest possible setup?
  YES --> Graph DDP (Ddp::setup)
```

Both approaches auto-detect available CUDA devices and fall back to
single-GPU/CPU mode when fewer than 2 GPUs are available.

---

## Graph DDP

### Ddp::setup()

One-liner to auto-detect GPUs, distribute the model, set per-replica
optimizers, and enable training mode.

```rust
use flodl::*;

let model = FlowBuilder::from(Linear::new(784, 256)?)
    .through(ReLU::new())
    .through(Linear::new(256, 10)?)
    .label("classifier")
    .build()?;

// Single call: detect GPUs, replicate, set optimizer, training mode
Ddp::setup(&model, &builder, |p| Adam::new(p, 0.001))?;

// Training loop -- identical for 1 or N GPUs
for epoch in 0..100 {
    for batch in &dataset {
        let loss = model.forward(&batch)?.mse(&target)?;
        model.step()?;  // AllReduce + sync + optimizer + zero_grad
    }
}
```

Prints hardware diagnostics to stderr:
```
  ddp: 2 GPUs (heterogeneous) | RTX 5060 Ti (16.0 GB) | GTX 1060 (6.0 GB)
```

**Behavior by hardware:**
- 2+ CUDA devices: full DDP with NCCL
- 1 CUDA device: sets optimizer + training mode, no distribution
- CPU only: same as single device

### Ddp::setup_with()

Same as `setup()` but accepts a `DdpConfig` for explicit configuration:

```rust
let config = DdpConfig::new()
    .speed_hint(1, 0.4)        // GPU 1 is ~40% the speed of GPU 0
    .overhead_target(0.10)     // keep AllReduce < 10% of compute
    .max_anchor(Some(200))     // gradient staleness cap
    .max_grad_norm(5.0);       // per-rank gradient clipping

Ddp::setup_with(&model, &builder, |p| Adam::new(p, 0.001), config)?;
```

### Graph::distribute()

Called internally by `Ddp::setup()`. Can also be called directly for
manual setup:

```rust
model.distribute(|dev| {
    FlowBuilder::from(Linear::on_device(784, 256, dev)?)
        .through(ReLU::new())
        .through(Linear::on_device(256, 10, dev)?)
        .label("classifier")
        .build()
})?;
```

Creates one replica per available CUDA device. Broadcasts parameters from
device 0 to all replicas. Cross-device autograd is preserved: `to_device()`
wraps the transfer in `ToCopyBackward` so gradients flow back naturally.

### Graph::step()

Performs the full synchronization cycle in one call:

1. AllReduce gradients across all replicas (NCCL)
2. Sync buffers (BatchNorm running stats, etc.)
3. Optimizer step on each replica
4. Zero gradients

With El Che enabled, step additionally:
- Normalizes accumulated gradients by `1/count[rank]`
- Performs weighted AllReduce (each replica scaled by batch contribution)
- Reports timing to ElChe for adaptive cadence
- Updates DataLoader batch counts for the next window

### DataLoader integration

```rust
let loader = DataLoader::from_batch_dataset(dataset)
    .batch_size(32)
    .names(&["image", "label"])
    .build()?;

model.set_data_loader(loader, "image");  // auto-wires to graph input

for batch in model.epoch(0) {
    let batch = batch?;
    let loss = model.forward_batch(&batch)?;
    model.step()?;
}
```

When distributed, `set_data_loader()` creates per-device backends:
- Each GPU independently selects resident (data fits in VRAM) or streaming
  (prefetch worker with async H2D transfers)
- No lowest-common-denominator constraint: a 16 GB GPU can go resident
  while a 6 GB GPU uses streaming
- Presharded forward: each replica forwards its local shard with zero
  cross-device input transfer

### DdpConfig

| Field | Default | Description |
|-------|---------|-------------|
| `speed_hint(rank, ratio)` | None | Initial speed estimate (self-corrects after first timing) |
| `overhead_target(f64)` | 0.10 | AllReduce overhead ceiling as fraction of compute |
| `max_anchor(Option<usize>)` | None (auto) | `None` = auto, `Some(0)` = disable El Che, `Some(n)` = fixed cap |
| `max_grad_norm(f64)` | None | Per-rank gradient clipping before AllReduce. Clips accumulated gradients on all ranks (including replicas the caller cannot reach). Uses fused C++ kernel. |
| `timeline(Arc<Timeline>)` | None | Attach a [`Timeline`](https://docs.rs/flodl/latest/flodl/monitor/struct.Timeline.html) so the DDP runtime injects sync/epoch/anchor events into the profiler stream. |

### Graph DDP — LR scheduling

A scheduler attached on the Graph DDP path drives every replica's
optimizer LR through `Graph::step()`:

| Method | Description |
|--------|-------------|
| `Graph::set_scheduler(Arc<dyn Scheduler>)` | Attach a per-batch scheduler. `step()` updates LR to `scheduler.lr(training_step) * lr_scale` before applying gradients. |
| `Graph::set_lr_scale(f64)` | Linear-scaling multiplier (Goyal et al., 2017). Default `1.0`. Has no effect without a scheduler — bake the scaling into the optimizer's base LR instead. |
| `Graph::training_step()` | Current step counter (increments once per `step()` call). |

```rust
use std::sync::Arc;
use flodl::nn::MultiStepLR;

let sched: Arc<dyn flodl::nn::Scheduler> =
    Arc::new(MultiStepLR::new(0.1, &[100, 150], 0.1));
graph.set_scheduler(sched);
graph.set_lr_scale(world_size as f64);   // optional linear scaling
```

### Manual DDP: Ddp::wrap()

For complex training patterns (GAN, RL, progressive growing) where you need
explicit control over synchronization:

```rust
let ddp = Ddp::wrap(&[&model_a, &model_b], &devices)?;

// Explicit sync cycle
ddp.sync_params()?;
// ... forward + backward ...
ddp.all_reduce_gradients()?;
// or with weighted averaging for El Che:
ddp.weighted_all_reduce_gradients(&batch_counts)?;
ddp.sync_buffers()?;
```

---

## El Che Cadence Strategy

Named after Che Guevara's marching principle: "the column marches at the
slowest one's pace."

### The problem

Traditional DDP forces all GPUs to synchronize after every batch. With
heterogeneous hardware (e.g., RTX 5060 Ti + GTX 1060), the fast GPU idles
60% of the time waiting for the slow one.

### The solution

The slow device anchors the sync cadence. The fast device processes more
batches between sync points, filling what would otherwise be idle wall time.
AllReduce happens when the slow device completes its anchor count.

### How it works

1. **Anchor**: number of batches the slow device processes per sync window
2. **Batch counts**: `counts[rank] = round(anchor * speed_ratio[rank])`
3. **Speed ratios**: discovered from CudaEvent timing after the first sync

After each sync, `report_timing(wall_ms, sync_ms)` is called:

**Speed discovery:**
- Each rank's `ms_per_batch` is computed as `wall_ms[rank] / batch_count[rank]`
- EMA-smoothed with error-adaptive alpha: `alpha = clamp(prediction_error, 0.1, 0.8)`.
  Large corrections use high alpha for fast catch-up; small jitter uses low alpha
  for stability
- Speed ratios derived from relative ms_per_batch values (slowest = 1.0)

**Anchor auto-tuning:**
- `overhead_ratio = sync_ms / (wall_ms - sync_ms)` measures what fraction
  of compute time was spent in AllReduce
- If overhead > target: increase anchor by `ceil(anchor * overhead / target)`
  (proportional to the excess, because overhead is wasted GPU time)
- If overhead < target/2: decrease anchor by 1 (gradual, because lower
  anchor means fresher gradients)
- Anchor is clamped to `[1, max_anchor]`

**Batch count distribution:**
- `counts[rank] = round(anchor * speed_ratio[rank])`
- `clamp_total(max)`: proportionally clamp counts near epoch boundaries
  so workers do not overshoot the remaining samples

### Configuration

| Method | Default | Description |
|--------|---------|-------------|
| `ElChe::new(world_size, anchor)` | -- | Create with initial anchor |
| `.with_speed_ratio(slow_rank, ratio)` | Equal | Seed speed estimate |
| `.with_overhead_target(target)` | 0.10 | AllReduce overhead ceiling |
| `.with_max_anchor(max)` | 200 | Gradient staleness cap |
| `.with_max_batch_diff(max)` | None | Max batch lead of fastest over slowest |

### Weighted gradient averaging

When batch counts are unequal, each replica's gradient is scaled by its
batch contribution before AllReduce Sum:

```
weight[rank] = count[rank] / sum(counts)
grad_avg = sum(weight[rank] * grad[rank])
```

This produces the mathematically correct mean gradient regardless of
per-device batch counts.

---

## DDP Builder

### Ddp::builder()

Recommended entry point. Returns a builder that launches training
non-blocking:

```rust
let ddp = Ddp::builder(
    |dev| MyModel::on_device(dev),              // model factory
    |params| Adam::new(params, 0.001),          // optimizer factory
    |model, batch| {                            // train function
        let input = Variable::new(batch[0].clone(), false);
        let target = Variable::new(batch[1].clone(), false);
        model.forward(&input)?.mse(&target)?.mean()
    },
)
.dataset(dataset)
.batch_size(32)
.num_epochs(10)
.policy(ApplyPolicy::Cadence)
.backend(AverageBackend::Nccl)
.run()?;                                        // spawns threads

let state = ddp.join()?;                        // blocks until done
// state.params, state.buffers are CPU tensors
```

**Why closures?** Each GPU thread needs its own model and optimizer.
`Rc<RefCell<...>>` types (Variable, Buffer) are not Send, so they must be
constructed inside each thread. The factories are called once per GPU.

### Ddp::builder() quick-start

All arguments can be passed directly via the builder:

```rust
let ddp = Ddp::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset)
    .batch_size(32)
    .num_epochs(10)
    .policy(ApplyPolicy::Cadence)
    .backend(AverageBackend::Nccl)
    .run()?;
```

### Builder methods

| Method | Required | Default | Description |
|--------|----------|---------|-------------|
| `.dataset(Arc<dyn BatchDataSet>)` | Yes | -- | Training dataset |
| `.batch_size(usize)` | Yes | -- | Batch size per GPU |
| `.num_epochs(usize)` | Yes | -- | Number of epochs |
| `.policy(ApplyPolicy)` | No | Cadence | When to average |
| `.backend(AverageBackend)` | No | Nccl | How to average |
| `.overhead_target(f64)` | No | 0.10 | AllReduce overhead ceiling |
| `.max_anchor(usize)` | No | 200 | Gradient staleness cap |
| `.anchor(usize)` | No | 10 | Initial anchor count |
| `.divergence_threshold(f64)` | No | 0.05 | Async mode divergence threshold |
| `.max_batch_diff(usize)` | No | None | Max batch lead (0 = lockstep) |
| `.max_grad_norm(f64)` | No | None | Per-worker gradient clipping between backward and optimizer step |
| `.progressive_dispatch(bool)` | No | Auto | Stream work in small chunks (auto: true for Cadence/Async) |
| `.checkpoint_every(usize)` | No | None | Checkpoint interval (averaging events or epochs) |
| `.checkpoint_fn(Fn)` | No | None | Checkpoint callback on rank 0 |
| `.epoch_fn(Fn)` | No | None | Per-epoch callback inside each worker thread |
| `.scheduler(factory)` | No | None | Per-worker LR scheduler factory closure. Each rank instantiates its own scheduler. Pairs with `.lr_scale_ratio()` for linear scaling. |
| `.lr_scale_ratio(f64)` | No | 1.0 | Auto LR scaling factor for the linear scaling rule (Goyal et al., 2017). Effective `lr_scale = 1 + ratio * (world_size - 1)`. `1.0` (default) for full linear scaling, `0.0` to disable. |
| `.no_divergence_guard()` | No | (guard on) | Disable the convergence guard entirely. Useful during calibration runs when divergence trend logging adds more noise than signal. |
| `.max_overshoot(usize)` | No | (auto-tuned) | Async-only: cap how many extra batches the fastest rank may run past the slowest before the next averaging event. Bounds the worst case explicitly when the auto-tuner is too permissive. |
| `.timeline(Arc<Timeline>)` | No | None | Attach a `monitor::Timeline` so the DDP runtime injects `EpochStart/End`, `SyncStart/End`, `CpuAvgStart/End`, `AnchorChanged`, `Throttle` events into the profiler stream. |

### DdpRunConfig

Advanced config via `DdpRunConfig` (passed through the builder methods above):

| Field | Default | Description |
|-------|---------|-------------|
| `partition_ratios` | None | Fixed per-rank data splits (e.g. `[0.7, 0.3]`). Disables auto-rebalancing. |
| `snapshot_timeout_secs` | 5 | CPU averaging timeout before soft-abort |
| `progressive_dispatch` | Auto | When true, coordinator streams small chunks to workers instead of full epoch partitions. Auto enables for Cadence/Async policies. |
| `no_divergence_guard` | false | Disable the convergence guard. Builder shortcut: `.no_divergence_guard()`. |
| `max_overshoot` | None (auto) | Async-only overshoot cap. Builder shortcut: `.max_overshoot(N)`. |
| `lr_scale_ratio` | 1.0 | Linear LR scaling ratio. Builder shortcut: `.lr_scale_ratio(F)`. |
| `timeline` | None | `Arc<Timeline>` for profiler event injection. Builder shortcut: `.timeline(tl)`. |

### ApplyPolicy

Controls WHEN parameter averaging occurs (the interval K).

| Policy | K | Trigger | Behavior | Best for |
|--------|---|---------|----------|----------|
| `Sync` | 1 | Every batch | Average after every batch. Fast GPU waits. | Homogeneous GPUs, correctness-first |
| `Cadence` | N (ElChe) | Wall-time | Fires when slowest rank's accumulated wall time reaches anchor wall-time. Slow GPU anchors cadence. Fast GPU fills wall time. | Heterogeneous GPUs (default) |
| `Async` | Adaptive | Batch-count | Fires when all ranks complete their assigned batch counts. Overshooting is intentional: replicas explore different parameter neighborhoods, producing diversity that benefits convergence. Auto-tunes from divergence monitoring. | Maximum throughput, large models |

**Why Cadence uses wall-time but Async uses batch-count**: Cadence needs
predictable rendezvous points for the AllReduce barrier. Wall-time gives
a stable anchor tied to the slow device's actual pace. Async benefits from
letting fast devices overshoot: the slight divergence between replicas acts
like implicit exploration. Benchmark evidence shows async with wall-time
trigger produces worse convergence than batch-count trigger.

### AverageBackend

Controls HOW parameter averaging is performed. Orthogonal to policy.

| Backend | Mechanism | Memory | Blocking | Fault tolerance |
|---------|-----------|--------|----------|-----------------|
| `Nccl` | In-place AllReduce via GPU DMA | Zero extra | All GPUs sync at barrier | Abort handles unblock stuck ops |
| `Cpu` | Snapshots to coordinator, CPU average, distribute back | O(W * M) CPU RAM | No GPU ever blocks | Timeout (5s) detects dead workers |

All 6 combinations (3 policies x 2 backends) are valid. This enables A/B
testing: same model, same K, swap only the backend.

### Worker lifecycle

1. Main thread creates model on device[0], extracts initial params
2. NCCL comms initialized from main thread (`NcclComms::new()` + `split()`)
3. One thread spawned per GPU
4. Each thread: create model + optimizer from factories, copy initial params
5. Training loop: `wait_for_epoch_plan()` blocks for coordinator's `EpochPlan`, then `run_epoch_plan()` calls `train_step()` per batch
6. After all epochs (coordinator sends `Shutdown`): `send_final_snapshot()`, `report_exiting()`
7. `drain_until_shutdown()`: keeps handling control messages until coordinator sends Shutdown
8. Thread exits, NCCL comm dropped

### Coordinator lifecycle

1. Spawned as a dedicated thread
2. Sends initial epoch plans to all workers via `send_all_plans(0)`
3. Main loop: `drain_timing_blocking()` with 100us timeout
4. Each tick: `check_throttle()`, `poll_cpu_averaging()`, `drain_metrics()`
5. `drain_metrics()` triggers `try_aggregate_epochs()`: when all ranks report for an epoch, `on_epoch_aggregated()` dispatches the next epoch's plans (or `Shutdown` after the last epoch)
6. When `should_average()`: `trigger_averaging()`
7. On shutdown or all workers exited: `drain_avg_state()`, `shutdown_workers()`
8. Collects final snapshots, returns `TrainedState`

### Global epoch management

The coordinator owns epochs globally. Workers are mode-agnostic: they wait
for an `EpochPlan` from the coordinator and process it. Policy lives
entirely in the coordinator's dispatch timing.

```
EpochPlan { epoch, partition_offset, partition_size }
```

**Control flow:**

1. Coordinator sends `send_all_plans(0)` at startup (throughput-proportional if ElChe has speed hints)
2. Workers block in `wait_for_epoch_plan()`, receive `StartEpoch(plan)`, run their partition
3. Workers send `MetricsMsg` at partition end
4. Coordinator's `drain_metrics()` calls `on_rank_done()` (Auto per-rank dispatch) and `try_aggregate_epochs()` (sorted epoch processing)
5. `on_epoch_aggregated()` sends next epoch's plans (Sync/Cadence) or unblocks waiting ranks (Auto), or sends `Shutdown` when the last epoch completes

**Partition sizing:** throughput-proportional (faster GPUs get more samples) when ElChe is calibrated, equal sizes otherwise. Fixed ratios via `partition_ratios` override auto-sizing. Partitions are deterministic: all ranks share the same seed-based global permutation, with consecutive non-overlapping slices.

**Auto lookahead:** in `Async` mode, fast ranks may run 1 epoch ahead of the last globally-aggregated epoch, keeping GPUs busy while the slow rank finishes.

### Progressive dispatch

When `progressive_dispatch` is enabled (default for Cadence/Async), the
coordinator sends work in small chunks instead of full epoch partitions.
This provides continuous adaptation to throughput changes within an epoch.

Without progressive dispatch (Sync mode default), each worker receives
its full epoch partition upfront and processes it sequentially. This is
simpler and has lower coordination overhead, but cannot react to
throughput changes mid-epoch.

```rust
// Explicitly enable (auto for Cadence/Async)
.progressive_dispatch(true)

// Explicitly disable (auto for Sync)
.progressive_dispatch(false)
```

### Epoch callbacks

The `epoch_fn` callback runs at the start of each epoch inside each
worker thread, before training begins. It receives the epoch number and
a mutable reference to the `GpuWorker`:

```rust
.epoch_fn(|epoch, worker| {
    // Learning rate schedule
    let lr = 0.001 * (0.95_f64).powi(epoch as i32);
    worker.set_lr(lr);
})
```

The callback runs on every GPU thread independently. Use it for:
- Learning rate schedules (`worker.set_lr()`)
- Noise curricula (adjusting dropout or data augmentation)
- Dynamic loss weights that change per epoch
- Logging epoch transitions

**`GpuWorker` methods available in callbacks:**

| Method | Description |
|--------|-------------|
| `rank()` | This worker's rank (0-based) |
| `device()` | CUDA device for this rank |
| `local_step()` | Batches processed by this rank so far |
| `current_version()` | Latest averaging version applied |
| `current_epoch()` | Current epoch number |
| `current_lr()` | Current learning rate |
| `set_lr(f64)` | Set learning rate directly |
| `scale_lr(f64)` | Multiply current LR by a factor |
| `set_lr_scale(f64)` | Set the linear scaling multiplier |
| `set_scheduler(Arc<dyn Scheduler>)` | Replace the LR scheduler |
| `model()` | Reference to the rank-local model |

### Convergence guard

The builder includes a weight-space divergence guard that monitors
parameter drift between sync points. After each averaging event, it
measures `||params_before - params_after|| / ||params_after||` per
parameter group, producing a `DivergenceReport`.

The guard maintains a ring buffer of the last 5 divergence values
and watches for trends:

- **`Stable`**: divergence within threshold, no action needed
- **`SuppressGrowth`**: 3 consecutive rising values detected, hold
  current cadence (don't increase anchor)
- **`NudgeDown`**: divergence exceeds threshold with growth trend,
  reduce anchor to sync more frequently

```rust
// Configure the guard (default: enabled with auto threshold)
.divergence_threshold(0.05)   // custom threshold
.no_divergence_guard()        // disable entirely
```

The guard interacts with El Che's cadence: when it detects instability,
it prevents the anchor from increasing and can actively reduce it,
keeping replicas within the basin of constructive averaging.

### CPU averaging state machine

The CPU backend operates as a non-blocking 3-phase state machine:

```
Idle --> Collecting --> Computing --> Idle
         (try_recv       (thread
          per tick)       join)
```

- **Idle**: no averaging in progress
- **Collecting**: `try_recv` for worker snapshots each tick. Transitions to
  Computing when all ranks respond, or soft-aborts on timeout.
- **Computing**: background thread runs `average_params()` + divergence check.
  When done, sends `Update` to all workers.

`check_throttle()` runs every tick, even during averaging.

### Metrics pipeline

The DDP Builder provides a structured metrics pipeline for monitoring
training progress from outside the worker threads.

**Inside the train function** -- record custom scalars:

```rust
|model, batch| {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].clone(), false);
    let pred = model.forward(&input)?;
    let loss = pred.mse(&target)?.mean()?;

    // Record custom metrics (thread-local, zero overhead)
    let accuracy = compute_accuracy(&pred, &target);
    record_scalar("accuracy", accuracy);

    Ok(loss)
}
```

**Outside** -- consume aggregated epoch metrics:

```rust
let ddp = Ddp::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset)
    .batch_size(32)
    .num_epochs(100)
    .run()?;

// Non-blocking polling loop
loop {
    for metrics in ddp.poll_metrics() {
        println!(
            "epoch {} | loss={:.4} | accuracy={:.4} | {:.0}ms",
            metrics.epoch, metrics.avg_loss,
            metrics.scalars.get("accuracy").unwrap_or(&0.0),
            metrics.epoch_ms,
        );
    }
}
let state = ddp.join()?;
```

**`EpochMetrics` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | `usize` | Epoch number (0-based) |
| `avg_loss` | `f64` | Loss averaged across all ranks |
| `epoch_ms` | `f64` | Wall time for the epoch (slowest rank) |
| `scalars` | `HashMap<String, f64>` | Aggregated custom scalars (averaged across ranks) |
| `per_rank` | `Vec<HashMap<String, f64>>` | Per-rank custom scalars |
| `per_rank_throughput` | `Vec<f64>` | Per-rank batches per second |
| `per_rank_batch_share` | `Vec<f64>` | Fraction of total batches handled per rank |
| `device_indices` | `Vec<u8>` | CUDA device index for each rank |

### Monitor integration

Wire the DDP handle into a training `Monitor` for the live dashboard:

```rust
let ddp = Ddp::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset)
    .batch_size(32)
    .num_epochs(100)
    .run()?;

let mut monitor = Monitor::new(100);
ddp.setup_monitor(&mut monitor);
monitor.serve(3000)?;

// Feed metrics to the monitor
while let Some(metrics) = ddp.next_metrics() {
    let elapsed = std::time::Duration::from_millis(metrics.epoch_ms as u64);
    monitor.log(metrics.epoch, elapsed, &metrics);
}
let state = ddp.join()?;
monitor.finish();
```

`setup_monitor()` attaches the graph identity (label + structural hash),
architecture SVG, and training configuration (policy, backend, world size)
to the monitor. The dashboard shows per-GPU tabs, throughput charts, and
batch share distribution automatically.

### DdpHandle methods

| Method | Description |
|--------|-------------|
| `next_metrics()` | Block until next epoch completes, returns `Some(EpochMetrics)` or `None` when done |
| `poll_metrics()` | Non-blocking: returns all completed epoch metrics since last poll |
| `join()` | Wait for training to finish, returns `TrainedState` |
| `world_size()` | Number of GPU workers |
| `devices()` | CUDA devices used by each rank |
| `architecture_svg()` | Graph architecture SVG (if model is a Graph) |
| `setup_monitor(&self, &mut Monitor)` | Wire into live dashboard |

### TrainedState

Returned by `DdpHandle::join()`:

```rust
pub struct TrainedState {
    pub params: Vec<Tensor>,   // averaged, on CPU
    pub buffers: Vec<Tensor>,  // averaged, on CPU
}
```

On partial failure (some workers died), contains the average of surviving
workers' final snapshots. If averaging fails, falls back to the first
snapshot's tensors.

### Single-GPU fallback

With fewer than 2 CUDA devices, `DdpHandle` runs training on the main
thread. No worker threads, no coordinator, no averaging. The API is
identical: `join()` returns `TrainedState`. This means you can develop on a
laptop and deploy to a multi-GPU server with zero code changes.

---

## Strategy Guide: Start with A/B Testing

You have 6 valid configurations (3 policies x 2 backends). You don't know
which one works best for your model until you try it. That's the point:
**run a few epochs with different configs, compare loss curves, then commit
to the winner for your full training run.**

This takes minutes, not hours, and prevents you from discovering 50 epochs
in that a different config would have converged faster (or at all).

### The 3x2 matrix

| Policy | Backend | Use Case | Throughput | Convergence | Complexity |
|--------|---------|----------|------------|-------------|------------|
| Async | Nccl | **Best overall (recommended)** | Best | Best with clipping | Low |
| Cadence | Nccl | Strong second, predictable sync | Good | Good | Low |
| Sync | Nccl | Strict sync baseline | Baseline | Good | Lowest |
| Async | Cpu | Non-blocking GPUs, fault-tolerant | Good | Good | Medium |
| Cadence | Cpu | Non-blocking for heterogeneous clusters | Good | Good | Medium |
| Sync | Cpu | Strict sync without GPU barrier | Baseline | Good | Medium |

> **CPU backend: when to use it.** The CPU backend trades GPU-to-GPU DMA for
> a snapshot / CPU-average / distribute round-trip. It costs O(W * M) CPU RAM
> (W = world size, M = model size), and no GPU ever blocks on a collective
> barrier. Fault tolerance is via a 5-second timeout: a dead worker unwedges
> the coordinator instead of stalling the cluster. Throughput is competitive
> with NCCL on small models and marginally behind on large ones. Use it when
> NCCL is unavailable, when you want non-blocking GPUs, or for A/B testing
> against NCCL on the same model and seed.

### Recommended workflow

```
1. Start with Async + Nccl (El Che -- best overall in practice)
2. A/B test against Cadence + Nccl for 3-5 epochs (strong second)
3. A/B test against Sync + Nccl if you want a strict-sync baseline
4. Full training run with the winning NCCL combo
5. Swap in `AverageBackend::Cpu` if you want non-blocking GPUs or NCCL is unavailable
```

The code change between runs is one line:

```rust
// Run A -- start here
.policy(ApplyPolicy::Async).backend(AverageBackend::Nccl)

// Run B -- strong alternative
.policy(ApplyPolicy::Cadence).backend(AverageBackend::Nccl)

// Run C -- strict sync baseline
.policy(ApplyPolicy::Sync).backend(AverageBackend::Nccl)

// Run D -- CPU backend (non-blocking GPUs, fault-tolerant)
.policy(ApplyPolicy::Async).backend(AverageBackend::Cpu)
```

### Decision tree

```
Start with Async + Nccl (El Che).
  Best overall: fast GPUs overshoot, creating parameter diversity
  that benefits convergence. Auto-tunes from divergence monitoring.

Convergence not stable enough?
  --> A/B test Cadence + Nccl (strong second, more predictable sync points)

Want a strict-sync baseline?
  --> A/B test Sync + Nccl for 3-5 epochs, compare loss curves.

No NCCL available, or want non-blocking GPUs?
  --> AverageBackend::Cpu with any policy. Competitive on small models,
      marginally slower on large ones, never blocks GPUs, tolerates dead
      workers via the 5s timeout.
```

---

## A/B Testing

### Why it matters

Every model responds differently to averaging frequency and transport
timing. A config that works for a transformer may not work for a conv net.
The only way to know is to test, and floDl makes this a one-line change
instead of a rewrite.

### How it works

`ApplyPolicy` and `AverageBackend` are orthogonal. The policy determines
K (how many batches between averaging). The backend determines the
transport (GPU-to-GPU DMA vs CPU round-trip). The mathematical operation
is the same: weighted average of parameters.

All six combinations (3 policies x 2 backends) are validated. Same model,
same data, same seed. Change one knob, compare loss curves.

### Quick A/B test

```rust
// Build your base config once
let base = || {
    Ddp::builder(model_factory.clone(), optim_factory.clone(), train_fn.clone())
        .dataset(dataset.clone())
        .batch_size(32)
        .num_epochs(5)   // just enough to see the trend
        .max_grad_norm(5.0)
};

// Run A: Async + NCCL (El Che -- best overall in practice)
let a = base().policy(ApplyPolicy::Async).backend(AverageBackend::Nccl).run()?;
let state_a = a.join()?;

// Run B: Cadence + NCCL (strong second, more predictable sync)
let b = base().policy(ApplyPolicy::Cadence).backend(AverageBackend::Nccl).run()?;
let state_b = b.join()?;

// Run C: Sync + NCCL (strict sync baseline)
let c = base().policy(ApplyPolicy::Sync).backend(AverageBackend::Nccl).run()?;
let state_c = c.join()?;

// Compare: which reached the lowest loss in 5 epochs?
// For most workloads, Async + NCCL wins on loss-per-wall-second.
```

### What to compare

- **Loss at epoch N**: lower is better, obviously
- **Wall time per epoch**: Cadence should be faster than Sync on heterogeneous hardware
- **Loss per wall-second**: the real metric. A slightly higher loss in half the time often wins.

### NCCL vs CPU backend

NCCL uses hardware-level GPU-to-GPU AllReduce with implicit synchronization.
All GPUs block at the barrier, zero extra memory is required, hardware DMA
moves the tensors. Abort handles unblock stuck collectives if a worker dies
mid-op.

The CPU backend uses a snapshot / CPU-average / distribute round-trip. It
costs O(W * M) CPU RAM (W = world size, M = model size) and two GPU-to-CPU
copies per averaging event, but no GPU ever blocks on a barrier. Fault
tolerance comes from a 5-second timeout that unwedges the coordinator if a
worker goes dark.

Both backends are validated across all three policies. NCCL is typically
faster on large models (no CPU round-trip); CPU is competitive or faster on
small models (no GPU barrier) and the better choice when you need
non-blocking GPUs, fault tolerance, or NCCL is unavailable.

---

## Worked Example: ResNet-20 on CIFAR-10, A/B testable

The
[`ddp-bench/src/models/resnet_graph.rs`](https://github.com/fab2s/floDl/blob/main/ddp-bench/src/models/resnet_graph.rs)
+ [`harness.rs`](https://github.com/fab2s/floDl/blob/main/ddp-bench/src/harness.rs)
pair is the canonical end-to-end DDP recipe in the repo. It wires together
every moving part — model factory, train function, optimizer, scheduler,
`Timeline`, `Monitor`, `record_scalar`, both Graph and Builder modes —
behind a single CLI (`fdl ddp-bench --model resnet-graph --mode <mode>`)
so the same code runs in 8 backend × policy combinations without a
rewrite.

This section walks through the wiring. Use it as a template when
porting a real workload.

### 1. Model factory + train step

The model is built from a closure that takes a `Device` and returns a
`Box<dyn Module>`. The same closure is reused by every rank and every
A/B run — no shared state, no clones of GPU tensors.

```rust
fn build_model(device: Device) -> Result<Box<dyn Module>> {
    let d = device;
    let model = FlowBuilder::from(conv3x3(3, 16, 1, d)?)
        .through(BatchNorm2d::on_device(16, d)?)
        .through(ReLU)
        // 3 BasicBlocks at 16ch
        .also(res_main(16, 16, 1, d)?).through(ReLU)
        .also(res_main(16, 16, 1, d)?).through(ReLU)
        .also(res_main(16, 16, 1, d)?).through(ReLU)
        // 32ch — first block downsamples (1x1 skip via also_with)
        .also_with(downsample(16, 32, 2, d)?, res_main(16, 32, 2, d)?).through(ReLU)
        .also(res_main(32, 32, 1, d)?).through(ReLU)
        .also(res_main(32, 32, 1, d)?).through(ReLU)
        // 64ch
        .also_with(downsample(32, 64, 2, d)?, res_main(32, 64, 2, d)?).through(ReLU)
        .also(res_main(64, 64, 1, d)?).through(ReLU)
        .also(res_main(64, 64, 1, d)?).through(ReLU)
        // Head
        .through(AdaptiveAvgPool2d::new([1, 1]))
        .through(Flatten::default())
        .through(Linear::on_device(64, 10, d)?)
        .tag("logits")           // observable from the monitor
        .build()?;
    Ok(Box::new(model))
}

fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input  = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].to_dtype(DType::Int64)?, false);
    let pred   = model.forward(&input)?;

    // Per-batch training accuracy, aggregated across all DDP ranks.
    let predicted = pred.data().argmax(-1, false)?;
    let correct: f64 = predicted.eq_tensor(&target.data())?.sum()?.item()?;
    let total = target.data().shape()[0] as f64;
    flodl::record_scalar("train_acc", correct / total);

    flodl::cross_entropy_loss(&pred, &target)
}
```

`flodl::record_scalar("train_acc", ...)` works in both Graph and Builder
modes — the framework routes it to the right aggregator.

### 2. Builder mode (thread-per-GPU) — A/B testable

Wire `Timeline` + `Monitor` + per-worker scheduler factory + `record_scalar`
in one chain. Switching between Sync / Cadence / Async × NCCL / CPU is
literally two `.policy(...).backend(...)` lines.

```rust
let timeline = Timeline::new(100);   // 100ms poll interval
timeline.start();

let mut builder = Ddp::builder(
        build_model,                                   // model factory
        |params: &[Parameter]| SGD::new(params, 0.1, 0.9).weight_decay(1e-4),
        train_step,                                    // train fn
    )
    .dataset(dataset.clone())
    .batch_size(64)
    .num_epochs(200)
    .policy(ApplyPolicy::Cadence)                      // <-- A
    .backend(AverageBackend::Nccl)                     // <-- B
    .max_grad_norm(5.0)
    .lr_scale_ratio(1.0)                               // linear scaling rule
    .timeline(Arc::clone(&timeline));                  // profiler events

// Per-worker scheduler factory: each rank instantiates its own copy.
let total_steps = dataset.len() / 64 * 200;
builder = builder.scheduler(move |world_size| {
    Arc::from(MultiStepLR::new(
        0.1, &[total_steps / 2, total_steps * 3 / 4], 0.1,
    )) as Arc<dyn flodl::nn::Scheduler>
});

let handle = builder.run()?;

// Wire the live monitor (HTML dashboard + SSE).
// Monitor::new takes the total number of epochs (used for ETA + progress bars).
let mut monitor = Monitor::new(200);
handle.setup_monitor(&mut monitor);

// Stream per-epoch metrics as they land
while let Some(m) = handle.next_metrics() {
    flodl::msg!(
        "epoch {} | loss={:.4} | acc={:.3} | {:.0}ms",
        m.epoch, m.avg_loss,
        m.scalars.get("train_acc").copied().unwrap_or(0.0),
        m.epoch_ms,
    );
    monitor.log(m.epoch, Duration::from_millis(m.epoch_ms as u64), &m);
}

let state: TrainedState = handle.join()?;   // averaged params + buffers
timeline.stop();
timeline.save_html("runs/resnet-graph/cadence-nccl/timeline.html")?;
```

To produce a clean A/B comparison, capture `base()` as a closure factory:

```rust
let base = || {
    Ddp::builder(build_model, opt_factory.clone(), train_step)
        .dataset(dataset.clone())
        .batch_size(64)
        .num_epochs(5)                              // smoke run
        .max_grad_norm(5.0)
        .lr_scale_ratio(1.0)
        .timeline(Arc::clone(&timeline))            // shared profiler
};

// Three runs, one knob change each
let a = base().policy(ApplyPolicy::Async)  .backend(AverageBackend::Nccl).run()?.join()?;
let b = base().policy(ApplyPolicy::Cadence).backend(AverageBackend::Nccl).run()?.join()?;
let c = base().policy(ApplyPolicy::Sync)   .backend(AverageBackend::Nccl).run()?.join()?;
```

Same data, same seed, same model factory — only the policy/backend pair
differs. Compare the three saved timelines and per-epoch metrics to pick
a winner.

### 3. Graph mode (sync) — same wiring, fewer pieces

`Ddp::setup_with` + `DdpConfig::new().timeline(...)` gives the Graph DDP
path the same A/B-testable surface, with the user-owned training loop:

```rust
let graph = build_model(Device::CUDA(0))?;
let graph = graph.as_graph().expect("Graph required");

Ddp::setup_with(
    graph,
    move |dev| build_model(dev),
    move |params: &[Parameter]| SGD::new(params, 0.1, 0.9).weight_decay(1e-4),
    DdpConfig::new()
        .timeline(Arc::clone(&timeline)),            // event injection
)?;

// Attach the same scheduler used in builder mode
let total_steps = dataset.len() / 64 * 200;
let sched: Arc<dyn flodl::nn::Scheduler> =
    Arc::from(MultiStepLR::new(0.1, &[total_steps / 2, total_steps * 3 / 4], 0.1));
graph.set_scheduler(sched);
graph.set_lr_scale(graph.world_size() as f64);      // linear scaling

let mut monitor = Monitor::new(200);   // total_epochs
monitor.watch(graph);

for epoch in 0..200 {
    let t0 = Instant::now();
    for batch in load_epoch(&dataset, 64) {
        let loss = train_step(graph, &batch)?;
        loss.backward()?;
        graph.step()?;                              // AllReduce + opt + zero_grad + LR
    }
    monitor.log(epoch, t0.elapsed(), graph);
}
```

The cross-mode parity test (`graph_tests.rs`) guarantees that this loop
and the builder loop above produce the same LR schedule for the same
`MultiStepLR`, so A/B comparisons across modes are meaningful.

### 4. What you get for free

- **Live HTML dashboard** at `monitor.serve()`: per-rank loss curves,
  GPU utilization, VRAM, anchor/throughput evolution.
- **`timeline.save_html(...)`**: post-hoc swimlane view of CPU/GPU
  utilization, sync events, anchor changes, idle gaps.
- **`record_scalar("k", v)`**: any per-batch scalar shows up in the
  dashboard, the JSON archive, and `EpochMetrics::scalars`.
- **`flodl::msg!` / `verbose!` / `debug!` / `trace!`**: gated logging
  controlled by `fdl -v / -vv / -vvv / --quiet` or
  `FLODL_VERBOSITY=verbose`. Same code, three verbosity levels for
  development vs CI vs production.

### 5. Drive it from `fdl.yaml`

The
[`ddp-bench/fdl.yml.example`](https://github.com/fab2s/floDl/blob/main/ddp-bench/fdl.yml.example)
turns the matrix into named presets under the sub-command's `commands:`
map:

```yaml
commands:
  validate:
    description: Check convergence against structured baselines
    options: { model: all, mode: all, validate: true,
               baseline: baselines/structured.json }
  nccl-cadence:
    description: NCCL cadence for all models
    options: { model: all, mode: nccl-cadence }
```

```bash
fdl ddp-bench validate            # full sweep
fdl ddp-bench nccl-cadence -v     # one mode, verbose
```

Every run drops `training.log`, `timeline.{json,csv,html}`, and
`metrics.json` under `runs/<model>/<mode>/`. The reporter
(`fdl ddp-bench --report runs/report.md`) collates them into a
Markdown convergence table.

---

## Data Pipeline

The `DataLoader` is DDP-aware and adapts automatically to distributed
training. Understanding its modes helps get the best throughput.

### Modes

| Mode | Description | When |
|------|-------------|------|
| **Resident** | Entire dataset loaded into GPU VRAM once. Per-epoch reshuffling via GPU-side `index_select`. | Dataset fits in 75% of free VRAM |
| **Streaming** | Persistent background worker thread with async H2D on dedicated CUDA stream. Prefetch depth auto-adapts to VRAM. | Dataset too large for VRAM |
| **Distributed** | Per-device backends (each GPU independently selects resident or streaming). No lowest-common-denominator. | `Ddp::setup()` or `Graph::distribute()` |

### VRAM-aware prefetch

In streaming mode, the prefetch depth is computed automatically:

```
depth = clamp(free_vram * headroom / batch_bytes, 2, max_depth)
```

- **Bootstrap**: 4 batches at construction time (model not yet loaded)
- **epoch(0)**: re-probes VRAM after model allocation, fills to cap
- **epoch(N)**: re-probes each epoch, adapts to fragmentation
- **`vram_max_usage(0.90)`**: use up to 90% of total VRAM (default)
- **`.prefetch(n)`**: manual override, disables automatic adaptation
- **OOM fallback**: if resident mode fails with CUDA OOM, automatically
  retries with streaming mode

### Per-device backends (DDP)

When distributed across heterogeneous GPUs:

```
RTX 5060 Ti (16 GB):  resident (6 GB dataset fits easily)
GTX 1060 (6 GB):      streaming (only 2 GB free after model)
```

Each GPU independently selects the best mode. No constraint from the
smallest GPU forces the larger GPU into streaming. The gather device
(where outputs are collected) prefers the resident backend with the most
free VRAM.

### DataLoader builder reference

| Method | Default | Description |
|--------|---------|-------------|
| `.batch_size(usize)` | Required | Batch size per GPU |
| `.device(Device)` | CPU | Target device (leave as CPU for DDP) |
| `.seed(u64)` | 42 | RNG seed for shuffling (epoch-deterministic) |
| `.shuffle(bool)` | true | Enable shuffling (RandomSampler) |
| `.sampler(Box<dyn Sampler>)` | -- | Custom sampler (overrides shuffle) |
| `.prefetch(usize)` | Auto | Override auto-detected prefetch depth |
| `.vram_max_usage(f64)` | 0.90 | Max VRAM fraction for prefetch |
| `.streaming()` | Auto | Force streaming mode |
| `.names(&[&str])` | Positional | Name batch tensor positions |
| `.drop_last(bool)` | true | Drop incomplete final batch (BatchNorm safety) |

---

## NCCL Bindings

### NcclComms

Group communicator for multi-GPU collectives. RAII: destroyed on drop.

```rust
let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)])?;
comms.all_reduce(&[&tensor_a, &tensor_b], ReduceOp::Avg)?;
comms.broadcast(&[&params_0, &params_1], 0)?;  // broadcast from rank 0
```

Stream variants for overlapped communication:
```rust
comms.all_reduce_on_streams(&tensors, ReduceOp::Avg, &streams)?;
comms.broadcast_on_streams(&tensors, 0, &streams)?;
```

All operations save and restore the current CUDA device.

### NcclComms::split()

Extracts per-rank `NcclRankComm` from a group. Preferred over per-thread
`init_rank()` because `ncclCommInitRank` from worker threads corrupts CUDA
context on heterogeneous GPUs.

```rust
let group = NcclComms::new(&devices)?;
let rank_comms: Vec<NcclRankComm> = group.split()?;
// Move each NcclRankComm into its worker thread
```

### NcclRankComm

Per-rank communicator for multi-threaded DDP. `Send`, so it can be moved
into spawned threads.

```rust
// Inside a worker thread:
comm.all_reduce(&[&tensor], ReduceOp::Avg)?;
comm.all_reduce_on_stream(&[&tensor], ReduceOp::Avg, &stream)?;
```

### NcclAbortHandle

Arc-shared handle to abort a stuck communicator:

```rust
let handle = comm.abort_handle();  // Arc<NcclAbortHandle>
// In error recovery:
handle.abort()?;  // unblocks any thread stuck in AllReduce
```

After abort, the communicator's `Drop` is a no-op.

### ReduceOp

| Variant | Value | Description |
|---------|-------|-------------|
| `Sum` | 0 | Element-wise sum |
| `Prod` | 1 | Element-wise product |
| `Max` | 2 | Element-wise maximum |
| `Min` | 3 | Element-wise minimum |
| `Avg` | 4 | Element-wise average |

---

## CUDA Synchronization Primitives

### CudaEvent

Record and synchronize on CUDA streams. Used for timing and cross-stream
synchronization.

```rust
let event = CudaEvent::new(CudaEventFlags::Default)?;
event.record()?;                    // record on current stream
event.record_on(&stream)?;         // record on specific stream
event.synchronize()?;              // CPU blocks until event completes
let done = event.is_complete()?;   // non-blocking poll

// Timing between two events:
let ms = CudaEvent::elapsed_time(&start, &end)?;
```

Use `CudaEventFlags::DisableTiming` for pure synchronization (lower
overhead, but `elapsed_time` will error).

### CudaStream

Pool-managed CUDA streams. Used for overlapped compute/communication.

```rust
let stream = CudaStream::new(Device::CUDA(0), false)?;  // normal priority
stream.synchronize()?;
stream.wait_event(&event)?;    // stream waits for event before proceeding
```

### StreamGuard

RAII guard that sets a stream as current and restores the default on drop:

```rust
{
    let _guard = StreamGuard::new(&stream);
    // All CUDA ops here run on `stream`
    tensor.copy_(&source, true)?;  // non-blocking copy on this stream
}
// Default stream restored
```

---

## Troubleshooting

### NCCL init failure

**Error**: `ncclCommInitAll failed`

**Cause**: NCCL cannot establish peer-to-peer communication between devices.
Common on consumer GPUs without NVLink.

**Fix**: Check `nvidia-smi topo -m` for device connectivity. If devices
cannot communicate via NVLink or PCIe peer-to-peer, NCCL falls back to
shared memory. Ensure CUDA IPC is available. Or use
`AverageBackend::Cpu` to bypass NCCL entirely.

### Parameter count mismatch

**Error**: `GpuWorker rank N: model has M params but config has K`

**Cause**: The model factory produces a model with a different parameter
count than the initial model used to extract starting parameters.

**Fix**: Ensure `model_factory(dev)` produces an identical architecture
for every device.

### CUDA context corruption

**Error**: `CUBLAS_STATUS_EXECUTION_FAILED` or SIGABRT after NCCL init

**Cause**: `ncclCommInitRank` called from multiple threads on heterogeneous
GPUs corrupts the CUDA context.

**Fix**: Always use the init-on-main + split pattern:
```rust
let group = NcclComms::new(&devices)?;      // main thread
let rank_comms = group.split()?;            // extract per-rank
// Move rank_comms[i] into thread i
```

Never call `NcclRankComm::init_rank()` from worker threads on heterogeneous
hardware.

### NCCL deadlock (worker death)

**Error**: Training hangs indefinitely

**Cause**: One worker died mid-collective. Surviving workers are stuck in
AllReduce waiting for the dead rank.

**Fix**: `DdpHandle` handles this automatically via `NcclAbortHandle`. For
manual DDP, call `abort_handle.abort()` on all communicators when a worker
fails.

### OOM on smaller GPU

**Error**: CUDA out of memory on one device but not others

**Cause**: Heterogeneous GPUs with different VRAM. The smaller GPU cannot
fit the same batch size.

**Fix**: Use El Che (auto-enabled by `Ddp::setup()` for heterogeneous
hardware). It assigns fewer batches to the slower/smaller GPU. Or use
`Ddp::builder` with `Cadence` policy, which naturally partitions data
proportionally. The DataLoader's per-device backend selection also helps:
the large GPU can go resident while the small GPU streams.

### CPU averaging timeout

**Error**: `ddp: CPU averaging timeout, missing ranks: [1]`

**Cause**: A worker is not responding to `RequestParams` within the
timeout window (default 5 seconds).

**Fix**: Check if the worker is stuck in a long computation. Increase
the timeout via `.with_snapshot_timeout(10)` on `DdpRunConfig`, or
investigate why the worker is unresponsive. Repeated timeouts (check
`coordinator.abort_count()`) indicate a persistently sick worker.

---

Previous: [Tutorial 13: Data Loading](tutorials/13-data-loading.md) |
Next: [PyTorch Migration Guide](pytorch_migration.md)
