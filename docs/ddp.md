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
Provides `ApplyPolicy` x `AverageBackend` for fine-grained control and
A/B testing. Best for non-Graph modules or when you need maximum
configurability.

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
    .max_anchor(Some(200));    // gradient staleness cap

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
- EMA-smoothed `ms_per_batch` per rank (alpha = error-scaled, 0.1-0.5)
- Dead-zone hysteresis: ignores jitter < 5% of current estimate
- Anchor auto-tune: if sync overhead > target, increase anchor aggressively;
  if overhead < target/2, decrease gradually

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
| `.checkpoint_every(usize)` | No | None | Checkpoint interval |
| `.checkpoint_fn(Fn)` | No | None | Checkpoint callback |

### ApplyPolicy

Controls WHEN parameter averaging occurs (the interval K).

| Policy | K | Behavior | Best for |
|--------|---|----------|----------|
| `Sync` | 1 | Average after every batch. Fast GPU waits. | Homogeneous GPUs, correctness-first |
| `Cadence` | N (ElChe) | Slow GPU anchors cadence. Fast GPU fills wall time. | Heterogeneous GPUs (default) |
| `Async` | Adaptive | Auto-tunes from divergence monitoring. Starts at K=1. | Maximum throughput, large models |

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
5. Training loop: `run_epoch()` calls `train_step()` per batch
6. After all epochs: `send_final_snapshot()`, `report_exiting()`
7. `drain_until_shutdown()`: keeps handling control messages until coordinator sends Shutdown
8. Thread exits, NCCL comm dropped

### Coordinator lifecycle

1. Spawned as a dedicated thread
2. Main loop: `drain_timing_blocking()` with 100us timeout
3. Each tick: `check_throttle()`, `poll_cpu_averaging()`, `drain_metrics()`
4. When `should_average()`: `trigger_averaging()`
5. On shutdown or all workers exited: `drain_avg_state()`, `shutdown_workers()`
6. Collects final snapshots, returns `TrainedState`

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

## Strategy Guide

### The 3x2 matrix

| Policy | Backend | Use Case | Throughput | Convergence | Complexity |
|--------|---------|----------|------------|-------------|------------|
| Sync | Nccl | Homogeneous GPUs, small models | Baseline | Best | Lowest |
| Sync | Cpu | Debugging, validation, no NCCL | Lower | Best | Low |
| Cadence | Nccl | **Heterogeneous GPUs (recommended)** | Good | Good | Low |
| Cadence | Cpu | Heterogeneous + no NVLink | Moderate | Good | Low |
| Async | Nccl | Large models, expensive batches | Best | Monitor needed | Moderate |
| Async | Cpu | Research, max independence | Good | Monitor needed | Moderate |

### Decision tree

```
All GPUs same model and VRAM?
  YES --> Sync + Nccl (simplest, best convergence)
  NO  --> Mixed hardware?
          YES --> Cadence + Nccl (ElChe handles speed differences)
                  Large model or expensive batches?
                    YES --> Async + Nccl (monitor loss curves)
Need to validate without NCCL?
  --> Run the same config with AverageBackend::Cpu. Compare loss curves.
```

---

## A/B Testing

### Why it works

`ApplyPolicy` and `AverageBackend` are orthogonal. The policy determines
K (how many batches between averaging). The backend determines the
transport (GPU-to-GPU DMA vs CPU round-trip). The mathematical operation
is the same: weighted average of parameters.

This means you can run the same model with the same policy and compare
backends. If loss curves match, the cheaper backend is validated for your
workload.

### Protocol

1. Train with `Cadence + Nccl` for N epochs. Record final loss and
   wall time. Use a fixed seed for reproducibility.
2. Same seed, same data, same model, same policy. Change only
   `.backend(AverageBackend::Cpu)`.
3. Train again. Compare loss curves.

### Interpretation

- **Curves match within tolerance**: CPU backend is validated. Use it when
  NVLink is unavailable, for debugging, or when NCCL is problematic.
- **Curves diverge**: the blocking behavior of NCCL produces different
  gradient accumulation patterns that matter for your model. Stick with
  NCCL, or increase averaging frequency (lower K).

### When CPU can match NCCL

Both backends compute the same weighted average of the same parameters.
The difference is timing: NCCL blocks all GPUs at the barrier, so all
replicas see the averaged parameters at exactly the same training step.
CPU averaging is non-blocking, so replicas may process 1-2 extra batches
with stale parameters before the update arrives.

For most models, this difference is negligible. For models with sharp loss
landscapes or very small learning rates, the synchronization point matters.

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

Previous: [Tutorial 12: DDP Builder](tutorials/12-async-ddp.md) |
Next: [PyTorch Migration Guide](pytorch_migration.md)
