# Tutorial 11: Multi-GPU Training

Scale your Graph-based model to multiple GPUs with one line of code.
The training loop stays identical: `Ddp::setup()` handles replication,
gradient sync, and optimizer management transparently.

> **Prerequisites**: [Training](04-training.md) and
> [Graph Builder](05-graph-builder.md). Requires 2+ CUDA GPUs at runtime
> (works on single GPU/CPU with no code changes).

> **Time**: ~20 minutes.

## The one-liner: Ddp::setup()

```rust
use flodl::*;

// Build your model as usual
let model = FlowBuilder::from(Linear::new(784, 256)?)
    .through(ReLU::new())
    .through(Linear::new(256, 10)?)
    .label("classifier")
    .build()?;

// One call: detect GPUs, replicate, set optimizer, enable training
Ddp::setup(&model, &builder, |p| Adam::new(p, 0.001))?;

// Training loop -- identical for 1 or N GPUs
for epoch in 0..100 {
    for batch in &dataset {
        let input = Variable::new(batch[0].clone(), false);
        let target = Variable::new(batch[1].clone(), false);
        let loss = model.forward(&input)?.mse(&target)?.mean()?;
        model.step()?;  // AllReduce + sync + optimizer + zero_grad
    }
}
```

`Ddp::setup()` prints hardware diagnostics to stderr:
```
  ddp: 2 GPUs (heterogeneous) | RTX 5060 Ti (16.0 GB) | GTX 1060 (6.0 GB)
```

On a single GPU or CPU, it still sets the optimizer and training mode.
Your training loop needs zero conditional logic.

### PyTorch comparison

In PyTorch, multi-GPU requires process groups, environment variables,
`torchrun`, and a `DistributedSampler`:

```python
# PyTorch: 8+ lines of setup + torchrun launcher
dist.init_process_group("nccl")
model = DDP(model.to(rank))
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler)
```

In floDl:
```rust
// floDl: one line, no process groups, no torchrun
Ddp::setup(&model, &builder, |p| Adam::new(p, 0.001))?;
```

## What happens under the hood

When `Ddp::setup()` detects 2+ CUDA devices:

1. **Replicate**: creates a model replica on each GPU via the builder
   closure you provided when constructing the Graph
2. **Broadcast**: copies parameters from GPU 0 to all replicas
3. **Set optimizer**: creates a per-replica optimizer via your factory
4. **Training mode**: enables dropout, BatchNorm training stats

Each call to `model.step()`:

1. **AllReduce** gradients across all replicas (NCCL, in-place)
2. **Sync buffers** (BatchNorm running mean/var)
3. **Optimizer step** on each replica independently
4. **Zero gradients**

The forward pass scatters the input batch across GPUs (each gets a shard),
forwards in parallel, and gathers outputs. Gradients flow back through
cross-device transfers via libtorch autograd.

## DataLoader integration

For the full training experience, use `DataLoader` with the Graph:

```rust
let loader = DataLoader::from_batch_dataset(dataset)
    .batch_size(32)
    .names(&["image", "label"])
    .build()?;

// Wire the loader: "image" maps to the graph's input port
model.set_data_loader(loader, "image");

// Epoch iteration handles per-GPU data distribution
for batch in model.epoch(0) {
    let batch = batch?;
    let loss = model.forward_batch(&batch)?;
    model.step()?;
}
```

When distributed, `set_data_loader()` creates per-device data backends:

- Each GPU independently selects **resident** (dataset fits in VRAM,
  loaded once, reshuffled via GPU-side `index_select`) or **streaming**
  (prefetch worker with async H2D on a dedicated CUDA stream)
- A 16 GB GPU can go resident while a 6 GB GPU streams. No
  lowest-common-denominator constraint.
- **Presharded forward**: each replica forwards its local shard with zero
  cross-device input transfer. Outputs are gathered to the gather device.

## Heterogeneous GPUs: El Che

### The problem

Traditional DDP forces all GPUs to synchronize after every batch. If your
RTX 5060 Ti processes a batch in 10ms and your GTX 1060 takes 25ms, the
fast GPU idles 60% of the time.

### The solution: El Che

Named after Che Guevara's marching principle: "the column marches at the
slowest one's pace." The slow device anchors the sync cadence, and the
fast device processes more batches between sync points.

`Ddp::setup()` detects heterogeneous hardware automatically and enables
El Che. No configuration needed for the common case.

### How it works

- The slow GPU processes `anchor` batches (default 10)
- The fast GPU processes `round(anchor * speed_ratio)` batches
- Speed ratios are discovered from CudaEvent timing after the first sync
- The anchor auto-tunes to keep AllReduce overhead below 10% of compute

### Explicit configuration

For manual control, use `Ddp::setup_with()` with `DdpConfig`:

```rust
let config = DdpConfig::new()
    .speed_hint(1, 0.4)         // GPU 1 is ~40% speed of GPU 0
    .overhead_target(0.10)      // AllReduce < 10% of compute
    .max_anchor(Some(200));     // gradient staleness cap

Ddp::setup_with(&model, &builder, |p| Adam::new(p, 0.001), config)?;
```

`speed_hint` is optional and self-corrects after the first timing report.
Use it to avoid a slow first few batches when the speed difference is known.

### Weighted gradient averaging

When batch counts are unequal, each replica's gradient is scaled by its
contribution before AllReduce Sum. The result is the mathematically
correct mean gradient regardless of per-device batch counts:

```
weight[rank] = count[rank] / sum(counts)
gradient_avg = sum(weight[rank] * gradient[rank])
```

## Auto-balancing

The auto-balancer measures per-GPU throughput and adjusts batch
distribution:

- **CudaEvent-based timing**: zero overhead (async GPU recording, no CPU sync)
- **EMA throughput**: exponentially smoothed samples/ms per device (alpha=0.3)
- **Chunk ratios**: after 10 calibration steps with equal splits, ratios
  are recomputed proportional to measured throughput. Re-evaluated every
  50 steps.
- **Starvation guard**: `MIN_CHUNK_RATIO` (5%) prevents any GPU from
  receiving zero work

Query the current state:
```rust
let ratios = model.chunk_ratios();     // e.g., [0.7, 0.3]
let throughput = model.throughput();    // per-device samples/ms
```

## Dashboard integration

When using the training monitor, multi-GPU metrics are visible
automatically:

- **Per-GPU tabs**: VRAM usage, utilization, throughput, batch share
- **GPU Overview card**: compact row per GPU with VRAM bar and throughput
- **Fastest/slowest highlighting**: fastest GPU green, slowest yellow

No extra configuration. The monitor collects `GpuSnapshot` (hardware)
and `GpuMetrics` (DDP throughput, chunk ratio) each sample.

## Manual DDP: Ddp::wrap()

For training patterns that need explicit control over when gradients
sync (GAN discriminator vs generator, RL actor vs critic, progressive
growing):

```rust
let ddp = Ddp::wrap(&[&model], &devices)?;

// Explicit sync
ddp.sync_params()?;

for batch in &dataset {
    let loss = model.forward(&batch)?;
    loss.backward()?;

    // Sync gradients when YOU decide
    ddp.all_reduce_gradients()?;
    ddp.sync_buffers()?;

    optimizer.step()?;
    optimizer.zero_grad();
}
```

With El Che (weighted averaging):
```rust
ddp.weighted_all_reduce_gradients(&batch_counts)?;
```

## Quick reference

### Ddp methods

| Method | Description |
|--------|-------------|
| `Ddp::setup(&model, &builder, optim_fn)` | One-liner: detect, distribute, set optimizer |
| `Ddp::setup_with(..., config)` | Same with explicit DdpConfig |
| `Ddp::wrap(&[&model], &devices)` | Manual coordinator |
| `Ddp::is_heterogeneous()` | True if GPU models differ |
| `.sync_params()` | Broadcast params from rank 0 |
| `.all_reduce_gradients()` | AllReduce(Avg) all gradients |
| `.weighted_all_reduce_gradients(&counts)` | Weighted AllReduce for El Che |
| `.sync_buffers()` | Broadcast buffers from rank 0 |
| `.world_size()` | Number of GPUs |
| `.devices()` | Device list |

### Graph methods (DDP-aware)

| Method | Description |
|--------|-------------|
| `model.distribute(builder)` | Create replicas on all GPUs |
| `model.set_optimizer(factory)` | Per-replica optimizers |
| `model.step()` | AllReduce + sync + optimizer + zero_grad |
| `model.world_size()` | Number of GPUs (1 if not distributed) |
| `model.is_distributed()` | True if multi-GPU |
| `model.chunk_ratios()` | Per-GPU batch share |
| `model.throughput()` | Per-GPU EMA throughput |
| `model.has_el_che()` | True if El Che is active |
| `model.configure_el_che(config)` | Set El Che parameters |
| `model.set_data_loader(loader, input)` | Attach DataLoader |
| `model.epoch(n)` | Distributed epoch iterator |
| `model.forward_batch(&batch)` | Batch-aware forward |

### DdpConfig

| Method | Default | Description |
|--------|---------|-------------|
| `.speed_hint(rank, ratio)` | None | Initial speed estimate |
| `.overhead_target(f64)` | 0.10 | AllReduce overhead ceiling |
| `.max_anchor(Option<usize>)` | None | None=auto, Some(0)=disable El Che |

---

Previous: [Graph Tree](10-graph-tree.md) |
Next: [DDP Builder](12-async-ddp.md)
