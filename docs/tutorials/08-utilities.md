# Training Utilities

This tutorial covers the utilities that sit around the training loop:
gradient clipping, checkpointing, weight initialization, and trend-based
training control.

> **Prerequisites**: [Training](04-training.md) introduces the
> backward/step loop. [The Graph Builder](05-graph-builder.md)
> introduces tags.

## Gradient clipping

Deep models — especially those with loops or long chains — can suffer
from exploding gradients. floDl provides two clipping strategies. Both
are called between `backward()` and `optimizer.step()`.

### clip_grad_norm

Scales all parameter gradients so that the total L2 norm does not
exceed `max_norm`. Returns the original norm before clipping.

```rust
loss.backward()?;
let orig_norm = clip_grad_norm(&params, 1.0)?;
optimizer.step()?;
```

### clip_grad_value

Clamps each individual gradient element to `[-max_val, max_val]`.

```rust
loss.backward()?;
clip_grad_value(&params, 0.5)?;
optimizer.step()?;
```

Use `clip_grad_norm` as the default — it preserves gradient direction.

## Checkpoints

Save and restore model parameters with a compact binary format.

### Saving and loading

```rust
use flodl::{save_parameters_file, load_parameters_file};

// Save
save_parameters_file("/tmp/model.fdl", &model.parameters())?;

// Load
load_parameters_file("/tmp/model.fdl", &model.parameters())?;
```

`load_parameters` validates that the parameter count, names, and shapes
match exactly. If the model architecture has changed since the checkpoint
was written, loading fails with a descriptive error.

### Details

- Parameters store their native dtype — float16 params stay f16 on disk.
- Positional checkpoints match parameters by count, name, and shape.
- The `io::Write` / `io::Read` variants (`save_parameters`, `load_parameters`)
  work with any destination: files, buffers, network connections.

### Named checkpoints (partial loading)

For transfer learning, named checkpoints match by qualified name instead
of position. This allows loading a subset of parameters from a different
model:

```rust
use flodl::*;

// Save with qualified names from a Graph
let named = model.named_parameters();
save_named_parameters_file("/tmp/model.fdl", &named)?;

// Load into a different model — only matching names transfer
let new_named = new_model.named_parameters();
let report = load_named_parameters_file("/tmp/model.fdl", &new_named)?;

println!("loaded:  {:?}", report.loaded);   // matched and loaded
println!("skipped: {:?}", report.skipped);  // in checkpoint, not in model
println!("missing: {:?}", report.missing);  // in model, not in checkpoint
```

Names are `"prefix/param_name"` where the prefix is the tag name if
tagged (`"encoder/weight"`) or the node ID if not (`"linear_1/weight"`).
Shape mismatches on a matched name produce an error (not a silent skip).

### Freezing transferred parameters

After partial loading, freeze the transferred params and train only the
new ones:

```rust
for (name, param) in &new_named {
    if report.loaded.contains(name) {
        param.freeze()?;
    }
}

// Only unfrozen params get gradients — optimizer skips the rest
let fresh: Vec<Parameter> = new_named.iter()
    .filter(|(_, p)| !p.is_frozen())
    .map(|(_, p)| p.clone())
    .collect();
let mut opt = Adam::new(&fresh, 1e-3);
```

### Periodic checkpoints during training

```rust
for epoch in 0..num_epochs {
    // ... training loop ...

    if (epoch + 1) % 10 == 0 {
        let path = format!("/tmp/checkpoint_epoch_{}.fdl", epoch + 1);
        save_parameters_file(&path, &params)?;
    }
}
```

## Weight initialization

floDl modules use sensible defaults — `Linear` initializes weights with
Kaiming uniform (suitable for ReLU) and bias with uniform. But you can
override this when needed.

### Built-in initializers

| Function | Distribution | Best for |
|----------|-------------|----------|
| `kaiming_uniform(shape, fan_in, device)` | U(-bound, bound) | ReLU activations |
| `kaiming_normal(shape, fan_in, device)` | N(0, std) | ReLU activations |
| `xavier_uniform(shape, fan_in, fan_out, device)` | U(-bound, bound) | Sigmoid / Tanh |
| `xavier_normal(shape, fan_in, fan_out, device)` | N(0, std) | Sigmoid / Tanh |

Note: `kaiming_uniform` and `kaiming_normal` are available in the `nn::init` module
but are not re-exported at the crate root. Use `flodl::nn::init::kaiming_uniform(...)`.

### Custom initialization

Replace parameter data after constructing the module:

```rust
let layer = Linear::new(128, 64)?;

// Re-initialize weight with Xavier normal.
let w = xavier_normal(&[64, 128], 128, 64, Device::CPU)?;
layer.parameters()[0].set_data(&w);
```

## LR Scheduling

Schedulers are pure LR calculators, decoupled from the optimizer:

```rust
use flodl::*;

// Step decay: multiply by gamma every step_size steps
let scheduler = StepDecay::new(0.01, 30, 0.1);  // base_lr, step_size, gamma

// Cosine annealing
let scheduler = CosineScheduler::new(0.001, 1e-6, 100);  // base_lr, min_lr, total_steps

// Warmup wrapper (composes with any scheduler)
let inner = CosineScheduler::new(0.001, 1e-6, 100);
let scheduler = WarmupScheduler::new(inner, 0.001, 10);  // inner, target_lr, warmup_steps

// Reduce on plateau
let mut scheduler = PlateauScheduler::new(0.001, 5, 0.1, 1e-6);  // base_lr, patience, factor, min_lr
```

Step-based schedulers implement the `Scheduler` trait:

```rust
pub trait Scheduler {
    fn lr(&self, step: usize) -> f64;
}
```

`PlateauScheduler` is reactive (driven by metrics, not step count) and uses
`observe(metric)` / `lr()` instead.

## Trend-based training control

After collecting metrics and flushing epoch summaries (see
[Training — Observing Training](04-training.md#observing-training)),
query trends to make training decisions.

### Trend queries

Each `flush` adds one data point (the epoch mean) to the tag's history.
`trend` returns a queryable view:

```rust
let trend = g.trend("loss");
trend.latest();              // most recent epoch value
trend.mean();                // mean across all flushed epochs
trend.slope(5);              // OLS slope over last 5 epochs
trend.improving(5);          // is slope negative? (loss decreasing)
trend.stalled(5, 1e-4);     // is |slope| below tolerance?
trend.converged(5, 1e-5);   // is variance below tolerance?
```

### LR decay on plateau

```rust
if g.trend("loss").stalled(10, 1e-4) {
    // reduce learning rate
}
```

### Early stopping

```rust
if g.trend("loss").converged(5, 1e-5) {
    break;
}
```

### Recording external metrics

Losses computed outside the graph can be injected with `record`:

```rust
for epoch in 0..num_epochs {
    for (input, target) in &batches {
        let pred = g.forward(&input)?;
        let loss = cross_entropy_loss(&pred, &target)?;

        g.collect(&["hidden"])?;              // from graph tag
        g.record_scalar("loss", loss.item()?);  // external scalar
    }
    g.flush(&["hidden", "loss"]);
}
```

### Trend groups

When using `tag_group` with `split`, `trends` expands the group for
aggregate queries:

```rust
let g = FlowBuilder::from(encoder)
    .split(modules![head_a, head_b, head_c]).tag_group("head")
    .merge(MergeOp::Mean)
    .build()?;

// After training with collect/flush...
let tg = g.trends(&["head"]);  // expands to head_0, head_1, head_2
if tg.all_improving(5) {
    println!("all heads improving");
}
println!("mean slope: {:.4}", tg.mean_slope(5));
```

### ETA and timing

```rust
for epoch in 0..total_epochs {
    // ... training ...
    g.flush(&["loss"]);

    println!(
        "epoch {}  loss={:.4}  ETA {}",
        epoch + 1,
        g.trend("loss").latest(),
        format_duration(g.eta(total_epochs)),
    );
}
```

### Training curves

Export metrics as self-contained HTML or CSV:

```rust
g.plot_html("training.html", &["loss", "head"])?;
g.export_trends("metrics.csv", &["loss"])?;
g.write_log("training.log", total_epochs, &["loss"])?;
```

## Putting it together

A complete training script using clipping, checkpoints, trends, and
scheduling:

```rust
// Build model with tagged sections.
let g = FlowBuilder::from(Linear::new(4, 64)?).tag("encoder")
    .through(GELU)
    .through(Linear::new(64, 64)?).tag("body")
    .through(GELU)
    .through(Linear::new(64, 2)?).tag("head")
    .build()?;

// Set up training.
let params = g.parameters();
let optimizer = Adam::new(&params, 0.001);
let scheduler = CosineScheduler::new(0.001, 1e-6, 100);
g.set_training(true);

for epoch in 0..100 {
    for (input, target) in &batches {
        let input = Variable::new(input.clone(), true);
        let target = Variable::new(target.clone(), false);

        optimizer.zero_grad();
        let output = g.forward(&input)?;
        let loss = mse_loss(&output, &target)?;
        loss.backward()?;

        clip_grad_norm(&params, 1.0)?;
        optimizer.step()?;

        g.collect(&["head"])?;
        g.record_scalar("loss", loss.item()?);
    }
    g.flush(&["head", "loss"]);
    g.end_epoch();
    optimizer.set_lr(scheduler.lr(epoch));

    // Early stop when converged.
    if g.trend("loss").converged(5, 1e-5) {
        break;
    }
}

// Save.
g.set_training(false);
save_parameters_file("/tmp/model.fdl", &params)?;
```

---

Next: [Tutorial 9: Training Monitor](09-monitor.md)

Previous tutorials: [07-Visualization](07-visualization.md) |
[06-Advanced Graphs](06-advanced-graphs.md) |
[05-Graph Builder](05-graph-builder.md) |
[04-Training](04-training.md) |
[03-Modules](03-modules.md) |
[02-Autograd](02-autograd.md) |
[01-Tensors](01-tensors.md)
