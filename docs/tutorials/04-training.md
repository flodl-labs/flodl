# Tutorial 4: Training

This tutorial puts everything together: loss functions, optimizers, gradient
clipping, mixed precision training, and the training loop. It builds on
[Tutorial 3: Modules](03-modules.md).

## Loss Functions

```rust
use flodl::*;

// Mean Squared Error: mean((pred - target)^2)
let loss = mse_loss(&pred, &target)?;

// Cross-Entropy from raw logits (not probabilities).
// pred: [batch, classes] logits.
// target: [batch] class indices (Int64) or [batch, classes] one-hot/soft labels.
let loss = cross_entropy_loss(&logits, &target)?;

// Binary Cross-Entropy with logits
let loss = bce_with_logits_loss(&pred, &target)?;

// L1 Loss (Mean Absolute Error)
let loss = l1_loss(&pred, &target)?;

// Smooth L1 Loss (Huber)
let loss = smooth_l1_loss(&pred, &target, 1.0)?;

// KL Divergence
let loss = kl_div_loss(&log_pred, &target)?;
```

All return a scalar `Variable` ready for `backward()`.

## Optimizers

All optimizers implement the `Optimizer` trait:

```rust
pub trait Optimizer {
    fn step(&mut self) -> Result<()>;
    fn zero_grad(&self);
    fn set_lr(&mut self, lr: f64);
    fn set_group_lr(&mut self, group: usize, lr: f64);  // per-group LR
}
```

### SGD

```rust
// SGD with momentum
let optimizer = SGD::new(&params, 0.01, 0.9);
```

### Adam

```rust
// Default betas (0.9, 0.999), eps=1e-8
let optimizer = Adam::new(&params, 0.001);
```

### AdamW

Adam with decoupled weight decay:

```rust
let optimizer = AdamW::new(&params, 0.001, 0.01);
```

### Fused CUDA Optimizers

On CUDA, both `Adam` and `AdamW` automatically use `_fused_adamw_` -- a
single multi-tensor kernel that updates all parameters, gradients, and
moment buffers in one launch. A naive implementation would require 4N
separate kernels (one each for momentum update, variance update, bias
correction, and parameter update, per parameter). The fused path reduces
this to a single kernel launch for all parameters in a group.

This is completely automatic. `Adam::new()` and `AdamW::new()` use the
fused path whenever parameters live on CUDA. No API changes are needed.

The fused kernel also exposes `grad_scale` and `found_inf` tensor
parameters internally, which `GradScaler` uses for integrated mixed
precision training (see [Mixed Precision Training](#mixed-precision-training)
below).

## Gradient Clipping

Prevent exploding gradients by clipping after backward and before the
optimizer step:

```rust
// Scale gradients so total L2 norm <= max_norm
clip_grad_norm(&params, 1.0)?;

// Clamp each gradient element to [-max_val, max_val]
clip_grad_value(&params, 0.5)?;
```

Under the hood, `clip_grad_norm` uses `_foreach_norm` + `_foreach_mul_`
internally -- two kernels total regardless of the number of parameters,
instead of 2N kernels with a naive per-parameter approach. This is
particularly beneficial on CUDA where kernel launch overhead dominates
for small per-parameter operations.

## Device Placement

By default, all tensors and parameters live on CPU. To train on CUDA, use
`move_to_device` on the graph.

### Moving the model

```rust
let model = build_model()?;

if flodl::cuda_available() {
    model.move_to_device(Device::CUDA(0));
}

// Create optimizer AFTER move_to_device.
let params = model.parameters();
let optimizer = Adam::new(&params, 0.001);
```

## The Training Loop

The standard pattern is: **forward -> loss -> zero_grad -> backward -> clip -> step**.

```rust
model.train();

for (input_t, target_t) in &batches {
    let input = Variable::new(input_t.clone(), true);
    let target = Variable::new(target_t.clone(), false);

    // 1. Forward
    let pred = model.forward(&input)?;

    // 2. Loss
    let loss = mse_loss(&pred, &target)?;

    // 3. Zero gradients
    optimizer.zero_grad();

    // 4. Backward
    loss.backward()?;

    // 5. Clip gradients
    clip_grad_norm(&params, 1.0)?;

    // 6. Update parameters
    optimizer.step()?;
}
```

## Observing Training

Tag the nodes you want to monitor when building the graph:

```rust
let model = FlowBuilder::from(Linear::new(2, 16)?)
    .through(GELU)
    .through(Linear::new(16, 2)?).tag("output")
    .build()?;
```

### Collect and Flush

For epoch-level metrics, collect scalar values during the batch loop and
flush at epoch boundaries:

```rust
for epoch in 0..num_epochs {
    for (input, target) in &batches {
        let pred = model.forward(&Variable::new(input.clone(), true))?;
        let loss = mse_loss(&pred, &Variable::new(target.clone(), false))?;

        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;

        model.collect(&["output"])?;            // from graph tag
        model.record_scalar("loss", loss.item()?);  // external metric
    }
    model.flush(&["output", "loss"]);           // batch mean -> epoch history
    model.end_epoch();
}
```

`collect` appends the scalar value of each tagged node to a batch buffer.
`record` pushes raw `f64` values into the same buffer. `flush` computes
the mean, stores it in epoch history, and clears the buffer.

## Stateful Graphs — detach_state

Call `detach_state()` between training steps to sever autograd references
held by the graph. It detaches:

- **Forward-reference state buffers** (recurrent state carried between calls)
- **Tagged outputs** (Variables captured by `tag()` for observation)
- **Module internal state** (e.g., recurrent hidden state in custom modules)

```rust
model.train();

for (input_t, target_t) in &batches {
    let input = Variable::new(input_t.clone(), true);
    let target = Variable::new(target_t.clone(), false);

    let pred = model.forward(&input)?;
    let loss = mse_loss(&pred, &target)?;

    optimizer.zero_grad();
    loss.backward()?;
    clip_grad_norm(&params, 1.0)?;

    model.detach_state();  // break gradient chains on state buffers + tagged outputs
    optimizer.step()?;
}
```

**When is it needed?** For any graph with forward references (`using("x")`
before `tag("x")`) it is mandatory. For graphs that use `tag()` for
observation, it prevents tagged output Variables from holding stale
autograd graph references between batches.

## Parameter Groups

All optimizers support per-group learning rates via a builder API:

```rust
let mut opt = Adam::with_groups()
    .group(&scan_params, 1e-3)     // group 0: high LR
    .group(&read_params, 1e-5)     // group 1: low LR
    .build();

// Adjust one group
opt.set_group_lr(1, 1e-4);

// Adjust all groups at once
opt.set_lr(1e-3);
```

`Adam::new(&params, lr)` still works for single-group usage. `SGD` and
`AdamW` have the same builder pattern (`SGD::with_groups(momentum)`,
`AdamW::with_groups(weight_decay)`).

## Parameter Freezing

Freeze parameters to disable gradient tracking — useful for transfer
learning:

```rust
for param in &encoder_params {
    param.freeze()?;   // no gradients will accumulate
}

// Later, unfreeze for fine-tuning:
for param in &encoder_params {
    param.unfreeze()?;
}

// Check status:
if param.is_frozen() { /* ... */ }
```

Frozen parameters are automatically skipped by optimizers (they produce
no gradient). Freezing works through `Rc<RefCell>` — a freeze is visible
everywhere the parameter is referenced.

## Checkpoints

Save and restore model parameters using named checkpoints:

```rust
// Save — includes all parameters, buffers, and structural hash
model.save_checkpoint("/tmp/model.fdl")?;

// Load — validates architecture, returns a report
let report = model.load_checkpoint("/tmp/model.fdl")?;
```

For custom destinations (network, in-memory buffer), the lower-level API is available:

```rust
let named = model.named_parameters();
let buffers = model.named_buffers();
let hash = Some(model.structural_hash());
save_checkpoint(&mut writer, &named, &buffers, hash)?;
let report = load_checkpoint(&mut reader, &named, &buffers, hash)?;
```

### Partial loading (transfer learning)

Named checkpoints match by qualified name, so you can load a subset of parameters from a different model using `Graph::named_parameters()`:

```rust
// Save with qualified names
model.save_checkpoint("/tmp/model.fdl")?;

// Load into a different model — matches by name
// Use the lower-level API with None hash to skip architecture validation
let new_named = new_model.named_parameters();
let new_buffers = new_model.named_buffers();
let report = load_checkpoint_file("/tmp/model.fdl", &new_named, &new_buffers, None)?;

println!("loaded: {:?}", report.loaded);   // matched and loaded
println!("skipped: {:?}", report.skipped); // in checkpoint, not in model
println!("missing: {:?}", report.missing); // in model, not in checkpoint

// Freeze transferred params, train the rest
for (name, param) in &new_named {
    if report.loaded.contains(name) {
        param.freeze()?;
    }
}
```

Named parameters use the tag name as prefix when available (`"encoder/weight"`),
otherwise the node ID (`"linear_1/weight"`). Shape mismatches on matched names
are errors.

## LR Scheduling

Schedulers compute learning rates without owning the optimizer — they are
pure calculators:

```rust
let scheduler = CosineScheduler::new(0.001, 1e-6, 100);  // base_lr, min_lr, total_steps

for step in 0..100 {
    let lr = scheduler.lr(step);
    optimizer.set_lr(lr);
}
```

Compose with warmup:

```rust
let inner = CosineScheduler::new(0.001, 1e-6, 100);
let scheduler = WarmupScheduler::new(inner, 0.001, 10);  // inner, target_lr, warmup_steps
```

## Mixed Precision Training

Mixed precision training runs eligible operations (matmul, convolutions,
linear layers) in a reduced-precision dtype (typically `Float16` or
`BFloat16`) while keeping numerically sensitive operations (losses, norms,
softmax) in full `Float32`. On GPUs with Tensor Cores (RTX 30xx, RTX 40xx,
RTX 50xx), this can deliver up to 3x speedup with minimal accuracy impact.

### Autocast

The `AutocastGuard` RAII guard enables automatic dtype dispatch for the
duration of its lifetime. The `autocast()` closure helper provides a
convenient scoped interface:

```rust
use flodl::*;

// RAII guard style
let _amp = AutocastGuard::new(DType::Float16);
let output = model.forward(&input)?;  // matmul dispatches to fp16
let loss = mse_loss(&output, &target)?;  // stays fp32
drop(_amp);

// Closure style (preferred)
let loss = autocast(DType::Float16, || {
    let output = model.forward(&input)?;
    mse_loss(&output, &target)
})?;

// Query whether autocast is active
if is_autocast_enabled() {
    // inside an autocast region
}
```

### GradScaler

Half-precision gradients can underflow to zero. `GradScaler` solves this
by scaling the loss before backward (inflating gradient magnitudes), then
unscaling gradients before the optimizer step. It dynamically adjusts the
scale factor -- growing it when gradients stay finite, backing off when
inf/nan is detected.

```rust
let mut scaler = GradScaler::new();
// Initial scale: 65536, growth: 2x, backoff: 0.5x, interval: 2000 steps
```

The `step` method handles unscaling, inf/nan checking, and the optimizer
step in a single call. It returns `true` if the step was taken, or `false`
if it was skipped due to non-finite gradients:

```rust
let stepped = scaler.step(&params, &mut || optimizer.step())?;
scaler.update();  // adjust scale factor -- call after every step()
```

### Complete Mixed Precision Loop

```rust
let mut scaler = GradScaler::new();

model.train();
for (x, y) in &batches {
    let input = Variable::new(x.clone(), false);
    let target = Variable::new(y.clone(), false);

    // Forward under autocast -- eligible ops run in fp16
    let loss = autocast(DType::Float16, || {
        let pred = model.forward(&input)?;
        mse_loss(&pred, &target)
    })?;

    // Scale loss and backward
    let scaled = scaler.scale(&loss)?;
    optimizer.zero_grad();
    scaled.backward()?;

    // Unscale gradients, check for inf/nan, clip, and step
    clip_grad_norm(&params, 1.0)?;
    let stepped = scaler.step(&params, &mut || optimizer.step())?;
    scaler.update();
}
```

### Manual Dtype Conversion

For cases where you need explicit control over parameter dtypes rather
than relying on autocast:

```rust
// Cast all parameters to fp16
cast_parameters(&params, DType::Float16);

// Cast back to fp32
cast_parameters(&params, DType::Float32);
```

Parameters already at the target dtype are skipped (no-op).

## Eval Mode

Switch to eval mode for inference:

```rust
model.eval();
no_grad(|| {
    let output = model.forward(&input)?;
    // No graph built, no gradient tracking overhead.
    Ok(output)
})?;
```

## Training Housekeeping

The graph tracks step and epoch counts for schedulers and observation:

```rust
model.end_step();   // increment step counter
model.end_epoch();  // increment epoch counter, reset step
```

## Reproducibility

For deterministic training, seed both libtorch and CPU-side RNG at the start:

```rust
use flodl::*;

fn main() -> Result<()> {
    // Seed libtorch: controls rand, randn, dropout, weight init
    manual_seed(42);

    // CPU-side RNG: controls data shuffling, augmentation
    let mut rng = Rng::seed(42);

    // Build model AFTER seeding — weight initialization uses the seed
    let model = FlowBuilder::from(Linear::new(2, 16)?)
        .through(GELU)
        .through(Linear::new(16, 2)?)
        .build()?;

    // Shuffle training data deterministically
    let mut indices: Vec<usize> = (0..dataset_len).collect();
    rng.shuffle(&mut indices);

    // ...
    Ok(())
}
```

`manual_seed` covers all libtorch operations (tensor creation, dropout masks,
weight initialization). `Rng` covers everything else (data loading order,
augmentation parameters). Together they give full reproducibility.

## Complete Example

```rust
use flodl::*;

fn main() -> Result<()> {
    manual_seed(42);

    // Build model.
    let model = FlowBuilder::from(Linear::new(2, 16)?)
        .through(GELU)
        .through(LayerNorm::new(16)?)
        .also(Linear::new(16, 16)?)
        .through(Linear::new(16, 2)?)
        .build()?;

    // Set up training.
    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 0.01);
    model.train();

    // Training loop (simplified — no data loader yet).
    let input_t = Tensor::randn(&[20, 2], TensorOptions::default())?;
    let target_t = Tensor::randn(&[20, 2], TensorOptions::default())?;

    for epoch in 0..50 {
        let input = Variable::new(input_t.clone(), true);
        let target = Variable::new(target_t.clone(), false);

        let pred = model.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;

        optimizer.zero_grad();
        loss.backward()?;
        clip_grad_norm(&params, 1.0)?;
        optimizer.step()?;

        if epoch % 10 == 0 {
            println!("epoch {}  loss={:.6}", epoch, loss.item()?);
        }
    }

    // Eval.
    model.eval();
    let test_input = Tensor::from_f32(&[0.5, 0.3], &[1, 2], Device::CPU)?;
    let pred = no_grad(|| {
        model.forward(&Variable::new(test_input, false))
    })?;
    println!("pred: {:?}", pred.data().to_f32_vec()?);

    Ok(())
}
```

---

Next: [Tutorial 5: The Graph Builder](05-graph-builder.md)

Previous: [01-Tensors](01-tensors.md) | [02-Autograd](02-autograd.md) | [03-Modules](03-modules.md)
