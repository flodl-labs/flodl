# Tutorial 4: Training

This tutorial puts everything together: loss functions, optimizers, gradient
clipping, and the training loop. It builds on
[Tutorial 3: Modules](03-modules.md).

## Loss Functions

```rust
use flodl::*;

// Mean Squared Error: mean((pred - target)^2)
let loss = mse_loss(&pred, &target)?;

// Cross-Entropy from raw logits (not probabilities).
// pred: [batch, classes] logits. target: [batch, classes] one-hot.
let loss = cross_entropy_loss(&logits, &one_hot_target)?;

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

## Gradient Clipping

Prevent exploding gradients by clipping after backward and before the
optimizer step:

```rust
// Scale gradients so total L2 norm <= max_norm
clip_grad_norm(&params, 1.0)?;

// Clamp each gradient element to [-max_val, max_val]
clip_grad_value(&params, 0.5)?;
```

## Device Placement

By default, all tensors and parameters live on CPU. To train on CUDA, use
`set_device` on the graph.

### Moving the model

```rust
let model = build_model()?;

if flodl::cuda_available() {
    model.set_device(Device::CUDA);
}

// Create optimizer AFTER set_device.
let params = model.parameters();
let optimizer = Adam::new(&params, 0.001);
```

## The Training Loop

The standard pattern is: **forward -> loss -> zero_grad -> backward -> clip -> step**.

```rust
model.set_training(true);

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

If your model uses forward references (recurrent state carried between
forward calls), you **must** call `detach_state()` between training steps.
Without it, the autograd computation graph grows without bound.

```rust
model.set_training(true);

for (input_t, target_t) in &batches {
    let input = Variable::new(input_t.clone(), true);
    let target = Variable::new(target_t.clone(), false);

    let pred = model.forward(&input)?;
    let loss = mse_loss(&pred, &target)?;

    optimizer.zero_grad();
    loss.backward()?;
    clip_grad_norm(&params, 1.0)?;

    model.detach_state();  // break gradient chains on state buffers
    optimizer.step()?;
}
```

**When is it needed?** Only for graphs with forward references — where
`using("x")` appears before `tag("x")` in the builder chain.

## Checkpoints

Save and restore model parameters:

```rust
// Save
save_parameters_file("/tmp/model.fdl", &params)?;

// Load
load_parameters_file("/tmp/model.fdl", &params)?;
```

Parameters are always serialized as float32, making checkpoints portable
across CPU and CUDA. The `io::Write` / `io::Read` variants are also
available for custom destinations:

```rust
save_parameters(&mut writer, &params)?;
load_parameters(&mut reader, &params)?;
```

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

## Eval Mode

Switch to eval mode for inference:

```rust
model.set_training(false);
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

## Complete Example

```rust
use flodl::*;

fn main() -> Result<()> {
    // Build model.
    let model = FlowBuilder::from(Linear::new(2, 16)?)
        .through(GELU)
        .through(LayerNorm::new(16)?)
        .also(Linear::new(16, 16)?)
        .through(Linear::new(16, 2)?)
        .build()?;

    // Set up training.
    let params = model.parameters();
    let optimizer = Adam::new(&params, 0.01);
    model.set_training(true);

    // Training loop (simplified — no data loader yet).
    let input_t = Tensor::randn(&[20, 2], &TensorOptions::default())?;
    let target_t = Tensor::randn(&[20, 2], &TensorOptions::default())?;

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
    model.set_training(false);
    let test_input = Tensor::from_f32(&[0.5, 0.3], &[1, 2])?;
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
