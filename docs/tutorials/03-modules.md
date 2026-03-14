# Tutorial 3: Modules

The `nn` module provides neural network layers, activations, and the `Module`
trait that unifies them all. Modules compose naturally — a model is a Module
that contains other Modules.

This tutorial builds on [Tutorial 2: Automatic Differentiation](02-autograd.md).

## The Module Trait

Every layer in floDl implements this trait:

```rust
pub trait Module {
    fn forward(&self, input: &Variable) -> Result<Variable>;

    fn parameters(&self) -> Vec<Parameter> { vec![] }
    fn name(&self) -> &str { "module" }
    fn sub_modules(&self) -> Vec<Rc<dyn Module>> { vec![] }
    fn move_to_device(&self, _device: Device) {}
    fn set_training(&self, _training: bool) {}
    fn as_named_input(&self) -> Option<&dyn NamedInputModule> { None }
}
```

`forward` takes an input variable and returns an output variable. `parameters`
returns all learnable weights. Modules with no learnable parameters (like
activations) return an empty vec.

## Linear

Fully connected layer: `y = x @ W^T + b`.

```rust
let linear = Linear::new(784, 128)?;
```

Weights are Kaiming-initialized (suitable for ReLU). Input shape:
`[batch, in_features]`. Output shape: `[batch, out_features]`.

```rust
let output = linear.forward(&input)?;  // [batch, 784] -> [batch, 128]
```

### Builder options

```rust
// Without bias
let linear = Linear::no_bias(784, 128)?;

// On a specific device
let linear = Linear::on_device(784, 128, Device::CUDA)?;
```

## Conv2d

2D convolution over `[N, C, H, W]` inputs.

```rust
let conv = Conv2d::new(3, 64, 3)?;  // in=3, out=64, kernel=3 (stride=1, padding=0)
// For custom stride/padding:
let conv = Conv2d::build(3, 64, 3, true, [1,1], [1,1], [1,1], 1, Device::CPU)?;
```

## ConvTranspose2d

Transpose convolution (upsampling).

```rust
let deconv = ConvTranspose2d::new(64, 3, 3)?;  // in=64, out=3, kernel=3
```

## Normalization

### LayerNorm

Normalizes the last dimension. Commonly used in transformers.

```rust
let ln = LayerNorm::new(512)?;
let output = ln.forward(&input)?;  // [batch, 512] -> [batch, 512]
```

### BatchNorm

Normalizes over the batch dimension. Uses running statistics at inference.

```rust
// For fully-connected layers: input [batch, features]
let bn = BatchNorm::new(128)?;
let output = bn.forward(&input)?;  // [batch, 128] -> [batch, 128]

// For conv layers: input [batch, channels, height, width]
let bn2d = BatchNorm2d::new(64)?;
let output = bn2d.forward(&input)?;  // [B, 64, H, W] -> [B, 64, H, W]
```

Use `BatchNorm` after Linear layers and `BatchNorm2d` after Conv2d layers.
Both behave differently during training (batch statistics) vs. inference
(running statistics). They track `num_batches_tracked` and will error in eval
mode if no training has occurred — this catches a common silent bug.

See [Train/Eval Mode](#traineval-mode) below.

## Dropout

Randomly zeroes elements during training. Uses inverted dropout so no
scaling is needed at inference.

```rust
let drop = Dropout::new(0.1);  // 10% drop probability
let output = drop.forward(&input)?;
```

During inference, Dropout becomes an identity function.

## Embedding

Lookup table mapping integer indices to dense vectors.

```rust
let emb = Embedding::new(10000, 64)?;  // vocab=10000, dim=64
```

Input is a Variable wrapping an Int64 tensor:

```rust
// [batch, seq_len] -> [batch, seq_len, 64]
let output = emb.forward(&indices)?;
```

## Recurrent Cells

### GRUCell

Single GRU timestep:

```rust
let gru = GRUCell::new(128, 256)?;

let h = gru.forward_step(&x, None)?;      // first step: h initialized to zeros
let h = gru.forward_step(&x2, Some(&h))?; // subsequent steps
```

### LSTMCell

Single LSTM timestep. State packs hidden and cell states into one tensor:

```rust
let lstm = LSTMCell::new(128, 256)?;

let state = lstm.forward_step(&x, None)?;             // first step
let state = lstm.forward_step(&x2, Some(&state))?;    // subsequent steps
```

## Activations

Activation functions are also modules, making them composable in the graph
builder:

```rust
ReLU          // max(0, x)
Sigmoid       // 1 / (1 + exp(-x))
Tanh          // hyperbolic tangent
GELU          // Gaussian Error Linear Unit
SiLU          // x * sigmoid(x), also called Swish
```

All activations have no learnable parameters and are zero-sized types.

## Train/Eval Mode

Some modules (Dropout, BatchNorm) behave differently during training vs.
inference. The `set_training` method on Module controls this:

```rust
model.set_training(false);   // eval mode
model.set_training(true);    // training mode
```

When using the graph builder, `Graph::set_training(bool)` propagates to
all nodes recursively.

## Optional Module Traits

### NamedInputModule

For modules that receive `using` references as a named map instead of
positional arguments:

```rust
pub trait NamedInputModule: Module {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable>;
}
```

### Stateful Module Methods

For modules with per-forward mutable state (attention location, counter),
override `reset()` on Module. Loops auto-call it before iterating:

```rust
impl Module for AttentionStep {
    fn reset(&self) {
        self.location.set(None); // clear stale state
    }
    // ...
}
```

For modules holding Variables across forward calls (recurrent state),
override `detach_state()` on Module. `graph.detach_state()` propagates
to all modules:

```rust
impl Module for RecurrentModule {
    fn detach_state(&self) {
        // break gradient chain on retained hidden state
    }
    // ...
}
```

## Composing Modules Manually

Without the graph builder, you compose modules in plain Rust. Implement
`sub_modules()` to declare children — the framework then handles device
placement, training mode, and parameter collection:

```rust
struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    fn new() -> Result<Self> {
        Ok(MLP {
            fc1: Linear::new(784, 128)?,
            fc2: Linear::new(128, 10)?,
        })
    }
}

impl Module for MLP {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let x = self.fc1.forward(input)?;
        let x = x.relu()?;
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![Rc::new(self.fc1.clone()), Rc::new(self.fc2.clone())]
    }
}
```

This is the same pattern as PyTorch's `nn.Module` — declare children, let
the framework walk the tree. For anything involving residual connections,
parallel branches, loops, or conditional execution, the graph builder API
is more expressive and handles the wiring automatically.

---

Next: [Tutorial 4: Training](04-training.md)
