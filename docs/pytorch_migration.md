# PyTorch → flodl Migration Guide

A side-by-side reference for PyTorch users learning flodl.

## Imports

In PyTorch, `import torch` gives you almost everything. Rust uses explicit
imports — flodl re-exports its full API from the crate root for convenience:

```rust
use flodl::*;  // brings in Tensor, Variable, all nn modules, graph builder, etc.
```

**Or import selectively by module:**

```rust
use flodl::tensor::{Tensor, TensorOptions, DType, Device};
use flodl::autograd::{Variable, no_grad};
use flodl::nn::{Linear, Adam, mse_loss, Module, Parameter};
use flodl::graph::{FlowBuilder, Graph, MergeOp};
```

| If you're doing... | You need |
|---------------------|----------|
| Defining a model with the graph builder | `nn`, `graph` |
| Writing a training loop | `nn`, `autograd`, `tensor` |
| Creating raw tensors | `tensor` |
| Everything (main training script) | `use flodl::*` |

**How it maps to PyTorch:**

| PyTorch | flodl | What's in it |
|---------|-------|-------------|
| `torch.*` | `flodl::tensor` | Creation (`zeros`, `rand`, `arange`...), math ops, shape ops |
| `torch.autograd` | `flodl::autograd` | `Variable`, `no_grad`, gradient tracking |
| `torch.nn` | `flodl::nn` | Modules (`Linear`, `Conv2d`...), activations, losses |
| `torch.optim` | `flodl::nn` | Optimizers (`Adam`, `SGD`, `AdamW`), LR schedulers |
| *(no equivalent)* | `flodl::graph` | Fluent computation graph builder |

## Core Concepts

| PyTorch | flodl | Notes |
|---------|-------|-------|
| `torch.Tensor` | `Tensor` | Immutable, `Drop`-based VRAM cleanup, `Send`+`Sync` |
| `torch.autograd` | `Variable` | Wraps Tensor, tracks gradients via `Rc<RefCell>` |
| `torch.nn.Module` | `Module` trait | `forward(&self, &Variable) -> Result<Variable>` + `parameters()` |
| `model.train()` | `module.train()` | Called on individual modules |
| `model.eval()` | `module.eval()` | Disables dropout, freezes BatchNorm stats |
| `with torch.no_grad():` | `no_grad(\|\| { ... })` | RAII guard disables gradient tracking |

**Error handling:** flodl returns `Result<T>` instead of panicking. Use `?` to propagate errors:

```rust
let y = x.matmul(&w)?.add(&b)?.relu()?;
```

## Reproducibility

```python
# PyTorch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

```rust
// flodl
flodl::manual_seed(42);
flodl::cuda_manual_seed_all(42);  // no-op without CUDA feature
```

`manual_seed` controls all libtorch random operations: `Tensor::rand`, `Tensor::randn`, dropout masks, weight initialization. Call it before model creation for full reproducibility.

For CPU-side randomness (data shuffling, augmentation), use `Rng`:

```rust
use flodl::Rng;

let mut rng = Rng::seed(42);       // deterministic
rng.shuffle(&mut indices);          // Fisher-Yates shuffle
let val = rng.f32();               // uniform [0, 1)
let coin = rng.bernoulli(0.5);    // true ~50%
```

## Tensor Creation

```python
# PyTorch
x = torch.zeros(2, 3)
x = torch.ones(2, 3)
x = torch.rand(2, 3)
x = torch.randn(2, 3)
x = torch.full((2, 3), 7.0)
x = torch.eye(4)
x = torch.arange(0, 10, 2)
x = torch.tensor([1.0, 2.0, 3.0])
x = torch.tensor([0, 1, 2], dtype=torch.int64)
x = torch.linspace(0, 1, 10)
```

```rust
// flodl
let opts = TensorOptions::default();  // Float32, CPU
let x = Tensor::zeros(&[2, 3], opts)?;
let x = Tensor::ones(&[2, 3], opts)?;
let x = Tensor::rand(&[2, 3], opts)?;
let x = Tensor::randn(&[2, 3], opts)?;
let x = Tensor::full(&[2, 3], 7.0, opts)?;
let x = Tensor::eye(4, opts)?;
let x = Tensor::arange(0.0, 10.0, 2.0, opts)?;
let x = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], Device::CPU)?;
let x = Tensor::from_i64(&[0, 1, 2], &[3], Device::CPU)?;
let x = Tensor::linspace(0.0, 1.0, 10, opts)?;
```

**Creation helpers:**

```rust
// Like-tensors (same dtype/device as source)
let y = Tensor::zeros_like(&x)?;
let y = Tensor::ones_like(&x)?;

// Stacking
let y = Tensor::stack(&[&a, &b, &c], 0)?;
```

## Tensor Operations

### Arithmetic

```python
# PyTorch
c = a + b           # element-wise add
c = a - b           # element-wise sub
c = a * b           # element-wise mul
c = a / b           # element-wise div
c = a @ b           # matrix multiply
c = x * 2.0         # scalar multiply
c = x + 1.0         # scalar add
c = x / 3.0         # scalar divide
c = -x              # negation
```

```rust
// flodl
let c = a.add(&b)?;
let c = a.sub(&b)?;
let c = a.mul(&b)?;
let c = a.div(&b)?;
let c = a.matmul(&b)?;
let c = x.mul_scalar(2.0)?;
let c = x.add_scalar(1.0)?;
let c = x.div_scalar(3.0)?;
let c = x.neg()?;
```

### Math Functions

```python
# PyTorch
y = torch.exp(x)
y = torch.log(x)
y = torch.sqrt(x)
y = torch.abs(x)
y = torch.pow(x, 2.0)
y = torch.clamp(x, -1.0, 1.0)
y = torch.sin(x)
y = torch.cos(x)
y = torch.sign(x)
y = torch.floor(x)
y = torch.ceil(x)
y = torch.round(x)
y = torch.reciprocal(x)
```

```rust
// flodl
let y = x.exp()?;
let y = x.log()?;
let y = x.sqrt()?;
let y = x.abs()?;
let y = x.pow_scalar(2.0)?;
let y = x.clamp(-1.0, 1.0)?;
let y = x.sin()?;
let y = x.cos()?;
let y = x.sign()?;
let y = x.floor()?;
let y = x.ceil()?;
let y = x.round()?;
let y = x.reciprocal()?;
```

### Activations

```python
# PyTorch
y = torch.relu(x)
y = torch.sigmoid(x)
y = torch.tanh(x)
y = F.gelu(x)
y = F.silu(x)
y = torch.softmax(x, dim=1)
y = F.log_softmax(x, dim=1)
```

```rust
// flodl
let y = x.relu()?;
let y = x.sigmoid()?;
let y = x.tanh()?;
let y = x.gelu()?;
let y = x.silu()?;
let y = x.softmax(1)?;
let y = x.log_softmax(1)?;
```

### Reductions

```python
# PyTorch
s = x.sum()
s = x.sum(dim=1, keepdim=True)
m = x.mean()
m = x.mean(dim=1, keepdim=True)
v = x.var()
v = x.std()
v = x.max(dim=1, keepdim=True).values
v = x.min(dim=1, keepdim=True).values
idx = x.argmax(dim=1)
n = x.norm()
```

```rust
// flodl
let s = x.sum()?;
let s = x.sum_dim(1, true)?;
let m = x.mean()?;
let m = x.mean_dim(1, true)?;
let v = x.var()?;
let v = x.std()?;
let v = x.max_dim(1, true)?;
let v = x.min_dim(1, true)?;
let idx = x.argmax(1, false)?;
let n = x.norm()?;
```

### Shape Operations

```python
# PyTorch
y = x.reshape(2, 3)
y = x.view(2, 3)           # same as reshape
y = x.squeeze(0)
y = x.unsqueeze(0)
y = x.flatten(1)
y = x.permute(0, 2, 1)
y = x.transpose(0, 1)
y = x.expand(4, 3)
```

```rust
// flodl
let y = x.reshape(&[2, 3])?;
// no separate view — reshape handles it
let y = x.squeeze(0)?;
let y = x.unsqueeze(0)?;
let y = x.flatten(1, -1)?;
let y = x.permute(&[0, 2, 1])?;
let y = x.transpose(0, 1)?;
let y = x.expand(&[4, 3])?;
```

### Indexing and Slicing

```python
# PyTorch
y = x[0]                       # select first along dim 0
y = x[:, 1:3]                  # narrow: dim=1, start=1, length=2
y = x.index_select(0, indices) # gather rows
y = torch.cat([a, b], dim=0)
y = torch.stack([a, b], dim=0)
y = x.chunk(3, dim=1)
y = x.repeat(2, 3)
```

```rust
// flodl
let y = x.select(0, 0)?;
let y = x.narrow(1, 1, 2)?;
let y = x.index_select(0, &indices)?;
let y = a.cat(&b, 0)?;
let y = Tensor::stack(&[&a, &b], 0)?;
let chunks = x.chunk(3, 1)?;
let y = x.repeat(&[2, 3])?;

// Batch iteration (split along dim 0)
for batch in data.batches(32)? {
    let x = Variable::new(batch, false);
    // ...
}
```

### Comparisons and Conditionals

```python
# PyTorch
mask = x > threshold
mask = x >= 0
y = torch.where(mask, a, b)
```

```rust
// flodl
let mask = x.gt(&threshold)?;
let mask = x.ge_scalar(0.0)?;
let y = Tensor::where_cond(&mask, &a, &b)?;

// Scalar comparisons
let mask = x.gt_scalar(0.0)?;
let mask = x.lt_scalar(1.0)?;
let mask = x.ge_scalar(0.0)?;
let mask = x.le_scalar(1.0)?;
```

### Dtype Casting

```python
# PyTorch
y = x.float()       # → float32
y = x.double()      # → float64
y = x.half()        # → float16
y = x.to(torch.bfloat16)
```

```rust
// flodl
let y = x.to_dtype(DType::Float32)?;
let y = x.to_dtype(DType::Float64)?;
let y = x.to_dtype(DType::Float16)?;
let y = x.to_dtype(DType::BFloat16)?;
```

### Data Access

```python
# PyTorch
val = loss.item()          # scalar → float
data = x.numpy()           # → numpy array
data = x.tolist()          # → Python list
```

```rust
// flodl
let val = loss.item()?;           // scalar → f64
let data = x.to_f32_vec()?;      // → Vec<f32>
let data = x.to_f64_vec()?;      // → Vec<f64>
let data = x.to_i64_vec()?;      // → Vec<i64>
```

### Tensor Metadata

```python
# PyTorch
x.shape          # torch.Size([2, 3])
x.ndim           # 2
x.numel()        # 6
x.dtype          # torch.float32
x.device         # device(type='cpu')
```

```rust
// flodl
x.shape()        // Vec<i64>: [2, 3]
x.ndim()         // usize: 2
x.numel()        // i64: 6
x.dtype()        // DType::Float32
x.device()       // Device::CPU
```

## Autograd

```python
# PyTorch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # tensor([2.0, 4.0])
```

```rust
// flodl
let xt = Tensor::from_f32(&[1.0, 2.0], &[2], Device::CPU)?;
let x = Variable::new(xt, true);  // requires_grad = true
let y = x.pow_scalar(2.0)?.sum()?;
y.backward()?;
println!("{:?}", x.grad());  // Some(tensor([2.0, 4.0]))
```

### Key Differences

| Aspect | PyTorch | flodl |
|--------|---------|-------|
| Gradient access | `x.grad` (attribute) | `x.grad()` returns `Option<Tensor>` |
| Clear gradients | `x.grad.zero_()` | `x.zero_grad()` |
| Detach | `x.detach()` | `x.detach()` — returns new leaf Variable |
| No-grad block | `with torch.no_grad():` | `no_grad(\|\| { ... })` or `let _g = NoGradGuard::new();` |
| Check grad enabled | `torch.is_grad_enabled()` | `is_grad_enabled()` |
| Leaf check | `x.is_leaf` | `x.is_leaf()` |

**Differentiable ops on Variable:** Add, Sub, Mul, Div, Matmul, MulScalar, AddScalar,
DivScalar, Neg, Exp, Log, Sqrt, Abs, Pow, Sin, Cos, Sign, Floor, Ceil, Round,
Reciprocal, Clamp, ReLU, Sigmoid, Tanh, GELU, SiLU, Softmax, LogSoftmax,
Sum, SumDim, MeanDim, Mean, Var, Std, Min, Max, VarDim, StdDim,
Reshape, Transpose, Permute, Squeeze, Unsqueeze, Flatten, Expand,
Select, Narrow, Cat, Chunk, Repeat, Pad, IndexSelect, Gather, TopK, Sort,
Conv2d, ConvTranspose2d, MaxPool2d, AdaptiveAvgPool2d, GridSample, LayerNorm.

## Neural Network Layers

```python
# PyTorch
layer = nn.Linear(784, 128)
layer = nn.Linear(784, 128, bias=False)
layer = nn.Conv2d(3, 64, kernel_size=3, padding=1)
layer = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
layer = nn.MaxPool2d(kernel_size=2)
layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
layer = nn.LayerNorm(128)
layer = nn.BatchNorm1d(128)
layer = nn.BatchNorm2d(64)
layer = nn.Dropout(p=0.5)
layer = nn.Embedding(1000, 128)
cell = nn.GRUCell(128, 256)
cell = nn.LSTMCell(128, 256)
```

```rust
// flodl
let layer = Linear::new(784, 128)?;
let layer = Linear::no_bias(784, 128)?;
let layer = Conv2d::new(3, 64, 3)?;                                         // defaults: stride=1, padding=0
let layer = Conv2d::configure(3, 64, 3).with_padding(1).with_stride(2).done()?;   // fluent builder
let layer = Conv2d::build(3, 64, 3, true, [1,1], [1,1], [1,1], 1, Device::CPU)?;  // full control
let layer = ConvTranspose2d::new(64, 3, 4)?;                                // defaults: stride=1, padding=0
let pool = MaxPool2d::new(2);                                               // kernel=2, stride=2 (defaults to kernel)
let pool = MaxPool2d::with_stride(3, 2).padding(1);                         // kernel=3, stride=2, padding=1
let layer = LayerNorm::new(128)?;
let layer = BatchNorm::new(128)?;                                         // for [B, features] after Linear
let layer = BatchNorm2d::new(64)?;                                        // for [B, C, H, W] after Conv2d
let layer = Dropout::new(0.5);
let layer = Embedding::new(1000, 128)?;
let cell = GRUCell::new(128, 256)?;
let cell = LSTMCell::new(128, 256)?;

// On a specific device:
let layer = Linear::on_device(784, 128, Device::CUDA(0))?;
let layer = LayerNorm::on_device(128, Device::CUDA(0))?;
let layer = BatchNorm::on_device(128, Device::CUDA(0))?;
let layer = BatchNorm2d::on_device(64, Device::CUDA(0))?;
let layer = Embedding::on_device(1000, 128, Device::CUDA(0))?;
let cell = GRUCell::on_device(128, 256, Device::CUDA(0))?;
let cell = LSTMCell::on_device(128, 256, Device::CUDA(0))?;
```

## Activations (as Modules)

```python
# PyTorch
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.GELU()
nn.SiLU()
```

```rust
// flodl
ReLU
Sigmoid
Tanh
GELU
SiLU
```

## Composite Modules

In PyTorch, `nn.Module.__init__` auto-discovers child modules assigned to `self`.
In flodl, composite modules implement the `Module` trait and declare children
via `sub_modules()` — enabling recursive device placement, training mode
toggling, and parameter collection.

```python
# PyTorch — children auto-discovered
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

model.to(device)           # walks children automatically
model.parameters()         # collects from all children
model.train()              # propagates to children
```

```rust
// flodl — declare children via sub_modules()
struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl Module for MLP {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let h = self.fc1.forward(input)?.relu()?;
        self.fc2.forward(&h)
    }

    fn parameters(&self) -> Vec<Parameter> {
        nn::collect_parameters(&[&self.fc1 as &dyn Module, &self.fc2])
    }

    fn name(&self) -> &str { "MLP" }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![Rc::new(self.fc1.clone()), Rc::new(self.fc2.clone())]
    }
}
```

Or skip manual structs entirely — use the **graph builder** (see below).

| Aspect | PyTorch | flodl |
|--------|---------|-------|
| Child discovery | Implicit (`self.x = ...`) | Explicit (`sub_modules()`) |
| Parameter collection | Automatic | `collect_parameters()` walks tree |
| Device move | `model.to(device)` | `module.move_to_device(Device::CUDA(0))` |
| Training mode | `model.train()` / `model.eval()` | `module.train()` / `module.eval()` |

## Loss Functions

```python
# PyTorch
loss = F.mse_loss(pred, target)
loss = F.cross_entropy(logits, labels)
loss = F.binary_cross_entropy_with_logits(pred, target)
loss = F.l1_loss(pred, target)
loss = F.smooth_l1_loss(pred, target, beta=1.0)
loss = F.kl_div(log_probs, targets, reduction='batchmean')
```

```rust
// flodl — free functions, return Variable (differentiable)
let loss = mse_loss(&pred, &target)?;
let loss = cross_entropy_loss(&logits, &labels)?;  // labels: [B] indices or [B,C] one-hot
let loss = bce_with_logits_loss(&pred, &target)?;
let loss = l1_loss(&pred, &target)?;
let loss = smooth_l1_loss(&pred, &target, 1.0)?;
let loss = kl_div_loss(&log_probs, &targets)?;
```

## Optimizers

```python
# PyTorch
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

opt.zero_grad()
loss.backward()
opt.step()
```

```rust
// flodl — optimizers own a clone of the param list
let mut opt = SGD::new(&params, 0.01, 0.9);     // lr, momentum
let mut opt = Adam::new(&params, 0.001);         // lr
let mut opt = AdamW::new(&params, 0.001, 0.01);  // lr, weight_decay

opt.zero_grad();
loss.backward()?;
opt.step()?;  // returns Result<()>
```

### Parameter groups

```python
# PyTorch
opt = torch.optim.Adam([
    {"params": encoder.parameters(), "lr": 1e-5},
    {"params": decoder.parameters(), "lr": 1e-3},
])
```

```rust
// flodl — builder API
let mut opt = Adam::with_groups()
    .group(&encoder_params, 1e-5)
    .group(&decoder_params, 1e-3)
    .build();

opt.set_group_lr(0, 1e-6);  // adjust one group
opt.set_lr(1e-4);           // adjust all groups
```

### Freezing parameters

```python
# PyTorch
for param in model.encoder.parameters():
    param.requires_grad = False
```

```rust
// flodl
for param in &encoder_params {
    param.freeze()?;
}
// Later: param.unfreeze()?;
// Check: param.is_frozen()
```

## Learning Rate Scheduling

```python
# PyTorch
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
scheduler.step()
```

```rust
// flodl — schedulers produce an lr, you apply it
let sched = StepDecay::new(0.001, 30, 0.1);
let sched = CosineScheduler::new(0.001, 1e-6, 100);
let lr = sched.lr(step);
opt.set_lr(lr);

// Composable warmup:
let sched = WarmupScheduler::new(CosineScheduler::new(0.001, 1e-6, 100), 0.001, 10);

// Plateau (reduce on metric stall):
let mut sched = PlateauScheduler::new(0.001, 10, 0.1, 1e-6);
let lr = sched.observe(val_loss);
opt.set_lr(lr);
```

**Key difference:** PyTorch schedulers wrap optimizers. flodl schedulers are pure
functions — you call `.lr(step)` or `.observe(metric)` and set the lr yourself.

## Gradient Clipping

```python
# PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

```rust
// flodl
let total_norm = clip_grad_norm(&params, 1.0)?;
let max_val = clip_grad_value(&params, 0.5)?;
```

## Saving and Loading

```python
# PyTorch
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

```rust
// flodl — one-call checkpoint (saves params + buffers + structural hash)
model.save_checkpoint("model.fdl")?;
let report = model.load_checkpoint("model.fdl")?;
// report.loaded, report.skipped, report.missing

// Or with any io::Write / io::Read for custom I/O:
let named = model.named_parameters();
let buffers = model.named_buffers();
let hash = Some(model.structural_hash());
save_checkpoint(&mut writer, &named, &buffers, hash)?;
let report = load_checkpoint(&mut reader, &named, &buffers, hash)?;
```

### Full training resume (model + optimizer)

Optimizers implement the `Stateful` trait for save/load:

```rust
// Save
model.save_checkpoint("model.fdl")?;
let mut f = File::create("optimizer.fdl")?;
optimizer.save_state(&mut f)?;

// Load
let report = model.load_checkpoint("model.fdl")?;
let mut f = File::open("optimizer.fdl")?;
optimizer.load_state(&mut f)?;
```

## Device Placement

```python
# PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
x = x.to(device)
```

```rust
// flodl
let device = if cuda_available() { Device::CUDA(0) } else { Device::CPU };

// Move model parameters
module.move_to_device(device);

// Move tensors
let x = x.to_device(device)?;
let x = x.to_device_of(&weights)?;  // match another tensor's device

// Move variables
let x = x.to_device(device)?;

// Create directly on device
let opts = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
let x = Tensor::zeros(&[2, 3], opts)?;
```

| Aspect | PyTorch | flodl |
|--------|---------|-------|
| Device check | `torch.cuda.is_available()` | `cuda_available()` |
| Device count | `torch.cuda.device_count()` | `cuda_device_count()` |
| Model move | `model.to(device)` | `module.move_to_device(device)` |
| Tensor move | `x.to(device)` | `x.to_device(device)?` |
| cuDNN benchmark | `torch.backends.cudnn.benchmark = True` | `set_cudnn_benchmark(true)` |

## Weight Initialization

```python
# PyTorch
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)
```

```rust
// flodl — returns a new Tensor (immutable design)
let w = xavier_uniform(&[out_features, in_features], in_features, out_features, Device::CPU)?;
let w = xavier_normal(&[out_features, in_features], in_features, out_features, Device::CPU)?;
```

## Mixed Precision Training

```python
# PyTorch
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

```rust
// flodl
let mut scaler = GradScaler::new();

// Cast model to float16
cast_parameters(&params, DType::Float16);

// Forward + backward with scaled loss
let output = model.forward(&input)?;
let loss = mse_loss(&output, &target)?;
let scaled = scaler.scale(&loss)?;
scaled.backward()?;

// Step with automatic unscaling + inf/nan checking
let stepped = scaler.step(&params, &mut || opt.step())?;
scaler.update();
```

## Training Loop Pattern

```python
# PyTorch
model.train()
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

```rust
// flodl
model.train();
for epoch in 0..num_epochs {
    // your data loading here
    opt.zero_grad();
    let output = model.forward(&input)?;
    let loss = cross_entropy_loss(&output, &target)?;
    loss.backward()?;
    clip_grad_norm(&params, 1.0)?;
    opt.step()?;

    opt.set_lr(sched.lr(epoch));
    println!("Epoch {}: loss={:.4}", epoch, loss.item()?);
}

// Inference
model.eval();
let pred = no_grad(|| model.forward(&test_input))?;
```

## Error Handling

PyTorch raises Python exceptions. flodl returns `Result<T, TensorError>`.
Both `Result` and `TensorError` are re-exported from `flodl::*`.
Use Rust's `?` operator for clean propagation:

```rust
fn train_step(model: &Graph, input: &Variable, target: &Variable,
              optimizer: &mut Adam) -> Result<f64> {
    optimizer.zero_grad();
    let output = model.forward(input)?;
    let loss = mse_loss(&output, target)?;
    loss.backward()?;
    optimizer.step()?;
    loss.item()
}
```

## Memory Management

| Aspect | PyTorch | flodl |
|--------|---------|-------|
| Model memory | Python GC + reference counting | Rust `Drop` trait — deterministic deallocation |
| GPU memory | GC-delayed; `torch.cuda.empty_cache()` | Freed immediately when last reference drops |
| Gradient graph | Freed after `.backward()` | `backward()` also calls `detach_()` — grad_fn chain freed synchronously |
| No-grad inference | `with torch.no_grad():` | `no_grad(\|\| { ... })` or `NoGradGuard::new()` |
| Handle diagnostics | N/A | `live_tensor_count()`, `rss_kb()` |

No manual memory management needed. Rust's ownership system handles it.

## Graph Builder (flodl-specific)

flodl's unique feature: a fluent API for building computation graphs declaratively.
No PyTorch equivalent — this replaces manual `nn.Module` subclassing.

### Sequential

```python
# PyTorch
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

```rust
// flodl
let model = FlowBuilder::from(Linear::new(784, 128)?)
    .through(ReLU)
    .through(Linear::new(128, 10)?)
    .build()?;
```

### Residual Connections

```python
# PyTorch — manual
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Linear(128, 128))
    def forward(self, x):
        return x + self.net(x)
```

```rust
// flodl — one line
let model = FlowBuilder::from(Linear::new(128, 128)?)
    .also(ReLU)    // skip connection: output = input + ReLU(input)
    .build()?;
```

### Parallel Branches

```python
# PyTorch — manual fork/merge
class ParallelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Linear(128, 64)
        self.branch2 = nn.Linear(128, 64)
    def forward(self, x):
        return self.branch1(x) + self.branch2(x)
```

```rust
// flodl
let model = FlowBuilder::from(Linear::new(128, 128)?)
    .split(modules![Linear::new(128, 64)?, Linear::new(128, 64)?])
    .merge(MergeOp::Add)
    .build()?;
```

### Loops (Recurrent)

```rust
let model = FlowBuilder::from(init_module)
    .loop_body(step_module)
    .for_n(10)              // fixed 10 iterations
    .through(head_module)
    .build()?;

// Or with learned halting:
    .loop_body(step_module)
    .until_cond(ThresholdHalt::new(0.95), 20)  // max 20 iters
```

### Routing (Mixture of Experts)

```rust
// Hard routing — one expert per input
let model = FlowBuilder::from(encoder)
    .switch(ArgmaxSelector::new(128, 3)?, modules![expert1, expert2, expert3])
    .build()?;

// Soft routing — weighted mixture
let model = FlowBuilder::from(encoder)
    .gate(SoftmaxRouter::new(128, 3)?, modules![expert1, expert2, expert3])
    .build()?;
```

### Tags and Cross-References

```rust
// Tag intermediate outputs for later use
let model = FlowBuilder::from(encoder)
    .tag("encoded")
    .through(decoder)
    .input(&["encoded"])       // declare named input
    .using(&["encoded"])       // wire tagged value as named input
    .build()?;
```

### Observation and Profiling

```rust
let mut model = FlowBuilder::from(Linear::new(4, 8)?)
    .through(ReLU)
    .build()?;

// After each batch
model.end_step();

// After each epoch
model.end_epoch();

// Query metrics
let trend = model.trend("loss");
println!("mean: {}", trend.mean());

// Profiling
model.enable_profiling();
// ... run forward passes ...
if let Some(profile) = model.profile() {
    for t in &profile.nodes {
        println!("{}: {}", t.id, format_duration(t.duration.as_secs_f64()));
    }
}

// Visualization
model.dot();                         // GraphViz DOT string
model.svg(Some("model.svg"))?;      // render to SVG
```

The graph implements `Module`, so it works with optimizers, checkpointing, and everything else.

## Training Monitor (replaces TensorBoard)

PyTorch researchers typically use TensorBoard, Weights & Biases, or MLflow for
training visibility. In floDl, the training monitor is built in — no external
process, no pip install, no separate UI.

```python
# PyTorch + TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/experiment_1")

for epoch in range(num_epochs):
    # ... training ...
    writer.add_scalar("loss", loss.item(), epoch)
    writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

# Then: tensorboard --logdir runs/
```

```rust
// flodl — built-in monitor with live dashboard
use flodl::monitor::Monitor;

let mut monitor = Monitor::new(num_epochs);
monitor.serve(3000)?;                        // live dashboard at http://localhost:3000
monitor.watch(&model);                       // graph SVG in dashboard
monitor.save_html("training_report.html");   // archive at finish

for epoch in 0..num_epochs {
    let t = std::time::Instant::now();
    // ... training ...

    monitor.log(epoch, t.elapsed(), &[("loss", loss_val), ("lr", lr)]);
    // epoch  42/100  loss=0.0023  lr=0.001  [1.2s  ETA 1m 10s]  VRAM: 2.1/6.0 GB (82%)
}

monitor.finish_with(&model);  // profiled SVG + archive saved
```

| Feature | TensorBoard | flodl Monitor |
|---------|-------------|---------------|
| Setup | `pip install tensorboard` + `SummaryWriter` + `tensorboard --logdir` | `monitor.serve(3000)` |
| Terminal output | None (web only) | One-line per epoch with ETA |
| Resource tracking | Manual (no GPU metrics built in) | CPU/RAM/GPU/VRAM automatic |
| Live charts | Yes (web) | Yes (SSE, no polling) |
| Architecture viz | `add_graph()` (limited) | `monitor.watch(&model)` — full DOT/SVG with profiling heat map |
| Offline archive | Log files (need TensorBoard to view) | Self-contained HTML |
| Dependencies | protobuf, gRPC, webpack frontend | Zero — 16KB inline HTML/JS |

## GPU Memory Queries

```python
# PyTorch
torch.cuda.memory_allocated()     # bytes allocated by tensors
torch.cuda.memory_reserved()      # bytes reserved by caching allocator
torch.cuda.max_memory_allocated()  # peak allocated
```

```rust
// flodl — hardware-level via cudaMemGetInfo
let (used, total) = cuda_memory_info()?;   // (bytes_used, bytes_total)
let util = cuda_utilization();              // Option<u32> — GPU % via NVML

// Allocator-level queries
let active = cuda_active_bytes()?;             // bytes backing live tensors
let peak = cuda_peak_active_bytes()?;          // max since last reset
let peak_reserved = cuda_peak_reserved_bytes()?;  // max allocator reservation

// Reset peak tracking (e.g., between profiling phases)
cuda_empty_cache();
cuda_reset_peak_stats();
```

| PyTorch | flodl | What it reports |
|---------|-------|----------------|
| `torch.cuda.mem_get_info()` | `cuda_memory_info()?` | `(used, total)` bytes via `cudaMemGetInfo` |
| `torch.cuda.memory_allocated()` | `cuda_active_bytes()?` | Bytes currently backing live tensors |
| `torch.cuda.memory_reserved()` | `cuda_allocated_bytes()?` | Bytes reserved by caching allocator (includes spill) |
| `torch.cuda.max_memory_allocated()` | `cuda_peak_active_bytes()?` | Peak allocated since last reset |
| `torch.cuda.max_memory_reserved()` | `cuda_peak_reserved_bytes()?` | Peak reserved since last reset |
| `torch.cuda.reset_peak_memory_stats()` | `cuda_reset_peak_stats()` | Reset peak counters |
| `torch.cuda.empty_cache()` | `cuda_empty_cache()` | Release unused cached blocks |
| *(no built-in)* | `cuda_utilization()` | GPU compute % via NVML |

The monitor samples these automatically on every `log()` call — you don't need
to query them manually during training.

## Quick Reference Table

| PyTorch | flodl | Notes |
|---------|-------|-------|
| `torch.zeros(2,3)` | `Tensor::zeros(&[2,3], opts)?` | Shape as slice |
| `x + y` | `x.add(&y)?` | Returns `Result` |
| `x.requires_grad_(True)` | `Variable::new(x, true)` | Set at creation |
| `loss.backward()` | `loss.backward()?` | Same pattern |
| `optimizer.zero_grad()` | `opt.zero_grad()` | Same pattern |
| `optimizer.step()` | `opt.step()?` | Returns `Result` |
| `nn.Linear(in, out)` | `Linear::new(in, out)?` | Returns `Result` |
| `F.relu(x)` | `x.relu()?` | Method on Variable |
| `F.mse_loss(a, b)` | `mse_loss(&a, &b)?` | Free function |
| `model.to(device)` | `module.move_to_device(device)` | |
| `with torch.no_grad():` | `no_grad(\|\| { })` or `NoGradGuard::new()` | Closure or RAII guard |
| `nn.Sequential(...)` | `FlowBuilder::from(...).through(...).build()?` | Fluent builder |
| `model.train()` | `module.train()` | |
| `model.eval()` | `module.eval()` | |
| `torch.save(...)` / `torch.load(...)` | `model.save_checkpoint("m.fdl")?` / `model.load_checkpoint("m.fdl")?` | Named `.fdl` format with `LoadReport` + structural hash validation |
| `param.requires_grad = False` | `param.freeze()?` | Also: `unfreeze()`, `is_frozen()` |
| `Adam([{"params":..., "lr":...}])` | `Adam::with_groups().group(&p, lr).build()` | Per-group LR |
| `torch.cuda.memory_reserved()` | `cuda_allocated_bytes()?` | Bytes reserved by caching allocator |
| `x.pin_memory()` | `x.pin_memory()?` | Page-locked CPU memory for async transfers |
| `x.is_pinned()` | `x.is_pinned()` | Check if tensor is in pinned memory |
| `x.to(device, non_blocking=True)` | `x.to_device_async(device)?` | Non-blocking transfer (pair with `pin_memory`) |
| `x.to(memory_format=channels_last)` | `x.to_channels_last()?` | NHWC layout for Conv2d (8-35% on Tensor Cores) |
| `x.is_contiguous(channels_last)` | `x.is_channels_last()` | Check memory format |
| `torch.cuda.amp.autocast()` | `autocast(DType::Float16, \|\| { })` | Automatic mixed precision dispatch |
| `torch.cuda.amp.GradScaler()` | `GradScaler::new()` | Dynamic loss scaling for AMP |
| `torch.cuda.CUDAGraph()` | `CudaGraph::new()?` | CUDA graph capture/replay |
| `torch.cuda.graph(g)` | `cuda_graph_capture(warmup, pool, \|\| { })` | Convenience capture helper |
| `SummaryWriter` + TensorBoard | `Monitor::new(n).serve(3000)?` | Built-in live dashboard |
