# Tutorial 1: Tensors

Tensors are the fundamental data type in floDl — n-dimensional arrays of numbers backed by libtorch. This tutorial covers creation, operations, error handling, and memory management.

## Creating Tensors

All creation functions return `Result<Tensor>`.

```rust
use flodl::{Tensor, TensorOptions, Device, DType};

// From Rust data — data is copied into libtorch
let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Device::CPU)?;

// Filled tensors
let opts = TensorOptions::default();  // Float32, CPU
let zeros = Tensor::zeros(&[3, 4], opts)?;
let ones = Tensor::ones(&[3, 4], opts)?;

// Random tensors (seed with manual_seed() for reproducibility)
let uniform = Tensor::rand(&[2, 3], opts)?;   // values in [0, 1)
let normal = Tensor::randn(&[2, 3], opts)?;   // standard normal

// Integer tensor (for indices, e.g. Embedding lookups)
let idx = Tensor::from_i64(&[0, 3, 7], &[3], Device::CPU)?;
```

### Options

`TensorOptions` is a plain struct with `dtype` and `device` fields:

```rust
let opts = TensorOptions { dtype: DType::Float64, ..Default::default() };
let t = Tensor::ones(&[4], opts)?;

let gpu_opts = TensorOptions { device: Device::CUDA(0), ..Default::default() };
let t = Tensor::zeros(&[3, 3], gpu_opts)?;
```

## Shape Inspection

```rust
let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Device::CPU)?;

t.shape();   // [2, 3]
t.ndim();    // 2
t.numel();   // 6
t.dtype();   // DType::Float32
t.device();  // Device::CPU
```

## Operations

Operations return new tensors — originals are never modified. Every operation
returns `Result<Tensor>`, and the `?` operator propagates errors:

```rust
let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU)?;
let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2], Device::CPU)?;

let result = a.add(&b)?.matmul(&b)?.relu()?;
```

### Arithmetic

```rust
a.add(&b)?          // element-wise a + b
a.sub(&b)?          // element-wise a - b
a.mul(&b)?          // element-wise a * b (Hadamard product)
a.div(&b)?          // element-wise a / b
a.matmul(&b)?       // matrix multiplication

a.mul_scalar(0.5)?  // multiply every element by a scalar
a.add_scalar(1.0)?  // add a scalar to every element
```

### Activations and Math

```rust
t.relu()?           // max(0, x)
t.sigmoid()?        // 1 / (1 + exp(-x))
t.tanh()?           // hyperbolic tangent
t.exp()?            // element-wise e^x
t.log()?            // element-wise ln(x)
t.sqrt()?           // element-wise square root
t.neg()?            // element-wise negation
t.softmax(-1)?      // softmax along dimension
```

### Reductions

```rust
t.sum()?                      // reduce all elements to scalar
t.mean()?                     // mean of all elements
t.sum_dim(1, true)?           // reduce along dim, keep dimension
t.mean_dim(1, true)?          // mean along dim
t.max()?                      // scalar max
t.min()?                      // scalar min
t.argmax(-1)?                 // index of max along dim
```

### Shape Manipulation

```rust
t.reshape(&[6, 1])?          // new shape, same data
t.transpose(0, 1)?           // swap two dimensions
t.flatten(0, -1)?            // flatten all dims
t.squeeze(0)?                // remove dim of size 1
t.unsqueeze(0)?              // add dim of size 1
t.permute(&[1, 0])?          // arbitrary axis reorder
```

### Slicing and Joining

```rust
t.narrow(0, 1, 2)?           // extract a contiguous slice along dim
t.select(0, 1)?              // pick one index along dim, removing that dim
a.cat(&b, 0)?                // concatenate two tensors along dim
t.index_select(0, &indices)? // gather slices at given indices
```

## Extracting Data

Copy tensor data back to Rust vectors:

```rust
let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], Device::CPU)?;

let data: Vec<f32> = t.to_f32_vec()?;   // [1.0, 2.0, 3.0]
let item: f64 = t.select(0, 0)?.item()?; // scalar value as f64
```

`to_f64_vec()` and `to_i64_vec()` are also available.

## Memory Management

Tensors are backed by C++ memory managed through libtorch. Rust's ownership
system handles cleanup automatically via `Drop` — you never need to free
tensors manually.

```rust
{
    let t = Tensor::zeros(&[1000, 1000], TensorOptions::default())?;
    // ... use t ...
} // t is dropped here — C++ memory freed immediately
```

This is a fundamental advantage over garbage-collected languages. In Go,
tensor memory can linger until the GC runs, requiring explicit `Release()`
calls and VRAM budget heuristics. In Rust, memory is freed deterministically
at the end of the owning scope.

`Clone` on a `Tensor` shares the underlying data (like PyTorch's shallow
copy). The C++ `TensorImpl` is reference-counted internally by libtorch.

### Diagnostics

Utilities for debugging memory and performance issues:

```rust
use flodl::{live_tensor_count, rss_kb};

// Number of live C++ Tensor handles (created but not yet dropped).
// If this grows over time, there is a tensor handle leak.
println!("live handles: {}", live_tensor_count());

// Current process RSS in kilobytes (Linux only).
println!("RSS: {}MB", rss_kb() / 1024);

// Count autograd nodes reachable from a tensor — measures graph
// complexity. Compare against Python to detect decomposed-op bloat.
let loss = mse_loss(&pred, &target)?;
println!("autograd nodes: {}", loss.data().autograd_node_count());
```

### Peak VRAM Tracking

On CUDA devices, you can track peak GPU memory usage during a training step or
any other section of code. These match the standard PyTorch memory diagnostics:

| flodl | PyTorch equivalent |
|---|---|
| `cuda_peak_active_bytes()` | `torch.cuda.max_memory_allocated()` |
| `cuda_peak_reserved_bytes()` | `torch.cuda.max_memory_reserved()` |
| `cuda_reset_peak_stats()` | `torch.cuda.reset_peak_memory_stats()` |

"Active" bytes are memory currently holding tensor data. "Reserved" bytes
include the CUDA caching allocator's free pool — memory that libtorch has
obtained from the driver but is not currently in use by any tensor. The gap
between the two tells you how much allocator headroom exists.

A typical pattern for profiling a training step:

```rust
use flodl::{cuda_reset_peak_stats, cuda_peak_active_bytes, cuda_peak_reserved_bytes,
            cuda_empty_cache};

// Flush the allocator cache so reserved starts from a clean baseline
cuda_empty_cache();
cuda_reset_peak_stats();

// --- run one training step ---
let output = model.forward(&batch)?;
let loss = mse_loss(&output, &targets)?;
loss.backward()?;
optimizer.step()?;
optimizer.zero_grad()?;

// Read peaks
let active_mb = cuda_peak_active_bytes()? as f64 / 1048576.0;
let reserved_mb = cuda_peak_reserved_bytes()? as f64 / 1048576.0;
println!("peak active: {active_mb:.1} MB, peak reserved: {reserved_mb:.1} MB");
```

The `_idx` variants (`cuda_peak_active_bytes_idx`, `cuda_peak_reserved_bytes_idx`,
`cuda_reset_peak_stats_idx`) accept an explicit device index for multi-GPU setups.

## Device Transfer

```rust
let gpu = t.to_device(Device::CUDA(0))?;   // move to GPU
let cpu = gpu.to_device(Device::CPU)?;  // move back to CPU

if flodl::cuda_available() {
    println!("CUDA devices: {}", flodl::cuda_device_count());
}
```

### Non-blocking Device Transfer

The default `to_device()` blocks until the transfer completes. For CPU-to-GPU
transfers, you can overlap the copy with CPU work by using pinned memory and an
asynchronous transfer:

```rust
use flodl::{Tensor, Device, TensorOptions, cuda_synchronize};

let cpu_tensor = Tensor::randn(&[256, 512], TensorOptions::default())?;

// Pin the CPU tensor into page-locked memory (requires CUDA)
let pinned = cpu_tensor.pin_memory()?;

// Launch the transfer — returns immediately
let gpu = pinned.to_device_async(Device::CUDA(0))?;

// ... do CPU work while the DMA transfer runs ...

// Synchronize before using the GPU tensor
cuda_synchronize(0);
```

`pin_memory()` allocates the tensor in page-locked (pinned) host memory, which
the GPU can DMA directly without an intermediate staging copy. This matters most
when you are streaming large batches to the GPU and want to keep the CPU busy
with data preprocessing for the next batch.

You can check whether a tensor is already pinned with `t.is_pinned()`.

## Reproducibility

Seed libtorch's RNG before creating random tensors or models:

```rust
flodl::manual_seed(42);  // seeds CPU + CUDA RNGs
```

After seeding, `Tensor::rand`, `Tensor::randn`, dropout masks, and weight
initialization all produce deterministic results.

For CPU-side randomness (shuffling datasets, augmentation), use the `Rng`
struct:

```rust
use flodl::Rng;

let mut rng = Rng::seed(42);
rng.shuffle(&mut data);
```

See [Tutorial 4: Training](04-training.md) for full reproducibility setup.

## cuDNN Benchmark Mode

For fixed-size workloads (fixed batch size, fixed image dimensions), enabling cuDNN
benchmark mode lets cuDNN auto-tune convolution algorithms on the first call:

```rust
flodl::set_cudnn_benchmark(true);  // opt-in, 5-10% speedup for fixed shapes
```

Leave this off for dynamic-shape workloads (variable-length sequences,
multi-resolution images) — the warmup cost can hurt throughput.

## Memory Format (Channels Last)

By default, 4D tensors use the NCHW memory layout (batch, channels, height, width).
GPUs with Tensor Cores (NVIDIA Volta and newer) can run convolutions significantly
faster when tensors are stored in NHWC order, also called "channels last."

```rust
let images = Tensor::randn(&[8, 3, 224, 224], TensorOptions::default())?;

// Convert to channels-last layout
let images_cl = images.to_channels_last()?;

assert!(images_cl.is_channels_last());
assert_eq!(images_cl.shape(), &[8, 3, 224, 224]); // logical shape unchanged
```

The logical shape stays the same — only the physical memory stride order changes.
This avoids format-conversion overhead inside cuDNN convolution kernels. On Tensor
Core GPUs, expect an 8--35% speedup for Conv2d-heavy workloads, depending on layer
sizes and batch dimensions.

Convert your input tensors and model weights to channels-last format before
training. flodl's Conv2d will pick up the layout automatically.

## Foreach Operations

Foreach operations apply the same operation to a list of tensors in a single
fused CUDA kernel launch. When you have dozens or hundreds of parameter tensors
(typical in any real model), this eliminates per-tensor kernel launch overhead.

All foreach functions are associated functions on `Tensor`:

```rust
use flodl::Tensor;

let params: Vec<Tensor> = model.parameters();

// Zero all parameter gradients in one launch
Tensor::foreach_zero_(&params)?;

// Scale every tensor by a constant
Tensor::foreach_mul_scalar_(&params, 0.99)?;

// Add a constant to every tensor
Tensor::foreach_add_scalar_(&params, 1e-8)?;
```

There are also multi-list variants that operate on pairs of tensor lists
element-wise:

```rust
// params[i] += grads[i] * alpha, for all i
Tensor::foreach_add_list_(&params, &grads, -0.01)?;

// params[i] = lerp(params[i], targets[i], weight), for all i
Tensor::foreach_lerp_scalar_(&params, &targets, 0.1)?;

// In-place sqrt of every tensor
Tensor::foreach_sqrt_(&params)?;

// Compute the L2 norm of each tensor — returns a Vec<Tensor> of scalars
let norms: Vec<Tensor> = Tensor::foreach_norm(&params, 2.0)?;
```

You typically do not need to call these directly. flodl's fused optimizers
(Adam, AdamW) and gradient clipping routines use foreach operations internally
to minimize kernel launch overhead. They are exposed in the public API for
advanced use cases like custom optimizers or manual parameter surgery.

---

Previous: [Tutorial 0: Rust for PyTorch Users](00-rust-primer.md) |
Next: [Tutorial 2: Automatic Differentiation](02-autograd.md)
