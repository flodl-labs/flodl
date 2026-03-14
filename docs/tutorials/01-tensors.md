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

// Random tensors
let uniform = Tensor::rand(&[2, 3], opts)?;   // values in [0, 1)
let normal = Tensor::randn(&[2, 3], opts)?;   // standard normal

// Integer tensor (for indices, e.g. Embedding lookups)
let idx = Tensor::from_i64(&[0, 3, 7], &[3])?;
```

### Options

`TensorOptions` is a plain struct with `dtype` and `device` fields:

```rust
let opts = TensorOptions { dtype: DType::Float64, ..Default::default() };
let t = Tensor::ones(&[4], opts)?;

let gpu_opts = TensorOptions { device: Device::CUDA, ..Default::default() };
let t = Tensor::zeros(&[3, 3], gpu_opts)?;
```

## Shape Inspection

```rust
let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

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
let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2])?;

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
t.tanh_op()?        // hyperbolic tangent
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
let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3])?;

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

## Device Transfer

```rust
let gpu = t.to_device(Device::CUDA)?;   // move to GPU
let cpu = gpu.to_device(Device::CPU)?;  // move back to CPU

if flodl::cuda_available() {
    println!("CUDA devices: {}", flodl::cuda_device_count());
}
```

---

Previous: [Tutorial 0: Rust for PyTorch Users](00-rust-primer.md) |
Next: [Tutorial 2: Automatic Differentiation](02-autograd.md)
