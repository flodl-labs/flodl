# Tutorial 2: Automatic Differentiation

The `autograd` module provides reverse-mode automatic differentiation.
Variables wrap tensors with gradient tracking. When you perform operations
on variables, a computation graph is built behind the scenes. Calling
`backward()` walks that graph in reverse, accumulating gradients at each
leaf variable.

This tutorial builds on [Tutorial 1: Tensors](01-tensors.md).

## Variables

A `Variable` wraps a tensor and optionally tracks gradients:

```rust
use flodl::{Tensor, Variable};

let t = Tensor::from_f32(&[2.0], &[1])?;

// requires_grad=true: operations on this variable build a computation graph
let x = Variable::new(t, true);

// requires_grad=false: just a constant, no tracking
let c = Variable::new(some_tensor, false);
```

Variables created by the user are **leaf variables**. Variables produced by
operations are non-leaf (intermediate) nodes in the computation graph.

## Forward Pass: Building the Graph

Operations on variables mirror the tensor API. Each operation records its
inputs and backward function:

```rust
let w_t = Tensor::from_f32(&[3.0], &[1])?;
let x_t = Tensor::from_f32(&[2.0], &[1])?;

let w = Variable::new(w_t, true);
let x = Variable::new(x_t, true);

// y = x * w, then reduce to scalar
let y = x.mul(&w)?.sum()?;
// The graph now records: Sum <- Mul <- (x, w)
```

The full set of differentiable operations includes: `add`, `sub`, `mul`,
`div`, `matmul`, `relu`, `sigmoid`, `tanh_act`, `gelu`, `silu`, `exp`,
`log`, `neg`, `sqrt`, `abs`, `pow_scalar`, `sin`, `cos`, `sign`, `floor`,
`ceil`, `round`, `reciprocal`, `clamp`, `sum`, `sum_dim`, `mean`, `mean_dim`,
`min`, `max`, `min_dim`, `max_dim`, `var`, `std`, `var_dim`, `std_dim`, `mul_scalar`, `add_scalar`, `div_scalar`,
`softmax`, `log_softmax`, `transpose`, `permute`, `reshape`, `flatten`,
`squeeze`, `unsqueeze`, `expand`, `narrow`, `select`, `cat`, `chunk`,
`repeat`, `pad`, `index_select`, `gather`, `topk`, `sort`,
`conv2d`, `conv_transpose2d`, `adaptive_avg_pool2d`, `grid_sample`,
`layer_norm`, and more.

## Backward Pass: Computing Gradients

Call `backward()` on a scalar variable to compute gradients for all leaf
variables that contributed to it:

```rust
y.backward()?;
```

`backward()` requires a scalar (single-element) output. After the backward
pass, the calling variable's grad_fn chain is severed in-place (via
`detach_()`) — this immediately frees the C++ autograd Node objects rather
than waiting for the variable to be dropped. Leaf variables hold their
accumulated gradients:

```rust
println!("{:?}", w.grad());  // dy/dw — the gradient tensor
println!("{:?}", x.grad());  // dy/dx
```

## Complete Example: Manual Gradient Check

```rust
// y = x * w, where x=2, w=3
// dy/dw = x = 2
// dy/dx = w = 3

let x_t = Tensor::from_f32(&[2.0], &[1])?;
let w_t = Tensor::from_f32(&[3.0], &[1])?;

let x = Variable::new(x_t, true);
let w = Variable::new(w_t, true);

let y = x.mul(&w)?.sum()?;
y.backward()?;

let w_grad = w.grad().unwrap().to_f32_vec()?;  // [2.0] — dy/dw = x
let x_grad = x.grad().unwrap().to_f32_vec()?;  // [3.0] — dy/dx = w
```

## ZeroGrad

Gradients accumulate across multiple backward passes. Reset them before
each training step:

```rust
w.zero_grad();  // reset gradient to None
```

In practice you will call `optimizer.zero_grad()` which does this for all
parameters (see [Tutorial 4](04-training.md)).

## Detach

Stop gradient flow by detaching a variable. This creates a new leaf variable
sharing the same tensor data but with no gradient tracking:

```rust
let detached = v.detach();
// Operations on detached do not build a graph
```

The underlying `Tensor` also has an in-place variant, `detach_()`, which
severs the grad_fn chain without allocating a new handle. This is used
internally by `backward()` to release the autograd graph immediately.

## no_grad: Disabling Tracking for Inference

Wrap inference code in `no_grad` to skip graph construction. This saves
memory and computation:

```rust
use flodl::no_grad;

no_grad(|| {
    let output = model.forward(&input)?;
    // No computation graph is built, even if inputs require gradients.
    Ok(output)
})?;
```

`no_grad` blocks can nest safely.

## Error Handling

Unlike goDl's error-chain pattern (where errors propagate silently through
a chain), floDl uses Rust's `Result<T>` type. Every operation that can fail
returns a `Result`, and the `?` operator propagates errors immediately:

```rust
let result = x.matmul(&w)?.add(&b)?;
// If matmul fails (shape mismatch), the error returns immediately.
// No silent propagation — you handle errors explicitly.
```

This is more explicit than error chains but catches bugs earlier and
produces clearer error messages.

---

Next: [Tutorial 3: Modules](03-modules.md)
