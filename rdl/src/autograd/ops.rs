use crate::tensor::{Result, Tensor, TensorOptions};

use super::context::is_grad_enabled;
use super::variable::{GradFn, Variable};

/// Check if any input requires gradient and gradient computation is enabled.
fn needs_grad(vars: &[&Variable]) -> bool {
    if !is_grad_enabled() {
        return false;
    }
    vars.iter().any(|v| v.inner.borrow().requires_grad)
}

/// Reduce a gradient tensor to match the original input shape,
/// undoing any broadcasting that occurred during the forward pass.
fn unbroadcast(grad: &Tensor, target_shape: &[i64]) -> Result<Tensor> {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return Ok(grad.clone());
    }

    let mut result = grad.clone();
    let mut result_shape = grad_shape;

    // Sum leading dimensions if gradient has more dims than target
    while result_shape.len() > target_shape.len() {
        result = result.sum_dim(0, false)?;
        result_shape = result.shape();
    }

    // Sum along dimensions where target has size 1 (broadcast dims)
    for i in 0..target_shape.len() {
        if target_shape[i] == 1 && result_shape[i] != 1 {
            result = result.sum_dim(i as i32, true)?;
        }
    }

    Ok(result)
}

impl Variable {
    // --- Arithmetic ---

    pub fn add(&self, other: &Variable) -> Result<Variable> {
        let a = self.inner.borrow();
        let b = other.inner.borrow();
        let result = a.data.add(&b.data)?;

        if !needs_grad(&[self, other]) {
            return Ok(Variable::leaf(result, false));
        }

        let a_shape = a.data.shape();
        let b_shape = b.data.shape();
        drop(a);
        drop(b);

        let grad_fn = GradFn {
            name: "AddBackward",
            inputs: vec![self.clone(), other.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let ga = unbroadcast(grad, &a_shape)?;
                let gb = unbroadcast(grad, &b_shape)?;
                Ok(vec![ga, gb])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    pub fn sub(&self, other: &Variable) -> Result<Variable> {
        let a = self.inner.borrow();
        let b = other.inner.borrow();
        let result = a.data.sub(&b.data)?;

        if !needs_grad(&[self, other]) {
            return Ok(Variable::leaf(result, false));
        }

        let a_shape = a.data.shape();
        let b_shape = b.data.shape();
        drop(a);
        drop(b);

        let grad_fn = GradFn {
            name: "SubBackward",
            inputs: vec![self.clone(), other.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let ga = unbroadcast(grad, &a_shape)?;
                let gb = unbroadcast(&grad.neg()?, &b_shape)?;
                Ok(vec![ga, gb])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    pub fn mul(&self, other: &Variable) -> Result<Variable> {
        let result = {
            let a = self.inner.borrow();
            let b = other.inner.borrow();
            a.data.mul(&b.data)?
        };

        if !needs_grad(&[self, other]) {
            return Ok(Variable::leaf(result, false));
        }

        let (a_shape, b_shape, saved_a, saved_b) = {
            let a = self.inner.borrow();
            let b = other.inner.borrow();
            (a.data.shape(), b.data.shape(), a.data.clone(), b.data.clone())
        };

        let grad_fn = GradFn {
            name: "MulBackward",
            inputs: vec![self.clone(), other.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let ga = unbroadcast(&grad.mul(&saved_b)?, &a_shape)?;
                let gb = unbroadcast(&grad.mul(&saved_a)?, &b_shape)?;
                Ok(vec![ga, gb])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    pub fn div(&self, other: &Variable) -> Result<Variable> {
        let result = {
            let a = self.inner.borrow();
            let b = other.inner.borrow();
            a.data.div(&b.data)?
        };

        if !needs_grad(&[self, other]) {
            return Ok(Variable::leaf(result, false));
        }

        let (a_shape, b_shape, saved_a, saved_b) = {
            let a = self.inner.borrow();
            let b = other.inner.borrow();
            (a.data.shape(), b.data.shape(), a.data.clone(), b.data.clone())
        };

        // d(a/b)/da = 1/b, d(a/b)/db = -a/b²
        let grad_fn = GradFn {
            name: "DivBackward",
            inputs: vec![self.clone(), other.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let ga = unbroadcast(&grad.div(&saved_b)?, &a_shape)?;
                let b_sq = saved_b.mul(&saved_b)?;
                let gb = unbroadcast(&grad.neg()?.mul(&saved_a)?.div(&b_sq)?, &b_shape)?;
                Ok(vec![ga, gb])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    pub fn matmul(&self, other: &Variable) -> Result<Variable> {
        let result = {
            let a = self.inner.borrow();
            let b = other.inner.borrow();
            a.data.matmul(&b.data)?
        };

        if !needs_grad(&[self, other]) {
            return Ok(Variable::leaf(result, false));
        }

        let (saved_a, saved_b) = {
            let a = self.inner.borrow();
            let b = other.inner.borrow();
            (a.data.clone(), b.data.clone())
        };

        // 2D: dL/dA = grad @ B^T, dL/dB = A^T @ grad
        let grad_fn = GradFn {
            name: "MatmulBackward",
            inputs: vec![self.clone(), other.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let a_ndim = saved_a.ndim();
                let b_ndim = saved_b.ndim();

                let bt = saved_b.transpose(b_ndim as i32 - 2, b_ndim as i32 - 1)?;
                let ga = grad.matmul(&bt)?;

                let at = saved_a.transpose(a_ndim as i32 - 2, a_ndim as i32 - 1)?;
                let gb = at.matmul(grad)?;

                Ok(vec![ga, gb])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    pub fn mul_scalar(&self, scalar: f64) -> Result<Variable> {
        let result = self.inner.borrow().data.mul_scalar(scalar)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let grad_fn = GradFn {
            name: "MulScalarBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| Ok(vec![grad.mul_scalar(scalar)?])),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    pub fn add_scalar(&self, scalar: f64) -> Result<Variable> {
        let result = self.inner.borrow().data.add_scalar(scalar)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let grad_fn = GradFn {
            name: "AddScalarBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| Ok(vec![grad.clone()])),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    // --- Activations ---

    pub fn relu(&self) -> Result<Variable> {
        let result = self.inner.borrow().data.relu()?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let saved_input = self.inner.borrow().data.clone();

        let grad_fn = GradFn {
            name: "ReluBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let mask = saved_input.gt_scalar(0.0)?;
                Ok(vec![grad.mul(&mask)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    pub fn sigmoid(&self) -> Result<Variable> {
        let result = self.inner.borrow().data.sigmoid()?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let saved_output = result.clone();

        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        let grad_fn = GradFn {
            name: "SigmoidBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let one = Tensor::ones_like(&saved_output)?;
                let one_minus = one.sub(&saved_output)?;
                let sig_prime = saved_output.mul(&one_minus)?;
                Ok(vec![grad.mul(&sig_prime)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    pub fn tanh_act(&self) -> Result<Variable> {
        let result = self.inner.borrow().data.tanh_op()?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let saved_output = result.clone();

        // tanh'(x) = 1 - tanh²(x)
        let grad_fn = GradFn {
            name: "TanhBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let sq = saved_output.mul(&saved_output)?;
                let one = Tensor::ones_like(&sq)?;
                let one_minus_sq = one.sub(&sq)?;
                Ok(vec![grad.mul(&one_minus_sq)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    // --- Reductions ---

    pub fn sum(&self) -> Result<Variable> {
        let result = self.inner.borrow().data.sum()?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let input_shape = self.inner.borrow().data.shape();
        let input_dtype = self.inner.borrow().data.dtype();
        let input_device = self.inner.borrow().data.device();

        let grad_fn = GradFn {
            name: "SumBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let opts = TensorOptions {
                    dtype: input_dtype,
                    device: input_device,
                };
                let ones = Tensor::ones(&input_shape, opts)?;
                Ok(vec![ones.mul_scalar(grad.item()?)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// Sum along a dimension, optionally keeping the dimension.
    pub fn sum_dim(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.inner.borrow().data.sum_dim(dim, keepdim)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let input_shape = self.inner.borrow().data.shape();

        let grad_fn = GradFn {
            name: "SumDimBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                // Expand grad back to input shape
                let grad_for_expand = if keepdim {
                    grad.clone()
                } else {
                    // Re-insert the summed dimension (size 1) via reshape
                    let mut shape = input_shape.clone();
                    shape[dim as usize] = 1;
                    grad.reshape(&shape)?
                };
                Ok(vec![grad_for_expand.expand(&input_shape)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    // --- Element-wise math ---

    /// Absolute value with differentiable backward (sign function).
    pub fn abs(&self) -> Result<Variable> {
        let result = self.inner.borrow().data.abs()?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let saved_input = self.inner.borrow().data.clone();
        let saved_abs = result.clone();

        // d|x|/dx = sign(x) ≈ x / (|x| + eps) — smooth at zero
        let grad_fn = GradFn {
            name: "AbsBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let sign = saved_input.div(&saved_abs.add_scalar(1e-12)?)?;
                Ok(vec![grad.mul(&sign)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    pub fn exp(&self) -> Result<Variable> {
        let result = self.inner.borrow().data.exp()?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let saved_output = result.clone();

        // exp'(x) = exp(x)
        let grad_fn = GradFn {
            name: "ExpBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| Ok(vec![grad.mul(&saved_output)?])),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    pub fn log(&self) -> Result<Variable> {
        let result = self.inner.borrow().data.log()?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let saved_input = self.inner.borrow().data.clone();

        // log'(x) = 1/x
        let grad_fn = GradFn {
            name: "LogBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let one = Tensor::ones_like(&saved_input)?;
                let reciprocal = one.div(&saved_input)?;
                Ok(vec![grad.mul(&reciprocal)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    pub fn neg(&self) -> Result<Variable> {
        let result = self.inner.borrow().data.neg()?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let grad_fn = GradFn {
            name: "NegBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| Ok(vec![grad.neg()?])),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    // --- Shape operations ---

    pub fn transpose(&self, dim0: i32, dim1: i32) -> Result<Variable> {
        let result = self.inner.borrow().data.transpose(dim0, dim1)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let grad_fn = GradFn {
            name: "TransposeBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| Ok(vec![grad.transpose(dim0, dim1)?])),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    pub fn reshape(&self, shape: &[i64]) -> Result<Variable> {
        let orig_shape = self.inner.borrow().data.shape();
        let result = self.inner.borrow().data.reshape(shape)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let grad_fn = GradFn {
            name: "ReshapeBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| Ok(vec![grad.reshape(&orig_shape)?])),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// Narrow (slice) along a dimension.
    pub fn narrow(&self, dim: i32, start: i64, length: i64) -> Result<Variable> {
        let result = self.inner.borrow().data.narrow(dim, start, length)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let full_shape = self.inner.borrow().data.shape();
        let grad_fn = GradFn {
            name: "NarrowBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                // Scatter narrow gradient back into full-sized zero tensor
                let mut full_grad = Tensor::zeros(&full_shape, TensorOptions::default())?;
                // narrow_scatter: places grad into full_grad at the correct position
                full_grad = full_grad.narrow_scatter(grad, dim, start)?;
                Ok(vec![full_grad])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// Select rows along a dimension using an index tensor.
    pub fn index_select(&self, dim: i32, index: &Tensor) -> Result<Variable> {
        let result = self.inner.borrow().data.index_select(dim, index)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let input_shape = self.inner.borrow().data.shape();
        let saved_index = index.clone();

        let grad_fn = GradFn {
            name: "IndexSelectBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                // Scatter gradient back: zeros.index_add(dim, index, grad)
                let zeros = Tensor::zeros(&input_shape, TensorOptions::default())?;
                let result = zeros.index_add(dim, &saved_index, grad)?;
                Ok(vec![result])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// Concatenate two variables along a dimension.
    pub fn cat(&self, other: &Variable, dim: i32) -> Result<Variable> {
        let result = self.inner.borrow().data.cat(&other.inner.borrow().data, dim)?;

        if !needs_grad(&[self, other]) {
            return Ok(Variable::leaf(result, false));
        }

        let a_size = self.inner.borrow().data.shape()[dim as usize];
        let grad_fn = GradFn {
            name: "CatBackward",
            inputs: vec![self.clone(), other.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let ga = grad.narrow(dim, 0, a_size)?;
                let gb = grad.narrow(dim, a_size, grad.shape()[dim as usize] - a_size)?;
                Ok(vec![ga, gb])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    // --- Backward entry point ---

    pub fn backward(&self) -> Result<()> {
        super::engine::backward(self)
    }
}
