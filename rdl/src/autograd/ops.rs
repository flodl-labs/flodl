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

    /// Mean along a dimension, optionally keeping the dimension.
    pub fn mean_dim(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.inner.borrow().data.mean_dim(dim, keepdim)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let input_shape = self.inner.borrow().data.shape();
        let dim_size = input_shape[dim as usize] as f64;

        let grad_fn = GradFn {
            name: "MeanDimBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let grad_for_expand = if keepdim {
                    grad.clone()
                } else {
                    let mut shape = input_shape.clone();
                    shape[dim as usize] = 1;
                    grad.reshape(&shape)?
                };
                Ok(vec![grad_for_expand.expand(&input_shape)?.mul_scalar(1.0 / dim_size)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// Softmax along a dimension with native libtorch forward.
    pub fn softmax(&self, dim: i32) -> Result<Variable> {
        let result = self.inner.borrow().data.softmax(dim)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let saved_output = result.clone();

        // d(softmax)/dx_i = softmax_i * (delta_ij - softmax_j)
        // grad_input = softmax * (grad - sum(grad * softmax, dim, keepdim))
        let grad_fn = GradFn {
            name: "SoftmaxBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let gs = grad.mul(&saved_output)?;
                let sum_gs = gs.sum_dim(dim, true)?;
                let correction = saved_output.mul(&sum_gs)?;
                Ok(vec![gs.sub(&correction)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// Clamp values to [min, max].
    pub fn clamp(&self, min: f64, max: f64) -> Result<Variable> {
        let result = self.inner.borrow().data.clamp(min, max)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let saved_input = self.inner.borrow().data.clone();

        // Gradient passes through where input was not clamped
        let grad_fn = GradFn {
            name: "ClampBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                // Where clamp didn't change the value, pass gradient through
                let clamped = saved_input.clamp(min, max)?;
                let diff = saved_input.sub(&clamped)?.abs()?;
                let mask = Tensor::ones_like(&diff)?.sub(&diff.gt_scalar(1e-30)?)?;
                Ok(vec![grad.mul(&mask)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// Element-wise power with scalar exponent.
    pub fn pow_scalar(&self, exponent: f64) -> Result<Variable> {
        let result = self.inner.borrow().data.pow_scalar(exponent)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let saved_input = self.inner.borrow().data.clone();

        // d(x^n)/dx = n * x^(n-1)
        let grad_fn = GradFn {
            name: "PowScalarBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let x_pow = saved_input.pow_scalar(exponent - 1.0)?;
                Ok(vec![grad.mul(&x_pow)?.mul_scalar(exponent)?])
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

    /// Square root with native libtorch forward.
    pub fn sqrt(&self) -> Result<Variable> {
        let result = self.inner.borrow().data.sqrt()?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let saved_output = result.clone();

        // d(sqrt(x))/dx = 0.5 / sqrt(x)
        let grad_fn = GradFn {
            name: "SqrtBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                Ok(vec![grad.mul_scalar(0.5)?.div(&saved_output)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// Log-softmax with native libtorch forward.
    pub fn log_softmax(&self, dim: i32) -> Result<Variable> {
        let result = self.inner.borrow().data.log_softmax(dim)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let saved_output = result.clone();

        // d(log_softmax)/dx = I - softmax(x)  applied as:
        // grad_input = grad - exp(log_softmax) * sum(grad, dim, keepdim)
        let grad_fn = GradFn {
            name: "LogSoftmaxBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let softmax = saved_output.exp()?;
                let sum_grad = grad.sum_dim(dim, true)?;
                let correction = softmax.mul(&sum_grad)?;
                Ok(vec![grad.sub(&correction)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// GELU activation with native libtorch forward.
    pub fn gelu(&self) -> Result<Variable> {
        let result = self.inner.borrow().data.gelu()?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let saved_input = self.inner.borrow().data.clone();

        // Exact GELU backward (tanh approximation):
        // a = sqrt(2/pi) * (x + 0.044715 * x^3)
        // grad * (0.5 + 0.5 * tanh(a) + 0.5 * x * (1 - tanh(a)^2) * sqrt(2/pi) * (1 + 3*0.044715*x^2))
        let grad_fn = GradFn {
            name: "GeluBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let x = &saved_input;
                let x_sq = x.mul(x)?;
                let x_cu = x_sq.mul(x)?;
                let inner = x.add(&x_cu.mul_scalar(0.044715)?)?.mul_scalar(0.7978845608)?;
                let tanh_inner = inner.tanh_op()?;
                let tanh_sq = tanh_inner.mul(&tanh_inner)?;
                let one = Tensor::ones_like(&tanh_sq)?;
                let sech_sq = one.sub(&tanh_sq)?;
                let cubic_deriv = x_sq.mul_scalar(3.0 * 0.044715)?.add_scalar(1.0)?;
                let right = x.mul(&sech_sq)?.mul(&cubic_deriv)?.mul_scalar(0.5 * 0.7978845608)?;
                let left = tanh_inner.mul_scalar(0.5)?.add_scalar(0.5)?;
                let deriv = left.add(&right)?;
                Ok(vec![grad.mul(&deriv)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// SiLU activation with native libtorch forward.
    pub fn silu(&self) -> Result<Variable> {
        let result = self.inner.borrow().data.silu()?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let saved_input = self.inner.borrow().data.clone();

        // d(x * sigmoid(x))/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        let grad_fn = GradFn {
            name: "SiluBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                let sig = saved_input.sigmoid()?;
                let one = Tensor::ones_like(&sig)?;
                let one_minus_sig = one.sub(&sig)?;
                let x_term = saved_input.mul(&one_minus_sig)?;
                let bracket = x_term.add_scalar(1.0)?;
                let deriv = sig.mul(&bracket)?;
                Ok(vec![grad.mul(&deriv)?])
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

    /// Select a single index along a dimension (removes that dim).
    pub fn select(&self, dim: i32, index: i64) -> Result<Variable> {
        let result = self.inner.borrow().data.select(dim, index)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let full_shape = self.inner.borrow().data.shape();

        let grad_fn = GradFn {
            name: "SelectBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                // Unsqueeze grad to restore the selected dimension, then scatter
                let mut expanded_shape = full_shape.clone();
                expanded_shape[dim as usize] = 1;
                let grad_unsqueezed = grad.reshape(&expanded_shape)?;
                let mut zeros = Tensor::zeros(&full_shape, TensorOptions::default())?;
                zeros = zeros.narrow_scatter(&grad_unsqueezed, dim, index)?;
                Ok(vec![zeros])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// Flatten dimensions from `start_dim` to `end_dim` (inclusive).
    pub fn flatten(&self, start_dim: i32, end_dim: i32) -> Result<Variable> {
        let shape = self.inner.borrow().data.shape();
        let ndim = shape.len() as i32;
        let start = if start_dim < 0 { ndim + start_dim } else { start_dim } as usize;
        let end = if end_dim < 0 { ndim + end_dim } else { end_dim } as usize;

        let mut new_shape = Vec::new();
        for i in 0..start {
            new_shape.push(shape[i]);
        }
        let mut flat_size: i64 = 1;
        for i in start..=end {
            flat_size *= shape[i];
        }
        new_shape.push(flat_size);
        for i in (end + 1)..shape.len() {
            new_shape.push(shape[i]);
        }

        self.reshape(&new_shape)
    }

    /// Expand (broadcast) to a larger shape.
    pub fn expand(&self, shape: &[i64]) -> Result<Variable> {
        let result = self.inner.borrow().data.expand(shape)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let input_shape = self.inner.borrow().data.shape();

        let grad_fn = GradFn {
            name: "ExpandBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                Ok(vec![unbroadcast(grad, &input_shape)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// Permute dimensions.
    pub fn permute(&self, dims: &[i64]) -> Result<Variable> {
        let result = self.inner.borrow().data.permute(dims)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        // Inverse permutation: if dims = [2, 0, 1], inverse = [1, 2, 0]
        let mut inv = vec![0i64; dims.len()];
        for (i, &d) in dims.iter().enumerate() {
            inv[d as usize] = i as i64;
        }

        let grad_fn = GradFn {
            name: "PermuteBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                Ok(vec![grad.permute(&inv)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// Squeeze (remove) a dimension of size 1.
    pub fn squeeze(&self, dim: i32) -> Result<Variable> {
        let result = self.inner.borrow().data.squeeze(dim)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let grad_fn = GradFn {
            name: "SqueezeBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                Ok(vec![grad.unsqueeze(dim)?])
            }),
        };

        Ok(Variable::from_op(result, grad_fn))
    }

    /// Unsqueeze (insert) a dimension of size 1.
    pub fn unsqueeze(&self, dim: i32) -> Result<Variable> {
        let result = self.inner.borrow().data.unsqueeze(dim)?;

        if !needs_grad(&[self]) {
            return Ok(Variable::leaf(result, false));
        }

        let grad_fn = GradFn {
            name: "UnsqueezeBackward",
            inputs: vec![self.clone()],
            apply: Box::new(move |grad: &Tensor| {
                Ok(vec![grad.squeeze(dim)?])
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

/// Native layer normalization with differentiable backward for all three inputs.
pub fn layer_norm(
    input: &Variable, weight: &Variable, bias: &Variable,
    normalized_size: i64, eps: f64,
) -> Result<Variable> {
    let input_data = input.inner.borrow().data.clone();
    let weight_data = weight.inner.borrow().data.clone();
    let bias_data = bias.inner.borrow().data.clone();

    let (output, mean, rstd) = input_data.native_layer_norm(
        &weight_data, &bias_data, normalized_size, eps,
    )?;

    if !needs_grad(&[input, weight, bias]) {
        return Ok(Variable::leaf(output, false));
    }

    let saved_input = input_data;
    let saved_mean = mean;
    let saved_rstd = rstd;
    let saved_weight = weight_data;
    let saved_bias = bias_data;

    let grad_fn = GradFn {
        name: "LayerNormBackward",
        inputs: vec![input.clone(), weight.clone(), bias.clone()],
        apply: Box::new(move |grad: &Tensor| {
            let (gi, gw, gb) = Tensor::native_layer_norm_backward(
                grad, &saved_input, &saved_mean, &saved_rstd,
                &saved_weight, &saved_bias, normalized_size,
            )?;
            Ok(vec![gi, gw, gb])
        }),
    };

    Ok(Variable::from_op(output, grad_fn))
}

/// 2D convolution with differentiable backward for input, weight, and optional bias.
pub fn conv2d(
    input: &Variable, weight: &Variable, bias: Option<&Variable>,
    stride: [i64; 2], padding: [i64; 2], dilation: [i64; 2], groups: i64,
) -> Result<Variable> {
    let input_data = input.inner.borrow().data.clone();
    let weight_data = weight.inner.borrow().data.clone();
    let bias_data = bias.map(|b| b.inner.borrow().data.clone());

    let result = input_data.conv2d(
        &weight_data, bias_data.as_ref(),
        stride, padding, dilation, groups,
    )?;

    let grad_vars: Vec<&Variable> = if let Some(b) = bias {
        vec![input, weight, b]
    } else {
        vec![input, weight]
    };

    if !needs_grad(&grad_vars) {
        return Ok(Variable::leaf(result, false));
    }

    let saved_input = input_data;
    let saved_weight = weight_data;
    let has_bias = bias.is_some();
    let inputs: Vec<Variable> = grad_vars.iter().map(|v| (*v).clone()).collect();

    let grad_fn = GradFn {
        name: "Conv2dBackward",
        inputs,
        apply: Box::new(move |grad: &Tensor| {
            let (gi, gw, gb) = Tensor::conv2d_backward(
                grad, &saved_input, &saved_weight,
                stride, padding, dilation, groups, has_bias,
            )?;
            if has_bias {
                Ok(vec![gi, gw, gb.unwrap()])
            } else {
                Ok(vec![gi, gw])
            }
        }),
    };

    Ok(Variable::from_op(result, grad_fn))
}

/// Adaptive average pooling to a target spatial size.
pub fn adaptive_avg_pool2d(
    input: &Variable, output_size: [i64; 2],
) -> Result<Variable> {
    let input_data = input.inner.borrow().data.clone();
    let result = input_data.adaptive_avg_pool2d(output_size)?;

    if !needs_grad(&[input]) {
        return Ok(Variable::leaf(result, false));
    }

    let saved_input = input_data;

    let grad_fn = GradFn {
        name: "AdaptiveAvgPool2dBackward",
        inputs: vec![input.clone()],
        apply: Box::new(move |grad: &Tensor| {
            let gi = Tensor::adaptive_avg_pool2d_backward(grad, &saved_input)?;
            Ok(vec![gi])
        }),
    };

    Ok(Variable::from_op(result, grad_fn))
}

/// Grid sampling with differentiable backward for both input and grid.
pub fn grid_sample(
    input: &Variable, grid: &Variable,
    mode: i32, padding_mode: i32, align_corners: bool,
) -> Result<Variable> {
    let input_data = input.inner.borrow().data.clone();
    let grid_data = grid.inner.borrow().data.clone();
    let result = input_data.grid_sample(&grid_data, mode, padding_mode, align_corners)?;

    if !needs_grad(&[input, grid]) {
        return Ok(Variable::leaf(result, false));
    }

    let saved_input = input_data;
    let saved_grid = grid_data;

    let grad_fn = GradFn {
        name: "GridSampleBackward",
        inputs: vec![input.clone(), grid.clone()],
        apply: Box::new(move |grad: &Tensor| {
            let (gi, gg) = Tensor::grid_sample_backward(
                grad, &saved_input, &saved_grid, mode, padding_mode, align_corners,
            )?;
            Ok(vec![gi, gg])
        }),
    };

    Ok(Variable::from_op(result, grad_fn))
}

/// Transposed 2D convolution with differentiable backward.
pub fn conv_transpose2d(
    input: &Variable, weight: &Variable, bias: Option<&Variable>,
    stride: [i64; 2], padding: [i64; 2], output_padding: [i64; 2],
    dilation: [i64; 2], groups: i64,
) -> Result<Variable> {
    let input_data = input.inner.borrow().data.clone();
    let weight_data = weight.inner.borrow().data.clone();
    let bias_data = bias.map(|b| b.inner.borrow().data.clone());

    let result = input_data.conv_transpose2d(
        &weight_data, bias_data.as_ref(),
        stride, padding, output_padding, dilation, groups,
    )?;

    let grad_vars: Vec<&Variable> = if let Some(b) = bias {
        vec![input, weight, b]
    } else {
        vec![input, weight]
    };

    if !needs_grad(&grad_vars) {
        return Ok(Variable::leaf(result, false));
    }

    let saved_input = input_data;
    let saved_weight = weight_data;
    let has_bias = bias.is_some();
    let inputs: Vec<Variable> = grad_vars.iter().map(|v| (*v).clone()).collect();

    let grad_fn = GradFn {
        name: "ConvTranspose2dBackward",
        inputs,
        apply: Box::new(move |grad: &Tensor| {
            let (gi, gw, gb) = Tensor::conv_transpose2d_backward(
                grad, &saved_input, &saved_weight,
                stride, padding, output_padding, dilation, groups, has_bias,
            )?;
            if has_bias {
                Ok(vec![gi, gw, gb.unwrap()])
            } else {
                Ok(vec![gi, gw])
            }
        }),
    };

    Ok(Variable::from_op(result, grad_fn))
}
