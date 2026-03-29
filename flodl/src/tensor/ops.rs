//! Pure tensor operations: arithmetic, element-wise math, activations,
//! reductions, comparisons, masking, sorting, and advanced indexing.

use std::ptr;
use flodl_sys::{self as ffi, FlodlTensor};
use super::{Tensor, check_err, Result};

impl Tensor {
    // --- Arithmetic (chainable) ---

    /// Element-wise addition. Shapes must be broadcastable.
    ///
    /// ```ignore
    /// let c = a.add(&b)?; // [2, 3] + [2, 3] → [2, 3]
    /// ```
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_add(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise subtraction. Shapes must be broadcastable.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sub(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise (Hadamard) multiplication. Shapes must be broadcastable.
    /// For matrix multiplication, use [`matmul`](Self::matmul).
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_mul(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Matrix multiplication.
    ///
    /// ```ignore
    /// // [batch, M, K] @ [batch, K, N] → [batch, M, N]
    /// let c = a.matmul(&b)?;
    /// ```
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_matmul(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Multiply every element by a scalar. Like `tensor * 0.5` in PyTorch.
    pub fn mul_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_mul_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise division.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_div(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Negate every element.
    pub fn neg(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_neg(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Add a scalar to every element.
    pub fn add_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_add_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Divide every element by a scalar.
    pub fn div_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_div_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Element-wise math ---

    /// Element-wise exponential.
    pub fn exp(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_exp(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_log(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sqrt(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_abs(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Upper triangle of a matrix (or batch of matrices).
    /// Elements below the `diagonal`-th diagonal are zeroed.
    /// `diagonal=0` keeps the main diagonal; `diagonal=1` excludes it.
    pub fn triu(&self, diagonal: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_triu(self.handle, diagonal, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Lower triangle of a matrix (or batch of matrices).
    /// Elements above the `diagonal`-th diagonal are zeroed.
    /// `diagonal=0` keeps the main diagonal; `diagonal=-1` excludes it.
    pub fn tril(&self, diagonal: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_tril(self.handle, diagonal, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Raise every element to a scalar exponent.
    pub fn pow_scalar(&self, exponent: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_pow_scalar(self.handle, exponent, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Clamp all elements to `[min, max]`.
    pub fn clamp(&self, min: f64, max: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_clamp(self.handle, min, max, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Clamp all elements to be at least `min`.
    pub fn clamp_min(&self, min: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_clamp_min(self.handle, min, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Clamp all elements to be at most `max`.
    pub fn clamp_max(&self, max: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_clamp_max(self.handle, max, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise `log(1 + x)`, numerically stable for small x.
    pub fn log1p(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_log1p(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise `exp(x) - 1`, numerically stable for small x.
    pub fn expm1(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_expm1(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise base-2 logarithm.
    pub fn log2(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_log2(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise base-10 logarithm.
    pub fn log10(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_log10(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise sine.
    pub fn sin(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sin(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise cosine.
    pub fn cos(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_cos(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise tangent.
    pub fn tan(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_tan(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise arcsine (inverse sine).
    pub fn asin(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_asin(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise arccosine (inverse cosine).
    pub fn acos(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_acos(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise arctangent (inverse tangent).
    pub fn atan(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_atan(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise sign (-1, 0, or +1).
    pub fn sign(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sign(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise floor.
    pub fn floor(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_floor(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise ceiling.
    pub fn ceil(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ceil(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise rounding to nearest integer.
    pub fn round(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_round(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise reciprocal (1/x).
    pub fn reciprocal(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_reciprocal(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise Gauss error function.
    pub fn erf(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_erf(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise complementary error function (1 - erf(x)).
    pub fn erfc(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_erfc(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise truncation (round towards zero).
    pub fn trunc(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_trunc(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise fractional part (x - trunc(x)).
    pub fn frac(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_frac(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise floating-point remainder (C fmod semantics).
    pub fn fmod(&self, divisor: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_fmod_scalar(self.handle, divisor, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise floating-point remainder with tensor divisor.
    pub fn fmod_tensor(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_fmod_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise remainder (Python modulo semantics).
    pub fn remainder(&self, divisor: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_remainder_scalar(self.handle, divisor, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise remainder with tensor divisor (Python modulo semantics).
    pub fn remainder_tensor(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_remainder_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Linear interpolation: self + weight * (end - self).
    pub fn lerp(&self, end: &Tensor, weight: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_lerp(self.handle, end.handle, weight, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Linear interpolation with per-element weight tensor.
    pub fn lerp_tensor(&self, end: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_lerp_tensor(self.handle, end.handle, weight.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise closeness check: |self - other| <= atol + rtol * |other|.
    pub fn isclose(&self, other: &Tensor, rtol: f64, atol: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_isclose(self.handle, other.handle, rtol, atol, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused: beta * self + alpha * (mat1 @ mat2).
    pub fn addmm(&self, mat1: &Tensor, mat2: &Tensor, beta: f64, alpha: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_addmm(self.handle, mat1.handle, mat2.handle, beta, alpha, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused: self + value * (tensor1 * tensor2).
    pub fn addcmul(&self, tensor1: &Tensor, tensor2: &Tensor, value: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_addcmul(self.handle, tensor1.handle, tensor2.handle, value, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused: self + value * (tensor1 / tensor2).
    pub fn addcdiv(&self, tensor1: &Tensor, tensor2: &Tensor, value: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_addcdiv(self.handle, tensor1.handle, tensor2.handle, value, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Activations ---

    /// SELU: `lambda * (max(0, x) + min(0, alpha * (exp(x) - 1)))`.
    /// Self-normalizing activation with fixed alpha and lambda.
    pub fn selu(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_selu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Hardswish: `x * clamp(x + 3, 0, 6) / 6`.
    pub fn hardswish(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_hardswish(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Hardsigmoid: `clamp(x + 3, 0, 6) / 6`.
    pub fn hardsigmoid(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_hardsigmoid(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// PReLU: `max(0, x) + weight * min(0, x)` (learnable weight).
    pub fn prelu(&self, weight: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_prelu(self.handle, weight.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// ReLU activation: max(0, x).
    pub fn relu(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_relu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Sigmoid activation: 1 / (1 + exp(-x)).
    pub fn sigmoid(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sigmoid(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Tanh activation: element-wise hyperbolic tangent.
    pub fn tanh(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_tanh_op(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Softmax along a dimension.
    pub fn softmax(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_softmax(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Log-softmax along a dimension (numerically stable).
    pub fn log_softmax(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_log_softmax(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// GELU activation (native libtorch).
    pub fn gelu(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_gelu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// SiLU activation (native libtorch).
    pub fn silu(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_silu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Leaky ReLU: `max(0, x) + negative_slope * min(0, x)`.
    pub fn leaky_relu(&self, negative_slope: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_leaky_relu(self.handle, negative_slope, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// ELU: `max(0, x) + min(0, alpha * (exp(x) - 1))`.
    pub fn elu(&self, alpha: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_elu(self.handle, alpha, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Softplus: `(1/beta) * log(1 + exp(beta * x))`.
    /// Reverts to linear when `beta * x > threshold`.
    pub fn softplus(&self, beta: f64, threshold: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_softplus(self.handle, beta, threshold, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Mish: `x * tanh(softplus(x))`.
    pub fn mish(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_mish(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Reductions ---

    /// Sum of all elements (scalar result).
    pub fn sum(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sum(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Mean of all elements (scalar result).
    pub fn mean(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_mean(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Sum along a dimension.
    pub fn sum_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_sum_dim(self.handle, dim, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Mean along a dimension.
    pub fn mean_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_mean_dim(self.handle, dim, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Product of all elements (scalar result).
    pub fn prod(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_prod(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Product along a dimension.
    pub fn prod_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_prod_dim(self.handle, dim, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Cumulative sum along a dimension.
    pub fn cumsum(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_cumsum(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Log of summed exponentials along a dimension (numerically stable).
    pub fn logsumexp(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_logsumexp(self.handle, dim, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scalar minimum.
    pub fn min(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_min(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scalar maximum.
    pub fn max(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_max(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// L2 (Frobenius) norm of all elements.
    pub fn norm(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_norm(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// p-norm along a dimension.
    pub fn norm_p(&self, p: f64, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_norm_p_dim(self.handle, p, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Sum over multiple dimensions at once.
    pub fn sum_dims(&self, dims: &[i32], keepdim: bool) -> Result<Tensor> {
        let mut dims64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_sum_dims(self.handle, dims64.as_mut_ptr(), dims.len() as i32, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Cumulative product along a dimension.
    pub fn cumprod(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_cumprod(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Median of all elements (scalar).
    pub fn median(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_median(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Median along a dimension, returns (values, indices).
    pub fn median_dim(&self, dim: i32, keepdim: bool) -> Result<(Tensor, Tensor)> {
        let mut vals: FlodlTensor = ptr::null_mut();
        let mut idxs: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_median_dim(self.handle, dim, keepdim as i32, &mut vals, &mut idxs) };
        check_err(err)?;
        Ok((Tensor::from_raw(vals), Tensor::from_raw(idxs)))
    }

    /// Count of non-zero elements (scalar).
    pub fn count_nonzero(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_count_nonzero(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Count of non-zero elements along a dimension.
    pub fn count_nonzero_dim(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_count_nonzero_dim(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Indices of non-zero elements (N x ndim tensor).
    pub fn nonzero(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_nonzero(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Unique elements. Returns (output, inverse_indices).
    /// If `return_inverse` is false, inverse_indices is empty.
    pub fn unique(&self, sorted: bool, return_inverse: bool) -> Result<(Tensor, Tensor)> {
        let mut output: FlodlTensor = ptr::null_mut();
        let mut inverse: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_unique(self.handle, sorted as i32, return_inverse as i32, &mut output, &mut inverse)
        };
        check_err(err)?;
        let inv = if inverse.is_null() {
            Tensor::from_i64(&[], &[0], super::Device::CPU)?
        } else {
            Tensor::from_raw(inverse)
        };
        Ok((Tensor::from_raw(output), inv))
    }

    /// Binary search: find insertion indices in a sorted sequence.
    pub fn searchsorted(&self, values: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_searchsorted(self.handle, values.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Minimum along a dimension (values only).
    pub fn min_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_min_dim(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Maximum along a dimension (values only).
    pub fn max_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_max_dim(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Argmax along a dimension.
    pub fn argmax(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_argmax(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Argmin along a dimension.
    pub fn argmin(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_argmin(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Variance of all elements (Bessel-corrected).
    pub fn var(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_var(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Standard deviation of all elements (Bessel-corrected).
    #[allow(clippy::should_implement_trait)]
    pub fn std(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_std_op(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Variance along a dimension (Bessel-corrected).
    pub fn var_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_var_dim(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Standard deviation along a dimension (Bessel-corrected).
    pub fn std_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_std_dim(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Comparisons ---

    /// Element-wise greater-than comparison against a scalar.
    pub fn gt_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_gt_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise greater-than-or-equal comparison against a scalar.
    pub fn ge_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ge_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise less-than-or-equal comparison against a scalar.
    pub fn le_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_le_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise less-than comparison against a scalar.
    pub fn lt_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_lt_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise equality comparison against a scalar (returns float mask: 0.0 or 1.0).
    pub fn eq_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_eq_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise not-equal comparison against a scalar (returns float mask: 0.0 or 1.0).
    pub fn ne_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ne_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise NaN detection (returns float mask: 0.0 or 1.0).
    pub fn isnan(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_isnan(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise infinity detection (returns float mask: 0.0 or 1.0).
    pub fn isinf(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_isinf(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise logical AND of two tensors (returns float mask).
    pub fn logical_and(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_logical_and(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise logical OR of two tensors (returns float mask).
    pub fn logical_or(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_logical_or(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise logical NOT (returns float mask).
    pub fn logical_not(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_logical_not(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Returns a scalar float tensor: 1.0 if any element is non-zero, 0.0 otherwise.
    pub fn any(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_any(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Returns a scalar float tensor: 1.0 if all elements are non-zero, 0.0 otherwise.
    pub fn all(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_all(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise atan2 (arc tangent of y/x).
    pub fn atan2(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_atan2(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise maximum of two tensors.
    pub fn maximum(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_maximum(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise minimum of two tensors.
    pub fn minimum(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_minimum(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise greater-than (returns float mask: 0.0 or 1.0).
    pub fn gt(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_gt_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise less-than (returns float mask: 0.0 or 1.0).
    pub fn lt(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_lt_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise greater-than-or-equal (returns float mask: 0.0 or 1.0).
    pub fn ge(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ge_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise less-than-or-equal (returns float mask: 0.0 or 1.0).
    pub fn le(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_le_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise equality. Returns a mask (0.0 or 1.0) in the input's
    /// dtype for float inputs, or Float32 for integer/bool inputs.
    pub fn eq_tensor(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_eq_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise not-equal. Returns a mask (0.0 or 1.0) in the input's
    /// dtype for float inputs, or Float32 for integer/bool inputs.
    pub fn ne_tensor(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ne_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Masking/conditional ---

    /// Fill elements where `mask` is true (non-zero) with `value`.
    /// The mask is broadcast to match the tensor shape.
    pub fn masked_fill(&self, mask: &Tensor, value: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_masked_fill(self.handle, mask.handle, value, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Conditional select: where(condition, self, other).
    pub fn where_cond(condition: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_where(condition.handle, x.handle, y.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Sorting ---

    /// Top-k values and indices along a dimension. Returns (values, indices).
    pub fn topk(&self, k: i64, dim: i32, largest: bool, sorted: bool) -> Result<(Tensor, Tensor)> {
        let mut values: FlodlTensor = ptr::null_mut();
        let mut indices: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_topk(
                self.handle, k, dim, largest as i32, sorted as i32,
                &mut values, &mut indices,
            )
        };
        check_err(err)?;
        Ok((Tensor::from_raw(values), Tensor::from_raw(indices)))
    }

    /// Sort along a dimension. Returns (sorted_values, indices).
    pub fn sort(&self, dim: i32, descending: bool) -> Result<(Tensor, Tensor)> {
        let mut values: FlodlTensor = ptr::null_mut();
        let mut indices: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_sort(self.handle, dim, descending as i32, &mut values, &mut indices)
        };
        check_err(err)?;
        Ok((Tensor::from_raw(values), Tensor::from_raw(indices)))
    }

    /// Return indices that would sort the tensor along a dimension.
    pub fn argsort(&self, dim: i32, descending: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_argsort(self.handle, dim, descending as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Advanced indexing ---

    /// Gather values along a dimension using an index tensor.
    pub fn gather(&self, dim: i32, index: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_gather(self.handle, dim, index.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter-add: accumulate src into self at index positions along dim.
    pub fn scatter_add(&self, dim: i32, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_scatter_add(self.handle, dim, index.handle, src.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter: write src values into self at index positions along dim (replaces, not adds).
    pub fn scatter(&self, dim: i32, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_scatter(self.handle, dim, index.handle, src.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Select rows/elements along a dimension using an index tensor.
    pub fn index_select(&self, dim: i32, index: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_index_select(self.handle, dim, index.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter-add src into self along dim at positions given by index.
    pub fn index_add(&self, dim: i32, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_index_add(self.handle, dim, index.handle, src.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter a selected index back into a tensor.
    pub fn select_scatter(&self, src: &Tensor, dim: i32, index: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_select_scatter(self.handle, src.handle, dim, index, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Other ---

    /// L_p normalize along a dimension (default: L2, dim=-1).
    pub fn normalize(&self, p: f64, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_normalize(self.handle, p, dim, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Draw samples from a multinomial distribution.
    /// `self` contains unnormalized probabilities (one row per distribution).
    pub fn multinomial(&self, num_samples: i64, replacement: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_multinomial(
                self.handle, num_samples, replacement as i32, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Pairwise L2 distance between rows of two batched matrices.
    /// Input shapes: `[B, P, D]` and `[B, R, D]` -> output `[B, P, R]`.
    pub fn cdist(&self, other: &Tensor) -> Result<Tensor> {
        self.cdist_p(other, 2.0)
    }

    /// Pairwise distance with custom p-norm.
    pub fn cdist_p(&self, other: &Tensor, p: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_cdist(self.handle, other.handle, p, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Cosine similarity between two tensors along a dimension.
    /// Default dim=1, eps=1e-8 (matches PyTorch).
    pub fn cosine_similarity(&self, other: &Tensor, dim: i64, eps: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_cosine_similarity(self.handle, other.handle, dim, eps, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Cast to a different dtype.
    pub fn to_dtype(&self, dtype: super::DType) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_to_dtype(self.handle, dtype as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Check if all elements are finite (no inf/nan).
    pub fn all_finite(&self) -> Result<bool> {
        let mut result: i32 = 0;
        let err = unsafe { ffi::flodl_all_finite(self.handle, &mut result) };
        check_err(err)?;
        Ok(result != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_add() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[4.0, 5.0, 6.0], &[3], test_device()).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_f32_vec().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2], test_device()).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.to_f32_vec().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_chaining() {
        let a = Tensor::from_f32(&[1.0, -2.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[1.0, 1.0, 1.0], &[3], test_device()).unwrap();
        let result = a.add(&b).unwrap().relu().unwrap().sum().unwrap();
        // [1+1, -2+1, 3+1] = [2, -1, 4] -> relu -> [2, 0, 4] -> sum -> 6
        let val = result.item().unwrap();
        assert!((val - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_div_scalar() {
        let t = Tensor::from_f32(&[6.0, 9.0], &[2], test_device()).unwrap();
        let r = t.div_scalar(3.0).unwrap();
        let data = r.to_f32_vec().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean() {
        let t = Tensor::from_f32(&[2.0, 4.0, 6.0], &[3], test_device()).unwrap();
        let m = t.mean().unwrap();
        assert!((m.item().unwrap() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_sub_mul_div() {
        let a = Tensor::from_f32(&[6.0, 8.0], &[2], test_device()).unwrap();
        let b = Tensor::from_f32(&[2.0, 3.0], &[2], test_device()).unwrap();
        assert_eq!(a.sub(&b).unwrap().to_f32_vec().unwrap(), vec![4.0, 5.0]);
        assert_eq!(a.mul(&b).unwrap().to_f32_vec().unwrap(), vec![12.0, 24.0]);
        let d = a.div(&b).unwrap().to_f32_vec().unwrap();
        assert!((d[0] - 3.0).abs() < 1e-5);
        assert!((d[1] - 8.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_scalar_ops() {
        let t = Tensor::from_f32(&[2.0, 4.0], &[2], test_device()).unwrap();
        assert_eq!(t.add_scalar(1.0).unwrap().to_f32_vec().unwrap(), vec![3.0, 5.0]);
        assert_eq!(t.mul_scalar(3.0).unwrap().to_f32_vec().unwrap(), vec![6.0, 12.0]);
        assert_eq!(t.neg().unwrap().to_f32_vec().unwrap(), vec![-2.0, -4.0]);
    }

    #[test]
    fn test_exp_log_sqrt_abs_pow() {
        let t = Tensor::from_f32(&[1.0, 4.0], &[2], test_device()).unwrap();
        let e = t.exp().unwrap().to_f32_vec().unwrap();
        assert!((e[0] - 1.0_f32.exp()).abs() < 1e-5);

        let l = t.log().unwrap().to_f32_vec().unwrap();
        assert!((l[1] - 4.0_f32.ln()).abs() < 1e-5);

        let s = t.sqrt().unwrap().to_f32_vec().unwrap();
        assert!((s[1] - 2.0).abs() < 1e-5);

        let a = Tensor::from_f32(&[-3.0, 5.0], &[2], test_device()).unwrap();
        assert_eq!(a.abs().unwrap().to_f32_vec().unwrap(), vec![3.0, 5.0]);

        let p = t.pow_scalar(2.0).unwrap().to_f32_vec().unwrap();
        assert!((p[0] - 1.0).abs() < 1e-5);
        assert!((p[1] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_clamp() {
        let t = Tensor::from_f32(&[-1.0, 0.5, 2.0], &[3], test_device()).unwrap();
        let c = t.clamp(0.0, 1.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(c, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_sum_dim_mean_dim() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let s = t.sum_dim(1, false).unwrap().to_f32_vec().unwrap();
        assert_eq!(s, vec![3.0, 7.0]);

        let m = t.mean_dim(0, false).unwrap().to_f32_vec().unwrap();
        assert!((m[0] - 2.0).abs() < 1e-5);
        assert!((m[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm() {
        let t = Tensor::from_f32(&[3.0, 4.0], &[2], test_device()).unwrap();
        let n = t.norm().unwrap().item().unwrap();
        assert!((n - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_activations() {
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        assert_eq!(t.relu().unwrap().to_f32_vec().unwrap(), vec![0.0, 0.0, 1.0]);

        let sig = t.sigmoid().unwrap().to_f32_vec().unwrap();
        assert!((sig[2] - 0.7310586).abs() < 1e-5);

        let th = t.tanh().unwrap().to_f32_vec().unwrap();
        assert!((th[2] - 1.0_f32.tanh()).abs() < 1e-5);

        // gelu/silu just check they don't crash and return right shape
        assert_eq!(t.gelu().unwrap().shape(), vec![3]);
        assert_eq!(t.silu().unwrap().shape(), vec![3]);
    }

    #[test]
    fn test_softmax_log_softmax() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let sm = t.softmax(0).unwrap().to_f32_vec().unwrap();
        let total: f32 = sm.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);

        let lsm = t.log_softmax(0).unwrap().to_f32_vec().unwrap();
        assert!(lsm[0] < 0.0 && lsm[1] < 0.0 && lsm[2] < 0.0);
    }

    #[test]
    fn test_eq_ne_tensor() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[1.0, 5.0, 3.0], &[3], test_device()).unwrap();

        let eq = a.eq_tensor(&b).unwrap().to_f32_vec().unwrap();
        assert_eq!(eq, vec![1.0, 0.0, 1.0]);

        let ne = a.ne_tensor(&b).unwrap().to_f32_vec().unwrap();
        assert_eq!(ne, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_gt_lt_ge_le_tensor() {
        let a = Tensor::from_f32(&[1.0, 3.0, 2.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[2.0, 2.0, 2.0], &[3], test_device()).unwrap();

        assert_eq!(a.gt(&b).unwrap().to_f32_vec().unwrap(), vec![0.0, 1.0, 0.0]);
        assert_eq!(a.lt(&b).unwrap().to_f32_vec().unwrap(), vec![1.0, 0.0, 0.0]);
        assert_eq!(a.ge(&b).unwrap().to_f32_vec().unwrap(), vec![0.0, 1.0, 1.0]);
        assert_eq!(a.le(&b).unwrap().to_f32_vec().unwrap(), vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sign_floor_ceil_round() {
        let t = Tensor::from_f32(&[-2.7, 0.0, 1.3], &[3], test_device()).unwrap();
        assert_eq!(t.sign().unwrap().to_f32_vec().unwrap(), vec![-1.0, 0.0, 1.0]);
        assert_eq!(t.floor().unwrap().to_f32_vec().unwrap(), vec![-3.0, 0.0, 1.0]);
        assert_eq!(t.ceil().unwrap().to_f32_vec().unwrap(), vec![-2.0, 0.0, 2.0]);

        let r = Tensor::from_f32(&[-0.6, 0.4, 1.5], &[3], test_device()).unwrap();
        let rv = r.round().unwrap().to_f32_vec().unwrap();
        assert!((rv[0] - (-1.0)).abs() < 1e-5);
        assert!((rv[1] - 0.0).abs() < 1e-5);
        assert!((rv[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmin() {
        let t = Tensor::from_f32(&[3.0, 1.0, 2.0], &[3], test_device()).unwrap();
        let idx = t.argmin(0, false).unwrap().to_i64_vec().unwrap();
        assert_eq!(idx, vec![1]);
    }

    #[test]
    fn test_var_std() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        // Bessel: var = ((1-2)^2+(2-2)^2+(3-2)^2)/2 = 1.0
        assert!((t.var().unwrap().item().unwrap() - 1.0).abs() < 1e-5);
        assert!((t.std().unwrap().item().unwrap() - 1.0).abs() < 1e-5);

        // dim variant: [[1,2],[3,4]] var along dim=1 = [0.5, 0.5]
        let t2 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let vd = t2.var_dim(1, false).unwrap().to_f32_vec().unwrap();
        assert!((vd[0] - 0.5).abs() < 1e-5);
        assert!((vd[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_sin_cos_reciprocal() {
        let t = Tensor::from_f32(&[0.0, 1.0], &[2], test_device()).unwrap();
        let s = t.sin().unwrap().to_f32_vec().unwrap();
        assert!((s[0] - 0.0).abs() < 1e-5);
        assert!((s[1] - 1.0_f32.sin()).abs() < 1e-5);

        let c = t.cos().unwrap().to_f32_vec().unwrap();
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 1.0_f32.cos()).abs() < 1e-5);

        let r = Tensor::from_f32(&[2.0, 5.0], &[2], test_device()).unwrap();
        let rec = r.reciprocal().unwrap().to_f32_vec().unwrap();
        assert!((rec[0] - 0.5).abs() < 1e-5);
        assert!((rec[1] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_gather_scatter_add() {
        // gather: pick elements by index
        let t = Tensor::from_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2], test_device()).unwrap();
        let idx = Tensor::from_i64(&[1, 0, 0, 1], &[2, 2], test_device()).unwrap();
        let g = t.gather(1, &idx).unwrap().to_f32_vec().unwrap();
        assert_eq!(g, vec![20.0, 10.0, 30.0, 40.0]);

        // scatter_add: accumulate into base at positions
        let base = Tensor::zeros(&[2, 3], test_opts()).unwrap();
        let src = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let idx2 = Tensor::from_i64(&[0, 2, 1, 0], &[2, 2], test_device()).unwrap();
        let sa = base.scatter_add(1, &idx2, &src).unwrap();
        let data = sa.to_f32_vec().unwrap();
        // Row 0: pos 0 += 1.0, pos 2 += 2.0 -> [1, 0, 2]
        // Row 1: pos 1 += 3.0, pos 0 += 4.0 -> [4, 3, 0]
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[2] - 2.0).abs() < 1e-5);
        assert!((data[3] - 4.0).abs() < 1e-5);
        assert!((data[4] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_topk_sort() {
        let t = Tensor::from_f32(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5], test_device()).unwrap();
        let (vals, idxs) = t.topk(3, 0, true, true).unwrap();
        assert_eq!(vals.to_f32_vec().unwrap(), vec![5.0, 4.0, 3.0]);
        let idx_data = idxs.to_i64_vec().unwrap();
        assert_eq!(idx_data, vec![4, 2, 0]);

        let (svals, sidxs) = t.sort(0, false).unwrap();
        assert_eq!(svals.to_f32_vec().unwrap(), vec![1.0, 1.0, 3.0, 4.0, 5.0]);
        let si = sidxs.to_i64_vec().unwrap();
        assert_eq!(si[4], 4); // 5.0 was at index 4
    }

    #[test]
    fn test_masked_fill() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let mask = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2], test_device()).unwrap();
        let filled = t.masked_fill(&mask, -1e9).unwrap().to_f32_vec().unwrap();
        assert!(filled[0] < -1e8); // masked
        assert!((filled[1] - 2.0).abs() < 1e-5); // kept
        assert!((filled[2] - 3.0).abs() < 1e-5); // kept
        assert!(filled[3] < -1e8); // masked
    }

    #[test]
    fn test_tril() {
        let t = Tensor::from_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3], test_device(),
        ).unwrap();
        let lo = t.tril(0).unwrap().to_f32_vec().unwrap();
        assert_eq!(lo, vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_prod() {
        let t = Tensor::from_f32(&[2.0, 3.0, 4.0], &[3], test_device()).unwrap();
        let p = t.prod().unwrap().item().unwrap();
        assert!((p - 24.0).abs() < 1e-4);
    }

    #[test]
    fn test_prod_dim() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let p = t.prod_dim(1, false).unwrap().to_f32_vec().unwrap();
        assert!((p[0] - 2.0).abs() < 1e-4);
        assert!((p[1] - 12.0).abs() < 1e-4);
    }

    #[test]
    fn test_cumsum() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let c = t.cumsum(1).unwrap().to_f32_vec().unwrap();
        assert_eq!(c, vec![1.0, 3.0, 3.0, 7.0]);
    }

    #[test]
    fn test_logsumexp() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let lse = t.logsumexp(0, false).unwrap().item().unwrap();
        // log(e^1 + e^2 + e^3) ~ 3.4076
        assert!((lse - 3.4076).abs() < 1e-3);
    }

    #[test]
    fn test_multinomial() {
        let probs = Tensor::from_f32(&[0.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let samples = probs.multinomial(2, true).unwrap();
        // All probability mass on index 2 -- both samples must be 2.
        let vals = samples.to_i64_vec().unwrap();
        assert_eq!(vals, vec![2, 2]);
    }

    #[test]
    fn test_normalize() {
        let t = Tensor::from_f32(&[3.0, 4.0], &[2], test_device()).unwrap();
        let n = t.normalize(2.0, 0).unwrap().to_f32_vec().unwrap();
        // L2 norm is 5, so [0.6, 0.8]
        assert!((n[0] - 0.6).abs() < 1e-5);
        assert!((n[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_leaky_relu() {
        let t = Tensor::from_f32(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5], test_device()).unwrap();
        let r = t.leaky_relu(0.1).unwrap().to_f32_vec().unwrap();
        assert!((r[0] - (-0.2)).abs() < 1e-5);
        assert!((r[1] - (-0.1)).abs() < 1e-5);
        assert!((r[2] - 0.0).abs() < 1e-5);
        assert!((r[3] - 1.0).abs() < 1e-5);
        assert!((r[4] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_elu() {
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let r = t.elu(1.0).unwrap().to_f32_vec().unwrap();
        // ELU(-1) = 1*(exp(-1)-1) ~ -0.6321
        assert!((r[0] - (-0.6321)).abs() < 1e-3);
        assert!((r[1] - 0.0).abs() < 1e-5);
        assert!((r[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softplus() {
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let r = t.softplus(1.0, 20.0).unwrap().to_f32_vec().unwrap();
        // softplus(0) = ln(2)
        assert!((r[1] - std::f32::consts::LN_2).abs() < 1e-3);
        // softplus(x) > 0 for all x
        assert!(r[0] > 0.0);
    }

    #[test]
    fn test_mish() {
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let r = t.mish().unwrap().to_f32_vec().unwrap();
        // mish(0) = 0 * tanh(softplus(0)) = 0
        assert!((r[1] - 0.0).abs() < 1e-5);
        // mish(1) ~ 0.8651
        assert!((r[2] - 0.8651).abs() < 1e-3);
    }

    #[test]
    fn test_cdist() {
        // Two 2D points: [0,0] and [3,4] -> distance = 5
        let x = Tensor::from_f32(&[0.0, 0.0], &[1, 1, 2], test_device()).unwrap();
        let y = Tensor::from_f32(&[3.0, 4.0], &[1, 1, 2], test_device()).unwrap();
        let d = x.cdist(&y).unwrap();
        assert_eq!(d.shape(), vec![1, 1, 1]);
        assert!((d.item().unwrap() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_cdist_p1() {
        // L1: |3| + |4| = 7
        let x = Tensor::from_f32(&[0.0, 0.0], &[1, 1, 2], test_device()).unwrap();
        let y = Tensor::from_f32(&[3.0, 4.0], &[1, 1, 2], test_device()).unwrap();
        let d = x.cdist_p(&y, 1.0).unwrap();
        assert!((d.item().unwrap() - 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_clamp_min_max() {
        let t = Tensor::from_f32(&[-2.0, 0.5, 3.0], &[3], test_device()).unwrap();
        let cmin = t.clamp_min(0.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(cmin, vec![0.0, 0.5, 3.0]);
        let cmax = t.clamp_max(1.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(cmax, vec![-2.0, 0.5, 1.0]);
    }

    #[test]
    fn test_log1p_expm1() {
        let t = Tensor::from_f32(&[0.0, 1.0], &[2], test_device()).unwrap();
        let l = t.log1p().unwrap().to_f32_vec().unwrap();
        assert!((l[0] - 0.0).abs() < 1e-5); // log(1+0) = 0
        assert!((l[1] - 2.0_f32.ln()).abs() < 1e-5); // log(1+1) = ln(2)

        let e = t.expm1().unwrap().to_f32_vec().unwrap();
        assert!((e[0] - 0.0).abs() < 1e-5); // exp(0)-1 = 0
        assert!((e[1] - (1.0_f32.exp() - 1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_log2_log10() {
        let t = Tensor::from_f32(&[1.0, 8.0, 100.0], &[3], test_device()).unwrap();
        let l2 = t.log2().unwrap().to_f32_vec().unwrap();
        assert!((l2[0] - 0.0).abs() < 1e-5);
        assert!((l2[1] - 3.0).abs() < 1e-4);

        let l10 = t.log10().unwrap().to_f32_vec().unwrap();
        assert!((l10[0] - 0.0).abs() < 1e-5);
        assert!((l10[2] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_eq_ne_scalar() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let eq = t.eq_scalar(2.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(eq, vec![0.0, 1.0, 0.0]);
        let ne = t.ne_scalar(2.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(ne, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_isnan_isinf() {
        let t = Tensor::from_f32(&[1.0, f32::NAN, f32::INFINITY], &[3], test_device()).unwrap();
        let nan = t.isnan().unwrap().to_f32_vec().unwrap();
        assert_eq!(nan, vec![0.0, 1.0, 0.0]);
        let inf = t.isinf().unwrap().to_f32_vec().unwrap();
        assert_eq!(inf, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_logical_ops() {
        let a = Tensor::from_f32(&[1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[0.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let and = a.logical_and(&b).unwrap().to_f32_vec().unwrap();
        assert_eq!(and, vec![0.0, 0.0, 1.0]);
        let or = a.logical_or(&b).unwrap().to_f32_vec().unwrap();
        assert_eq!(or, vec![1.0, 0.0, 1.0]);
        let not = a.logical_not().unwrap().to_f32_vec().unwrap();
        assert_eq!(not, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_any_all() {
        let t = Tensor::from_f32(&[0.0, 0.0, 1.0], &[3], test_device()).unwrap();
        assert!((t.any().unwrap().item().unwrap() - 1.0).abs() < 1e-5);
        assert!((t.all().unwrap().item().unwrap() - 0.0).abs() < 1e-5);

        let all_true = Tensor::from_f32(&[1.0, 1.0], &[2], test_device()).unwrap();
        assert!((all_true.all().unwrap().item().unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_atan2() {
        let y = Tensor::from_f32(&[1.0, 0.0], &[2], test_device()).unwrap();
        let x = Tensor::from_f32(&[0.0, 1.0], &[2], test_device()).unwrap();
        let result = y.atan2(&x).unwrap().to_f32_vec().unwrap();
        // atan2(1, 0) = pi/2
        assert!((result[0] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        // atan2(0, 1) = 0
        assert!((result[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_maximum_minimum() {
        let a = Tensor::from_f32(&[1.0, 5.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[4.0, 2.0, 3.0], &[3], test_device()).unwrap();
        assert_eq!(a.maximum(&b).unwrap().to_f32_vec().unwrap(), vec![4.0, 5.0, 3.0]);
        assert_eq!(a.minimum(&b).unwrap().to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_argsort() {
        let t = Tensor::from_f32(&[3.0, 1.0, 2.0], &[3], test_device()).unwrap();
        let idx = t.argsort(0, false).unwrap().to_i64_vec().unwrap();
        assert_eq!(idx, vec![1, 2, 0]); // ascending: 1.0(1), 2.0(2), 3.0(0)
    }

    #[test]
    fn test_scatter() {
        let base = Tensor::zeros(&[2, 3], test_opts()).unwrap();
        let idx = Tensor::from_i64(&[0, 2, 1, 0], &[2, 2], test_device()).unwrap();
        let src = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let result = base.scatter(1, &idx, &src).unwrap().to_f32_vec().unwrap();
        // Row 0: pos 0 = 1.0, pos 2 = 2.0 -> [1, 0, 2]
        // Row 1: pos 1 = 3.0, pos 0 = 4.0 -> [4, 3, 0]
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[2] - 2.0).abs() < 1e-5);
        assert!((result[3] - 4.0).abs() < 1e-5);
        assert!((result[4] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_randperm() {
        let mut opts = test_opts();
        opts.dtype = DType::Int64;
        let p = Tensor::randperm(5, opts).unwrap();
        assert_eq!(p.shape(), vec![5]);
        // All values 0..5 must be present (it's a permutation).
        let mut vals = p.to_i64_vec().unwrap();
        vals.sort();
        assert_eq!(vals, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_tan() {
        let t = Tensor::from_f32(&[0.0, std::f32::consts::FRAC_PI_4], &[2], test_device()).unwrap();
        let r = t.tan().unwrap().to_f32_vec().unwrap();
        assert!((r[0] - 0.0).abs() < 1e-5);
        assert!((r[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_asin_acos_atan() {
        let t = Tensor::from_f32(&[0.0, 0.5, 1.0], &[3], test_device()).unwrap();
        let as_ = t.asin().unwrap().to_f32_vec().unwrap();
        assert!((as_[0] - 0.0).abs() < 1e-5);
        assert!((as_[1] - std::f32::consts::FRAC_PI_6).abs() < 1e-3);

        let ac = t.acos().unwrap().to_f32_vec().unwrap();
        assert!((ac[0] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        assert!((ac[2] - 0.0).abs() < 1e-5);

        let at = t.atan().unwrap().to_f32_vec().unwrap();
        assert!((at[0] - 0.0).abs() < 1e-5);
        assert!((at[2] - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
    }

    #[test]
    fn test_erf_erfc() {
        let t = Tensor::from_f32(&[0.0, 1.0], &[2], test_device()).unwrap();
        let e = t.erf().unwrap().to_f32_vec().unwrap();
        assert!((e[0] - 0.0).abs() < 1e-5);
        assert!((e[1] - 0.8427).abs() < 1e-3);

        let ec = t.erfc().unwrap().to_f32_vec().unwrap();
        assert!((ec[0] - 1.0).abs() < 1e-5);
        assert!((ec[1] - 0.1573).abs() < 1e-3);
    }

    #[test]
    fn test_trunc_frac() {
        let t = Tensor::from_f32(&[2.7, -1.3], &[2], test_device()).unwrap();
        let tr = t.trunc().unwrap().to_f32_vec().unwrap();
        assert!((tr[0] - 2.0).abs() < 1e-5);
        assert!((tr[1] - (-1.0)).abs() < 1e-5);

        let fr = t.frac().unwrap().to_f32_vec().unwrap();
        assert!((fr[0] - 0.7).abs() < 1e-5);
        assert!((fr[1] - (-0.3)).abs() < 1e-5);
    }

    #[test]
    fn test_fmod() {
        let t = Tensor::from_f32(&[5.0, -5.0, 7.5], &[3], test_device()).unwrap();
        let r = t.fmod(3.0).unwrap().to_f32_vec().unwrap();
        assert!((r[0] - 2.0).abs() < 1e-5);
        assert!((r[1] - (-2.0)).abs() < 1e-5);
        assert!((r[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_remainder() {
        let t = Tensor::from_f32(&[5.0, -5.0], &[2], test_device()).unwrap();
        let r = t.remainder(3.0).unwrap().to_f32_vec().unwrap();
        assert!((r[0] - 2.0).abs() < 1e-5);
        assert!((r[1] - 1.0).abs() < 1e-5); // Python semantics: sign matches divisor
    }

    #[test]
    fn test_lerp() {
        let a = Tensor::from_f32(&[0.0, 10.0], &[2], test_device()).unwrap();
        let b = Tensor::from_f32(&[10.0, 20.0], &[2], test_device()).unwrap();
        let r = a.lerp(&b, 0.3).unwrap().to_f32_vec().unwrap();
        assert!((r[0] - 3.0).abs() < 1e-5);
        assert!((r[1] - 13.0).abs() < 1e-5);
    }

    #[test]
    fn test_isclose() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[1.0, 2.001, 5.0], &[3], test_device()).unwrap();
        let r = a.isclose(&b, 1e-5, 1e-2).unwrap().to_f32_vec().unwrap();
        assert!((r[0] - 1.0).abs() < 1e-5); // exact match
        assert!((r[1] - 1.0).abs() < 1e-5); // within atol=0.01
        assert!((r[2] - 0.0).abs() < 1e-5); // not close
    }

    #[test]
    fn test_addmm() {
        let bias = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let m1 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let m2 = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2], test_device()).unwrap();
        // 1.0 * bias + 1.0 * (m1 @ m2) = bias + m1 (identity)
        let r = bias.addmm(&m1, &m2, 1.0, 1.0).unwrap().to_f32_vec().unwrap();
        assert!((r[0] - 2.0).abs() < 1e-5); // 1 + 1
        assert!((r[1] - 4.0).abs() < 1e-5); // 2 + 2
        assert!((r[2] - 4.0).abs() < 1e-5); // 1 + 3
        assert!((r[3] - 6.0).abs() < 1e-5); // 2 + 4
    }

    #[test]
    fn test_addcmul_addcdiv() {
        let s = Tensor::from_f32(&[1.0, 1.0], &[2], test_device()).unwrap();
        let t1 = Tensor::from_f32(&[2.0, 3.0], &[2], test_device()).unwrap();
        let t2 = Tensor::from_f32(&[4.0, 5.0], &[2], test_device()).unwrap();

        // addcmul: 1 + 0.5 * (2*4) = 5, 1 + 0.5 * (3*5) = 8.5
        let cm = s.addcmul(&t1, &t2, 0.5).unwrap().to_f32_vec().unwrap();
        assert!((cm[0] - 5.0).abs() < 1e-5);
        assert!((cm[1] - 8.5).abs() < 1e-5);

        // addcdiv: 1 + 0.5 * (2/4) = 1.25, 1 + 0.5 * (3/5) = 1.3
        let cd = s.addcdiv(&t1, &t2, 0.5).unwrap().to_f32_vec().unwrap();
        assert!((cd[0] - 1.25).abs() < 1e-5);
        assert!((cd[1] - 1.3).abs() < 1e-5);
    }

    #[test]
    fn test_cumprod() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], test_device()).unwrap();
        let r = t.cumprod(0).unwrap().to_f32_vec().unwrap();
        assert_eq!(r, vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_norm_p_dim() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        // L1 norm along dim 1: [3.0, 7.0]
        let l1 = t.norm_p(1.0, 1, false).unwrap().to_f32_vec().unwrap();
        assert!((l1[0] - 3.0).abs() < 1e-5);
        assert!((l1[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_sum_dims() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], test_device()).unwrap();
        // Sum over both dims should give scalar 21
        let s = t.sum_dims(&[0, 1], false).unwrap();
        assert!((s.item().unwrap() - 21.0).abs() < 1e-5);
    }

    #[test]
    fn test_median() {
        let t = Tensor::from_f32(&[3.0, 1.0, 2.0], &[3], test_device()).unwrap();
        let m = t.median().unwrap().item().unwrap();
        assert!((m - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_median_dim() {
        let t = Tensor::from_f32(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3], test_device()).unwrap();
        let (vals, idxs) = t.median_dim(1, false).unwrap();
        let v = vals.to_f32_vec().unwrap();
        let i = idxs.to_i64_vec().unwrap();
        assert!((v[0] - 2.0).abs() < 1e-5);
        assert!((v[1] - 5.0).abs() < 1e-5);
        assert_eq!(i[0], 2);
        assert_eq!(i[1], 2);
    }

    #[test]
    fn test_count_nonzero() {
        let t = Tensor::from_f32(&[0.0, 1.0, 0.0, 2.0, 3.0], &[5], test_device()).unwrap();
        let c = t.count_nonzero().unwrap().to_i64_vec().unwrap();
        assert_eq!(c[0], 3);
    }

    #[test]
    fn test_nonzero() {
        let t = Tensor::from_f32(&[0.0, 1.0, 0.0, 2.0], &[4], test_device()).unwrap();
        let nz = t.nonzero().unwrap();
        assert_eq!(nz.shape(), vec![2, 1]); // 2 non-zero entries, 1D
        let vals = nz.to_i64_vec().unwrap();
        assert_eq!(vals, vec![1, 3]);
    }

    #[test]
    fn test_unique() {
        let t = Tensor::from_f32(&[3.0, 1.0, 2.0, 1.0, 3.0], &[5], test_device()).unwrap();
        let (u, inv) = t.unique(true, true).unwrap();
        let uv = u.to_f32_vec().unwrap();
        assert_eq!(uv, vec![1.0, 2.0, 3.0]);
        let iv = inv.to_i64_vec().unwrap();
        // 3->2, 1->0, 2->1, 1->0, 3->2
        assert_eq!(iv, vec![2, 0, 1, 0, 2]);
    }

    #[test]
    fn test_searchsorted() {
        let sorted = Tensor::from_f32(&[1.0, 3.0, 5.0, 7.0], &[4], test_device()).unwrap();
        let vals = Tensor::from_f32(&[2.0, 4.0, 6.0], &[3], test_device()).unwrap();
        let idx = sorted.searchsorted(&vals).unwrap().to_i64_vec().unwrap();
        assert_eq!(idx, vec![1, 2, 3]);
    }
}
