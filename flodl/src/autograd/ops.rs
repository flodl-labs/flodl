//! Variable operations backed by libtorch native autograd.
//!
//! Every op calls the Tensor FFI method and wraps the result.
//! libtorch's C++ autograd engine tracks the computation graph
//! and computes gradients natively — no Rust-side backward closures.

use crate::tensor::{Result, Tensor};

use super::variable::Variable;

impl Variable {
    // --- Arithmetic ---

    pub fn add(&self, other: &Variable) -> Result<Variable> {
        let result = self.data().add(&other.data())?;
        Ok(Variable::wrap(result))
    }

    pub fn sub(&self, other: &Variable) -> Result<Variable> {
        let result = self.data().sub(&other.data())?;
        Ok(Variable::wrap(result))
    }

    pub fn mul(&self, other: &Variable) -> Result<Variable> {
        let result = self.data().mul(&other.data())?;
        Ok(Variable::wrap(result))
    }

    pub fn div(&self, other: &Variable) -> Result<Variable> {
        let result = self.data().div(&other.data())?;
        Ok(Variable::wrap(result))
    }

    pub fn matmul(&self, other: &Variable) -> Result<Variable> {
        let result = self.data().matmul(&other.data())?;
        Ok(Variable::wrap(result))
    }

    pub fn mul_scalar(&self, scalar: f64) -> Result<Variable> {
        let result = self.data().mul_scalar(scalar)?;
        Ok(Variable::wrap(result))
    }

    pub fn div_scalar(&self, scalar: f64) -> Result<Variable> {
        let result = self.data().div_scalar(scalar)?;
        Ok(Variable::wrap(result))
    }

    pub fn add_scalar(&self, scalar: f64) -> Result<Variable> {
        let result = self.data().add_scalar(scalar)?;
        Ok(Variable::wrap(result))
    }

    pub fn neg(&self) -> Result<Variable> {
        let result = self.data().neg()?;
        Ok(Variable::wrap(result))
    }

    // --- Activations ---

    pub fn relu(&self) -> Result<Variable> {
        let result = self.data().relu()?;
        Ok(Variable::wrap(result))
    }

    pub fn sigmoid(&self) -> Result<Variable> {
        let result = self.data().sigmoid()?;
        Ok(Variable::wrap(result))
    }

    pub fn tanh_act(&self) -> Result<Variable> {
        let result = self.data().tanh_op()?;
        Ok(Variable::wrap(result))
    }

    pub fn gelu(&self) -> Result<Variable> {
        let result = self.data().gelu()?;
        Ok(Variable::wrap(result))
    }

    pub fn silu(&self) -> Result<Variable> {
        let result = self.data().silu()?;
        Ok(Variable::wrap(result))
    }

    // --- Reductions ---

    pub fn sum(&self) -> Result<Variable> {
        let result = self.data().sum()?;
        Ok(Variable::wrap(result))
    }

    pub fn mean(&self) -> Result<Variable> {
        let result = self.data().mean()?;
        Ok(Variable::wrap(result))
    }

    pub fn sum_dim(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.data().sum_dim(dim, keepdim)?;
        Ok(Variable::wrap(result))
    }

    pub fn mean_dim(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.data().mean_dim(dim, keepdim)?;
        Ok(Variable::wrap(result))
    }

    pub fn min(&self) -> Result<Variable> {
        let result = self.data().min()?;
        Ok(Variable::wrap(result))
    }

    pub fn max(&self) -> Result<Variable> {
        let result = self.data().max()?;
        Ok(Variable::wrap(result))
    }

    pub fn var(&self) -> Result<Variable> {
        let result = self.data().var()?;
        Ok(Variable::wrap(result))
    }

    pub fn std(&self) -> Result<Variable> {
        let result = self.data().std()?;
        Ok(Variable::wrap(result))
    }

    pub fn var_dim(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.data().var_dim(dim, keepdim)?;
        Ok(Variable::wrap(result))
    }

    pub fn std_dim(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.data().std_dim(dim, keepdim)?;
        Ok(Variable::wrap(result))
    }

    // --- Softmax ---

    pub fn softmax(&self, dim: i32) -> Result<Variable> {
        let result = self.data().softmax(dim)?;
        Ok(Variable::wrap(result))
    }

    pub fn log_softmax(&self, dim: i32) -> Result<Variable> {
        let result = self.data().log_softmax(dim)?;
        Ok(Variable::wrap(result))
    }

    // --- Element-wise math ---

    pub fn exp(&self) -> Result<Variable> {
        let result = self.data().exp()?;
        Ok(Variable::wrap(result))
    }

    pub fn log(&self) -> Result<Variable> {
        let result = self.data().log()?;
        Ok(Variable::wrap(result))
    }

    pub fn sqrt(&self) -> Result<Variable> {
        let result = self.data().sqrt()?;
        Ok(Variable::wrap(result))
    }

    pub fn abs(&self) -> Result<Variable> {
        let result = self.data().abs()?;
        Ok(Variable::wrap(result))
    }

    pub fn pow_scalar(&self, exponent: f64) -> Result<Variable> {
        let result = self.data().pow_scalar(exponent)?;
        Ok(Variable::wrap(result))
    }

    pub fn triu(&self, diagonal: i64) -> Result<Variable> {
        let result = self.data().triu(diagonal)?;
        Ok(Variable::wrap(result))
    }

    pub fn sin(&self) -> Result<Variable> {
        let result = self.data().sin()?;
        Ok(Variable::wrap(result))
    }

    pub fn cos(&self) -> Result<Variable> {
        let result = self.data().cos()?;
        Ok(Variable::wrap(result))
    }

    pub fn sign(&self) -> Result<Variable> {
        let result = self.data().sign()?;
        Ok(Variable::wrap(result))
    }

    pub fn floor(&self) -> Result<Variable> {
        let result = self.data().floor()?;
        Ok(Variable::wrap(result))
    }

    pub fn ceil(&self) -> Result<Variable> {
        let result = self.data().ceil()?;
        Ok(Variable::wrap(result))
    }

    pub fn round(&self) -> Result<Variable> {
        let result = self.data().round()?;
        Ok(Variable::wrap(result))
    }

    pub fn reciprocal(&self) -> Result<Variable> {
        let result = self.data().reciprocal()?;
        Ok(Variable::wrap(result))
    }

    pub fn clamp(&self, min: f64, max: f64) -> Result<Variable> {
        let result = self.data().clamp(min, max)?;
        Ok(Variable::wrap(result))
    }

    // --- Shape operations ---

    pub fn reshape(&self, shape: &[i64]) -> Result<Variable> {
        let result = self.data().reshape(shape)?;
        Ok(Variable::wrap(result))
    }

    pub fn transpose(&self, dim0: i32, dim1: i32) -> Result<Variable> {
        let result = self.data().transpose(dim0, dim1)?;
        Ok(Variable::wrap(result))
    }

    pub fn permute(&self, dims: &[i64]) -> Result<Variable> {
        let result = self.data().permute(dims)?;
        Ok(Variable::wrap(result))
    }

    pub fn squeeze(&self, dim: i32) -> Result<Variable> {
        let result = self.data().squeeze(dim)?;
        Ok(Variable::wrap(result))
    }

    pub fn unsqueeze(&self, dim: i32) -> Result<Variable> {
        let result = self.data().unsqueeze(dim)?;
        Ok(Variable::wrap(result))
    }

    pub fn flatten(&self, start_dim: i32, end_dim: i32) -> Result<Variable> {
        let result = self.data().flatten(start_dim, end_dim)?;
        Ok(Variable::wrap(result))
    }

    pub fn expand(&self, shape: &[i64]) -> Result<Variable> {
        let result = self.data().expand(shape)?;
        Ok(Variable::wrap(result))
    }

    // --- Indexing & slicing ---

    pub fn narrow(&self, dim: i32, start: i64, length: i64) -> Result<Variable> {
        let result = self.data().narrow(dim, start, length)?;
        Ok(Variable::wrap(result))
    }

    pub fn select(&self, dim: i32, index: i64) -> Result<Variable> {
        let result = self.data().select(dim, index)?;
        Ok(Variable::wrap(result))
    }

    pub fn index_select(&self, dim: i32, index: &Tensor) -> Result<Variable> {
        let result = self.data().index_select(dim, index)?;
        Ok(Variable::wrap(result))
    }

    pub fn gather(&self, dim: i32, index: &Tensor) -> Result<Variable> {
        let result = self.data().gather(dim, index)?;
        Ok(Variable::wrap(result))
    }

    pub fn cat(&self, other: &Variable, dim: i32) -> Result<Variable> {
        let result = self.data().cat(&other.data(), dim)?;
        Ok(Variable::wrap(result))
    }

    pub fn stack(vars: &[Variable], dim: i32) -> Result<Variable> {
        let tensors: Vec<Tensor> = vars.iter().map(|v| v.data()).collect();
        let refs: Vec<&Tensor> = tensors.iter().collect();
        let result = Tensor::stack(&refs, dim)?;
        Ok(Variable::wrap(result))
    }

    pub fn chunk(&self, chunks: i32, dim: i32) -> Result<Vec<Variable>> {
        let tensors = self.data().chunk(chunks, dim)?;
        Ok(tensors.into_iter().map(Variable::wrap).collect())
    }

    pub fn repeat(&self, repeats: &[i64]) -> Result<Variable> {
        let result = self.data().repeat(repeats)?;
        Ok(Variable::wrap(result))
    }

    pub fn pad(&self, padding: &[i64], value: f64) -> Result<Variable> {
        let result = self.data().pad(padding, value)?;
        Ok(Variable::wrap(result))
    }

    // --- Sorting ---

    pub fn topk(&self, k: i64, dim: i32, largest: bool, sorted: bool) -> Result<(Variable, Tensor)> {
        let (values, indices) = self.data().topk(k, dim, largest, sorted)?;
        Ok((Variable::wrap(values), indices))
    }

    pub fn sort(&self, dim: i32, descending: bool) -> Result<(Variable, Tensor)> {
        let (values, indices) = self.data().sort(dim, descending)?;
        Ok((Variable::wrap(values), indices))
    }
}

// --- Standalone differentiable ops ---

/// Layer normalization with autograd support.
pub fn layer_norm(
    input: &Variable,
    weight: &Variable,
    bias: &Variable,
    normalized_size: i64,
    eps: f64,
) -> Result<Variable> {
    let (output, _mean, _rstd) = input.data().native_layer_norm(
        &weight.data(), &bias.data(), normalized_size, eps,
    )?;
    Ok(Variable::wrap(output))
}

/// 2D convolution with autograd support.
pub fn conv2d(
    input: &Variable,
    weight: &Variable,
    bias: Option<&Variable>,
    stride: [i64; 2],
    padding: [i64; 2],
    dilation: [i64; 2],
    groups: i64,
) -> Result<Variable> {
    let bias_tensor = bias.map(|b| b.data());
    let result = input.data().conv2d(
        &weight.data(),
        bias_tensor.as_ref(),
        stride, padding, dilation, groups,
    )?;
    Ok(Variable::wrap(result))
}

/// Transposed 2D convolution with autograd support.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    input: &Variable,
    weight: &Variable,
    bias: Option<&Variable>,
    stride: [i64; 2],
    padding: [i64; 2],
    output_padding: [i64; 2],
    dilation: [i64; 2],
    groups: i64,
) -> Result<Variable> {
    let bias_tensor = bias.map(|b| b.data());
    let result = input.data().conv_transpose2d(
        &weight.data(),
        bias_tensor.as_ref(),
        stride, padding, output_padding, dilation, groups,
    )?;
    Ok(Variable::wrap(result))
}

/// Adaptive average pooling with autograd support.
pub fn adaptive_avg_pool2d(
    input: &Variable,
    output_size: [i64; 2],
) -> Result<Variable> {
    let result = input.data().adaptive_avg_pool2d(output_size)?;
    Ok(Variable::wrap(result))
}

/// Grid sampling with autograd support.
pub fn grid_sample(
    input: &Variable,
    grid: &Variable,
    mode: i32,
    padding_mode: i32,
    align_corners: bool,
) -> Result<Variable> {
    let result = input.data().grid_sample(
        &grid.data(), mode, padding_mode, align_corners,
    )?;
    Ok(Variable::wrap(result))
}
