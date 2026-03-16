use crate::tensor::{Device, Result, Tensor, TensorOptions};

/// Kaiming uniform initialization.
///
/// `a` is the negative slope of the nonlinearity (0 for ReLU, sqrt(5) for PyTorch Linear default).
/// - `a = 0.0` (ReLU):     bound = sqrt(6 / fan_in)       — standard He init
/// - `a = sqrt(5)` (Linear): bound = 1 / sqrt(fan_in)       — matches PyTorch nn.Linear
pub fn kaiming_uniform(shape: &[i64], fan_in: i64, a: f64, device: Device) -> Result<Tensor> {
    let gain = (2.0 / (1.0 + a * a)).sqrt();
    let std = gain / (fan_in as f64).sqrt();
    let bound = 3.0_f64.sqrt() * std;
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::rand(shape, opts)?.mul_scalar(2.0 * bound)?.add_scalar(-bound)
}

/// Kaiming normal initialization.
///
/// `a` is the negative slope of the nonlinearity (0 for ReLU, sqrt(5) for PyTorch Linear default).
/// - `a = 0.0` (ReLU):     std = sqrt(2 / fan_in)           — standard He init
/// - `a = sqrt(5)` (Linear): std = 1 / sqrt(3 * fan_in)       — matches PyTorch nn.Linear
pub fn kaiming_normal(shape: &[i64], fan_in: i64, a: f64, device: Device) -> Result<Tensor> {
    let gain = (2.0 / (1.0 + a * a)).sqrt();
    let std = gain / (fan_in as f64).sqrt();
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::randn(shape, opts)?.mul_scalar(std)
}

/// Xavier (Glorot) uniform initialization (for sigmoid/tanh networks).
/// bound = sqrt(6 / (fan_in + fan_out)), uniform(-bound, bound)
pub fn xavier_uniform(shape: &[i64], fan_in: i64, fan_out: i64, device: Device) -> Result<Tensor> {
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::rand(shape, opts)?.mul_scalar(2.0 * bound)?.add_scalar(-bound)
}

/// Xavier (Glorot) normal initialization (for sigmoid/tanh networks).
/// std = sqrt(2 / (fan_in + fan_out)), normal(0, std)
pub fn xavier_normal(shape: &[i64], fan_in: i64, fan_out: i64, device: Device) -> Result<Tensor> {
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::randn(shape, opts)?.mul_scalar(std)
}

/// Uniform bias initialization.
/// bound = 1 / sqrt(fan_in), uniform(-bound, bound)
pub fn uniform_bias(fan_in: i64, shape: &[i64], device: Device) -> Result<Tensor> {
    let bound = 1.0 / (fan_in as f64).sqrt();
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::rand(shape, opts)?.mul_scalar(2.0 * bound)?.add_scalar(-bound)
}
