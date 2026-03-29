use crate::autograd::Variable;
use crate::tensor::Result;

use super::Module;

/// Identity pass-through module. Returns its input unchanged.
///
/// Useful as a tagging entry point in graphs:
/// ```ignore
/// FlowBuilder::from(Identity).tag("image")
/// ```
pub struct Identity;

impl Default for Identity {
    fn default() -> Self {
        Identity
    }
}

impl Identity {
    /// Create an Identity module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Identity {
    fn name(&self) -> &str { "identity" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        Ok(input.clone())
    }
}

/// ReLU activation: `max(0, x)`. Zeroes negative values.
pub struct ReLU;

impl Default for ReLU {
    fn default() -> Self {
        ReLU
    }
}

impl ReLU {
    /// Create a ReLU activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for ReLU {
    fn name(&self) -> &str { "relu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.relu()
    }
}

/// Sigmoid activation: `1 / (1 + exp(-x))`. Maps to (0, 1).
pub struct Sigmoid;

impl Default for Sigmoid {
    fn default() -> Self {
        Sigmoid
    }
}

impl Sigmoid {
    /// Create a Sigmoid activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Sigmoid {
    fn name(&self) -> &str { "sigmoid" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.sigmoid()
    }
}

/// Tanh activation: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`. Maps to (-1, 1).
pub struct Tanh;

impl Default for Tanh {
    fn default() -> Self {
        Tanh
    }
}

impl Tanh {
    /// Create a Tanh activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Tanh {
    fn name(&self) -> &str { "tanh" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.tanh()
    }
}

/// GELU activation (Gaussian Error Linear Unit).
///
/// Uses the exact form: `0.5 * x * (1 + erf(x / sqrt(2)))`
pub struct GELU;

impl Default for GELU {
    fn default() -> Self {
        GELU
    }
}

impl GELU {
    /// Create a GELU activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for GELU {
    fn name(&self) -> &str { "gelu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.gelu()
    }
}

/// Sigmoid Linear Unit (Swish): `x * sigmoid(x)`.
/// Self-gated activation with smooth gradient flow.
pub struct SiLU;

impl Default for SiLU {
    fn default() -> Self {
        SiLU
    }
}

impl SiLU {
    /// Create a SiLU activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for SiLU {
    fn name(&self) -> &str { "silu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.silu()
    }
}

/// Leaky ReLU: `max(0, x) + negative_slope * min(0, x)`.
pub struct LeakyReLU {
    negative_slope: f64,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        LeakyReLU { negative_slope: 0.01 }
    }
}

impl LeakyReLU {
    /// Create a LeakyReLU with the given negative slope (default: 0.01).
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
}

impl Module for LeakyReLU {
    fn name(&self) -> &str { "leaky_relu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.leaky_relu(self.negative_slope)
    }
}

/// ELU: `max(0, x) + min(0, alpha * (exp(x) - 1))`.
pub struct ELU {
    alpha: f64,
}

impl Default for ELU {
    fn default() -> Self {
        ELU { alpha: 1.0 }
    }
}

impl ELU {
    /// Create an ELU with the given alpha (default: 1.0).
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Module for ELU {
    fn name(&self) -> &str { "elu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.elu(self.alpha)
    }
}

/// Softplus: smooth approximation to ReLU.
/// `(1/beta) * log(1 + exp(beta * x))`, reverts to linear above `threshold`.
pub struct Softplus {
    beta: f64,
    threshold: f64,
}

impl Default for Softplus {
    fn default() -> Self {
        Softplus { beta: 1.0, threshold: 20.0 }
    }
}

impl Softplus {
    /// Create a Softplus with given beta and threshold (defaults: 1.0, 20.0).
    pub fn new(beta: f64, threshold: f64) -> Self {
        Self { beta, threshold }
    }
}

impl Module for Softplus {
    fn name(&self) -> &str { "softplus" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.softplus(self.beta, self.threshold)
    }
}

/// Mish: `x * tanh(softplus(x))`. Self-regularizing activation.
pub struct Mish;

impl Default for Mish {
    fn default() -> Self {
        Mish
    }
}

impl Mish {
    /// Create a Mish activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Mish {
    fn name(&self) -> &str { "mish" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.mish()
    }
}

/// Softmax along a dimension. Output sums to 1.
pub struct Softmax {
    dim: i32,
}

impl Softmax {
    /// Create a Softmax module along the given dimension.
    pub fn new(dim: i32) -> Self {
        Self { dim }
    }
}

impl Module for Softmax {
    fn name(&self) -> &str { "softmax" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.softmax(self.dim)
    }
}

/// Log-softmax along a dimension. Numerically stable `log(softmax(x))`.
pub struct LogSoftmax {
    dim: i32,
}

impl LogSoftmax {
    /// Create a LogSoftmax module along the given dimension.
    pub fn new(dim: i32) -> Self {
        Self { dim }
    }
}

impl Module for LogSoftmax {
    fn name(&self) -> &str { "log_softmax" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.log_softmax(self.dim)
    }
}

/// Flatten dimensions into a single dimension.
/// Default: flattens all dims except batch (start=1, end=-1).
pub struct Flatten {
    start_dim: i32,
    end_dim: i32,
}

impl Default for Flatten {
    fn default() -> Self {
        Flatten { start_dim: 1, end_dim: -1 }
    }
}

impl Flatten {
    /// Create a Flatten module with custom start and end dimensions.
    pub fn new(start_dim: i32, end_dim: i32) -> Self {
        Self { start_dim, end_dim }
    }
}

impl Module for Flatten {
    fn name(&self) -> &str { "flatten" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.flatten(self.start_dim, self.end_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device};

    #[test]
    fn test_leaky_relu_module() {
        let m = LeakyReLU::new(0.2);
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        assert!((y[0] - (-0.2)).abs() < 1e-5);
        assert!((y[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_leaky_relu_default() {
        let m = LeakyReLU::default();
        let t = Tensor::from_f32(&[-1.0], &[1], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        assert!((y[0] - (-0.01)).abs() < 1e-5);
    }

    #[test]
    fn test_elu_module() {
        let m = ELU::default();
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        assert!(y[0] < 0.0); // negative for negative input
        assert!((y[1] - 0.0).abs() < 1e-5);
        assert!((y[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softplus_module() {
        let m = Softplus::default();
        let t = Tensor::from_f32(&[0.0], &[1], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        assert!((y[0] - std::f32::consts::LN_2).abs() < 1e-3);
    }

    #[test]
    fn test_mish_module() {
        let m = Mish::new();
        let t = Tensor::from_f32(&[0.0, 1.0], &[2], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        assert!((y[0] - 0.0).abs() < 1e-5);
        assert!((y[1] - 0.8651).abs() < 1e-3);
    }

    #[test]
    fn test_softmax_module() {
        let m = Softmax::new(-1);
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        let sum: f32 = y.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_module() {
        let m = LogSoftmax::new(-1);
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        // log_softmax values should all be negative
        assert!(y.iter().all(|&v| v < 0.0));
    }

    #[test]
    fn test_flatten_module() {
        let m = Flatten::default();
        let t = Tensor::from_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2], test_device(),
        ).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap();
        assert_eq!(y.data().shape(), vec![2, 4]); // batch dim preserved
    }

    #[test]
    fn test_flatten_all() {
        let m = Flatten::new(0, -1);
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap();
        assert_eq!(y.data().shape(), vec![4]);
    }
}
