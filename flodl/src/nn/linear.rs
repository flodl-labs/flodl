use crate::autograd::{self, Variable};
use crate::tensor::{Device, Result};

use super::init;
use super::parameter::Parameter;
use super::Module;

/// Fully connected layer: `y = x @ W^T + b`.
///
/// Weight shape: `[out_features, in_features]`.
/// Bias shape: `[out_features]` (optional).
///
/// Input shape: `[batch, in_features]`.
/// Output shape: `[batch, out_features]`.
///
/// ```ignore
/// let layer = Linear::new(4, 2)?;
/// let x = Variable::new(Tensor::randn(&[8, 4], opts)?, false);
/// let y = layer.forward(&x)?;
/// assert_eq!(y.shape(), vec![8, 2]);
/// ```
pub struct Linear {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
}

impl Linear {
    /// Create a linear layer on CPU with bias.
    pub fn new(in_features: i64, out_features: i64) -> Result<Self> {
        Self::on_device(in_features, out_features, Device::CPU)
    }

    /// Create a linear layer on a specific device with bias.
    pub fn on_device(in_features: i64, out_features: i64, device: Device) -> Result<Self> {
        let w = init::kaiming_uniform(&[out_features, in_features], in_features, 5.0_f64.sqrt(), device)?;
        let b = init::uniform_bias(in_features, &[out_features], device)?;
        Ok(Linear {
            weight: Parameter::new(w, "weight"),
            bias: Some(Parameter::new(b, "bias")),
        })
    }

    /// Create a linear layer without bias on CPU.
    pub fn no_bias(in_features: i64, out_features: i64) -> Result<Self> {
        Self::no_bias_on_device(in_features, out_features, Device::CPU)
    }

    /// Create a linear layer without bias on a specific device.
    pub fn no_bias_on_device(in_features: i64, out_features: i64, device: Device) -> Result<Self> {
        let w = init::kaiming_uniform(&[out_features, in_features], in_features, 5.0_f64.sqrt(), device)?;
        Ok(Linear {
            weight: Parameter::new(w, "weight"),
            bias: None,
        })
    }
}

impl Module for Linear {
    fn name(&self) -> &str { "linear" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::linear(
            input,
            &self.weight.variable,
            self.bias.as_ref().map(|b| &b.variable),
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device, test_opts};

    #[test]
    fn test_linear_forward_shape() {
        let dev = test_device();
        let layer = Linear::on_device(4, 2, dev).unwrap();
        let x = Variable::new(Tensor::randn(&[8, 4], test_opts()).unwrap(), false);
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![8, 2]);
    }

    #[test]
    fn test_linear_parameters_with_bias() {
        let layer = Linear::on_device(4, 2, test_device()).unwrap();
        let params = layer.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].variable.shape(), vec![2, 4]); // weight
        assert_eq!(params[1].variable.shape(), vec![2]);     // bias
    }

    #[test]
    fn test_linear_no_bias() {
        let layer = Linear::no_bias_on_device(4, 2, test_device()).unwrap();
        let params = layer.parameters();
        assert_eq!(params.len(), 1);
        assert!(layer.bias.is_none());

        let x = Variable::new(Tensor::randn(&[3, 4], test_opts()).unwrap(), false);
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![3, 2]);
    }

    #[test]
    fn test_linear_gradient_flow() {
        let dev = test_device();
        let layer = Linear::on_device(3, 2, dev).unwrap();
        let x = Variable::new(Tensor::randn(&[4, 3], test_opts()).unwrap(), false);
        let y = layer.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let params = layer.parameters();
        assert!(params[0].variable.grad().is_some(), "weight should have gradient");
        assert!(params[1].variable.grad().is_some(), "bias should have gradient");
    }

    #[test]
    fn test_linear_on_device() {
        let dev = test_device();
        let layer = Linear::on_device(4, 2, dev).unwrap();
        assert_eq!(layer.weight.variable.device(), dev);
        if let Some(ref b) = layer.bias {
            assert_eq!(b.variable.device(), dev);
        }
    }

    #[test]
    fn test_linear_name() {
        let layer = Linear::new(4, 2).unwrap();
        assert_eq!(layer.name(), "linear");
    }
}
