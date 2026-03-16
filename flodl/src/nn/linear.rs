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

    /// Create a linear layer without bias.
    pub fn no_bias(in_features: i64, out_features: i64) -> Result<Self> {
        let w = init::kaiming_uniform(&[out_features, in_features], in_features, 5.0_f64.sqrt(), Device::CPU)?;
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
