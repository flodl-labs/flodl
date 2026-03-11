use crate::autograd::Variable;
use crate::tensor::{Result, Tensor};

use super::parameter::Parameter;
use super::Module;

/// Layer normalization over the last dimension.
///
/// Computes: `gamma * (x - mean) / sqrt(var + eps) + beta`
pub struct LayerNorm {
    pub weight: Parameter, // gamma
    pub bias: Parameter,   // beta
    #[allow(dead_code)]
    size: i64,
    eps: f64,
}

impl LayerNorm {
    pub fn new(size: i64) -> Result<Self> {
        let weight = Variable::new(Tensor::ones(&[size], Default::default())?, true);
        let bias = Variable::new(Tensor::zeros(&[size], Default::default())?, true);

        Ok(LayerNorm {
            weight: Parameter {
                variable: weight,
                name: "weight".into(),
            },
            bias: Parameter {
                variable: bias,
                name: "bias".into(),
            },
            size,
            eps: 1e-5,
        })
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let ndim = input.shape().len();
        let last_dim = (ndim - 1) as i32;

        // mean along last dim, keepdim for broadcasting
        let count = input.shape()[ndim - 1] as f64;
        let mean = input.sum_dim(last_dim, true)?.mul_scalar(1.0 / count)?;

        // centered
        let centered = input.sub(&mean)?;

        // variance = mean(centered^2)
        let var = centered
            .mul(&centered)?
            .sum_dim(last_dim, true)?
            .mul_scalar(1.0 / count)?;

        // normalize: centered / sqrt(var + eps)
        // sqrt via exp(0.5 * log(x)) since we don't have differentiable sqrt
        let var_eps = var.add_scalar(self.eps)?;
        let std = var_eps.log()?.mul_scalar(0.5)?.exp()?;
        let normalized = centered.div(&std)?;

        // scale and shift: gamma * normalized + beta
        normalized.mul(&self.weight.variable)?.add(&self.bias.variable)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
