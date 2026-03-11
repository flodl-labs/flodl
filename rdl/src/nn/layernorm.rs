use crate::autograd::{Variable, layer_norm};
use crate::tensor::{Result, Tensor};

use super::parameter::Parameter;
use super::Module;

/// Layer normalization over the last dimension.
///
/// Uses native libtorch `layer_norm` for PyTorch numerical parity.
pub struct LayerNorm {
    pub weight: Parameter, // gamma
    pub bias: Parameter,   // beta
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
        layer_norm(input, &self.weight.variable, &self.bias.variable, self.size, self.eps)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
