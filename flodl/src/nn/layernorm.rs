use crate::autograd::{Variable, layer_norm};
use crate::tensor::{Device, DType, Result, Tensor, TensorOptions};

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
    /// Create a LayerNorm normalizing over the last `size` elements on CPU.
    pub fn new(size: i64) -> Result<Self> {
        Self::on_device(size, Device::CPU)
    }

    /// Create a LayerNorm on a specific device.
    pub fn on_device(size: i64, device: Device) -> Result<Self> {
        let opts = TensorOptions { dtype: DType::Float32, device };
        let weight = Variable::new(Tensor::ones(&[size], opts)?, true);
        let bias = Variable::new(Tensor::zeros(&[size], opts)?, true);

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
    fn name(&self) -> &str { "layernorm" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        layer_norm(input, &self.weight.variable, &self.bias.variable, self.size, self.eps)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device, test_opts};

    #[test]
    fn test_layernorm_forward_shape() {
        let ln = LayerNorm::on_device(8, test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 8], test_opts()).unwrap(), false,
        );
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 8]);
    }

    #[test]
    fn test_layernorm_normalizes() {
        let ln = LayerNorm::on_device(4, test_device()).unwrap();
        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4], test_device()).unwrap(),
            false,
        );
        let y = ln.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // Output should be approximately normalized (mean ~0, std ~1)
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 1e-4, "mean should be ~0, got {mean}");
    }

    #[test]
    fn test_layernorm_3d_input() {
        let ln = LayerNorm::on_device(16, test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 5, 16], test_opts()).unwrap(), false,
        );
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 5, 16]);
    }

    #[test]
    fn test_layernorm_gradient() {
        let ln = LayerNorm::on_device(8, test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[4, 8], test_opts()).unwrap(), true,
        );
        let y = ln.forward(&x).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert!(x.grad().is_some());
        // Weight and bias should also have gradients
        assert!(ln.weight.variable.grad().is_some());
        assert!(ln.bias.variable.grad().is_some());
    }

    #[test]
    fn test_layernorm_parameters() {
        let ln = LayerNorm::on_device(16, test_device()).unwrap();
        let params = ln.parameters();
        assert_eq!(params.len(), 2);
        // Weight should be ones, bias should be zeros
        let w = params[0].variable.data().to_f32_vec().unwrap();
        let b = params[1].variable.data().to_f32_vec().unwrap();
        assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(b.iter().all(|&v| v.abs() < 1e-6));
    }
}
