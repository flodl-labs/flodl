use std::cell::Cell;

use crate::autograd::Variable;
use crate::tensor::{Result, Tensor, TensorOptions};

use super::Module;

/// Inverted dropout module.
///
/// During training: randomly zeros elements with probability `p`,
/// scales remaining by `1/(1-p)`.
/// During eval: identity function.
pub struct Dropout {
    p: f64,
    training: Cell<bool>,
}

impl Dropout {
    /// Create a dropout module with drop probability `p` (0.0 to 1.0).
    /// Use `set_training(false)` to disable during inference.
    pub fn new(p: f64) -> Self {
        Dropout {
            p,
            training: Cell::new(true),
        }
    }

}

impl Module for Dropout {
    fn name(&self) -> &str { "dropout" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        if !self.training.get() || self.p == 0.0 {
            return Ok(input.clone());
        }

        let shape = input.shape();
        let opts = TensorOptions {
            dtype: input.dtype(),
            device: input.device(),
        };
        // Generate random mask: 1 where rand > p, 0 otherwise
        let rand_tensor = Tensor::rand(&shape, opts)?;
        let mask_tensor = rand_tensor.gt_scalar(self.p)?;
        let mask = Variable::new(mask_tensor, false);

        // Scale by 1/(1-p) for inverted dropout
        let scale = 1.0 / (1.0 - self.p);
        input.mul(&mask)?.mul_scalar(scale)
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }
}

/// 2D channel dropout — drops entire channels (feature maps) at once.
///
/// During training: randomly zeros entire channels with probability `p`,
/// scales remaining by `1/(1-p)`. Mask shape is `[B, C, 1, 1]`.
/// During eval: identity function.
///
/// Expects 4-D input `[B, C, H, W]`.
pub struct Dropout2d {
    p: f64,
    training: Cell<bool>,
}

impl Dropout2d {
    /// Create a 2D dropout module with channel drop probability `p`.
    pub fn new(p: f64) -> Self {
        Dropout2d {
            p,
            training: Cell::new(true),
        }
    }
}

impl Module for Dropout2d {
    fn name(&self) -> &str { "dropout2d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        if !self.training.get() || self.p == 0.0 {
            return Ok(input.clone());
        }

        let shape = input.shape();
        assert!(shape.len() == 4, "Dropout2d expects [B, C, H, W], got {:?}", shape);

        // Mask shape: [B, C, 1, 1] — broadcast across spatial dims
        let mask_shape = [shape[0], shape[1], 1, 1];
        let opts = TensorOptions {
            dtype: input.dtype(),
            device: input.device(),
        };
        let rand_tensor = Tensor::rand(&mask_shape, opts)?;
        let mask_tensor = rand_tensor.gt_scalar(self.p)?;
        let mask = Variable::new(mask_tensor, false);

        let scale = 1.0 / (1.0 - self.p);
        input.mul(&mask)?.mul_scalar(scale)
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DType;

    #[test]
    fn test_dropout2d_whole_channels_zeroed() {
        let d = Dropout2d::new(0.5);
        let opts = TensorOptions { dtype: DType::Float32, device: crate::tensor::test_device() };
        let input = Variable::new(Tensor::ones(&[2, 8, 4, 4], opts).unwrap(), false);

        let output = d.forward(&input).unwrap();
        let data = output.data().to_f32_vec().unwrap();

        // Each channel should be either all-zero or all-scaled
        let h = 4_usize;
        let w = 4_usize;
        let scale = 1.0 / 0.5;
        for b in 0..2_usize {
            for c in 0..8_usize {
                let start = b * 8 * h * w + c * h * w;
                let channel: Vec<f32> = data[start..start + h * w].to_vec();
                let first = channel[0];
                // All elements in channel should be equal (either 0 or scale)
                for &v in &channel {
                    assert!((v - first).abs() < 1e-5,
                        "channel [{},{}] not uniform: {} vs {}", b, c, v, first);
                }
                assert!(first.abs() < 1e-5 || (first - scale as f32).abs() < 1e-5,
                    "channel value should be 0 or {}: got {}", scale, first);
            }
        }
    }

    #[test]
    fn test_dropout2d_eval_identity() {
        let d = Dropout2d::new(0.5);
        d.set_training(false);
        let opts = TensorOptions { dtype: DType::Float32, device: crate::tensor::test_device() };
        let input = Variable::new(Tensor::ones(&[1, 3, 4, 4], opts).unwrap(), false);

        let output = d.forward(&input).unwrap();
        let data = output.data().to_f32_vec().unwrap();
        assert!(data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }
}
