use std::cell::Cell;

use crate::autograd::Variable;
use crate::tensor::{Result, Tensor, TensorOptions};

use super::parameter::Parameter;
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
    pub fn new(p: f64) -> Self {
        Dropout {
            p,
            training: Cell::new(true),
        }
    }

    pub fn set_training(&self, training: bool) {
        self.training.set(training);
    }
}

impl Module for Dropout {
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

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}
