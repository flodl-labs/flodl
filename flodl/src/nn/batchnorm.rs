use std::cell::{Cell, RefCell};

use crate::autograd::Variable;
use crate::tensor::{Result, Tensor, TensorError, TensorOptions};

use super::parameter::Parameter;
use super::Module;

/// Batch normalization over the first (batch) dimension.
///
/// Training mode: uses batch statistics, updates running stats.
/// Eval mode: uses running statistics.
pub struct BatchNorm {
    pub weight: Parameter,   // gamma
    pub bias: Parameter,     // beta
    running_mean: RefCell<Tensor>,
    running_var: RefCell<Tensor>,
    #[allow(dead_code)]
    num_features: i64,
    num_batches_tracked: Cell<usize>,
    eps: f64,
    momentum: f64,
    training: Cell<bool>,
}

impl BatchNorm {
    /// Create a BatchNorm layer for `num_features` channels/features.
    pub fn new(num_features: i64) -> Result<Self> {
        let opts = TensorOptions::default();
        let weight = Variable::new(Tensor::ones(&[num_features], opts)?, true);
        let bias = Variable::new(Tensor::zeros(&[num_features], opts)?, true);
        let running_mean = Tensor::zeros(&[num_features], opts)?;
        let running_var = Tensor::ones(&[num_features], opts)?;

        Ok(BatchNorm {
            weight: Parameter { variable: weight, name: "weight".into() },
            bias: Parameter { variable: bias, name: "bias".into() },
            running_mean: RefCell::new(running_mean),
            running_var: RefCell::new(running_var),
            num_features,
            num_batches_tracked: Cell::new(0),
            eps: 1e-5,
            momentum: 0.1,
            training: Cell::new(true),
        })
    }

    fn update_running_stats(&self, batch_mean: &Tensor, batch_var: &Tensor, batch_size: i64) -> Result<()> {
        let m = self.momentum;
        // Bessel's correction for unbiased variance estimate
        let correction = batch_size as f64 / (batch_size as f64 - 1.0);

        let mut rm = self.running_mean.borrow_mut();
        *rm = rm.mul_scalar(1.0 - m)?.add(&batch_mean.mul_scalar(m)?)?;

        let mut rv = self.running_var.borrow_mut();
        *rv = rv.mul_scalar(1.0 - m)?.add(&batch_var.mul_scalar(m * correction)?)?;

        Ok(())
    }
}

impl Module for BatchNorm {
    fn name(&self) -> &str { "batchnorm" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        if !self.training.get() {
            // Eval mode: use running statistics
            if self.num_batches_tracked.get() == 0 {
                return Err(TensorError::new(
                    "BatchNorm has no running statistics — run at least one training batch \
                     or use set_training(true) before forward"
                ));
            }
            let mean = Variable::new(self.running_mean.borrow().clone(), false);
            let var = Variable::new(self.running_var.borrow().clone(), false);
            let std = var.add_scalar(self.eps)?.sqrt()?;
            let normalized = input.sub(&mean)?.div(&std)?;
            return normalized.mul(&self.weight.variable)?.add(&self.bias.variable);
        }

        // Training mode: use batch statistics
        let batch_size = input.shape()[0];
        if batch_size < 2 {
            return Err(TensorError::new(
                "BatchNorm requires batch_size >= 2 in training mode \
                 (Bessel's correction divides by batch_size-1)"
            ));
        }
        let batch_mean = input.mean_dim(0, false)?;
        let centered = input.sub(&batch_mean)?;
        let batch_var = centered.mul(&centered)?.mean_dim(0, false)?;
        let std = batch_var.add_scalar(self.eps)?.sqrt()?;
        let normalized = centered.div(&std)?;
        let output = normalized.mul(&self.weight.variable)?.add(&self.bias.variable)?;

        // Update running stats (detached from graph)
        self.update_running_stats(&batch_mean.data(), &batch_var.data(), batch_size)?;
        self.num_batches_tracked.set(self.num_batches_tracked.get() + 1);

        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }

    fn move_to_device(&self, device: crate::tensor::Device) {
        let mut rm = self.running_mean.borrow_mut();
        if rm.device() != device
            && let Ok(t) = rm.to_device(device)
        {
            *rm = t;
        }
        let mut rv = self.running_var.borrow_mut();
        if rv.device() != device
            && let Ok(t) = rv.to_device(device)
        {
            *rv = t;
        }
    }
}
