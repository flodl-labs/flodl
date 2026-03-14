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

/// Batch normalization for 4D `[B, C, H, W]` inputs (conv layers).
///
/// Normalizes over the `(B, H, W)` dimensions per channel, matching
/// PyTorch's `nn.BatchNorm2d`. Internally reshapes to `[B*H*W, C]`,
/// applies `BatchNorm`, and reshapes back.
///
/// ```ignore
/// let bn = BatchNorm2d::new(64)?;  // 64 channels
/// let x: Variable  // [B, 64, H, W]
/// let y = bn.forward(&x)?;  // [B, 64, H, W], normalized per channel
/// ```
pub struct BatchNorm2d {
    inner: BatchNorm,
}

impl BatchNorm2d {
    /// Create a BatchNorm2d layer for `num_channels` channels.
    pub fn new(num_channels: i64) -> Result<Self> {
        Ok(Self { inner: BatchNorm::new(num_channels)? })
    }
}

impl Module for BatchNorm2d {
    fn name(&self) -> &str { "batchnorm2d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(TensorError::new(
                "BatchNorm2d: input must be 4D [B, C, H, W]"
            ));
        }
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        // [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        let permuted = input.permute(&[0, 2, 3, 1])?;
        let flat = permuted.reshape(&[b * h * w, c])?;

        // Apply 2D batch norm
        let normed = self.inner.forward(&flat)?;

        // [B*H*W, C] -> [B, H, W, C] -> [B, C, H, W]
        let unflat = normed.reshape(&[b, h, w, c])?;
        unflat.permute(&[0, 3, 1, 2])
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.inner.parameters()
    }

    fn set_training(&self, training: bool) {
        self.inner.set_training(training);
    }

    fn move_to_device(&self, device: crate::tensor::Device) {
        self.inner.move_to_device(device);
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
