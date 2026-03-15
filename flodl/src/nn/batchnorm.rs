use std::cell::Cell;

use crate::autograd::Variable;
use crate::tensor::{Result, Tensor, TensorError, TensorOptions};

use super::buffer::Buffer;
use super::parameter::Parameter;
use super::Module;

/// Batch normalization over the first (batch) dimension.
///
/// Training mode: uses batch statistics, updates running stats.
/// Eval mode: uses running statistics (persisted via `buffers()`).
pub struct BatchNorm {
    pub weight: Parameter,   // gamma
    pub bias: Parameter,     // beta
    running_mean: Buffer,
    running_var: Buffer,
    #[allow(dead_code)]
    num_features: i64,
    eps: f64,
    momentum: f64,
    training: Cell<bool>,
}

impl BatchNorm {
    /// Create a BatchNorm layer for `num_features` channels/features on CPU.
    pub fn new(num_features: i64) -> Result<Self> {
        Self::on_device(num_features, crate::tensor::Device::CPU)
    }

    /// Create a BatchNorm layer on a specific device.
    pub fn on_device(num_features: i64, device: crate::tensor::Device) -> Result<Self> {
        let opts = TensorOptions { dtype: crate::tensor::DType::Float32, device };
        let weight = Variable::new(Tensor::ones(&[num_features], opts)?, true);
        let bias = Variable::new(Tensor::zeros(&[num_features], opts)?, true);

        Ok(BatchNorm {
            weight: Parameter { variable: weight, name: "weight".into() },
            bias: Parameter { variable: bias, name: "bias".into() },
            running_mean: Buffer::new(Tensor::zeros(&[num_features], opts)?, "running_mean"),
            running_var: Buffer::new(Tensor::ones(&[num_features], opts)?, "running_var"),
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            training: Cell::new(true),
        })
    }

    fn update_running_stats(&self, batch_mean: &Tensor, batch_var: &Tensor, batch_size: i64) -> Result<()> {
        let m = self.momentum;
        // Bessel's correction for unbiased variance estimate
        let correction = batch_size as f64 / (batch_size as f64 - 1.0);

        let rm = self.running_mean.get();
        self.running_mean.set(rm.mul_scalar(1.0 - m)?.add(&batch_mean.mul_scalar(m)?)?);

        let rv = self.running_var.get();
        self.running_var.set(rv.mul_scalar(1.0 - m)?.add(&batch_var.mul_scalar(m * correction)?)?);

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
    /// Create a BatchNorm2d layer for `num_channels` channels on CPU.
    pub fn new(num_channels: i64) -> Result<Self> {
        Ok(Self { inner: BatchNorm::new(num_channels)? })
    }

    /// Create a BatchNorm2d layer on a specific device.
    pub fn on_device(num_channels: i64, device: crate::tensor::Device) -> Result<Self> {
        Ok(Self { inner: BatchNorm::on_device(num_channels, device)? })
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

    fn buffers(&self) -> Vec<Buffer> {
        self.inner.buffers()
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
            let mean = Variable::new(self.running_mean.get(), false);
            let var = Variable::new(self.running_var.get(), false);
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

        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn buffers(&self) -> Vec<Buffer> {
        vec![self.running_mean.clone(), self.running_var.clone()]
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }

    fn move_to_device(&self, device: crate::tensor::Device) {
        let _ = self.running_mean.to_device(device);
        let _ = self.running_var.to_device(device);
    }
}
