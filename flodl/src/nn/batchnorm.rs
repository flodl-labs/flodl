use std::cell::Cell;

use crate::autograd::Variable;
use crate::tensor::{Result, Tensor, TensorError, TensorOptions};

use super::buffer::Buffer;
use super::parameter::Parameter;
use super::Module;

/// Batch normalization over the first (batch) dimension.
///
/// Uses a single fused `torch::batch_norm` kernel (1 autograd node).
/// Training mode: uses batch statistics, updates running stats in-place.
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
}

/// Batch normalization for 4D `[B, C, H, W]` inputs (conv layers).
///
/// Uses fused `torch::batch_norm` which handles 4D input natively —
/// normalizes over `(B, H, W)` dimensions per channel, matching
/// PyTorch's `nn.BatchNorm2d`.
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
        // Fused batch_norm handles 4D input natively
        self.inner.forward(input)
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
        let training = self.training.get();
        if training {
            let batch_size = input.shape()[0];
            if batch_size < 2 {
                return Err(TensorError::new(
                    "BatchNorm requires batch_size >= 2 in training mode \
                     (Bessel's correction divides by batch_size-1)"
                ));
            }
        }

        // Single fused call: computes norm, applies affine, updates running stats.
        // Running stats are updated in-place via shared storage when training=true.
        let rm = self.running_mean.get();
        let rv = self.running_var.get();
        let result = input.data().batch_norm(
            Some(&self.weight.variable.data()),
            Some(&self.bias.variable.data()),
            Some(&rm), Some(&rv),
            training, self.momentum, self.eps,
        )?;
        Ok(Variable::wrap(result))
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
