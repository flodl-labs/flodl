use crate::autograd::{Variable, conv_transpose1d};
use crate::tensor::{Result, Device};

use super::init::{kaiming_uniform, uniform_bias};
use super::parameter::Parameter;
use super::Module;

/// Transposed 1D convolution (deconvolution) layer.
///
/// Weight shape: `[in_channels, out_channels / groups, kernel_size]`.
/// Bias shape: `[out_channels]` (optional).
pub struct ConvTranspose1d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub stride: i64,
    pub padding: i64,
    pub output_padding: i64,
    pub dilation: i64,
    pub groups: i64,
}

impl ConvTranspose1d {
    /// Create a ConvTranspose1d layer with default settings and bias.
    pub fn new(
        in_channels: i64, out_channels: i64, kernel_size: i64,
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, true, 1, 0, 0, 1, 1, Device::CPU)
    }

    /// Create a ConvTranspose1d layer on a specific device.
    pub fn on_device(
        in_channels: i64, out_channels: i64, kernel_size: i64, device: Device,
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, true, 1, 0, 0, 1, 1, device)
    }

    /// Fully configurable ConvTranspose1d constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        in_channels: i64, out_channels: i64, kernel_size: i64,
        with_bias: bool,
        stride: i64, padding: i64, output_padding: i64,
        dilation: i64, groups: i64, device: Device,
    ) -> Result<Self> {
        // Note: weight shape is [in_channels, out_channels/groups, K]
        let shape = [in_channels, out_channels / groups, kernel_size];
        let fan_in = in_channels * kernel_size;

        let weight_data = kaiming_uniform(&shape, fan_in, 5.0_f64.sqrt(), device)?;
        let weight = Variable::new(weight_data, true);

        let bias = if with_bias {
            let bias_data = uniform_bias(fan_in, &[out_channels], device)?;
            Some(Parameter {
                variable: Variable::new(bias_data, true),
                name: "bias".into(),
            })
        } else {
            None
        };

        Ok(ConvTranspose1d {
            weight: Parameter { variable: weight, name: "weight".into() },
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        })
    }
}

impl Module for ConvTranspose1d {
    fn name(&self) -> &str { "conv_t1d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        conv_transpose1d(
            input,
            &self.weight.variable,
            self.bias.as_ref().map(|b| &b.variable),
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
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
