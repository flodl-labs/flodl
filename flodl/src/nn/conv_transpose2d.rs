use crate::autograd::{Variable, conv_transpose2d};
use crate::tensor::{Result, Device};

use super::init::{kaiming_uniform, uniform_bias};
use super::parameter::Parameter;
use super::Module;

/// Transposed 2D convolution (deconvolution) layer.
///
/// Weight shape: `[in_channels, out_channels / groups, kernel_h, kernel_w]`.
/// Bias shape: `[out_channels]` (optional).
pub struct ConvTranspose2d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub stride: [i64; 2],
    pub padding: [i64; 2],
    pub output_padding: [i64; 2],
    pub dilation: [i64; 2],
    pub groups: i64,
}

impl ConvTranspose2d {
    /// Create a ConvTranspose2d layer with default settings and bias.
    pub fn new(
        in_channels: i64, out_channels: i64, kernel_size: i64,
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, true, [1, 1], [0, 0], [0, 0], [1, 1], 1, Device::CPU)
    }

    /// Fully configurable ConvTranspose2d constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        in_channels: i64, out_channels: i64, kernel_size: i64,
        with_bias: bool,
        stride: [i64; 2], padding: [i64; 2], output_padding: [i64; 2],
        dilation: [i64; 2], groups: i64, device: Device,
    ) -> Result<Self> {
        // Note: weight shape is [in_channels, out_channels/groups, kH, kW]
        let shape = [in_channels, out_channels / groups, kernel_size, kernel_size];
        let fan_in = in_channels * kernel_size * kernel_size;

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

        Ok(ConvTranspose2d {
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

impl Module for ConvTranspose2d {
    fn name(&self) -> &str { "conv_t2d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        conv_transpose2d(
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
