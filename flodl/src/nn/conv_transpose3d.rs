use crate::autograd::{Variable, conv_transpose3d};
use crate::tensor::{Result, Device};

use super::init::{kaiming_uniform, uniform_bias};
use super::parameter::Parameter;
use super::Module;

/// Transposed 3D convolution (fractionally-strided convolution).
///
/// Weight shape: `[in_channels, out_channels / groups, kD, kH, kW]`.
/// Input: `[batch, in_channels, D, H, W]`.
pub struct ConvTranspose3d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub stride: [i64; 3],
    pub padding: [i64; 3],
    pub output_padding: [i64; 3],
    pub dilation: [i64; 3],
    pub groups: i64,
}

impl ConvTranspose3d {
    pub fn new(
        in_channels: i64, out_channels: i64, kernel_size: [i64; 3],
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, true,
                    [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], 1, Device::CPU)
    }

    pub fn on_device(
        in_channels: i64, out_channels: i64, kernel_size: [i64; 3], device: Device,
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, true,
                    [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], 1, device)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build(
        in_channels: i64, out_channels: i64, kernel_size: [i64; 3],
        with_bias: bool,
        stride: [i64; 3], padding: [i64; 3], output_padding: [i64; 3],
        dilation: [i64; 3], groups: i64, device: Device,
    ) -> Result<Self> {
        let shape = [in_channels, out_channels / groups,
                     kernel_size[0], kernel_size[1], kernel_size[2]];
        let fan_in = (in_channels / groups) * kernel_size[0] * kernel_size[1] * kernel_size[2];

        let weight_data = kaiming_uniform(&shape, fan_in, 5.0_f64.sqrt(), device)?;
        let weight = Variable::new(weight_data, true);

        let bias = if with_bias {
            let bias_data = uniform_bias(fan_in, &[out_channels], device)?;
            Some(Parameter { variable: Variable::new(bias_data, true), name: "bias".into() })
        } else {
            None
        };

        Ok(ConvTranspose3d {
            weight: Parameter { variable: weight, name: "weight".into() },
            bias, stride, padding, output_padding, dilation, groups,
        })
    }
}

impl Module for ConvTranspose3d {
    fn name(&self) -> &str { "conv_transpose3d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        conv_transpose3d(
            input, &self.weight.variable,
            self.bias.as_ref().map(|b| &b.variable),
            self.stride, self.padding, self.output_padding,
            self.dilation, self.groups,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device, test_opts};

    #[test]
    fn test_conv_transpose3d_forward() {
        let conv = ConvTranspose3d::on_device(4, 1, [3, 3, 3], test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 4, 4, 4, 4], test_opts()).unwrap(),
            false,
        );
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 6, 6, 6]);
    }
}
