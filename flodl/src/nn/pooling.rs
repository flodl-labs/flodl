//! Pooling layers.

use crate::autograd::{self, Variable};
use crate::tensor::Result;

use super::Module;
use super::parameter::Parameter;

/// 2D max pooling layer.
///
/// Applies max pooling over a 4D input `[B, C, H, W]`, equivalent to
/// PyTorch's `nn.MaxPool2d`.
///
/// ```ignore
/// let pool = MaxPool2d::new(2);  // kernel_size=2, stride=2
/// let y = pool.forward(&x)?;    // [B, C, H/2, W/2]
/// ```
pub struct MaxPool2d {
    kernel_size: [i64; 2],
    stride: [i64; 2],
    padding: [i64; 2],
    dilation: [i64; 2],
    ceil_mode: bool,
}

impl MaxPool2d {
    /// Create a MaxPool2d with the given kernel size. Stride defaults to kernel size.
    pub fn new(kernel_size: i64) -> Self {
        Self {
            kernel_size: [kernel_size, kernel_size],
            stride: [kernel_size, kernel_size],
            padding: [0, 0],
            dilation: [1, 1],
            ceil_mode: false,
        }
    }

    /// Create with explicit stride.
    pub fn with_stride(kernel_size: i64, stride: i64) -> Self {
        Self {
            kernel_size: [kernel_size, kernel_size],
            stride: [stride, stride],
            padding: [0, 0],
            dilation: [1, 1],
            ceil_mode: false,
        }
    }

    /// Set padding (applied symmetrically).
    pub fn padding(mut self, padding: i64) -> Self {
        self.padding = [padding, padding];
        self
    }

    /// Set dilation.
    pub fn dilation(mut self, dilation: i64) -> Self {
        self.dilation = [dilation, dilation];
        self
    }

    /// Enable ceiling mode for output size computation.
    pub fn ceil_mode(mut self, ceil_mode: bool) -> Self {
        self.ceil_mode = ceil_mode;
        self
    }
}

impl Module for MaxPool2d {
    fn name(&self) -> &str { "maxpool2d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![] // pooling has no learnable parameters
    }
}

/// 2D average pooling layer.
///
/// Applies average pooling over a 4D input `[B, C, H, W]`, equivalent to
/// PyTorch's `nn.AvgPool2d`.
///
/// ```ignore
/// let pool = AvgPool2d::new(2);  // kernel_size=2, stride=2
/// let y = pool.forward(&x)?;    // [B, C, H/2, W/2]
/// ```
pub struct AvgPool2d {
    kernel_size: [i64; 2],
    stride: [i64; 2],
    padding: [i64; 2],
    ceil_mode: bool,
    count_include_pad: bool,
}

impl AvgPool2d {
    /// Create an AvgPool2d with the given kernel size. Stride defaults to kernel size.
    pub fn new(kernel_size: i64) -> Self {
        Self {
            kernel_size: [kernel_size, kernel_size],
            stride: [kernel_size, kernel_size],
            padding: [0, 0],
            ceil_mode: false,
            count_include_pad: true,
        }
    }

    /// Create with explicit stride.
    pub fn with_stride(kernel_size: i64, stride: i64) -> Self {
        Self {
            kernel_size: [kernel_size, kernel_size],
            stride: [stride, stride],
            padding: [0, 0],
            ceil_mode: false,
            count_include_pad: true,
        }
    }

    /// Set padding (applied symmetrically).
    pub fn padding(mut self, padding: i64) -> Self {
        self.padding = [padding, padding];
        self
    }

    /// Enable ceiling mode for output size computation.
    pub fn ceil_mode(mut self, ceil_mode: bool) -> Self {
        self.ceil_mode = ceil_mode;
        self
    }

    /// Whether to include zero-padding in the average calculation.
    pub fn count_include_pad(mut self, count_include_pad: bool) -> Self {
        self.count_include_pad = count_include_pad;
        self
    }
}

impl Module for AvgPool2d {
    fn name(&self) -> &str { "avgpool2d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::avg_pool2d(
            input, self.kernel_size, self.stride, self.padding,
            self.ceil_mode, self.count_include_pad,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_max_pool2d_basic() {
        let opts = crate::tensor::test_opts();
        // [1, 1, 4, 4] → kernel=2, stride=2 → [1, 1, 2, 2]
        let x = Variable::new(
            Tensor::randn(&[1, 1, 4, 4], opts).unwrap(),
            false,
        );
        let pool = MaxPool2d::new(2);
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_max_pool2d_with_padding() {
        let opts = crate::tensor::test_opts();
        // [2, 3, 8, 8] → kernel=3, stride=2, padding=1 → [2, 3, 4, 4]
        let x = Variable::new(
            Tensor::randn(&[2, 3, 8, 8], opts).unwrap(),
            false,
        );
        let pool = MaxPool2d::with_stride(3, 2).padding(1);
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 3, 4, 4]);
    }

    #[test]
    fn test_max_pool2d_gradient() {
        let opts = crate::tensor::test_opts();
        let x = Variable::new(
            Tensor::randn(&[2, 1, 4, 4], opts).unwrap(),
            true,
        );
        let pool = MaxPool2d::new(2);
        let y = pool.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap();
        assert_eq!(grad.shape(), vec![2, 1, 4, 4]);
        // Gradient should be sparse: only max elements get gradient
    }

    #[test]
    fn test_max_pool2d_values() {
        let device = crate::tensor::test_device();
        // Manual check: 2x2 max pool on a known 4x4 input
        let data = vec![
            1.0_f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let x = Variable::new(
            Tensor::from_f32(&data, &[1, 1, 4, 4], device).unwrap(),
            false,
        );
        let pool = MaxPool2d::new(2);
        let y = pool.forward(&x).unwrap();
        let y_data = y.data().to_f32_vec().unwrap();
        // Each 2x2 block → max: [6, 8, 14, 16]
        assert_eq!(y_data, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_avg_pool2d_basic() {
        let opts = crate::tensor::test_opts();
        let x = Variable::new(
            Tensor::randn(&[1, 1, 4, 4], opts).unwrap(),
            false,
        );
        let pool = AvgPool2d::new(2);
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_avg_pool2d_values() {
        let device = crate::tensor::test_device();
        let data = vec![
            1.0_f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let x = Variable::new(
            Tensor::from_f32(&data, &[1, 1, 4, 4], device).unwrap(),
            false,
        );
        let pool = AvgPool2d::new(2);
        let y = pool.forward(&x).unwrap();
        let y_data = y.data().to_f32_vec().unwrap();
        // Each 2x2 block → mean: [3.5, 5.5, 11.5, 13.5]
        assert_eq!(y_data, vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn test_avg_pool2d_gradient() {
        let opts = crate::tensor::test_opts();
        let x = Variable::new(
            Tensor::randn(&[2, 1, 4, 4], opts).unwrap(),
            true,
        );
        let pool = AvgPool2d::new(2);
        let y = pool.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap();
        assert_eq!(grad.shape(), vec![2, 1, 4, 4]);
    }
}
