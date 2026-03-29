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

/// 1D max pooling layer.
///
/// Applies max pooling over a 3D input `[B, C, L]`, equivalent to
/// PyTorch's `nn.MaxPool1d`.
pub struct MaxPool1d {
    kernel_size: i64,
    stride: i64,
    padding: i64,
    dilation: i64,
    ceil_mode: bool,
}

impl MaxPool1d {
    /// Create a MaxPool1d with the given kernel size. Stride defaults to kernel size.
    pub fn new(kernel_size: i64) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: 0,
            dilation: 1,
            ceil_mode: false,
        }
    }

    /// Create with explicit stride.
    pub fn with_stride(kernel_size: i64, stride: i64) -> Self {
        Self {
            kernel_size,
            stride,
            padding: 0,
            dilation: 1,
            ceil_mode: false,
        }
    }

    /// Set padding.
    pub fn padding(mut self, padding: i64) -> Self {
        self.padding = padding;
        self
    }
}

impl Module for MaxPool1d {
    fn name(&self) -> &str { "maxpool1d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::max_pool1d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// 1D average pooling layer.
pub struct AvgPool1d {
    kernel_size: i64,
    stride: i64,
    padding: i64,
    ceil_mode: bool,
    count_include_pad: bool,
}

impl AvgPool1d {
    /// Create an AvgPool1d with the given kernel size. Stride defaults to kernel size.
    pub fn new(kernel_size: i64) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: 0,
            ceil_mode: false,
            count_include_pad: true,
        }
    }

    /// Create with explicit stride.
    pub fn with_stride(kernel_size: i64, stride: i64) -> Self {
        Self {
            kernel_size,
            stride,
            padding: 0,
            ceil_mode: false,
            count_include_pad: true,
        }
    }

    /// Set padding.
    pub fn padding(mut self, padding: i64) -> Self {
        self.padding = padding;
        self
    }
}

impl Module for AvgPool1d {
    fn name(&self) -> &str { "avgpool1d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::avg_pool1d(
            input, self.kernel_size, self.stride, self.padding,
            self.ceil_mode, self.count_include_pad,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Adaptive max pooling 2D. Outputs a fixed `[H, W]` regardless of input size.
pub struct AdaptiveMaxPool2d {
    output_size: [i64; 2],
}

impl AdaptiveMaxPool2d {
    /// Create with target output size `[H, W]`.
    pub fn new(output_h: i64, output_w: i64) -> Self {
        Self { output_size: [output_h, output_w] }
    }
}

impl Module for AdaptiveMaxPool2d {
    fn name(&self) -> &str { "adaptive_maxpool2d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::adaptive_max_pool2d(input, self.output_size)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Pixel shuffle: rearranges `[N, C*r^2, H, W]` to `[N, C, H*r, W*r]`.
/// Used for efficient sub-pixel convolution upsampling (ESPCN).
pub struct PixelShuffle {
    upscale_factor: i64,
}

impl PixelShuffle {
    /// Create a PixelShuffle with the given upscale factor.
    pub fn new(upscale_factor: i64) -> Self {
        Self { upscale_factor }
    }
}

impl Module for PixelShuffle {
    fn name(&self) -> &str { "pixel_shuffle" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::pixel_shuffle(input, self.upscale_factor)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Pixel unshuffle: inverse of PixelShuffle.
/// Rearranges `[N, C, H*r, W*r]` to `[N, C*r^2, H, W]`.
pub struct PixelUnshuffle {
    downscale_factor: i64,
}

impl PixelUnshuffle {
    /// Create a PixelUnshuffle with the given downscale factor.
    pub fn new(downscale_factor: i64) -> Self {
        Self { downscale_factor }
    }
}

impl Module for PixelUnshuffle {
    fn name(&self) -> &str { "pixel_unshuffle" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::pixel_unshuffle(input, self.downscale_factor)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Upsample module. Wraps `F.interpolate` as an `nn.Module`.
///
/// `mode`: 0=nearest, 1=bilinear, 2=bicubic, 3=trilinear.
pub struct Upsample {
    output_size: Vec<i64>,
    mode: i32,
    align_corners: bool,
}

impl Upsample {
    /// Create an Upsample module with fixed output size.
    pub fn new(output_size: &[i64], mode: i32) -> Self {
        Self {
            output_size: output_size.to_vec(),
            mode,
            align_corners: false,
        }
    }

    /// Set align_corners flag (only used for bilinear/bicubic/trilinear).
    pub fn align_corners(mut self, align: bool) -> Self {
        self.align_corners = align;
        self
    }
}

impl Module for Upsample {
    fn name(&self) -> &str { "upsample" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let result = input.data().interpolate(&self.output_size, self.mode, self.align_corners)?;
        Ok(Variable::wrap(result))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Unfold extracts sliding local blocks from a batched input tensor.
///
/// Input: `[N, C, H, W]`.
/// Output: `[N, C * kH * kW, L]` where L is the number of valid blocks.
///
/// Equivalent to PyTorch's `nn.Unfold` (im2col).
pub struct Unfold {
    kernel_size: [i64; 2],
    dilation: [i64; 2],
    padding: [i64; 2],
    stride: [i64; 2],
}

impl Unfold {
    /// Create with square kernel.
    pub fn new(kernel_size: i64) -> Self {
        Self {
            kernel_size: [kernel_size, kernel_size],
            dilation: [1, 1],
            padding: [0, 0],
            stride: [1, 1],
        }
    }

    /// Create with rectangular kernel.
    pub fn with_kernel(kernel_size: [i64; 2]) -> Self {
        Self {
            kernel_size,
            dilation: [1, 1],
            padding: [0, 0],
            stride: [1, 1],
        }
    }

    /// Set dilation (spacing between kernel elements).
    pub fn dilation(mut self, dilation: [i64; 2]) -> Self { self.dilation = dilation; self }
    /// Set zero-padding added to both sides of the input.
    pub fn padding(mut self, padding: [i64; 2]) -> Self { self.padding = padding; self }
    /// Set stride of the sliding blocks.
    pub fn stride(mut self, stride: [i64; 2]) -> Self { self.stride = stride; self }
}

impl Module for Unfold {
    fn name(&self) -> &str { "unfold" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::im2col(input, self.kernel_size, self.dilation, self.padding, self.stride)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Fold reassembles columns into a batched image tensor.
///
/// Input: `[N, C * kH * kW, L]`.
/// Output: `[N, C, output_H, output_W]`.
///
/// Equivalent to PyTorch's `nn.Fold` (col2im). Inverse of Unfold.
pub struct Fold {
    output_size: [i64; 2],
    kernel_size: [i64; 2],
    dilation: [i64; 2],
    padding: [i64; 2],
    stride: [i64; 2],
}

impl Fold {
    /// Create with target output size and square kernel.
    pub fn new(output_size: [i64; 2], kernel_size: i64) -> Self {
        Self {
            output_size,
            kernel_size: [kernel_size, kernel_size],
            dilation: [1, 1],
            padding: [0, 0],
            stride: [1, 1],
        }
    }

    /// Create with target output size and rectangular kernel.
    pub fn with_kernel(output_size: [i64; 2], kernel_size: [i64; 2]) -> Self {
        Self {
            output_size,
            kernel_size,
            dilation: [1, 1],
            padding: [0, 0],
            stride: [1, 1],
        }
    }

    /// Set dilation (spacing between kernel elements).
    pub fn dilation(mut self, dilation: [i64; 2]) -> Self { self.dilation = dilation; self }
    /// Set zero-padding added to both sides of the input.
    pub fn padding(mut self, padding: [i64; 2]) -> Self { self.padding = padding; self }
    /// Set stride of the sliding blocks.
    pub fn stride(mut self, stride: [i64; 2]) -> Self { self.stride = stride; self }
}

impl Module for Fold {
    fn name(&self) -> &str { "fold" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::col2im(input, self.output_size, self.kernel_size, self.dilation, self.padding, self.stride)
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

    #[test]
    fn test_max_pool1d() {
        let x = Variable::new(
            Tensor::randn(&[1, 1, 8], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let pool = MaxPool1d::new(2);
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 4]);
    }

    #[test]
    fn test_avg_pool1d() {
        let x = Variable::new(
            Tensor::randn(&[1, 1, 8], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let pool = AvgPool1d::new(2);
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 4]);
    }

    #[test]
    fn test_adaptive_max_pool2d() {
        let x = Variable::new(
            Tensor::randn(&[1, 1, 8, 8], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let pool = AdaptiveMaxPool2d::new(3, 3);
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 3, 3]);
    }

    #[test]
    fn test_pixel_shuffle() {
        let x = Variable::new(
            Tensor::randn(&[1, 4, 2, 2], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let ps = PixelShuffle::new(2);
        let y = ps.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_pixel_unshuffle() {
        let x = Variable::new(
            Tensor::randn(&[1, 1, 4, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let pu = PixelUnshuffle::new(2);
        let y = pu.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 4, 2, 2]);
    }

    #[test]
    fn test_upsample() {
        let x = Variable::new(
            Tensor::randn(&[1, 1, 2, 2], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let up = Upsample::new(&[4, 4], 0); // nearest
        let y = up.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_unfold() {
        // [1, 1, 4, 4] with kernel=2, stride=1 -> L = 3*3 = 9 blocks
        let x = Variable::new(
            Tensor::randn(&[1, 1, 4, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let unfold = Unfold::new(2);
        let y = unfold.forward(&x).unwrap();
        // Output: [1, 1*2*2, 3*3] = [1, 4, 9]
        assert_eq!(y.shape(), vec![1, 4, 9]);
    }

    #[test]
    fn test_unfold_with_stride() {
        // [1, 1, 4, 4] with kernel=2, stride=2 -> L = 2*2 = 4 blocks
        let x = Variable::new(
            Tensor::randn(&[1, 1, 4, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let unfold = Unfold::new(2).stride([2, 2]);
        let y = unfold.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 4, 4]);
    }

    #[test]
    fn test_fold_unfold_roundtrip() {
        let x = Variable::new(
            Tensor::randn(&[1, 1, 4, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        // Unfold with non-overlapping blocks (stride=kernel)
        let unfold = Unfold::new(2).stride([2, 2]);
        let cols = unfold.forward(&x).unwrap();
        assert_eq!(cols.shape(), vec![1, 4, 4]);

        // Fold back
        let fold = Fold::new([4, 4], 2).stride([2, 2]);
        let y = fold.forward(&cols).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 4, 4]);

        // Non-overlapping roundtrip should be exact
        let xv = x.data().to_f32_vec().unwrap();
        let yv = y.data().to_f32_vec().unwrap();
        for (a, b) in xv.iter().zip(yv.iter()) {
            assert!((a - b).abs() < 1e-5, "roundtrip mismatch: {} vs {}", a, b);
        }
    }
}
