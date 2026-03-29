//! Fused neural network operations: layer norm, convolution, linear, RNN cells,
//! pooling, grid sampling, losses, batch norm, and dropout.

use std::ptr;
use flodl_sys::{self as ffi, FlodlTensor};
use super::{Tensor, check_err, Result};

impl Tensor {
    /// Native layer normalization. Returns (output, mean, rstd).
    pub fn native_layer_norm(
        &self, weight: &Tensor, bias: &Tensor, normalized_size: i64, eps: f64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mut out: FlodlTensor = ptr::null_mut();
        let mut mean: FlodlTensor = ptr::null_mut();
        let mut rstd: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_native_layer_norm(
                self.handle, weight.handle, bias.handle,
                normalized_size, eps,
                &mut out, &mut mean, &mut rstd,
            )
        };
        check_err(err)?;
        Ok((Tensor::from_raw(out), Tensor::from_raw(mean), Tensor::from_raw(rstd)))
    }

    /// 2D convolution. bias may be a null-handle tensor for no bias.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &self, weight: &Tensor, bias: Option<&Tensor>,
        stride: [i64; 2], padding: [i64; 2], dilation: [i64; 2], groups: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut stride = stride;
        let mut padding = padding;
        let mut dilation = dilation;
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_conv2d(
                self.handle, weight.handle, bias_handle,
                stride.as_mut_ptr(), padding.as_mut_ptr(), dilation.as_mut_ptr(),
                groups, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Transposed 2D convolution.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose2d(
        &self, weight: &Tensor, bias: Option<&Tensor>,
        stride: [i64; 2], padding: [i64; 2], output_padding: [i64; 2],
        dilation: [i64; 2], groups: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut stride = stride;
        let mut padding = padding;
        let mut output_padding = output_padding;
        let mut dilation = dilation;
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_conv_transpose2d(
                self.handle, weight.handle, bias_handle,
                stride.as_mut_ptr(), padding.as_mut_ptr(),
                output_padding.as_mut_ptr(), dilation.as_mut_ptr(),
                groups, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// 1D convolution. bias may be None for no bias.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &self, weight: &Tensor, bias: Option<&Tensor>,
        stride: i64, padding: i64, dilation: i64, groups: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_conv1d(
                self.handle, weight.handle, bias_handle,
                stride, padding, dilation,
                groups, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Transposed 1D convolution.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d(
        &self, weight: &Tensor, bias: Option<&Tensor>,
        stride: i64, padding: i64, output_padding: i64,
        dilation: i64, groups: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_conv_transpose1d(
                self.handle, weight.handle, bias_handle,
                stride, padding, output_padding, dilation,
                groups, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Group normalization. weight and bias are optional (shape `[num_channels]`).
    pub fn group_norm(
        &self, num_groups: i64,
        weight: Option<&Tensor>, bias: Option<&Tensor>,
        eps: f64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let w = weight.map_or(ptr::null_mut(), |t| t.handle);
        let b = bias.map_or(ptr::null_mut(), |t| t.handle);
        let err = unsafe {
            ffi::flodl_group_norm(
                self.handle, num_groups, w, b, eps, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused linear: `y = input @ weight^T + bias` (single ATen kernel).
    pub fn linear(&self, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_linear(self.handle, weight.handle, bias_handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused GRU cell: single ATen `gru_cell` kernel.
    /// Returns new hidden state h'.
    #[allow(clippy::too_many_arguments)]
    pub fn gru_cell(
        &self, hx: &Tensor,
        w_ih: &Tensor, w_hh: &Tensor,
        b_ih: &Tensor, b_hh: &Tensor,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_gru_cell(
                self.handle, hx.handle,
                w_ih.handle, w_hh.handle,
                b_ih.handle, b_hh.handle,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused LSTM cell: single ATen `lstm_cell` kernel.
    /// Returns `(h', c')`.
    #[allow(clippy::too_many_arguments)]
    pub fn lstm_cell(
        &self, hx: &Tensor, cx: &Tensor,
        w_ih: &Tensor, w_hh: &Tensor,
        b_ih: &Tensor, b_hh: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mut h_out: FlodlTensor = ptr::null_mut();
        let mut c_out: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_lstm_cell(
                self.handle, hx.handle, cx.handle,
                w_ih.handle, w_hh.handle,
                b_ih.handle, b_hh.handle,
                &mut h_out, &mut c_out,
            )
        };
        check_err(err)?;
        Ok((Tensor::from_raw(h_out), Tensor::from_raw(c_out)))
    }

    /// Max pooling over a 2D input (`[B, C, H, W]`).
    ///
    /// Equivalent to `torch.nn.functional.max_pool2d`.
    pub fn max_pool2d(
        &self,
        kernel_size: [i64; 2],
        stride: [i64; 2],
        padding: [i64; 2],
        dilation: [i64; 2],
        ceil_mode: bool,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut ks = kernel_size;
        let mut st = stride;
        let mut pd = padding;
        let mut dl = dilation;
        let err = unsafe {
            ffi::flodl_max_pool2d(
                self.handle,
                ks.as_mut_ptr(), st.as_mut_ptr(),
                pd.as_mut_ptr(), dl.as_mut_ptr(),
                ceil_mode as i32, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Average pooling over spatial dimensions.
    pub fn avg_pool2d(
        &self,
        kernel_size: [i64; 2],
        stride: [i64; 2],
        padding: [i64; 2],
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut ks = kernel_size;
        let mut st = stride;
        let mut pd = padding;
        let err = unsafe {
            ffi::flodl_avg_pool2d(
                self.handle,
                ks.as_mut_ptr(), st.as_mut_ptr(), pd.as_mut_ptr(),
                ceil_mode as i32, count_include_pad as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Adaptive average pooling to target spatial size.
    pub fn adaptive_avg_pool2d(&self, output_size: [i64; 2]) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut os = output_size;
        let err = unsafe {
            ffi::flodl_adaptive_avg_pool2d(self.handle, os.as_mut_ptr(), &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Grid sampling (bilinear/nearest interpolation).
    pub fn grid_sample(
        &self, grid: &Tensor, mode: i32, padding_mode: i32, align_corners: bool,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_grid_sample(
                self.handle, grid.handle, mode, padding_mode,
                align_corners as i32, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Fused loss functions ---

    /// Fused MSE loss: single libtorch kernel.
    /// reduction: 0=None, 1=Mean, 2=Sum.
    pub fn mse_loss(&self, target: &Tensor, reduction: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_mse_loss(self.handle, target.handle, reduction, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused cross-entropy loss: single libtorch kernel.
    /// pred: \[N,C\] logits. target: \[N\] Int64 indices or \[N,C\] Float probs.
    /// reduction: 0=None, 1=Mean, 2=Sum.
    #[allow(clippy::too_many_arguments)]
    pub fn cross_entropy_loss(
        &self, target: &Tensor, reduction: i64,
        ignore_index: i64, label_smoothing: f64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_cross_entropy_loss(
                self.handle, target.handle,
                reduction, ignore_index, label_smoothing,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Binary cross-entropy loss from probabilities (NOT logits).
    /// Input must be in \[0, 1\] (e.g. after sigmoid).
    /// reduction: 0=None, 1=Mean, 2=Sum.
    pub fn bce_loss(&self, target: &Tensor, reduction: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_bce_loss(
                self.handle, target.handle, reduction, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused BCE with logits loss: single libtorch kernel.
    /// Numerically stable binary cross-entropy from raw logits.
    /// reduction: 0=None, 1=Mean, 2=Sum.
    pub fn bce_with_logits_loss(&self, target: &Tensor, reduction: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_bce_with_logits_loss(
                self.handle, target.handle, reduction, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused L1 loss: single libtorch kernel.
    /// reduction: 0=None, 1=Mean, 2=Sum.
    pub fn l1_loss(&self, target: &Tensor, reduction: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_l1_loss(self.handle, target.handle, reduction, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused Smooth L1 (Huber) loss: single libtorch kernel.
    /// reduction: 0=None, 1=Mean, 2=Sum. beta: transition point.
    pub fn smooth_l1_loss(&self, target: &Tensor, reduction: i64, beta: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_smooth_l1_loss(
                self.handle, target.handle, reduction, beta, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused KL divergence loss: single libtorch kernel.
    /// input: log-probabilities. target: probabilities.
    /// reduction: 0=None, 1=Mean, 2=Sum, 5=BatchMean.
    pub fn kl_div_loss(&self, target: &Tensor, reduction: i64, log_target: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_kl_div_loss(
                self.handle, target.handle, reduction, log_target as i32, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Fused batch normalization ---

    /// Fused batch normalization: single libtorch kernel.
    /// When training=true, updates running_mean/running_var in-place.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm(
        &self, weight: Option<&Tensor>, bias: Option<&Tensor>,
        running_mean: Option<&Tensor>, running_var: Option<&Tensor>,
        training: bool, momentum: f64, eps: f64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let w = weight.map_or(ptr::null_mut(), |t| t.handle);
        let b = bias.map_or(ptr::null_mut(), |t| t.handle);
        let rm = running_mean.map_or(ptr::null_mut(), |t| t.handle);
        let rv = running_var.map_or(ptr::null_mut(), |t| t.handle);
        let err = unsafe {
            ffi::flodl_batch_norm(
                self.handle, w, b, rm, rv,
                training as i32, momentum, eps, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Fused dropout ---

    /// Fused dropout: single libtorch kernel with inverted scaling.
    pub fn dropout(&self, p: f64, training: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_dropout(self.handle, p, training as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused 2D feature dropout: drops entire channels.
    pub fn feature_dropout(&self, p: f64, training: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_feature_dropout(self.handle, p, training as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused embedding lookup + reduction (sum / mean / max).
    ///
    /// `weight`: `[num_embeddings, embedding_dim]` embedding table.
    /// `indices`: 1-D i64 tensor of token indices.
    /// `offsets`: 1-D i64 tensor marking the start of each bag.
    /// `mode`: 0 = sum, 1 = mean, 2 = max.
    ///
    /// Returns one row per bag with shape `[num_bags, embedding_dim]`.
    pub fn embedding_bag(
        weight: &Tensor, indices: &Tensor, offsets: &Tensor, mode: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_embedding_bag(
                weight.handle, indices.handle, offsets.handle,
                mode, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Interpolate (resize) a tensor using nearest, bilinear, bicubic, or trilinear mode.
    ///
    /// `output_size`: target spatial dimensions (1D, 2D, or 3D depending on input).
    /// `mode`: 0=nearest, 1=bilinear, 2=bicubic, 3=trilinear.
    /// `align_corners`: whether to align corner pixels (ignored for nearest).
    pub fn interpolate(
        &self, output_size: &[i64], mode: i32, align_corners: bool,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut os = output_size.to_vec();
        let err = unsafe {
            ffi::flodl_interpolate(
                self.handle, os.as_mut_ptr(), os.len() as i32,
                mode, align_corners as i32, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }
}
