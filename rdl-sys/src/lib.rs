//! Raw FFI bindings to the libtorch C++ shim.
//!
//! Every function that can fail returns a `*mut i8` error string (caller
//! must free it with [`rdl_free_string`]). A null pointer means success.
//!
//! `RdlTensor` is an opaque `*mut c_void` handle to a heap-allocated
//! `torch::Tensor`. Caller owns it and must free with [`rdl_free_tensor`].

use std::ffi::c_void;

/// Opaque handle to a `torch::Tensor` on the C++ side.
pub type RdlTensor = *mut c_void;

// --- DType constants (must match shim.h) ---
pub const RDL_FLOAT16: i32 = 5;
pub const RDL_BFLOAT16: i32 = 15;
pub const RDL_FLOAT32: i32 = 6;
pub const RDL_FLOAT64: i32 = 7;
pub const RDL_INT32: i32 = 3;
pub const RDL_INT64: i32 = 4;

// --- Device constants (must match shim.h) ---
pub const RDL_CPU: i32 = 0;
pub const RDL_CUDA: i32 = 1;

unsafe extern "C" {
    // --- Tensor creation ---

    pub fn rdl_zeros(
        shape: *mut i64, ndim: i32, dtype: i32, device: i32,
        result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_ones(
        shape: *mut i64, ndim: i32, dtype: i32, device: i32,
        result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_rand(
        shape: *mut i64, ndim: i32, dtype: i32, device: i32,
        result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_randn(
        shape: *mut i64, ndim: i32, dtype: i32, device: i32,
        result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_from_blob(
        data: *mut c_void, shape: *mut i64, ndim: i32,
        dtype: i32, device: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_linspace(
        start: f64, end: f64, steps: i64,
        dtype: i32, device: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_arange(
        start: f64, end: f64, step: f64,
        dtype: i32, device: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_expand(
        t: RdlTensor, new_shape: *mut i64, ndim: i32,
        result: *mut RdlTensor,
    ) -> *mut i8;

    // --- Tensor lifecycle ---

    pub fn rdl_free_tensor(t: RdlTensor);
    pub fn rdl_shallow_clone(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;

    // --- Tensor metadata ---

    pub fn rdl_ndim(t: RdlTensor) -> i32;
    pub fn rdl_shape(t: RdlTensor, dim: i32) -> i64;
    pub fn rdl_dtype(t: RdlTensor) -> i32;
    pub fn rdl_device(t: RdlTensor) -> i32;
    pub fn rdl_numel(t: RdlTensor) -> i64;

    // --- Data access ---

    pub fn rdl_copy_data(
        t: RdlTensor, buffer: *mut c_void, buffer_bytes: i64,
    ) -> *mut i8;

    // --- Arithmetic ---

    pub fn rdl_add(a: RdlTensor, b: RdlTensor, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_sub(a: RdlTensor, b: RdlTensor, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_mul(a: RdlTensor, b: RdlTensor, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_div(a: RdlTensor, b: RdlTensor, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_matmul(a: RdlTensor, b: RdlTensor, result: *mut RdlTensor) -> *mut i8;

    pub fn rdl_add_scalar(
        t: RdlTensor, scalar: f64, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_mul_scalar(
        t: RdlTensor, scalar: f64, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_neg(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;

    // --- Activations ---

    pub fn rdl_relu(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_sigmoid(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_tanh_op(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_softmax(t: RdlTensor, dim: i32, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_log_softmax(t: RdlTensor, dim: i32, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_gelu(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_silu(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;

    // --- Layer normalization ---

    pub fn rdl_native_layer_norm(
        input: RdlTensor, weight: RdlTensor, bias: RdlTensor,
        normalized_size: i64, eps: f64,
        output: *mut RdlTensor, mean: *mut RdlTensor, rstd: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_native_layer_norm_backward(
        grad_output: RdlTensor, input: RdlTensor,
        mean: RdlTensor, rstd: RdlTensor,
        weight: RdlTensor, bias: RdlTensor,
        normalized_size: i64,
        grad_input: *mut RdlTensor, grad_weight: *mut RdlTensor,
        grad_bias: *mut RdlTensor,
    ) -> *mut i8;

    // --- Element-wise math ---

    pub fn rdl_exp(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_log(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_sqrt(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_abs(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;

    pub fn rdl_pow_scalar(
        t: RdlTensor, exponent: f64, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_clamp(
        t: RdlTensor, min_val: f64, max_val: f64, result: *mut RdlTensor,
    ) -> *mut i8;

    // --- Reductions ---

    pub fn rdl_sum(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;

    pub fn rdl_sum_dim(
        t: RdlTensor, dim: i32, keepdim: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_mean_dim(
        t: RdlTensor, dim: i32, keepdim: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_min(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;

    pub fn rdl_min_dim(
        t: RdlTensor, dim: i32, keepdim: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_max_dim(
        t: RdlTensor, dim: i32, keepdim: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_argmax(
        t: RdlTensor, dim: i32, keepdim: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    // --- Comparison (return float masks: 0.0 or 1.0) ---

    pub fn rdl_gt_scalar(
        t: RdlTensor, scalar: f64, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_ge_scalar(
        t: RdlTensor, scalar: f64, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_le_scalar(
        t: RdlTensor, scalar: f64, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_lt_scalar(
        t: RdlTensor, scalar: f64, result: *mut RdlTensor,
    ) -> *mut i8;

    // --- Shape operations ---

    pub fn rdl_reshape(
        t: RdlTensor, shape: *mut i64, ndim: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_transpose(
        t: RdlTensor, dim0: i32, dim1: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_permute(
        t: RdlTensor, dims: *mut i64, ndim: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_select(
        t: RdlTensor, dim: i32, index: i64, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_narrow(
        t: RdlTensor, dim: i32, start: i64, length: i64,
        result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_squeeze(
        t: RdlTensor, dim: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_unsqueeze(
        t: RdlTensor, dim: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    // --- Scatter ---

    pub fn rdl_select_scatter(
        input: RdlTensor, src: RdlTensor, dim: i32, index: i64,
        result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_narrow_scatter(
        input: RdlTensor, src: RdlTensor, dim: i32, start: i64,
        result: *mut RdlTensor,
    ) -> *mut i8;

    // --- Indexing ---

    pub fn rdl_index_select(
        t: RdlTensor, dim: i32, index: RdlTensor,
        result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_index_add(
        t: RdlTensor, dim: i32, index: RdlTensor, src: RdlTensor,
        result: *mut RdlTensor,
    ) -> *mut i8;

    // --- Concatenation ---

    pub fn rdl_cat2(
        a: RdlTensor, b: RdlTensor, dim: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    // --- Conditional ---

    pub fn rdl_where(
        condition: RdlTensor, x: RdlTensor, y: RdlTensor,
        result: *mut RdlTensor,
    ) -> *mut i8;

    // --- Like constructors ---

    pub fn rdl_zeros_like(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;
    pub fn rdl_ones_like(t: RdlTensor, result: *mut RdlTensor) -> *mut i8;

    // --- Convolution ---

    pub fn rdl_conv2d(
        input: RdlTensor, weight: RdlTensor, bias: RdlTensor,
        stride: *mut i64, padding: *mut i64, dilation: *mut i64,
        groups: i64, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_conv2d_backward(
        grad_output: RdlTensor, input: RdlTensor, weight: RdlTensor,
        stride: *mut i64, padding: *mut i64, dilation: *mut i64,
        groups: i64, compute_bias: i32,
        grad_input: *mut RdlTensor, grad_weight: *mut RdlTensor,
        grad_bias: *mut RdlTensor,
    ) -> *mut i8;

    // --- Transposed convolution ---

    pub fn rdl_conv_transpose2d(
        input: RdlTensor, weight: RdlTensor, bias: RdlTensor,
        stride: *mut i64, padding: *mut i64,
        output_padding: *mut i64, dilation: *mut i64,
        groups: i64, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_conv_transpose2d_backward(
        grad_output: RdlTensor, input: RdlTensor, weight: RdlTensor,
        stride: *mut i64, padding: *mut i64,
        output_padding: *mut i64, dilation: *mut i64,
        groups: i64, compute_bias: i32,
        grad_input: *mut RdlTensor, grad_weight: *mut RdlTensor,
        grad_bias: *mut RdlTensor,
    ) -> *mut i8;

    // --- Pooling ---

    pub fn rdl_adaptive_avg_pool2d(
        input: RdlTensor, output_size: *mut i64,
        result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_adaptive_avg_pool2d_backward(
        grad_output: RdlTensor, input: RdlTensor,
        grad_input: *mut RdlTensor,
    ) -> *mut i8;

    // --- Grid sampling ---

    pub fn rdl_grid_sample(
        input: RdlTensor, grid: RdlTensor,
        mode: i32, padding_mode: i32, align_corners: i32,
        result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_grid_sample_backward(
        grad_output: RdlTensor, input: RdlTensor, grid: RdlTensor,
        mode: i32, padding_mode: i32, align_corners: i32,
        grad_input: *mut RdlTensor, grad_grid: *mut RdlTensor,
    ) -> *mut i8;

    // --- Device ---

    pub fn rdl_to_device(
        t: RdlTensor, device: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_cuda_is_available() -> i32;
    pub fn rdl_cuda_device_count() -> i32;

    // --- Dtype casting ---

    pub fn rdl_to_dtype(
        t: RdlTensor, dtype: i32, result: *mut RdlTensor,
    ) -> *mut i8;

    pub fn rdl_all_finite(t: RdlTensor, result: *mut i32) -> *mut i8;

    // --- Utility ---

    pub fn rdl_free_string(s: *mut i8);
}
