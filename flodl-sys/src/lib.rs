//! Raw FFI bindings to the libtorch C++ shim.
//!
//! Every function that can fail returns a `*mut i8` error string (caller
//! must free it with [`flodl_free_string`]). A null pointer means success.
//!
//! `FlodlTensor` is an opaque `*mut c_void` handle to a heap-allocated
//! `torch::Tensor`. Caller owns it and must free with [`flodl_free_tensor`].

use std::ffi::c_void;

/// Opaque handle to a `torch::Tensor` on the C++ side.
pub type FlodlTensor = *mut c_void;

// --- DType constants (must match shim.h) ---
pub const FLODL_FLOAT16: i32 = 5;
pub const FLODL_BFLOAT16: i32 = 15;
pub const FLODL_FLOAT32: i32 = 6;
pub const FLODL_FLOAT64: i32 = 7;
pub const FLODL_INT32: i32 = 3;
pub const FLODL_INT64: i32 = 4;

// --- Device constants (must match shim.h) ---
pub const FLODL_CPU: i32 = 0;
pub const FLODL_CUDA: i32 = 1;

unsafe extern "C" {
    // --- Tensor creation ---

    pub fn flodl_zeros(
        shape: *mut i64, ndim: i32, dtype: i32, device: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_ones(
        shape: *mut i64, ndim: i32, dtype: i32, device: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_rand(
        shape: *mut i64, ndim: i32, dtype: i32, device: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_randn(
        shape: *mut i64, ndim: i32, dtype: i32, device: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_from_blob(
        data: *mut c_void, shape: *mut i64, ndim: i32,
        dtype: i32, device: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_linspace(
        start: f64, end: f64, steps: i64,
        dtype: i32, device: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_arange(
        start: f64, end: f64, step: f64,
        dtype: i32, device: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_expand(
        t: FlodlTensor, new_shape: *mut i64, ndim: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Tensor lifecycle ---

    pub fn flodl_free_tensor(t: FlodlTensor);
    pub fn flodl_shallow_clone(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    // --- Tensor metadata ---

    pub fn flodl_ndim(t: FlodlTensor) -> i32;
    pub fn flodl_shape(t: FlodlTensor, dim: i32) -> i64;
    pub fn flodl_dtype(t: FlodlTensor) -> i32;
    pub fn flodl_device(t: FlodlTensor) -> i32;
    pub fn flodl_numel(t: FlodlTensor) -> i64;

    // --- Data access ---

    pub fn flodl_copy_data(
        t: FlodlTensor, buffer: *mut c_void, buffer_bytes: i64,
    ) -> *mut i8;

    // --- Arithmetic ---

    pub fn flodl_add(a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_sub(a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_mul(a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_div(a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_matmul(a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    pub fn flodl_add_scalar(
        t: FlodlTensor, scalar: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_mul_scalar(
        t: FlodlTensor, scalar: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_div_scalar(
        t: FlodlTensor, scalar: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_neg(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    // --- Activations ---

    pub fn flodl_relu(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_sigmoid(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_tanh_op(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_softmax(t: FlodlTensor, dim: i32, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_log_softmax(t: FlodlTensor, dim: i32, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_gelu(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_silu(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    // --- Layer normalization ---

    pub fn flodl_native_layer_norm(
        input: FlodlTensor, weight: FlodlTensor, bias: FlodlTensor,
        normalized_size: i64, eps: f64,
        output: *mut FlodlTensor, mean: *mut FlodlTensor, rstd: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Element-wise math ---

    pub fn flodl_exp(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_log(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_sqrt(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_abs(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_triu(t: FlodlTensor, diagonal: i64, result: *mut FlodlTensor) -> *mut i8;

    pub fn flodl_pow_scalar(
        t: FlodlTensor, exponent: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_clamp(
        t: FlodlTensor, min_val: f64, max_val: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Reductions ---

    pub fn flodl_sum(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_mean(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    pub fn flodl_sum_dim(
        t: FlodlTensor, dim: i32, keepdim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_mean_dim(
        t: FlodlTensor, dim: i32, keepdim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_min(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_max(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_norm(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    pub fn flodl_min_dim(
        t: FlodlTensor, dim: i32, keepdim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_max_dim(
        t: FlodlTensor, dim: i32, keepdim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_argmax(
        t: FlodlTensor, dim: i32, keepdim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Comparison (return float masks: 0.0 or 1.0) ---

    pub fn flodl_gt_scalar(
        t: FlodlTensor, scalar: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_ge_scalar(
        t: FlodlTensor, scalar: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_le_scalar(
        t: FlodlTensor, scalar: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_lt_scalar(
        t: FlodlTensor, scalar: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Shape operations ---

    pub fn flodl_reshape(
        t: FlodlTensor, shape: *mut i64, ndim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_transpose(
        t: FlodlTensor, dim0: i32, dim1: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_permute(
        t: FlodlTensor, dims: *mut i64, ndim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_select(
        t: FlodlTensor, dim: i32, index: i64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_narrow(
        t: FlodlTensor, dim: i32, start: i64, length: i64,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_squeeze(
        t: FlodlTensor, dim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_unsqueeze(
        t: FlodlTensor, dim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_flatten(
        t: FlodlTensor, start_dim: i32, end_dim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Scatter ---

    pub fn flodl_select_scatter(
        input: FlodlTensor, src: FlodlTensor, dim: i32, index: i64,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_narrow_scatter(
        input: FlodlTensor, src: FlodlTensor, dim: i32, start: i64,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Indexing ---

    pub fn flodl_index_select(
        t: FlodlTensor, dim: i32, index: FlodlTensor,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_index_add(
        t: FlodlTensor, dim: i32, index: FlodlTensor, src: FlodlTensor,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Concatenation ---

    pub fn flodl_cat2(
        a: FlodlTensor, b: FlodlTensor, dim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_stack(
        tensors: *mut FlodlTensor, count: i32, dim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Conditional ---

    pub fn flodl_where(
        condition: FlodlTensor, x: FlodlTensor, y: FlodlTensor,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Like constructors ---

    pub fn flodl_zeros_like(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_ones_like(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    // --- Convolution ---

    pub fn flodl_conv2d(
        input: FlodlTensor, weight: FlodlTensor, bias: FlodlTensor,
        stride: *mut i64, padding: *mut i64, dilation: *mut i64,
        groups: i64, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Transposed convolution ---

    pub fn flodl_conv_transpose2d(
        input: FlodlTensor, weight: FlodlTensor, bias: FlodlTensor,
        stride: *mut i64, padding: *mut i64,
        output_padding: *mut i64, dilation: *mut i64,
        groups: i64, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Pooling ---

    pub fn flodl_adaptive_avg_pool2d(
        input: FlodlTensor, output_size: *mut i64,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Grid sampling ---

    pub fn flodl_grid_sample(
        input: FlodlTensor, grid: FlodlTensor,
        mode: i32, padding_mode: i32, align_corners: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Device ---

    pub fn flodl_to_device(
        t: FlodlTensor, device: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_cuda_is_available() -> i32;
    pub fn flodl_cuda_device_count() -> i32;
    pub fn flodl_force_cuda_link() -> i32;

    // --- CUDA memory/utilization (monitor support) ---

    pub fn flodl_cuda_mem_info(
        used_bytes: *mut u64, total_bytes: *mut u64,
    ) -> *mut i8;

    pub fn flodl_cuda_utilization(device_index: i32) -> i32;

    // --- Dtype casting ---

    pub fn flodl_to_dtype(
        t: FlodlTensor, dtype: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_all_finite(t: FlodlTensor, result: *mut i32) -> *mut i8;

    // --- Comparison (tensor-tensor, return float masks: 0.0 or 1.0) ---

    pub fn flodl_gt_tensor(
        a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_lt_tensor(
        a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_ge_tensor(
        a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_le_tensor(
        a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_eq_tensor(
        a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_ne_tensor(
        a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Additional reductions ---

    pub fn flodl_argmin(
        t: FlodlTensor, dim: i32, keepdim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_var(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_std_op(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    pub fn flodl_var_dim(
        t: FlodlTensor, dim: i32, keepdim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_std_dim(
        t: FlodlTensor, dim: i32, keepdim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Element-wise math (trig, rounding, sign) ---

    pub fn flodl_sin(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_cos(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_sign(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_floor(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_ceil(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_round(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_reciprocal(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    // --- Advanced indexing ---

    pub fn flodl_gather(
        t: FlodlTensor, dim: i32, index: FlodlTensor,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_scatter_add(
        t: FlodlTensor, dim: i32, index: FlodlTensor, src: FlodlTensor,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Sorting ---

    pub fn flodl_topk(
        t: FlodlTensor, k: i64, dim: i32, largest: i32, sorted: i32,
        values: *mut FlodlTensor, indices: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_sort(
        t: FlodlTensor, dim: i32, descending: i32,
        values: *mut FlodlTensor, indices: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Tensor creation (additional) ---

    pub fn flodl_eye(
        n: i64, dtype: i32, device: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_full(
        shape: *mut i64, ndim: i32, value: f64, dtype: i32, device: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Shape operations (additional) ---

    pub fn flodl_chunk(
        t: FlodlTensor, chunks: i32, dim: i32,
        results: *mut *mut FlodlTensor, count: *mut i32,
    ) -> *mut i8;

    pub fn flodl_repeat(
        t: FlodlTensor, repeats: *mut i64, ndim: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_pad(
        t: FlodlTensor, padding: *mut i64, pad_len: i32, value: f64,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Autograd ---

    pub fn flodl_set_requires_grad(
        t: FlodlTensor, requires_grad: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_requires_grad(t: FlodlTensor) -> i32;

    pub fn flodl_backward(t: FlodlTensor) -> *mut i8;

    pub fn flodl_grad(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    pub fn flodl_set_grad(t: FlodlTensor, grad: FlodlTensor) -> *mut i8;

    pub fn flodl_zero_grad(t: FlodlTensor) -> *mut i8;

    pub fn flodl_detach(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    pub fn flodl_is_leaf(t: FlodlTensor) -> i32;

    // --- Autograd context ---

    pub fn flodl_no_grad_guard_new() -> *mut c_void;
    pub fn flodl_no_grad_guard_delete(guard: *mut c_void);
    pub fn flodl_is_grad_enabled() -> i32;

    // --- In-place operations ---

    pub fn flodl_add_(t: FlodlTensor, other: FlodlTensor) -> *mut i8;
    pub fn flodl_sub_(t: FlodlTensor, other: FlodlTensor) -> *mut i8;
    pub fn flodl_mul_scalar_(t: FlodlTensor, scalar: f64) -> *mut i8;
    pub fn flodl_add_scalar_(t: FlodlTensor, scalar: f64) -> *mut i8;
    pub fn flodl_zero_(t: FlodlTensor) -> *mut i8;

    // --- Utility ---

    pub fn flodl_free_string(s: *mut i8);
}
