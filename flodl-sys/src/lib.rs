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
        shape: *mut i64, ndim: i32, dtype: i32,
        device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_ones(
        shape: *mut i64, ndim: i32, dtype: i32,
        device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_rand(
        shape: *mut i64, ndim: i32, dtype: i32,
        device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_randn(
        shape: *mut i64, ndim: i32, dtype: i32,
        device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_from_blob(
        data: *mut c_void, shape: *mut i64, ndim: i32,
        dtype: i32, device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_linspace(
        start: f64, end: f64, steps: i64,
        dtype: i32, device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_arange(
        start: f64, end: f64, step: f64,
        dtype: i32, device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
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
    pub fn flodl_device_type(t: FlodlTensor) -> i32;
    pub fn flodl_device_index(t: FlodlTensor) -> i32;
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
    pub fn flodl_leaky_relu(
        t: FlodlTensor, negative_slope: f64, result: *mut FlodlTensor,
    ) -> *mut i8;
    pub fn flodl_elu(t: FlodlTensor, alpha: f64, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_softplus(
        t: FlodlTensor, beta: f64, threshold: f64, result: *mut FlodlTensor,
    ) -> *mut i8;
    pub fn flodl_mish(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    // --- Layer normalization ---

    pub fn flodl_native_layer_norm(
        input: FlodlTensor, weight: FlodlTensor, bias: FlodlTensor,
        normalized_size: i64, eps: f64,
        output: *mut FlodlTensor, mean: *mut FlodlTensor, rstd: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Group normalization ---

    pub fn flodl_group_norm(
        input: FlodlTensor, num_groups: i64,
        weight: FlodlTensor, bias: FlodlTensor,
        eps: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Element-wise math ---

    pub fn flodl_exp(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_log(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_sqrt(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_abs(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_triu(t: FlodlTensor, diagonal: i64, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_tril(t: FlodlTensor, diagonal: i64, result: *mut FlodlTensor) -> *mut i8;

    pub fn flodl_pow_scalar(
        t: FlodlTensor, exponent: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_clamp(
        t: FlodlTensor, min_val: f64, max_val: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_clamp_min(
        t: FlodlTensor, min_val: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_clamp_max(
        t: FlodlTensor, max_val: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_log1p(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_expm1(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_log2(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_log10(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    // --- Reductions ---

    pub fn flodl_sum(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_mean(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    pub fn flodl_sum_dim(
        t: FlodlTensor, dim: i32, keepdim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_mean_dim(
        t: FlodlTensor, dim: i32, keepdim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_prod(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    pub fn flodl_prod_dim(
        t: FlodlTensor, dim: i32, keepdim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_cumsum(
        t: FlodlTensor, dim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_logsumexp(
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

    pub fn flodl_eq_scalar(
        t: FlodlTensor, scalar: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_ne_scalar(
        t: FlodlTensor, scalar: f64, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Boolean / detection (return float masks) ---

    pub fn flodl_isnan(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_isinf(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_logical_and(
        a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor,
    ) -> *mut i8;
    pub fn flodl_logical_or(
        a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor,
    ) -> *mut i8;
    pub fn flodl_logical_not(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_any(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_all(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

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

    pub fn flodl_cat(
        tensors: *mut FlodlTensor, count: i32, dim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_stack(
        tensors: *mut FlodlTensor, count: i32, dim: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Masking ---

    pub fn flodl_masked_fill(
        t: FlodlTensor, mask: FlodlTensor, value: f64,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Conditional ---

    pub fn flodl_where(
        condition: FlodlTensor, x: FlodlTensor, y: FlodlTensor,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Like constructors ---

    pub fn flodl_zeros_like(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_ones_like(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_full_like(
        t: FlodlTensor, value: f64, result: *mut FlodlTensor,
    ) -> *mut i8;
    pub fn flodl_rand_like(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_randn_like(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    // --- Tensor creation (tier 2) ---

    pub fn flodl_randint(
        low: i64, high: i64, shape: *mut i64, ndim: i32,
        dtype: i32, device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_empty(
        shape: *mut i64, ndim: i32, dtype: i32,
        device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_one_hot(
        t: FlodlTensor, num_classes: i64,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_bernoulli(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;

    // --- Convolution ---

    pub fn flodl_conv2d(
        input: FlodlTensor, weight: FlodlTensor, bias: FlodlTensor,
        stride: *mut i64, padding: *mut i64, dilation: *mut i64,
        groups: i64, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- 1D convolution ---

    pub fn flodl_conv1d(
        input: FlodlTensor, weight: FlodlTensor, bias: FlodlTensor,
        stride: i64, padding: i64, dilation: i64,
        groups: i64, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Transposed convolution ---

    pub fn flodl_conv_transpose2d(
        input: FlodlTensor, weight: FlodlTensor, bias: FlodlTensor,
        stride: *mut i64, padding: *mut i64,
        output_padding: *mut i64, dilation: *mut i64,
        groups: i64, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Transposed 1D convolution ---

    pub fn flodl_conv_transpose1d(
        input: FlodlTensor, weight: FlodlTensor, bias: FlodlTensor,
        stride: i64, padding: i64,
        output_padding: i64, dilation: i64,
        groups: i64, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Pooling ---

    pub fn flodl_max_pool2d(
        input: FlodlTensor, kernel_size: *mut i64,
        stride: *mut i64, padding: *mut i64, dilation: *mut i64,
        ceil_mode: i32, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_avg_pool2d(
        input: FlodlTensor, kernel_size: *mut i64,
        stride: *mut i64, padding: *mut i64,
        ceil_mode: i32, count_include_pad: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

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
        t: FlodlTensor, device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_to_device_async(
        t: FlodlTensor, device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_cuda_is_available() -> i32;
    pub fn flodl_cuda_device_count() -> i32;
    pub fn flodl_force_cuda_link() -> i32;
    pub fn flodl_set_current_device(device_index: i32);
    pub fn flodl_get_current_device() -> i32;
    pub fn flodl_cuda_synchronize(device_index: i32);

    // --- CUDA memory/utilization (monitor support) ---

    pub fn flodl_cuda_mem_info(
        device_index: i32, used_bytes: *mut u64, total_bytes: *mut u64,
    ) -> *mut i8;

    pub fn flodl_cuda_alloc_bytes(
        device_index: i32, allocated_bytes: *mut u64,
    ) -> *mut i8;

    pub fn flodl_cuda_active_bytes(
        device_index: i32, active_bytes: *mut u64,
    ) -> *mut i8;

    pub fn flodl_cuda_peak_active_bytes(
        device_index: i32, peak_bytes: *mut u64,
    ) -> *mut i8;

    pub fn flodl_cuda_peak_reserved_bytes(
        device_index: i32, peak_bytes: *mut u64,
    ) -> *mut i8;

    pub fn flodl_cuda_reset_peak_stats(device_index: i32);

    pub fn flodl_cuda_empty_cache();

    pub fn flodl_cuda_utilization(device_index: i32) -> i32;

    pub fn flodl_cuda_device_name(
        device_index: i32, buf: *mut i8, buf_len: i32,
    ) -> *mut i8;

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

    // --- Element-wise binary (differentiable) ---

    pub fn flodl_atan2(
        a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_maximum(
        a: FlodlTensor, b: FlodlTensor, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_minimum(
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
        n: i64, dtype: i32, device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_full(
        shape: *mut i64, ndim: i32, value: f64, dtype: i32,
        device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_randperm(
        n: i64, dtype: i32, device_type: i32, device_index: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_multinomial(
        probs: FlodlTensor, num_samples: i64, replacement: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Normalization ---

    pub fn flodl_normalize(
        t: FlodlTensor, p: f64, dim: i32, result: *mut FlodlTensor,
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

    // mode: 0=constant, 1=reflect, 2=replicate, 3=circular
    pub fn flodl_pad_mode(
        t: FlodlTensor, padding: *mut i64, pad_len: i32,
        mode: i32, value: f64,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // mode: 0=nearest, 1=bilinear, 2=bicubic, 3=trilinear
    pub fn flodl_interpolate(
        input: FlodlTensor, output_size: *mut i64, ndim: i32,
        mode: i32, align_corners: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_flip(
        t: FlodlTensor, dims: *mut i64, ndim: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_roll(
        t: FlodlTensor, shift: i64, dim: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_split(
        t: FlodlTensor, split_size: i64, dim: i32,
        results: *mut *mut FlodlTensor, count: *mut i32,
    ) -> *mut i8;

    pub fn flodl_unbind(
        t: FlodlTensor, dim: i32,
        results: *mut *mut FlodlTensor, count: *mut i32,
    ) -> *mut i8;

    pub fn flodl_contiguous(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_is_contiguous(t: FlodlTensor) -> i32;

    pub fn flodl_argsort(
        t: FlodlTensor, dim: i32, descending: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_scatter(
        t: FlodlTensor, dim: i32, index: FlodlTensor, src: FlodlTensor,
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

    pub fn flodl_detach_(t: FlodlTensor) -> *mut i8;

    pub fn flodl_is_leaf(t: FlodlTensor) -> i32;

    // --- Autograd context ---

    pub fn flodl_no_grad_guard_new() -> *mut c_void;
    pub fn flodl_no_grad_guard_delete(guard: *mut c_void);
    pub fn flodl_is_grad_enabled() -> i32;

    // --- Autocast (automatic mixed precision) ---

    pub fn flodl_autocast_guard_new(device_type: i32, dtype: i32) -> *mut c_void;
    pub fn flodl_autocast_guard_delete(guard: *mut c_void);
    pub fn flodl_is_autocast_enabled(device_type: i32) -> i32;

    // --- Meshgrid ---

    pub fn flodl_meshgrid(
        tensors: *mut FlodlTensor, count: i32,
        results: *mut *mut FlodlTensor, result_count: *mut i32,
    ) -> *mut i8;

    // --- Pairwise distance ---

    pub fn flodl_cdist(
        x: FlodlTensor, y: FlodlTensor, p: f64,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Cosine similarity ---

    pub fn flodl_cosine_similarity(
        a: FlodlTensor, b: FlodlTensor,
        dim: i64, eps: f64,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Fused ops ---

    pub fn flodl_linear(
        input: FlodlTensor, weight: FlodlTensor, bias: FlodlTensor,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_gru_cell(
        input: FlodlTensor, hx: FlodlTensor,
        w_ih: FlodlTensor, w_hh: FlodlTensor,
        b_ih: FlodlTensor, b_hh: FlodlTensor,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_lstm_cell(
        input: FlodlTensor, hx: FlodlTensor, cx: FlodlTensor,
        w_ih: FlodlTensor, w_hh: FlodlTensor,
        b_ih: FlodlTensor, b_hh: FlodlTensor,
        h_out: *mut FlodlTensor, c_out: *mut FlodlTensor,
    ) -> *mut i8;

    // --- cuDNN benchmark ---

    pub fn flodl_set_cudnn_benchmark(enable: i32);

    // --- RNG seed ---

    pub fn flodl_manual_seed(seed: u64);
    pub fn flodl_cuda_manual_seed_all(seed: u64);

    // --- In-place operations ---

    pub fn flodl_add_(t: FlodlTensor, other: FlodlTensor) -> *mut i8;
    pub fn flodl_sub_(t: FlodlTensor, other: FlodlTensor) -> *mut i8;
    pub fn flodl_mul_scalar_(t: FlodlTensor, scalar: f64) -> *mut i8;
    pub fn flodl_add_scalar_(t: FlodlTensor, scalar: f64) -> *mut i8;
    pub fn flodl_zero_(t: FlodlTensor) -> *mut i8;
    pub fn flodl_mul_(t: FlodlTensor, other: FlodlTensor) -> *mut i8;
    pub fn flodl_div_scalar_(t: FlodlTensor, scalar: f64) -> *mut i8;
    pub fn flodl_div_(t: FlodlTensor, other: FlodlTensor) -> *mut i8;
    pub fn flodl_fill_(t: FlodlTensor, value: f64) -> *mut i8;

    // --- Fused Adam step ---

    pub fn flodl_adam_step(
        param: FlodlTensor, grad: FlodlTensor,
        m: FlodlTensor, v: FlodlTensor,
        lr: f64, beta1: f64, beta2: f64, eps: f64,
        weight_decay: f64, step: i64,
    ) -> *mut i8;

    // --- Batched Adam step ---

    pub fn flodl_adam_step_batched(
        params: *mut FlodlTensor, grads: *mut FlodlTensor,
        ms: *mut FlodlTensor, vs: *mut FlodlTensor,
        lrs: *mut f64, count: i32,
        beta1: f64, beta2: f64, eps: f64,
        weight_decay: f64, step: i64,
    ) -> *mut i8;

    // --- Fused Adam/AdamW (multi-tensor kernel) ---

    pub fn flodl_fused_adam_(
        params: *mut FlodlTensor, grads: *mut FlodlTensor,
        exp_avgs: *mut FlodlTensor, exp_avg_sqs: *mut FlodlTensor,
        count: i32, lr: f64,
        beta1: f64, beta2: f64, eps: f64,
        weight_decay: f64, step: i64,
        grad_scale: FlodlTensor, found_inf: FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_fused_adamw_(
        params: *mut FlodlTensor, grads: *mut FlodlTensor,
        exp_avgs: *mut FlodlTensor, exp_avg_sqs: *mut FlodlTensor,
        count: i32, lr: f64,
        beta1: f64, beta2: f64, eps: f64,
        weight_decay: f64, step: i64,
        grad_scale: FlodlTensor, found_inf: FlodlTensor,
    ) -> *mut i8;

    // --- Pinned memory ---

    pub fn flodl_pin_memory(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_is_pinned(t: FlodlTensor) -> i32;

    // --- Memory diagnostics ---

    pub fn flodl_malloc_trim() -> i32;

    // --- Zero grad (set_to_none) ---

    pub fn flodl_zero_grad_set_to_none(t: FlodlTensor);

    // --- Fused clip_grad_norm ---

    pub fn flodl_clip_grad_norm(
        params: *mut FlodlTensor, count: i32,
        max_norm: f64, total_norm_out: *mut f64,
    ) -> *mut i8;

    // --- Multi-tensor foreach operations ---

    pub fn flodl_foreach_add_scalar_(
        tensors: *mut FlodlTensor, count: i32, scalar: f64,
    ) -> *mut i8;

    pub fn flodl_foreach_mul_scalar_(
        tensors: *mut FlodlTensor, count: i32, scalar: f64,
    ) -> *mut i8;

    pub fn flodl_foreach_zero_(
        tensors: *mut FlodlTensor, count: i32,
    ) -> *mut i8;

    pub fn flodl_foreach_add_list_(
        tensors1: *mut FlodlTensor, tensors2: *mut FlodlTensor,
        count: i32, alpha: f64,
    ) -> *mut i8;

    pub fn flodl_foreach_norm(
        tensors: *mut FlodlTensor, count: i32, ord: f64,
        results: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_foreach_lerp_scalar_(
        tensors1: *mut FlodlTensor, tensors2: *mut FlodlTensor,
        count: i32, weight: f64,
    ) -> *mut i8;

    pub fn flodl_foreach_sqrt_(
        tensors: *mut FlodlTensor, count: i32,
    ) -> *mut i8;

    // --- Autograd diagnostics ---

    pub fn flodl_autograd_node_count(t: FlodlTensor) -> i64;

    // --- Fused loss functions ---

    pub fn flodl_mse_loss(
        pred: FlodlTensor, target: FlodlTensor,
        reduction: i64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_cross_entropy_loss(
        pred: FlodlTensor, target: FlodlTensor,
        reduction: i64, ignore_index: i64, label_smoothing: f64,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_bce_with_logits_loss(
        pred: FlodlTensor, target: FlodlTensor,
        reduction: i64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_bce_loss(
        pred: FlodlTensor, target: FlodlTensor,
        reduction: i64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_l1_loss(
        pred: FlodlTensor, target: FlodlTensor,
        reduction: i64, result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_smooth_l1_loss(
        pred: FlodlTensor, target: FlodlTensor,
        reduction: i64, beta: f64,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_kl_div_loss(
        input: FlodlTensor, target: FlodlTensor,
        reduction: i64, log_target: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Fused batch normalization ---

    pub fn flodl_batch_norm(
        input: FlodlTensor, weight: FlodlTensor,
        bias: FlodlTensor, running_mean: FlodlTensor,
        running_var: FlodlTensor, training: i32,
        momentum: f64, eps: f64,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- Fused dropout ---

    pub fn flodl_dropout(
        input: FlodlTensor, p: f64, training: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    pub fn flodl_feature_dropout(
        input: FlodlTensor, p: f64, training: i32,
        result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- In-place copy ---

    pub fn flodl_copy_(dst: FlodlTensor, src: FlodlTensor, non_blocking: i32) -> *mut i8;

    // --- Memory format ---

    pub fn flodl_to_channels_last(t: FlodlTensor, result: *mut FlodlTensor) -> *mut i8;
    pub fn flodl_is_channels_last(t: FlodlTensor) -> i32;

    // --- Embedding bag ---

    pub fn flodl_embedding_bag(
        weight: FlodlTensor, indices: FlodlTensor, offsets: FlodlTensor,
        mode: i64, result: *mut FlodlTensor,
    ) -> *mut i8;

    // --- CUDA Graphs ---

    pub fn flodl_cuda_graph_new(graph_out: *mut *mut c_void) -> *mut i8;
    pub fn flodl_cuda_graph_capture_begin(
        graph: *mut c_void, pool_hi: u64, pool_lo: u64, mode: i32,
    ) -> *mut i8;
    pub fn flodl_cuda_graph_capture_end(graph: *mut c_void) -> *mut i8;
    pub fn flodl_cuda_graph_replay(graph: *mut c_void) -> *mut i8;
    pub fn flodl_cuda_graph_reset(graph: *mut c_void) -> *mut i8;
    pub fn flodl_cuda_graph_delete(graph: *mut c_void);
    pub fn flodl_cuda_graph_pool(
        graph: *mut c_void, pool_hi: *mut u64, pool_lo: *mut u64,
    );
    pub fn flodl_cuda_graph_pool_handle(pool_hi: *mut u64, pool_lo: *mut u64);

    // --- Utility ---

    pub fn flodl_free_string(s: *mut i8);
}
