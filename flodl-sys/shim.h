// shim.h — C interface to libtorch's C++ API.
//
// Rust calls these via extern "C" FFI. The implementations live in
// shim.cpp (compiled as C++) and export C linkage.
//
// Design: every function that can fail returns an error string (caller
// must free it with flodl_free_string). NULL means success.

#ifndef FLODL_SHIM_H
#define FLODL_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a torch::Tensor.
typedef void* FlodlTensor;

// --- Tensor creation ---

char* flodl_zeros(int64_t* shape, int ndim, int dtype, int device,
                FlodlTensor* result);
char* flodl_ones(int64_t* shape, int ndim, int dtype, int device,
               FlodlTensor* result);
char* flodl_rand(int64_t* shape, int ndim, int dtype, int device,
               FlodlTensor* result);
char* flodl_randn(int64_t* shape, int ndim, int dtype, int device,
                FlodlTensor* result);
char* flodl_from_blob(void* data, int64_t* shape, int ndim, int dtype,
                    int device, FlodlTensor* result);
char* flodl_linspace(double start, double end, int64_t steps, int dtype,
                   int device, FlodlTensor* result);
char* flodl_arange(double start, double end, double step, int dtype,
                 int device, FlodlTensor* result);
char* flodl_expand(FlodlTensor t, int64_t* new_shape, int ndim,
                 FlodlTensor* result);

// --- Tensor lifecycle ---

void flodl_free_tensor(FlodlTensor t);
char* flodl_shallow_clone(FlodlTensor t, FlodlTensor* result);

// --- Tensor metadata ---

int flodl_ndim(FlodlTensor t);
int64_t flodl_shape(FlodlTensor t, int dim);
int flodl_dtype(FlodlTensor t);
int flodl_device(FlodlTensor t);
int64_t flodl_numel(FlodlTensor t);

// --- Data access ---

char* flodl_copy_data(FlodlTensor t, void* buffer, int64_t buffer_bytes);

// --- Arithmetic ---

char* flodl_add(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_sub(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_mul(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_div(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_matmul(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_add_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_mul_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_div_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_neg(FlodlTensor t, FlodlTensor* result);

// --- Activations ---

char* flodl_relu(FlodlTensor t, FlodlTensor* result);
char* flodl_sigmoid(FlodlTensor t, FlodlTensor* result);
char* flodl_tanh_op(FlodlTensor t, FlodlTensor* result);
char* flodl_softmax(FlodlTensor t, int dim, FlodlTensor* result);
char* flodl_log_softmax(FlodlTensor t, int dim, FlodlTensor* result);
char* flodl_gelu(FlodlTensor t, FlodlTensor* result);
char* flodl_silu(FlodlTensor t, FlodlTensor* result);

// --- Layer normalization ---

char* flodl_native_layer_norm(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                             int64_t normalized_size, double eps,
                             FlodlTensor* output, FlodlTensor* mean, FlodlTensor* rstd);
// --- Element-wise math ---

char* flodl_exp(FlodlTensor t, FlodlTensor* result);
char* flodl_log(FlodlTensor t, FlodlTensor* result);
char* flodl_sqrt(FlodlTensor t, FlodlTensor* result);
char* flodl_abs(FlodlTensor t, FlodlTensor* result);
char* flodl_pow_scalar(FlodlTensor t, double exponent, FlodlTensor* result);
char* flodl_triu(FlodlTensor t, int64_t diagonal, FlodlTensor* result);
char* flodl_clamp(FlodlTensor t, double min_val, double max_val,
                FlodlTensor* result);

// --- Reductions ---

char* flodl_sum(FlodlTensor t, FlodlTensor* result);
char* flodl_mean(FlodlTensor t, FlodlTensor* result);
char* flodl_sum_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_mean_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_min(FlodlTensor t, FlodlTensor* result);
char* flodl_max(FlodlTensor t, FlodlTensor* result);
char* flodl_norm(FlodlTensor t, FlodlTensor* result);
char* flodl_min_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_max_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_argmax(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);

// --- Comparison (return float masks: 0.0 or 1.0) ---

char* flodl_gt_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_ge_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_le_scalar(FlodlTensor t, double scalar, FlodlTensor* result);
char* flodl_lt_scalar(FlodlTensor t, double scalar, FlodlTensor* result);

// --- Comparison (tensor-tensor, return float masks: 0.0 or 1.0) ---

char* flodl_gt_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_lt_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_ge_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_le_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_eq_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);
char* flodl_ne_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result);

// --- Additional reductions ---

char* flodl_argmin(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_var(FlodlTensor t, FlodlTensor* result);
char* flodl_std_op(FlodlTensor t, FlodlTensor* result);
char* flodl_var_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);
char* flodl_std_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result);

// --- Element-wise math (trig, rounding, sign) ---

char* flodl_sin(FlodlTensor t, FlodlTensor* result);
char* flodl_cos(FlodlTensor t, FlodlTensor* result);
char* flodl_sign(FlodlTensor t, FlodlTensor* result);
char* flodl_floor(FlodlTensor t, FlodlTensor* result);
char* flodl_ceil(FlodlTensor t, FlodlTensor* result);
char* flodl_round(FlodlTensor t, FlodlTensor* result);
char* flodl_reciprocal(FlodlTensor t, FlodlTensor* result);

// --- Advanced indexing ---

char* flodl_gather(FlodlTensor t, int dim, FlodlTensor index,
                  FlodlTensor* result);
char* flodl_scatter_add(FlodlTensor t, int dim, FlodlTensor index,
                       FlodlTensor src, FlodlTensor* result);

// --- Sorting ---

char* flodl_topk(FlodlTensor t, int64_t k, int dim, int largest, int sorted,
                FlodlTensor* values, FlodlTensor* indices);
char* flodl_sort(FlodlTensor t, int dim, int descending,
                FlodlTensor* values, FlodlTensor* indices);

// --- Tensor creation ---

char* flodl_eye(int64_t n, int dtype, int device, FlodlTensor* result);
char* flodl_full(int64_t* shape, int ndim, double value, int dtype, int device,
                FlodlTensor* result);

// --- Shape operations (additional) ---

char* flodl_chunk(FlodlTensor t, int chunks, int dim,
                 FlodlTensor** results, int* count);
char* flodl_repeat(FlodlTensor t, int64_t* repeats, int ndim,
                  FlodlTensor* result);
char* flodl_pad(FlodlTensor t, int64_t* padding, int pad_len, double value,
               FlodlTensor* result);

// --- Shape operations ---

char* flodl_reshape(FlodlTensor t, int64_t* shape, int ndim, FlodlTensor* result);
char* flodl_transpose(FlodlTensor t, int dim0, int dim1, FlodlTensor* result);
char* flodl_permute(FlodlTensor t, int64_t* dims, int ndim, FlodlTensor* result);
char* flodl_select(FlodlTensor t, int dim, int64_t index, FlodlTensor* result);
char* flodl_narrow(FlodlTensor t, int dim, int64_t start, int64_t length,
                 FlodlTensor* result);
char* flodl_squeeze(FlodlTensor t, int dim, FlodlTensor* result);
char* flodl_unsqueeze(FlodlTensor t, int dim, FlodlTensor* result);
char* flodl_flatten(FlodlTensor t, int start_dim, int end_dim, FlodlTensor* result);

// --- Scatter ---

char* flodl_select_scatter(FlodlTensor input, FlodlTensor src, int dim,
                         int64_t index, FlodlTensor* result);
char* flodl_narrow_scatter(FlodlTensor input, FlodlTensor src, int dim,
                         int64_t start, FlodlTensor* result);

// --- Indexing ---

char* flodl_index_select(FlodlTensor t, int dim, FlodlTensor index,
                       FlodlTensor* result);
char* flodl_index_add(FlodlTensor t, int dim, FlodlTensor index,
                    FlodlTensor src, FlodlTensor* result);

// --- Concatenation ---

char* flodl_cat2(FlodlTensor a, FlodlTensor b, int dim, FlodlTensor* result);
char* flodl_stack(FlodlTensor* tensors, int count, int dim, FlodlTensor* result);

// --- Conditional ---

char* flodl_where(FlodlTensor condition, FlodlTensor x, FlodlTensor y,
                FlodlTensor* result);

// --- Like constructors ---

char* flodl_zeros_like(FlodlTensor t, FlodlTensor* result);
char* flodl_ones_like(FlodlTensor t, FlodlTensor* result);

// --- Convolution ---

char* flodl_conv2d(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                 int64_t* stride, int64_t* padding, int64_t* dilation,
                 int64_t groups, FlodlTensor* result);
// --- Transposed convolution ---

char* flodl_conv_transpose2d(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                           int64_t* stride, int64_t* padding,
                           int64_t* output_padding, int64_t* dilation,
                           int64_t groups, FlodlTensor* result);
// --- Pooling ---

char* flodl_adaptive_avg_pool2d(FlodlTensor input, int64_t* output_size,
                              FlodlTensor* result);
// --- Grid sampling ---

char* flodl_grid_sample(FlodlTensor input, FlodlTensor grid,
                      int mode, int padding_mode, int align_corners,
                      FlodlTensor* result);
// --- Fused ops ---

char* flodl_linear(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                  FlodlTensor* result);
char* flodl_gru_cell(FlodlTensor input, FlodlTensor hx,
                    FlodlTensor w_ih, FlodlTensor w_hh,
                    FlodlTensor b_ih, FlodlTensor b_hh,
                    FlodlTensor* result);
char* flodl_lstm_cell(FlodlTensor input, FlodlTensor hx, FlodlTensor cx,
                     FlodlTensor w_ih, FlodlTensor w_hh,
                     FlodlTensor b_ih, FlodlTensor b_hh,
                     FlodlTensor* h_out, FlodlTensor* c_out);

// --- Device ---

char* flodl_to_device(FlodlTensor t, int device, FlodlTensor* result);
int flodl_cuda_is_available(void);
int flodl_cuda_device_count(void);
int flodl_force_cuda_link(void);
void flodl_set_cudnn_benchmark(int enable);

// --- CUDA memory/utilization (monitor support) ---

// Query CUDA memory: writes used and total bytes for the current device.
// Returns error string on failure (caller must free), NULL on success.
char* flodl_cuda_mem_info(uint64_t* used_bytes, uint64_t* total_bytes);

// Query GPU utilization percentage (0-100) via NVML.
// Returns -1 if NVML is not available or query fails.
int flodl_cuda_utilization(int device_index);

// --- Dtype casting ---

char* flodl_to_dtype(FlodlTensor t, int dtype, FlodlTensor* result);
char* flodl_all_finite(FlodlTensor t, int* result);

// --- Autograd ---

char* flodl_set_requires_grad(FlodlTensor t, int requires_grad, FlodlTensor* result);
int flodl_requires_grad(FlodlTensor t);
char* flodl_backward(FlodlTensor t);
char* flodl_grad(FlodlTensor t, FlodlTensor* result);
char* flodl_set_grad(FlodlTensor t, FlodlTensor grad);
char* flodl_zero_grad(FlodlTensor t);
char* flodl_detach(FlodlTensor t, FlodlTensor* result);
char* flodl_detach_(FlodlTensor t);
int flodl_is_leaf(FlodlTensor t);

// --- Autograd context ---

void* flodl_no_grad_guard_new(void);
void flodl_no_grad_guard_delete(void* guard);
int flodl_is_grad_enabled(void);

// --- In-place operations ---

char* flodl_add_(FlodlTensor t, FlodlTensor other);
char* flodl_sub_(FlodlTensor t, FlodlTensor other);
char* flodl_mul_scalar_(FlodlTensor t, double scalar);
char* flodl_add_scalar_(FlodlTensor t, double scalar);
char* flodl_zero_(FlodlTensor t);

// --- Meshgrid ---

char* flodl_meshgrid(FlodlTensor* tensors, int count,
                    FlodlTensor** results, int* result_count);

// --- Pairwise distance ---

char* flodl_cdist(FlodlTensor x, FlodlTensor y, double p,
                 FlodlTensor* result);

// --- Memory diagnostics ---

// Ask glibc to return free memory to the OS. Returns 1 if memory was
// actually released, 0 otherwise. Useful for distinguishing allocator
// fragmentation from real leaks.
int flodl_malloc_trim(void);

// --- Utility ---

void flodl_free_string(char* s);

// --- DType constants ---
#define FLODL_FLOAT16  5
#define FLODL_BFLOAT16 15
#define FLODL_FLOAT32  6
#define FLODL_FLOAT64  7
#define FLODL_INT32    3
#define FLODL_INT64    4

// --- Device constants ---
#define FLODL_CPU  0
#define FLODL_CUDA 1

#ifdef __cplusplus
}
#endif

#endif // FLODL_SHIM_H
