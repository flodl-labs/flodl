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
char* flodl_native_layer_norm_backward(FlodlTensor grad_output, FlodlTensor input,
                                      FlodlTensor mean, FlodlTensor rstd,
                                      FlodlTensor weight, FlodlTensor bias,
                                      int64_t normalized_size,
                                      FlodlTensor* grad_input, FlodlTensor* grad_weight,
                                      FlodlTensor* grad_bias);

// --- Element-wise math ---

char* flodl_exp(FlodlTensor t, FlodlTensor* result);
char* flodl_log(FlodlTensor t, FlodlTensor* result);
char* flodl_sqrt(FlodlTensor t, FlodlTensor* result);
char* flodl_abs(FlodlTensor t, FlodlTensor* result);
char* flodl_pow_scalar(FlodlTensor t, double exponent, FlodlTensor* result);
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
char* flodl_conv2d_backward(FlodlTensor grad_output, FlodlTensor input,
                          FlodlTensor weight,
                          int64_t* stride, int64_t* padding,
                          int64_t* dilation, int64_t groups,
                          int compute_bias,
                          FlodlTensor* grad_input, FlodlTensor* grad_weight,
                          FlodlTensor* grad_bias);

// --- Transposed convolution ---

char* flodl_conv_transpose2d(FlodlTensor input, FlodlTensor weight, FlodlTensor bias,
                           int64_t* stride, int64_t* padding,
                           int64_t* output_padding, int64_t* dilation,
                           int64_t groups, FlodlTensor* result);
char* flodl_conv_transpose2d_backward(FlodlTensor grad_output, FlodlTensor input,
                                    FlodlTensor weight,
                                    int64_t* stride, int64_t* padding,
                                    int64_t* output_padding, int64_t* dilation,
                                    int64_t groups, int compute_bias,
                                    FlodlTensor* grad_input,
                                    FlodlTensor* grad_weight,
                                    FlodlTensor* grad_bias);

// --- Pooling ---

char* flodl_adaptive_avg_pool2d(FlodlTensor input, int64_t* output_size,
                              FlodlTensor* result);
char* flodl_adaptive_avg_pool2d_backward(FlodlTensor grad_output, FlodlTensor input,
                                       FlodlTensor* grad_input);

// --- Grid sampling ---

char* flodl_grid_sample(FlodlTensor input, FlodlTensor grid,
                      int mode, int padding_mode, int align_corners,
                      FlodlTensor* result);
char* flodl_grid_sample_backward(FlodlTensor grad_output,
                               FlodlTensor input, FlodlTensor grid,
                               int mode, int padding_mode, int align_corners,
                               FlodlTensor* grad_input, FlodlTensor* grad_grid);

// --- Device ---

char* flodl_to_device(FlodlTensor t, int device, FlodlTensor* result);
int flodl_cuda_is_available(void);
int flodl_cuda_device_count(void);

// --- Dtype casting ---

char* flodl_to_dtype(FlodlTensor t, int dtype, FlodlTensor* result);
char* flodl_all_finite(FlodlTensor t, int* result);

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
