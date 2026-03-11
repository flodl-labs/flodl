// shim.h — C interface to libtorch's C++ API.
//
// Rust calls these via extern "C" FFI. The implementations live in
// shim.cpp (compiled as C++) and export C linkage.
//
// Design: every function that can fail returns an error string (caller
// must free it with rdl_free_string). NULL means success.

#ifndef RDL_SHIM_H
#define RDL_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a torch::Tensor.
typedef void* RdlTensor;

// --- Tensor creation ---

char* rdl_zeros(int64_t* shape, int ndim, int dtype, int device,
                RdlTensor* result);
char* rdl_ones(int64_t* shape, int ndim, int dtype, int device,
               RdlTensor* result);
char* rdl_rand(int64_t* shape, int ndim, int dtype, int device,
               RdlTensor* result);
char* rdl_randn(int64_t* shape, int ndim, int dtype, int device,
                RdlTensor* result);
char* rdl_from_blob(void* data, int64_t* shape, int ndim, int dtype,
                    int device, RdlTensor* result);
char* rdl_linspace(double start, double end, int64_t steps, int dtype,
                   int device, RdlTensor* result);
char* rdl_arange(double start, double end, double step, int dtype,
                 int device, RdlTensor* result);
char* rdl_expand(RdlTensor t, int64_t* new_shape, int ndim,
                 RdlTensor* result);

// --- Tensor lifecycle ---

void rdl_free_tensor(RdlTensor t);
char* rdl_shallow_clone(RdlTensor t, RdlTensor* result);

// --- Tensor metadata ---

int rdl_ndim(RdlTensor t);
int64_t rdl_shape(RdlTensor t, int dim);
int rdl_dtype(RdlTensor t);
int rdl_device(RdlTensor t);
int64_t rdl_numel(RdlTensor t);

// --- Data access ---

char* rdl_copy_data(RdlTensor t, void* buffer, int64_t buffer_bytes);

// --- Arithmetic ---

char* rdl_add(RdlTensor a, RdlTensor b, RdlTensor* result);
char* rdl_sub(RdlTensor a, RdlTensor b, RdlTensor* result);
char* rdl_mul(RdlTensor a, RdlTensor b, RdlTensor* result);
char* rdl_div(RdlTensor a, RdlTensor b, RdlTensor* result);
char* rdl_matmul(RdlTensor a, RdlTensor b, RdlTensor* result);
char* rdl_add_scalar(RdlTensor t, double scalar, RdlTensor* result);
char* rdl_mul_scalar(RdlTensor t, double scalar, RdlTensor* result);
char* rdl_neg(RdlTensor t, RdlTensor* result);

// --- Activations ---

char* rdl_relu(RdlTensor t, RdlTensor* result);
char* rdl_sigmoid(RdlTensor t, RdlTensor* result);
char* rdl_tanh_op(RdlTensor t, RdlTensor* result);
char* rdl_softmax(RdlTensor t, int dim, RdlTensor* result);
char* rdl_log_softmax(RdlTensor t, int dim, RdlTensor* result);
char* rdl_gelu(RdlTensor t, RdlTensor* result);
char* rdl_silu(RdlTensor t, RdlTensor* result);

// --- Layer normalization ---

char* rdl_native_layer_norm(RdlTensor input, RdlTensor weight, RdlTensor bias,
                             int64_t normalized_size, double eps,
                             RdlTensor* output, RdlTensor* mean, RdlTensor* rstd);
char* rdl_native_layer_norm_backward(RdlTensor grad_output, RdlTensor input,
                                      RdlTensor mean, RdlTensor rstd,
                                      RdlTensor weight, RdlTensor bias,
                                      int64_t normalized_size,
                                      RdlTensor* grad_input, RdlTensor* grad_weight,
                                      RdlTensor* grad_bias);

// --- Element-wise math ---

char* rdl_exp(RdlTensor t, RdlTensor* result);
char* rdl_log(RdlTensor t, RdlTensor* result);
char* rdl_sqrt(RdlTensor t, RdlTensor* result);
char* rdl_abs(RdlTensor t, RdlTensor* result);
char* rdl_pow_scalar(RdlTensor t, double exponent, RdlTensor* result);
char* rdl_clamp(RdlTensor t, double min_val, double max_val,
                RdlTensor* result);

// --- Reductions ---

char* rdl_sum(RdlTensor t, RdlTensor* result);
char* rdl_sum_dim(RdlTensor t, int dim, int keepdim, RdlTensor* result);
char* rdl_mean_dim(RdlTensor t, int dim, int keepdim, RdlTensor* result);
char* rdl_min(RdlTensor t, RdlTensor* result);
char* rdl_min_dim(RdlTensor t, int dim, int keepdim, RdlTensor* result);
char* rdl_max_dim(RdlTensor t, int dim, int keepdim, RdlTensor* result);
char* rdl_argmax(RdlTensor t, int dim, int keepdim, RdlTensor* result);

// --- Comparison (return float masks: 0.0 or 1.0) ---

char* rdl_gt_scalar(RdlTensor t, double scalar, RdlTensor* result);
char* rdl_ge_scalar(RdlTensor t, double scalar, RdlTensor* result);
char* rdl_le_scalar(RdlTensor t, double scalar, RdlTensor* result);
char* rdl_lt_scalar(RdlTensor t, double scalar, RdlTensor* result);

// --- Shape operations ---

char* rdl_reshape(RdlTensor t, int64_t* shape, int ndim, RdlTensor* result);
char* rdl_transpose(RdlTensor t, int dim0, int dim1, RdlTensor* result);
char* rdl_permute(RdlTensor t, int64_t* dims, int ndim, RdlTensor* result);
char* rdl_select(RdlTensor t, int dim, int64_t index, RdlTensor* result);
char* rdl_narrow(RdlTensor t, int dim, int64_t start, int64_t length,
                 RdlTensor* result);
char* rdl_squeeze(RdlTensor t, int dim, RdlTensor* result);
char* rdl_unsqueeze(RdlTensor t, int dim, RdlTensor* result);

// --- Scatter ---

char* rdl_select_scatter(RdlTensor input, RdlTensor src, int dim,
                         int64_t index, RdlTensor* result);
char* rdl_narrow_scatter(RdlTensor input, RdlTensor src, int dim,
                         int64_t start, RdlTensor* result);

// --- Indexing ---

char* rdl_index_select(RdlTensor t, int dim, RdlTensor index,
                       RdlTensor* result);
char* rdl_index_add(RdlTensor t, int dim, RdlTensor index,
                    RdlTensor src, RdlTensor* result);

// --- Concatenation ---

char* rdl_cat2(RdlTensor a, RdlTensor b, int dim, RdlTensor* result);

// --- Conditional ---

char* rdl_where(RdlTensor condition, RdlTensor x, RdlTensor y,
                RdlTensor* result);

// --- Like constructors ---

char* rdl_zeros_like(RdlTensor t, RdlTensor* result);
char* rdl_ones_like(RdlTensor t, RdlTensor* result);

// --- Convolution ---

char* rdl_conv2d(RdlTensor input, RdlTensor weight, RdlTensor bias,
                 int64_t* stride, int64_t* padding, int64_t* dilation,
                 int64_t groups, RdlTensor* result);
char* rdl_conv2d_backward(RdlTensor grad_output, RdlTensor input,
                          RdlTensor weight,
                          int64_t* stride, int64_t* padding,
                          int64_t* dilation, int64_t groups,
                          int compute_bias,
                          RdlTensor* grad_input, RdlTensor* grad_weight,
                          RdlTensor* grad_bias);

// --- Transposed convolution ---

char* rdl_conv_transpose2d(RdlTensor input, RdlTensor weight, RdlTensor bias,
                           int64_t* stride, int64_t* padding,
                           int64_t* output_padding, int64_t* dilation,
                           int64_t groups, RdlTensor* result);
char* rdl_conv_transpose2d_backward(RdlTensor grad_output, RdlTensor input,
                                    RdlTensor weight,
                                    int64_t* stride, int64_t* padding,
                                    int64_t* output_padding, int64_t* dilation,
                                    int64_t groups, int compute_bias,
                                    RdlTensor* grad_input,
                                    RdlTensor* grad_weight,
                                    RdlTensor* grad_bias);

// --- Pooling ---

char* rdl_adaptive_avg_pool2d(RdlTensor input, int64_t* output_size,
                              RdlTensor* result);
char* rdl_adaptive_avg_pool2d_backward(RdlTensor grad_output, RdlTensor input,
                                       RdlTensor* grad_input);

// --- Grid sampling ---

char* rdl_grid_sample(RdlTensor input, RdlTensor grid,
                      int mode, int padding_mode, int align_corners,
                      RdlTensor* result);
char* rdl_grid_sample_backward(RdlTensor grad_output,
                               RdlTensor input, RdlTensor grid,
                               int mode, int padding_mode, int align_corners,
                               RdlTensor* grad_input, RdlTensor* grad_grid);

// --- Device ---

char* rdl_to_device(RdlTensor t, int device, RdlTensor* result);
int rdl_cuda_is_available(void);
int rdl_cuda_device_count(void);

// --- Dtype casting ---

char* rdl_to_dtype(RdlTensor t, int dtype, RdlTensor* result);
char* rdl_all_finite(RdlTensor t, int* result);

// --- Utility ---

void rdl_free_string(char* s);

// --- DType constants ---
#define RDL_FLOAT16  5
#define RDL_BFLOAT16 15
#define RDL_FLOAT32  6
#define RDL_FLOAT64  7
#define RDL_INT32    3
#define RDL_INT64    4

// --- Device constants ---
#define RDL_CPU  0
#define RDL_CUDA 1

#ifdef __cplusplus
}
#endif

#endif // RDL_SHIM_H
