// shim.cpp — C++ implementations of the C shim functions.
//
// Each function wraps a libtorch C++ call and:
// 1. Catches any C++ exception
// 2. Returns the error message as a malloc'd C string
// 3. Returns NULL on success
//
// This prevents C++ exceptions from crossing the Rust FFI boundary
// (which would be instant UB).

#include "shim.h"
#include <torch/torch.h>
#include <cstring>
#include <string>

// Helper: convert a C++ exception to a malloc'd C string.
static char* make_error(const std::string& msg) {
    char* err = (char*)malloc(msg.size() + 1);
    if (err) {
        memcpy(err, msg.c_str(), msg.size() + 1);
    }
    return err;
}

// Helper: convert our dtype constant to torch::ScalarType.
static torch::ScalarType to_scalar_type(int dtype) {
    switch (dtype) {
        case FLODL_FLOAT16:  return torch::kFloat16;
        case FLODL_BFLOAT16: return torch::kBFloat16;
        case FLODL_FLOAT32:  return torch::kFloat32;
        case FLODL_FLOAT64:  return torch::kFloat64;
        case FLODL_INT32:    return torch::kInt32;
        case FLODL_INT64:    return torch::kInt64;
        default:           return torch::kFloat32;
    }
}

// Helper: convert our dtype constant back from torch::ScalarType.
static int from_scalar_type(torch::ScalarType st) {
    switch (st) {
        case torch::kFloat16:  return FLODL_FLOAT16;
        case torch::kBFloat16: return FLODL_BFLOAT16;
        case torch::kFloat32:  return FLODL_FLOAT32;
        case torch::kFloat64:  return FLODL_FLOAT64;
        case torch::kInt32:    return FLODL_INT32;
        case torch::kInt64:    return FLODL_INT64;
        default:               return FLODL_FLOAT32;
    }
}

// Helper: convert our device constant to torch::Device.
static torch::Device to_device(int device) {
    if (device == FLODL_CUDA) {
        return torch::Device(torch::kCUDA, 0);
    }
    return torch::Device(torch::kCPU);
}

// Helper: convert torch::Device back to our constant.
static int from_device(const torch::Device& dev) {
    if (dev.is_cuda()) return FLODL_CUDA;
    return FLODL_CPU;
}

// Helper: wrap a new torch::Tensor into a heap-allocated pointer.
// The caller (Rust) owns this pointer and must call flodl_free_tensor.
static FlodlTensor wrap(torch::Tensor t) {
    auto* p = new torch::Tensor(std::move(t));
    return (FlodlTensor)p;
}

// Helper: unwrap an FlodlTensor handle back to a reference.
static torch::Tensor& unwrap(FlodlTensor t) {
    return *((torch::Tensor*)t);
}

// Helper: build IntArrayRef from C array.
static torch::IntArrayRef make_shape(int64_t* shape, int ndim) {
    return torch::IntArrayRef(shape, ndim);
}

// --- Tensor creation ---

extern "C" char* flodl_zeros(int64_t* shape, int ndim, int dtype, int device,
                            FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device));
        *result = wrap(torch::zeros(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_ones(int64_t* shape, int ndim, int dtype, int device,
                           FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device));
        *result = wrap(torch::ones(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_rand(int64_t* shape, int ndim, int dtype, int device,
                           FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device));
        *result = wrap(torch::rand(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_randn(int64_t* shape, int ndim, int dtype, int device,
                             FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device));
        *result = wrap(torch::randn(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_from_blob(void* data, int64_t* shape, int ndim,
                                int dtype, int device, FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions().dtype(to_scalar_type(dtype));
        // from_blob does not take ownership — clone to get an independent copy.
        auto t = torch::from_blob(data, make_shape(shape, ndim), options).clone();
        if (device == FLODL_CUDA) {
            t = t.to(torch::kCUDA);
        }
        *result = wrap(std::move(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_linspace(double start, double end, int64_t steps,
                               int dtype, int device, FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions().dtype(to_scalar_type(dtype));
        auto t = torch::linspace(start, end, steps, options);
        if (device == FLODL_CUDA) {
            t = t.to(torch::kCUDA);
        }
        *result = wrap(std::move(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_arange(double start, double end, double step,
                              int dtype, int device, FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device));
        *result = wrap(torch::arange(start, end, step, options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_expand(FlodlTensor t, int64_t* new_shape, int ndim,
                             FlodlTensor* result) {
    try {
        // expand returns a view; contiguous() makes an owned copy.
        *result = wrap(unwrap(t).expand(make_shape(new_shape, ndim)).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Tensor lifecycle ---

extern "C" void flodl_free_tensor(FlodlTensor t) {
    if (t) {
        delete (torch::Tensor*)t;
    }
}

extern "C" char* flodl_shallow_clone(FlodlTensor t, FlodlTensor* result) {
    try {
        auto* src = reinterpret_cast<torch::Tensor*>(t);
        *result = new torch::Tensor(*src);  // Copy ctor: shares TensorImpl, bumps refcount.
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Tensor metadata ---

extern "C" int flodl_ndim(FlodlTensor t) {
    return (int)unwrap(t).dim();
}

extern "C" int64_t flodl_shape(FlodlTensor t, int dim) {
    return unwrap(t).size(dim);
}

extern "C" int flodl_dtype(FlodlTensor t) {
    return from_scalar_type(unwrap(t).scalar_type());
}

extern "C" int flodl_device(FlodlTensor t) {
    return from_device(unwrap(t).device());
}

extern "C" int64_t flodl_numel(FlodlTensor t) {
    return unwrap(t).numel();
}

// --- Data access ---

extern "C" char* flodl_copy_data(FlodlTensor t, void* buffer,
                                int64_t buffer_bytes) {
    try {
        auto tensor = unwrap(t);
        // Move to CPU if on another device
        if (!tensor.is_cpu()) {
            tensor = tensor.to(torch::kCPU);
        }
        // Ensure contiguous layout
        tensor = tensor.contiguous();
        int64_t data_bytes = tensor.numel() * tensor.element_size();
        if (buffer_bytes < data_bytes) {
            return make_error("buffer too small: need " +
                              std::to_string(data_bytes) + " bytes, got " +
                              std::to_string(buffer_bytes));
        }
        memcpy(buffer, tensor.data_ptr(), data_bytes);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Arithmetic ---

extern "C" char* flodl_add(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(a) + unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_sub(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(a) - unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_mul(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(a) * unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_div(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(a) / unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_matmul(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        *result = wrap(torch::matmul(unwrap(a), unwrap(b)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_add_scalar(FlodlTensor t, double scalar,
                                 FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t) + scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_mul_scalar(FlodlTensor t, double scalar,
                                 FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t) * scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_div_scalar(FlodlTensor t, double scalar,
                                 FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t) / scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_neg(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(-unwrap(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Activations ---

extern "C" char* flodl_relu(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::relu(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_sigmoid(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::sigmoid(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_tanh_op(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::tanh(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_softmax(FlodlTensor t, int dim, FlodlTensor* result) {
    try {
        *result = wrap(torch::softmax(unwrap(t), dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_log_softmax(FlodlTensor t, int dim, FlodlTensor* result) {
    try {
        *result = wrap(torch::log_softmax(unwrap(t), dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_gelu(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::gelu(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_silu(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::silu(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Layer normalization ---

extern "C" char* flodl_native_layer_norm(FlodlTensor input, FlodlTensor weight,
                                         FlodlTensor bias, int64_t normalized_size,
                                         double eps,
                                         FlodlTensor* output, FlodlTensor* mean,
                                         FlodlTensor* rstd) {
    try {
        auto result = at::native_layer_norm(
            unwrap(input), {normalized_size},
            weight ? c10::optional<torch::Tensor>(unwrap(weight)) : c10::nullopt,
            bias ? c10::optional<torch::Tensor>(unwrap(bias)) : c10::nullopt,
            eps);
        *output = wrap(std::get<0>(result));
        *mean = wrap(std::get<1>(result));
        *rstd = wrap(std::get<2>(result));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_native_layer_norm_backward(FlodlTensor grad_output,
                                                  FlodlTensor input,
                                                  FlodlTensor mean, FlodlTensor rstd,
                                                  FlodlTensor weight, FlodlTensor bias,
                                                  int64_t normalized_size,
                                                  FlodlTensor* grad_input,
                                                  FlodlTensor* grad_weight,
                                                  FlodlTensor* grad_bias) {
    try {
        c10::optional<torch::Tensor> w_opt = weight ? c10::optional<torch::Tensor>(unwrap(weight)) : c10::nullopt;
        c10::optional<torch::Tensor> b_opt = bias ? c10::optional<torch::Tensor>(unwrap(bias)) : c10::nullopt;
        std::array<bool, 3> mask = {true, true, true};
        auto result = at::native_layer_norm_backward(
            unwrap(grad_output), unwrap(input),
            {normalized_size},
            unwrap(mean), unwrap(rstd),
            w_opt, b_opt, mask);
        *grad_input = wrap(std::get<0>(result));
        *grad_weight = wrap(std::get<1>(result));
        *grad_bias = wrap(std::get<2>(result));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Element-wise math ---

extern "C" char* flodl_exp(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::exp(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_log(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::log(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_sqrt(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::sqrt(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_abs(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).abs());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_pow_scalar(FlodlTensor t, double exponent,
                                 FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).pow(exponent));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_clamp(FlodlTensor t, double min_val, double max_val,
                             FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).clamp(min_val, max_val));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Reductions ---

extern "C" char* flodl_sum(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).sum());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_mean(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).mean());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_sum_dim(FlodlTensor t, int dim, int keepdim,
                              FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).sum(dim, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_mean_dim(FlodlTensor t, int dim, int keepdim,
                               FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).mean(dim, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_min(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).min());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_max(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).max());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_norm(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).norm());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_min_dim(FlodlTensor t, int dim, int keepdim,
                              FlodlTensor* result) {
    try {
        auto [values, indices] = unwrap(t).min(dim, (bool)keepdim);
        *result = wrap(values);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_max_dim(FlodlTensor t, int dim, int keepdim,
                              FlodlTensor* result) {
    try {
        // std::get<0> gets the values (not the indices)
        *result = wrap(std::get<0>(unwrap(t).max(dim, keepdim != 0)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_argmax(FlodlTensor t, int dim, int keepdim,
                              FlodlTensor* result) {
    try {
        *result = wrap(torch::argmax(unwrap(t), dim, (bool)keepdim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Comparison (return float masks: 0.0 or 1.0) ---

extern "C" char* flodl_gt_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::gt(unwrap(t), scalar);
        *result = wrap(mask.to(unwrap(t).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_ge_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::ge(unwrap(t), scalar);
        *result = wrap(mask.to(unwrap(t).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_le_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::le(unwrap(t), scalar);
        *result = wrap(mask.to(unwrap(t).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_lt_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::lt(unwrap(t), scalar);
        *result = wrap(mask.to(unwrap(t).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Shape operations ---

extern "C" char* flodl_reshape(FlodlTensor t, int64_t* shape, int ndim,
                              FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).reshape(make_shape(shape, ndim)).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_transpose(FlodlTensor t, int dim0, int dim1,
                                FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).transpose(dim0, dim1).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_permute(FlodlTensor t, int64_t* dims, int ndim,
                              FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).permute(torch::IntArrayRef(dims, ndim)).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_select(FlodlTensor t, int dim, int64_t index,
                             FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).select(dim, index).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_narrow(FlodlTensor t, int dim, int64_t start,
                             int64_t length, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).narrow(dim, start, length).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_squeeze(FlodlTensor t, int dim, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).squeeze(dim).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_unsqueeze(FlodlTensor t, int dim, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).unsqueeze(dim).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_flatten(FlodlTensor t, int start_dim, int end_dim,
                              FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).flatten(start_dim, end_dim).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Scatter ---

extern "C" char* flodl_select_scatter(FlodlTensor input, FlodlTensor src,
                                     int dim, int64_t index,
                                     FlodlTensor* result) {
    try {
        auto out = unwrap(input).clone();
        out.select(dim, index).copy_(unwrap(src));
        *result = wrap(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_narrow_scatter(FlodlTensor input, FlodlTensor src,
                                     int dim, int64_t start,
                                     FlodlTensor* result) {
    try {
        auto out = unwrap(input).clone();
        out.narrow(dim, start, unwrap(src).size(dim)).copy_(unwrap(src));
        *result = wrap(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Indexing ---

extern "C" char* flodl_index_select(FlodlTensor t, int dim, FlodlTensor index,
                                   FlodlTensor* result) {
    try {
        *result = wrap(torch::index_select(unwrap(t), dim, unwrap(index)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_index_add(FlodlTensor t, int dim, FlodlTensor index,
                                FlodlTensor src, FlodlTensor* result) {
    try {
        // Out-of-place: returns t with src scattered at index positions.
        *result = wrap(unwrap(t).index_add(dim, unwrap(index), unwrap(src)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Concatenation ---

extern "C" char* flodl_cat2(FlodlTensor a, FlodlTensor b, int dim,
                           FlodlTensor* result) {
    try {
        *result = wrap(torch::cat({unwrap(a), unwrap(b)}, dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_stack(FlodlTensor* tensors, int count, int dim,
                            FlodlTensor* result) {
    try {
        std::vector<at::Tensor> vec;
        vec.reserve(count);
        for (int i = 0; i < count; i++) {
            vec.push_back(unwrap(tensors[i]));
        }
        *result = wrap(torch::stack(vec, dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Conditional ---

extern "C" char* flodl_where(FlodlTensor condition, FlodlTensor x,
                             FlodlTensor y, FlodlTensor* result) {
    try {
        auto cond = unwrap(condition).to(torch::kBool);
        *result = wrap(torch::where(cond, unwrap(x), unwrap(y)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Like constructors ---

extern "C" char* flodl_zeros_like(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::zeros_like(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_ones_like(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::ones_like(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Convolution ---

extern "C" char* flodl_conv2d(FlodlTensor input, FlodlTensor weight,
                             FlodlTensor bias,
                             int64_t* stride, int64_t* padding,
                             int64_t* dilation,
                             int64_t groups, FlodlTensor* result) {
    try {
        auto in = unwrap(input);
        auto w = unwrap(weight);
        c10::optional<torch::Tensor> b;
        if (bias != nullptr) {
            b = unwrap(bias);
        }
        *result = wrap(torch::conv2d(in, w, b,
            torch::IntArrayRef(stride, 2),
            torch::IntArrayRef(padding, 2),
            torch::IntArrayRef(dilation, 2),
            groups));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_conv2d_backward(FlodlTensor grad_output, FlodlTensor input,
                                      FlodlTensor weight,
                                      int64_t* stride, int64_t* padding,
                                      int64_t* dilation,
                                      int64_t groups, int compute_bias,
                                      FlodlTensor* grad_input,
                                      FlodlTensor* grad_weight,
                                      FlodlTensor* grad_bias) {
    try {
        auto go_ = unwrap(grad_output);
        auto in = unwrap(input);
        auto w = unwrap(weight);

        c10::OptionalIntArrayRef bias_sizes = c10::nullopt;
        std::vector<int64_t> bias_sizes_vec;
        if (compute_bias) {
            bias_sizes_vec = {w.size(0)};
            bias_sizes = bias_sizes_vec;
        }

        std::vector<int64_t> output_padding = {0, 0};
        auto result = at::convolution_backward(
            go_, in, w,
            bias_sizes,
            torch::IntArrayRef(stride, 2),
            torch::IntArrayRef(padding, 2),
            torch::IntArrayRef(dilation, 2),
            false, // transposed
            output_padding,
            groups,
            {true, true, compute_bias != 0}
        );

        *grad_input = wrap(std::get<0>(result));
        *grad_weight = wrap(std::get<1>(result));
        if (compute_bias) {
            *grad_bias = wrap(std::get<2>(result));
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Transposed convolution ---

extern "C" char* flodl_conv_transpose2d(FlodlTensor input, FlodlTensor weight,
                                       FlodlTensor bias,
                                       int64_t* stride, int64_t* padding,
                                       int64_t* output_padding, int64_t* dilation,
                                       int64_t groups, FlodlTensor* result) {
    try {
        auto in = unwrap(input);
        auto w = unwrap(weight);
        c10::optional<torch::Tensor> b;
        if (bias != nullptr) {
            b = unwrap(bias);
        }
        *result = wrap(torch::conv_transpose2d(in, w, b,
            torch::IntArrayRef(stride, 2),
            torch::IntArrayRef(padding, 2),
            torch::IntArrayRef(output_padding, 2),
            groups,
            torch::IntArrayRef(dilation, 2)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_conv_transpose2d_backward(FlodlTensor grad_output,
                                                 FlodlTensor input,
                                                 FlodlTensor weight,
                                                 int64_t* stride, int64_t* padding,
                                                 int64_t* output_padding,
                                                 int64_t* dilation,
                                                 int64_t groups, int compute_bias,
                                                 FlodlTensor* grad_input,
                                                 FlodlTensor* grad_weight,
                                                 FlodlTensor* grad_bias) {
    try {
        auto go_ = unwrap(grad_output);
        auto in = unwrap(input);
        auto w = unwrap(weight);

        c10::OptionalIntArrayRef bias_sizes = c10::nullopt;
        std::vector<int64_t> bias_sizes_vec;
        if (compute_bias) {
            bias_sizes_vec = {w.size(1) * groups};
            bias_sizes = bias_sizes_vec;
        }

        auto result = at::convolution_backward(
            go_, in, w,
            bias_sizes,
            torch::IntArrayRef(stride, 2),
            torch::IntArrayRef(padding, 2),
            torch::IntArrayRef(dilation, 2),
            true, // transposed
            torch::IntArrayRef(output_padding, 2),
            groups,
            {true, true, compute_bias != 0}
        );

        *grad_input = wrap(std::get<0>(result));
        *grad_weight = wrap(std::get<1>(result));
        if (compute_bias) {
            *grad_bias = wrap(std::get<2>(result));
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Pooling ---

extern "C" char* flodl_adaptive_avg_pool2d(FlodlTensor input, int64_t* output_size,
                                          FlodlTensor* result) {
    try {
        *result = wrap(at::adaptive_avg_pool2d(
            unwrap(input), torch::IntArrayRef(output_size, 2)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_adaptive_avg_pool2d_backward(FlodlTensor grad_output,
                                                    FlodlTensor input,
                                                    FlodlTensor* grad_input) {
    try {
        *grad_input = wrap(at::_adaptive_avg_pool2d_backward(
            unwrap(grad_output), unwrap(input)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Grid sampling ---

extern "C" char* flodl_grid_sample(FlodlTensor input, FlodlTensor grid,
                                  int mode, int padding_mode,
                                  int align_corners, FlodlTensor* result) {
    try {
        *result = wrap(at::grid_sampler(
            unwrap(input), unwrap(grid), mode, padding_mode, align_corners != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_grid_sample_backward(FlodlTensor grad_output,
                                           FlodlTensor input, FlodlTensor grid,
                                           int mode, int padding_mode,
                                           int align_corners,
                                           FlodlTensor* grad_input,
                                           FlodlTensor* grad_grid) {
    try {
        auto result = at::grid_sampler_2d_backward(
            unwrap(grad_output), unwrap(input), unwrap(grid),
            mode, padding_mode, align_corners != 0,
            {true, true});
        *grad_input = wrap(std::get<0>(result));
        *grad_grid = wrap(std::get<1>(result));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Dtype casting ---

extern "C" char* flodl_to_dtype(FlodlTensor t, int dtype, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).to(to_scalar_type(dtype)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_all_finite(FlodlTensor t, int* result) {
    try {
        auto& tensor = unwrap(t);
        *result = torch::isfinite(tensor).all().item<bool>() ? 1 : 0;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Device operations ---

extern "C" char* flodl_to_device(FlodlTensor t, int device,
                                FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).to(to_device(device)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" int flodl_cuda_is_available(void) {
    return torch::cuda::is_available() ? 1 : 0;
}

extern "C" int flodl_cuda_device_count(void) {
    return (int)torch::cuda::device_count();
}

// --- Comparison (tensor-tensor, return float masks: 0.0 or 1.0) ---

extern "C" char* flodl_gt_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::gt(unwrap(a), unwrap(b));
        *result = wrap(mask.to(unwrap(a).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_lt_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::lt(unwrap(a), unwrap(b));
        *result = wrap(mask.to(unwrap(a).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_ge_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::ge(unwrap(a), unwrap(b));
        *result = wrap(mask.to(unwrap(a).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_le_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::le(unwrap(a), unwrap(b));
        *result = wrap(mask.to(unwrap(a).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_eq_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::eq(unwrap(a), unwrap(b));
        *result = wrap(mask.to(unwrap(a).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_ne_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::ne(unwrap(a), unwrap(b));
        *result = wrap(mask.to(unwrap(a).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Additional reductions ---

extern "C" char* flodl_argmin(FlodlTensor t, int dim, int keepdim, FlodlTensor* result) {
    try {
        *result = wrap(torch::argmin(unwrap(t), dim, (bool)keepdim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_var(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).var());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_std_op(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).std());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_var_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).var({(int64_t)dim}, 1, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_std_dim(FlodlTensor t, int dim, int keepdim, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).std({(int64_t)dim}, 1, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Element-wise math (trig, rounding, sign) ---

extern "C" char* flodl_sin(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::sin(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_cos(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::cos(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_sign(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::sign(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_floor(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::floor(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_ceil(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::ceil(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_round(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::round(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_reciprocal(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(torch::reciprocal(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Advanced indexing ---

extern "C" char* flodl_gather(FlodlTensor t, int dim, FlodlTensor index,
                              FlodlTensor* result) {
    try {
        *result = wrap(torch::gather(unwrap(t), dim, unwrap(index)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_scatter_add(FlodlTensor t, int dim, FlodlTensor index,
                                   FlodlTensor src, FlodlTensor* result) {
    try {
        auto out = unwrap(t).clone();
        out.scatter_add_(dim, unwrap(index), unwrap(src));
        *result = wrap(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Sorting ---

extern "C" char* flodl_topk(FlodlTensor t, int64_t k, int dim, int largest, int sorted,
                            FlodlTensor* values, FlodlTensor* indices) {
    try {
        auto [v, i] = torch::topk(unwrap(t), k, dim, largest != 0, sorted != 0);
        *values = wrap(v);
        *indices = wrap(i);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_sort(FlodlTensor t, int dim, int descending,
                            FlodlTensor* values, FlodlTensor* indices) {
    try {
        auto [v, i] = torch::sort(unwrap(t), dim, descending != 0);
        *values = wrap(v);
        *indices = wrap(i);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Tensor creation (additional) ---

extern "C" char* flodl_eye(int64_t n, int dtype, int device, FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device));
        *result = wrap(torch::eye(n, options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_full(int64_t* shape, int ndim, double value, int dtype, int device,
                            FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device));
        *result = wrap(torch::full(make_shape(shape, ndim), value, options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Shape operations (additional) ---

extern "C" char* flodl_chunk(FlodlTensor t, int chunks, int dim,
                             FlodlTensor** results, int* count) {
    try {
        auto chunks_vec = torch::chunk(unwrap(t), chunks, dim);
        int n = (int)chunks_vec.size();
        auto* arr = (FlodlTensor*)malloc(sizeof(FlodlTensor) * n);
        if (!arr) {
            return make_error("malloc failed");
        }
        for (int i = 0; i < n; i++) {
            arr[i] = wrap(chunks_vec[i].contiguous());
        }
        *results = arr;
        *count = n;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_repeat(FlodlTensor t, int64_t* repeats, int ndim,
                              FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).repeat(make_shape(repeats, ndim)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_pad(FlodlTensor t, int64_t* padding, int pad_len, double value,
                           FlodlTensor* result) {
    try {
        *result = wrap(at::constant_pad_nd(unwrap(t),
                       torch::IntArrayRef(padding, pad_len), value));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Utility ---

extern "C" void flodl_free_string(char* s) {
    free(s);
}
