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
#include <torch/csrc/autograd/function.h>
#include <cstring>
#include <string>
#include <queue>
#include <unordered_set>
#ifdef __linux__
#include <malloc.h>
#endif

#ifdef FLODL_BUILD_CUDA
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

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

// Helper: convert our device (type, index) to torch::Device.
static torch::Device to_device(int device_type, int device_index) {
    if (device_type == FLODL_CUDA) {
        return torch::Device(torch::kCUDA, (c10::DeviceIndex)device_index);
    }
    return torch::Device(torch::kCPU);
}

// Helper: convert torch::Device back to our (type, index) pair.
static int from_device_type(const torch::Device& dev) {
    if (dev.is_cuda()) return FLODL_CUDA;
    return FLODL_CPU;
}

static int from_device_index(const torch::Device& dev) {
    if (dev.is_cuda()) return (int)dev.index();
    return 0;
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

extern "C" char* flodl_zeros(int64_t* shape, int ndim, int dtype,
                            int device_type, int device_index,
                            FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::zeros(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_ones(int64_t* shape, int ndim, int dtype,
                           int device_type, int device_index,
                           FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::ones(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_rand(int64_t* shape, int ndim, int dtype,
                           int device_type, int device_index,
                           FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::rand(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_randn(int64_t* shape, int ndim, int dtype,
                             int device_type, int device_index,
                             FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::randn(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_from_blob(void* data, int64_t* shape, int ndim,
                                int dtype, int device_type, int device_index,
                                FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions().dtype(to_scalar_type(dtype));
        // from_blob does not take ownership — clone to get an independent copy.
        auto t = torch::from_blob(data, make_shape(shape, ndim), options).clone();
        if (device_type == FLODL_CUDA) {
            t = t.to(to_device(device_type, device_index));
        }
        *result = wrap(std::move(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_linspace(double start, double end, int64_t steps,
                               int dtype, int device_type, int device_index,
                               FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions().dtype(to_scalar_type(dtype));
        auto t = torch::linspace(start, end, steps, options);
        if (device_type == FLODL_CUDA) {
            t = t.to(to_device(device_type, device_index));
        }
        *result = wrap(std::move(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_arange(double start, double end, double step,
                              int dtype, int device_type, int device_index,
                              FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
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

extern "C" int flodl_device_type(FlodlTensor t) {
    return from_device_type(unwrap(t).device());
}

extern "C" int flodl_device_index(FlodlTensor t) {
    return from_device_index(unwrap(t).device());
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

extern "C" char* flodl_triu(FlodlTensor t, int64_t diagonal,
                            FlodlTensor* result) {
    try {
        *result = wrap(torch::triu(unwrap(t), diagonal));
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
// Float inputs: mask uses input's dtype. Non-float (Int64, Bool): mask is Float32.
static inline torch::ScalarType mask_dtype(const torch::Tensor& t) {
    return t.is_floating_point() ? t.scalar_type() : torch::kFloat32;
}

extern "C" char* flodl_gt_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::gt(unwrap(t), scalar);
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_ge_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::ge(unwrap(t), scalar);
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_le_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::le(unwrap(t), scalar);
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_lt_scalar(FlodlTensor t, double scalar,
                                FlodlTensor* result) {
    try {
        auto mask = torch::lt(unwrap(t), scalar);
        *result = wrap(mask.to(mask_dtype(unwrap(t))));
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

extern "C" char* flodl_cat(FlodlTensor* tensors, int count, int dim,
                           FlodlTensor* result) {
    try {
        std::vector<at::Tensor> vec;
        vec.reserve(count);
        for (int i = 0; i < count; i++) {
            vec.push_back(unwrap(tensors[i]));
        }
        *result = wrap(torch::cat(vec, dim));
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

// --- Pooling ---

extern "C" char* flodl_max_pool2d(FlodlTensor input, int64_t* kernel_size,
                                 int64_t* stride, int64_t* padding, int64_t* dilation,
                                 int ceil_mode, FlodlTensor* result) {
    try {
        *result = wrap(at::max_pool2d(
            unwrap(input),
            torch::IntArrayRef(kernel_size, 2),
            torch::IntArrayRef(stride, 2),
            torch::IntArrayRef(padding, 2),
            torch::IntArrayRef(dilation, 2),
            ceil_mode != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

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

extern "C" char* flodl_to_device(FlodlTensor t, int device_type,
                                int device_index, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).to(to_device(device_type, device_index)));
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

extern "C" void flodl_set_current_device(int device_index) {
#ifdef FLODL_BUILD_CUDA
    c10::cuda::set_device((c10::DeviceIndex)device_index);
#else
    (void)device_index;
#endif
}

extern "C" int flodl_get_current_device(void) {
#ifdef FLODL_BUILD_CUDA
    return (int)c10::cuda::current_device();
#else
    return 0;
#endif
}

extern "C" void flodl_cuda_synchronize(int device_index) {
#ifdef FLODL_BUILD_CUDA
    if (torch::cuda::is_available()) {
        c10::cuda::set_device((c10::DeviceIndex)device_index);
        cudaDeviceSynchronize();
    }
#else
    (void)device_index;
#endif
}

// Force real symbol references to BOTH c10_cuda.so and libtorch_cuda.so
// so that --as-needed doesn't drop them from the link. Without this,
// no Rust code directly references symbols in these libraries, so the
// linker silently drops them and torch::cuda::is_available() returns
// false even with a GPU present.
//
// c10_cuda.so      -> c10::cuda::device_count()
// libtorch_cuda.so -> torch::CudaIPCCollect() (static initializers in
//                     this library register the CUDA backend)
#ifdef FLODL_BUILD_CUDA
#include <cuda_runtime.h>
#include <dlfcn.h>
namespace torch { void CudaIPCCollect(); }
#endif

extern "C" int flodl_force_cuda_link(void) {
#ifdef FLODL_BUILD_CUDA
    // c10_cuda.so dependency
    volatile int n = (int)c10::cuda::device_count();
    // libtorch_cuda.so dependency -- take address, don't call
    volatile auto p = &torch::CudaIPCCollect;
    (void)p;
    return n;
#else
    return 0;
#endif
}

// --- CUDA memory info via cudaMemGetInfo ---

extern "C" char* flodl_cuda_mem_info(int device_index,
                                    uint64_t* used_bytes, uint64_t* total_bytes) {
#ifdef FLODL_BUILD_CUDA
    if (!torch::cuda::is_available()) {
        return make_error("CUDA not available");
    }
    // Switch to target device, query, then restore
    auto prev = c10::cuda::current_device();
    c10::cuda::set_device((c10::DeviceIndex)device_index);
    size_t free_b = 0, total_b = 0;
    auto err = cudaMemGetInfo(&free_b, &total_b);
    c10::cuda::set_device(prev);
    if (err != cudaSuccess) {
        return make_error(cudaGetErrorString(err));
    }
    *total_bytes = (uint64_t)total_b;
    *used_bytes  = (uint64_t)(total_b - free_b);
    return nullptr;
#else
    (void)device_index; (void)used_bytes; (void)total_bytes;
    return make_error("CUDA not available (built without cuda feature)");
#endif
}

// --- CUDA caching allocator stats ---

extern "C" char* flodl_cuda_alloc_bytes(int device_index,
                                         uint64_t* allocated_bytes) {
#ifdef FLODL_BUILD_CUDA
    if (!torch::cuda::is_available()) {
        return make_error("CUDA not available");
    }
    try {
        auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
            (c10::DeviceIndex)device_index);
        // reserved_bytes = total memory grabbed from CUDA driver (including
        // unified-memory spill to host RAM).  allocated_bytes only counts
        // actively-used sub-blocks, which never exceeds physical VRAM.
        *allocated_bytes = (uint64_t)stats.reserved_bytes[0].current;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
#else
    (void)device_index; (void)allocated_bytes;
    return make_error("CUDA not available (built without cuda feature)");
#endif
}

// Active allocator bytes (tensors actually in use, not cached free blocks).
// Matches torch.cuda.max_memory_allocated() semantics.
extern "C" char* flodl_cuda_active_bytes(int device_index,
                                          uint64_t* active_bytes) {
#ifdef FLODL_BUILD_CUDA
    if (!torch::cuda::is_available()) {
        return make_error("CUDA not available");
    }
    try {
        auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
            (c10::DeviceIndex)device_index);
        *active_bytes = (uint64_t)stats.allocated_bytes[0].current;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
#else
    (void)device_index; (void)active_bytes;
    return make_error("CUDA not available (built without cuda feature)");
#endif
}

// --- CUDA empty cache ---

extern "C" void flodl_cuda_empty_cache(void) {
#ifdef FLODL_BUILD_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
}

// --- GPU utilization via NVML (dynamically loaded) ---

#ifdef FLODL_BUILD_CUDA
namespace {
    typedef int nvml_ret_t;
    typedef void* nvml_device_t;
    struct NvmlUtil { unsigned int gpu; unsigned int memory; };

    struct NvmlState {
        bool tried = false;
        bool ok = false;
        nvml_ret_t (*init)(void) = nullptr;
        nvml_ret_t (*getHandle)(unsigned int, nvml_device_t*) = nullptr;
        nvml_ret_t (*getUtil)(nvml_device_t, NvmlUtil*) = nullptr;
    };
    static NvmlState nvml;

    static void nvml_try_load() {
        if (nvml.tried) return;
        nvml.tried = true;
        void* lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (!lib) return;
        nvml.init      = (decltype(nvml.init))dlsym(lib, "nvmlInit_v2");
        nvml.getHandle = (decltype(nvml.getHandle))dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
        nvml.getUtil   = (decltype(nvml.getUtil))dlsym(lib, "nvmlDeviceGetUtilizationRates");
        if (!nvml.init || !nvml.getHandle || !nvml.getUtil) return;
        nvml.ok = (nvml.init() == 0);
    }
} // anonymous namespace
#endif

extern "C" int flodl_cuda_utilization(int device_index) {
#ifdef FLODL_BUILD_CUDA
    nvml_try_load();
    if (!nvml.ok) return -1;
    nvml_device_t dev;
    if (nvml.getHandle((unsigned int)device_index, &dev) != 0) return -1;
    NvmlUtil util;
    if (nvml.getUtil(dev, &util) != 0) return -1;
    return (int)util.gpu;
#else
    (void)device_index;
    return -1;
#endif
}

// --- GPU device name ---

extern "C" char* flodl_cuda_device_name(int device_index, char* buf, int buf_len) {
#ifdef FLODL_BUILD_CUDA
    if (!torch::cuda::is_available()) {
        return make_error("CUDA not available");
    }
    cudaDeviceProp prop;
    auto err = cudaGetDeviceProperties(&prop, device_index);
    if (err != cudaSuccess) {
        return make_error(cudaGetErrorString(err));
    }
    snprintf(buf, buf_len, "%s", prop.name);
    return nullptr;
#else
    (void)device_index; (void)buf; (void)buf_len;
    return make_error("CUDA not available (built without cuda feature)");
#endif
}

// --- Comparison (tensor-tensor, return float masks: 0.0 or 1.0) ---

extern "C" char* flodl_gt_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::gt(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_lt_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::lt(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_ge_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::ge(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_le_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::le(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_eq_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::eq(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_ne_tensor(FlodlTensor a, FlodlTensor b, FlodlTensor* result) {
    try {
        auto mask = torch::ne(unwrap(a), unwrap(b));
        *result = wrap(mask.to(mask_dtype(unwrap(a))));
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

extern "C" char* flodl_eye(int64_t n, int dtype, int device_type,
                          int device_index, FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
        *result = wrap(torch::eye(n, options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_full(int64_t* shape, int ndim, double value, int dtype,
                            int device_type, int device_index,
                            FlodlTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device_type, device_index));
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

// --- Autograd ---

extern "C" char* flodl_set_requires_grad(FlodlTensor t, int requires_grad,
                                          FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).set_requires_grad(requires_grad != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" int flodl_requires_grad(FlodlTensor t) {
    return unwrap(t).requires_grad() ? 1 : 0;
}

extern "C" char* flodl_backward(FlodlTensor t) {
    try {
        unwrap(t).backward();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_grad(FlodlTensor t, FlodlTensor* result) {
    try {
        auto g = unwrap(t).grad();
        if (g.defined()) {
            *result = wrap(g);
        } else {
            *result = nullptr;
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_set_grad(FlodlTensor t, FlodlTensor grad) {
    try {
        unwrap(t).mutable_grad() = unwrap(grad);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_zero_grad(FlodlTensor t) {
    try {
        auto& tensor = unwrap(t);
        if (tensor.grad().defined()) {
            tensor.mutable_grad().zero_();
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" void flodl_zero_grad_set_to_none(FlodlTensor t) {
    auto& tensor = unwrap(t);
    if (tensor.grad().defined()) {
        tensor.mutable_grad().reset();
    }
}

extern "C" char* flodl_detach(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).detach());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_detach_(FlodlTensor t) {
    try {
        unwrap(t).detach_();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" int flodl_is_leaf(FlodlTensor t) {
    return unwrap(t).is_leaf() ? 1 : 0;
}

// --- Autograd context ---

extern "C" void* flodl_no_grad_guard_new() {
    return new torch::NoGradGuard();
}

extern "C" void flodl_no_grad_guard_delete(void* guard) {
    delete static_cast<torch::NoGradGuard*>(guard);
}

extern "C" int flodl_is_grad_enabled() {
    return torch::GradMode::is_enabled() ? 1 : 0;
}

// --- In-place operations ---

extern "C" char* flodl_add_(FlodlTensor t, FlodlTensor other) {
    try {
        unwrap(t).add_(unwrap(other));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_sub_(FlodlTensor t, FlodlTensor other) {
    try {
        unwrap(t).sub_(unwrap(other));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_mul_scalar_(FlodlTensor t, double scalar) {
    try {
        unwrap(t).mul_(scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_add_scalar_(FlodlTensor t, double scalar) {
    try {
        unwrap(t).add_(scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_zero_(FlodlTensor t) {
    try {
        unwrap(t).zero_();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Fused ops ---

extern "C" char* flodl_linear(FlodlTensor input, FlodlTensor weight,
                              FlodlTensor bias, FlodlTensor* result) {
    try {
        c10::optional<torch::Tensor> b;
        if (bias != nullptr) {
            b = unwrap(bias);
        }
        *result = wrap(torch::linear(unwrap(input), unwrap(weight), b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_gru_cell(FlodlTensor input, FlodlTensor hx,
                                FlodlTensor w_ih, FlodlTensor w_hh,
                                FlodlTensor b_ih, FlodlTensor b_hh,
                                FlodlTensor* result) {
    try {
        *result = wrap(torch::gru_cell(
            unwrap(input), unwrap(hx),
            unwrap(w_ih), unwrap(w_hh),
            unwrap(b_ih), unwrap(b_hh)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_lstm_cell(FlodlTensor input, FlodlTensor hx,
                                 FlodlTensor cx,
                                 FlodlTensor w_ih, FlodlTensor w_hh,
                                 FlodlTensor b_ih, FlodlTensor b_hh,
                                 FlodlTensor* h_out, FlodlTensor* c_out) {
    try {
        auto result = torch::lstm_cell(
            unwrap(input), {unwrap(hx), unwrap(cx)},
            unwrap(w_ih), unwrap(w_hh),
            unwrap(b_ih), unwrap(b_hh));
        *h_out = wrap(std::get<0>(result));
        *c_out = wrap(std::get<1>(result));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- cuDNN benchmark ---

extern "C" void flodl_set_cudnn_benchmark(int enable) {
    at::globalContext().setBenchmarkCuDNN(enable != 0);
}

// --- RNG seed ---

extern "C" void flodl_manual_seed(uint64_t seed) {
    torch::manual_seed(static_cast<int64_t>(seed));
}

extern "C" void flodl_cuda_manual_seed_all(uint64_t seed) {
#ifdef FLODL_BUILD_CUDA
    torch::cuda::manual_seed_all(static_cast<int64_t>(seed));
#else
    (void)seed;
#endif
}

// --- Meshgrid ---

extern "C" char* flodl_meshgrid(FlodlTensor* tensors, int count,
                                FlodlTensor** results, int* result_count) {
    try {
        std::vector<torch::Tensor> vec;
        vec.reserve(count);
        for (int i = 0; i < count; i++) {
            vec.push_back(unwrap(tensors[i]));
        }
        auto grids = torch::meshgrid(vec, "ij");
        int n = (int)grids.size();
        *result_count = n;
        FlodlTensor* arr = (FlodlTensor*)malloc(n * sizeof(FlodlTensor));
        for (int i = 0; i < n; i++) {
            arr[i] = new torch::Tensor(grids[i]);
        }
        *results = arr;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Pairwise distance ---

extern "C" char* flodl_cdist(FlodlTensor x, FlodlTensor y, double p,
                             FlodlTensor* result) {
    try {
        auto out = torch::cdist(unwrap(x), unwrap(y), p);
        *result = new torch::Tensor(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Fused Adam step ---

extern "C" char* flodl_adam_step(FlodlTensor param, FlodlTensor grad,
                                 FlodlTensor m, FlodlTensor v,
                                 double lr, double beta1, double beta2, double eps,
                                 double weight_decay, int64_t step) {
    try {
        // Get autograd-free view of param data
        auto p = unwrap(param).data();
        auto& g = unwrap(grad);
        auto& m_ref = unwrap(m);
        auto& v_ref = unwrap(v);

        // Decoupled weight decay (AdamW): p *= (1 - lr * wd)
        if (weight_decay > 0.0) {
            p.mul_(1.0 - lr * weight_decay);
        }

        // First moment: m = beta1 * m + (1 - beta1) * g  [1 fused kernel]
        m_ref.mul_(beta1).add_(g, 1.0 - beta1);

        // Second moment: v = beta2 * v + (1 - beta2) * g^2  [1 fused kernel]
        v_ref.mul_(beta2).addcmul_(g, g, 1.0 - beta2);

        // Bias correction
        double bc1 = 1.0 - std::pow(beta1, (double)step);
        double bc2 = 1.0 - std::pow(beta2, (double)step);
        double step_size = lr / bc1;

        // denom = sqrt(v / bc2) + eps
        auto denom = (v_ref / bc2).sqrt_().add_(eps);

        // p -= step_size * m / denom  [1 fused kernel]
        p.addcdiv_(m_ref, denom, -step_size);

        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Batched Adam step ---

extern "C" char* flodl_adam_step_batched(
        FlodlTensor* params, FlodlTensor* grads,
        FlodlTensor* ms, FlodlTensor* vs,
        double* lrs, int count,
        double beta1, double beta2, double eps,
        double weight_decay, int64_t step) {
    try {
        double bc1 = 1.0 - std::pow(beta1, (double)step);
        double bc2 = 1.0 - std::pow(beta2, (double)step);

        for (int i = 0; i < count; i++) {
            auto p = unwrap(params[i]).data();
            auto& g = unwrap(grads[i]);
            auto& m = unwrap(ms[i]);
            auto& v = unwrap(vs[i]);
            double lr = lrs[i];

            if (weight_decay > 0.0) {
                p.mul_(1.0 - lr * weight_decay);
            }
            m.mul_(beta1).add_(g, 1.0 - beta1);
            v.mul_(beta2).addcmul_(g, g, 1.0 - beta2);

            double step_size = lr / bc1;
            auto denom = (v / bc2).sqrt_().add_(eps);
            p.addcdiv_(m, denom, -step_size);
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Pinned memory ---

extern "C" char* flodl_pin_memory(FlodlTensor t, FlodlTensor* result) {
    try {
        *result = wrap(unwrap(t).pin_memory());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" int flodl_is_pinned(FlodlTensor t) {
    return unwrap(t).is_pinned() ? 1 : 0;
}

// --- Memory diagnostics ---

extern "C" int flodl_malloc_trim() {
#ifdef __linux__
    return malloc_trim(0);
#else
    return 0;
#endif
}

// --- Fused clip_grad_norm ---

extern "C" char* flodl_clip_grad_norm(FlodlTensor* params, int count,
                                       double max_norm, double* total_norm_out) {
    try {
        // Pass 1: compute global L2 norm across all param grads
        torch::Tensor total_norm_sq;
        bool has_grads = false;
        for (int i = 0; i < count; i++) {
            auto& p = unwrap(params[i]);
            auto g = p.grad();
            if (g.defined()) {
                auto n2 = g.norm().pow(2);
                if (!has_grads) {
                    total_norm_sq = n2;
                    has_grads = true;
                } else {
                    total_norm_sq.add_(n2);
                }
            }
        }
        if (!has_grads) {
            *total_norm_out = 0.0;
            return nullptr;
        }
        double total = total_norm_sq.sqrt().item<double>();
        *total_norm_out = total;

        // Pass 2: scale grads if needed
        if (total > max_norm) {
            double scale = max_norm / (total + 1e-6);
            for (int i = 0; i < count; i++) {
                auto& p = unwrap(params[i]);
                auto g = p.grad();
                if (g.defined()) {
                    p.mutable_grad().mul_(scale);
                }
            }
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Autograd diagnostics ---

extern "C" int64_t flodl_autograd_node_count(FlodlTensor t) {
    auto& tensor = unwrap(t);
    auto fn = tensor.grad_fn();
    if (!fn) return 0;

    std::unordered_set<torch::autograd::Node*> visited;
    std::queue<torch::autograd::Node*> q;
    q.push(fn.get());
    visited.insert(fn.get());

    while (!q.empty()) {
        auto* node = q.front();
        q.pop();
        for (auto& edge : node->next_edges()) {
            auto* next = edge.function.get();
            if (next && visited.insert(next).second) {
                q.push(next);
            }
        }
    }
    return static_cast<int64_t>(visited.size());
}

// --- Fused loss functions ---

extern "C" char* flodl_mse_loss(FlodlTensor pred, FlodlTensor target,
                                 int64_t reduction, FlodlTensor* result) {
    try {
        *result = wrap(torch::mse_loss(unwrap(pred), unwrap(target), reduction));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_cross_entropy_loss(FlodlTensor pred, FlodlTensor target,
                                           int64_t reduction, int64_t ignore_index,
                                           double label_smoothing, FlodlTensor* result) {
    try {
        *result = wrap(at::cross_entropy_loss(
            unwrap(pred), unwrap(target),
            /*weight=*/{}, reduction, ignore_index, label_smoothing));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_bce_with_logits_loss(FlodlTensor pred, FlodlTensor target,
                                              int64_t reduction, FlodlTensor* result) {
    try {
        *result = wrap(torch::binary_cross_entropy_with_logits(
            unwrap(pred), unwrap(target),
            /*weight=*/{}, /*pos_weight=*/{}, reduction));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_l1_loss(FlodlTensor pred, FlodlTensor target,
                                int64_t reduction, FlodlTensor* result) {
    try {
        *result = wrap(torch::l1_loss(unwrap(pred), unwrap(target), reduction));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_smooth_l1_loss(FlodlTensor pred, FlodlTensor target,
                                       int64_t reduction, double beta,
                                       FlodlTensor* result) {
    try {
        *result = wrap(torch::smooth_l1_loss(unwrap(pred), unwrap(target),
                                              reduction, beta));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_kl_div_loss(FlodlTensor input, FlodlTensor target,
                                    int64_t reduction, int log_target,
                                    FlodlTensor* result) {
    try {
        *result = wrap(torch::kl_div(unwrap(input), unwrap(target),
                                      reduction, log_target != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Fused batch normalization ---

extern "C" char* flodl_batch_norm(FlodlTensor input, FlodlTensor weight,
                                   FlodlTensor bias, FlodlTensor running_mean,
                                   FlodlTensor running_var, int training,
                                   double momentum, double eps,
                                   FlodlTensor* result) {
    try {
        c10::optional<torch::Tensor> w = weight ? c10::make_optional(unwrap(weight))
                                                : c10::nullopt;
        c10::optional<torch::Tensor> b = bias ? c10::make_optional(unwrap(bias))
                                              : c10::nullopt;
        c10::optional<torch::Tensor> rm = running_mean
            ? c10::make_optional(unwrap(running_mean)) : c10::nullopt;
        c10::optional<torch::Tensor> rv = running_var
            ? c10::make_optional(unwrap(running_var)) : c10::nullopt;

        *result = wrap(torch::batch_norm(unwrap(input), w, b, rm, rv,
                       training != 0, momentum, eps, /*cudnn_enabled=*/true));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Fused dropout ---

extern "C" char* flodl_dropout(FlodlTensor input, double p, int training,
                                FlodlTensor* result) {
    try {
        *result = wrap(torch::dropout(unwrap(input), p, training != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* flodl_feature_dropout(FlodlTensor input, double p, int training,
                                        FlodlTensor* result) {
    try {
        *result = wrap(torch::feature_dropout(unwrap(input), p, training != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Utility ---

extern "C" void flodl_free_string(char* s) {
    free(s);
}
