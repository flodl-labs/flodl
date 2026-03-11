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
        case RDL_FLOAT16:  return torch::kFloat16;
        case RDL_BFLOAT16: return torch::kBFloat16;
        case RDL_FLOAT32:  return torch::kFloat32;
        case RDL_FLOAT64:  return torch::kFloat64;
        case RDL_INT32:    return torch::kInt32;
        case RDL_INT64:    return torch::kInt64;
        default:           return torch::kFloat32;
    }
}

// Helper: convert our dtype constant back from torch::ScalarType.
static int from_scalar_type(torch::ScalarType st) {
    switch (st) {
        case torch::kFloat16:  return RDL_FLOAT16;
        case torch::kBFloat16: return RDL_BFLOAT16;
        case torch::kFloat32:  return RDL_FLOAT32;
        case torch::kFloat64:  return RDL_FLOAT64;
        case torch::kInt32:    return RDL_INT32;
        case torch::kInt64:    return RDL_INT64;
        default:               return RDL_FLOAT32;
    }
}

// Helper: convert our device constant to torch::Device.
static torch::Device to_device(int device) {
    if (device == RDL_CUDA) {
        return torch::Device(torch::kCUDA, 0);
    }
    return torch::Device(torch::kCPU);
}

// Helper: convert torch::Device back to our constant.
static int from_device(const torch::Device& dev) {
    if (dev.is_cuda()) return RDL_CUDA;
    return RDL_CPU;
}

// Helper: wrap a new torch::Tensor into a heap-allocated pointer.
// The caller (Rust) owns this pointer and must call rdl_free_tensor.
static RdlTensor wrap(torch::Tensor t) {
    auto* p = new torch::Tensor(std::move(t));
    return (RdlTensor)p;
}

// Helper: unwrap an RdlTensor handle back to a reference.
static torch::Tensor& unwrap(RdlTensor t) {
    return *((torch::Tensor*)t);
}

// Helper: build IntArrayRef from C array.
static torch::IntArrayRef make_shape(int64_t* shape, int ndim) {
    return torch::IntArrayRef(shape, ndim);
}

// --- Tensor creation ---

extern "C" char* rdl_zeros(int64_t* shape, int ndim, int dtype, int device,
                            RdlTensor* result) {
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

extern "C" char* rdl_ones(int64_t* shape, int ndim, int dtype, int device,
                           RdlTensor* result) {
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

extern "C" char* rdl_rand(int64_t* shape, int ndim, int dtype, int device,
                           RdlTensor* result) {
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

extern "C" char* rdl_randn(int64_t* shape, int ndim, int dtype, int device,
                             RdlTensor* result) {
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

extern "C" char* rdl_from_blob(void* data, int64_t* shape, int ndim,
                                int dtype, int device, RdlTensor* result) {
    try {
        auto options = torch::TensorOptions().dtype(to_scalar_type(dtype));
        // from_blob does not take ownership — clone to get an independent copy.
        auto t = torch::from_blob(data, make_shape(shape, ndim), options).clone();
        if (device == RDL_CUDA) {
            t = t.to(torch::kCUDA);
        }
        *result = wrap(std::move(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_linspace(double start, double end, int64_t steps,
                               int dtype, int device, RdlTensor* result) {
    try {
        auto options = torch::TensorOptions().dtype(to_scalar_type(dtype));
        auto t = torch::linspace(start, end, steps, options);
        if (device == RDL_CUDA) {
            t = t.to(torch::kCUDA);
        }
        *result = wrap(std::move(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_arange(double start, double end, double step,
                              int dtype, int device, RdlTensor* result) {
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

extern "C" char* rdl_expand(RdlTensor t, int64_t* new_shape, int ndim,
                             RdlTensor* result) {
    try {
        // expand returns a view; contiguous() makes an owned copy.
        *result = wrap(unwrap(t).expand(make_shape(new_shape, ndim)).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Tensor lifecycle ---

extern "C" void rdl_free_tensor(RdlTensor t) {
    if (t) {
        delete (torch::Tensor*)t;
    }
}

extern "C" char* rdl_shallow_clone(RdlTensor t, RdlTensor* result) {
    try {
        auto* src = reinterpret_cast<torch::Tensor*>(t);
        *result = new torch::Tensor(*src);  // Copy ctor: shares TensorImpl, bumps refcount.
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Tensor metadata ---

extern "C" int rdl_ndim(RdlTensor t) {
    return (int)unwrap(t).dim();
}

extern "C" int64_t rdl_shape(RdlTensor t, int dim) {
    return unwrap(t).size(dim);
}

extern "C" int rdl_dtype(RdlTensor t) {
    return from_scalar_type(unwrap(t).scalar_type());
}

extern "C" int rdl_device(RdlTensor t) {
    return from_device(unwrap(t).device());
}

extern "C" int64_t rdl_numel(RdlTensor t) {
    return unwrap(t).numel();
}

// --- Data access ---

extern "C" char* rdl_copy_data(RdlTensor t, void* buffer,
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

extern "C" char* rdl_add(RdlTensor a, RdlTensor b, RdlTensor* result) {
    try {
        *result = wrap(unwrap(a) + unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_sub(RdlTensor a, RdlTensor b, RdlTensor* result) {
    try {
        *result = wrap(unwrap(a) - unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_mul(RdlTensor a, RdlTensor b, RdlTensor* result) {
    try {
        *result = wrap(unwrap(a) * unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_div(RdlTensor a, RdlTensor b, RdlTensor* result) {
    try {
        *result = wrap(unwrap(a) / unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_matmul(RdlTensor a, RdlTensor b, RdlTensor* result) {
    try {
        *result = wrap(torch::matmul(unwrap(a), unwrap(b)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_add_scalar(RdlTensor t, double scalar,
                                 RdlTensor* result) {
    try {
        *result = wrap(unwrap(t) + scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_mul_scalar(RdlTensor t, double scalar,
                                 RdlTensor* result) {
    try {
        *result = wrap(unwrap(t) * scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_neg(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(-unwrap(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Activations ---

extern "C" char* rdl_relu(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(torch::relu(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_sigmoid(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(torch::sigmoid(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_tanh_op(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(torch::tanh(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_softmax(RdlTensor t, int dim, RdlTensor* result) {
    try {
        *result = wrap(torch::softmax(unwrap(t), dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_log_softmax(RdlTensor t, int dim, RdlTensor* result) {
    try {
        *result = wrap(torch::log_softmax(unwrap(t), dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_gelu(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(torch::gelu(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_silu(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(torch::silu(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Layer normalization ---

extern "C" char* rdl_native_layer_norm(RdlTensor input, RdlTensor weight,
                                         RdlTensor bias, int64_t normalized_size,
                                         double eps,
                                         RdlTensor* output, RdlTensor* mean,
                                         RdlTensor* rstd) {
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

extern "C" char* rdl_native_layer_norm_backward(RdlTensor grad_output,
                                                  RdlTensor input,
                                                  RdlTensor mean, RdlTensor rstd,
                                                  RdlTensor weight, RdlTensor bias,
                                                  int64_t normalized_size,
                                                  RdlTensor* grad_input,
                                                  RdlTensor* grad_weight,
                                                  RdlTensor* grad_bias) {
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

extern "C" char* rdl_exp(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(torch::exp(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_log(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(torch::log(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_sqrt(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(torch::sqrt(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_abs(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).abs());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_pow_scalar(RdlTensor t, double exponent,
                                 RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).pow(exponent));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_clamp(RdlTensor t, double min_val, double max_val,
                             RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).clamp(min_val, max_val));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Reductions ---

extern "C" char* rdl_sum(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).sum());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_sum_dim(RdlTensor t, int dim, int keepdim,
                              RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).sum(dim, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_mean_dim(RdlTensor t, int dim, int keepdim,
                               RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).mean(dim, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_min(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).min());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_min_dim(RdlTensor t, int dim, int keepdim,
                              RdlTensor* result) {
    try {
        auto [values, indices] = unwrap(t).min(dim, (bool)keepdim);
        *result = wrap(values);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_max_dim(RdlTensor t, int dim, int keepdim,
                              RdlTensor* result) {
    try {
        // std::get<0> gets the values (not the indices)
        *result = wrap(std::get<0>(unwrap(t).max(dim, keepdim != 0)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_argmax(RdlTensor t, int dim, int keepdim,
                              RdlTensor* result) {
    try {
        *result = wrap(torch::argmax(unwrap(t), dim, (bool)keepdim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Comparison (return float masks: 0.0 or 1.0) ---

extern "C" char* rdl_gt_scalar(RdlTensor t, double scalar,
                                RdlTensor* result) {
    try {
        auto mask = torch::gt(unwrap(t), scalar);
        *result = wrap(mask.to(unwrap(t).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_ge_scalar(RdlTensor t, double scalar,
                                RdlTensor* result) {
    try {
        auto mask = torch::ge(unwrap(t), scalar);
        *result = wrap(mask.to(unwrap(t).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_le_scalar(RdlTensor t, double scalar,
                                RdlTensor* result) {
    try {
        auto mask = torch::le(unwrap(t), scalar);
        *result = wrap(mask.to(unwrap(t).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_lt_scalar(RdlTensor t, double scalar,
                                RdlTensor* result) {
    try {
        auto mask = torch::lt(unwrap(t), scalar);
        *result = wrap(mask.to(unwrap(t).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Shape operations ---

extern "C" char* rdl_reshape(RdlTensor t, int64_t* shape, int ndim,
                              RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).reshape(make_shape(shape, ndim)).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_transpose(RdlTensor t, int dim0, int dim1,
                                RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).transpose(dim0, dim1).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_permute(RdlTensor t, int64_t* dims, int ndim,
                              RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).permute(torch::IntArrayRef(dims, ndim)).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_select(RdlTensor t, int dim, int64_t index,
                             RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).select(dim, index).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_narrow(RdlTensor t, int dim, int64_t start,
                             int64_t length, RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).narrow(dim, start, length).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_squeeze(RdlTensor t, int dim, RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).squeeze(dim).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_unsqueeze(RdlTensor t, int dim, RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).unsqueeze(dim).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Scatter ---

extern "C" char* rdl_select_scatter(RdlTensor input, RdlTensor src,
                                     int dim, int64_t index,
                                     RdlTensor* result) {
    try {
        auto out = unwrap(input).clone();
        out.select(dim, index).copy_(unwrap(src));
        *result = wrap(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_narrow_scatter(RdlTensor input, RdlTensor src,
                                     int dim, int64_t start,
                                     RdlTensor* result) {
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

extern "C" char* rdl_index_select(RdlTensor t, int dim, RdlTensor index,
                                   RdlTensor* result) {
    try {
        *result = wrap(torch::index_select(unwrap(t), dim, unwrap(index)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_index_add(RdlTensor t, int dim, RdlTensor index,
                                RdlTensor src, RdlTensor* result) {
    try {
        // Out-of-place: returns t with src scattered at index positions.
        *result = wrap(unwrap(t).index_add(dim, unwrap(index), unwrap(src)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Concatenation ---

extern "C" char* rdl_cat2(RdlTensor a, RdlTensor b, int dim,
                           RdlTensor* result) {
    try {
        *result = wrap(torch::cat({unwrap(a), unwrap(b)}, dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Conditional ---

extern "C" char* rdl_where(RdlTensor condition, RdlTensor x,
                             RdlTensor y, RdlTensor* result) {
    try {
        auto cond = unwrap(condition).to(torch::kBool);
        *result = wrap(torch::where(cond, unwrap(x), unwrap(y)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Like constructors ---

extern "C" char* rdl_zeros_like(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(torch::zeros_like(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_ones_like(RdlTensor t, RdlTensor* result) {
    try {
        *result = wrap(torch::ones_like(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Convolution ---

extern "C" char* rdl_conv2d(RdlTensor input, RdlTensor weight,
                             RdlTensor bias,
                             int64_t* stride, int64_t* padding,
                             int64_t* dilation,
                             int64_t groups, RdlTensor* result) {
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

extern "C" char* rdl_conv2d_backward(RdlTensor grad_output, RdlTensor input,
                                      RdlTensor weight,
                                      int64_t* stride, int64_t* padding,
                                      int64_t* dilation,
                                      int64_t groups, int compute_bias,
                                      RdlTensor* grad_input,
                                      RdlTensor* grad_weight,
                                      RdlTensor* grad_bias) {
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

extern "C" char* rdl_conv_transpose2d(RdlTensor input, RdlTensor weight,
                                       RdlTensor bias,
                                       int64_t* stride, int64_t* padding,
                                       int64_t* output_padding, int64_t* dilation,
                                       int64_t groups, RdlTensor* result) {
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

extern "C" char* rdl_conv_transpose2d_backward(RdlTensor grad_output,
                                                 RdlTensor input,
                                                 RdlTensor weight,
                                                 int64_t* stride, int64_t* padding,
                                                 int64_t* output_padding,
                                                 int64_t* dilation,
                                                 int64_t groups, int compute_bias,
                                                 RdlTensor* grad_input,
                                                 RdlTensor* grad_weight,
                                                 RdlTensor* grad_bias) {
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

extern "C" char* rdl_adaptive_avg_pool2d(RdlTensor input, int64_t* output_size,
                                          RdlTensor* result) {
    try {
        *result = wrap(at::adaptive_avg_pool2d(
            unwrap(input), torch::IntArrayRef(output_size, 2)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_adaptive_avg_pool2d_backward(RdlTensor grad_output,
                                                    RdlTensor input,
                                                    RdlTensor* grad_input) {
    try {
        *grad_input = wrap(at::_adaptive_avg_pool2d_backward(
            unwrap(grad_output), unwrap(input)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Grid sampling ---

extern "C" char* rdl_grid_sample(RdlTensor input, RdlTensor grid,
                                  int mode, int padding_mode,
                                  int align_corners, RdlTensor* result) {
    try {
        *result = wrap(at::grid_sampler(
            unwrap(input), unwrap(grid), mode, padding_mode, align_corners != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_grid_sample_backward(RdlTensor grad_output,
                                           RdlTensor input, RdlTensor grid,
                                           int mode, int padding_mode,
                                           int align_corners,
                                           RdlTensor* grad_input,
                                           RdlTensor* grad_grid) {
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

extern "C" char* rdl_to_dtype(RdlTensor t, int dtype, RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).to(to_scalar_type(dtype)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* rdl_all_finite(RdlTensor t, int* result) {
    try {
        auto& tensor = unwrap(t);
        *result = torch::isfinite(tensor).all().item<bool>() ? 1 : 0;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Device operations ---

extern "C" char* rdl_to_device(RdlTensor t, int device,
                                RdlTensor* result) {
    try {
        *result = wrap(unwrap(t).to(to_device(device)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" int rdl_cuda_is_available(void) {
    return torch::cuda::is_available() ? 1 : 0;
}

extern "C" int rdl_cuda_device_count(void) {
    return (int)torch::cuda::device_count();
}

// --- Utility ---

extern "C" void rdl_free_string(char* s) {
    free(s);
}
