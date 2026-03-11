//! Tensor — immutable, chainable wrapper around a libtorch tensor.
//!
//! Every tensor owns its C++ handle and frees it on drop. This is the
//! entire VRAM management story — no GC, no scopes, no finalizers.
//!
//! Operations are chainable and return `Result<Tensor>`:
//!
//! ```ignore
//! let z = a.add(&b)?.relu()?.sum()?;
//! ```

use std::ffi::{c_void, CStr};
use std::fmt;
use std::ptr;

use rdl_sys::{self as ffi, RdlTensor};

/// DType represents the data type of tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DType {
    Float16 = ffi::RDL_FLOAT16,
    BFloat16 = ffi::RDL_BFLOAT16,
    Float32 = ffi::RDL_FLOAT32,
    Float64 = ffi::RDL_FLOAT64,
    Int32 = ffi::RDL_INT32,
    Int64 = ffi::RDL_INT64,
}

impl DType {
    fn from_raw(v: i32) -> Self {
        match v {
            ffi::RDL_FLOAT16 => DType::Float16,
            ffi::RDL_BFLOAT16 => DType::BFloat16,
            ffi::RDL_FLOAT32 => DType::Float32,
            ffi::RDL_FLOAT64 => DType::Float64,
            ffi::RDL_INT32 => DType::Int32,
            ffi::RDL_INT64 => DType::Int64,
            _ => DType::Float32,
        }
    }

    /// Size of one element in bytes.
    pub fn element_size(self) -> usize {
        match self {
            DType::Float16 | DType::BFloat16 => 2,
            DType::Float32 | DType::Int32 => 4,
            DType::Float64 | DType::Int64 => 8,
        }
    }
}

/// Device represents where a tensor's data lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Device {
    CPU = ffi::RDL_CPU,
    CUDA = ffi::RDL_CUDA,
}

impl Device {
    fn from_raw(v: i32) -> Self {
        match v {
            ffi::RDL_CUDA => Device::CUDA,
            _ => Device::CPU,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::CPU => write!(f, "cpu"),
            Device::CUDA => write!(f, "cuda"),
        }
    }
}

/// Error type for tensor operations.
#[derive(Debug, Clone)]
pub struct TensorError(String);

impl TensorError {
    pub fn new(msg: &str) -> Self {
        TensorError(msg.to_string())
    }
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for TensorError {}

pub type Result<T> = std::result::Result<T, TensorError>;

/// Convert a C error string to Result. Frees the C string.
fn check_err(err: *mut i8) -> Result<()> {
    if err.is_null() {
        Ok(())
    } else {
        let msg = unsafe { CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned();
        unsafe { ffi::rdl_free_string(err) };
        Err(TensorError(msg))
    }
}

/// Options for tensor creation.
#[derive(Debug, Clone, Copy)]
pub struct TensorOptions {
    pub dtype: DType,
    pub device: Device,
}

impl Default for TensorOptions {
    fn default() -> Self {
        Self {
            dtype: DType::Float32,
            device: Device::CPU,
        }
    }
}

/// A tensor wrapping a libtorch C++ tensor.
///
/// Owns the underlying C++ handle. When dropped, the C++ tensor is
/// freed immediately — including any GPU memory. This is the entire
/// VRAM management story.
pub struct Tensor {
    handle: RdlTensor,
}

// Safety: libtorch tensors are reference-counted internally and
// thread-safe for read access. Mutations go through the shim which
// creates new tensors.
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::rdl_free_tensor(self.handle) };
        }
    }
}

impl Clone for Tensor {
    /// Shallow clone: creates a new C++ Tensor handle sharing the same
    /// TensorImpl (and thus the same data storage). Cheap — just bumps
    /// libtorch's internal refcount.
    fn clone(&self) -> Self {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_shallow_clone(self.handle, &mut handle) };
        if !err.is_null() {
            let msg = unsafe { CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned();
            unsafe { ffi::rdl_free_string(err) };
            panic!("tensor clone failed: {}", msg);
        }
        Self::from_raw(handle)
    }
}

impl Tensor {
    /// Wrap a raw handle. The Tensor takes ownership.
    fn from_raw(handle: RdlTensor) -> Self {
        debug_assert!(!handle.is_null());
        Self { handle }
    }

    /// Access the raw handle (for passing to FFI in sibling modules).
    #[allow(dead_code)]
    pub(crate) fn raw(&self) -> RdlTensor {
        self.handle
    }

    // --- Creation ---

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_zeros(
                shape.as_mut_ptr(),
                shape.len() as i32,
                opts.dtype as i32,
                opts.device as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_ones(
                shape.as_mut_ptr(),
                shape.len() as i32,
                opts.dtype as i32,
                opts.device as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a tensor from f32 data.
    pub fn from_f32(data: &[f32], shape: &[i64], device: Device) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_from_blob(
                data.as_ptr() as *mut c_void,
                shape.as_mut_ptr(),
                shape.len() as i32,
                DType::Float32 as i32,
                device as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a tensor from i64 data (for indices).
    pub fn from_i64(data: &[i64], shape: &[i64]) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_from_blob(
                data.as_ptr() as *mut c_void,
                shape.as_mut_ptr(),
                shape.len() as i32,
                DType::Int64 as i32,
                Device::CPU as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    // --- Metadata ---

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        unsafe { ffi::rdl_ndim(self.handle) as usize }
    }

    /// Shape as a Vec.
    pub fn shape(&self) -> Vec<i64> {
        let n = self.ndim();
        (0..n)
            .map(|i| unsafe { ffi::rdl_shape(self.handle, i as i32) })
            .collect()
    }

    /// Total number of elements.
    pub fn numel(&self) -> i64 {
        unsafe { ffi::rdl_numel(self.handle) }
    }

    /// Data type.
    pub fn dtype(&self) -> DType {
        DType::from_raw(unsafe { ffi::rdl_dtype(self.handle) })
    }

    /// Device (CPU or CUDA).
    pub fn device(&self) -> Device {
        Device::from_raw(unsafe { ffi::rdl_device(self.handle) })
    }

    // --- Data access ---

    /// Copy tensor data to a Vec<f32>. Moves to CPU if needed.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        let n = self.numel() as usize;
        let mut buf = vec![0f32; n];
        let bytes = (n * 4) as i64;
        let err = unsafe {
            ffi::rdl_copy_data(self.handle, buf.as_mut_ptr() as *mut c_void, bytes)
        };
        check_err(err)?;
        Ok(buf)
    }

    /// Extract a scalar value as f64 (for loss values, metrics, etc.).
    pub fn item(&self) -> Result<f64> {
        let data = self.to_f32_vec()?;
        Ok(data.first().copied().unwrap_or(0.0) as f64)
    }

    // --- Arithmetic (chainable) ---

    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_add(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_sub(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_mul(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_matmul(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn mul_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_mul_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Activations ---

    pub fn relu(&self) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_relu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn sigmoid(&self) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_sigmoid(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Reductions ---

    pub fn sum(&self) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_sum(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Additional arithmetic ---

    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_div(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn neg(&self) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_neg(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn add_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_add_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Activations ---

    pub fn tanh_op(&self) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_tanh_op(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Element-wise math ---

    pub fn exp(&self) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_exp(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn log(&self) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_log(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn sqrt(&self) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_sqrt(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn abs(&self) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_abs(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn pow_scalar(&self, exponent: f64) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_pow_scalar(self.handle, exponent, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Reductions ---

    pub fn sum_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_sum_dim(self.handle, dim, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Comparisons ---

    pub fn gt_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_gt_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Shape operations ---

    pub fn reshape(&self, shape: &[i64]) -> Result<Tensor> {
        let mut shape = shape.to_vec();
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_reshape(self.handle, shape.as_mut_ptr(), shape.len() as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn transpose(&self, dim0: i32, dim1: i32) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_transpose(self.handle, dim0, dim1, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn expand(&self, shape: &[i64]) -> Result<Tensor> {
        let mut shape = shape.to_vec();
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_expand(self.handle, shape.as_mut_ptr(), shape.len() as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Slicing and indexing ---

    /// Narrow (slice) along a dimension: returns a view.
    pub fn narrow(&self, dim: i32, start: i64, length: i64) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_narrow(self.handle, dim, start, length, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter a narrow slice back into a tensor (for narrow backward).
    pub fn narrow_scatter(&self, src: &Tensor, dim: i32, start: i64) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_narrow_scatter(self.handle, src.handle, dim, start, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Concatenate two tensors along a dimension.
    pub fn cat(&self, other: &Tensor, dim: i32) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_cat2(self.handle, other.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Softmax along a dimension.
    pub fn softmax(&self, dim: i32) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_softmax(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Log-softmax along a dimension (numerically stable).
    pub fn log_softmax(&self, dim: i32) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_log_softmax(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// GELU activation (native libtorch).
    pub fn gelu(&self) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_gelu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// SiLU activation (native libtorch).
    pub fn silu(&self) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_silu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Native layer normalization. Returns (output, mean, rstd).
    pub fn native_layer_norm(
        &self, weight: &Tensor, bias: &Tensor, normalized_size: i64, eps: f64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mut out: RdlTensor = ptr::null_mut();
        let mut mean: RdlTensor = ptr::null_mut();
        let mut rstd: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_native_layer_norm(
                self.handle, weight.handle, bias.handle,
                normalized_size, eps,
                &mut out, &mut mean, &mut rstd,
            )
        };
        check_err(err)?;
        Ok((Tensor::from_raw(out), Tensor::from_raw(mean), Tensor::from_raw(rstd)))
    }

    /// Native layer normalization backward. Returns (grad_input, grad_weight, grad_bias).
    pub fn native_layer_norm_backward(
        grad_output: &Tensor, input: &Tensor, mean: &Tensor, rstd: &Tensor,
        weight: &Tensor, bias: &Tensor, normalized_size: i64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mut gi: RdlTensor = ptr::null_mut();
        let mut gw: RdlTensor = ptr::null_mut();
        let mut gb: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_native_layer_norm_backward(
                grad_output.handle, input.handle,
                mean.handle, rstd.handle,
                weight.handle, bias.handle, normalized_size,
                &mut gi, &mut gw, &mut gb,
            )
        };
        check_err(err)?;
        Ok((Tensor::from_raw(gi), Tensor::from_raw(gw), Tensor::from_raw(gb)))
    }

    /// Select a single index along a dimension (reduces that dim).
    pub fn select(&self, dim: i32, index: i64) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_select(self.handle, dim, index, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Mean along a dimension.
    pub fn mean_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_mean_dim(self.handle, dim, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Select rows/elements along a dimension using an index tensor.
    pub fn index_select(&self, dim: i32, index: &Tensor) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_index_select(self.handle, dim, index.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter-add src into self along dim at positions given by index.
    pub fn index_add(&self, dim: i32, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_index_add(self.handle, dim, index.handle, src.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Like constructors ---

    pub fn zeros_like(t: &Tensor) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_zeros_like(t.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    pub fn ones_like(t: &Tensor) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_ones_like(t.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Random ---

    pub fn rand(shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_rand(
                shape.as_mut_ptr(), shape.len() as i32,
                opts.dtype as i32, opts.device as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    pub fn randn(shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::rdl_randn(
                shape.as_mut_ptr(), shape.len() as i32,
                opts.dtype as i32, opts.device as i32,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    // --- Device ---

    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        let mut handle: RdlTensor = ptr::null_mut();
        let err = unsafe { ffi::rdl_to_device(self.handle, device as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor({:?}, {:?}, {:?})",
            self.shape(),
            self.dtype(),
            self.device()
        )
    }
}

/// Returns true if CUDA is available.
pub fn cuda_available() -> bool {
    unsafe { ffi::rdl_cuda_is_available() != 0 }
}

/// Returns the number of CUDA devices.
pub fn cuda_device_count() -> i32 {
    unsafe { ffi::rdl_cuda_device_count() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[2, 3], TensorOptions::default()).unwrap();
        assert_eq!(t.shape(), vec![2, 3]);
        assert_eq!(t.dtype(), DType::Float32);
        assert_eq!(t.device(), Device::CPU);
        assert_eq!(t.numel(), 6);

        let data = t.to_f32_vec().unwrap();
        assert_eq!(data, vec![0.0; 6]);
    }

    #[test]
    fn test_from_f32() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], Device::CPU).unwrap();
        assert_eq!(t.shape(), vec![3]);
        let data = t.to_f32_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_add() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], Device::CPU).unwrap();
        let b = Tensor::from_f32(&[4.0, 5.0, 6.0], &[3], Device::CPU).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_f32_vec().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU).unwrap();
        let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2], Device::CPU).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.to_f32_vec().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_chaining() {
        let a = Tensor::from_f32(&[1.0, -2.0, 3.0], &[3], Device::CPU).unwrap();
        let b = Tensor::from_f32(&[1.0, 1.0, 1.0], &[3], Device::CPU).unwrap();
        let result = a.add(&b).unwrap().relu().unwrap().sum().unwrap();
        // [1+1, -2+1, 3+1] = [2, -1, 4] -> relu -> [2, 0, 4] -> sum -> 6
        let val = result.item().unwrap();
        assert!((val - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_drop_frees_memory() {
        // Create and immediately drop — verifies Drop doesn't crash.
        let _ = Tensor::zeros(&[1000, 1000], TensorOptions::default()).unwrap();
        // If Drop is broken, this would leak or crash.
    }

    #[test]
    fn test_debug_format() {
        let t = Tensor::zeros(&[2, 3], TensorOptions::default()).unwrap();
        let s = format!("{:?}", t);
        assert!(s.contains("[2, 3]"));
        assert!(s.contains("Float32"));
    }
}
