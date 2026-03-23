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
use std::sync::atomic::{AtomicU64, Ordering};

use flodl_sys::{self as ffi, FlodlTensor};

/// Global counter of live C++ Tensor handles. Incremented on creation,
/// decremented on Drop. If this grows over time during training, there
/// is a Tensor handle leak. If it stays stable but RSS grows, the leak
/// is inside libtorch internals (not a handle leak).
static LIVE_TENSOR_COUNT: AtomicU64 = AtomicU64::new(0);

/// Element data type of a tensor. Maps to PyTorch's `torch.dtype`.
///
/// Float32 is the default. Use Float16/BFloat16 for mixed precision,
/// Int64 for indices and labels, Float64 when extra precision is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DType {
    Float16 = ffi::FLODL_FLOAT16,
    BFloat16 = ffi::FLODL_BFLOAT16,
    Float32 = ffi::FLODL_FLOAT32,
    Float64 = ffi::FLODL_FLOAT64,
    Int32 = ffi::FLODL_INT32,
    Int64 = ffi::FLODL_INT64,
}

impl DType {
    fn from_raw(v: i32) -> Self {
        match v {
            ffi::FLODL_FLOAT16 => DType::Float16,
            ffi::FLODL_BFLOAT16 => DType::BFloat16,
            ffi::FLODL_FLOAT32 => DType::Float32,
            ffi::FLODL_FLOAT64 => DType::Float64,
            ffi::FLODL_INT32 => DType::Int32,
            ffi::FLODL_INT64 => DType::Int64,
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
///
/// `Device::CPU` is the host. `Device::CUDA(n)` is GPU index `n`.
/// Most single-GPU code uses `Device::CUDA(0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    CUDA(u8),
}

impl Device {
    /// Convert to (device_type, device_index) for FFI calls.
    pub(crate) fn to_ffi(self) -> (i32, i32) {
        match self {
            Device::CPU => (ffi::FLODL_CPU, 0),
            Device::CUDA(idx) => (ffi::FLODL_CUDA, idx as i32),
        }
    }

    /// Reconstruct from FFI (device_type, device_index).
    pub(crate) fn from_ffi(device_type: i32, device_index: i32) -> Self {
        match device_type {
            ffi::FLODL_CUDA => Device::CUDA(device_index as u8),
            _ => Device::CPU,
        }
    }

    /// Whether this is a CUDA device.
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::CUDA(_))
    }

    /// Device index (0 for CPU, GPU index for CUDA).
    pub fn index(&self) -> u8 {
        match self {
            Device::CPU => 0,
            Device::CUDA(idx) => *idx,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::CPU => write!(f, "cpu"),
            Device::CUDA(0) => write!(f, "cuda"),
            Device::CUDA(idx) => write!(f, "cuda:{}", idx),
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
pub(crate) fn check_err(err: *mut i8) -> Result<()> {
    if err.is_null() {
        Ok(())
    } else {
        let msg = unsafe { CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned();
        unsafe { ffi::flodl_free_string(err) };
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
///
/// Operations are chainable and return `Result<Tensor>`:
///
/// ```ignore
/// let y = x.matmul(&w)?.add(&b)?.relu()?;
/// ```
pub struct Tensor {
    handle: FlodlTensor,
}

// Safety: libtorch tensors are reference-counted internally and
// thread-safe for read access. Mutations go through the shim which
// creates new tensors.
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            LIVE_TENSOR_COUNT.fetch_sub(1, Ordering::Relaxed);
            unsafe { ffi::flodl_free_tensor(self.handle) };
        }
    }
}

impl Clone for Tensor {
    /// Shallow clone: creates a new C++ Tensor handle sharing the same
    /// TensorImpl (and thus the same data storage). Cheap — just bumps
    /// libtorch's internal refcount.
    fn clone(&self) -> Self {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_shallow_clone(self.handle, &mut handle) };
        if !err.is_null() {
            let msg = unsafe { CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned();
            unsafe { ffi::flodl_free_string(err) };
            panic!("tensor clone failed: {}", msg);
        }
        Self::from_raw(handle)
    }
}

impl Tensor {
    /// Wrap a raw handle. The Tensor takes ownership.
    fn from_raw(handle: FlodlTensor) -> Self {
        debug_assert!(!handle.is_null());
        LIVE_TENSOR_COUNT.fetch_add(1, Ordering::Relaxed);
        Self { handle }
    }

    /// Wrap a raw handle (crate-visible). The Tensor takes ownership.
    ///
    /// # Safety
    /// Caller must ensure the handle is valid and not owned elsewhere.
    pub(crate) unsafe fn from_raw_handle(handle: FlodlTensor) -> Self {
        Self::from_raw(handle)
    }

    /// Access the raw handle (for passing to FFI in sibling modules).
    pub(crate) fn raw(&self) -> FlodlTensor {
        self.handle
    }

    // --- Creation ---

    /// Create a tensor filled with zeros.
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[2, 3], TensorOptions::default())?;
    /// assert_eq!(t.shape(), vec![2, 3]);
    /// ```
    pub fn zeros(shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_zeros(
                shape.as_mut_ptr(),
                shape.len() as i32,
                opts.dtype as i32,
                dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a tensor filled with ones. Like `torch.ones()`.
    ///
    /// ```ignore
    /// let t = Tensor::ones(&[2, 3], TensorOptions::default())?;
    /// ```
    pub fn ones(shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_ones(
                shape.as_mut_ptr(),
                shape.len() as i32,
                opts.dtype as i32,
                dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a tensor from f32 data.
    ///
    /// ```ignore
    /// let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU)?;
    /// assert_eq!(t.shape(), vec![2, 2]);
    /// ```
    pub fn from_f32(data: &[f32], shape: &[i64], device: Device) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = device.to_ffi();
        let err = unsafe {
            ffi::flodl_from_blob(
                data.as_ptr() as *mut c_void,
                shape.as_mut_ptr(),
                shape.len() as i32,
                DType::Float32 as i32,
                dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a Float64 tensor from f64 data. Use when full double precision
    /// is needed (e.g. loss accumulation, high-precision metrics).
    pub fn from_f64(data: &[f64], shape: &[i64], device: Device) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = device.to_ffi();
        let err = unsafe {
            ffi::flodl_from_blob(
                data.as_ptr() as *mut c_void,
                shape.as_mut_ptr(),
                shape.len() as i32,
                DType::Float64 as i32,
                dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create an Int64 tensor from i64 data. Commonly used for class labels,
    /// token indices, and any integer indexing (e.g. `cross_entropy_loss` targets).
    pub fn from_i64(data: &[i64], shape: &[i64], device: Device) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = device.to_ffi();
        let err = unsafe {
            ffi::flodl_from_blob(
                data.as_ptr() as *mut c_void,
                shape.as_mut_ptr(),
                shape.len() as i32,
                DType::Int64 as i32,
                dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    // --- Metadata ---

    /// Number of dimensions (rank). Like `tensor.ndim` in PyTorch.
    pub fn ndim(&self) -> usize {
        unsafe { ffi::flodl_ndim(self.handle) as usize }
    }

    /// Shape of each dimension as a Vec. Like `tensor.shape` in PyTorch.
    pub fn shape(&self) -> Vec<i64> {
        let n = self.ndim();
        (0..n)
            .map(|i| unsafe { ffi::flodl_shape(self.handle, i as i32) })
            .collect()
    }

    /// Total number of elements (product of all dimensions). Like `tensor.numel()`.
    pub fn numel(&self) -> i64 {
        unsafe { ffi::flodl_numel(self.handle) }
    }

    /// Element data type of this tensor. Like `tensor.dtype` in PyTorch.
    pub fn dtype(&self) -> DType {
        DType::from_raw(unsafe { ffi::flodl_dtype(self.handle) })
    }

    /// Device where this tensor's data resides (CPU or CUDA). Like `tensor.device`.
    pub fn device(&self) -> Device {
        let dt = unsafe { ffi::flodl_device_type(self.handle) };
        let di = unsafe { ffi::flodl_device_index(self.handle) };
        Device::from_ffi(dt, di)
    }

    // --- Data access ---

    /// Copy tensor data to a `Vec<f32>`. Transparently moves to CPU first
    /// if the tensor lives on CUDA. Non-f32 dtypes are cast via libtorch.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        let n = self.numel() as usize;
        let mut buf = vec![0f32; n];
        let bytes = (n * 4) as i64;
        let err = unsafe {
            ffi::flodl_copy_data(self.handle, buf.as_mut_ptr() as *mut c_void, bytes)
        };
        check_err(err)?;
        Ok(buf)
    }

    /// Copy tensor data to a `Vec<f64>`. Moves to CPU if needed.
    /// Float64 tensors are copied at full precision. All other dtypes
    /// go through f32 (lossless for f16/bf16, and the best f32 can offer).
    pub fn to_f64_vec(&self) -> Result<Vec<f64>> {
        if self.dtype() == DType::Float64 {
            let n = self.numel() as usize;
            let mut buf = vec![0.0f64; n];
            let bytes = (n * 8) as i64;
            let err = unsafe {
                ffi::flodl_copy_data(self.handle, buf.as_mut_ptr() as *mut c_void, bytes)
            };
            check_err(err)?;
            Ok(buf)
        } else {
            let f32s = self.to_f32_vec()?;
            Ok(f32s.into_iter().map(|v| v as f64).collect())
        }
    }

    /// Copy tensor data to a `Vec<i64>`. Moves to CPU if needed.
    /// Intended for Int64 tensors (indices, labels).
    pub fn to_i64_vec(&self) -> Result<Vec<i64>> {
        let n = self.numel() as usize;
        let mut buf = vec![0i64; n];
        let bytes = (n * 8) as i64;
        let err = unsafe {
            ffi::flodl_copy_data(self.handle, buf.as_mut_ptr() as *mut c_void, bytes)
        };
        check_err(err)?;
        Ok(buf)
    }

    /// Extract a scalar value as f64. Like PyTorch's `.item()`.
    ///
    /// The tensor must contain exactly one element (any shape is fine,
    /// e.g. `[1]`, `[1, 1]`, or `[]`). Returns an error otherwise.
    /// Preserves full precision for Float64 tensors.
    ///
    /// ```ignore
    /// let loss_val = loss_tensor.item()?;
    /// println!("loss: {:.4}", loss_val);
    /// ```
    pub fn item(&self) -> Result<f64> {
        if self.numel() != 1 {
            return Err(TensorError::new(&format!(
                "item() requires exactly 1 element, got {} (shape {:?})",
                self.numel(), self.shape()
            )));
        }
        if self.dtype() == DType::Float64 {
            let mut buf = [0.0f64; 1];
            let err = unsafe {
                ffi::flodl_copy_data(self.handle, buf.as_mut_ptr() as *mut c_void, 8)
            };
            check_err(err)?;
            Ok(buf[0])
        } else {
            let mut buf = [0.0f32; 1];
            let err = unsafe {
                ffi::flodl_copy_data(self.handle, buf.as_mut_ptr() as *mut c_void, 4)
            };
            check_err(err)?;
            Ok(buf[0] as f64)
        }
    }

    // --- Arithmetic (chainable) ---

    /// Element-wise addition. Shapes must be broadcastable.
    ///
    /// ```ignore
    /// let c = a.add(&b)?; // [2, 3] + [2, 3] → [2, 3]
    /// ```
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_add(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise subtraction. Shapes must be broadcastable.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sub(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise (Hadamard) multiplication. Shapes must be broadcastable.
    /// For matrix multiplication, use [`matmul`](Self::matmul).
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_mul(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Matrix multiplication.
    ///
    /// ```ignore
    /// // [batch, M, K] @ [batch, K, N] → [batch, M, N]
    /// let c = a.matmul(&b)?;
    /// ```
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_matmul(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Multiply every element by a scalar. Like `tensor * 0.5` in PyTorch.
    pub fn mul_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_mul_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Activations ---

    /// ReLU activation: max(0, x).
    pub fn relu(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_relu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Sigmoid activation: 1 / (1 + exp(-x)).
    pub fn sigmoid(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sigmoid(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Reductions ---

    /// Sum of all elements (scalar result).
    pub fn sum(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sum(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Mean of all elements (scalar result).
    pub fn mean(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_mean(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Flatten dimensions `[start_dim..=end_dim]` into one.
    pub fn flatten(&self, start_dim: i32, end_dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_flatten(self.handle, start_dim, end_dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Additional arithmetic ---

    /// Element-wise division.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_div(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Negate every element.
    pub fn neg(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_neg(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Add a scalar to every element.
    pub fn add_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_add_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Divide every element by a scalar.
    pub fn div_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_div_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Activations ---

    /// Tanh activation: element-wise hyperbolic tangent.
    pub fn tanh(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_tanh_op(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Element-wise math ---

    /// Element-wise exponential.
    pub fn exp(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_exp(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_log(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sqrt(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_abs(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Upper triangle of a matrix (or batch of matrices).
    /// Elements below the `diagonal`-th diagonal are zeroed.
    /// `diagonal=0` keeps the main diagonal; `diagonal=1` excludes it.
    pub fn triu(&self, diagonal: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_triu(self.handle, diagonal, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Raise every element to a scalar exponent.
    pub fn pow_scalar(&self, exponent: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_pow_scalar(self.handle, exponent, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Reductions ---

    /// Sum along a dimension.
    pub fn sum_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_sum_dim(self.handle, dim, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Clamp all elements to `[min, max]`.
    pub fn clamp(&self, min: f64, max: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_clamp(self.handle, min, max, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Comparisons ---

    /// Element-wise greater-than comparison against a scalar.
    pub fn gt_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_gt_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Shape operations ---

    /// Reshape to a new shape (must have same total elements).
    /// Use -1 for one inferred dimension.
    ///
    /// ```ignore
    /// let flat = t.reshape(&[-1])?; // [2, 3] → [6]
    /// ```
    pub fn reshape(&self, shape: &[i64]) -> Result<Tensor> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_reshape(self.handle, shape.as_mut_ptr(), shape.len() as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Swap two dimensions.
    ///
    /// ```ignore
    /// let t = x.transpose(0, 1)?; // [M, N] → [N, M]
    /// ```
    pub fn transpose(&self, dim0: i32, dim1: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_transpose(self.handle, dim0, dim1, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Broadcast to a larger shape.
    pub fn expand(&self, shape: &[i64]) -> Result<Tensor> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_expand(self.handle, shape.as_mut_ptr(), shape.len() as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Slicing and indexing ---

    /// Narrow (slice) along a dimension: returns a view.
    pub fn narrow(&self, dim: i32, start: i64, length: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_narrow(self.handle, dim, start, length, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter a narrow slice back into a tensor (for narrow backward).
    pub fn narrow_scatter(&self, src: &Tensor, dim: i32, start: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_narrow_scatter(self.handle, src.handle, dim, start, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Concatenate two tensors along a dimension.
    pub fn cat(&self, other: &Tensor, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_cat2(self.handle, other.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Concatenate multiple tensors along an existing dimension.
    ///
    /// All tensors must have the same shape except in the concatenation dimension.
    /// Uses a single kernel launch regardless of the number of tensors.
    pub fn cat_many(tensors: &[&Tensor], dim: i32) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(TensorError::new("cat_many: empty tensor list"));
        }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let mut result: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_cat(handles.as_mut_ptr(), handles.len() as i32, dim, &mut result)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(result))
    }

    /// Stack tensors along a new dimension.
    ///
    /// All tensors must have the same shape. A new dimension is inserted at `dim`.
    pub fn stack(tensors: &[&Tensor], dim: i32) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(TensorError::new("stack: empty tensor list"));
        }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let mut result: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_stack(handles.as_mut_ptr(), handles.len() as i32, dim, &mut result)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(result))
    }

    /// Softmax along a dimension.
    pub fn softmax(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_softmax(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Log-softmax along a dimension (numerically stable).
    pub fn log_softmax(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_log_softmax(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// GELU activation (native libtorch).
    pub fn gelu(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_gelu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// SiLU activation (native libtorch).
    pub fn silu(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_silu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Native layer normalization. Returns (output, mean, rstd).
    pub fn native_layer_norm(
        &self, weight: &Tensor, bias: &Tensor, normalized_size: i64, eps: f64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mut out: FlodlTensor = ptr::null_mut();
        let mut mean: FlodlTensor = ptr::null_mut();
        let mut rstd: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_native_layer_norm(
                self.handle, weight.handle, bias.handle,
                normalized_size, eps,
                &mut out, &mut mean, &mut rstd,
            )
        };
        check_err(err)?;
        Ok((Tensor::from_raw(out), Tensor::from_raw(mean), Tensor::from_raw(rstd)))
    }

    /// Permute dimensions.
    pub fn permute(&self, dims: &[i64]) -> Result<Tensor> {
        let mut dims = dims.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_permute(self.handle, dims.as_mut_ptr(), dims.len() as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Select a single index along a dimension (reduces that dim).
    pub fn select(&self, dim: i32, index: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_select(self.handle, dim, index, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Mean along a dimension.
    pub fn mean_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_mean_dim(self.handle, dim, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Select rows/elements along a dimension using an index tensor.
    pub fn index_select(&self, dim: i32, index: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_index_select(self.handle, dim, index.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter-add src into self along dim at positions given by index.
    pub fn index_add(&self, dim: i32, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_index_add(self.handle, dim, index.handle, src.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Like constructors ---

    /// Create a tensor of zeros with the same shape, dtype, and device as `t`.
    pub fn zeros_like(t: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_zeros_like(t.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Create a tensor of ones with the same shape, dtype, and device as `t`.
    pub fn ones_like(t: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ones_like(t.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Random ---

    /// Create a tensor with uniform random values in [0, 1).
    pub fn rand(shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_rand(
                shape.as_mut_ptr(), shape.len() as i32,
                opts.dtype as i32, dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a tensor with standard normal random values (mean=0, std=1).
    pub fn randn(shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_randn(
                shape.as_mut_ptr(), shape.len() as i32,
                opts.dtype as i32, dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    // --- Convolution (many args unavoidable — maps 1:1 to libtorch C API) ---

    /// 2D convolution. bias may be a null-handle tensor for no bias.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &self, weight: &Tensor, bias: Option<&Tensor>,
        stride: [i64; 2], padding: [i64; 2], dilation: [i64; 2], groups: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut stride = stride;
        let mut padding = padding;
        let mut dilation = dilation;
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_conv2d(
                self.handle, weight.handle, bias_handle,
                stride.as_mut_ptr(), padding.as_mut_ptr(), dilation.as_mut_ptr(),
                groups, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Transposed 2D convolution.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose2d(
        &self, weight: &Tensor, bias: Option<&Tensor>,
        stride: [i64; 2], padding: [i64; 2], output_padding: [i64; 2],
        dilation: [i64; 2], groups: i64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut stride = stride;
        let mut padding = padding;
        let mut output_padding = output_padding;
        let mut dilation = dilation;
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_conv_transpose2d(
                self.handle, weight.handle, bias_handle,
                stride.as_mut_ptr(), padding.as_mut_ptr(),
                output_padding.as_mut_ptr(), dilation.as_mut_ptr(),
                groups, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Fused ops ---

    /// Fused linear: `y = input @ weight^T + bias` (single ATen kernel).
    pub fn linear(&self, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let bias_handle = bias.map_or(ptr::null_mut(), |b| b.handle);
        let err = unsafe {
            ffi::flodl_linear(self.handle, weight.handle, bias_handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused GRU cell: single ATen `gru_cell` kernel.
    /// Returns new hidden state h'.
    #[allow(clippy::too_many_arguments)]
    pub fn gru_cell(
        &self, hx: &Tensor,
        w_ih: &Tensor, w_hh: &Tensor,
        b_ih: &Tensor, b_hh: &Tensor,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_gru_cell(
                self.handle, hx.handle,
                w_ih.handle, w_hh.handle,
                b_ih.handle, b_hh.handle,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused LSTM cell: single ATen `lstm_cell` kernel.
    /// Returns `(h', c')`.
    #[allow(clippy::too_many_arguments)]
    pub fn lstm_cell(
        &self, hx: &Tensor, cx: &Tensor,
        w_ih: &Tensor, w_hh: &Tensor,
        b_ih: &Tensor, b_hh: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mut h_out: FlodlTensor = ptr::null_mut();
        let mut c_out: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_lstm_cell(
                self.handle, hx.handle, cx.handle,
                w_ih.handle, w_hh.handle,
                b_ih.handle, b_hh.handle,
                &mut h_out, &mut c_out,
            )
        };
        check_err(err)?;
        Ok((Tensor::from_raw(h_out), Tensor::from_raw(c_out)))
    }

    // --- Fused loss functions ---

    /// Fused MSE loss: single libtorch kernel.
    /// reduction: 0=None, 1=Mean, 2=Sum.
    pub fn mse_loss(&self, target: &Tensor, reduction: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_mse_loss(self.handle, target.handle, reduction, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused cross-entropy loss: single libtorch kernel.
    /// pred: [N,C] logits. target: [N] Int64 indices or [N,C] Float probs.
    /// reduction: 0=None, 1=Mean, 2=Sum.
    #[allow(clippy::too_many_arguments)]
    pub fn cross_entropy_loss(
        &self, target: &Tensor, reduction: i64,
        ignore_index: i64, label_smoothing: f64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_cross_entropy_loss(
                self.handle, target.handle,
                reduction, ignore_index, label_smoothing,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused BCE with logits loss: single libtorch kernel.
    /// Numerically stable binary cross-entropy from raw logits.
    /// reduction: 0=None, 1=Mean, 2=Sum.
    pub fn bce_with_logits_loss(&self, target: &Tensor, reduction: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_bce_with_logits_loss(
                self.handle, target.handle, reduction, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused L1 loss: single libtorch kernel.
    /// reduction: 0=None, 1=Mean, 2=Sum.
    pub fn l1_loss(&self, target: &Tensor, reduction: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_l1_loss(self.handle, target.handle, reduction, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused Smooth L1 (Huber) loss: single libtorch kernel.
    /// reduction: 0=None, 1=Mean, 2=Sum. beta: transition point.
    pub fn smooth_l1_loss(&self, target: &Tensor, reduction: i64, beta: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_smooth_l1_loss(
                self.handle, target.handle, reduction, beta, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused KL divergence loss: single libtorch kernel.
    /// input: log-probabilities. target: probabilities.
    /// reduction: 0=None, 1=Mean, 2=Sum, 5=BatchMean.
    pub fn kl_div_loss(&self, target: &Tensor, reduction: i64, log_target: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_kl_div_loss(
                self.handle, target.handle, reduction, log_target as i32, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Fused batch normalization ---

    /// Fused batch normalization: single libtorch kernel.
    /// When training=true, updates running_mean/running_var in-place.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm(
        &self, weight: Option<&Tensor>, bias: Option<&Tensor>,
        running_mean: Option<&Tensor>, running_var: Option<&Tensor>,
        training: bool, momentum: f64, eps: f64,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let w = weight.map_or(ptr::null_mut(), |t| t.handle);
        let b = bias.map_or(ptr::null_mut(), |t| t.handle);
        let rm = running_mean.map_or(ptr::null_mut(), |t| t.handle);
        let rv = running_var.map_or(ptr::null_mut(), |t| t.handle);
        let err = unsafe {
            ffi::flodl_batch_norm(
                self.handle, w, b, rm, rv,
                training as i32, momentum, eps, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Fused dropout ---

    /// Fused dropout: single libtorch kernel with inverted scaling.
    pub fn dropout(&self, p: f64, training: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_dropout(self.handle, p, training as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused 2D feature dropout: drops entire channels.
    pub fn feature_dropout(&self, p: f64, training: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_feature_dropout(self.handle, p, training as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Missing wrappers for existing shims ---

    /// Create evenly spaced values.
    pub fn linspace(start: f64, end: f64, steps: i64, opts: TensorOptions) -> Result<Self> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_linspace(start, end, steps, opts.dtype as i32, dt, di, &mut handle)
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a range of values [start, end) with given step.
    pub fn arange(start: f64, end: f64, step: f64, opts: TensorOptions) -> Result<Self> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_arange(start, end, step, opts.dtype as i32, dt, di, &mut handle)
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Scalar minimum.
    pub fn min(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_min(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scalar maximum.
    pub fn max(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_max(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// L2 (Frobenius) norm of all elements.
    pub fn norm(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_norm(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Minimum along a dimension (values only).
    pub fn min_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_min_dim(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Maximum along a dimension (values only).
    pub fn max_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_max_dim(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Argmax along a dimension.
    pub fn argmax(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_argmax(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise greater-than-or-equal comparison against a scalar.
    pub fn ge_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ge_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise less-than-or-equal comparison against a scalar.
    pub fn le_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_le_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise less-than comparison against a scalar.
    pub fn lt_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_lt_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter a selected index back into a tensor.
    pub fn select_scatter(&self, src: &Tensor, dim: i32, index: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_select_scatter(self.handle, src.handle, dim, index, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Conditional select: where(condition, self, other).
    pub fn where_cond(condition: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_where(condition.handle, x.handle, y.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Squeeze (remove) a dimension of size 1.
    pub fn squeeze(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_squeeze(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Unsqueeze (insert) a dimension of size 1.
    pub fn unsqueeze(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_unsqueeze(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Max pooling over a 2D input (`[B, C, H, W]`).
    ///
    /// Equivalent to `torch.nn.functional.max_pool2d`.
    pub fn max_pool2d(
        &self,
        kernel_size: [i64; 2],
        stride: [i64; 2],
        padding: [i64; 2],
        dilation: [i64; 2],
        ceil_mode: bool,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut ks = kernel_size;
        let mut st = stride;
        let mut pd = padding;
        let mut dl = dilation;
        let err = unsafe {
            ffi::flodl_max_pool2d(
                self.handle,
                ks.as_mut_ptr(), st.as_mut_ptr(),
                pd.as_mut_ptr(), dl.as_mut_ptr(),
                ceil_mode as i32, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Adaptive average pooling to target spatial size.
    pub fn adaptive_avg_pool2d(&self, output_size: [i64; 2]) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let mut os = output_size;
        let err = unsafe {
            ffi::flodl_adaptive_avg_pool2d(self.handle, os.as_mut_ptr(), &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Grid sampling (bilinear/nearest interpolation).
    pub fn grid_sample(
        &self, grid: &Tensor, mode: i32, padding_mode: i32, align_corners: bool,
    ) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_grid_sample(
                self.handle, grid.handle, mode, padding_mode,
                align_corners as i32, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Cast to a different dtype.
    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_to_dtype(self.handle, dtype as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Check if all elements are finite (no inf/nan).
    pub fn all_finite(&self) -> Result<bool> {
        let mut result: i32 = 0;
        let err = unsafe { ffi::flodl_all_finite(self.handle, &mut result) };
        check_err(err)?;
        Ok(result != 0)
    }

    // --- Comparison (tensor-tensor) ---

    /// Element-wise greater-than (returns float mask: 0.0 or 1.0).
    pub fn gt(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_gt_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise less-than (returns float mask: 0.0 or 1.0).
    pub fn lt(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_lt_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise greater-than-or-equal (returns float mask: 0.0 or 1.0).
    pub fn ge(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ge_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise less-than-or-equal (returns float mask: 0.0 or 1.0).
    pub fn le(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_le_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise equality. Returns a mask (0.0 or 1.0) in the input's
    /// dtype for float inputs, or Float32 for integer/bool inputs.
    pub fn eq_tensor(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_eq_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise not-equal. Returns a mask (0.0 or 1.0) in the input's
    /// dtype for float inputs, or Float32 for integer/bool inputs.
    pub fn ne_tensor(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ne_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Additional reductions ---

    /// Argmin along a dimension.
    pub fn argmin(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_argmin(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Variance of all elements (Bessel-corrected).
    pub fn var(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_var(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Standard deviation of all elements (Bessel-corrected).
    #[allow(clippy::should_implement_trait)]
    pub fn std(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_std_op(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Variance along a dimension (Bessel-corrected).
    pub fn var_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_var_dim(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Standard deviation along a dimension (Bessel-corrected).
    pub fn std_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_std_dim(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Element-wise math (trig, rounding, sign) ---

    /// Element-wise sine.
    pub fn sin(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sin(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise cosine.
    pub fn cos(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_cos(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise sign (-1, 0, or +1).
    pub fn sign(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sign(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise floor.
    pub fn floor(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_floor(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise ceiling.
    pub fn ceil(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ceil(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise rounding to nearest integer.
    pub fn round(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_round(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise reciprocal (1/x).
    pub fn reciprocal(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_reciprocal(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Advanced indexing ---

    /// Gather values along a dimension using an index tensor.
    pub fn gather(&self, dim: i32, index: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_gather(self.handle, dim, index.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter-add: accumulate src into self at index positions along dim.
    pub fn scatter_add(&self, dim: i32, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_scatter_add(self.handle, dim, index.handle, src.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Sorting ---

    /// Top-k values and indices along a dimension. Returns (values, indices).
    pub fn topk(&self, k: i64, dim: i32, largest: bool, sorted: bool) -> Result<(Tensor, Tensor)> {
        let mut values: FlodlTensor = ptr::null_mut();
        let mut indices: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_topk(
                self.handle, k, dim, largest as i32, sorted as i32,
                &mut values, &mut indices,
            )
        };
        check_err(err)?;
        Ok((Tensor::from_raw(values), Tensor::from_raw(indices)))
    }

    /// Sort along a dimension. Returns (sorted_values, indices).
    pub fn sort(&self, dim: i32, descending: bool) -> Result<(Tensor, Tensor)> {
        let mut values: FlodlTensor = ptr::null_mut();
        let mut indices: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_sort(self.handle, dim, descending as i32, &mut values, &mut indices)
        };
        check_err(err)?;
        Ok((Tensor::from_raw(values), Tensor::from_raw(indices)))
    }

    // --- Tensor creation (additional) ---

    /// Create an identity matrix of size n x n.
    pub fn eye(n: i64, opts: TensorOptions) -> Result<Self> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_eye(n, opts.dtype as i32, dt, di, &mut handle)
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a tensor filled with a scalar value.
    pub fn full(shape: &[i64], value: f64, opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_full(
                shape.as_mut_ptr(), shape.len() as i32, value,
                opts.dtype as i32, dt, di, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    // --- Shape operations (additional) ---

    /// Split tensor into batches of `batch_size` along dimension 0.
    /// The last batch may be smaller if the tensor size isn't evenly divisible.
    ///
    /// ```ignore
    /// let data = Tensor::randn(&[100, 4], opts)?;
    /// for batch in data.batches(32)? {
    ///     let x = Variable::new(batch, false);
    ///     // ...
    /// }
    /// ```
    pub fn batches(&self, batch_size: i64) -> Result<Vec<Tensor>> {
        let n = self.shape()[0];
        let mut result = Vec::new();
        let mut start = 0i64;
        while start < n {
            let len = (batch_size).min(n - start);
            result.push(self.narrow(0, start, len)?);
            start += len;
        }
        Ok(result)
    }

    /// Split tensor into chunks along a dimension.
    pub fn chunk(&self, chunks: i32, dim: i32) -> Result<Vec<Tensor>> {
        let mut results_ptr: *mut FlodlTensor = ptr::null_mut();
        let mut count: i32 = 0;
        let err = unsafe {
            ffi::flodl_chunk(self.handle, chunks, dim, &mut results_ptr, &mut count)
        };
        check_err(err)?;
        let mut tensors = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let handle = unsafe { *results_ptr.add(i) };
            tensors.push(Tensor::from_raw(handle));
        }
        if !results_ptr.is_null() {
            // Free the C-allocated array (tensors are now owned by Rust).
            // flodl_free_string is just free() — safe for any malloc'd pointer.
            unsafe { ffi::flodl_free_string(results_ptr as *mut i8) };
        }
        Ok(tensors)
    }

    /// Repeat the tensor along each dimension.
    pub fn repeat(&self, repeats: &[i64]) -> Result<Tensor> {
        let mut repeats = repeats.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_repeat(self.handle, repeats.as_mut_ptr(), repeats.len() as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Constant-value padding. Padding format matches PyTorch: [left, right, top, bottom, ...].
    pub fn pad(&self, padding: &[i64], value: f64) -> Result<Tensor> {
        let mut padding = padding.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_pad(
                self.handle, padding.as_mut_ptr(), padding.len() as i32,
                value, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Insert multiple dimensions of size 1.
    /// Dims are sorted ascending and applied sequentially.
    pub fn unsqueeze_many(&self, dims: &[i32]) -> Result<Tensor> {
        let mut sorted = dims.to_vec();
        sorted.sort();
        let mut t = self.unsqueeze(sorted[0])?;
        for &d in &sorted[1..] {
            t = t.unsqueeze(d)?;
        }
        Ok(t)
    }

    /// Compute meshgrid from a slice of 1-D tensors (always "ij" indexing).
    pub fn meshgrid(tensors: &[&Tensor]) -> Result<Vec<Tensor>> {
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let mut results_ptr: *mut FlodlTensor = ptr::null_mut();
        let mut count: i32 = 0;
        let err = unsafe {
            ffi::flodl_meshgrid(
                handles.as_mut_ptr(), handles.len() as i32,
                &mut results_ptr, &mut count,
            )
        };
        check_err(err)?;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let handle = unsafe { *results_ptr.add(i) };
            out.push(Tensor::from_raw(handle));
        }
        if !results_ptr.is_null() {
            unsafe { ffi::flodl_free_string(results_ptr as *mut i8) };
        }
        Ok(out)
    }

    /// Pairwise L2 distance between rows of two batched matrices.
    /// Input shapes: `[B, P, D]` and `[B, R, D]` -> output `[B, P, R]`.
    pub fn cdist(&self, other: &Tensor) -> Result<Tensor> {
        self.cdist_p(other, 2.0)
    }

    /// Pairwise distance with custom p-norm.
    pub fn cdist_p(&self, other: &Tensor, p: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_cdist(self.handle, other.handle, p, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Device ---

    /// Move this tensor to a different device (CPU or CUDA).
    /// Returns a new tensor; the original is unchanged.
    ///
    /// ```ignore
    /// let gpu = t.to_device(Device::CUDA(0))?;
    /// ```
    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = device.to_ffi();
        let err = unsafe { ffi::flodl_to_device(self.handle, dt, di, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Move this tensor to the same device as `other`.
    /// No-op (returns a clone) if both are already on the same device.
    ///
    /// ```ignore
    /// let x = x.to_device_of(&weights)?;  // ensure same device
    /// ```
    pub fn to_device_of(&self, other: &Tensor) -> Result<Tensor> {
        let target = other.device();
        if self.device() == target {
            return Ok(self.clone());
        }
        self.to_device(target)
    }

    /// Non-blocking device transfer. Combined with [`pin_memory`] for CPU->GPU,
    /// this allows the transfer to overlap with host computation.
    ///
    /// ```ignore
    /// let pinned = cpu_tensor.pin_memory()?;
    /// let gpu = pinned.to_device_async(Device::CUDA(0))?;
    /// // ... do CPU work while transfer runs ...
    /// cuda_synchronize(0); // ensure transfer is done before using gpu tensor
    /// ```
    pub fn to_device_async(&self, device: Device) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = device.to_ffi();
        let err = unsafe { ffi::flodl_to_device_async(self.handle, dt, di, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Autograd ---

    /// Set requires_grad on this tensor. Returns a new tensor that shares
    /// storage but has the grad flag set. This enables libtorch's native
    /// autograd tracking for all subsequent operations.
    pub fn set_requires_grad(&self, requires_grad: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_set_requires_grad(self.handle, requires_grad as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Check whether this tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        unsafe { ffi::flodl_requires_grad(self.handle) != 0 }
    }

    /// Run backward pass from this scalar tensor. Populates .grad() on
    /// all leaf tensors in the computation graph.
    pub fn backward(&self) -> Result<()> {
        let err = unsafe { ffi::flodl_backward(self.handle) };
        check_err(err)
    }

    /// Get the accumulated gradient for this tensor, if any.
    /// Returns None if no gradient has been computed.
    pub fn grad(&self) -> Option<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_grad(self.handle, &mut handle) };
        if !err.is_null() {
            unsafe { ffi::flodl_free_string(err) };
            return None;
        }
        if handle.is_null() {
            None
        } else {
            Some(Tensor::from_raw(handle))
        }
    }

    /// Replace the gradient tensor (for gradient clipping / unscaling).
    pub fn set_grad(&self, grad: &Tensor) -> Result<()> {
        let err = unsafe { ffi::flodl_set_grad(self.handle, grad.handle) };
        check_err(err)
    }

    /// Zero out the accumulated gradient.
    pub fn zero_grad(&self) -> Result<()> {
        let err = unsafe { ffi::flodl_zero_grad(self.handle) };
        check_err(err)
    }

    /// Null out the gradient pointer instead of zeroing the data.
    /// No CUDA kernel — just resets the grad tensor to undefined.
    /// This is what PyTorch does by default since 1.7.
    pub fn zero_grad_set_to_none(&self) {
        unsafe { ffi::flodl_zero_grad_set_to_none(self.handle) }
    }

    /// Fused clip_grad_norm: compute global L2 norm across all param grads
    /// and scale in-place if it exceeds max_norm. Single C++ call.
    /// Returns the original total norm before clipping.
    pub fn clip_grad_norm_fused(params: &[Tensor], max_norm: f64) -> Result<f64> {
        if params.is_empty() {
            return Ok(0.0);
        }
        let mut handles: Vec<FlodlTensor> = params.iter().map(|t| t.handle).collect();
        let mut total_norm: f64 = 0.0;
        let err = unsafe {
            ffi::flodl_clip_grad_norm(
                handles.as_mut_ptr(),
                handles.len() as i32,
                max_norm,
                &mut total_norm,
            )
        };
        check_err(err)?;
        Ok(total_norm)
    }

    /// Whether this tensor is a leaf in the autograd graph.
    /// A tensor is a leaf if it was created by the user (not by an op)
    /// or if it doesn't require grad.
    pub fn is_leaf(&self) -> bool {
        unsafe { ffi::flodl_is_leaf(self.handle) != 0 }
    }

    /// Count unique autograd nodes reachable from this tensor's grad_fn.
    /// Returns 0 for leaf tensors or tensors without gradient tracking.
    /// This is the number of backward operations libtorch will execute.
    pub fn autograd_node_count(&self) -> i64 {
        unsafe { ffi::flodl_autograd_node_count(self.handle) }
    }

    /// Detach from the computation graph. Returns a new tensor that shares
    /// storage but has no autograd history.
    pub fn detach(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_detach(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// In-place detach: sever the grad_fn chain on this tensor without
    /// allocating a new handle. After this call the tensor's autograd_meta
    /// no longer references any C++ Node objects, allowing the autograd
    /// graph to be freed immediately rather than when the tensor is dropped.
    pub fn detach_(&self) -> Result<()> {
        let err = unsafe { ffi::flodl_detach_(self.handle) };
        check_err(err)
    }

    // --- In-place operations ---

    /// In-place add: self += other
    pub fn add_(&self, other: &Tensor) -> Result<()> {
        let err = unsafe { ffi::flodl_add_(self.handle, other.handle) };
        check_err(err)
    }

    /// In-place subtract: self -= other
    pub fn sub_(&self, other: &Tensor) -> Result<()> {
        let err = unsafe { ffi::flodl_sub_(self.handle, other.handle) };
        check_err(err)
    }

    /// In-place scalar multiply: self *= scalar
    pub fn mul_scalar_(&self, scalar: f64) -> Result<()> {
        let err = unsafe { ffi::flodl_mul_scalar_(self.handle, scalar) };
        check_err(err)
    }

    /// In-place scalar add: self += scalar
    pub fn add_scalar_(&self, scalar: f64) -> Result<()> {
        let err = unsafe { ffi::flodl_add_scalar_(self.handle, scalar) };
        check_err(err)
    }

    /// In-place zero: self = 0
    pub fn zero_(&self) -> Result<()> {
        let err = unsafe { ffi::flodl_zero_(self.handle) };
        check_err(err)
    }

    /// In-place copy: `self = src`.
    ///
    /// Copies the data from `src` into `self`. Both tensors must have the
    /// same shape. When `non_blocking` is true, cross-device copies may
    /// be asynchronous (useful inside CUDA Graph capture).
    pub fn copy_(&self, src: &Tensor, non_blocking: bool) -> Result<()> {
        let err = unsafe { ffi::flodl_copy_(self.handle, src.handle, non_blocking as i32) };
        check_err(err)
    }

    /// Fused Adam/AdamW step: updates param, m, and v tensors in-place.
    #[allow(clippy::too_many_arguments)]
    ///
    /// Performs the full Adam update in a single FFI call (~5 kernel launches
    /// instead of ~16), eliminating temporary tensor allocations.
    ///
    /// - `self` — parameter tensor (updated in-place)
    /// - `grad` — gradient (read-only)
    /// - `m`, `v` — moment buffers (updated in-place)
    /// - `weight_decay` — 0.0 for Adam, >0 for AdamW (decoupled)
    /// - `step` — timestep for bias correction
    pub fn adam_step(
        &self, grad: &Tensor, m: &Tensor, v: &Tensor,
        lr: f64, beta1: f64, beta2: f64, eps: f64,
        weight_decay: f64, step: i64,
    ) -> Result<()> {
        let err = unsafe {
            ffi::flodl_adam_step(
                self.handle, grad.handle, m.handle, v.handle,
                lr, beta1, beta2, eps, weight_decay, step,
            )
        };
        check_err(err)
    }

    // --- Batched Adam step ---

    /// Perform Adam/AdamW update on all params in one C++ loop.
    /// Eliminates per-param FFI overhead. `lrs[i]` supports per-group LR.
    #[allow(clippy::too_many_arguments)]
    pub fn adam_step_batched(
        params: &[Tensor], grads: &[Tensor], ms: &[Tensor], vs: &[Tensor],
        lrs: &mut [f64], beta1: f64, beta2: f64, eps: f64,
        weight_decay: f64, step: i64,
    ) -> Result<()> {
        let count = params.len() as i32;
        let mut p_handles: Vec<FlodlTensor> = params.iter().map(|t| t.handle).collect();
        let mut g_handles: Vec<FlodlTensor> = grads.iter().map(|t| t.handle).collect();
        let mut m_handles: Vec<FlodlTensor> = ms.iter().map(|t| t.handle).collect();
        let mut v_handles: Vec<FlodlTensor> = vs.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_adam_step_batched(
                p_handles.as_mut_ptr(), g_handles.as_mut_ptr(),
                m_handles.as_mut_ptr(), v_handles.as_mut_ptr(),
                lrs.as_mut_ptr(), count,
                beta1, beta2, eps, weight_decay, step,
            )
        };
        check_err(err)
    }

    // --- Fused Adam/AdamW (multi-tensor kernel) ---
    // Uses libtorch's _fused_adam_ / _fused_adamw_ to perform the complete
    // Adam update across ALL params in a single kernel launch on CUDA.

    /// Fused Adam update (L2 weight decay) across all params in one kernel.
    ///
    /// On CUDA, this launches a single multi-tensor kernel instead of ~4N
    /// separate kernels for N parameters. On CPU, falls back to a fused loop.
    ///
    /// - `grad_scale` / `found_inf`: pass `None` to skip mixed-precision integration.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_adam_(
        params: &[Tensor], grads: &[Tensor], exp_avgs: &[Tensor], exp_avg_sqs: &[Tensor],
        lr: f64, beta1: f64, beta2: f64, eps: f64,
        weight_decay: f64, step: i64,
        grad_scale: Option<&Tensor>, found_inf: Option<&Tensor>,
    ) -> Result<()> {
        if params.is_empty() { return Ok(()); }
        let count = params.len() as i32;
        let mut p = Self::handles(params);
        let mut g = Self::handles(grads);
        let mut m = Self::handles(exp_avgs);
        let mut v = Self::handles(exp_avg_sqs);
        let gs = grad_scale.map_or(ptr::null_mut(), |t| t.handle);
        let fi = found_inf.map_or(ptr::null_mut(), |t| t.handle);
        let err = unsafe {
            ffi::flodl_fused_adam_(
                p.as_mut_ptr(), g.as_mut_ptr(), m.as_mut_ptr(), v.as_mut_ptr(),
                count, lr, beta1, beta2, eps, weight_decay, step, gs, fi,
            )
        };
        check_err(err)
    }

    /// Fused AdamW update (decoupled weight decay) across all params in one kernel.
    ///
    /// Same as [`fused_adam_`] but applies decoupled weight decay:
    /// `param *= (1 - lr * weight_decay)` before the Adam step.
    /// With `weight_decay = 0.0`, identical to `fused_adam_`.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_adamw_(
        params: &[Tensor], grads: &[Tensor], exp_avgs: &[Tensor], exp_avg_sqs: &[Tensor],
        lr: f64, beta1: f64, beta2: f64, eps: f64,
        weight_decay: f64, step: i64,
        grad_scale: Option<&Tensor>, found_inf: Option<&Tensor>,
    ) -> Result<()> {
        if params.is_empty() { return Ok(()); }
        let count = params.len() as i32;
        let mut p = Self::handles(params);
        let mut g = Self::handles(grads);
        let mut m = Self::handles(exp_avgs);
        let mut v = Self::handles(exp_avg_sqs);
        let gs = grad_scale.map_or(ptr::null_mut(), |t| t.handle);
        let fi = found_inf.map_or(ptr::null_mut(), |t| t.handle);
        let err = unsafe {
            ffi::flodl_fused_adamw_(
                p.as_mut_ptr(), g.as_mut_ptr(), m.as_mut_ptr(), v.as_mut_ptr(),
                count, lr, beta1, beta2, eps, weight_decay, step, gs, fi,
            )
        };
        check_err(err)
    }

    /// Collect FlodlTensor handles from a slice.
    fn handles(tensors: &[Tensor]) -> Vec<FlodlTensor> {
        tensors.iter().map(|t| t.handle).collect()
    }

    // --- Multi-tensor foreach operations ---
    // These use libtorch's _foreach_* ops which batch the same operation
    // across all tensors into fewer kernel launches on CUDA.

    /// In-place add scalar to all tensors: `tensors[i] += scalar`.
    /// Single batched kernel on CUDA instead of N separate launches.
    pub fn foreach_add_scalar_(tensors: &[Tensor], scalar: f64) -> Result<()> {
        if tensors.is_empty() { return Ok(()); }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_foreach_add_scalar_(handles.as_mut_ptr(), handles.len() as i32, scalar)
        };
        check_err(err)
    }

    /// In-place multiply all tensors by scalar: `tensors[i] *= scalar`.
    /// Single batched kernel on CUDA instead of N separate launches.
    pub fn foreach_mul_scalar_(tensors: &[Tensor], scalar: f64) -> Result<()> {
        if tensors.is_empty() { return Ok(()); }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_foreach_mul_scalar_(handles.as_mut_ptr(), handles.len() as i32, scalar)
        };
        check_err(err)
    }

    /// In-place zero all tensors: `tensors[i] = 0`.
    /// Single batched kernel on CUDA instead of N separate launches.
    pub fn foreach_zero_(tensors: &[Tensor]) -> Result<()> {
        if tensors.is_empty() { return Ok(()); }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_foreach_zero_(handles.as_mut_ptr(), handles.len() as i32)
        };
        check_err(err)
    }

    /// In-place add two tensor lists: `tensors1[i] += alpha * tensors2[i]`.
    /// Single batched kernel on CUDA instead of N separate launches.
    pub fn foreach_add_list_(tensors1: &[Tensor], tensors2: &[Tensor], alpha: f64) -> Result<()> {
        if tensors1.is_empty() { return Ok(()); }
        assert_eq!(tensors1.len(), tensors2.len(), "foreach_add_list_: list length mismatch");
        let mut h1: Vec<FlodlTensor> = tensors1.iter().map(|t| t.handle).collect();
        let mut h2: Vec<FlodlTensor> = tensors2.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_foreach_add_list_(
                h1.as_mut_ptr(), h2.as_mut_ptr(), h1.len() as i32, alpha,
            )
        };
        check_err(err)
    }

    /// Compute per-tensor norms. Returns a Vec of scalar tensors.
    /// Single batched kernel on CUDA instead of N separate norm calls.
    pub fn foreach_norm(tensors: &[Tensor], ord: f64) -> Result<Vec<Tensor>> {
        if tensors.is_empty() { return Ok(vec![]); }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let mut results: Vec<FlodlTensor> = vec![ptr::null_mut(); tensors.len()];
        let err = unsafe {
            ffi::flodl_foreach_norm(
                handles.as_mut_ptr(), handles.len() as i32, ord,
                results.as_mut_ptr(),
            )
        };
        check_err(err)?;
        Ok(results.into_iter().map(Tensor::from_raw).collect())
    }

    /// In-place lerp: `tensors1[i] += weight * (tensors2[i] - tensors1[i])`.
    /// Single batched kernel on CUDA instead of N separate launches.
    pub fn foreach_lerp_scalar_(tensors1: &[Tensor], tensors2: &[Tensor], weight: f64) -> Result<()> {
        if tensors1.is_empty() { return Ok(()); }
        assert_eq!(tensors1.len(), tensors2.len(), "foreach_lerp_scalar_: list length mismatch");
        let mut h1: Vec<FlodlTensor> = tensors1.iter().map(|t| t.handle).collect();
        let mut h2: Vec<FlodlTensor> = tensors2.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_foreach_lerp_scalar_(
                h1.as_mut_ptr(), h2.as_mut_ptr(), h1.len() as i32, weight,
            )
        };
        check_err(err)
    }

    /// In-place sqrt: `tensors[i] = sqrt(tensors[i])`.
    /// Single batched kernel on CUDA instead of N separate launches.
    pub fn foreach_sqrt_(tensors: &[Tensor]) -> Result<()> {
        if tensors.is_empty() { return Ok(()); }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_foreach_sqrt_(handles.as_mut_ptr(), handles.len() as i32)
        };
        check_err(err)
    }

    // --- Pinned memory ---

    /// Copy this CPU tensor into page-locked (pinned) memory.
    ///
    /// Pinned memory enables async CPU→GPU transfers via `cudaMemcpyAsync`.
    /// Only valid for CPU tensors. Returns a new tensor in pinned memory.
    pub fn pin_memory(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_pin_memory(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Returns true if this tensor is stored in pinned (page-locked) memory.
    pub fn is_pinned(&self) -> bool {
        unsafe { ffi::flodl_is_pinned(self.handle) != 0 }
    }

    // --- Memory format ---

    /// Convert to channels-last (NHWC) memory format. Only meaningful for 4D tensors.
    /// This is the Rust equivalent of `tensor.to(memory_format=torch.channels_last)`.
    pub fn to_channels_last(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_to_channels_last(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Returns true if this tensor is contiguous in channels-last format.
    pub fn is_channels_last(&self) -> bool {
        unsafe { ffi::flodl_is_channels_last(self.handle) != 0 }
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
///
/// On Linux, this also ensures CUDA libraries are loaded (they can be
/// dropped by the linker's `--as-needed` flag since no Rust code
/// directly references symbols in `libtorch_cuda.so`).
pub fn cuda_available() -> bool {
    // flodl_force_cuda_link references c10::cuda::device_count(),
    // creating a real symbol dependency on c10_cuda.so. This prevents
    // --as-needed from dropping CUDA libs. The call is cheap (no-op on
    // non-CUDA builds since the symbol resolves to a stub returning 0).
    unsafe { let _ = ffi::flodl_force_cuda_link(); }
    unsafe { ffi::flodl_cuda_is_available() != 0 }
}

/// Returns the number of CUDA devices.
pub fn cuda_device_count() -> i32 {
    unsafe { ffi::flodl_cuda_device_count() }
}

/// Query CUDA memory usage for a specific device.
/// Returns `(used_bytes, total_bytes)` or an error if CUDA is not available.
pub fn cuda_memory_info_idx(device_index: i32) -> Result<(u64, u64)> {
    let mut used: u64 = 0;
    let mut total: u64 = 0;
    check_err(unsafe { ffi::flodl_cuda_mem_info(device_index, &mut used, &mut total) })?;
    Ok((used, total))
}

/// Query CUDA memory usage for device 0.
/// Returns `(used_bytes, total_bytes)` or an error if CUDA is not available.
pub fn cuda_memory_info() -> Result<(u64, u64)> {
    cuda_memory_info_idx(0)
}

/// Query bytes reserved by the CUDA caching allocator on a specific device.
///
/// This is the Rust equivalent of `torch.cuda.memory_reserved()`. It can exceed
/// physical VRAM when unified memory spills to host RAM.
pub fn cuda_allocated_bytes_idx(device_index: i32) -> Result<u64> {
    let mut allocated: u64 = 0;
    check_err(unsafe { ffi::flodl_cuda_alloc_bytes(device_index, &mut allocated) })?;
    Ok(allocated)
}

/// Query bytes reserved by the CUDA caching allocator on device 0.
pub fn cuda_allocated_bytes() -> Result<u64> {
    cuda_allocated_bytes_idx(0)
}

/// Query bytes actively used by tensors on a specific device.
///
/// This is the Rust equivalent of `torch.cuda.memory_allocated()`. Unlike
/// `cuda_allocated_bytes` (which reports the allocator's total reservation),
/// this only counts sub-blocks currently backing live tensors.
pub fn cuda_active_bytes_idx(device_index: i32) -> Result<u64> {
    let mut active: u64 = 0;
    check_err(unsafe { ffi::flodl_cuda_active_bytes(device_index, &mut active) })?;
    Ok(active)
}

/// Query bytes actively used by tensors on device 0.
pub fn cuda_active_bytes() -> Result<u64> {
    cuda_active_bytes_idx(0)
}

/// Peak bytes allocated to tensors since last `cuda_reset_peak_stats()` on a specific device.
///
/// This is the Rust equivalent of `torch.cuda.max_memory_allocated()`.
pub fn cuda_peak_active_bytes_idx(device_index: i32) -> Result<u64> {
    let mut peak: u64 = 0;
    check_err(unsafe { ffi::flodl_cuda_peak_active_bytes(device_index, &mut peak) })?;
    Ok(peak)
}

/// Peak bytes allocated to tensors since last `cuda_reset_peak_stats()` on device 0.
pub fn cuda_peak_active_bytes() -> Result<u64> {
    cuda_peak_active_bytes_idx(0)
}

/// Peak bytes reserved by the CUDA caching allocator since last `cuda_reset_peak_stats()` on a specific device.
///
/// This is the Rust equivalent of `torch.cuda.max_memory_reserved()`.
pub fn cuda_peak_reserved_bytes_idx(device_index: i32) -> Result<u64> {
    let mut peak: u64 = 0;
    check_err(unsafe { ffi::flodl_cuda_peak_reserved_bytes(device_index, &mut peak) })?;
    Ok(peak)
}

/// Peak bytes reserved by the CUDA caching allocator since last `cuda_reset_peak_stats()` on device 0.
pub fn cuda_peak_reserved_bytes() -> Result<u64> {
    cuda_peak_reserved_bytes_idx(0)
}

/// Reset peak memory statistics for a specific device.
/// Equivalent to `torch.cuda.reset_peak_memory_stats()`.
pub fn cuda_reset_peak_stats_idx(device_index: i32) {
    unsafe { ffi::flodl_cuda_reset_peak_stats(device_index) }
}

/// Reset peak memory statistics for device 0.
pub fn cuda_reset_peak_stats() {
    cuda_reset_peak_stats_idx(0)
}

/// Release all unused cached memory from the CUDA caching allocator.
/// Equivalent to `torch.cuda.empty_cache()`.
pub fn cuda_empty_cache() {
    unsafe { ffi::flodl_cuda_empty_cache() }
}

/// Query GPU utilization percentage (0-100) via NVML.
/// Returns `None` if NVML is not available or the query fails.
pub fn cuda_utilization() -> Option<u32> {
    cuda_utilization_idx(0)
}

/// Query GPU utilization percentage for a specific device (0-100) via NVML.
pub fn cuda_utilization_idx(device_index: i32) -> Option<u32> {
    let val = unsafe { ffi::flodl_cuda_utilization(device_index) };
    if val >= 0 { Some(val as u32) } else { None }
}

/// Set the current CUDA device.
pub fn set_current_cuda_device(device_index: u8) {
    unsafe { ffi::flodl_set_current_device(device_index as i32) };
}

/// Get the current CUDA device index.
pub fn current_cuda_device() -> u8 {
    unsafe { ffi::flodl_get_current_device() as u8 }
}

/// Synchronize a CUDA device (wait for all pending work to complete).
pub fn cuda_synchronize(device_index: u8) {
    unsafe { ffi::flodl_cuda_synchronize(device_index as i32) };
}

/// Returns the GPU device name for the given index (e.g. "NVIDIA GeForce GTX 1060 6GB").
pub fn cuda_device_name_idx(device: i32) -> Option<String> {
    let mut buf = [0i8; 256];
    let err = unsafe { ffi::flodl_cuda_device_name(device, buf.as_mut_ptr(), 256) };
    if err.is_null() {
        let name = unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Some(name)
    } else {
        unsafe { ffi::flodl_free_string(err) };
        None
    }
}

/// Returns the GPU device name for device 0 (e.g. "NVIDIA GeForce GTX 1060 6GB").
pub fn cuda_device_name() -> Option<String> {
    cuda_device_name_idx(0)
}

/// Information about a CUDA device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device index (0-based).
    pub index: u8,
    /// Device name (e.g. "NVIDIA GeForce GTX 1060 6GB").
    pub name: String,
    /// Total device memory in bytes.
    pub total_memory: u64,
}

/// Enumerate all available CUDA devices.
pub fn cuda_devices() -> Vec<DeviceInfo> {
    let n = cuda_device_count();
    (0..n).filter_map(|i| {
        let name = cuda_device_name_idx(i)?;
        let total_memory = cuda_memory_info_idx(i).map(|(_, t)| t).unwrap_or(0);
        Some(DeviceInfo { index: i as u8, name, total_memory })
    }).collect()
}

/// One-line hardware summary for dashboard headers.
///
/// Returns something like:
/// `"CPU: AMD Ryzen 9 5900X (64GB) | GPU: NVIDIA GeForce GTX 1060 (6GB)"`
pub fn hardware_summary() -> String {
    let cpu = cpu_model_name().unwrap_or_else(|| "Unknown CPU".into());
    let threads = cpu_thread_count();
    let ram = total_ram_gb();
    let mut s = format!("{} ({} threads, {}GB)", cpu, threads, ram);

    if cuda_available() {
        let n = cuda_device_count();
        for i in 0..n {
            if let Some(gpu) = cuda_device_name_idx(i) {
                let vram_str = cuda_memory_info_idx(i)
                    .map(|(_, total)| format!(" ({}GB)", total / (1024 * 1024 * 1024)))
                    .unwrap_or_default();
                let _ = std::fmt::Write::write_fmt(&mut s, format_args!(
                    " | {}{}", gpu, vram_str
                ));
            }
        }
    }
    s
}

/// Count logical CPU threads from /proc/cpuinfo (Linux).
fn cpu_thread_count() -> usize {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .map(|s| s.lines().filter(|l| l.starts_with("processor")).count())
        .unwrap_or(1)
}

/// Read CPU model name from /proc/cpuinfo (Linux).
fn cpu_model_name() -> Option<String> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in info.lines() {
        if line.starts_with("model name") && let Some(val) = line.split(':').nth(1) {
            return Some(val.trim().to_string());
        }
    }
    None
}

/// Total physical RAM in GB (Linux).
fn total_ram_gb() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            for line in s.lines() {
                if line.starts_with("MemTotal:") {
                    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
                    return Some(kb / (1024 * 1024));
                }
            }
            None
        })
        .unwrap_or(0)
}

/// Enable or disable cuDNN benchmark mode.
///
/// When enabled, cuDNN will benchmark multiple convolution algorithms
/// on the first call and cache the fastest. Benefits fixed-size workloads
/// (FBRL, fixed image dims) with 5-10% speedup. Can hurt dynamic-shape
/// workloads due to warmup cost. Off by default — users opt in.
pub fn set_cudnn_benchmark(enable: bool) {
    unsafe { ffi::flodl_set_cudnn_benchmark(enable as i32) }
}

/// Seed all libtorch RNGs (CPU + CUDA) for reproducible tensor ops.
///
/// This sets the global seed for `Tensor::rand`, `Tensor::randn`,
/// dropout masks, and all other libtorch random operations.
/// Call before model creation and training for full reproducibility.
pub fn manual_seed(seed: u64) {
    unsafe { ffi::flodl_manual_seed(seed) }
}

/// Seed all CUDA device RNGs. No-op when built without CUDA.
///
/// Usually you want `manual_seed()` instead, which seeds both CPU
/// and CUDA. Use this only when you need to re-seed CUDA independently.
pub fn cuda_manual_seed_all(seed: u64) {
    unsafe { ffi::flodl_cuda_manual_seed_all(seed) }
}

/// Ask glibc to return free memory to the OS (Linux only).
///
/// Returns `true` if memory was actually released. Useful for
/// distinguishing allocator fragmentation from real leaks:
/// if RSS drops after calling this, the growth was fragmentation.
pub fn malloc_trim() -> bool {
    unsafe { ffi::flodl_malloc_trim() != 0 }
}

/// Number of live C++ Tensor handles (created but not yet dropped).
/// If this grows over time during training, there is a handle leak.
/// If it stays stable but RSS grows, the leak is inside libtorch.
pub fn live_tensor_count() -> u64 {
    LIVE_TENSOR_COUNT.load(Ordering::Relaxed)
}

/// Read current process RSS in kilobytes (Linux only).
/// Returns 0 on non-Linux or if /proc/self/statm is unreadable.
pub fn rss_kb() -> usize {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<usize>().ok())
        .map(|pages| pages * 4)
        .unwrap_or(0)
}

/// Returns the device to use in tests: CUDA when compiled with `--features cuda`
/// and a GPU is available, CPU otherwise.
#[cfg(test)]
pub fn test_device() -> Device {
    use std::sync::Once;
    static PRINT: Once = Once::new();
    let dev = if cfg!(feature = "cuda") && cuda_available() { Device::CUDA(0) } else { Device::CPU };
    PRINT.call_once(|| eprintln!("\n*** flodl test device: {} ***\n", dev));
    dev
}

/// Returns `TensorOptions` for tests (Float32 on `test_device()`).
#[cfg(test)]
pub fn test_opts() -> TensorOptions {
    TensorOptions { dtype: DType::Float32, device: test_device() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[2, 3], test_opts()).unwrap();
        assert_eq!(t.shape(), vec![2, 3]);
        assert_eq!(t.dtype(), DType::Float32);
        assert_eq!(t.device(), test_device());
        assert_eq!(t.numel(), 6);

        let data = t.to_f32_vec().unwrap();
        assert_eq!(data, vec![0.0; 6]);
    }

    #[test]
    fn test_from_f32() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        assert_eq!(t.shape(), vec![3]);
        let data = t.to_f32_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_add() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[4.0, 5.0, 6.0], &[3], test_device()).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_f32_vec().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2], test_device()).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.to_f32_vec().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_chaining() {
        let a = Tensor::from_f32(&[1.0, -2.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[1.0, 1.0, 1.0], &[3], test_device()).unwrap();
        let result = a.add(&b).unwrap().relu().unwrap().sum().unwrap();
        // [1+1, -2+1, 3+1] = [2, -1, 4] -> relu -> [2, 0, 4] -> sum -> 6
        let val = result.item().unwrap();
        assert!((val - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_drop_frees_memory() {
        // Create and immediately drop — verifies Drop doesn't crash.
        let _ = Tensor::zeros(&[1000, 1000], test_opts()).unwrap();
        // If Drop is broken, this would leak or crash.
    }

    #[test]
    fn test_debug_format() {
        let t = Tensor::zeros(&[2, 3], test_opts()).unwrap();
        let s = format!("{:?}", t);
        assert!(s.contains("[2, 3]"));
        assert!(s.contains("Float32"));
    }

    #[test]
    fn test_div_scalar() {
        let t = Tensor::from_f32(&[6.0, 9.0], &[2], test_device()).unwrap();
        let r = t.div_scalar(3.0).unwrap();
        let data = r.to_f32_vec().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean() {
        let t = Tensor::from_f32(&[2.0, 4.0, 6.0], &[3], test_device()).unwrap();
        let m = t.mean().unwrap();
        assert!((m.item().unwrap() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_flatten() {
        let t = Tensor::ones(&[2, 3, 4], test_opts()).unwrap();
        let f = t.flatten(1, 2).unwrap();
        assert_eq!(f.shape(), vec![2, 12]);
    }

    #[test]
    fn test_stack() {
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0], &[2], test_device()).unwrap();
        let c = Tensor::from_f32(&[5.0, 6.0], &[2], test_device()).unwrap();

        // Stack along dim 0: [3, 2]
        let s = Tensor::stack(&[&a, &b, &c], 0).unwrap();
        assert_eq!(s.shape(), vec![3, 2]);
        let data = s.to_f32_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Stack along dim 1: [2, 3]
        let s1 = Tensor::stack(&[&a, &b, &c], 1).unwrap();
        assert_eq!(s1.shape(), vec![2, 3]);
        let data1 = s1.to_f32_vec().unwrap();
        assert_eq!(data1, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_ones_from_f64_from_i64() {
        let o = Tensor::ones(&[2, 3], test_opts()).unwrap();
        assert_eq!(o.to_f32_vec().unwrap(), vec![1.0; 6]);

        let f = Tensor::from_f64(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        assert_eq!(f.dtype(), DType::Float64);
        assert_eq!(f.to_f64_vec().unwrap(), vec![1.0, 2.0, 3.0]);

        let i = Tensor::from_i64(&[10, 20, 30], &[3], test_device()).unwrap();
        assert_eq!(i.dtype(), DType::Int64);
        assert_eq!(i.to_i64_vec().unwrap(), vec![10, 20, 30]);
    }

    #[test]
    fn test_sub_mul_div() {
        let a = Tensor::from_f32(&[6.0, 8.0], &[2], test_device()).unwrap();
        let b = Tensor::from_f32(&[2.0, 3.0], &[2], test_device()).unwrap();
        assert_eq!(a.sub(&b).unwrap().to_f32_vec().unwrap(), vec![4.0, 5.0]);
        assert_eq!(a.mul(&b).unwrap().to_f32_vec().unwrap(), vec![12.0, 24.0]);
        let d = a.div(&b).unwrap().to_f32_vec().unwrap();
        assert!((d[0] - 3.0).abs() < 1e-5);
        assert!((d[1] - 8.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_scalar_ops() {
        let t = Tensor::from_f32(&[2.0, 4.0], &[2], test_device()).unwrap();
        assert_eq!(t.add_scalar(1.0).unwrap().to_f32_vec().unwrap(), vec![3.0, 5.0]);
        assert_eq!(t.mul_scalar(3.0).unwrap().to_f32_vec().unwrap(), vec![6.0, 12.0]);
        assert_eq!(t.neg().unwrap().to_f32_vec().unwrap(), vec![-2.0, -4.0]);
    }

    #[test]
    fn test_exp_log_sqrt_abs_pow() {
        let t = Tensor::from_f32(&[1.0, 4.0], &[2], test_device()).unwrap();
        let e = t.exp().unwrap().to_f32_vec().unwrap();
        assert!((e[0] - 1.0_f32.exp()).abs() < 1e-5);

        let l = t.log().unwrap().to_f32_vec().unwrap();
        assert!((l[1] - 4.0_f32.ln()).abs() < 1e-5);

        let s = t.sqrt().unwrap().to_f32_vec().unwrap();
        assert!((s[1] - 2.0).abs() < 1e-5);

        let a = Tensor::from_f32(&[-3.0, 5.0], &[2], test_device()).unwrap();
        assert_eq!(a.abs().unwrap().to_f32_vec().unwrap(), vec![3.0, 5.0]);

        let p = t.pow_scalar(2.0).unwrap().to_f32_vec().unwrap();
        assert!((p[0] - 1.0).abs() < 1e-5);
        assert!((p[1] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_clamp() {
        let t = Tensor::from_f32(&[-1.0, 0.5, 2.0], &[3], test_device()).unwrap();
        let c = t.clamp(0.0, 1.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(c, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_sum_dim_mean_dim() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let s = t.sum_dim(1, false).unwrap().to_f32_vec().unwrap();
        assert_eq!(s, vec![3.0, 7.0]);

        let m = t.mean_dim(0, false).unwrap().to_f32_vec().unwrap();
        assert!((m[0] - 2.0).abs() < 1e-5);
        assert!((m[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm() {
        let t = Tensor::from_f32(&[3.0, 4.0], &[2], test_device()).unwrap();
        let n = t.norm().unwrap().item().unwrap();
        assert!((n - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_reshape_transpose_narrow_select() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], test_device()).unwrap();
        let r = t.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape(), vec![3, 2]);
        assert_eq!(r.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let tr = t.transpose(0, 1).unwrap();
        assert_eq!(tr.shape(), vec![3, 2]);
        assert_eq!(tr.to_f32_vec().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let n = t.narrow(1, 0, 2).unwrap();
        assert_eq!(n.shape(), vec![2, 2]);
        assert_eq!(n.to_f32_vec().unwrap(), vec![1.0, 2.0, 4.0, 5.0]);

        let s = t.select(0, 1).unwrap();
        assert_eq!(s.shape(), vec![3]);
        assert_eq!(s.to_f32_vec().unwrap(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_permute_expand() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], test_device()).unwrap();
        let p = t.permute(&[1, 0]).unwrap();
        assert_eq!(p.shape(), vec![3, 2]);

        let s = Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], test_device()).unwrap();
        let e = s.expand(&[4, 3]).unwrap();
        assert_eq!(e.shape(), vec![4, 3]);
        let data = e.to_f32_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cat_many() {
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0, 5.0], &[3], test_device()).unwrap();
        let c = Tensor::from_f32(&[6.0], &[1], test_device()).unwrap();

        // Concatenate 3 tensors along dim 0
        let result = Tensor::cat_many(&[&a, &b, &c], 0).unwrap();
        assert_eq!(result.shape(), vec![6]);
        assert_eq!(result.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // 2D: concat along dim 1
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let y = Tensor::from_f32(&[5.0, 6.0], &[2, 1], test_device()).unwrap();
        let z = Tensor::from_f32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3], test_device()).unwrap();
        let result2 = Tensor::cat_many(&[&x, &y, &z], 1).unwrap();
        assert_eq!(result2.shape(), vec![2, 6]);
        assert_eq!(
            result2.to_f32_vec().unwrap(),
            vec![1.0, 2.0, 5.0, 7.0, 8.0, 9.0, 3.0, 4.0, 6.0, 10.0, 11.0, 12.0]
        );

        // Single tensor — should just return a copy
        let single = Tensor::cat_many(&[&a], 0).unwrap();
        assert_eq!(single.to_f32_vec().unwrap(), vec![1.0, 2.0]);

        // Empty list — should error
        let empty: Vec<&Tensor> = vec![];
        assert!(Tensor::cat_many(&empty, 0).is_err());
    }

    #[test]
    fn test_cat_index_select_index_add() {
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0, 5.0], &[3], test_device()).unwrap();
        let c = a.cat(&b, 0).unwrap();
        assert_eq!(c.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let t = Tensor::from_f32(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5], test_device()).unwrap();
        let idx = Tensor::from_i64(&[0, 2, 4], &[3], test_device()).unwrap();
        let sel = t.index_select(0, &idx).unwrap();
        assert_eq!(sel.to_f32_vec().unwrap(), vec![10.0, 30.0, 50.0]);

        let base = Tensor::zeros(&[5], test_opts()).unwrap();
        let src = Tensor::from_f32(&[1.0, 1.0, 1.0], &[3], test_device()).unwrap();
        let r = base.index_add(0, &idx, &src).unwrap();
        let data = r.to_f32_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
        assert!((data[4] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_narrow_scatter_select_scatter() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], test_device()).unwrap();
        let src = Tensor::from_f32(&[10.0, 20.0], &[2], test_device()).unwrap();
        let ns = t.narrow_scatter(&src, 0, 1).unwrap();
        assert_eq!(ns.to_f32_vec().unwrap(), vec![1.0, 10.0, 20.0, 4.0]);

        let t2 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], test_device()).unwrap();
        let row = Tensor::from_f32(&[10.0, 20.0, 30.0], &[3], test_device()).unwrap();
        let ss = t2.select_scatter(&row, 0, 0).unwrap();
        assert_eq!(ss.to_f32_vec().unwrap(), vec![10.0, 20.0, 30.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_activations() {
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        assert_eq!(t.relu().unwrap().to_f32_vec().unwrap(), vec![0.0, 0.0, 1.0]);

        let sig = t.sigmoid().unwrap().to_f32_vec().unwrap();
        assert!((sig[2] - 0.7310586).abs() < 1e-5);

        let th = t.tanh().unwrap().to_f32_vec().unwrap();
        assert!((th[2] - 1.0_f32.tanh()).abs() < 1e-5);

        // gelu/silu just check they don't crash and return right shape
        assert_eq!(t.gelu().unwrap().shape(), vec![3]);
        assert_eq!(t.silu().unwrap().shape(), vec![3]);
    }

    #[test]
    fn test_softmax_log_softmax() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let sm = t.softmax(0).unwrap().to_f32_vec().unwrap();
        let total: f32 = sm.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);

        let lsm = t.log_softmax(0).unwrap().to_f32_vec().unwrap();
        assert!(lsm[0] < 0.0 && lsm[1] < 0.0 && lsm[2] < 0.0);
    }

    #[test]
    fn test_eq_ne_tensor() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[1.0, 5.0, 3.0], &[3], test_device()).unwrap();

        let eq = a.eq_tensor(&b).unwrap().to_f32_vec().unwrap();
        assert_eq!(eq, vec![1.0, 0.0, 1.0]);

        let ne = a.ne_tensor(&b).unwrap().to_f32_vec().unwrap();
        assert_eq!(ne, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_gt_lt_ge_le_tensor() {
        let a = Tensor::from_f32(&[1.0, 3.0, 2.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[2.0, 2.0, 2.0], &[3], test_device()).unwrap();

        assert_eq!(a.gt(&b).unwrap().to_f32_vec().unwrap(), vec![0.0, 1.0, 0.0]);
        assert_eq!(a.lt(&b).unwrap().to_f32_vec().unwrap(), vec![1.0, 0.0, 0.0]);
        assert_eq!(a.ge(&b).unwrap().to_f32_vec().unwrap(), vec![0.0, 1.0, 1.0]);
        assert_eq!(a.le(&b).unwrap().to_f32_vec().unwrap(), vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sign_floor_ceil_round() {
        let t = Tensor::from_f32(&[-2.7, 0.0, 1.3], &[3], test_device()).unwrap();
        assert_eq!(t.sign().unwrap().to_f32_vec().unwrap(), vec![-1.0, 0.0, 1.0]);
        assert_eq!(t.floor().unwrap().to_f32_vec().unwrap(), vec![-3.0, 0.0, 1.0]);
        assert_eq!(t.ceil().unwrap().to_f32_vec().unwrap(), vec![-2.0, 0.0, 2.0]);

        let r = Tensor::from_f32(&[-0.6, 0.4, 1.5], &[3], test_device()).unwrap();
        let rv = r.round().unwrap().to_f32_vec().unwrap();
        assert!((rv[0] - (-1.0)).abs() < 1e-5);
        assert!((rv[1] - 0.0).abs() < 1e-5);
        assert!((rv[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmin() {
        let t = Tensor::from_f32(&[3.0, 1.0, 2.0], &[3], test_device()).unwrap();
        let idx = t.argmin(0, false).unwrap().to_i64_vec().unwrap();
        assert_eq!(idx, vec![1]);
    }

    #[test]
    fn test_var_std() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        // Bessel: var = ((1-2)²+(2-2)²+(3-2)²)/2 = 1.0
        assert!((t.var().unwrap().item().unwrap() - 1.0).abs() < 1e-5);
        assert!((t.std().unwrap().item().unwrap() - 1.0).abs() < 1e-5);

        // dim variant: [[1,2],[3,4]] var along dim=1 = [0.5, 0.5]
        let t2 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let vd = t2.var_dim(1, false).unwrap().to_f32_vec().unwrap();
        assert!((vd[0] - 0.5).abs() < 1e-5);
        assert!((vd[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_sin_cos_reciprocal() {
        let t = Tensor::from_f32(&[0.0, 1.0], &[2], test_device()).unwrap();
        let s = t.sin().unwrap().to_f32_vec().unwrap();
        assert!((s[0] - 0.0).abs() < 1e-5);
        assert!((s[1] - 1.0_f32.sin()).abs() < 1e-5);

        let c = t.cos().unwrap().to_f32_vec().unwrap();
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 1.0_f32.cos()).abs() < 1e-5);

        let r = Tensor::from_f32(&[2.0, 5.0], &[2], test_device()).unwrap();
        let rec = r.reciprocal().unwrap().to_f32_vec().unwrap();
        assert!((rec[0] - 0.5).abs() < 1e-5);
        assert!((rec[1] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_eye_full() {
        let eye = Tensor::eye(3, test_opts()).unwrap();
        assert_eq!(eye.shape(), vec![3, 3]);
        let data = eye.to_f32_vec().unwrap();
        assert_eq!(data, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        let f = Tensor::full(&[2, 3], 7.0, test_opts()).unwrap();
        assert_eq!(f.shape(), vec![2, 3]);
        assert_eq!(f.to_f32_vec().unwrap(), vec![7.0; 6]);
    }

    #[test]
    fn test_gather_scatter_add() {
        // gather: pick elements by index
        let t = Tensor::from_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2], test_device()).unwrap();
        let idx = Tensor::from_i64(&[1, 0, 0, 1], &[2, 2], test_device()).unwrap();
        let g = t.gather(1, &idx).unwrap().to_f32_vec().unwrap();
        assert_eq!(g, vec![20.0, 10.0, 30.0, 40.0]);

        // scatter_add: accumulate into base at positions
        let base = Tensor::zeros(&[2, 3], test_opts()).unwrap();
        let src = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let idx2 = Tensor::from_i64(&[0, 2, 1, 0], &[2, 2], test_device()).unwrap();
        let sa = base.scatter_add(1, &idx2, &src).unwrap();
        let data = sa.to_f32_vec().unwrap();
        // Row 0: pos 0 += 1.0, pos 2 += 2.0 → [1, 0, 2]
        // Row 1: pos 1 += 3.0, pos 0 += 4.0 → [4, 3, 0]
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[2] - 2.0).abs() < 1e-5);
        assert!((data[3] - 4.0).abs() < 1e-5);
        assert!((data[4] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_topk_sort() {
        let t = Tensor::from_f32(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5], test_device()).unwrap();
        let (vals, idxs) = t.topk(3, 0, true, true).unwrap();
        assert_eq!(vals.to_f32_vec().unwrap(), vec![5.0, 4.0, 3.0]);
        let idx_data = idxs.to_i64_vec().unwrap();
        assert_eq!(idx_data, vec![4, 2, 0]);

        let (svals, sidxs) = t.sort(0, false).unwrap();
        assert_eq!(svals.to_f32_vec().unwrap(), vec![1.0, 1.0, 3.0, 4.0, 5.0]);
        let si = sidxs.to_i64_vec().unwrap();
        assert_eq!(si[4], 4); // 5.0 was at index 4
    }

    #[test]
    fn test_chunk_repeat_pad() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], test_device()).unwrap();
        let chunks = t.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].to_f32_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(chunks[1].to_f32_vec().unwrap(), vec![3.0, 4.0]);
        assert_eq!(chunks[2].to_f32_vec().unwrap(), vec![5.0, 6.0]);

        let s = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let rep = s.repeat(&[3]).unwrap();
        assert_eq!(rep.to_f32_vec().unwrap(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        let pad = s.pad(&[1, 2], 0.0).unwrap();
        assert_eq!(pad.shape(), vec![5]);
        assert_eq!(pad.to_f32_vec().unwrap(), vec![0.0, 1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_zeros_like_ones_like() {
        let t = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let zl = Tensor::zeros_like(&t).unwrap();
        assert_eq!(zl.to_f32_vec().unwrap(), vec![0.0, 0.0]);
        assert_eq!(zl.dtype(), DType::Float32);

        let ol = Tensor::ones_like(&t).unwrap();
        assert_eq!(ol.to_f32_vec().unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_unsqueeze_many() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let u = t.unsqueeze_many(&[1, 2]).unwrap();
        assert_eq!(u.shape(), vec![3, 1, 1]);
        // Should match sequential unsqueeze
        let u2 = t.unsqueeze(1).unwrap().unsqueeze(2).unwrap();
        assert_eq!(u.shape(), u2.shape());
        assert_eq!(u.to_f32_vec().unwrap(), u2.to_f32_vec().unwrap());
    }

    #[test]
    fn test_meshgrid() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[4.0, 5.0], &[2], test_device()).unwrap();
        let grids = Tensor::meshgrid(&[&a, &b]).unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), vec![3, 2]);
        assert_eq!(grids[1].shape(), vec![3, 2]);
        // Grid 0: rows repeat a values
        assert_eq!(grids[0].to_f32_vec().unwrap(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        // Grid 1: cols repeat b values
        assert_eq!(grids[1].to_f32_vec().unwrap(), vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0]);
    }

    #[test]
    fn test_cdist() {
        // Two 2D points: [0,0] and [3,4] -> distance = 5
        let x = Tensor::from_f32(&[0.0, 0.0], &[1, 1, 2], test_device()).unwrap();
        let y = Tensor::from_f32(&[3.0, 4.0], &[1, 1, 2], test_device()).unwrap();
        let d = x.cdist(&y).unwrap();
        assert_eq!(d.shape(), vec![1, 1, 1]);
        assert!((d.item().unwrap() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_cdist_p1() {
        // L1: |3| + |4| = 7
        let x = Tensor::from_f32(&[0.0, 0.0], &[1, 1, 2], test_device()).unwrap();
        let y = Tensor::from_f32(&[3.0, 4.0], &[1, 1, 2], test_device()).unwrap();
        let d = x.cdist_p(&y, 1.0).unwrap();
        assert!((d.item().unwrap() - 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_from_i64_device() {
        let t = Tensor::from_i64(&[1, 2, 3], &[3], test_device()).unwrap();
        assert_eq!(t.device(), test_device());
        assert_eq!(t.dtype(), DType::Int64);
        assert_eq!(t.to_i64_vec().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_pin_memory() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], Device::CPU).unwrap();
        assert!(!t.is_pinned(), "regular CPU tensor should not be pinned");

        if cuda_available() {
            let pinned = t.pin_memory().unwrap();
            assert!(pinned.is_pinned(), "pin_memory() result should be pinned");
            assert_eq!(pinned.device(), Device::CPU, "pinned tensor should stay on CPU");
            assert_eq!(pinned.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0],
                "data should be preserved after pinning");
        } else {
            // pin_memory requires CUDA — verify it returns an error on CPU-only
            assert!(t.pin_memory().is_err(),
                "pin_memory should fail without CUDA");
        }
    }

    #[test]
    fn test_channels_last() {
        let t = Tensor::randn(&[1, 3, 4, 4], test_opts()).unwrap();
        assert!(!t.is_channels_last());
        let cl = t.to_channels_last().unwrap();
        assert!(cl.is_channels_last());
        assert_eq!(cl.shape(), vec![1, 3, 4, 4]); // shape unchanged
    }

    #[test]
    fn test_adam_step_basic() {
        // Basic smoke test for the fused adam_step at tensor level
        let param = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let grad = Tensor::from_f32(&[0.5, 0.5], &[2], test_device()).unwrap();
        let m = Tensor::zeros(&[2], test_opts()).unwrap();
        let v = Tensor::zeros(&[2], test_opts()).unwrap();

        param.adam_step(&grad, &m, &v, 0.001, 0.9, 0.999, 1e-8, 0.0, 1).unwrap();

        let p = param.to_f32_vec().unwrap();
        assert!(p[0] < 1.0, "param[0] should decrease");
        assert!(p[1] < 2.0, "param[1] should decrease");
        // m and v should be non-zero after the step
        let m_data = m.to_f32_vec().unwrap();
        let v_data = v.to_f32_vec().unwrap();
        assert!(m_data[0] > 0.0, "m should be updated");
        assert!(v_data[0] > 0.0, "v should be updated");
    }

    // --- Device model tests ---

    #[test]
    fn test_device_enum_basics() {
        assert_eq!(Device::CPU, Device::CPU);
        assert_eq!(Device::CUDA(0), Device::CUDA(0));
        assert_ne!(Device::CUDA(0), Device::CUDA(1));
        assert_ne!(Device::CPU, Device::CUDA(0));

        assert!(!Device::CPU.is_cuda());
        assert!(Device::CUDA(0).is_cuda());
        assert!(Device::CUDA(1).is_cuda());

        assert_eq!(Device::CPU.index(), 0);
        assert_eq!(Device::CUDA(0).index(), 0);
        assert_eq!(Device::CUDA(1).index(), 1);
    }

    #[test]
    fn test_device_display() {
        assert_eq!(format!("{}", Device::CPU), "cpu");
        assert_eq!(format!("{}", Device::CUDA(0)), "cuda");
        assert_eq!(format!("{}", Device::CUDA(1)), "cuda:1");
    }

    #[test]
    fn test_device_ffi_roundtrip() {
        let devices = [Device::CPU, Device::CUDA(0), Device::CUDA(1), Device::CUDA(7)];
        for dev in &devices {
            let (dt, di) = dev.to_ffi();
            let back = Device::from_ffi(dt, di);
            assert_eq!(*dev, back, "FFI roundtrip failed for {:?}", dev);
        }
    }

    #[test]
    fn test_device_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Device::CPU);
        set.insert(Device::CUDA(0));
        set.insert(Device::CUDA(1));
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Device::CPU));
        assert!(set.contains(&Device::CUDA(0)));
        assert!(set.contains(&Device::CUDA(1)));
    }

    // --- Send + Sync compile-time checks ---

    #[test]
    fn test_tensor_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Tensor>();
    }

    #[test]
    fn test_manual_seed_reproducible() {
        let opts = test_opts();
        manual_seed(123);
        let a = Tensor::randn(&[4, 4], opts).unwrap().to_f32_vec().unwrap();
        manual_seed(123);
        let b = Tensor::randn(&[4, 4], opts).unwrap().to_f32_vec().unwrap();
        assert_eq!(a, b);
    }

    // --- fused adam tests ---

    #[test]
    fn test_fused_adamw_matches_batched() {
        // Run the same update with both implementations, verify results match
        let dev = test_device();
        let opts = test_opts();

        // Create two identical copies of params/moments
        manual_seed(42);
        let p1 = Tensor::randn(&[4, 3], opts).unwrap();
        let p2 = Tensor::from_f32(&p1.to_f32_vec().unwrap(), &[4, 3], dev).unwrap();
        let g = Tensor::randn(&[4, 3], opts).unwrap();
        let m1 = Tensor::zeros(&[4, 3], opts).unwrap();
        let m2 = Tensor::zeros(&[4, 3], opts).unwrap();
        let v1 = Tensor::zeros(&[4, 3], opts).unwrap();
        let v2 = Tensor::zeros(&[4, 3], opts).unwrap();

        let lr = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let wd = 0.01;

        // Batched (old path)
        p1.adam_step(&g, &m1, &v1, lr, beta1, beta2, eps, wd, 1).unwrap();

        // Fused (new path)
        Tensor::fused_adamw_(
            std::slice::from_ref(&p2), std::slice::from_ref(&g),
            std::slice::from_ref(&m2), std::slice::from_ref(&v2),
            lr, beta1, beta2, eps, wd, 1, None, None,
        ).unwrap();

        let p1_data = p1.to_f32_vec().unwrap();
        let p2_data = p2.to_f32_vec().unwrap();
        for (i, (a, b)) in p1_data.iter().zip(&p2_data).enumerate() {
            assert!((a - b).abs() < 1e-5,
                "param mismatch at {}: batched={}, fused={}", i, a, b);
        }

        let m1_data = m1.to_f32_vec().unwrap();
        let m2_data = m2.to_f32_vec().unwrap();
        for (i, (a, b)) in m1_data.iter().zip(&m2_data).enumerate() {
            assert!((a - b).abs() < 1e-6,
                "m mismatch at {}: batched={}, fused={}", i, a, b);
        }
    }

    #[test]
    fn test_fused_adam_no_weight_decay() {
        let opts = test_opts();
        let p = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], test_device()).unwrap();
        let g = Tensor::from_f32(&[0.1, 0.2, 0.3, 0.4], &[4], test_device()).unwrap();
        let m = Tensor::zeros(&[4], opts).unwrap();
        let v = Tensor::zeros(&[4], opts).unwrap();

        Tensor::fused_adamw_(
            std::slice::from_ref(&p), std::slice::from_ref(&g),
            std::slice::from_ref(&m), std::slice::from_ref(&v),
            0.001, 0.9, 0.999, 1e-8, 0.0, 1, None, None,
        ).unwrap();

        let p_data = p.to_f32_vec().unwrap();
        // Each param should decrease by ~lr
        let orig = [1.0f32, 2.0, 3.0, 4.0];
        for (i, &o) in orig.iter().enumerate() {
            assert!((p_data[i] - (o - 0.001)).abs() < 1e-4,
                "p[{}]: got {}, expected ~{}", i, p_data[i], o - 0.001);
        }
    }

    #[test]
    fn test_fused_adam_multi_step() {
        let opts = test_opts();
        let p = Tensor::from_f32(&[5.0], &[1], test_device()).unwrap();
        let g = Tensor::from_f32(&[1.0], &[1], test_device()).unwrap();
        let m = Tensor::zeros(&[1], opts).unwrap();
        let v = Tensor::zeros(&[1], opts).unwrap();

        for step in 1..=10 {
            Tensor::fused_adamw_(
                std::slice::from_ref(&p), std::slice::from_ref(&g),
                std::slice::from_ref(&m), std::slice::from_ref(&v),
                0.01, 0.9, 0.999, 1e-8, 0.0, step, None, None,
            ).unwrap();
        }

        let p_data = p.to_f32_vec().unwrap();
        assert!(p_data[0] < 5.0, "param should decrease: got {}", p_data[0]);
        let m_data = m.to_f32_vec().unwrap();
        assert!((m_data[0] - 0.6513).abs() < 0.01,
            "m after 10 steps: got {}", m_data[0]);
    }

    #[test]
    fn test_fused_adam_empty_is_noop() {
        Tensor::fused_adamw_(&[], &[], &[], &[], 0.001, 0.9, 0.999, 1e-8, 0.0, 1, None, None).unwrap();
        Tensor::fused_adam_(&[], &[], &[], &[], 0.001, 0.9, 0.999, 1e-8, 0.0, 1, None, None).unwrap();
    }

    // --- foreach ops tests ---

    #[test]
    fn test_foreach_add_scalar() {
        let dev = test_device();
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0, 5.0], &[3], dev).unwrap();
        Tensor::foreach_add_scalar_(&[a.clone(), b.clone()], 10.0).unwrap();
        assert_eq!(a.to_f32_vec().unwrap(), vec![11.0, 12.0]);
        assert_eq!(b.to_f32_vec().unwrap(), vec![13.0, 14.0, 15.0]);
    }

    #[test]
    fn test_foreach_mul_scalar() {
        let dev = test_device();
        let a = Tensor::from_f32(&[2.0, 3.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[4.0, 5.0], &[2], dev).unwrap();
        Tensor::foreach_mul_scalar_(&[a.clone(), b.clone()], 0.5).unwrap();
        assert_eq!(a.to_f32_vec().unwrap(), vec![1.0, 1.5]);
        assert_eq!(b.to_f32_vec().unwrap(), vec![2.0, 2.5]);
    }

    #[test]
    fn test_foreach_zero() {
        let dev = test_device();
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0], &[2], dev).unwrap();
        Tensor::foreach_zero_(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(a.to_f32_vec().unwrap(), vec![0.0, 0.0]);
        assert_eq!(b.to_f32_vec().unwrap(), vec![0.0, 0.0]);
    }

    #[test]
    fn test_foreach_add_list() {
        let dev = test_device();
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[10.0, 20.0], &[2], dev).unwrap();
        let x = Tensor::from_f32(&[0.5, 0.5], &[2], dev).unwrap();
        let y = Tensor::from_f32(&[1.0, 1.0], &[2], dev).unwrap();
        // a += 2.0 * x, b += 2.0 * y
        Tensor::foreach_add_list_(
            &[a.clone(), b.clone()],
            &[x, y],
            2.0,
        ).unwrap();
        assert_eq!(a.to_f32_vec().unwrap(), vec![2.0, 3.0]);
        assert_eq!(b.to_f32_vec().unwrap(), vec![12.0, 22.0]);
    }

    #[test]
    fn test_foreach_norm() {
        let dev = test_device();
        let a = Tensor::from_f32(&[3.0, 4.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[1.0, 0.0], &[1, 2], dev).unwrap();
        let norms = Tensor::foreach_norm(&[a, b], 2.0).unwrap();
        assert_eq!(norms.len(), 2);
        let n0: f64 = norms[0].item().unwrap();
        let n1: f64 = norms[1].item().unwrap();
        assert!((n0 - 5.0).abs() < 1e-5, "norm of [3,4] should be 5, got {}", n0);
        assert!((n1 - 1.0).abs() < 1e-5, "norm of [1,0] should be 1, got {}", n1);
    }

    #[test]
    fn test_foreach_lerp_scalar() {
        let dev = test_device();
        let a = Tensor::from_f32(&[0.0, 10.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[10.0, 0.0], &[2], dev).unwrap();
        // a = a + 0.5 * (b_target - a), where b_target is the second list
        let a_target = Tensor::from_f32(&[10.0, 10.0], &[2], dev).unwrap();
        let b_target = Tensor::from_f32(&[10.0, 10.0], &[2], dev).unwrap();
        Tensor::foreach_lerp_scalar_(
            &[a.clone(), b.clone()],
            &[a_target, b_target],
            0.5,
        ).unwrap();
        // a = 0 + 0.5*(10-0) = 5, 10 + 0.5*(10-10) = 10
        assert_eq!(a.to_f32_vec().unwrap(), vec![5.0, 10.0]);
        // b = 10 + 0.5*(10-10) = 10, 0 + 0.5*(10-0) = 5
        assert_eq!(b.to_f32_vec().unwrap(), vec![10.0, 5.0]);
    }

    #[test]
    fn test_foreach_sqrt() {
        let dev = test_device();
        let a = Tensor::from_f32(&[4.0, 9.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[16.0, 25.0], &[2], dev).unwrap();
        Tensor::foreach_sqrt_(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(a.to_f32_vec().unwrap(), vec![2.0, 3.0]);
        assert_eq!(b.to_f32_vec().unwrap(), vec![4.0, 5.0]);
    }

    #[test]
    fn test_foreach_empty_list_is_noop() {
        // All foreach ops should handle empty lists gracefully
        Tensor::foreach_add_scalar_(&[], 1.0).unwrap();
        Tensor::foreach_mul_scalar_(&[], 1.0).unwrap();
        Tensor::foreach_zero_(&[]).unwrap();
        Tensor::foreach_add_list_(&[], &[], 1.0).unwrap();
        assert!(Tensor::foreach_norm(&[], 2.0).unwrap().is_empty());
        Tensor::foreach_lerp_scalar_(&[], &[], 0.5).unwrap();
        Tensor::foreach_sqrt_(&[]).unwrap();
    }
}
