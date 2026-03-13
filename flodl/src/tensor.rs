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

use flodl_sys::{self as ffi, FlodlTensor};

/// DType represents the data type of tensor elements.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Device {
    CPU = ffi::FLODL_CPU,
    CUDA = ffi::FLODL_CUDA,
}

impl Device {
    fn from_raw(v: i32) -> Self {
        match v {
            ffi::FLODL_CUDA => Device::CUDA,
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
        let err = unsafe {
            ffi::flodl_zeros(
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
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_ones(
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
    ///
    /// ```ignore
    /// let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU)?;
    /// assert_eq!(t.shape(), vec![2, 2]);
    /// ```
    pub fn from_f32(data: &[f32], shape: &[i64], device: Device) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_from_blob(
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

    /// Create a tensor from f64 data.
    pub fn from_f64(data: &[f64], shape: &[i64], device: Device) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_from_blob(
                data.as_ptr() as *mut c_void,
                shape.as_mut_ptr(),
                shape.len() as i32,
                DType::Float64 as i32,
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
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_from_blob(
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
        unsafe { ffi::flodl_ndim(self.handle) as usize }
    }

    /// Shape as a Vec.
    pub fn shape(&self) -> Vec<i64> {
        let n = self.ndim();
        (0..n)
            .map(|i| unsafe { ffi::flodl_shape(self.handle, i as i32) })
            .collect()
    }

    /// Total number of elements.
    pub fn numel(&self) -> i64 {
        unsafe { ffi::flodl_numel(self.handle) }
    }

    /// Data type.
    pub fn dtype(&self) -> DType {
        DType::from_raw(unsafe { ffi::flodl_dtype(self.handle) })
    }

    /// Device (CPU or CUDA).
    pub fn device(&self) -> Device {
        Device::from_raw(unsafe { ffi::flodl_device(self.handle) })
    }

    // --- Data access ---

    /// Copy tensor data to a `Vec<f32>`. Moves to CPU if needed.
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

    /// Copy tensor data to a `Vec<f64>`.
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

    /// Copy tensor data to a `Vec<i64>`. For integer-typed tensors.
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

    /// Extract a scalar value as f64 (for loss values, metrics, etc.).
    ///
    /// Preserves full precision for Float64 tensors. Works on any
    /// single-element tensor regardless of shape (like PyTorch's `.item()`).
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

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sub(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise multiplication.
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

    /// Multiply every element by a scalar.
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

    /// Tanh activation.
    pub fn tanh_op(&self) -> Result<Tensor> {
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
        let err = unsafe {
            ffi::flodl_rand(
                shape.as_mut_ptr(), shape.len() as i32,
                opts.dtype as i32, opts.device as i32,
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
        let err = unsafe {
            ffi::flodl_randn(
                shape.as_mut_ptr(), shape.len() as i32,
                opts.dtype as i32, opts.device as i32,
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

    // --- Missing wrappers for existing shims ---

    /// Create evenly spaced values.
    pub fn linspace(start: f64, end: f64, steps: i64, opts: TensorOptions) -> Result<Self> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_linspace(start, end, steps, opts.dtype as i32, opts.device as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a range of values [start, end) with given step.
    pub fn arange(start: f64, end: f64, step: f64, opts: TensorOptions) -> Result<Self> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_arange(start, end, step, opts.dtype as i32, opts.device as i32, &mut handle)
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

    /// Element-wise equality (returns float mask: 0.0 or 1.0).
    pub fn eq_tensor(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_eq_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise not-equal (returns float mask: 0.0 or 1.0).
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
        let err = unsafe {
            ffi::flodl_eye(n, opts.dtype as i32, opts.device as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a tensor filled with a scalar value.
    pub fn full(shape: &[i64], value: f64, opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_full(
                shape.as_mut_ptr(), shape.len() as i32, value,
                opts.dtype as i32, opts.device as i32, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    // --- Shape operations (additional) ---

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

    // --- Device ---

    /// Move this tensor to a different device (CPU or CUDA).
    /// Returns a new tensor; the original is unchanged.
    ///
    /// ```ignore
    /// let gpu = t.to_device(Device::CUDA)?;
    /// ```
    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_to_device(self.handle, device as i32, &mut handle) };
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

    /// Whether this tensor is a leaf in the autograd graph.
    /// A tensor is a leaf if it was created by the user (not by an op)
    /// or if it doesn't require grad.
    pub fn is_leaf(&self) -> bool {
        unsafe { ffi::flodl_is_leaf(self.handle) != 0 }
    }

    /// Detach from the computation graph. Returns a new tensor that shares
    /// storage but has no autograd history.
    pub fn detach(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_detach(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
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

/// Query CUDA memory usage for the current device.
/// Returns `(used_bytes, total_bytes)` or an error if CUDA is not available.
pub fn cuda_memory_info() -> Result<(u64, u64)> {
    let mut used: u64 = 0;
    let mut total: u64 = 0;
    check_err(unsafe { ffi::flodl_cuda_mem_info(&mut used, &mut total) })?;
    Ok((used, total))
}

/// Query GPU utilization percentage (0-100) via NVML.
/// Returns `None` if NVML is not available or the query fails.
pub fn cuda_utilization() -> Option<u32> {
    let val = unsafe { ffi::flodl_cuda_utilization(0) };
    if val >= 0 { Some(val as u32) } else { None }
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

    #[test]
    fn test_div_scalar() {
        let t = Tensor::from_f32(&[6.0, 9.0], &[2], Device::CPU).unwrap();
        let r = t.div_scalar(3.0).unwrap();
        let data = r.to_f32_vec().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean() {
        let t = Tensor::from_f32(&[2.0, 4.0, 6.0], &[3], Device::CPU).unwrap();
        let m = t.mean().unwrap();
        assert!((m.item().unwrap() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_flatten() {
        let t = Tensor::ones(&[2, 3, 4], TensorOptions::default()).unwrap();
        let f = t.flatten(1, 2).unwrap();
        assert_eq!(f.shape(), vec![2, 12]);
    }

    #[test]
    fn test_stack() {
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], Device::CPU).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0], &[2], Device::CPU).unwrap();
        let c = Tensor::from_f32(&[5.0, 6.0], &[2], Device::CPU).unwrap();

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
        let o = Tensor::ones(&[2, 3], TensorOptions::default()).unwrap();
        assert_eq!(o.to_f32_vec().unwrap(), vec![1.0; 6]);

        let f = Tensor::from_f64(&[1.0, 2.0, 3.0], &[3], Device::CPU).unwrap();
        assert_eq!(f.dtype(), DType::Float64);
        assert_eq!(f.to_f64_vec().unwrap(), vec![1.0, 2.0, 3.0]);

        let i = Tensor::from_i64(&[10, 20, 30], &[3]).unwrap();
        assert_eq!(i.dtype(), DType::Int64);
        assert_eq!(i.to_i64_vec().unwrap(), vec![10, 20, 30]);
    }

    #[test]
    fn test_sub_mul_div() {
        let a = Tensor::from_f32(&[6.0, 8.0], &[2], Device::CPU).unwrap();
        let b = Tensor::from_f32(&[2.0, 3.0], &[2], Device::CPU).unwrap();
        assert_eq!(a.sub(&b).unwrap().to_f32_vec().unwrap(), vec![4.0, 5.0]);
        assert_eq!(a.mul(&b).unwrap().to_f32_vec().unwrap(), vec![12.0, 24.0]);
        let d = a.div(&b).unwrap().to_f32_vec().unwrap();
        assert!((d[0] - 3.0).abs() < 1e-5);
        assert!((d[1] - 8.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_scalar_ops() {
        let t = Tensor::from_f32(&[2.0, 4.0], &[2], Device::CPU).unwrap();
        assert_eq!(t.add_scalar(1.0).unwrap().to_f32_vec().unwrap(), vec![3.0, 5.0]);
        assert_eq!(t.mul_scalar(3.0).unwrap().to_f32_vec().unwrap(), vec![6.0, 12.0]);
        assert_eq!(t.neg().unwrap().to_f32_vec().unwrap(), vec![-2.0, -4.0]);
    }

    #[test]
    fn test_exp_log_sqrt_abs_pow() {
        let t = Tensor::from_f32(&[1.0, 4.0], &[2], Device::CPU).unwrap();
        let e = t.exp().unwrap().to_f32_vec().unwrap();
        assert!((e[0] - 1.0_f32.exp()).abs() < 1e-5);

        let l = t.log().unwrap().to_f32_vec().unwrap();
        assert!((l[1] - 4.0_f32.ln()).abs() < 1e-5);

        let s = t.sqrt().unwrap().to_f32_vec().unwrap();
        assert!((s[1] - 2.0).abs() < 1e-5);

        let a = Tensor::from_f32(&[-3.0, 5.0], &[2], Device::CPU).unwrap();
        assert_eq!(a.abs().unwrap().to_f32_vec().unwrap(), vec![3.0, 5.0]);

        let p = t.pow_scalar(2.0).unwrap().to_f32_vec().unwrap();
        assert!((p[0] - 1.0).abs() < 1e-5);
        assert!((p[1] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_clamp() {
        let t = Tensor::from_f32(&[-1.0, 0.5, 2.0], &[3], Device::CPU).unwrap();
        let c = t.clamp(0.0, 1.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(c, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_sum_dim_mean_dim() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU).unwrap();
        let s = t.sum_dim(1, false).unwrap().to_f32_vec().unwrap();
        assert_eq!(s, vec![3.0, 7.0]);

        let m = t.mean_dim(0, false).unwrap().to_f32_vec().unwrap();
        assert!((m[0] - 2.0).abs() < 1e-5);
        assert!((m[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm() {
        let t = Tensor::from_f32(&[3.0, 4.0], &[2], Device::CPU).unwrap();
        let n = t.norm().unwrap().item().unwrap();
        assert!((n - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_reshape_transpose_narrow_select() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Device::CPU).unwrap();
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
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Device::CPU).unwrap();
        let p = t.permute(&[1, 0]).unwrap();
        assert_eq!(p.shape(), vec![3, 2]);

        let s = Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], Device::CPU).unwrap();
        let e = s.expand(&[4, 3]).unwrap();
        assert_eq!(e.shape(), vec![4, 3]);
        let data = e.to_f32_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cat_index_select_index_add() {
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], Device::CPU).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0, 5.0], &[3], Device::CPU).unwrap();
        let c = a.cat(&b, 0).unwrap();
        assert_eq!(c.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let t = Tensor::from_f32(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5], Device::CPU).unwrap();
        let idx = Tensor::from_i64(&[0, 2, 4], &[3]).unwrap();
        let sel = t.index_select(0, &idx).unwrap();
        assert_eq!(sel.to_f32_vec().unwrap(), vec![10.0, 30.0, 50.0]);

        let base = Tensor::zeros(&[5], TensorOptions::default()).unwrap();
        let src = Tensor::from_f32(&[1.0, 1.0, 1.0], &[3], Device::CPU).unwrap();
        let r = base.index_add(0, &idx, &src).unwrap();
        let data = r.to_f32_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
        assert!((data[4] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_narrow_scatter_select_scatter() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], Device::CPU).unwrap();
        let src = Tensor::from_f32(&[10.0, 20.0], &[2], Device::CPU).unwrap();
        let ns = t.narrow_scatter(&src, 0, 1).unwrap();
        assert_eq!(ns.to_f32_vec().unwrap(), vec![1.0, 10.0, 20.0, 4.0]);

        let t2 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Device::CPU).unwrap();
        let row = Tensor::from_f32(&[10.0, 20.0, 30.0], &[3], Device::CPU).unwrap();
        let ss = t2.select_scatter(&row, 0, 0).unwrap();
        assert_eq!(ss.to_f32_vec().unwrap(), vec![10.0, 20.0, 30.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_activations() {
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], Device::CPU).unwrap();
        assert_eq!(t.relu().unwrap().to_f32_vec().unwrap(), vec![0.0, 0.0, 1.0]);

        let sig = t.sigmoid().unwrap().to_f32_vec().unwrap();
        assert!((sig[2] - 0.7310586).abs() < 1e-5);

        let th = t.tanh_op().unwrap().to_f32_vec().unwrap();
        assert!((th[2] - 1.0_f32.tanh()).abs() < 1e-5);

        // gelu/silu just check they don't crash and return right shape
        assert_eq!(t.gelu().unwrap().shape(), vec![3]);
        assert_eq!(t.silu().unwrap().shape(), vec![3]);
    }

    #[test]
    fn test_softmax_log_softmax() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], Device::CPU).unwrap();
        let sm = t.softmax(0).unwrap().to_f32_vec().unwrap();
        let total: f32 = sm.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);

        let lsm = t.log_softmax(0).unwrap().to_f32_vec().unwrap();
        assert!(lsm[0] < 0.0 && lsm[1] < 0.0 && lsm[2] < 0.0);
    }

    #[test]
    fn test_eq_ne_tensor() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], Device::CPU).unwrap();
        let b = Tensor::from_f32(&[1.0, 5.0, 3.0], &[3], Device::CPU).unwrap();

        let eq = a.eq_tensor(&b).unwrap().to_f32_vec().unwrap();
        assert_eq!(eq, vec![1.0, 0.0, 1.0]);

        let ne = a.ne_tensor(&b).unwrap().to_f32_vec().unwrap();
        assert_eq!(ne, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_gt_lt_ge_le_tensor() {
        let a = Tensor::from_f32(&[1.0, 3.0, 2.0], &[3], Device::CPU).unwrap();
        let b = Tensor::from_f32(&[2.0, 2.0, 2.0], &[3], Device::CPU).unwrap();

        assert_eq!(a.gt(&b).unwrap().to_f32_vec().unwrap(), vec![0.0, 1.0, 0.0]);
        assert_eq!(a.lt(&b).unwrap().to_f32_vec().unwrap(), vec![1.0, 0.0, 0.0]);
        assert_eq!(a.ge(&b).unwrap().to_f32_vec().unwrap(), vec![0.0, 1.0, 1.0]);
        assert_eq!(a.le(&b).unwrap().to_f32_vec().unwrap(), vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sign_floor_ceil_round() {
        let t = Tensor::from_f32(&[-2.7, 0.0, 1.3], &[3], Device::CPU).unwrap();
        assert_eq!(t.sign().unwrap().to_f32_vec().unwrap(), vec![-1.0, 0.0, 1.0]);
        assert_eq!(t.floor().unwrap().to_f32_vec().unwrap(), vec![-3.0, 0.0, 1.0]);
        assert_eq!(t.ceil().unwrap().to_f32_vec().unwrap(), vec![-2.0, 0.0, 2.0]);

        let r = Tensor::from_f32(&[-0.6, 0.4, 1.5], &[3], Device::CPU).unwrap();
        let rv = r.round().unwrap().to_f32_vec().unwrap();
        assert!((rv[0] - (-1.0)).abs() < 1e-5);
        assert!((rv[1] - 0.0).abs() < 1e-5);
        assert!((rv[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmin() {
        let t = Tensor::from_f32(&[3.0, 1.0, 2.0], &[3], Device::CPU).unwrap();
        let idx = t.argmin(0, false).unwrap().to_i64_vec().unwrap();
        assert_eq!(idx, vec![1]);
    }

    #[test]
    fn test_var_std() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], Device::CPU).unwrap();
        // Bessel: var = ((1-2)²+(2-2)²+(3-2)²)/2 = 1.0
        assert!((t.var().unwrap().item().unwrap() - 1.0).abs() < 1e-5);
        assert!((t.std().unwrap().item().unwrap() - 1.0).abs() < 1e-5);

        // dim variant: [[1,2],[3,4]] var along dim=1 = [0.5, 0.5]
        let t2 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU).unwrap();
        let vd = t2.var_dim(1, false).unwrap().to_f32_vec().unwrap();
        assert!((vd[0] - 0.5).abs() < 1e-5);
        assert!((vd[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_sin_cos_reciprocal() {
        let t = Tensor::from_f32(&[0.0, 1.0], &[2], Device::CPU).unwrap();
        let s = t.sin().unwrap().to_f32_vec().unwrap();
        assert!((s[0] - 0.0).abs() < 1e-5);
        assert!((s[1] - 1.0_f32.sin()).abs() < 1e-5);

        let c = t.cos().unwrap().to_f32_vec().unwrap();
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 1.0_f32.cos()).abs() < 1e-5);

        let r = Tensor::from_f32(&[2.0, 5.0], &[2], Device::CPU).unwrap();
        let rec = r.reciprocal().unwrap().to_f32_vec().unwrap();
        assert!((rec[0] - 0.5).abs() < 1e-5);
        assert!((rec[1] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_eye_full() {
        let eye = Tensor::eye(3, TensorOptions::default()).unwrap();
        assert_eq!(eye.shape(), vec![3, 3]);
        let data = eye.to_f32_vec().unwrap();
        assert_eq!(data, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        let f = Tensor::full(&[2, 3], 7.0, TensorOptions::default()).unwrap();
        assert_eq!(f.shape(), vec![2, 3]);
        assert_eq!(f.to_f32_vec().unwrap(), vec![7.0; 6]);
    }

    #[test]
    fn test_gather_scatter_add() {
        // gather: pick elements by index
        let t = Tensor::from_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2], Device::CPU).unwrap();
        let idx = Tensor::from_i64(&[1, 0, 0, 1], &[2, 2]).unwrap();
        let g = t.gather(1, &idx).unwrap().to_f32_vec().unwrap();
        assert_eq!(g, vec![20.0, 10.0, 30.0, 40.0]);

        // scatter_add: accumulate into base at positions
        let base = Tensor::zeros(&[2, 3], TensorOptions::default()).unwrap();
        let src = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU).unwrap();
        let idx2 = Tensor::from_i64(&[0, 2, 1, 0], &[2, 2]).unwrap();
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
        let t = Tensor::from_f32(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5], Device::CPU).unwrap();
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
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], Device::CPU).unwrap();
        let chunks = t.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].to_f32_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(chunks[1].to_f32_vec().unwrap(), vec![3.0, 4.0]);
        assert_eq!(chunks[2].to_f32_vec().unwrap(), vec![5.0, 6.0]);

        let s = Tensor::from_f32(&[1.0, 2.0], &[2], Device::CPU).unwrap();
        let rep = s.repeat(&[3]).unwrap();
        assert_eq!(rep.to_f32_vec().unwrap(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        let pad = s.pad(&[1, 2], 0.0).unwrap();
        assert_eq!(pad.shape(), vec![5]);
        assert_eq!(pad.to_f32_vec().unwrap(), vec![0.0, 1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_zeros_like_ones_like() {
        let t = Tensor::from_f32(&[1.0, 2.0], &[2], Device::CPU).unwrap();
        let zl = Tensor::zeros_like(&t).unwrap();
        assert_eq!(zl.to_f32_vec().unwrap(), vec![0.0, 0.0]);
        assert_eq!(zl.dtype(), DType::Float32);

        let ol = Tensor::ones_like(&t).unwrap();
        assert_eq!(ol.to_f32_vec().unwrap(), vec![1.0, 1.0]);
    }
}
