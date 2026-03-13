use std::io::{Read, Write};

use crate::tensor::{Device, DType, Result, Tensor, TensorError};

use super::parameter::Parameter;

const MAGIC: [u8; 4] = [b'G', b'O', b'D', b'L'];
const VERSION: u32 = 2;

/// Save parameters to a binary checkpoint.
///
/// Format v2: GODL magic (4 bytes) + version (u32) + count (u32) + per-param data.
/// Each parameter stores its name, shape, dtype, and data in native precision.
/// Float16 params stay f16 on disk (half the size of f32).
pub fn save_parameters<W: Write>(w: &mut W, params: &[Parameter]) -> Result<()> {
    w.write_all(&MAGIC).map_err(io_err)?;
    w.write_all(&VERSION.to_le_bytes()).map_err(io_err)?;
    w.write_all(&(params.len() as u32).to_le_bytes()).map_err(io_err)?;

    for p in params {
        write_param(w, p)?;
    }
    Ok(())
}

/// Load parameters from a binary checkpoint.
///
/// Validates magic, version, parameter count, names, and shapes.
/// Loaded tensors are cast to the model parameter's current dtype and device.
pub fn load_parameters<R: Read>(r: &mut R, params: &[Parameter]) -> Result<()> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(io_err)?;
    if magic != MAGIC {
        return Err(TensorError::new("invalid checkpoint: bad magic"));
    }

    let version = read_u32(r)?;
    if version != 1 && version != 2 {
        return Err(TensorError::new(&format!(
            "unsupported checkpoint version {} (want 1 or 2)", version
        )));
    }

    let count = read_u32(r)? as usize;
    if count != params.len() {
        return Err(TensorError::new(&format!(
            "parameter count mismatch: checkpoint={} model={}", count, params.len()
        )));
    }

    for (i, p) in params.iter().enumerate() {
        read_param(r, p, i, version)?;
    }
    Ok(())
}

/// Save parameters to a file path.
pub fn save_parameters_file(path: &str, params: &[Parameter]) -> Result<()> {
    let mut f = std::fs::File::create(path).map_err(io_err)?;
    save_parameters(&mut f, params)
}

/// Load parameters from a file path.
pub fn load_parameters_file(path: &str, params: &[Parameter]) -> Result<()> {
    let mut f = std::fs::File::open(path).map_err(io_err)?;
    load_parameters(&mut f, params)
}

// --- Tensor state helpers for optimizer save/load ---

/// Write an optional tensor (for optimizer buffers that may not be initialized).
/// Uses native dtype — same format as v2 parameters.
pub(crate) fn write_tensor_state<W: Write>(w: &mut W, t: Option<&Tensor>) -> Result<()> {
    match t {
        None => {
            w.write_all(&[0u8]).map_err(io_err)?;
        }
        Some(t) => {
            w.write_all(&[1u8]).map_err(io_err)?;
            write_tensor_data(w, t)?;
        }
    }
    Ok(())
}

/// Read an optional tensor (returns None if the tensor was nil when saved).
pub(crate) fn read_tensor_state<R: Read>(r: &mut R, device: Device) -> Result<Option<Tensor>> {
    let mut present = [0u8; 1];
    r.read_exact(&mut present).map_err(io_err)?;
    if present[0] == 0 {
        return Ok(None);
    }

    let t = read_tensor_data(r)?;
    if device != Device::CPU {
        Ok(Some(t.to_device(device)?))
    } else {
        Ok(Some(t))
    }
}

// --- Internal: dtype-aware tensor serialization ---

/// DType tag byte for checkpoint format.
fn dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::Float16  => 1,
        DType::BFloat16 => 2,
        DType::Float32  => 3,
        DType::Float64  => 4,
        DType::Int32    => 5,
        DType::Int64    => 6,
    }
}

fn dtype_from_tag(tag: u8) -> Result<DType> {
    match tag {
        1 => Ok(DType::Float16),
        2 => Ok(DType::BFloat16),
        3 => Ok(DType::Float32),
        4 => Ok(DType::Float64),
        5 => Ok(DType::Int32),
        6 => Ok(DType::Int64),
        _ => Err(TensorError::new(&format!("unknown dtype tag: {}", tag))),
    }
}

/// Write tensor data in native dtype: shape + dtype tag + raw bytes.
fn write_tensor_data<W: Write>(w: &mut W, t: &Tensor) -> Result<()> {
    let shape = t.shape();
    w.write_all(&(shape.len() as u32).to_le_bytes()).map_err(io_err)?;
    for &s in &shape {
        w.write_all(&s.to_le_bytes()).map_err(io_err)?;
    }

    let dtype = t.dtype();
    w.write_all(&[dtype_tag(dtype)]).map_err(io_err)?;

    let numel = t.numel() as usize;
    let elem_size = dtype.element_size();
    let byte_count = numel * elem_size;

    // Copy raw bytes from tensor (handles any dtype)
    let raw = copy_raw_bytes(t, byte_count)?;
    w.write_all(&(byte_count as u64).to_le_bytes()).map_err(io_err)?;
    w.write_all(&raw).map_err(io_err)?;

    Ok(())
}

/// Read tensor data written by write_tensor_data.
fn read_tensor_data<R: Read>(r: &mut R) -> Result<Tensor> {
    let ndim = read_u32(r)? as usize;
    let mut shape = vec![0i64; ndim];
    for s in &mut shape {
        *s = read_i64(r)?;
    }

    let mut tag = [0u8; 1];
    r.read_exact(&mut tag).map_err(io_err)?;
    let dtype = dtype_from_tag(tag[0])?;

    let byte_count = read_u64(r)? as usize;
    let mut raw = vec![0u8; byte_count];
    r.read_exact(&mut raw).map_err(io_err)?;

    tensor_from_raw_bytes(&raw, &shape, dtype)
}

/// Copy raw bytes from a tensor (any dtype). Moves to CPU if needed.
fn copy_raw_bytes(t: &Tensor, byte_count: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; byte_count];
    let err = unsafe {
        flodl_sys::flodl_copy_data(
            t.raw(),
            buf.as_mut_ptr() as *mut std::ffi::c_void,
            byte_count as i64,
        )
    };
    check_err_raw(err)?;
    Ok(buf)
}

/// Construct a tensor from raw bytes + shape + dtype.
fn tensor_from_raw_bytes(raw: &[u8], shape: &[i64], dtype: DType) -> Result<Tensor> {
    // Route through the typed constructors to get a proper owned tensor
    match dtype {
        DType::Float32 => {
            let data: Vec<f32> = raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::from_f32(&data, shape, Device::CPU)
        }
        DType::Float64 => {
            let data: Vec<f64> = raw.chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Tensor::from_f64(&data, shape, Device::CPU)
        }
        DType::Int64 => {
            let data: Vec<i64> = raw.chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Tensor::from_i64(&data, shape)
        }
        DType::Float16 | DType::BFloat16 | DType::Int32 => {
            // For f16/bf16/i32: load as f32, then cast to target dtype.
            // This works because from_blob supports these dtypes via the shim,
            // but we don't have typed Rust constructors for them.
            // Load raw bytes via from_blob directly.
            let mut shape_v = shape.to_vec();
            let mut handle: flodl_sys::FlodlTensor = std::ptr::null_mut();
            let err = unsafe {
                flodl_sys::flodl_from_blob(
                    raw.as_ptr() as *mut std::ffi::c_void,
                    shape_v.as_mut_ptr(),
                    shape_v.len() as i32,
                    dtype as i32,
                    crate::tensor::Device::CPU as i32,
                    &mut handle,
                )
            };
            check_err_raw(err)?;
            debug_assert!(!handle.is_null());
            // Safety: from_blob clones the data in the shim, so handle is independent
            Ok(unsafe { Tensor::from_raw_handle(handle) })
        }
    }
}

// --- V1 legacy reader (f32-only format) ---

fn read_param_v1<R: Read>(r: &mut R, p: &Parameter, index: usize) -> Result<()> {
    let name_len = read_u32(r)? as usize;
    let mut name_bytes = vec![0u8; name_len];
    r.read_exact(&mut name_bytes).map_err(io_err)?;
    let name = String::from_utf8_lossy(&name_bytes);
    if name != p.name {
        return Err(TensorError::new(&format!(
            "parameter {}: name mismatch: checkpoint={:?} model={:?}", index, name, p.name
        )));
    }

    let ndim = read_u32(r)? as usize;
    let mut shape = vec![0i64; ndim];
    for s in &mut shape {
        *s = read_i64(r)?;
    }

    let model_shape = p.variable.shape();
    if shape != model_shape {
        return Err(TensorError::new(&format!(
            "parameter {:?}: shape mismatch: checkpoint={:?} model={:?}", p.name, shape, model_shape
        )));
    }

    let count = read_u64(r)? as usize;
    let data = read_f32_vec(r, count)?;
    let t = Tensor::from_f32(&data, &shape, Device::CPU)?;

    // Cast to model dtype if needed, then move to device
    let model_dtype = p.variable.data().dtype();
    let t = if t.dtype() != model_dtype { t.to_dtype(model_dtype)? } else { t };
    let dev = p.variable.data().device();
    if dev != Device::CPU {
        p.variable.set_data(t.to_device(dev)?);
    } else {
        p.variable.set_data(t);
    }
    Ok(())
}

// --- V2 param write/read ---

fn write_param<W: Write>(w: &mut W, p: &Parameter) -> Result<()> {
    let name = p.name.as_bytes();
    w.write_all(&(name.len() as u32).to_le_bytes()).map_err(io_err)?;
    w.write_all(name).map_err(io_err)?;

    write_tensor_data(w, &p.variable.data())?;

    Ok(())
}

fn read_param<R: Read>(r: &mut R, p: &Parameter, index: usize, version: u32) -> Result<()> {
    if version == 1 {
        return read_param_v1(r, p, index);
    }

    let name_len = read_u32(r)? as usize;
    let mut name_bytes = vec![0u8; name_len];
    r.read_exact(&mut name_bytes).map_err(io_err)?;
    let name = String::from_utf8_lossy(&name_bytes);
    if name != p.name {
        return Err(TensorError::new(&format!(
            "parameter {}: name mismatch: checkpoint={:?} model={:?}", index, name, p.name
        )));
    }

    let t = read_tensor_data(r)?;

    let model_shape = p.variable.shape();
    if t.shape() != model_shape {
        return Err(TensorError::new(&format!(
            "parameter {:?}: shape mismatch: checkpoint={:?} model={:?}", p.name, t.shape(), model_shape
        )));
    }

    // Cast to model dtype if checkpoint dtype differs, then move to device
    let model_dtype = p.variable.data().dtype();
    let t = if t.dtype() != model_dtype { t.to_dtype(model_dtype)? } else { t };
    let dev = p.variable.data().device();
    if dev != Device::CPU {
        p.variable.set_data(t.to_device(dev)?);
    } else {
        p.variable.set_data(t);
    }
    Ok(())
}

// --- Shared helpers ---

fn io_err(e: impl std::fmt::Display) -> TensorError {
    TensorError::new(&format!("io: {}", e))
}

fn check_err_raw(err: *mut i8) -> Result<()> {
    if err.is_null() {
        Ok(())
    } else {
        let msg = unsafe { std::ffi::CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned();
        unsafe { flodl_sys::flodl_free_string(err) };
        Err(TensorError::new(&msg))
    }
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32_vec<R: Read>(r: &mut R, count: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; count * 4];
    r.read_exact(&mut bytes).map_err(io_err)?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

// Pub(crate) helpers for optimizer state serialization
pub(crate) fn read_f64_le<R: Read>(r: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(f64::from_le_bytes(buf))
}
pub(crate) fn write_f64_le<W: Write>(w: &mut W, v: f64) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(io_err)?;
    Ok(())
}
pub(crate) fn write_u32_le<W: Write>(w: &mut W, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(io_err)?;
    Ok(())
}
pub(crate) fn write_i64_le<W: Write>(w: &mut W, v: i64) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(io_err)?;
    Ok(())
}
pub(crate) fn read_u32_le<R: Read>(r: &mut R) -> Result<u32> {
    read_u32(r)
}
pub(crate) fn read_i64_le<R: Read>(r: &mut R) -> Result<i64> {
    read_i64(r)
}
