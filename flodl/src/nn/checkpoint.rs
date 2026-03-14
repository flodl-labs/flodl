use std::io::{Read, Write};

use crate::tensor::{Device, DType, Result, Tensor, TensorError};

use super::parameter::Parameter;

/// Magic bytes for `.fdl` checkpoint files.
const MAGIC: [u8; 4] = *b"FDLC";
const VERSION: u32 = 2;

/// Report from a named parameter load: what was loaded, skipped, or missing.
#[derive(Debug, Clone)]
pub struct LoadReport {
    /// Parameters matched by name and loaded successfully.
    pub loaded: Vec<String>,
    /// Checkpoint entries with no matching model parameter (ignored).
    pub skipped: Vec<String>,
    /// Model parameters with no matching checkpoint entry (kept at init values).
    pub missing: Vec<String>,
}

/// Save named parameters to a binary checkpoint.
/// Uses the same v2 `.fdl` format — the qualified name from the tuple replaces the
/// parameter's own name.
pub fn save_named_parameters<W: Write>(w: &mut W, params: &[(String, Parameter)]) -> Result<()> {
    w.write_all(&MAGIC).map_err(io_err)?;
    w.write_all(&VERSION.to_le_bytes()).map_err(io_err)?;
    w.write_all(&(params.len() as u32).to_le_bytes()).map_err(io_err)?;

    for (name, p) in params {
        let name_bytes = name.as_bytes();
        w.write_all(&(name_bytes.len() as u32).to_le_bytes()).map_err(io_err)?;
        w.write_all(name_bytes).map_err(io_err)?;
        write_tensor_data(w, &p.variable.data())?;
    }
    Ok(())
}

/// Load named parameters from a checkpoint, matching by qualified name.
///
/// Returns a `LoadReport` describing what was matched, skipped, and missing.
/// Shape mismatches on a matched name are errors (not silent skips).
pub fn load_named_parameters<R: Read>(r: &mut R, params: &[(String, Parameter)]) -> Result<LoadReport> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(io_err)?;
    if magic != MAGIC {
        return Err(TensorError::new(
            "invalid checkpoint: bad magic (expected .fdl checkpoint)"
        ));
    }

    let version = read_u32(r)?;
    if version != 2 {
        return Err(TensorError::new(&format!(
            "unsupported checkpoint version {} (want 2)", version
        )));
    }

    let count = read_u32(r)? as usize;

    // Read all checkpoint entries into a map
    let mut ckpt: std::collections::HashMap<String, (Vec<i64>, DType, Vec<u8>)> =
        std::collections::HashMap::with_capacity(count);

    for _ in 0..count {
        let name_len = read_u32(r)? as usize;
        let mut name_bytes = vec![0u8; name_len];
        r.read_exact(&mut name_bytes).map_err(io_err)?;
        let name = String::from_utf8_lossy(&name_bytes).into_owned();

        let ndim = read_u32(r)? as usize;
        let mut shape = vec![0i64; ndim];
        for s in &mut shape { *s = read_i64(r)?; }
        let mut tag = [0u8; 1];
        r.read_exact(&mut tag).map_err(io_err)?;
        let dtype = dtype_from_tag(tag[0])?;
        let byte_count = read_u64(r)? as usize;
        let mut raw = vec![0u8; byte_count];
        r.read_exact(&mut raw).map_err(io_err)?;
        ckpt.insert(name, (shape, dtype, raw));
    }

    let mut loaded = Vec::new();
    let mut missing = Vec::new();

    for (name, p) in params {
        if let Some((shape, dtype, raw)) = ckpt.remove(name) {
            let model_shape = p.variable.shape();
            if shape != model_shape {
                return Err(TensorError::new(&format!(
                    "named parameter {:?}: shape mismatch: checkpoint={:?} model={:?}",
                    name, shape, model_shape
                )));
            }
            let t = tensor_from_raw_bytes(&raw, &shape, dtype)?;
            let model_dtype = p.variable.data().dtype();
            let t = if t.dtype() != model_dtype { t.to_dtype(model_dtype)? } else { t };
            let dev = p.variable.data().device();
            if dev != Device::CPU {
                p.variable.set_data(t.to_device(dev)?);
            } else {
                p.variable.set_data(t);
            }
            loaded.push(name.clone());
        } else {
            missing.push(name.clone());
        }
    }

    let skipped: Vec<String> = ckpt.into_keys().collect();

    Ok(LoadReport { loaded, skipped, missing })
}

/// Save named parameters to a file path. Uses gzip compression if path ends with `.gz`.
pub fn save_named_parameters_file(path: &str, params: &[(String, Parameter)]) -> Result<()> {
    let f = std::fs::File::create(path).map_err(io_err)?;
    if path.ends_with(".gz") {
        let mut w = flate2::write::GzEncoder::new(f, flate2::Compression::default());
        save_named_parameters(&mut w, params)?;
        w.finish().map_err(io_err)?;
        Ok(())
    } else {
        let mut w = std::io::BufWriter::new(f);
        save_named_parameters(&mut w, params)
    }
}

/// Load named parameters from a file path. Detects gzip from `.gz` extension.
pub fn load_named_parameters_file(path: &str, params: &[(String, Parameter)]) -> Result<LoadReport> {
    let f = std::fs::File::open(path).map_err(io_err)?;
    if path.ends_with(".gz") {
        let mut r = flate2::read::GzDecoder::new(f);
        load_named_parameters(&mut r, params)
    } else {
        let mut r = std::io::BufReader::new(f);
        load_named_parameters(&mut r, params)
    }
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
            Tensor::from_i64(&data, shape, Device::CPU)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorOptions;

    fn make_named_params(sizes: &[(i64, i64)]) -> Vec<(String, Parameter)> {
        sizes.iter().enumerate().map(|(i, &(rows, cols))| {
            let t = Tensor::randn(&[rows, cols], TensorOptions {
                dtype: DType::Float32,
                device: Device::CPU,
            }).unwrap();
            let name = format!("layer_{}/weight", i);
            (name.clone(), Parameter::new(t, "weight"))
        }).collect()
    }

    #[test]
    fn test_named_roundtrip() {
        let params = make_named_params(&[(4, 8), (8, 2)]);

        // Save
        let mut buf = Vec::new();
        save_named_parameters(&mut buf, &params).unwrap();

        // Clone params for loading (fresh init)
        let load_params = make_named_params(&[(4, 8), (8, 2)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_named_parameters(&mut cursor, &load_params).unwrap();

        assert_eq!(report.loaded.len(), 2);
        assert!(report.skipped.is_empty());
        assert!(report.missing.is_empty());

        // Verify data matches
        for ((_, src), (_, dst)) in params.iter().zip(load_params.iter()) {
            let src_data = src.variable.data().to_f32_vec().unwrap();
            let dst_data = dst.variable.data().to_f32_vec().unwrap();
            assert_eq!(src_data, dst_data);
        }
    }

    #[test]
    fn test_named_partial_load() {
        let params_3 = make_named_params(&[(4, 8), (8, 4), (4, 2)]);

        // Save 3 layers
        let mut buf = Vec::new();
        save_named_parameters(&mut buf, &params_3).unwrap();

        // Load into 4 layers (different model — only first 3 names match)
        let mut params_4 = make_named_params(&[(4, 8), (8, 4), (4, 2), (2, 1)]);
        // Rename 4th to something not in checkpoint
        params_4[3].0 = "extra/weight".to_string();

        let before_extra = params_4[3].1.variable.data().to_f32_vec().unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_named_parameters(&mut cursor, &params_4).unwrap();

        assert_eq!(report.loaded.len(), 3);
        assert_eq!(report.missing.len(), 1);
        assert_eq!(report.missing[0], "extra/weight");
        assert!(report.skipped.is_empty());

        // Extra param kept its init values
        let after_extra = params_4[3].1.variable.data().to_f32_vec().unwrap();
        assert_eq!(before_extra, after_extra);
    }

    #[test]
    fn test_named_skipped_checkpoint_params() {
        let params = make_named_params(&[(4, 8), (8, 2)]);

        let mut buf = Vec::new();
        save_named_parameters(&mut buf, &params).unwrap();

        // Load into model with only the first layer
        let model = vec![params[0].clone()];
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_named_parameters(&mut cursor, &model).unwrap();

        assert_eq!(report.loaded.len(), 1);
        assert_eq!(report.skipped.len(), 1); // layer_1/weight is extra in checkpoint
        assert!(report.missing.is_empty());
    }

    #[test]
    fn test_named_shape_mismatch_error() {
        let params = make_named_params(&[(4, 8)]);

        let mut buf = Vec::new();
        save_named_parameters(&mut buf, &params).unwrap();

        // Load into model with same name but different shape
        let wrong_shape = vec![(
            "layer_0/weight".to_string(),
            Parameter::new(
                Tensor::randn(&[4, 4], TensorOptions {
                    dtype: DType::Float32,
                    device: Device::CPU,
                }).unwrap(),
                "weight",
            ),
        )];
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_named_parameters(&mut cursor, &wrong_shape);
        assert!(result.is_err(), "shape mismatch should be an error");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("shape mismatch"), "error should mention shape: {}", err_msg);
    }

    #[test]
    fn test_compressed_roundtrip() {
        let params = make_named_params(&[(16, 32), (32, 8)]);

        let dir = std::env::temp_dir();
        let gz_path = dir.join("test_ckpt.fdl.gz");
        let plain_path = dir.join("test_ckpt.fdl");
        let gz = gz_path.to_str().unwrap();
        let plain = plain_path.to_str().unwrap();

        // Save both compressed and uncompressed
        save_named_parameters_file(gz, &params).unwrap();
        save_named_parameters_file(plain, &params).unwrap();

        // Compressed should be smaller
        let gz_size = std::fs::metadata(gz).unwrap().len();
        let plain_size = std::fs::metadata(plain).unwrap().len();
        assert!(gz_size < plain_size, "gz={} should be < plain={}", gz_size, plain_size);

        // Load from compressed and verify
        let load_params = make_named_params(&[(16, 32), (32, 8)]);
        let report = load_named_parameters_file(gz, &load_params).unwrap();
        assert_eq!(report.loaded.len(), 2);

        for ((_, src), (_, dst)) in params.iter().zip(load_params.iter()) {
            assert_eq!(src.variable.data().to_f32_vec().unwrap(),
                       dst.variable.data().to_f32_vec().unwrap());
        }

        std::fs::remove_file(gz).ok();
        std::fs::remove_file(plain).ok();
    }
}
