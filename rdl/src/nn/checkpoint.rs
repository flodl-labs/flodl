use std::io::{Read, Write};

use crate::tensor::{Device, Result, Tensor, TensorError};

use super::parameter::Parameter;

const MAGIC: [u8; 4] = [b'G', b'O', b'D', b'L'];
const VERSION: u32 = 1;

/// Save parameters to a binary format compatible with goDl.
///
/// Format: GODL magic (4 bytes) + version (u32) + count (u32) + per-param data.
/// All data is stored as little-endian float32 on CPU.
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
/// Loaded tensors are moved to the device the parameter is currently on.
pub fn load_parameters<R: Read>(r: &mut R, params: &[Parameter]) -> Result<()> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(io_err)?;
    if magic != MAGIC {
        return Err(TensorError::new("invalid checkpoint: bad magic"));
    }

    let version = read_u32(r)?;
    if version != VERSION {
        return Err(TensorError::new(&format!(
            "unsupported checkpoint version {} (want {})", version, VERSION
        )));
    }

    let count = read_u32(r)? as usize;
    if count != params.len() {
        return Err(TensorError::new(&format!(
            "parameter count mismatch: checkpoint={} model={}", count, params.len()
        )));
    }

    for (i, p) in params.iter().enumerate() {
        read_param(r, p, i)?;
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
pub(crate) fn write_tensor_state<W: Write>(w: &mut W, t: Option<&Tensor>) -> Result<()> {
    match t {
        None => {
            w.write_all(&[0u8]).map_err(io_err)?;
        }
        Some(t) => {
            w.write_all(&[1u8]).map_err(io_err)?;
            let shape = t.shape();
            w.write_all(&(shape.len() as u32).to_le_bytes()).map_err(io_err)?;
            for &s in &shape {
                w.write_all(&s.to_le_bytes()).map_err(io_err)?;
            }
            let data = t.to_f32_vec()?;
            w.write_all(&(data.len() as u64).to_le_bytes()).map_err(io_err)?;
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            w.write_all(&bytes).map_err(io_err)?;
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

    let ndim = read_u32(r)? as usize;
    let mut shape = vec![0i64; ndim];
    for s in &mut shape {
        *s = read_i64(r)?;
    }
    let count = read_u64(r)? as usize;
    let data = read_f32_vec(r, count)?;
    let t = Tensor::from_f32(&data, &shape, Device::CPU)?;
    if device != Device::CPU {
        Ok(Some(t.to_device(device)?))
    } else {
        Ok(Some(t))
    }
}

// --- Internal helpers ---

fn write_param<W: Write>(w: &mut W, p: &Parameter) -> Result<()> {
    let name = p.name.as_bytes();
    w.write_all(&(name.len() as u32).to_le_bytes()).map_err(io_err)?;
    w.write_all(name).map_err(io_err)?;

    let shape = p.variable.shape();
    w.write_all(&(shape.len() as u32).to_le_bytes()).map_err(io_err)?;
    for &s in &shape {
        w.write_all(&s.to_le_bytes()).map_err(io_err)?;
    }

    let data = p.variable.data().to_f32_vec()?;
    w.write_all(&(data.len() as u64).to_le_bytes()).map_err(io_err)?;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    w.write_all(&bytes).map_err(io_err)?;

    Ok(())
}

fn read_param<R: Read>(r: &mut R, p: &Parameter, index: usize) -> Result<()> {
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
    let dev = p.variable.data().device();
    if dev != Device::CPU {
        p.variable.set_data(t.to_device(dev)?);
    } else {
        p.variable.set_data(t);
    }

    Ok(())
}

fn io_err(e: impl std::fmt::Display) -> TensorError {
    TensorError::new(&format!("io: {}", e))
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
