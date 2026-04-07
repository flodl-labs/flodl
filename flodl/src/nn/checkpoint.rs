use std::io::{Read, Write};

use crate::tensor::{Device, DType, Result, Tensor, TensorError};

use super::buffer::Buffer;
use super::parameter::Parameter;

/// Magic bytes for `.fdl` checkpoint files.
pub(crate) const MAGIC: [u8; 4] = *b"FDLC";
/// Current checkpoint format version.
/// v1 = flodl 0.1.x naming, v2 = flodl 0.2.0+ naming (identical binary layout).
pub(crate) const VERSION: u32 = 2;
/// Maximum checkpoint version we can read.
const MAX_VERSION: u32 = 2;
/// Size of the structural hash field in the checkpoint header.
pub(crate) const HASH_LEN: usize = 32;

/// Report from a checkpoint load: what was loaded, skipped, or missing.
#[derive(Debug, Clone)]
pub struct LoadReport {
    /// Entries matched by name and loaded successfully.
    pub loaded: Vec<String>,
    /// Checkpoint entries with no matching model parameter or buffer (ignored).
    pub skipped: Vec<String>,
    /// Model parameters/buffers with no matching checkpoint entry (kept at init values).
    pub missing: Vec<String>,
}

/// Save parameters and buffers to a binary checkpoint.
///
/// Both params and buffers are stored as named tensors in the same flat list.
/// The format is: `MAGIC(4) | VERSION(u32=1) | hash(32 bytes) | num_entries(u32) | entries...`
///
/// Pass `structural_hash` from `Graph::structural_hash()` to embed architecture
/// identity. Pass `None` to write 32 zero bytes (hash validation skipped on load).
pub fn save_checkpoint<W: Write>(
    w: &mut W,
    params: &[(String, Parameter)],
    buffers: &[(String, Buffer)],
    structural_hash: Option<&str>,
) -> Result<()> {
    w.write_all(&MAGIC).map_err(io_err)?;
    w.write_all(&VERSION.to_le_bytes()).map_err(io_err)?;

    // Write 32-byte hash (or zeros)
    let hash_bytes = match structural_hash {
        Some(hex) => hex_to_bytes(hex)?,
        None => [0u8; HASH_LEN],
    };
    w.write_all(&hash_bytes).map_err(io_err)?;

    let total = (params.len() + buffers.len()) as u32;
    w.write_all(&total.to_le_bytes()).map_err(io_err)?;

    for (name, p) in params {
        let name_bytes = name.as_bytes();
        w.write_all(&(name_bytes.len() as u32).to_le_bytes()).map_err(io_err)?;
        w.write_all(name_bytes).map_err(io_err)?;
        write_tensor_data(w, &p.variable.data())?;
    }

    for (name, b) in buffers {
        let name_bytes = name.as_bytes();
        w.write_all(&(name_bytes.len() as u32).to_le_bytes()).map_err(io_err)?;
        w.write_all(name_bytes).map_err(io_err)?;
        write_tensor_data(w, &b.get())?;
    }

    Ok(())
}

/// Load a checkpoint, matching entries by qualified name against both
/// parameters and buffers.
///
/// Returns a `LoadReport` describing what was matched, skipped, and missing.
/// Shape mismatches on a matched name are errors (not silent skips).
///
/// Pass `structural_hash` from `Graph::structural_hash()` to validate that the
/// checkpoint was saved from the same architecture. Pass `None` to skip validation.
/// If both the file hash and expected hash are non-zero and they differ, returns an error.
pub fn load_checkpoint<R: Read>(
    r: &mut R,
    params: &[(String, Parameter)],
    buffers: &[(String, Buffer)],
    structural_hash: Option<&str>,
) -> Result<LoadReport> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(io_err)?;
    if magic != MAGIC {
        return Err(TensorError::new(
            "invalid checkpoint: bad magic (expected .fdl checkpoint)"
        ));
    }

    let version = read_u32(r)?;
    if version == 0 || version > MAX_VERSION {
        return Err(TensorError::new(&format!(
            "unsupported checkpoint version {} (this build supports 1..={})",
            version, MAX_VERSION,
        )));
    }

    // Read and validate structural hash
    let mut file_hash = [0u8; HASH_LEN];
    r.read_exact(&mut file_hash).map_err(io_err)?;

    let file_nonzero = file_hash.iter().any(|&b| b != 0);
    if let Some(expected_hex) = structural_hash {
        let expected = hex_to_bytes(expected_hex)?;
        let expected_nonzero = expected.iter().any(|&b| b != 0);
        if file_nonzero && expected_nonzero && file_hash != expected {
            return Err(TensorError::new(&format!(
                "checkpoint architecture mismatch: file={} model={}",
                bytes_to_hex(&file_hash),
                expected_hex,
            )));
        }
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

    // Match parameters
    for (name, p) in params {
        if let Some((shape, dtype, raw)) = ckpt.remove(name) {
            let model_shape = p.variable.shape();
            if shape != model_shape {
                return Err(TensorError::new(&format!(
                    "parameter {:?}: shape mismatch: checkpoint={:?} model={:?}",
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

    // Match buffers
    for (name, b) in buffers {
        if let Some((shape, dtype, raw)) = ckpt.remove(name) {
            let model_shape = b.shape();
            if shape != model_shape {
                return Err(TensorError::new(&format!(
                    "buffer {:?}: shape mismatch: checkpoint={:?} model={:?}",
                    name, shape, model_shape
                )));
            }
            let t = tensor_from_raw_bytes(&raw, &shape, dtype)?;
            let model_dtype = b.get().dtype();
            let t = if t.dtype() != model_dtype { t.to_dtype(model_dtype)? } else { t };
            let dev = b.device();
            if dev != Device::CPU {
                b.set(t.to_device(dev)?);
            } else {
                b.set(t);
            }
            loaded.push(name.clone());
        } else {
            missing.push(name.clone());
        }
    }

    let skipped: Vec<String> = ckpt.into_keys().collect();

    Ok(LoadReport { loaded, skipped, missing })
}

/// Save checkpoint to a file path. Uses gzip compression if path ends with `.gz`.
pub fn save_checkpoint_file(
    path: &str,
    params: &[(String, Parameter)],
    buffers: &[(String, Buffer)],
    structural_hash: Option<&str>,
) -> Result<()> {
    let f = std::fs::File::create(path).map_err(io_err)?;
    if path.ends_with(".gz") {
        let mut w = flate2::write::GzEncoder::new(f, flate2::Compression::default());
        save_checkpoint(&mut w, params, buffers, structural_hash)?;
        w.finish().map_err(io_err)?;
        Ok(())
    } else {
        let mut w = std::io::BufWriter::new(f);
        save_checkpoint(&mut w, params, buffers, structural_hash)
    }
}

/// Load checkpoint from a file path. Detects gzip from `.gz` extension.
pub fn load_checkpoint_file(
    path: &str,
    params: &[(String, Parameter)],
    buffers: &[(String, Buffer)],
    structural_hash: Option<&str>,
) -> Result<LoadReport> {
    let f = std::fs::File::open(path).map_err(io_err)?;
    if path.ends_with(".gz") {
        let mut r = flate2::read::GzDecoder::new(f);
        load_checkpoint(&mut r, params, buffers, structural_hash)
    } else {
        let mut r = std::io::BufReader::new(f);
        load_checkpoint(&mut r, params, buffers, structural_hash)
    }
}

/// Peek at the version number of a checkpoint file without reading the full contents.
///
/// Returns the version field (1 for flodl 0.1.x, 2 for flodl 0.2.0+).
/// Useful to decide whether a checkpoint needs migration before loading.
pub fn checkpoint_version(path: &str) -> Result<u32> {
    let f = std::fs::File::open(path).map_err(io_err)?;
    let mut r: Box<dyn Read> = if path.ends_with(".gz") {
        Box::new(flate2::read::GzDecoder::new(f))
    } else {
        Box::new(std::io::BufReader::new(f))
    };
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(io_err)?;
    if magic != MAGIC {
        return Err(TensorError::new(
            "invalid checkpoint: bad magic (expected .fdl checkpoint)"
        ));
    }
    read_u32(&mut r)
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
pub(crate) fn write_tensor_data<W: Write>(w: &mut W, t: &Tensor) -> Result<()> {
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
            // For f16/bf16/i32: load raw bytes via from_blob directly.
            let mut shape_v = shape.to_vec();
            let mut handle: flodl_sys::FlodlTensor = std::ptr::null_mut();
            let (dev_type, dev_idx) = crate::tensor::Device::CPU.to_ffi();
            let err = unsafe {
                flodl_sys::flodl_from_blob(
                    raw.as_ptr() as *mut std::ffi::c_void,
                    shape_v.as_mut_ptr(),
                    shape_v.len() as i32,
                    dtype as i32,
                    dev_type, dev_idx,
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

// --- Checkpoint migration ---

/// Report from a checkpoint migration.
#[derive(Debug, Clone)]
pub struct MigrateReport {
    /// Entries that kept their original name (exact match in old and new model).
    pub unchanged: Vec<String>,
    /// Entries remapped by shape+dtype matching: `(old_name, new_name)`.
    pub remapped: Vec<(String, String)>,
    /// Checkpoint entries with no matching model parameter/buffer (not migrated).
    pub dropped: Vec<String>,
    /// Model parameters/buffers with no matching checkpoint entry (will use init values).
    pub missing: Vec<String>,
}

impl MigrateReport {
    /// True if every checkpoint entry was matched (nothing dropped or missing).
    pub fn is_complete(&self) -> bool {
        self.dropped.is_empty() && self.missing.is_empty()
    }
}

impl std::fmt::Display for MigrateReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.unchanged.is_empty() {
            writeln!(f, "unchanged ({}):", self.unchanged.len())?;
            for name in &self.unchanged { writeln!(f, "  {}", name)?; }
        }
        if !self.remapped.is_empty() {
            writeln!(f, "remapped ({}):", self.remapped.len())?;
            for (old, new) in &self.remapped { writeln!(f, "  {} -> {}", old, new)?; }
        }
        if !self.dropped.is_empty() {
            writeln!(f, "dropped ({}):", self.dropped.len())?;
            for name in &self.dropped { writeln!(f, "  {}", name)?; }
        }
        if !self.missing.is_empty() {
            writeln!(f, "missing ({}):", self.missing.len())?;
            for name in &self.missing { writeln!(f, "  {}", name)?; }
        }
        Ok(())
    }
}

/// Raw checkpoint entry for migration (not loaded into a live Tensor).
struct RawEntry {
    name: String,
    shape: Vec<i64>,
    dtype: DType,
    raw: Vec<u8>,
}

/// Read checkpoint header and all raw entries without constructing tensors.
fn read_raw_checkpoint<R: Read>(r: &mut R) -> Result<Vec<RawEntry>> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(io_err)?;
    if magic != MAGIC {
        return Err(TensorError::new(
            "invalid checkpoint: bad magic (expected .fdl checkpoint)"
        ));
    }
    let version = read_u32(r)?;
    if version == 0 || version > MAX_VERSION {
        return Err(TensorError::new(&format!(
            "unsupported checkpoint version {} (this build supports 1..={})",
            version, MAX_VERSION,
        )));
    }
    // Skip structural hash
    let mut _hash = [0u8; HASH_LEN];
    r.read_exact(&mut _hash).map_err(io_err)?;

    let count = read_u32(r)? as usize;
    let mut entries = Vec::with_capacity(count);

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

        entries.push(RawEntry { name, shape, dtype, raw });
    }

    Ok(entries)
}

/// Write a single raw entry (name + tensor data) into a checkpoint stream.
fn write_raw_entry<W: Write>(w: &mut W, name: &str, e: &RawEntry) -> Result<()> {
    let name_bytes = name.as_bytes();
    w.write_all(&(name_bytes.len() as u32).to_le_bytes()).map_err(io_err)?;
    w.write_all(name_bytes).map_err(io_err)?;
    w.write_all(&(e.shape.len() as u32).to_le_bytes()).map_err(io_err)?;
    for &s in &e.shape {
        w.write_all(&s.to_le_bytes()).map_err(io_err)?;
    }
    w.write_all(&[dtype_tag(e.dtype)]).map_err(io_err)?;
    w.write_all(&(e.raw.len() as u64).to_le_bytes()).map_err(io_err)?;
    w.write_all(&e.raw).map_err(io_err)?;
    Ok(())
}

/// Migrate a checkpoint to match a model's current parameter and buffer naming.
///
/// Reads the source checkpoint and matches each entry against the model's
/// `named_parameters` and `named_buffers`:
///
/// 1. **Exact name match** — entries whose name and shape match a model target
///    are passed through unchanged.
/// 2. **Shape+dtype match** — remaining entries are matched to remaining model
///    targets by shape and dtype, in checkpoint order. This handles the common
///    case where only tag/node prefixes changed between versions.
///
/// The migrated checkpoint is written with a zeroed structural hash so it can
/// be loaded without architecture validation.
///
/// # Example
///
/// ```ignore
/// let graph = FlowBuilder::from(input)
///     .through(encoder).tag("encoder")
///     .build()?;
///
/// let report = migrate_checkpoint(
///     &mut src_reader,
///     &mut dst_writer,
///     &graph.named_parameters(),
///     &graph.named_buffers(),
/// )?;
/// println!("{}", report);
/// ```
pub fn migrate_checkpoint<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    params: &[(String, Parameter)],
    buffers: &[(String, Buffer)],
) -> Result<MigrateReport> {
    let entries = read_raw_checkpoint(r)?;

    // Build model expectations in order: params then buffers
    let mut targets: Vec<(String, Vec<i64>, DType)> = Vec::with_capacity(
        params.len() + buffers.len()
    );
    for (name, p) in params {
        targets.push((name.clone(), p.variable.shape(), p.variable.data().dtype()));
    }
    for (name, b) in buffers {
        targets.push((name.clone(), b.shape(), b.get().dtype()));
    }

    let mut unchanged = Vec::new();
    let mut remapped = Vec::new();
    let mut missing = Vec::new();
    let mut used = vec![false; entries.len()];

    // output: (new_name, checkpoint_index) in model order
    let mut output: Vec<(String, usize)> = Vec::new();

    // Index checkpoint entries by name for O(1) exact lookup
    let name_index: std::collections::HashMap<&str, usize> =
        entries.iter().enumerate().map(|(i, e)| (e.name.as_str(), i)).collect();

    // Indices of model targets not yet matched
    let mut unmatched: Vec<usize> = Vec::new();

    // Pass 1: exact name + shape match
    for (mi, (name, shape, _)) in targets.iter().enumerate() {
        if let Some(&ci) = name_index.get(name.as_str()) {
            if !used[ci] && entries[ci].shape == *shape {
                unchanged.push(name.clone());
                used[ci] = true;
                output.push((name.clone(), ci));
                continue;
            }
        }
        unmatched.push(mi);
    }

    // Pass 2: shape+dtype matching in checkpoint order
    for &mi in &unmatched {
        let (name, shape, dtype) = &targets[mi];

        let found = entries.iter().enumerate()
            .find(|(ci, e)| !used[*ci] && e.shape == *shape && e.dtype == *dtype)
            .map(|(ci, _)| ci);

        if let Some(ci) = found {
            remapped.push((entries[ci].name.clone(), name.clone()));
            used[ci] = true;
            output.push((name.clone(), ci));
        } else {
            missing.push(name.clone());
        }
    }

    let dropped: Vec<String> = entries.iter().enumerate()
        .filter(|(i, _)| !used[*i])
        .map(|(_, e)| e.name.clone())
        .collect();

    // Write migrated checkpoint with zeroed structural hash
    w.write_all(&MAGIC).map_err(io_err)?;
    w.write_all(&VERSION.to_le_bytes()).map_err(io_err)?;
    w.write_all(&[0u8; HASH_LEN]).map_err(io_err)?;
    w.write_all(&(output.len() as u32).to_le_bytes()).map_err(io_err)?;

    for (name, ci) in &output {
        write_raw_entry(w, name, &entries[*ci])?;
    }

    Ok(MigrateReport { unchanged, remapped, dropped, missing })
}

/// Migrate a checkpoint file. Detects gzip from `.gz` extension on both paths.
///
/// Source and destination must be different paths.
pub fn migrate_checkpoint_file(
    src: &str,
    dst: &str,
    params: &[(String, Parameter)],
    buffers: &[(String, Buffer)],
) -> Result<MigrateReport> {
    let sf = std::fs::File::open(src).map_err(io_err)?;
    let df = std::fs::File::create(dst).map_err(io_err)?;

    match (src.ends_with(".gz"), dst.ends_with(".gz")) {
        (true, true) => {
            let mut r = flate2::read::GzDecoder::new(sf);
            let mut w = flate2::write::GzEncoder::new(df, flate2::Compression::default());
            let report = migrate_checkpoint(&mut r, &mut w, params, buffers)?;
            w.finish().map_err(io_err)?;
            Ok(report)
        }
        (true, false) => {
            let mut r = flate2::read::GzDecoder::new(sf);
            let mut w = std::io::BufWriter::new(df);
            migrate_checkpoint(&mut r, &mut w, params, buffers)
        }
        (false, true) => {
            let mut r = std::io::BufReader::new(sf);
            let mut w = flate2::write::GzEncoder::new(df, flate2::Compression::default());
            let report = migrate_checkpoint(&mut r, &mut w, params, buffers)?;
            w.finish().map_err(io_err)?;
            Ok(report)
        }
        (false, false) => {
            let mut r = std::io::BufReader::new(sf);
            let mut w = std::io::BufWriter::new(df);
            migrate_checkpoint(&mut r, &mut w, params, buffers)
        }
    }
}

// --- Shared helpers ---

pub(crate) fn io_err(e: impl std::fmt::Display) -> TensorError {
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

/// Decode a hex string to a 32-byte array.
fn hex_to_bytes(hex: &str) -> Result<[u8; HASH_LEN]> {
    if hex.len() != HASH_LEN * 2 {
        return Err(TensorError::new(&format!(
            "expected {} hex chars, got {}",
            HASH_LEN * 2,
            hex.len()
        )));
    }
    let mut out = [0u8; HASH_LEN];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(TensorError::new(&format!("invalid hex byte: {}", b))),
    }
}

/// Encode a byte slice as a lowercase hex string.
fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        use std::fmt::Write;
        let _ = write!(s, "{:02x}", b);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorOptions;

    fn make_named_params(sizes: &[(i64, i64)]) -> Vec<(String, Parameter)> {
        sizes.iter().enumerate().map(|(i, &(rows, cols))| {
            let t = Tensor::randn(&[rows, cols], TensorOptions {
                dtype: DType::Float32,
                device: crate::tensor::test_device(),
            }).unwrap();
            let name = format!("layer_{}/weight", i);
            (name.clone(), Parameter::new(t, "weight"))
        }).collect()
    }

    fn make_named_buffers(sizes: &[i64]) -> Vec<(String, Buffer)> {
        sizes.iter().enumerate().map(|(i, &features)| {
            let t = Tensor::randn(&[features], TensorOptions {
                dtype: DType::Float32,
                device: crate::tensor::test_device(),
            }).unwrap();
            let name = format!("bn_{}/running_mean", i);
            (name.clone(), Buffer::new(t, "running_mean"))
        }).collect()
    }

    #[test]
    fn test_named_roundtrip() {
        let params = make_named_params(&[(4, 8), (8, 2)]);

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], None).unwrap();

        let load_params = make_named_params(&[(4, 8), (8, 2)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &load_params, &[], None).unwrap();

        assert_eq!(report.loaded.len(), 2);
        assert!(report.skipped.is_empty());
        assert!(report.missing.is_empty());

        for ((_, src), (_, dst)) in params.iter().zip(load_params.iter()) {
            let src_data = src.variable.data().to_f32_vec().unwrap();
            let dst_data = dst.variable.data().to_f32_vec().unwrap();
            assert_eq!(src_data, dst_data);
        }
    }

    #[test]
    fn test_buffer_roundtrip() {
        let params = make_named_params(&[(4, 8)]);
        let buffers = make_named_buffers(&[8]);

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &buffers, None).unwrap();

        // Fresh model with same structure
        let load_params = make_named_params(&[(4, 8)]);
        let load_buffers = make_named_buffers(&[8]);
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &load_params, &load_buffers, None).unwrap();

        assert_eq!(report.loaded.len(), 2); // 1 param + 1 buffer
        assert!(report.skipped.is_empty());
        assert!(report.missing.is_empty());

        // Verify buffer data matches
        let src_data = buffers[0].1.get().to_f32_vec().unwrap();
        let dst_data = load_buffers[0].1.get().to_f32_vec().unwrap();
        assert_eq!(src_data, dst_data);
    }

    #[test]
    fn test_named_partial_load() {
        let params_3 = make_named_params(&[(4, 8), (8, 4), (4, 2)]);

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params_3, &[], None).unwrap();

        let mut params_4 = make_named_params(&[(4, 8), (8, 4), (4, 2), (2, 1)]);
        params_4[3].0 = "extra/weight".to_string();

        let before_extra = params_4[3].1.variable.data().to_f32_vec().unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &params_4, &[], None).unwrap();

        assert_eq!(report.loaded.len(), 3);
        assert_eq!(report.missing.len(), 1);
        assert_eq!(report.missing[0], "extra/weight");
        assert!(report.skipped.is_empty());

        let after_extra = params_4[3].1.variable.data().to_f32_vec().unwrap();
        assert_eq!(before_extra, after_extra);
    }

    #[test]
    fn test_named_skipped_checkpoint_params() {
        let params = make_named_params(&[(4, 8), (8, 2)]);

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], None).unwrap();

        let model = vec![params[0].clone()];
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &model, &[], None).unwrap();

        assert_eq!(report.loaded.len(), 1);
        assert_eq!(report.skipped.len(), 1);
        assert!(report.missing.is_empty());
    }

    #[test]
    fn test_named_shape_mismatch_error() {
        let params = make_named_params(&[(4, 8)]);

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], None).unwrap();

        let wrong_shape = vec![(
            "layer_0/weight".to_string(),
            Parameter::new(
                Tensor::randn(&[4, 4], TensorOptions {
                    dtype: DType::Float32,
                    device: crate::tensor::test_device(),
                }).unwrap(),
                "weight",
            ),
        )];
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &wrong_shape, &[], None);
        assert!(result.is_err(), "shape mismatch should be an error");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("shape mismatch"), "error should mention shape: {}", err_msg);
    }

    #[test]
    fn test_buffer_shape_mismatch_error() {
        let buffers = make_named_buffers(&[8]);

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &[], &buffers, None).unwrap();

        let wrong_buffers = vec![(
            "bn_0/running_mean".to_string(),
            Buffer::new(
                Tensor::zeros(&[4], crate::tensor::test_opts()).unwrap(),
                "running_mean",
            ),
        )];
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &[], &wrong_buffers, None);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("shape mismatch"));
    }

    #[test]
    fn test_compressed_roundtrip() {
        let params = make_named_params(&[(16, 32), (32, 8)]);
        let buffers = make_named_buffers(&[32]);

        let dir = std::env::temp_dir();
        let gz_path = dir.join("test_ckpt_v2.fdl.gz");
        let plain_path = dir.join("test_ckpt_v2.fdl");
        let gz = gz_path.to_str().unwrap();
        let plain = plain_path.to_str().unwrap();

        save_checkpoint_file(gz, &params, &buffers, None).unwrap();
        save_checkpoint_file(plain, &params, &buffers, None).unwrap();

        // Compressed should be smaller
        let gz_size = std::fs::metadata(gz).unwrap().len();
        let plain_size = std::fs::metadata(plain).unwrap().len();
        assert!(gz_size < plain_size, "gz={} should be < plain={}", gz_size, plain_size);

        // Load from compressed and verify
        let load_params = make_named_params(&[(16, 32), (32, 8)]);
        let load_buffers = make_named_buffers(&[32]);
        let report = load_checkpoint_file(gz, &load_params, &load_buffers, None).unwrap();
        assert_eq!(report.loaded.len(), 3); // 2 params + 1 buffer

        for ((_, src), (_, dst)) in params.iter().zip(load_params.iter()) {
            assert_eq!(src.variable.data().to_f32_vec().unwrap(),
                       dst.variable.data().to_f32_vec().unwrap());
        }

        let src_buf = buffers[0].1.get().to_f32_vec().unwrap();
        let dst_buf = load_buffers[0].1.get().to_f32_vec().unwrap();
        assert_eq!(src_buf, dst_buf);

        std::fs::remove_file(gz).ok();
        std::fs::remove_file(plain).ok();
    }

    #[test]
    fn test_hash_roundtrip() {
        let params = make_named_params(&[(4, 8)]);
        // Use a known 64-char hex hash
        let hash = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6a7b8c9d0e1f2a3b4c5d6a7b8c9d0e1f2";

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], Some(hash)).unwrap();

        let load_params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        // Same hash: should succeed
        let report = load_checkpoint(&mut cursor, &load_params, &[], Some(hash)).unwrap();
        assert_eq!(report.loaded.len(), 1);
    }

    #[test]
    fn test_hash_mismatch_error() {
        let params = make_named_params(&[(4, 8)]);
        let hash_a = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6a7b8c9d0e1f2a3b4c5d6a7b8c9d0e1f2";
        let hash_b = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], Some(hash_a)).unwrap();

        let load_params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &load_params, &[], Some(hash_b));
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("architecture mismatch"), "error: {}", msg);
    }

    #[test]
    fn test_zero_hash_skips_validation() {
        let params = make_named_params(&[(4, 8)]);

        // Save with no hash (zero bytes)
        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], None).unwrap();

        // Load with a hash expectation — should still succeed (file has zeros)
        let hash = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
        let load_params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &load_params, &[], Some(hash)).unwrap();
        assert_eq!(report.loaded.len(), 1);

        // Save with hash, load with None — should succeed (no expected hash)
        let mut buf2 = Vec::new();
        save_checkpoint(&mut buf2, &params, &[], Some(hash)).unwrap();
        let load_params2 = make_named_params(&[(4, 8)]);
        let mut cursor2 = std::io::Cursor::new(&buf2);
        let report2 = load_checkpoint(&mut cursor2, &load_params2, &[], None).unwrap();
        assert_eq!(report2.loaded.len(), 1);
    }

    /// Write a checkpoint with an explicit version byte (for testing v1 migration).
    fn save_checkpoint_versioned<W: std::io::Write>(
        w: &mut W,
        version: u32,
        params: &[(String, Parameter)],
        buffers: &[(String, Buffer)],
    ) {
        w.write_all(&MAGIC).unwrap();
        w.write_all(&version.to_le_bytes()).unwrap();
        w.write_all(&[0u8; HASH_LEN]).unwrap();
        let total = (params.len() + buffers.len()) as u32;
        w.write_all(&total.to_le_bytes()).unwrap();
        for (name, p) in params {
            let name_bytes = name.as_bytes();
            w.write_all(&(name_bytes.len() as u32).to_le_bytes()).unwrap();
            w.write_all(name_bytes).unwrap();
            write_tensor_data(w, &p.variable.data()).unwrap();
        }
        for (name, b) in buffers {
            let name_bytes = name.as_bytes();
            w.write_all(&(name_bytes.len() as u32).to_le_bytes()).unwrap();
            w.write_all(name_bytes).unwrap();
            write_tensor_data(w, &b.get()).unwrap();
        }
    }

    #[test]
    fn test_migrate_all_renamed() {
        // Simulate v1 checkpoint with old-style names
        let old_params = vec![
            ("linear_0/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("linear_1/weight".to_string(), Parameter::new(
                Tensor::randn(&[8, 2], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &old_params, &[]);

        // New model with renamed tags
        let new_params = vec![
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("decoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[8, 2], crate::tensor::test_opts()).unwrap(), "weight")),
        ];

        let mut out = Vec::new();
        let report = migrate_checkpoint(
            &mut std::io::Cursor::new(&ckpt), &mut out,
            &new_params, &[],
        ).unwrap();

        assert!(report.unchanged.is_empty());
        assert_eq!(report.remapped.len(), 2);
        assert!(report.dropped.is_empty());
        assert!(report.missing.is_empty());
        assert!(report.is_complete());

        // Verify the migrated checkpoint loads correctly
        let verify_params = vec![
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("decoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[8, 2], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut cursor = std::io::Cursor::new(&out);
        let load_report = load_checkpoint(&mut cursor, &verify_params, &[], None).unwrap();
        assert_eq!(load_report.loaded.len(), 2);
        assert!(load_report.missing.is_empty());

        // Verify data preserved: old param data matches loaded data
        for (i, (_, vp)) in verify_params.iter().enumerate() {
            let expected = old_params[i].1.variable.data().to_f32_vec().unwrap();
            let got = vp.variable.data().to_f32_vec().unwrap();
            assert_eq!(expected, got, "data mismatch for param {}", i);
        }
    }

    #[test]
    fn test_migrate_partial_rename() {
        // Some names match, some don't
        let old_params = vec![
            ("shared/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("linear_0/weight".to_string(), Parameter::new(
                Tensor::randn(&[8, 2], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &old_params, &[]);

        let new_params = vec![
            ("shared/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[8, 2], crate::tensor::test_opts()).unwrap(), "weight")),
        ];

        let mut out = Vec::new();
        let report = migrate_checkpoint(
            &mut std::io::Cursor::new(&ckpt), &mut out,
            &new_params, &[],
        ).unwrap();

        assert_eq!(report.unchanged, vec!["shared/weight"]);
        assert_eq!(report.remapped.len(), 1);
        assert_eq!(report.remapped[0], ("linear_0/weight".to_string(), "encoder/weight".to_string()));
        assert!(report.is_complete());
    }

    #[test]
    fn test_migrate_with_buffers() {
        let old_params = vec![
            ("linear_0/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let old_buffers = vec![
            ("bn_0/running_mean".to_string(), Buffer::new(
                Tensor::zeros(&[8], crate::tensor::test_opts()).unwrap(), "running_mean")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &old_params, &old_buffers);

        let new_params = vec![
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let new_buffers = vec![
            ("norm/running_mean".to_string(), Buffer::new(
                Tensor::zeros(&[8], crate::tensor::test_opts()).unwrap(), "running_mean")),
        ];

        let mut out = Vec::new();
        let report = migrate_checkpoint(
            &mut std::io::Cursor::new(&ckpt), &mut out,
            &new_params, &new_buffers,
        ).unwrap();

        assert_eq!(report.remapped.len(), 2);
        assert!(report.is_complete());

        // Verify migrated checkpoint loads with new names
        let vp = vec![
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let vb = vec![
            ("norm/running_mean".to_string(), Buffer::new(
                Tensor::zeros(&[8], crate::tensor::test_opts()).unwrap(), "running_mean")),
        ];
        let mut cursor = std::io::Cursor::new(&out);
        let load_report = load_checkpoint(&mut cursor, &vp, &vb, None).unwrap();
        assert_eq!(load_report.loaded.len(), 2);
    }

    #[test]
    fn test_migrate_dropped_and_missing() {
        let old_params = vec![
            ("old/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("removed/weight".to_string(), Parameter::new(
                Tensor::randn(&[16, 16], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &old_params, &[]);

        // New model: one matching shape, one entirely new
        let new_params = vec![
            ("new/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
            ("added/weight".to_string(), Parameter::new(
                Tensor::randn(&[32, 32], crate::tensor::test_opts()).unwrap(), "weight")),
        ];

        let mut out = Vec::new();
        let report = migrate_checkpoint(
            &mut std::io::Cursor::new(&ckpt), &mut out,
            &new_params, &[],
        ).unwrap();

        assert_eq!(report.remapped.len(), 1);
        assert_eq!(report.dropped, vec!["removed/weight"]);
        assert_eq!(report.missing, vec!["added/weight"]);
        assert!(!report.is_complete());
    }

    #[test]
    fn test_migrate_positional_disambiguation() {
        // Two params with identical shape — must match by position
        let old_params = vec![
            ("linear_0/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 4], crate::tensor::test_opts()).unwrap(), "weight")),
            ("linear_1/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 4], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &old_params, &[]);

        let new_params = vec![
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 4], crate::tensor::test_opts()).unwrap(), "weight")),
            ("decoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 4], crate::tensor::test_opts()).unwrap(), "weight")),
        ];

        let mut out = Vec::new();
        let report = migrate_checkpoint(
            &mut std::io::Cursor::new(&ckpt), &mut out,
            &new_params, &[],
        ).unwrap();

        assert_eq!(report.remapped.len(), 2);
        // Positional: first old → first new, second old → second new
        assert_eq!(report.remapped[0].0, "linear_0/weight");
        assert_eq!(report.remapped[0].1, "encoder/weight");
        assert_eq!(report.remapped[1].0, "linear_1/weight");
        assert_eq!(report.remapped[1].1, "decoder/weight");

        // Verify correct data assignment
        let vp = vec![
            ("encoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 4], crate::tensor::test_opts()).unwrap(), "weight")),
            ("decoder/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 4], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut cursor = std::io::Cursor::new(&out);
        load_checkpoint(&mut cursor, &vp, &[], None).unwrap();

        // encoder/weight should have linear_0's data, decoder/weight should have linear_1's data
        let enc_data = vp[0].1.variable.data().to_f32_vec().unwrap();
        let dec_data = vp[1].1.variable.data().to_f32_vec().unwrap();
        let old_0 = old_params[0].1.variable.data().to_f32_vec().unwrap();
        let old_1 = old_params[1].1.variable.data().to_f32_vec().unwrap();
        assert_eq!(enc_data, old_0);
        assert_eq!(dec_data, old_1);
    }

    #[test]
    fn test_migrate_v1_writes_v2() {
        let old_params = vec![
            ("x/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &old_params, &[]);

        // Confirm source is v1
        let mut peek = std::io::Cursor::new(&ckpt);
        let mut magic = [0u8; 4];
        std::io::Read::read_exact(&mut peek, &mut magic).unwrap();
        let mut vbuf = [0u8; 4];
        std::io::Read::read_exact(&mut peek, &mut vbuf).unwrap();
        assert_eq!(u32::from_le_bytes(vbuf), 1);

        let new_params = vec![
            ("y/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];

        let mut out = Vec::new();
        migrate_checkpoint(
            &mut std::io::Cursor::new(&ckpt), &mut out,
            &new_params, &[],
        ).unwrap();

        // Confirm output is v2
        let mut peek2 = std::io::Cursor::new(&out);
        std::io::Read::read_exact(&mut peek2, &mut magic).unwrap();
        assert_eq!(&magic, b"FDLC");
        std::io::Read::read_exact(&mut peek2, &mut vbuf).unwrap();
        assert_eq!(u32::from_le_bytes(vbuf), VERSION); // should be 2
    }

    #[test]
    fn test_migrate_file_roundtrip() {
        let old_params = vec![
            ("old/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let dir = std::env::temp_dir();
        let src = dir.join("test_migrate_src.fdl");
        let dst = dir.join("test_migrate_dst.fdl");

        // Write v1 checkpoint to file
        {
            let f = std::fs::File::create(&src).unwrap();
            let mut w = std::io::BufWriter::new(f);
            save_checkpoint_versioned(&mut w, 1, &old_params, &[]);
        }

        let new_params = vec![
            ("new/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];

        let report = migrate_checkpoint_file(
            src.to_str().unwrap(),
            dst.to_str().unwrap(),
            &new_params, &[],
        ).unwrap();
        assert_eq!(report.remapped.len(), 1);
        assert!(report.is_complete());

        // Load migrated file
        let vp = vec![
            ("new/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let load_report = load_checkpoint_file(
            dst.to_str().unwrap(), &vp, &[], None,
        ).unwrap();
        assert_eq!(load_report.loaded.len(), 1);

        // Verify data preserved
        let expected = old_params[0].1.variable.data().to_f32_vec().unwrap();
        let got = vp[0].1.variable.data().to_f32_vec().unwrap();
        assert_eq!(expected, got);

        std::fs::remove_file(src).ok();
        std::fs::remove_file(dst).ok();
    }

    #[test]
    fn test_migrate_display() {
        let report = MigrateReport {
            unchanged: vec!["shared/weight".to_string()],
            remapped: vec![("old/bias".to_string(), "new/bias".to_string())],
            dropped: vec!["removed/weight".to_string()],
            missing: vec!["added/weight".to_string()],
        };
        let text = format!("{}", report);
        assert!(text.contains("unchanged (1)"));
        assert!(text.contains("remapped (1)"));
        assert!(text.contains("old/bias -> new/bias"));
        assert!(text.contains("dropped (1)"));
        assert!(text.contains("missing (1)"));
    }

    #[test]
    fn test_checkpoint_version_peek() {
        let params = make_named_params(&[(4, 8)]);
        let dir = std::env::temp_dir();
        let path = dir.join("test_version_peek.fdl");
        save_checkpoint_file(path.to_str().unwrap(), &params, &[], None).unwrap();

        let v = checkpoint_version(path.to_str().unwrap()).unwrap();
        assert_eq!(v, VERSION);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_accepts_v1() {
        // v1 checkpoints must still load in v2 builds
        let params = vec![
            ("x/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut ckpt = Vec::new();
        save_checkpoint_versioned(&mut ckpt, 1, &params, &[]);

        let load_params = vec![
            ("x/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut cursor = std::io::Cursor::new(&ckpt);
        let report = load_checkpoint(&mut cursor, &load_params, &[], None).unwrap();
        assert_eq!(report.loaded.len(), 1);

        let expected = params[0].1.variable.data().to_f32_vec().unwrap();
        let got = load_params[0].1.variable.data().to_f32_vec().unwrap();
        assert_eq!(expected, got);
    }

    // --- Edge case / corruption tests ---

    #[test]
    fn test_truncated_checkpoint_header_only() {
        // Write valid header but truncate before any entry data
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&[0u8; HASH_LEN]);
        // Claim 5 entries, but provide none
        buf.extend_from_slice(&5u32.to_le_bytes());

        let params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &params, &[], None);
        assert!(result.is_err(), "truncated checkpoint should return Err, not panic");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("io:"), "should be an IO error: {}", msg);
    }

    #[test]
    fn test_truncated_checkpoint_mid_entry() {
        // Save a valid checkpoint, then truncate in the middle of the first entry
        let params = make_named_params(&[(4, 8)]);
        let mut full = Vec::new();
        save_checkpoint(&mut full, &params, &[], None).unwrap();

        // Header = 4 (magic) + 4 (version) + 32 (hash) + 4 (count) = 44
        // Truncate partway through the first entry (e.g., keep only 50 bytes)
        let truncated = full[..50.min(full.len())].to_vec();

        let load_params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&truncated);
        let result = load_checkpoint(&mut cursor, &load_params, &[], None);
        assert!(result.is_err(), "truncated mid-entry should return Err");
    }

    #[test]
    fn test_empty_file() {
        // Zero bytes: read_exact for magic should fail
        let buf: Vec<u8> = Vec::new();
        let params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &params, &[], None);
        assert!(result.is_err(), "empty file should return Err");
    }

    #[test]
    fn test_invalid_magic_bytes() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JUNK"); // wrong magic
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&[0u8; HASH_LEN]);
        buf.extend_from_slice(&0u32.to_le_bytes());

        let params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &params, &[], None);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("bad magic"), "error should mention bad magic: {}", msg);
    }

    #[test]
    fn test_invalid_magic_checkpoint_version() {
        // checkpoint_version() should also reject bad magic
        let dir = std::env::temp_dir();
        let path = dir.join("test_bad_magic_version.fdl");
        std::fs::write(&path, b"NOT_FDLC_data").unwrap();

        let result = checkpoint_version(path.to_str().unwrap());
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("bad magic"), "error: {}", msg);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_unsupported_version_high() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&99u32.to_le_bytes()); // version 99
        buf.extend_from_slice(&[0u8; HASH_LEN]);
        buf.extend_from_slice(&0u32.to_le_bytes());

        let params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &params, &[], None);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("unsupported checkpoint version"), "error: {}", msg);
        assert!(msg.contains("99"), "should mention version 99: {}", msg);
    }

    #[test]
    fn test_unsupported_version_zero() {
        // Version 0 is also rejected (valid range is 1..=MAX_VERSION)
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&0u32.to_le_bytes()); // version 0
        buf.extend_from_slice(&[0u8; HASH_LEN]);
        buf.extend_from_slice(&0u32.to_le_bytes());

        let params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &params, &[], None);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("unsupported checkpoint version"), "error: {}", msg);
    }

    #[test]
    fn test_hash_mismatch_both_nonzero() {
        // Both file and expected have nonzero hashes that differ
        let params = make_named_params(&[(4, 8)]);
        let hash_a = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let hash_b = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], Some(hash_a)).unwrap();

        let load_params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &load_params, &[], Some(hash_b));
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("architecture mismatch"), "error: {}", msg);
        // Error message should include both hashes for diagnostics
        assert!(msg.contains(hash_b), "should show expected hash: {}", msg);
    }

    #[test]
    fn test_zero_entries_empty_model() {
        // Save a checkpoint with no parameters and no buffers
        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &[], &[], None).unwrap();

        // Load into an empty model
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &[], &[], None).unwrap();
        assert!(report.loaded.is_empty());
        assert!(report.skipped.is_empty());
        assert!(report.missing.is_empty());
    }

    #[test]
    fn test_zero_entries_nonempty_model() {
        // Save empty checkpoint, load into model that expects params
        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &[], &[], None).unwrap();

        let load_params = make_named_params(&[(4, 8)]);
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &load_params, &[], None).unwrap();
        assert!(report.loaded.is_empty());
        assert!(report.skipped.is_empty());
        assert_eq!(report.missing.len(), 1, "model param should be reported as missing");
    }

    #[test]
    fn test_shape_mismatch_transposed() {
        // Save [4, 8], try to load into [8, 4] (transposed, same numel)
        let params = vec![
            ("layer/weight".to_string(), Parameter::new(
                Tensor::randn(&[4, 8], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &params, &[], None).unwrap();

        let wrong_params = vec![
            ("layer/weight".to_string(), Parameter::new(
                Tensor::randn(&[8, 4], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &wrong_params, &[], None);
        assert!(result.is_err(), "transposed shape should be a mismatch error");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("shape mismatch"), "error: {}", msg);
        assert!(msg.contains("[4, 8]"), "should show checkpoint shape: {}", msg);
        assert!(msg.contains("[8, 4]"), "should show model shape: {}", msg);
    }

    #[test]
    fn test_dtype_mismatch_auto_cast() {
        // Save as f32, load into f64 parameter. The code does to_dtype() automatically.
        let f32_param = vec![
            ("layer/weight".to_string(), Parameter::new(
                Tensor::ones(&[2, 3], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &f32_param, &[], None).unwrap();

        // Create f64 parameter with same shape
        let f64_param = vec![
            ("layer/weight".to_string(), Parameter::new(
                Tensor::zeros(&[2, 3], TensorOptions {
                    dtype: DType::Float64,
                    device: crate::tensor::test_device(),
                }).unwrap(), "weight")),
        ];
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &f64_param, &[], None).unwrap();
        assert_eq!(report.loaded.len(), 1, "dtype auto-cast should succeed");

        // Verify the loaded data is correct and in f64
        let loaded = f64_param[0].1.variable.data();
        assert_eq!(loaded.dtype(), DType::Float64);
        let vals = loaded.to_f64_vec().unwrap();
        for v in vals {
            assert!((v - 1.0).abs() < 1e-6, "expected ~1.0, got {}", v);
        }
    }

    #[test]
    fn test_dtype_mismatch_buffer_auto_cast() {
        // Same auto-cast test for buffers
        let f32_buffers = vec![
            ("norm/running_mean".to_string(), Buffer::new(
                Tensor::ones(&[8], crate::tensor::test_opts()).unwrap(), "running_mean")),
        ];
        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &[], &f32_buffers, None).unwrap();

        let f64_buffers = vec![
            ("norm/running_mean".to_string(), Buffer::new(
                Tensor::zeros(&[8], TensorOptions {
                    dtype: DType::Float64,
                    device: crate::tensor::test_device(),
                }).unwrap(), "running_mean")),
        ];
        let mut cursor = std::io::Cursor::new(&buf);
        let report = load_checkpoint(&mut cursor, &[], &f64_buffers, None).unwrap();
        assert_eq!(report.loaded.len(), 1);
        assert_eq!(f64_buffers[0].1.get().dtype(), DType::Float64);
        let vals = f64_buffers[0].1.get().to_f64_vec().unwrap();
        for v in vals {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compressed_roundtrip_with_hash() {
        // Test gz compression with structural hash validation
        let params = make_named_params(&[(8, 16)]);
        let hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

        let dir = std::env::temp_dir();
        let gz_path = dir.join("test_ckpt_hash_gz.fdl.gz");
        let path_str = gz_path.to_str().unwrap();

        save_checkpoint_file(path_str, &params, &[], Some(hash)).unwrap();

        // Load with matching hash
        let load_params = make_named_params(&[(8, 16)]);
        let report = load_checkpoint_file(path_str, &load_params, &[], Some(hash)).unwrap();
        assert_eq!(report.loaded.len(), 1);

        // Load with wrong hash should fail
        let bad_hash = "1111111111111111111111111111111111111111111111111111111111111111";
        let load_params2 = make_named_params(&[(8, 16)]);
        let result = load_checkpoint_file(path_str, &load_params2, &[], Some(bad_hash));
        assert!(result.is_err());

        std::fs::remove_file(gz_path).ok();
    }

    #[test]
    fn test_corrupted_gz_file() {
        // Write valid gz header then garbage: should produce an error
        let dir = std::env::temp_dir();
        let path = dir.join("test_corrupt.fdl.gz");
        // Write some garbage that is not valid gzip
        std::fs::write(&path, b"\x1f\x8b\x08\x00GARBAGE_NOT_VALID_GZ").unwrap();

        let params = make_named_params(&[(4, 8)]);
        let result = load_checkpoint_file(path.to_str().unwrap(), &params, &[], None);
        assert!(result.is_err(), "corrupted gz should return Err");

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_unknown_dtype_tag() {
        // Manually craft a checkpoint with an invalid dtype tag byte
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&[0u8; HASH_LEN]);
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 entry

        // Entry name
        let name = b"layer/weight";
        buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
        buf.extend_from_slice(name);

        // ndim = 1, shape = [4]
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4i64.to_le_bytes());

        // Invalid dtype tag (255)
        buf.push(255);

        // byte_count = 16 (4 * f32), then dummy data
        buf.extend_from_slice(&16u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 16]);

        let params = vec![
            ("layer/weight".to_string(), Parameter::new(
                Tensor::zeros(&[4], crate::tensor::test_opts()).unwrap(), "weight")),
        ];
        let mut cursor = std::io::Cursor::new(&buf);
        let result = load_checkpoint(&mut cursor, &params, &[], None);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("unknown dtype tag"), "error: {}", msg);
    }
}
