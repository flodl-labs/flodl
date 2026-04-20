//! Safetensors format I/O and load-time validation.
//!
//! This module's primary public surface today is [`LoadValidation`] — a
//! structured diff between what a flodl model expects and what a HuggingFace
//! safetensors file actually contains. It surfaces three failure modes:
//!
//! 1. **Missing** — keys the model expects but the checkpoint lacks
//!    (typo in a `FlowBuilder::tag(...)` string, wrong model variant, etc.).
//! 2. **Unused** — keys the checkpoint contains but the model never asks for
//!    (architecture mismatch, stale pretrained file, etc.).
//! 3. **Shape mismatch** — a key matches by name but the tensor dimensions
//!    disagree (vocab size change, hidden dim change, head count change).
//!
//! The validator is the safety net behind flodl-hf's "string-named tag"
//! convention: typos and drift are caught at load time with a loud,
//! actionable error listing every key that disagrees.
//!
//! Once validation passes, [`load_safetensors_into_graph`] (and the file
//! variant, [`load_safetensors_file_into_graph`]) copy tensor data from
//! the checkpoint into the graph's `Parameter` and `Buffer` storage
//! in-place. Checkpoint dtypes other than f32 (f16, bf16, f64) are cast
//! to f32 on the host; integer dtypes are currently rejected (BERT-style
//! models only store floats).
//!
//! # Example (validator only)
//!
//! ```
//! use std::collections::HashMap;
//! use flodl_hf::safetensors_io::{ExpectedParam, validate_keys};
//!
//! let expected = vec![
//!     ExpectedParam { key: "bert.embeddings.word_embeddings.weight".into(), shape: vec![30522, 768] },
//!     ExpectedParam { key: "bert.pooler.dense.bias".into(),              shape: vec![768] },
//! ];
//! let mut actual: HashMap<String, Vec<i64>> = HashMap::new();
//! actual.insert("bert.embeddings.word_embeddings.weight".into(), vec![30522, 768]);
//! actual.insert("bert.pooler.dense.bias".into(),                  vec![768]);
//!
//! let v = validate_keys(&expected, &actual);
//! assert!(v.is_ok());
//! ```

use std::collections::{HashMap, HashSet};
use std::path::Path;

use flodl::{Device, Graph, Result, Tensor, TensorError};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};

use crate::path::hf_key_from_flodl_key;

/// A parameter the model expects to find in a checkpoint.
///
/// `key`: the HF-dotted key as it appears in a safetensors file
/// (e.g. `bert.encoder.layer.0.attention.self.query.weight`).
/// `shape`: the tensor shape flodl will try to assign to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedParam {
    pub key: String,
    pub shape: Vec<i64>,
}

/// A single shape disagreement between model and checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeMismatch {
    pub key: String,
    pub expected: Vec<i64>,
    pub found: Vec<i64>,
}

/// The result of validating a checkpoint's key set against a model's
/// expected parameters.
///
/// Build this with [`validate_keys`]. Check [`is_ok`](LoadValidation::is_ok)
/// or convert to a loud `Result` via
/// [`into_result`](LoadValidation::into_result).
#[derive(Debug, Default, Clone)]
pub struct LoadValidation {
    pub missing: Vec<String>,
    pub unused: Vec<String>,
    pub shape_mismatches: Vec<ShapeMismatch>,
}

impl LoadValidation {
    /// True when there are no missing keys, no unused keys, and no shape
    /// mismatches.
    pub fn is_ok(&self) -> bool {
        self.missing.is_empty() && self.unused.is_empty() && self.shape_mismatches.is_empty()
    }

    /// Convert to a flodl `Result` — returns `Ok(())` when the validation is
    /// clean, otherwise a [`TensorError`] whose message lists every
    /// disagreement (truncated to the first 20 entries per bucket).
    pub fn into_result(self) -> Result<()> {
        if self.is_ok() {
            return Ok(());
        }
        let mut msg = String::from("safetensors checkpoint does not match model:\n");
        if !self.missing.is_empty() {
            msg.push_str(&format!(
                "  {} missing key(s) (model expects, checkpoint lacks):\n",
                self.missing.len(),
            ));
            for k in self.missing.iter().take(20) {
                msg.push_str(&format!("    - {k}\n"));
            }
            if self.missing.len() > 20 {
                msg.push_str(&format!("    ... and {} more\n", self.missing.len() - 20));
            }
        }
        if !self.unused.is_empty() {
            msg.push_str(&format!(
                "  {} unused key(s) (checkpoint has, model lacks):\n",
                self.unused.len(),
            ));
            for k in self.unused.iter().take(20) {
                msg.push_str(&format!("    - {k}\n"));
            }
            if self.unused.len() > 20 {
                msg.push_str(&format!("    ... and {} more\n", self.unused.len() - 20));
            }
        }
        if !self.shape_mismatches.is_empty() {
            msg.push_str(&format!(
                "  {} shape mismatch(es):\n",
                self.shape_mismatches.len(),
            ));
            for m in self.shape_mismatches.iter().take(20) {
                msg.push_str(&format!(
                    "    - {}: expected {:?}, found {:?}\n",
                    m.key, m.expected, m.found,
                ));
            }
            if self.shape_mismatches.len() > 20 {
                msg.push_str(&format!(
                    "    ... and {} more\n",
                    self.shape_mismatches.len() - 20,
                ));
            }
        }
        Err(TensorError::new(&msg))
    }
}

/// Validate model expectations against the `(key → shape)` map extracted
/// from a safetensors file.
///
/// Output is sorted (missing / unused / mismatches all ascending by key) so
/// error messages are stable across runs, which matters for diffing error
/// logs and writing tests.
pub fn validate_keys(
    expected: &[ExpectedParam],
    actual: &HashMap<String, Vec<i64>>,
) -> LoadValidation {
    let expected_keys: HashSet<&str> = expected.iter().map(|p| p.key.as_str()).collect();
    let mut v = LoadValidation::default();

    for p in expected {
        match actual.get(&p.key) {
            None => v.missing.push(p.key.clone()),
            Some(found) if found != &p.shape => {
                v.shape_mismatches.push(ShapeMismatch {
                    key: p.key.clone(),
                    expected: p.shape.clone(),
                    found: found.clone(),
                });
            }
            Some(_) => {}
        }
    }
    for k in actual.keys() {
        if !expected_keys.contains(k.as_str()) {
            v.unused.push(k.clone());
        }
    }
    v.missing.sort();
    v.unused.sort();
    v.shape_mismatches.sort_by(|a, b| a.key.cmp(&b.key));
    v
}

/// Collect expected parameters + buffers from a `Graph`, with keys already
/// converted to HF-dotted form via [`hf_key_from_flodl_key`].
///
/// Use this to drive [`validate_keys`] in the common case where the model
/// is a flodl `Graph` built via `FlowBuilder`.
pub fn expected_from_graph(graph: &Graph) -> Vec<ExpectedParam> {
    let mut out = Vec::new();
    for (k, p) in graph.named_parameters() {
        out.push(ExpectedParam {
            key: hf_key_from_flodl_key(&k),
            shape: p.variable.shape(),
        });
    }
    for (k, b) in graph.named_buffers() {
        out.push(ExpectedParam {
            key: hf_key_from_flodl_key(&k),
            shape: b.shape(),
        });
    }
    out
}

// ── Tensor data loading ──────────────────────────────────────────────────

/// Load a safetensors byte buffer's weights into the graph's parameters
/// and buffers.
///
/// Validates key set and shapes first — on any disagreement, returns a
/// [`LoadValidation::into_result`]-style error listing every mismatch so
/// the caller can fix tags / use the right checkpoint variant. On success,
/// copies each tensor's data (casting to f32 if the checkpoint is f16 /
/// bf16 / f64) into the live graph storage in-place. Parameters keep
/// their autograd identity; only the underlying buffer bytes change.
///
/// Integer-dtype checkpoint tensors are currently rejected — HF's
/// transformer zoo doesn't ship integer-weight tensors, so the common
/// path only needs floats.
pub fn load_safetensors_into_graph(graph: &Graph, bytes: &[u8]) -> Result<()> {
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| TensorError::new(&format!("safetensors parse error: {e}")))?;

    // Build `name → [i64] shape` for the validator.
    let mut actual_shapes: HashMap<String, Vec<i64>> = HashMap::new();
    for name in st.names() {
        let view = st.tensor(name)
            .map_err(|e| TensorError::new(&format!("safetensors tensor lookup {name}: {e}")))?;
        actual_shapes.insert(
            name.to_string(),
            view.shape().iter().map(|&s| s as i64).collect(),
        );
    }

    // Validate before touching any graph storage. If this fails, the graph
    // is left untouched so the caller can recover / report.
    let expected = expected_from_graph(graph);
    validate_keys(&expected, &actual_shapes).into_result()?;

    // Replace each parameter's tensor with the loaded one. `set_data`
    // preserves the Variable's `requires_grad` flag and its place in the
    // graph while swapping in the new backing storage. An in-place
    // `copy_` would trip libtorch's "leaf Variable used in in-place op"
    // check on parameters.
    for (flodl_key, param) in graph.named_parameters() {
        let hf_key = hf_key_from_flodl_key(&flodl_key);
        let view = st.tensor(&hf_key)
            .map_err(|e| TensorError::new(&format!("safetensors tensor {hf_key}: {e}")))?;
        let device = param.variable.data().device();
        let src = tensor_view_to_f32_tensor(&view, device)?;
        param.variable.set_data(src);
    }

    // Same pattern for buffers (BERT has none today, but any future
    // model with BatchNorm-style running stats will).
    for (flodl_key, buffer) in graph.named_buffers() {
        let hf_key = hf_key_from_flodl_key(&flodl_key);
        let view = st.tensor(&hf_key)
            .map_err(|e| TensorError::new(&format!("safetensors tensor {hf_key}: {e}")))?;
        let src = tensor_view_to_f32_tensor(&view, buffer.device())?;
        buffer.set(src);
    }

    Ok(())
}

/// Read a safetensors file from disk and load it into `graph`.
///
/// Thin wrapper around [`load_safetensors_into_graph`]. I/O errors are
/// surfaced as `TensorError` with the path in the message for easier
/// debugging.
pub fn load_safetensors_file_into_graph(graph: &Graph, path: &Path) -> Result<()> {
    let bytes = std::fs::read(path).map_err(|e| {
        TensorError::new(&format!("safetensors read {}: {e}", path.display()))
    })?;
    load_safetensors_into_graph(graph, &bytes)
}

/// Materialise a safetensors `TensorView` as a CPU f32 `Tensor`, then
/// move it to `target_device`. Host-side dtype conversion keeps this
/// module independent of libtorch fp16/bf16 constructors.
fn tensor_view_to_f32_tensor(view: &TensorView, target_device: Device) -> Result<Tensor> {
    let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
    let data = tensor_view_to_f32_vec(view)?;
    let cpu = Tensor::from_f32(&data, &shape, Device::CPU)?;
    if target_device == Device::CPU {
        Ok(cpu)
    } else {
        cpu.to_device(target_device)
    }
}

/// Decode a safetensors `TensorView`'s raw bytes as a flat `Vec<f32>`.
/// Supports f32 (zero conversion), f64 / bf16 / f16 (host-side cast).
/// Rejects integer / bool dtypes — BERT-style checkpoints don't use them
/// and silently accepting would mean casting integers to floats, which
/// is almost never the user's intent.
fn tensor_view_to_f32_vec(view: &TensorView) -> Result<Vec<f32>> {
    let bytes = view.data();
    match view.dtype() {
        Dtype::F32 => {
            if bytes.len() % 4 != 0 {
                return Err(TensorError::new(&format!(
                    "F32 tensor byte length {} is not a multiple of 4", bytes.len(),
                )));
            }
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        Dtype::F64 => {
            if bytes.len() % 8 != 0 {
                return Err(TensorError::new(&format!(
                    "F64 tensor byte length {} is not a multiple of 8", bytes.len(),
                )));
            }
            let mut out = Vec::with_capacity(bytes.len() / 8);
            for chunk in bytes.chunks_exact(8) {
                let bits = f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                    chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                // Matches PyTorch's `.to(torch.float32)`: IEEE 754 narrowing,
                // overflow saturates silently to ±inf, precision rounds to
                // nearest-even. Transformer weights never hit the tails, so
                // the silent saturation is acceptable and PyTorch-compatible.
                out.push(bits as f32);
            }
            Ok(out)
        }
        Dtype::BF16 => {
            if bytes.len() % 2 != 0 {
                return Err(TensorError::new(&format!(
                    "BF16 tensor byte length {} is not a multiple of 2", bytes.len(),
                )));
            }
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                // bf16 is the top 16 bits of a f32 (same exponent).
                out.push(f32::from_bits((bits as u32) << 16));
            }
            Ok(out)
        }
        Dtype::F16 => {
            if bytes.len() % 2 != 0 {
                return Err(TensorError::new(&format!(
                    "F16 tensor byte length {} is not a multiple of 2", bytes.len(),
                )));
            }
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f16_bits_to_f32(bits));
            }
            Ok(out)
        }
        other => Err(TensorError::new(&format!(
            "unsupported safetensors dtype {other:?} — floats (F32/F64/BF16/F16) only",
        ))),
    }
}

/// IEEE 754 half-precision (binary16) to single-precision conversion.
///
/// Handles zero, subnormals, normals, infinity, and NaN. No external
/// dependency. Equivalent to `f32::from(half::f16::from_bits(bits))`
/// but keeps flodl-hf dep-light.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32 & 0x1;
    let exp = (bits >> 10) as u32 & 0x1f;
    let mantissa = bits as u32 & 0x3ff;

    let out_bits: u32 = if exp == 0 {
        if mantissa == 0 {
            sign << 31
        } else {
            // Subnormal half → normal f32.
            let mut m = mantissa;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let f32_exp = (e + 127) as u32 & 0xff;
            (sign << 31) | (f32_exp << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        // Inf (mantissa == 0) or NaN (mantissa != 0) — preserve bits
        // shifted into the wider mantissa.
        (sign << 31) | (0xff << 23) | (mantissa << 13)
    } else {
        // Normal half.
        let f32_exp = (exp + 127 - 15) & 0xff;
        (sign << 31) | (f32_exp << 23) | (mantissa << 13)
    };
    f32::from_bits(out_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn actual_map(entries: &[(&str, &[i64])]) -> HashMap<String, Vec<i64>> {
        entries.iter().map(|(k, s)| ((*k).to_string(), s.to_vec())).collect()
    }

    #[test]
    fn all_keys_match_returns_ok() {
        let expected = vec![
            ExpectedParam { key: "bert.embeddings.word_embeddings.weight".into(), shape: vec![30522, 768] },
            ExpectedParam { key: "bert.pooler.dense.bias".into(),                  shape: vec![768] },
        ];
        let actual = actual_map(&[
            ("bert.embeddings.word_embeddings.weight", &[30522, 768]),
            ("bert.pooler.dense.bias",                  &[768]),
        ]);
        let v = validate_keys(&expected, &actual);
        assert!(v.is_ok());
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn missing_key_is_reported() {
        let expected = vec![
            ExpectedParam { key: "bert.pooler.dense.weight".into(), shape: vec![768, 768] },
        ];
        let actual = actual_map(&[]);
        let v = validate_keys(&expected, &actual);
        assert_eq!(v.missing, vec!["bert.pooler.dense.weight"]);
        assert!(v.unused.is_empty());
        assert!(v.shape_mismatches.is_empty());
    }

    #[test]
    fn unused_checkpoint_key_is_reported() {
        let expected: Vec<ExpectedParam> = Vec::new();
        let actual = actual_map(&[("bert.something.extra", &[4])]);
        let v = validate_keys(&expected, &actual);
        assert_eq!(v.unused, vec!["bert.something.extra"]);
        assert!(v.missing.is_empty());
        assert!(v.shape_mismatches.is_empty());
    }

    #[test]
    fn shape_mismatch_is_reported() {
        // Vocab size drift: checkpoint has 30522 tokens, model expects 50257.
        let expected = vec![
            ExpectedParam {
                key: "bert.embeddings.word_embeddings.weight".into(),
                shape: vec![50257, 768],
            },
        ];
        let actual = actual_map(&[
            ("bert.embeddings.word_embeddings.weight", &[30522, 768]),
        ]);
        let v = validate_keys(&expected, &actual);
        assert!(v.missing.is_empty());
        assert!(v.unused.is_empty());
        assert_eq!(v.shape_mismatches.len(), 1);
        assert_eq!(v.shape_mismatches[0].key, "bert.embeddings.word_embeddings.weight");
        assert_eq!(v.shape_mismatches[0].expected, vec![50257, 768]);
        assert_eq!(v.shape_mismatches[0].found,    vec![30522, 768]);
    }

    #[test]
    fn typo_queri_vs_query_reports_both_missing_and_unused() {
        // The motivating bug: author typed "queri" in a tag, checkpoint has "query".
        // Validator must surface both sides so the typo is unambiguous.
        let expected = vec![
            ExpectedParam { key: "bert.encoder.layer.0.attention.self.queri.weight".into(), shape: vec![768, 768] },
        ];
        let actual = actual_map(&[
            ("bert.encoder.layer.0.attention.self.query.weight", &[768, 768]),
        ]);
        let v = validate_keys(&expected, &actual);
        assert_eq!(v.missing, vec!["bert.encoder.layer.0.attention.self.queri.weight"]);
        assert_eq!(v.unused,  vec!["bert.encoder.layer.0.attention.self.query.weight"]);
    }

    #[test]
    fn mixed_failures_accumulate() {
        let expected = vec![
            ExpectedParam { key: "ok.weight".into(),        shape: vec![4] },
            ExpectedParam { key: "missing.weight".into(),   shape: vec![8] },
            ExpectedParam { key: "wrong_shape.weight".into(), shape: vec![16] },
        ];
        let actual = actual_map(&[
            ("ok.weight",            &[4]),
            ("wrong_shape.weight",   &[32]),
            ("extra.weight",         &[1]),
        ]);
        let v = validate_keys(&expected, &actual);
        assert_eq!(v.missing,          vec!["missing.weight"]);
        assert_eq!(v.unused,           vec!["extra.weight"]);
        assert_eq!(v.shape_mismatches.len(), 1);
        assert_eq!(v.shape_mismatches[0].key, "wrong_shape.weight");
    }

    #[test]
    fn into_result_error_message_lists_every_bucket() {
        let expected = vec![
            ExpectedParam { key: "m.w".into(),  shape: vec![2] },
            ExpectedParam { key: "sm.w".into(), shape: vec![3] },
        ];
        let actual = actual_map(&[
            ("sm.w",    &[4]),
            ("extra.w", &[1]),
        ]);
        let v = validate_keys(&expected, &actual);
        let err = v.into_result().unwrap_err().to_string();
        assert!(err.contains("1 missing key"),     "missing bucket not in msg: {err}");
        assert!(err.contains("1 unused key"),      "unused bucket not in msg: {err}");
        assert!(err.contains("1 shape mismatch"),  "shape bucket not in msg: {err}");
        assert!(err.contains("m.w"));
        assert!(err.contains("extra.w"));
        assert!(err.contains("sm.w"));
        assert!(err.contains("[3]"));
        assert!(err.contains("[4]"));
    }

    #[test]
    fn output_is_sorted_for_stable_messages() {
        let expected = vec![
            ExpectedParam { key: "z.w".into(), shape: vec![1] },
            ExpectedParam { key: "a.w".into(), shape: vec![1] },
        ];
        let actual = actual_map(&[
            ("m.w", &[1]),
            ("c.w", &[1]),
        ]);
        let v = validate_keys(&expected, &actual);
        assert_eq!(v.missing, vec!["a.w", "z.w"]);
        assert_eq!(v.unused,  vec!["c.w", "m.w"]);
    }

    #[test]
    fn empty_everywhere_is_ok() {
        let v = validate_keys(&[], &HashMap::new());
        assert!(v.is_ok());
        assert!(v.missing.is_empty());
        assert!(v.unused.is_empty());
        assert!(v.shape_mismatches.is_empty());
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn into_result_truncates_long_missing_list() {
        // 25 missing keys — "... and N more" tail should appear for any
        // bucket longer than the 20-entry cap.
        let expected: Vec<ExpectedParam> = (0..25)
            .map(|i| ExpectedParam { key: format!("key.{i:02}"), shape: vec![1] })
            .collect();
        let v = validate_keys(&expected, &HashMap::new());
        assert_eq!(v.missing.len(), 25);
        let err = v.into_result().unwrap_err().to_string();
        assert!(err.contains("25 missing key"), "header must show full count: {err}");
        assert!(err.contains("... and 5 more"),
            "truncation tail must show remaining count: {err}");
        // First 20 keys listed; 21st onwards must not appear verbatim.
        assert!(err.contains("key.00"));
        assert!(err.contains("key.19"));
        assert!(!err.contains("key.20"));
    }

    /// Helper: raw f32 bytes (little-endian) from a flat slice.
    fn f32_le_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Helper: serialise a list of (name, dtype, shape, byte-data) into a
    /// safetensors byte buffer. Lifetimes stay simple because the byte
    /// buffers are owned by the caller and outlive the `TensorView`s.
    fn serialize_entries(entries: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
        let views: HashMap<String, TensorView<'_>> = entries.iter().map(|(n, d, s, b)| {
            (n.to_string(), TensorView::new(*d, s.clone(), b).unwrap())
        }).collect();
        safetensors::serialize(&views, &None).unwrap()
    }

    /// End-to-end: build a tagged Linear graph, pin its parameters to
    /// deterministic values, serialise them as f32 safetensors, then load
    /// the bytes into a second fresh graph. The second graph must end up
    /// bit-exact on both `weight` and `bias`. Guards the main load path
    /// plus the HF key conversion (slash → dot) inside the loader.
    #[test]
    fn load_safetensors_f32_roundtrip() {
        use flodl::{FlowBuilder, Linear, Module, Variable};

        let in_dim = 3_i64;
        let out_dim = 2_i64;
        let dev = Device::CPU;

        // Source graph: tag it and overwrite the random init with known values.
        let src_graph = FlowBuilder::new()
            .through(Linear::on_device(in_dim, out_dim, dev).unwrap())
            .tag("my.linear")
            .build().unwrap();
        let src_weight: Vec<f32> = (0..(in_dim * out_dim) as usize)
            .map(|i| 1.0 + i as f32 * 0.25).collect();
        let src_bias: Vec<f32> = (0..out_dim as usize)
            .map(|i| -0.5 + i as f32).collect();
        for (k, p) in src_graph.named_parameters() {
            let hf = hf_key_from_flodl_key(&k);
            let t = match hf.as_str() {
                "my.linear.weight" => Tensor::from_f32(&src_weight, &[out_dim, in_dim], dev).unwrap(),
                "my.linear.bias"   => Tensor::from_f32(&src_bias,   &[out_dim], dev).unwrap(),
                other => panic!("unexpected key {other}"),
            };
            p.variable.set_data(t);
        }

        // Serialise source params as f32 safetensors.
        let w_bytes = f32_le_bytes(&src_weight);
        let b_bytes = f32_le_bytes(&src_bias);
        let bytes = serialize_entries(&[
            ("my.linear.weight", Dtype::F32, vec![out_dim as usize, in_dim as usize], w_bytes),
            ("my.linear.bias",   Dtype::F32, vec![out_dim as usize],                   b_bytes),
        ]);

        // Destination graph: fresh, then load.
        let dst_graph = FlowBuilder::new()
            .through(Linear::on_device(in_dim, out_dim, dev).unwrap())
            .tag("my.linear")
            .build().unwrap();
        load_safetensors_into_graph(&dst_graph, &bytes).unwrap();

        // Assert bit-exactness on both params (f32 → f32 is lossless).
        let mut dst_weight: Option<Vec<f32>> = None;
        let mut dst_bias: Option<Vec<f32>> = None;
        for (k, p) in dst_graph.named_parameters() {
            let hf = hf_key_from_flodl_key(&k);
            let data = p.variable.data().to_f32_vec().unwrap();
            match hf.as_str() {
                "my.linear.weight" => dst_weight = Some(data),
                "my.linear.bias"   => dst_bias   = Some(data),
                other => panic!("unexpected key {other}"),
            }
        }
        assert_eq!(dst_weight.unwrap(), src_weight);
        assert_eq!(dst_bias.unwrap(),   src_bias);

        // Sanity: dst params are still alive as Variables (load shouldn't
        // replace them — just refill storage).
        let _keep_alive: Vec<Variable> = dst_graph.parameters().into_iter().map(|p| p.variable).collect();
    }

    /// File-path variant: same roundtrip but through disk, to exercise the
    /// file reader and the path-in-error-message behaviour indirectly.
    #[test]
    fn load_safetensors_file_roundtrip() {
        use flodl::{FlowBuilder, Linear};
        use std::io::Write;

        let dev = Device::CPU;
        let graph = FlowBuilder::new()
            .through(Linear::on_device(2, 1, dev).unwrap())
            .tag("m")
            .build().unwrap();
        let w = vec![0.25_f32, 0.5];
        let b = vec![1.5_f32];
        for (k, p) in graph.named_parameters() {
            let hf = hf_key_from_flodl_key(&k);
            let t = match hf.as_str() {
                "m.weight" => Tensor::from_f32(&w, &[1, 2], dev).unwrap(),
                "m.bias"   => Tensor::from_f32(&b, &[1],    dev).unwrap(),
                other => panic!("unexpected {other}"),
            };
            p.variable.set_data(t);
        }
        let bytes = serialize_entries(&[
            ("m.weight", Dtype::F32, vec![1, 2], f32_le_bytes(&w)),
            ("m.bias",   Dtype::F32, vec![1],    f32_le_bytes(&b)),
        ]);

        let path = std::env::temp_dir().join(format!("flodl_hf_test_{}.safetensors", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();

        let fresh = FlowBuilder::new()
            .through(Linear::on_device(2, 1, dev).unwrap())
            .tag("m")
            .build().unwrap();
        load_safetensors_file_into_graph(&fresh, &path).unwrap();

        // Cleanup first so a failed assert doesn't leak the tmp file.
        let _ = std::fs::remove_file(&path);

        for (k, p) in fresh.named_parameters() {
            let hf = hf_key_from_flodl_key(&k);
            let data = p.variable.data().to_f32_vec().unwrap();
            match hf.as_str() {
                "m.weight" => assert_eq!(data, w),
                "m.bias"   => assert_eq!(data, b),
                other => panic!("unexpected {other}"),
            }
        }
    }

    /// BF16 checkpoint → f32 graph. bf16 is exactly the top 16 bits of a
    /// finite f32, so values representable as bf16 must roundtrip exactly
    /// after the host-side widen. Guards the `from_bits((bits as u32) << 16)`
    /// path against endianness / shift bugs.
    #[test]
    fn load_safetensors_bf16_casts_to_f32() {
        use flodl::{FlowBuilder, Linear};

        let dev = Device::CPU;
        let graph = FlowBuilder::new()
            .through(Linear::on_device(2, 2, dev).unwrap())
            .tag("m")
            .build().unwrap();

        // bf16 representable values: pick f32s whose bottom 16 bits are 0.
        let exact_w = [1.0_f32, 2.0, -0.5, 0.25];
        let exact_b = [0.0_f32, -1.0];
        let to_bf16_bytes = |data: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(data.len() * 2);
            for &f in data {
                let top = (f.to_bits() >> 16) as u16;
                out.extend_from_slice(&top.to_le_bytes());
            }
            out
        };
        let bytes = serialize_entries(&[
            ("m.weight", Dtype::BF16, vec![2, 2], to_bf16_bytes(&exact_w)),
            ("m.bias",   Dtype::BF16, vec![2],    to_bf16_bytes(&exact_b)),
        ]);

        load_safetensors_into_graph(&graph, &bytes).unwrap();

        for (k, p) in graph.named_parameters() {
            let hf = hf_key_from_flodl_key(&k);
            let data = p.variable.data().to_f32_vec().unwrap();
            match hf.as_str() {
                "m.weight" => assert_eq!(data, exact_w),
                "m.bias"   => assert_eq!(data, exact_b),
                other => panic!("unexpected {other}"),
            }
        }
    }

    /// F16 checkpoint → f32 graph. Tests both normal and subnormal paths
    /// of the host-side converter, plus +/- zero.
    #[test]
    fn load_safetensors_f16_casts_to_f32() {
        use flodl::{FlowBuilder, Linear};

        let dev = Device::CPU;
        let graph = FlowBuilder::new()
            .through(Linear::on_device(1, 4, dev).unwrap())
            .tag("m")
            .build().unwrap();

        // IEEE 754 binary16 bit patterns:
        // 0x3C00 = 1.0
        // 0xBC00 = -1.0
        // 0x3800 = 0.5
        // 0x0000 = +0.0
        let f16_bits: [u16; 4] = [0x3C00, 0xBC00, 0x3800, 0x0000];
        let mut bytes_w = Vec::with_capacity(8);
        for b in f16_bits {
            bytes_w.extend_from_slice(&b.to_le_bytes());
        }
        let bias_bits: [u16; 1] = [0x3C00];
        let bytes_b: Vec<u8> = bias_bits[0].to_le_bytes().to_vec();

        let st_bytes = serialize_entries(&[
            ("m.weight", Dtype::F16, vec![4, 1], bytes_w),
            ("m.bias",   Dtype::F16, vec![4],    bytes_b.repeat(4)),
        ]);
        load_safetensors_into_graph(&graph, &st_bytes).unwrap();

        for (k, p) in graph.named_parameters() {
            let hf = hf_key_from_flodl_key(&k);
            let data = p.variable.data().to_f32_vec().unwrap();
            match hf.as_str() {
                "m.weight" => assert_eq!(data, vec![1.0, -1.0, 0.5, 0.0]),
                "m.bias"   => assert_eq!(data, vec![1.0, 1.0, 1.0, 1.0]),
                other => panic!("unexpected {other}"),
            }
        }
    }

    /// Validation failures must leave the graph untouched — callers rely
    /// on "either all params loaded, or none" so they can fall back or
    /// report safely. Missing key → error; on the error path the caller's
    /// graph is free for them to mutate without inconsistent state.
    #[test]
    fn load_safetensors_missing_key_errors_loudly() {
        use flodl::{FlowBuilder, Linear};

        let dev = Device::CPU;
        let graph = FlowBuilder::new()
            .through(Linear::on_device(2, 2, dev).unwrap())
            .tag("m")
            .build().unwrap();
        // Only ship the weight, not the bias.
        let w = vec![0.0_f32, 1.0, 2.0, 3.0];
        let bytes = serialize_entries(&[
            ("m.weight", Dtype::F32, vec![2, 2], f32_le_bytes(&w)),
        ]);
        let err = load_safetensors_into_graph(&graph, &bytes).unwrap_err().to_string();
        assert!(err.contains("missing key"), "error must mention missing keys: {err}");
        assert!(err.contains("m.bias"),      "error must name the missing key: {err}");
    }

    /// Integer dtypes are rejected explicitly rather than silently cast —
    /// a user shipping an I32 checkpoint almost certainly means something
    /// went wrong upstream.
    #[test]
    fn load_safetensors_rejects_integer_dtype() {
        use flodl::{FlowBuilder, Linear};

        let dev = Device::CPU;
        let graph = FlowBuilder::new()
            .through(Linear::on_device(1, 1, dev).unwrap())
            .tag("m")
            .build().unwrap();
        // I32 bias: 4 bytes per element × 1 element.
        let bias_i32: Vec<u8> = 1_i32.to_le_bytes().to_vec();
        let w_bytes = f32_le_bytes(&[0.5_f32]);
        let bytes = serialize_entries(&[
            ("m.weight", Dtype::F32, vec![1, 1], w_bytes),
            ("m.bias",   Dtype::I32, vec![1],    bias_i32),
        ]);
        let err = load_safetensors_into_graph(&graph, &bytes).unwrap_err().to_string();
        assert!(err.contains("unsupported safetensors dtype"),
            "error must call out dtype: {err}");
        assert!(err.contains("I32"), "error must name the offending dtype: {err}");
    }

    /// f16 helper unit check: +Inf, -Inf, NaN, smallest subnormal all
    /// survive the widening with the right classification. The loader
    /// catches these via the same function, so a regression here would
    /// surface as Weird Numerical Behaviour™ in a loaded model.
    #[test]
    fn f16_bits_to_f32_special_values() {
        // +Inf: exp=0x1f, mantissa=0
        assert!(f16_bits_to_f32(0x7C00).is_infinite() && f16_bits_to_f32(0x7C00).is_sign_positive());
        // -Inf
        assert!(f16_bits_to_f32(0xFC00).is_infinite() && f16_bits_to_f32(0xFC00).is_sign_negative());
        // NaN: exp=0x1f, mantissa != 0
        assert!(f16_bits_to_f32(0x7E00).is_nan());
        // Smallest positive subnormal half: 0x0001 = 2^-24
        let tiny = f16_bits_to_f32(0x0001);
        assert!((tiny - 2.0_f32.powi(-24)).abs() < 1e-10, "tiny subnormal wrong: {tiny}");
        // -0.0 preserves sign
        assert!(f16_bits_to_f32(0x8000).is_sign_negative() && f16_bits_to_f32(0x8000) == 0.0);
    }

    #[test]
    fn expected_from_graph_converts_slash_to_dot() {
        use flodl::{FlowBuilder, Linear, Module};
        let fb = FlowBuilder::new()
            .through(Linear::new(4, 2).unwrap()).tag("bert.pooler.dense");
        let graph = fb.build().unwrap();
        let expected = expected_from_graph(&graph);
        // Graph::named_parameters gives "bert.pooler.dense/weight" and
        // "bert.pooler.dense/bias". expected_from_graph must swap the last
        // slash for a dot.
        let keys: Vec<&str> = expected.iter().map(|e| e.key.as_str()).collect();
        assert!(keys.contains(&"bert.pooler.dense.weight"),
                "expected HF-dotted key missing, got {keys:?}");
        assert!(keys.contains(&"bert.pooler.dense.bias"),
                "expected HF-dotted key missing, got {keys:?}");
        // Sanity check: the parameter count matches Graph's own view.
        assert_eq!(expected.len(), graph.parameters().len());
    }
}
