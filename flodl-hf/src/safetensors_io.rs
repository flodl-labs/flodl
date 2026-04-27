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

use flodl::{DType, Device, Graph, Result, Tensor, TensorError};
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
        self.into_result_impl(false)
    }

    /// Like [`into_result`](Self::into_result), but unused checkpoint keys
    /// don't count as an error — only missing keys and shape mismatches
    /// do. Used when loading a base model out of a checkpoint that also
    /// contains task-specific heads (e.g. pulling `BertModel` weights
    /// out of a `BertForPreTraining` checkpoint — the MLM / NSP head
    /// tensors are "unused" from `BertModel`'s point of view but their
    /// presence is expected, not an error).
    pub fn into_result_allow_unused(self) -> Result<()> {
        self.into_result_impl(true)
    }

    fn into_result_impl(mut self, allow_unused: bool) -> Result<()> {
        if allow_unused {
            self.unused.clear();
        }
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
    load_safetensors_into_graph_with_rename(graph, bytes, |k| k.to_string())
}

/// Same as [`load_safetensors_into_graph`] but applies `rename` to every
/// checkpoint key before matching against the graph.
///
/// `rename(checkpoint_key) -> canonical_key` lets callers paper over
/// legacy HF naming (e.g. `LayerNorm.gamma` → `LayerNorm.weight`). Use
/// [`bert_legacy_key_rename`] for the standard BERT mapping.
pub fn load_safetensors_into_graph_with_rename<F>(
    graph: &Graph,
    bytes: &[u8],
    rename: F,
) -> Result<()>
where
    F: Fn(&str) -> String,
{
    load_safetensors_core(graph, bytes, &rename, false)?;
    Ok(())
}

/// Like [`load_safetensors_into_graph_with_rename`] but tolerates
/// checkpoint keys that the graph does not ask for — useful when
/// loading a base model out of a checkpoint that also ships task heads
/// (e.g. `BertForPreTraining` on the Hub carries MLM + NSP heads that
/// a bare `BertModel` has no slot for). Missing keys and shape
/// mismatches are still hard errors.
///
/// Returns the list of checkpoint keys that were present but not used,
/// sorted alphabetically, so callers can surface them to the user.
pub fn load_safetensors_into_graph_with_rename_allow_unused<F>(
    graph: &Graph,
    bytes: &[u8],
    rename: F,
) -> Result<Vec<String>>
where
    F: Fn(&str) -> String,
{
    load_safetensors_core(graph, bytes, &rename, true)
}

fn load_safetensors_core(
    graph: &Graph,
    bytes: &[u8],
    rename: &dyn Fn(&str) -> String,
    allow_unused: bool,
) -> Result<Vec<String>> {
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| TensorError::new(&format!("safetensors parse error: {e}")))?;

    // Index: canonical (renamed) key → original checkpoint key. The rename
    // must be injective across the checkpoint's key set, otherwise two
    // checkpoint tensors would collapse onto the same canonical slot —
    // surface that as a validation-shaped error so the caller knows.
    let mut canonical_to_original: HashMap<String, String> = HashMap::new();
    let mut actual_shapes: HashMap<String, Vec<i64>> = HashMap::new();
    for name in st.names() {
        let canonical = rename(name);
        if let Some(prev) = canonical_to_original.insert(canonical.clone(), name.to_string()) {
            return Err(TensorError::new(&format!(
                "safetensors key rename collision: both {prev:?} and {name:?} \
                 map to canonical key {canonical:?}",
            )));
        }
        let view = st.tensor(name)
            .map_err(|e| TensorError::new(&format!("safetensors tensor lookup {name}: {e}")))?;
        actual_shapes.insert(
            canonical,
            view.shape().iter().map(|&s| s as i64).collect(),
        );
    }

    // Validate before touching any graph storage. If this fails, the graph
    // is left untouched so the caller can recover / report.
    let expected = expected_from_graph(graph);
    let validation = validate_keys(&expected, &actual_shapes);
    let unused = validation.unused.clone();
    if allow_unused {
        validation.into_result_allow_unused()?;
    } else {
        validation.into_result()?;
    }

    // Replace each parameter's tensor with the loaded one. `set_data`
    // preserves the Variable's `requires_grad` flag and its place in the
    // graph while swapping in the new backing storage. An in-place
    // `copy_` would trip libtorch's "leaf Variable used in in-place op"
    // check on parameters.
    for (flodl_key, param) in graph.named_parameters() {
        let hf_key = hf_key_from_flodl_key(&flodl_key);
        let original = canonical_to_original.get(&hf_key).ok_or_else(|| {
            TensorError::new(&format!(
                "canonical key {hf_key:?} missing from checkpoint after rename \
                 (validation should have caught this)",
            ))
        })?;
        let view = st.tensor(original)
            .map_err(|e| TensorError::new(&format!("safetensors tensor {original}: {e}")))?;
        let device = param.variable.data().device();
        let src = tensor_view_to_tensor(&view, device)?;
        param.variable.set_data(src);
    }

    // Same pattern for buffers (BERT has none today, but any future
    // model with BatchNorm-style running stats will).
    for (flodl_key, buffer) in graph.named_buffers() {
        let hf_key = hf_key_from_flodl_key(&flodl_key);
        let original = canonical_to_original.get(&hf_key).ok_or_else(|| {
            TensorError::new(&format!(
                "canonical buffer key {hf_key:?} missing after rename",
            ))
        })?;
        let view = st.tensor(original)
            .map_err(|e| TensorError::new(&format!("safetensors tensor {original}: {e}")))?;
        let src = tensor_view_to_tensor(&view, buffer.device())?;
        buffer.set(src);
    }

    Ok(unused)
}

/// Rewrite legacy HF BERT-family checkpoint keys to the form flodl's
/// MLM-head graphs expect.
///
/// 1. **LayerNorm gamma/beta** (TensorFlow-era): `LayerNorm.gamma` →
///    `LayerNorm.weight`, `LayerNorm.beta` → `LayerNorm.bias`.
///    `bert-base-*` and other pre-2020 checkpoints still ship with
///    `gamma`/`beta`. HF Python's `BertModel.from_pretrained` applies
///    the same remap at load time.
///
/// 2. **MLM decoder bias tying** (BERT / RoBERTa MLM):
///    `cls.predictions.bias` → `cls.predictions.decoder.bias`,
///    `lm_head.bias` → `lm_head.decoder.bias`. HF's `BertForMaskedLM`
///    and `RobertaForMaskedLM` both tie their decoder's bias to a
///    top-level `bias` Parameter via `self.decoder.bias = self.bias`.
///    PyTorch's `state_dict` dedupes tied Parameters on save, so
///    checkpoints ship only the top-level key. Our graphs store the
///    bias directly on the decoder `Linear` (one of the entry points
///    of weight tying via [`flodl::Linear::from_shared_weight`]), so
///    we rename the checkpoint's key onto the decoder at load time.
pub fn bert_legacy_key_rename(checkpoint_key: &str) -> String {
    // MLM decoder-bias tying: exact-match renames. Exact rather than
    // suffix so we don't accidentally eat `*.cls.predictions.bias`
    // sub-keys in some future nested head.
    if checkpoint_key == "cls.predictions.bias" {
        return "cls.predictions.decoder.bias".to_string();
    }
    if checkpoint_key == "lm_head.bias" {
        return "lm_head.decoder.bias".to_string();
    }
    if let Some(prefix) = checkpoint_key.strip_suffix("LayerNorm.gamma") {
        format!("{prefix}LayerNorm.weight")
    } else if let Some(prefix) = checkpoint_key.strip_suffix("LayerNorm.beta") {
        format!("{prefix}LayerNorm.bias")
    } else {
        checkpoint_key.to_string()
    }
}

/// HF-canonical LayerNorm rename: rewrite legacy `LayerNorm.gamma` /
/// `LayerNorm.beta` suffixes to the modern `LayerNorm.weight` /
/// `LayerNorm.bias` form. Pure suffix rename, leaves every other key
/// untouched. Used by the round-trip comparators in
/// `tests/roundtrip_common/mod.rs` to canonicalise HF-reference
/// safetensors against flodl exports — flodl always saves the modern
/// names; older HF checkpoints (e.g. `bert-base-uncased`) ship the
/// legacy names. Distinct from [`bert_legacy_key_rename`], which is
/// the *load-side* HF-to-flodl rename and additionally maps the MLM
/// decoder-bias tying alias (no longer needed by the comparator since
/// [`hf_canonical_save_key`] makes flodl saves match HF canonical
/// keys for that case).
pub fn bert_legacy_layernorm_rename(checkpoint_key: &str) -> String {
    if let Some(prefix) = checkpoint_key.strip_suffix("LayerNorm.gamma") {
        format!("{prefix}LayerNorm.weight")
    } else if let Some(prefix) = checkpoint_key.strip_suffix("LayerNorm.beta") {
        format!("{prefix}LayerNorm.bias")
    } else {
        checkpoint_key.to_string()
    }
}

/// Inverse of [`bert_legacy_key_rename`] for the MLM decoder-bias tying
/// case: rewrite flodl's internal `cls.predictions.decoder.bias` /
/// `lm_head.decoder.bias` parameter name back to the canonical HF key
/// (`cls.predictions.bias` / `lm_head.bias`) at save time.
///
/// Why save-side renames matter: HF Python's `BertForMaskedLM` /
/// `RobertaForMaskedLM` declare `self.bias` as the storage parameter
/// and then alias it via `self.decoder.bias = self.bias`. When HF's
/// `from_pretrained` loads a state_dict that has only
/// `cls.predictions.decoder.bias` (flodl's internal name), the
/// owning `cls.predictions.bias` parameter ends up on the meta device
/// and `tie_weights()` doesn't always materialise it on torch 2.x —
/// forward then fails with "Tensor on device meta is not on the
/// expected device cpu" inside the decoder's `addmm`. Emitting the
/// canonical HF key makes the load route through the correct owning
/// parameter and keeps the alias materialised.
///
/// Applied by [`save_safetensors_from_graph`] after [`crate::path::hf_key_from_flodl_key`]
/// converts the flodl tag separator to dotted form. The
/// `LayerNorm.gamma`/`beta` legacy names are NOT inverted on save —
/// flodl emits the modern `weight`/`bias` form, which both HF Python
/// and the Rust `_live` head-roundtrip comparator already canonicalise
/// to.
pub fn hf_canonical_save_key(hf_key: &str) -> String {
    if hf_key == "cls.predictions.decoder.bias" {
        return "cls.predictions.bias".to_string();
    }
    if hf_key == "lm_head.decoder.bias" {
        return "lm_head.bias".to_string();
    }
    hf_key.to_string()
}

/// Predicate: does `key` end with one of the pooler suffixes?
///
/// Matches both BERT-style `pooler.dense.{weight,bias}` (a wrapper
/// around a `BertPooler { dense: Linear }`) and ALBERT-style flat
/// `pooler.{weight,bias}` (HF's `AlbertModel.pooler` is a bare
/// `nn.Linear`). Pooler-less families (DistilBERT, DeBERTa-v2) never
/// match either shape, so the predicate is a safe no-op for them.
///
/// Normalises the `/` tag separator that flodl checkpoints use between
/// qualified tag boundaries (e.g. `bert.pooler/dense.weight`) so the
/// same predicate works for both raw safetensors keys and flodl's
/// internal tag form.
fn is_pooler_key(key: &str) -> bool {
    let normalised = key.replace('/', ".");
    normalised.ends_with("pooler.dense.weight")
        || normalised.ends_with("pooler.dense.bias")
        || normalised.ends_with("pooler.weight")
        || normalised.ends_with("pooler.bias")
}

/// Inspect a safetensors blob and report whether it carries the pooler
/// `Linear` weights for any of the pooler-bearing families (BERT,
/// RoBERTa, XLM-R, ALBERT).
///
/// Used by every pooler-bearing family's `from_pretrained_on_device`
/// (and `AutoModel::from_pretrained_for_export_on_device`) to pick
/// `on_device` vs `on_device_without_pooler` based on what the
/// checkpoint actually ships, rather than baking a per-family default
/// that's always wrong for some Hub repos (e.g. `roberta-base` has no
/// pooler; `bert-base-uncased` does).
pub fn weights_have_pooler(weights: &[u8]) -> Result<bool> {
    let st = SafeTensors::deserialize(weights)
        .map_err(|e| TensorError::new(&format!("safetensors parse error: {e}")))?;
    Ok(st.names().iter().any(|n| is_pooler_key(n)))
}

/// Detect pooler presence from a list of checkpoint keys (e.g. the
/// output of [`flodl::checkpoint_keys`]). Mirrors [`weights_have_pooler`]
/// for the checkpoint-keys input shape; the flodl checkpoint key form
/// uses `/` as the tag-boundary separator and this helper normalises
/// that internally so both safetensors-style dotted keys and tagged
/// flodl keys are handled by the same predicate.
pub fn keys_have_pooler(keys: &[String]) -> bool {
    keys.iter().any(|k| is_pooler_key(k))
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

/// Read a safetensors file from disk and load it into `graph`, applying
/// `rename` to every checkpoint key first. See
/// [`load_safetensors_into_graph_with_rename`].
pub fn load_safetensors_file_into_graph_with_rename<F>(
    graph: &Graph,
    path: &Path,
    rename: F,
) -> Result<()>
where
    F: Fn(&str) -> String,
{
    let bytes = std::fs::read(path).map_err(|e| {
        TensorError::new(&format!("safetensors read {}: {e}", path.display()))
    })?;
    load_safetensors_into_graph_with_rename(graph, &bytes, rename)
}

// ── Saving ────────────────────────────────────────────────────────────────

/// Serialise a graph's parameters and buffers as safetensors bytes.
///
/// Iterates [`Graph::named_parameters`] and [`Graph::named_buffers`],
/// converts each flodl key (slash form, e.g. `bert.pooler.dense/weight`)
/// to its HF-dotted equivalent via [`hf_key_from_flodl_key`], and writes
/// every tensor as f32 — the storage dtype flodl uses internally.
///
/// Tied parameters (the same `Variable` reachable through multiple tags)
/// are deduped upstream by `named_parameters`, so each weight ships
/// once. If two distinct tensors collide on the same HF key after
/// renaming, the function returns a loud error rather than silently
/// dropping one — that condition signals a tag-naming conflict in the
/// model, not a save-layer bug.
///
/// Output ordering is deterministic: keys serialise in HF-dotted
/// alphabetical order so the resulting file diffs cleanly across runs.
///
/// The bytes produced are byte-for-byte loadable by HF Python's
/// `safe_open(...).load_state_dict(...)` for any model whose `state_dict`
/// matches the saved key set.
pub fn save_safetensors_from_graph(graph: &Graph) -> Result<Vec<u8>> {
    use std::collections::BTreeMap;

    // BTreeMap orders keys for deterministic byte output; (dtype, shape, bytes)
    // payload owns the data so the TensorView slices below stay valid.
    let mut entries: BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)> = BTreeMap::new();

    for (flodl_key, param) in graph.named_parameters() {
        let hf_key = hf_canonical_save_key(&hf_key_from_flodl_key(&flodl_key));
        let shape: Vec<usize> = param.variable.shape().iter().map(|&d| d as usize).collect();
        let dtype = param.variable.data().dtype();
        let bytes = param.variable.data().to_blob()?;
        if entries.contains_key(&hf_key) {
            return Err(TensorError::new(&format!(
                "save_safetensors: HF key {hf_key:?} collision \
                 — multiple distinct flodl tensors map to the same name; \
                 fix the conflicting `tag(...)` in the graph",
            )));
        }
        entries.insert(hf_key, (dtype_to_safetensors(dtype)?, shape, bytes));
    }

    for (flodl_key, buffer) in graph.named_buffers() {
        let hf_key = hf_canonical_save_key(&hf_key_from_flodl_key(&flodl_key));
        let shape: Vec<usize> = buffer.shape().iter().map(|&d| d as usize).collect();
        let dtype = buffer.get().dtype();
        let bytes = buffer.get().to_blob()?;
        if entries.contains_key(&hf_key) {
            return Err(TensorError::new(&format!(
                "save_safetensors: HF key {hf_key:?} collision \
                 — buffer collides with a parameter or another buffer",
            )));
        }
        entries.insert(hf_key, (dtype_to_safetensors(dtype)?, shape, bytes));
    }

    let views: HashMap<String, TensorView<'_>> = entries.iter()
        .map(|(k, (dtype, shape, bytes))| {
            let view = TensorView::new(*dtype, shape.clone(), bytes.as_slice())
                .map_err(|e| TensorError::new(&format!(
                    "safetensors view build for {k:?}: {e}",
                )))?;
            Ok::<(String, TensorView<'_>), TensorError>((k.clone(), view))
        })
        .collect::<std::result::Result<_, _>>()?;

    safetensors::serialize(&views, &None)
        .map_err(|e| TensorError::new(&format!("safetensors serialize: {e}")))
}

/// Map a flodl `DType` to the safetensors `Dtype` that uses the same
/// in-memory bit layout. Integer dtypes are rejected — flodl's exporter
/// only emits learnable parameter / buffer payloads, which are floats.
fn dtype_to_safetensors(dtype: DType) -> Result<Dtype> {
    match dtype {
        DType::Float32 => Ok(Dtype::F32),
        DType::Float64 => Ok(Dtype::F64),
        DType::Float16 => Ok(Dtype::F16),
        DType::BFloat16 => Ok(Dtype::BF16),
        DType::Int32 | DType::Int64 => Err(TensorError::new(&format!(
            "save_safetensors: integer dtype {dtype:?} not supported \
             — only floating-point parameters / buffers can be serialised",
        ))),
    }
}

/// Serialise a graph and write the bytes to `path`. Thin file wrapper
/// over [`save_safetensors_from_graph`]; I/O errors carry the path in
/// the message for easier debugging.
pub fn save_safetensors_file_from_graph(graph: &Graph, path: &Path) -> Result<()> {
    let bytes = save_safetensors_from_graph(graph)?;
    std::fs::write(path, &bytes).map_err(|e| {
        TensorError::new(&format!("safetensors write {}: {e}", path.display()))
    })
}

/// Materialise a safetensors `TensorView` as a `Tensor` on
/// `target_device`, **preserving the source dtype**. Raw bytes are
/// shuttled through libtorch's `from_blob` (which copies internally),
/// so the resulting tensor has the same dtype as the safetensors file
/// — F16 stays F16, BF16 stays BF16, F32 stays F32, F64 stays F64.
fn tensor_view_to_tensor(view: &TensorView, target_device: Device) -> Result<Tensor> {
    let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
    let dtype = match view.dtype() {
        Dtype::F32 => DType::Float32,
        Dtype::F64 => DType::Float64,
        Dtype::F16 => DType::Float16,
        Dtype::BF16 => DType::BFloat16,
        other => {
            return Err(TensorError::new(&format!(
                "unsupported safetensors dtype {other:?} — floats (F32/F64/BF16/F16) only",
            )));
        }
    };
    Tensor::from_blob(view.data(), &shape, dtype, target_device)
}

/// Decode a safetensors `TensorView`'s raw bytes as a flat `Vec<f32>`.
/// Supports f32 (zero conversion), f64 / bf16 / f16 (host-side cast).
/// Rejects integer / bool dtypes — BERT-style checkpoints don't use them
/// and silently accepting would mean casting integers to floats, which
/// is almost never the user's intent.
///
/// Public so external callers (e.g. roundtrip tests, custom load
/// pipelines) can decode safetensors values to f32 with the exact same
/// dtype rules flodl uses on its load path. Equivalent on the f16 path
/// to `f32::from(half::f16::from_bits(_))` without pulling in `half`.
pub fn tensor_view_to_f32_vec(view: &TensorView) -> Result<Vec<f32>> {
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
    fn keys_have_pooler_detects_slash_separator() {
        let keys = vec![
            "encoder/layer/0/attention/query/weight".to_string(),
            "pooler/dense/weight".to_string(),
            "pooler/dense/bias".to_string(),
        ];
        assert!(keys_have_pooler(&keys));
    }

    #[test]
    fn keys_have_pooler_detects_dot_separator() {
        let keys = vec![
            "encoder.layer.0.attention.query.weight".to_string(),
            "pooler.dense.weight".to_string(),
        ];
        assert!(keys_have_pooler(&keys));
    }

    #[test]
    fn keys_have_pooler_returns_false_for_encoder_only() {
        let keys = vec![
            "encoder/layer/0/attention/query/weight".to_string(),
            "encoder/layer/0/attention/query/bias".to_string(),
        ];
        assert!(!keys_have_pooler(&keys));
    }

    #[test]
    fn keys_have_pooler_does_not_match_substrings() {
        // A key containing "pooler" mid-string must not false-positive.
        let keys = vec!["encoder/some_pooler_thing/weight".to_string()];
        assert!(!keys_have_pooler(&keys));
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

    /// BF16 checkpoint preserves dtype on load. bf16 is exactly the top
    /// 16 bits of a finite f32, so values representable as bf16 round-
    /// trip exactly through libtorch's f16/bf16 storage. Verifies both
    /// dtype preservation and value correctness via to_f32_vec()'s
    /// libtorch cast.
    #[test]
    fn load_safetensors_bf16_preserves_dtype() {
        use flodl::{DType, FlowBuilder, Linear};

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
            assert_eq!(p.variable.data().dtype(), DType::BFloat16,
                "{hf}: dtype must be preserved as BF16 after load");
            let data = p.variable.data().to_f32_vec().unwrap();
            match hf.as_str() {
                "m.weight" => assert_eq!(data, exact_w),
                "m.bias"   => assert_eq!(data, exact_b),
                other => panic!("unexpected {other}"),
            }
        }
    }

    /// F16 checkpoint preserves dtype on load. Tests +1, -1, +0.5, +0
    /// — values representable bit-exactly in f16, so to_f32_vec()'s
    /// libtorch f16→f32 cast is lossless and gives back the original
    /// mathematical values.
    #[test]
    fn load_safetensors_f16_preserves_dtype() {
        use flodl::{DType, FlowBuilder, Linear};

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
            assert_eq!(p.variable.data().dtype(), DType::Float16,
                "{hf}: dtype must be preserved as F16 after load");
            let data = p.variable.data().to_f32_vec().unwrap();
            match hf.as_str() {
                "m.weight" => assert_eq!(data, vec![1.0, -1.0, 0.5, 0.0]),
                "m.bias"   => assert_eq!(data, vec![1.0, 1.0, 1.0, 1.0]),
                other => panic!("unexpected {other}"),
            }
        }
    }

    /// Full round-trip: F16 safetensors → load → save → bit-exact F16
    /// safetensors. This is the contract the verify-export matrix runner
    /// relies on for DeBERTa-v3 (pure F16 upstream).
    #[test]
    fn save_safetensors_f16_roundtrip_byte_exact() {
        use flodl::{FlowBuilder, Linear};

        let dev = Device::CPU;
        let graph = FlowBuilder::new()
            .through(Linear::on_device(1, 4, dev).unwrap())
            .tag("m")
            .build().unwrap();

        let f16_bits: [u16; 4] = [0x3C00, 0xBC00, 0x3800, 0x0000];
        let bytes_w: Vec<u8> = f16_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let bytes_b: Vec<u8> = (0..4).flat_map(|_| 0x3C00u16.to_le_bytes()).collect();

        let src = serialize_entries(&[
            ("m.weight", Dtype::F16, vec![4, 1], bytes_w.clone()),
            ("m.bias",   Dtype::F16, vec![4],    bytes_b.clone()),
        ]);
        load_safetensors_into_graph(&graph, &src).unwrap();

        let saved = save_safetensors_from_graph(&graph).unwrap();
        let saved_st = SafeTensors::deserialize(&saved).unwrap();
        for (k, expected_bytes) in [("m.weight", &bytes_w), ("m.bias", &bytes_b)] {
            let v = saved_st.tensor(k).unwrap();
            assert_eq!(v.dtype(), Dtype::F16, "{k}: must save back as F16");
            assert_eq!(v.data(), expected_bytes.as_slice(),
                "{k}: F16 bytes must be bit-exact through load+save");
        }
    }

    /// Same contract as the F16 round-trip, but for BF16. BF16 is the
    /// dtype of choice for many recent LLMs — we want to be sure
    /// flodl preserves it without surprise downcasts.
    #[test]
    fn save_safetensors_bf16_roundtrip_byte_exact() {
        use flodl::{FlowBuilder, Linear};

        let dev = Device::CPU;
        let graph = FlowBuilder::new()
            .through(Linear::on_device(2, 2, dev).unwrap())
            .tag("m")
            .build().unwrap();

        // BF16 = top 16 bits of f32. These f32s have zero low bits → exact bf16.
        let exact_w = [1.0_f32, 2.0, -0.5, 0.25];
        let exact_b = [0.0_f32, -1.0];
        let to_bf16_bytes = |data: &[f32]| -> Vec<u8> {
            data.iter().flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes()).collect()
        };
        let bytes_w = to_bf16_bytes(&exact_w);
        let bytes_b = to_bf16_bytes(&exact_b);

        let src = serialize_entries(&[
            ("m.weight", Dtype::BF16, vec![2, 2], bytes_w.clone()),
            ("m.bias",   Dtype::BF16, vec![2],    bytes_b.clone()),
        ]);
        load_safetensors_into_graph(&graph, &src).unwrap();

        let saved = save_safetensors_from_graph(&graph).unwrap();
        let saved_st = SafeTensors::deserialize(&saved).unwrap();
        for (k, expected_bytes) in [("m.weight", &bytes_w), ("m.bias", &bytes_b)] {
            let v = saved_st.tensor(k).unwrap();
            assert_eq!(v.dtype(), Dtype::BF16, "{k}: must save back as BF16");
            assert_eq!(v.data(), expected_bytes.as_slice(),
                "{k}: BF16 bytes must be bit-exact through load+save");
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

    /// Legacy BERT checkpoint key rewriting: only `LayerNorm.gamma`
    /// and `LayerNorm.beta` suffixes are remapped; every other key
    /// passes through untouched. `bert-base-uncased` from the Hub ships
    /// with the legacy suffixes on every LayerNorm parameter, so this
    /// is the one knob that separates "loads" from "doesn't".
    #[test]
    fn bert_legacy_key_rename_rewrites_layernorm_suffixes() {
        assert_eq!(
            bert_legacy_key_rename("bert.embeddings.LayerNorm.gamma"),
            "bert.embeddings.LayerNorm.weight",
        );
        assert_eq!(
            bert_legacy_key_rename("bert.embeddings.LayerNorm.beta"),
            "bert.embeddings.LayerNorm.bias",
        );
        assert_eq!(
            bert_legacy_key_rename("bert.encoder.layer.3.attention.output.LayerNorm.gamma"),
            "bert.encoder.layer.3.attention.output.LayerNorm.weight",
        );
        // Non-LayerNorm keys pass through.
        assert_eq!(
            bert_legacy_key_rename("bert.encoder.layer.0.attention.self.query.weight"),
            "bert.encoder.layer.0.attention.self.query.weight",
        );
        // Partial matches (wrong suffix) are NOT remapped.
        assert_eq!(
            bert_legacy_key_rename("something.gamma"),
            "something.gamma",
        );
    }

    /// MLM decoder-bias tying: HF's `BertForMaskedLM` and
    /// `RobertaForMaskedLM` save a single top-level `bias` Parameter
    /// that is tied to `decoder.bias`; our graph stores the bias on the
    /// decoder `Linear` directly. The rename maps the checkpoint key
    /// onto our graph key so MLM checkpoints load cleanly.
    #[test]
    fn bert_legacy_key_rename_retags_mlm_tied_bias() {
        assert_eq!(
            bert_legacy_key_rename("cls.predictions.bias"),
            "cls.predictions.decoder.bias",
        );
        assert_eq!(
            bert_legacy_key_rename("lm_head.bias"),
            "lm_head.decoder.bias",
        );
        // Exact-match, not suffix — a hypothetical nested key
        // `something.cls.predictions.bias` is untouched.
        assert_eq!(
            bert_legacy_key_rename("something.cls.predictions.bias"),
            "something.cls.predictions.bias",
        );
    }

    /// If a checkpoint has BOTH `foo.LayerNorm.gamma` and
    /// `foo.LayerNorm.weight`, renaming collapses them onto the same
    /// canonical slot. The loader must surface this as a loud error
    /// rather than silently picking one — otherwise the user gets a
    /// non-deterministic load depending on HashMap iteration order.
    #[test]
    fn load_safetensors_rename_collision_errors_loudly() {
        use flodl::{FlowBuilder, Linear};

        let dev = Device::CPU;
        let graph = FlowBuilder::new()
            .through(Linear::on_device(2, 2, dev).unwrap())
            .tag("m")
            .build().unwrap();

        // Same numeric data, different legacy-vs-canonical key suffix.
        let w = f32_le_bytes(&[0.0, 1.0, 2.0, 3.0]);
        let b = f32_le_bytes(&[0.1, 0.2]);
        let bytes = serialize_entries(&[
            ("m.weight",       Dtype::F32, vec![2, 2], w.clone()),
            ("m.LayerNorm.gamma", Dtype::F32, vec![2], b.clone()),
            ("m.LayerNorm.weight", Dtype::F32, vec![2], b),
        ]);

        let err = load_safetensors_into_graph_with_rename(
            &graph, &bytes, bert_legacy_key_rename,
        ).unwrap_err().to_string();
        assert!(err.contains("rename collision"),
            "error must identify the collision: {err}");
        assert!(err.contains("LayerNorm.weight"),
            "error must name the canonical key involved: {err}");
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

    /// Save → load roundtrip via the public API: build a tagged graph
    /// with deterministic parameter values, save it as safetensors bytes,
    /// load those bytes into a fresh graph with the same structure, and
    /// assert every parameter is bit-exact f32 (lossless via the f32
    /// storage dtype).
    ///
    /// This is the strongest invariant the save layer must hold —
    /// "anything flodl loads, flodl can save and reload identically" —
    /// and it's what the per-family `_roundtrip_*_live` tests rely on.
    #[test]
    fn save_safetensors_load_roundtrip() {
        use flodl::{FlowBuilder, Linear, Module, Variable};

        let dev = Device::CPU;
        let in_dim = 3_i64;
        let out_dim = 2_i64;

        // Source graph with pinned weights.
        let src = FlowBuilder::new()
            .through(Linear::on_device(in_dim, out_dim, dev).unwrap())
            .tag("my.linear")
            .build().unwrap();
        let src_weight: Vec<f32> = (0..(in_dim * out_dim) as usize)
            .map(|i| 0.5 + i as f32 * 0.1).collect();
        let src_bias: Vec<f32> = (0..out_dim as usize)
            .map(|i| -1.0 + i as f32 * 0.25).collect();
        for (k, p) in src.named_parameters() {
            let hf = hf_key_from_flodl_key(&k);
            let t = match hf.as_str() {
                "my.linear.weight" => Tensor::from_f32(&src_weight, &[out_dim, in_dim], dev).unwrap(),
                "my.linear.bias"   => Tensor::from_f32(&src_bias,   &[out_dim], dev).unwrap(),
                other => panic!("unexpected key {other}"),
            };
            p.variable.set_data(t);
        }

        let bytes = save_safetensors_from_graph(&src).unwrap();

        // Destination graph: fresh, same structure, load the saved bytes.
        let dst = FlowBuilder::new()
            .through(Linear::on_device(in_dim, out_dim, dev).unwrap())
            .tag("my.linear")
            .build().unwrap();
        load_safetensors_into_graph(&dst, &bytes).unwrap();

        let mut dw: Option<Vec<f32>> = None;
        let mut db: Option<Vec<f32>> = None;
        for (k, p) in dst.named_parameters() {
            let hf = hf_key_from_flodl_key(&k);
            let data = p.variable.data().to_f32_vec().unwrap();
            match hf.as_str() {
                "my.linear.weight" => dw = Some(data),
                "my.linear.bias"   => db = Some(data),
                other => panic!("unexpected key {other}"),
            }
        }
        assert_eq!(dw.unwrap(), src_weight);
        assert_eq!(db.unwrap(), src_bias);

        let _keep_alive: Vec<Variable> =
            dst.parameters().into_iter().map(|p| p.variable).collect();
    }

    /// Saved keys land in HF-dotted form (slash → dot on the last
    /// segment) and the byte payload of each tensor matches the source's
    /// little-endian f32 representation. Guards against the save path
    /// drifting from `hf_key_from_flodl_key` and against endianness bugs
    /// in the byte assembly.
    #[test]
    fn save_safetensors_uses_hf_dotted_keys_and_le_f32() {
        use flodl::{FlowBuilder, Linear};

        let dev = Device::CPU;
        let graph = FlowBuilder::new()
            .through(Linear::on_device(2, 1, dev).unwrap())
            .tag("encoder.layer.0.attention.output.dense")
            .build().unwrap();
        let w = vec![0.25_f32, -0.5];
        let b = vec![1.0_f32];
        for (k, p) in graph.named_parameters() {
            let hf = hf_key_from_flodl_key(&k);
            let t = match hf.as_str() {
                "encoder.layer.0.attention.output.dense.weight"
                    => Tensor::from_f32(&w, &[1, 2], dev).unwrap(),
                "encoder.layer.0.attention.output.dense.bias"
                    => Tensor::from_f32(&b, &[1], dev).unwrap(),
                other => panic!("unexpected key {other}"),
            };
            p.variable.set_data(t);
        }

        let bytes = save_safetensors_from_graph(&graph).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();

        let names: HashSet<&str> = st.names().iter().map(|s| s.as_str()).collect();
        assert!(names.contains("encoder.layer.0.attention.output.dense.weight"),
            "expected HF-dotted key in output, got {names:?}");
        assert!(names.contains("encoder.layer.0.attention.output.dense.bias"),
            "expected HF-dotted key in output, got {names:?}");

        let w_view = st.tensor("encoder.layer.0.attention.output.dense.weight").unwrap();
        assert_eq!(w_view.dtype(), Dtype::F32);
        assert_eq!(w_view.shape(), &[1_usize, 2]);
        let w_back: Vec<f32> = w_view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(w_back, w);

        let b_view = st.tensor("encoder.layer.0.attention.output.dense.bias").unwrap();
        let b_back: Vec<f32> = b_view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(b_back, b);
    }

    /// Tied parameters — same `Variable` reachable under multiple tags —
    /// are deduped by `named_parameters` upstream, so the saved file
    /// contains the shared weight exactly once. Verifies the upstream
    /// guarantee actually flows through to the byte output: a
    /// hand-rolled tying via [`Linear::from_shared_weight`] yields one
    /// weight key, not two.
    #[test]
    fn save_safetensors_dedups_shared_weights() {
        use flodl::{FlowBuilder, Linear, Parameter};

        let dev = Device::CPU;
        let primary = Linear::on_device(2, 2, dev).unwrap();
        let shared_weight = primary.weight.clone(); // Rc clone, same storage
        let tied_bias = Parameter::new(
            Tensor::from_f32(&[0.0_f32, 0.0], &[2], dev).unwrap(),
            "bias",
        );
        let tied = Linear::from_shared_weight(shared_weight, Some(tied_bias));

        let graph = FlowBuilder::new()
            .through(primary).tag("primary")
            .through(tied).tag("tied")
            .build().unwrap();

        let bytes = save_safetensors_from_graph(&graph).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let names: HashSet<&str> = st.names().iter().map(|s| s.as_str()).collect();

        // Shared weight ships once under whichever tag named_parameters
        // visited first. Each Linear's own bias is a distinct Parameter,
        // so both bias keys appear.
        let weight_count = ["primary.weight", "tied.weight"].iter()
            .filter(|k| names.contains(*k))
            .count();
        assert_eq!(
            weight_count, 1,
            "shared weight must ship exactly once, got {names:?}",
        );
        assert!(names.contains("primary.bias"), "primary bias missing in {names:?}");
        assert!(names.contains("tied.bias"),    "tied bias missing in {names:?}");
    }

    /// File-path variant: same save → load roundtrip but through disk.
    /// Exercises the file writer and the path-in-error-message behaviour
    /// indirectly.
    #[test]
    fn save_safetensors_file_roundtrip() {
        use flodl::{FlowBuilder, Linear};

        let dev = Device::CPU;
        let graph = FlowBuilder::new()
            .through(Linear::on_device(2, 1, dev).unwrap())
            .tag("m")
            .build().unwrap();
        let w = vec![0.1_f32, 0.2];
        let b = vec![0.3_f32];
        for (k, p) in graph.named_parameters() {
            let hf = hf_key_from_flodl_key(&k);
            let t = match hf.as_str() {
                "m.weight" => Tensor::from_f32(&w, &[1, 2], dev).unwrap(),
                "m.bias"   => Tensor::from_f32(&b, &[1],    dev).unwrap(),
                other => panic!("unexpected {other}"),
            };
            p.variable.set_data(t);
        }

        let path = std::env::temp_dir()
            .join(format!("flodl_hf_save_test_{}.safetensors", std::process::id()));
        save_safetensors_file_from_graph(&graph, &path).unwrap();

        // Load back into a fresh graph through the file API; assert match.
        let fresh = FlowBuilder::new()
            .through(Linear::on_device(2, 1, dev).unwrap())
            .tag("m")
            .build().unwrap();
        load_safetensors_file_into_graph(&fresh, &path).unwrap();

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
}
