//! BERT encoder, compatible with HuggingFace `bert-base-uncased` checkpoints.
//!
//! Structure: [`BertEmbeddings`] (token + position + token-type embeddings
//! with LayerNorm and Dropout), a stack of
//! [`TransformerLayer`](crate::models::transformer_layer::TransformerLayer)
//! instances (self-attention + two-layer GELU feed-forward, both wrapped
//! with residual + LayerNorm), and [`BertPooler`] on the `[CLS]` position.
//! [`BertModel::build`] assembles all of this into a flat [`Graph`].
//!
//! The encoder layer itself is shared with the RoBERTa and DistilBERT
//! ports via
//! [`LayerNaming::BERT`](crate::models::transformer_layer::LayerNaming::BERT)
//! — the Q/K/V projections, output projection, two LayerNorms, and
//! feed-forward block are identical, only the HF weight-key suffixes
//! differ.
//!
//! Padding is handled via an additive attention mask threaded into every
//! encoder layer as a named graph input (see [`build_extended_attention_mask`]).
//!
//! Parameter names are chosen so `Graph::named_parameters()` output, once
//! passed through [`hf_key_from_flodl_key`](crate::path::hf_key_from_flodl_key),
//! matches safetensors checkpoint keys exactly. No remapping needed at load
//! time.

use std::collections::HashMap;

use flodl::nn::{Dropout, Embedding, LayerNorm, Linear, Module, NamedInputModule, Parameter};
use flodl::{DType, Device, FlowBuilder, Graph, Result, Tensor, TensorError, Variable};

use crate::models::transformer_layer::{LayerNaming, TransformerLayer, TransformerLayerConfig};
use crate::path::{prefix_params, HfPath};

/// Convert a `[batch, seq_len]` attention mask (0 = mask, 1 = attend,
/// any numeric dtype) into a `[batch, 1, 1, seq_len]` additive f32 mask
/// suitable as the fourth input to the BERT graph.
///
/// Masked positions receive `-1e4`, attended positions `0.0`. The additive
/// mask is broadcast into the QKᵀ pre-softmax scores inside
/// `scaled_dot_product_attention`. `-1e4` (rather than `-inf`) matches
/// HuggingFace's `get_extended_attention_mask` convention and stays
/// numerically safe under fp16.
pub fn build_extended_attention_mask(mask: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    assert_eq!(shape.len(), 2, "expected [batch, seq_len], got {shape:?}");
    let mask_f = mask.to_dtype(DType::Float32)?;
    let additive = mask_f.mul_scalar(-1.0)?.add_scalar(1.0)?.mul_scalar(-1e4)?;
    additive.reshape(&[shape[0], 1, 1, shape[1]])
}

/// BERT hyperparameters. Matches the fields of a HuggingFace
/// `BertConfig` JSON file that affect model shape.
///
/// Use [`BertConfig::bert_base_uncased`] for the standard 12-layer / 768-dim
/// preset.
#[derive(Debug, Clone)]
pub struct BertConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub type_vocab_size: i64,
    /// Padding token index. `None` when `pad_token_id == eos_token_id` or
    /// when padding is handled entirely via the attention mask; `Some(i)`
    /// freezes the gradient on row `i` of the word-embedding table.
    pub pad_token_id: Option<i64>,
    pub layer_norm_eps: f64,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    /// Number of output labels for classification-style task heads. `None`
    /// on base `BertModel` configs; `Some(N)` when the checkpoint was fine-
    /// tuned as `BertForSequenceClassification`, `BertForTokenClassification`,
    /// etc. Derived from the HF `num_labels` field, or from the length of
    /// `id2label` if only the label map is present.
    pub num_labels: Option<i64>,
    /// Label strings indexed by class id (`id2label[k]` is the name of class
    /// `k`). `None` for base configs; `Some(vec)` for fine-tuned heads that
    /// shipped with an `id2label` / `label2id` mapping. Ordered by integer
    /// id so `vec[k]` reads like HF Python's `config.id2label[k]`.
    pub id2label: Option<Vec<String>>,
}

impl BertConfig {
    /// Preset matching `bert-base-uncased` on the HuggingFace Hub.
    pub fn bert_base_uncased() -> Self {
        BertConfig {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            pad_token_id: Some(0),
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            num_labels: None,
            id2label: None,
        }
    }

    /// Parse a HuggingFace-style `config.json` string into a [`BertConfig`].
    ///
    /// Reads the fields that affect model shape
    /// (`vocab_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
    /// `intermediate_size`, `max_position_embeddings`, `type_vocab_size`,
    /// `pad_token_id`, `layer_norm_eps`, `hidden_dropout_prob`,
    /// `attention_probs_dropout_prob`) plus the task-head metadata
    /// (`num_labels`, `id2label`) used by
    /// [`BertForSequenceClassification`] / [`BertForTokenClassification`] /
    /// [`BertForQuestionAnswering`]. Unknown fields are ignored, so adding
    /// new HF metadata (architecture lists, model type, torch dtype, …)
    /// doesn't break existing checkpoints.
    ///
    /// Required integer fields return a clear error if missing; dropout and
    /// layer-norm-eps fall back to the BERT defaults.
    ///
    /// `hidden_act` is not checked: BERT's default (`gelu`) is hardcoded in
    /// [`BertIntermediate`]. A config shipping a non-GELU activation will
    /// silently be run with GELU — acceptable for now since every
    /// BERT-family checkpoint on the Hub uses GELU.
    pub fn from_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| TensorError::new(&format!("config.json parse error: {e}")))?;
        let get_i64 = |key: &str| -> Result<i64> {
            v.get(key)
                .and_then(|x| x.as_i64())
                .ok_or_else(|| TensorError::new(&format!(
                    "config.json missing required integer field: {key}",
                )))
        };
        let get_f64_or = |key: &str, default: f64| -> f64 {
            v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
        };
        // pad_token_id is allowed to be absent or explicitly `null`.
        let pad_token_id = v.get("pad_token_id").and_then(|x| x.as_i64());

        // id2label: HF stores {"0": "LABEL_0", "1": "LABEL_1", ...}. Sort by
        // integer key so callers get a Vec<String> indexable by class id.
        let id2label = match v.get("id2label").and_then(|x| x.as_object()) {
            Some(obj) => {
                let mut pairs: Vec<(i64, String)> = Vec::with_capacity(obj.len());
                for (k, val) in obj {
                    let id: i64 = k.parse().map_err(|_| {
                        TensorError::new(&format!(
                            "config.json: id2label key {k:?} is not an integer",
                        ))
                    })?;
                    let label = val.as_str().ok_or_else(|| {
                        TensorError::new(&format!(
                            "config.json: id2label[{k}] is not a string",
                        ))
                    })?;
                    pairs.push((id, label.to_string()));
                }
                pairs.sort_by_key(|(id, _)| *id);
                // Require contiguous 0..N ids. A gap means the fine-tuner
                // did something unusual and falling back to "LABEL_k" would
                // silently misalign names with logits.
                for (idx, (id, _)) in pairs.iter().enumerate() {
                    if *id != idx as i64 {
                        return Err(TensorError::new(&format!(
                            "config.json: id2label must have contiguous ids 0..N, \
                             but index {idx} has id {id}",
                        )));
                    }
                }
                Some(pairs.into_iter().map(|(_, s)| s).collect::<Vec<_>>())
            }
            None => None,
        };
        // num_labels: explicit field wins; otherwise derive from id2label.
        let num_labels = v
            .get("num_labels")
            .and_then(|x| x.as_i64())
            .or_else(|| id2label.as_ref().map(|v| v.len() as i64));

        Ok(BertConfig {
            vocab_size:              get_i64("vocab_size")?,
            hidden_size:             get_i64("hidden_size")?,
            num_hidden_layers:       get_i64("num_hidden_layers")?,
            num_attention_heads:     get_i64("num_attention_heads")?,
            intermediate_size:       get_i64("intermediate_size")?,
            max_position_embeddings: get_i64("max_position_embeddings")?,
            type_vocab_size:         get_i64("type_vocab_size")?,
            pad_token_id,
            layer_norm_eps:               get_f64_or("layer_norm_eps", 1e-12),
            hidden_dropout_prob:          get_f64_or("hidden_dropout_prob", 0.1),
            attention_probs_dropout_prob: get_f64_or("attention_probs_dropout_prob", 0.1),
            num_labels,
            id2label,
        })
    }
}

// ── BertEmbeddings ───────────────────────────────────────────────────────

/// Token + position + token-type embeddings with post-LN and Dropout.
///
/// Implements [`NamedInputModule`] so the graph can feed `position_ids` and
/// `token_type_ids` alongside the main `input_ids` stream via
/// `FlowBuilder::using(&["position_ids", "token_type_ids"])`.
pub struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertEmbeddings {
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(BertEmbeddings {
            word_embeddings: Embedding::on_device_with_padding_idx(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
                device,
            )?,
            position_embeddings: Embedding::on_device(
                config.max_position_embeddings,
                config.hidden_size,
                device,
            )?,
            token_type_embeddings: Embedding::on_device(
                config.type_vocab_size,
                config.hidden_size,
                device,
            )?,
            layer_norm: LayerNorm::on_device_with_eps(
                config.hidden_size,
                config.layer_norm_eps,
                device,
            )?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }
}

impl Module for BertEmbeddings {
    fn name(&self) -> &str { "bert_embeddings" }

    /// Single-input forward path: word ids only. Position and token-type
    /// embeddings are skipped, which is useful for narrow unit tests but
    /// does NOT produce HF-equivalent outputs. The graph drives the full
    /// three-input path via `forward_named`.
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let word = self.word_embeddings.forward(input)?;
        let ln = self.layer_norm.forward(&word)?;
        self.dropout.forward(&ln)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params("word_embeddings", self.word_embeddings.parameters()));
        out.extend(prefix_params("position_embeddings", self.position_embeddings.parameters()));
        out.extend(prefix_params("token_type_embeddings", self.token_type_embeddings.parameters()));
        out.extend(prefix_params("LayerNorm", self.layer_norm.parameters()));
        out
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }

    fn set_training(&self, training: bool) {
        self.dropout.set_training(training);
    }
}

impl NamedInputModule for BertEmbeddings {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let mut summed = self.word_embeddings.forward(input)?;
        if let Some(pos) = refs.get("position_ids") {
            let pe = self.position_embeddings.forward(pos)?;
            summed = summed.add(&pe)?;
        }
        if let Some(tt) = refs.get("token_type_ids") {
            let te = self.token_type_embeddings.forward(tt)?;
            summed = summed.add(&te)?;
        }
        let ln = self.layer_norm.forward(&summed)?;
        self.dropout.forward(&ln)
    }
}

// ── BertPooler ───────────────────────────────────────────────────────────

/// Pooler: take the `[CLS]` token (index 0 along the sequence axis), pass
/// through a learned dense layer, then tanh.
///
/// Input shape: `[batch, seq_len, hidden]`. Output shape: `[batch, hidden]`.
pub struct BertPooler {
    dense: Linear,
}

impl BertPooler {
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(BertPooler {
            dense: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
        })
    }
}

impl Module for BertPooler {
    fn name(&self) -> &str { "bert_pooler" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // input: [batch, seq_len, hidden] → take index 0 along seq axis.
        let cls = input.select(1, 0)?;   // [batch, hidden]
        let pooled = self.dense.forward(&cls)?;
        pooled.tanh()
    }

    fn parameters(&self) -> Vec<Parameter> {
        prefix_params("dense", self.dense.parameters())
    }
}

// ── BertModel ────────────────────────────────────────────────────────────

/// Translate a [`BertConfig`] into the subset [`TransformerLayer`]
/// consumes. Localizes the field-name mapping in one place.
fn bert_layer_config(config: &BertConfig) -> TransformerLayerConfig {
    TransformerLayerConfig {
        hidden_size:                  config.hidden_size,
        num_attention_heads:          config.num_attention_heads,
        intermediate_size:            config.intermediate_size,
        hidden_dropout_prob:          config.hidden_dropout_prob,
        attention_probs_dropout_prob: config.attention_probs_dropout_prob,
        layer_norm_eps:               config.layer_norm_eps,
    }
}

/// Assemble the BERT backbone onto a fresh [`FlowBuilder`], up to and
/// optionally including the pooler.
///
/// Shared by [`BertModel`] and the task-head constructors — HF's
/// `add_pooling_layer=False` shortcut is just this helper with
/// `with_pooler=false`. Task heads that operate on `last_hidden_state`
/// (token classification, question answering) drop the pooler; those
/// that operate on the `[CLS]` vector (sequence classification) keep
/// it. Callers can `through()` their own head modules onto the returned
/// builder and then `.build()`.
///
/// Graph shape: `bert.embeddings` → `bert.encoder.layer.{0..N-1}` →
/// (`bert.pooler`?). Four named inputs are pre-declared:
/// `input_ids` (implicit first), `position_ids`, `token_type_ids`,
/// `attention_mask`. Every encoder layer pulls `attention_mask` via
/// `.using()`.
fn bert_backbone_flow(
    config: &BertConfig,
    device: Device,
    with_pooler: bool,
) -> Result<FlowBuilder> {
    let mut fb = FlowBuilder::new()
        .input(&["position_ids", "token_type_ids", "attention_mask"])
        .through(BertEmbeddings::on_device(config, device)?)
        .tag("bert.embeddings")
        .using(&["position_ids", "token_type_ids"]);

    let layer_root = HfPath::new("bert").sub("encoder").sub("layer");
    let layer_cfg = bert_layer_config(config);
    for i in 0..config.num_hidden_layers {
        let tag = layer_root.sub(i).to_string();
        fb = fb
            .through(TransformerLayer::on_device(&layer_cfg, LayerNaming::BERT, device)?)
            .tag(&tag)
            .using(&["attention_mask"]);
    }
    if with_pooler {
        fb = fb
            .through(BertPooler::on_device(config, device)?)
            .tag("bert.pooler");
    }
    Ok(fb)
}

/// Assembled BERT graph.
///
/// The returned [`Graph`] accepts four inputs via `forward_multi`, in
/// declaration order:
///
/// 1. `input_ids` (i64, shape `[batch, seq_len]`)
/// 2. `position_ids` (i64, shape `[batch, seq_len]`)
/// 3. `token_type_ids` (i64, shape `[batch, seq_len]`)
/// 4. `attention_mask` (f32, shape `[batch, 1, 1, seq_len]`, additive —
///    build with [`build_extended_attention_mask`] from a plain
///    `[batch, seq_len]` 0/1 mask)
///
/// Graph layout: `bert.embeddings` → `bert.encoder.layer.{0..N-1}` →
/// `bert.pooler`, where `N = config.num_hidden_layers`. Every encoder
/// layer pulls `attention_mask` via `.using()` so the same mask tensor
/// is shared across layers without re-materialising.
pub struct BertModel;

impl BertModel {
    /// Build a BERT graph on CPU.
    pub fn build(config: &BertConfig) -> Result<Graph> {
        Self::on_device(config, Device::CPU)
    }

    /// Build a BERT graph on `device`. Includes the pooler node; the
    /// returned graph emits `pooler_output` (`[batch, hidden]`).
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Graph> {
        bert_backbone_flow(config, device, true)?.build()
    }

    /// Build a BERT graph on `device` *without* the pooler. The returned
    /// graph emits `last_hidden_state` (`[batch, seq_len, hidden]`) —
    /// the shape the token-classification and question-answering heads
    /// consume. Matches HF Python's `BertModel(config, add_pooling_layer=False)`.
    pub fn on_device_without_pooler(config: &BertConfig, device: Device) -> Result<Graph> {
        bert_backbone_flow(config, device, false)?.build()
    }
}

// ── Task heads ───────────────────────────────────────────────────────────

/// Default label names when a checkpoint ships without `id2label`.
/// Mirrors HF Python's `LABEL_0`, `LABEL_1`, ...
fn default_labels(n: i64) -> Vec<String> {
    (0..n).map(|i| format!("LABEL_{i}")).collect()
}

/// Clamp `num_labels` to a positive value or surface a loud error.
fn check_num_labels(n: i64) -> Result<i64> {
    if n <= 0 {
        return Err(TensorError::new(&format!(
            "num_labels must be > 0, got {n}",
        )));
    }
    Ok(n)
}

/// BERT with a sequence-classification head on top of the pooled
/// `[CLS]` output: `pooler_output → Dropout → Linear(hidden, num_labels)`.
///
/// Parameter keys for the head:
/// - `classifier.weight`  (`[num_labels, hidden]`)
/// - `classifier.bias`    (`[num_labels]`)
///
/// Matches HF Python's
/// [`BertForSequenceClassification`](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification).
/// Pre-trained checkpoints: `nateraw/bert-base-uncased-emotion` (6 emotions,
/// requires `fdl flodl-hf convert` first for `.bin`-only repos),
/// `nlptown/bert-base-multilingual-uncased-sentiment` (5-star rating),
/// `unitary/toxic-bert` (6-label toxicity).
pub struct BertForSequenceClassification {
    graph: Graph,
    id2label: Vec<String>,
    #[cfg(feature = "tokenizer")]
    tokenizer: Option<crate::tokenizer::HfTokenizer>,
}

impl BertForSequenceClassification {
    /// Build the full graph (backbone + classifier head) on `device`
    /// without loading any weights. `num_labels` determines the head's
    /// output dimension; `id2label` falls back to `["LABEL_0", ...]`.
    pub fn on_device(
        config: &BertConfig,
        num_labels: i64,
        device: Device,
    ) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = bert_backbone_flow(config, device, /*with_pooler=*/ true)?
            .through(Dropout::new(config.hidden_dropout_prob))
            .through(Linear::on_device(config.hidden_size, num_labels, device)?)
            .tag("classifier")
            .build()?;
        let id2label = config
            .id2label
            .clone()
            .unwrap_or_else(|| default_labels(num_labels));
        Ok(Self {
            graph,
            id2label,
            #[cfg(feature = "tokenizer")]
            tokenizer: None,
        })
    }

    /// Resolve `num_labels` from config if present; error otherwise.
    /// Used by `from_pretrained` paths where the config must carry
    /// head metadata.
    pub(crate) fn num_labels_from_config(config: &BertConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "BertForSequenceClassification: config.json has no `num_labels` \
                 (nor `id2label`); cannot infer head size",
            )
        })
    }

    /// Borrow the underlying [`Graph`] for direct inspection, inference
    /// via `forward_multi`, or custom training loops.
    pub fn graph(&self) -> &Graph { &self.graph }

    /// Label names indexed by class id. `labels()[k]` is the name of
    /// class `k` — either from the checkpoint's `id2label` or the
    /// `LABEL_k` fallback.
    pub fn labels(&self) -> &[String] { &self.id2label }

    /// Attach a tokenizer so [`predict`](Self::predict) can encode raw
    /// text. `from_pretrained` does this automatically.
    #[cfg(feature = "tokenizer")]
    pub fn with_tokenizer(mut self, tok: crate::tokenizer::HfTokenizer) -> Self {
        self.tokenizer = Some(tok);
        self
    }

    /// Classify a pre-tokenised batch. Returns one label distribution per
    /// input, sorted by descending probability.
    #[cfg(feature = "tokenizer")]
    pub fn classify(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        let logits = self.forward_from_encoded(enc)?;
        logits_to_sorted_labels(&logits, &self.id2label)
    }

    /// One-shot text → label distribution. Encodes with the attached
    /// tokenizer, runs the graph in eval mode, softmaxes the logits, and
    /// returns per-input label distributions sorted desc.
    #[cfg(feature = "tokenizer")]
    pub fn predict(&self, texts: &[&str]) -> Result<Vec<Vec<(String, f32)>>> {
        let tok = self.tokenizer.as_ref().ok_or_else(|| {
            TensorError::new(
                "BertForSequenceClassification::predict requires a tokenizer; \
                 use from_pretrained or .with_tokenizer(...) first",
            )
        })?;
        let enc = tok.encode(texts)?;
        self.classify(&enc)
    }

    #[cfg(feature = "tokenizer")]
    fn forward_from_encoded(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
    ) -> Result<Variable> {
        self.graph.eval();
        let mask_f32 = enc.attention_mask.data().to_dtype(DType::Float32)?;
        let mask = Variable::new(build_extended_attention_mask(&mask_f32)?, false);
        self.graph.forward_multi(&[
            enc.input_ids.clone(),
            enc.position_ids.clone(),
            enc.token_type_ids.clone(),
            mask,
        ])
    }
}

/// Apply softmax to a `[batch, num_labels]` logits tensor and return a
/// sorted `(label, score)` list per batch entry.
fn logits_to_sorted_labels(
    logits: &Variable,
    id2label: &[String],
) -> Result<Vec<Vec<(String, f32)>>> {
    let probs = logits.softmax(-1)?;
    let shape = probs.shape();
    assert_eq!(shape.len(), 2, "expected [batch, num_labels], got {shape:?}");
    let batch = shape[0] as usize;
    let n = shape[1] as usize;
    assert_eq!(
        n,
        id2label.len(),
        "classifier output width {n} != id2label count {}",
        id2label.len(),
    );
    let flat = probs.data().to_f32_vec()?;
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut row: Vec<(String, f32)> = (0..n)
            .map(|k| (id2label[k].clone(), flat[b * n + k]))
            .collect();
        row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.push(row);
    }
    Ok(out)
}

/// One labelled token inside a [`BertForTokenClassification`] prediction.
#[cfg(feature = "tokenizer")]
#[derive(Debug, Clone)]
pub struct TokenPrediction {
    /// The subword string the tokenizer produced for this position.
    pub token: String,
    /// The highest-scoring label for this token.
    pub label: String,
    /// Softmax probability of `label`.
    pub score: f32,
    /// 1 for real tokens, 0 for padding (`attention_mask == 0`). Lets
    /// callers drop padding without re-tokenising.
    pub attends: bool,
}

/// BERT with a per-token classification head: `last_hidden_state →
/// Dropout → Linear(hidden, num_labels)`. Typical use is NER, POS
/// tagging, or any sequence labelling task.
///
/// Parameter keys for the head:
/// - `classifier.weight`  (`[num_labels, hidden]`)
/// - `classifier.bias`    (`[num_labels]`)
///
/// Matches HF Python's `BertForTokenClassification`. Pre-trained
/// checkpoints: `dslim/bert-base-NER`,
/// `dbmdz/bert-large-cased-finetuned-conll03-english`, etc.
pub struct BertForTokenClassification {
    graph: Graph,
    id2label: Vec<String>,
    #[cfg(feature = "tokenizer")]
    tokenizer: Option<crate::tokenizer::HfTokenizer>,
}

impl BertForTokenClassification {
    /// Build the full graph (backbone without pooler + classifier head).
    pub fn on_device(
        config: &BertConfig,
        num_labels: i64,
        device: Device,
    ) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = bert_backbone_flow(config, device, /*with_pooler=*/ false)?
            .through(Dropout::new(config.hidden_dropout_prob))
            .through(Linear::on_device(config.hidden_size, num_labels, device)?)
            .tag("classifier")
            .build()?;
        let id2label = config
            .id2label
            .clone()
            .unwrap_or_else(|| default_labels(num_labels));
        Ok(Self {
            graph,
            id2label,
            #[cfg(feature = "tokenizer")]
            tokenizer: None,
        })
    }

    pub(crate) fn num_labels_from_config(config: &BertConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "BertForTokenClassification: config.json has no `num_labels` \
                 (nor `id2label`); cannot infer head size",
            )
        })
    }

    pub fn graph(&self) -> &Graph { &self.graph }
    pub fn labels(&self) -> &[String] { &self.id2label }

    #[cfg(feature = "tokenizer")]
    pub fn with_tokenizer(mut self, tok: crate::tokenizer::HfTokenizer) -> Self {
        self.tokenizer = Some(tok);
        self
    }

    /// Tag every token in a pre-tokenised batch. Output shape matches
    /// `enc.input_ids`: `result[b][s]` is the top-1 prediction for batch
    /// entry `b`, position `s`.
    ///
    /// `TokenPrediction::attends` mirrors the input attention mask so
    /// callers can drop `[PAD]` entries without re-tokenising.
    #[cfg(feature = "tokenizer")]
    pub fn tag(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
    ) -> Result<Vec<Vec<TokenPrediction>>> {
        let tok = self.tokenizer.as_ref().ok_or_else(|| {
            TensorError::new(
                "BertForTokenClassification::tag requires a tokenizer; \
                 attach one via .with_tokenizer(...) or from_pretrained",
            )
        })?;
        self.graph.eval();
        let mask_f32 = enc.attention_mask.data().to_dtype(DType::Float32)?;
        let mask = Variable::new(build_extended_attention_mask(&mask_f32)?, false);
        let logits = self.graph.forward_multi(&[
            enc.input_ids.clone(),
            enc.position_ids.clone(),
            enc.token_type_ids.clone(),
            mask,
        ])?;
        // logits: [B, S, num_labels]
        let probs = logits.softmax(-1)?;
        let shape = probs.shape();
        assert_eq!(shape.len(), 3, "expected [B, S, num_labels], got {shape:?}");
        let batch = shape[0] as usize;
        let seq = shape[1] as usize;
        let n = shape[2] as usize;
        let flat = probs.data().to_f32_vec()?;
        let input_ids: Vec<i64> = enc.input_ids.data().to_i64_vec()?;
        let attn_ids: Vec<i64> = enc.attention_mask.data().to_i64_vec()?;

        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut row = Vec::with_capacity(seq);
            for s in 0..seq {
                let base = (b * seq + s) * n;
                let (best_k, &best_p) = flat[base..base + n]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .expect("n > 0 checked by check_num_labels");
                let id = input_ids[b * seq + s] as u32;
                let token = tok
                    .inner()
                    .id_to_token(id)
                    .unwrap_or_else(|| format!("<unk_id={id}>"));
                row.push(TokenPrediction {
                    token,
                    label: self.id2label[best_k].clone(),
                    score: best_p,
                    attends: attn_ids[b * seq + s] != 0,
                });
            }
            out.push(row);
        }
        Ok(out)
    }

    /// One-shot text → per-token tags. Encodes with the attached
    /// tokenizer and calls [`tag`](Self::tag).
    #[cfg(feature = "tokenizer")]
    pub fn predict(&self, texts: &[&str]) -> Result<Vec<Vec<TokenPrediction>>> {
        let tok = self.tokenizer.as_ref().ok_or_else(|| {
            TensorError::new(
                "BertForTokenClassification::predict requires a tokenizer; \
                 use from_pretrained or .with_tokenizer(...) first",
            )
        })?;
        let enc = tok.encode(texts)?;
        self.tag(&enc)
    }
}

/// Extracted answer span from a [`BertForQuestionAnswering`] prediction.
#[cfg(feature = "tokenizer")]
#[derive(Debug, Clone)]
pub struct Answer {
    /// Decoded answer text. Built from the token ids between `start`
    /// and `end` (inclusive) via the attached tokenizer's `decode`.
    pub text: String,
    /// Index (inclusive) of the first answer token in the input sequence.
    pub start: usize,
    /// Index (inclusive) of the last answer token in the input sequence.
    pub end: usize,
    /// Sum of the start and end logit probabilities — higher is more
    /// confident. Not a normalised probability; callers comparing across
    /// models should treat this as a relative score.
    pub score: f32,
}

/// BERT with an extractive question-answering head: `last_hidden_state →
/// Linear(hidden, 2)` splitting into `start_logits` and `end_logits`.
///
/// Parameter keys for the head:
/// - `qa_outputs.weight` (`[2, hidden]`)
/// - `qa_outputs.bias`   (`[2]`)
///
/// Matches HF Python's `BertForQuestionAnswering`. Pre-trained
/// checkpoints: `csarron/bert-base-uncased-squad-v1`,
/// `bert-large-uncased-whole-word-masking-finetuned-squad`, etc.
pub struct BertForQuestionAnswering {
    graph: Graph,
    #[cfg(feature = "tokenizer")]
    tokenizer: Option<crate::tokenizer::HfTokenizer>,
}

impl BertForQuestionAnswering {
    /// Build the full graph (backbone without pooler + QA output head).
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        // QA is a fixed-width head: 2 outputs (start, end), independent
        // of num_labels. Hardcoding it here matches HF Python.
        let graph = bert_backbone_flow(config, device, /*with_pooler=*/ false)?
            .through(Linear::on_device(config.hidden_size, 2, device)?)
            .tag("qa_outputs")
            .build()?;
        Ok(Self {
            graph,
            #[cfg(feature = "tokenizer")]
            tokenizer: None,
        })
    }

    pub fn graph(&self) -> &Graph { &self.graph }

    #[cfg(feature = "tokenizer")]
    pub fn with_tokenizer(mut self, tok: crate::tokenizer::HfTokenizer) -> Self {
        self.tokenizer = Some(tok);
        self
    }

    /// Answer one `(question, context)` pair. Returns the highest-scoring
    /// span over the context tokens.
    #[cfg(feature = "tokenizer")]
    pub fn answer(&self, question: &str, context: &str) -> Result<Answer> {
        let mut out = self.answer_batch(&[(question, context)])?;
        Ok(out.pop().expect("answer_batch returns one per input"))
    }

    /// Batched variant of [`answer`](Self::answer).
    #[cfg(feature = "tokenizer")]
    pub fn answer_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<Answer>> {
        let tok = self.tokenizer.as_ref().ok_or_else(|| {
            TensorError::new(
                "BertForQuestionAnswering::answer requires a tokenizer; \
                 use from_pretrained or .with_tokenizer(...) first",
            )
        })?;
        let enc = tok.encode_pairs(pairs)?;
        self.extract(&enc)
    }

    /// Run the graph on a pre-tokenised `(question, context)` batch and
    /// extract best spans. The tokenizer's `sequence_ids` mark the
    /// context region (== 1); scoring is restricted to that region.
    ///
    /// Uses `sequence_ids` (the tokenizer-level segment tag: `0` =
    /// question, `1` = context, `-1` = special/pad) rather than
    /// `token_type_ids` so the same code works across BERT-family
    /// models with different token-type conventions (RoBERTa, for
    /// instance, keeps all `token_type_ids` at zero). `-1` already
    /// excludes padding, so no separate attention-mask check is needed.
    #[cfg(feature = "tokenizer")]
    pub fn extract(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
    ) -> Result<Vec<Answer>> {
        let tok = self.tokenizer.as_ref().ok_or_else(|| {
            TensorError::new(
                "BertForQuestionAnswering::extract requires a tokenizer; \
                 attach one via .with_tokenizer(...) or from_pretrained",
            )
        })?;
        self.graph.eval();
        let mask_f32 = enc.attention_mask.data().to_dtype(DType::Float32)?;
        let mask = Variable::new(build_extended_attention_mask(&mask_f32)?, false);
        let logits = self.graph.forward_multi(&[
            enc.input_ids.clone(),
            enc.position_ids.clone(),
            enc.token_type_ids.clone(),
            mask,
        ])?;
        // logits: [B, S, 2] — dim 2 is (start, end).
        let shape = logits.shape();
        assert_eq!(shape.len(), 3, "expected [B, S, 2], got {shape:?}");
        let batch = shape[0] as usize;
        let seq = shape[1] as usize;
        assert_eq!(shape[2], 2, "QA head must be 2-wide, got {}", shape[2]);

        // Softmax start and end independently along the sequence axis;
        // matches HF's "argmax over valid positions, then multiply".
        // Splitting `[B, S, 2]` along the last dim is two narrows.
        let starts = logits.narrow(-1, 0, 1)?.softmax(1)?;
        let ends = logits.narrow(-1, 1, 1)?.softmax(1)?;
        let starts_flat = starts.data().to_f32_vec()?;
        let ends_flat = ends.data().to_f32_vec()?;
        let sequence_ids: Vec<i64> = enc.sequence_ids.data().to_i64_vec()?;
        let input_ids: Vec<i64> = enc.input_ids.data().to_i64_vec()?;

        let mut answers = Vec::with_capacity(batch);
        for b in 0..batch {
            let offset = b * seq;
            // Valid span positions: context tokens (sequence_id == 1).
            // `-1` (specials / padding) is automatically excluded.
            let valid: Vec<usize> = (0..seq)
                .filter(|&s| sequence_ids[offset + s] == 1)
                .collect();
            if valid.is_empty() {
                // Pair had no context region; fall back to full sequence.
                return Err(TensorError::new(
                    "QA extract: no context tokens (sequence_id == 1) found; \
                     tokenizer did not produce a pair encoding",
                ));
            }
            // Best (start, end) with start <= end, restricted to valid.
            let mut best = (valid[0], valid[0], f32::NEG_INFINITY);
            for &i in &valid {
                let sp = starts_flat[offset + i];
                for &j in valid.iter().filter(|&&j| j >= i) {
                    let ep = ends_flat[offset + j];
                    let score = sp + ep;
                    if score > best.2 {
                        best = (i, j, score);
                    }
                }
            }
            let (start, end, score) = best;
            let span_ids: Vec<u32> = input_ids[offset + start..=offset + end]
                .iter()
                .map(|&x| x as u32)
                .collect();
            let text = tok
                .inner()
                .decode(&span_ids, /*skip_special_tokens=*/ true)
                .map_err(|e| TensorError::new(&format!("qa decode: {e}")))?;
            answers.push(Answer { text, start, end, score });
        }
        Ok(answers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors_io::expected_from_graph;
    use flodl::TensorOptions;

    /// The 16 parameter keys every encoder layer exposes, template-formatted
    /// for a given layer index.
    fn expected_layer_keys(i: i64) -> Vec<String> {
        let suffixes = [
            "attention.output.LayerNorm.bias",
            "attention.output.LayerNorm.weight",
            "attention.output.dense.bias",
            "attention.output.dense.weight",
            "attention.self.key.bias",
            "attention.self.key.weight",
            "attention.self.query.bias",
            "attention.self.query.weight",
            "attention.self.value.bias",
            "attention.self.value.weight",
            "intermediate.dense.bias",
            "intermediate.dense.weight",
            "output.LayerNorm.bias",
            "output.LayerNorm.weight",
            "output.dense.bias",
            "output.dense.weight",
        ];
        suffixes.iter().map(|s| format!("bert.encoder.layer.{i}.{s}")).collect()
    }

    /// The full BERT-base key set (199 keys: 5 embeddings + 16×12 layers +
    /// 2 pooler) matches HuggingFace safetensors dotted form exactly. The
    /// expected list is built from the layer template so adding or removing
    /// layers only changes the config, not the test.
    #[test]
    fn bert_parameter_keys_match_hf_dotted_form() {
        let config = BertConfig::bert_base_uncased();
        let graph = BertModel::build(&config).unwrap();
        let expected = expected_from_graph(&graph);

        let mut keys: Vec<String> = expected.iter().map(|p| p.key.clone()).collect();
        keys.sort();

        let mut want: Vec<String> = vec![
            "bert.embeddings.LayerNorm.bias".into(),
            "bert.embeddings.LayerNorm.weight".into(),
            "bert.embeddings.position_embeddings.weight".into(),
            "bert.embeddings.token_type_embeddings.weight".into(),
            "bert.embeddings.word_embeddings.weight".into(),
        ];
        for i in 0..config.num_hidden_layers {
            want.extend(expected_layer_keys(i));
        }
        want.extend([
            "bert.pooler.dense.bias".into(),
            "bert.pooler.dense.weight".into(),
        ]);
        want.sort();

        // Sanity: BERT-base has exactly 199 parameter tensors.
        assert_eq!(want.len(), 199, "expected-key list size drift");
        assert_eq!(keys, want, "BERT parameter keys must match HF exactly");
    }

    /// Parameter shapes must match the BERT-base-uncased reference. If any
    /// constant in `BertConfig::bert_base_uncased` regresses, this test
    /// pins it.
    #[test]
    fn bert_parameter_shapes_match_bert_base_uncased() {
        let config = BertConfig::bert_base_uncased();
        let graph = BertModel::build(&config).unwrap();
        let expected = expected_from_graph(&graph);
        let by_key: std::collections::HashMap<&str, &[i64]> = expected
            .iter()
            .map(|p| (p.key.as_str(), p.shape.as_slice()))
            .collect();

        assert_eq!(by_key["bert.embeddings.word_embeddings.weight"],       &[30522, 768]);
        assert_eq!(by_key["bert.embeddings.position_embeddings.weight"],   &[512, 768]);
        assert_eq!(by_key["bert.embeddings.token_type_embeddings.weight"], &[2, 768]);
        assert_eq!(by_key["bert.embeddings.LayerNorm.weight"],             &[768]);
        assert_eq!(by_key["bert.embeddings.LayerNorm.bias"],               &[768]);

        // Every encoder layer has the same shape profile. Sweep all
        // layers so any mis-wiring on a specific index surfaces here
        // rather than being masked by only checking layer 0.
        for i in 0..config.num_hidden_layers {
            let p = format!("bert.encoder.layer.{i}");
            assert_eq!(by_key[&*format!("{p}.attention.self.query.weight")],        &[768, 768]);
            assert_eq!(by_key[&*format!("{p}.attention.self.query.bias")],          &[768]);
            assert_eq!(by_key[&*format!("{p}.attention.self.key.weight")],          &[768, 768]);
            assert_eq!(by_key[&*format!("{p}.attention.self.value.weight")],        &[768, 768]);
            assert_eq!(by_key[&*format!("{p}.attention.output.dense.weight")],      &[768, 768]);
            assert_eq!(by_key[&*format!("{p}.attention.output.LayerNorm.weight")],  &[768]);
            assert_eq!(by_key[&*format!("{p}.intermediate.dense.weight")],          &[3072, 768]);
            assert_eq!(by_key[&*format!("{p}.intermediate.dense.bias")],            &[3072]);
            assert_eq!(by_key[&*format!("{p}.output.dense.weight")],                &[768, 3072]);
            assert_eq!(by_key[&*format!("{p}.output.dense.bias")],                  &[768]);
            assert_eq!(by_key[&*format!("{p}.output.LayerNorm.weight")],            &[768]);
        }

        assert_eq!(by_key["bert.pooler.dense.weight"], &[768, 768]);
        assert_eq!(by_key["bert.pooler.dense.bias"],   &[768]);
    }

    /// Encoder stack honours `config.num_hidden_layers`. Pins the loop so
    /// a future regression that (e.g.) wires in one hardcoded layer is
    /// caught immediately.
    #[test]
    fn bert_layer_count_scales_with_config() {
        for n in [1_i64, 3, 6] {
            let config = BertConfig {
                num_hidden_layers: n,
                ..BertConfig::bert_base_uncased()
            };
            let graph = BertModel::build(&config).unwrap();
            let expected = expected_from_graph(&graph);
            let total = expected.len();
            // 5 embedding keys + 16 per layer + 2 pooler keys.
            let want_total = 5 + 16 * n as usize + 2;
            assert_eq!(
                total, want_total,
                "num_hidden_layers={n}: got {total} keys, expected {want_total}",
            );

            // The highest-indexed layer actually exists — catches "stopped
            // one short" / "started at 1" bugs in the loop.
            let last_layer_key = format!(
                "bert.encoder.layer.{}.attention.self.query.weight", n - 1,
            );
            assert!(
                expected.iter().any(|p| p.key == last_layer_key),
                "last layer key {last_layer_key:?} missing from graph keys",
            );
        }
    }

    /// Small BERT preset the mask tests reuse. One layer, tiny hidden, no
    /// dropout — enough wiring to exercise embeddings + encoder + pooler
    /// without the cost of `bert-base`.
    fn tiny_bert_config() -> BertConfig {
        BertConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            intermediate_size: 32,
            max_position_embeddings: 8,
            type_vocab_size: 2,
            pad_token_id: Some(0),
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            num_labels: None,
            id2label: None,
        }
    }

    /// Standard `bert-base-uncased` config.json — round-trip through
    /// `from_json_str` must produce a config that matches the hardcoded
    /// `bert_base_uncased()` preset. Exercises every required field plus
    /// a few optional ones and unknown-field tolerance.
    #[test]
    fn bert_config_from_json_str_matches_base_preset() {
        let json = r#"{
            "architectures": ["BertForMaskedLM"],
            "attention_probs_dropout_prob": 0.1,
            "gradient_checkpointing": false,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.6.0.dev0",
            "type_vocab_size": 2,
            "use_cache": true,
            "vocab_size": 30522
        }"#;
        let got = BertConfig::from_json_str(json).unwrap();
        let want = BertConfig::bert_base_uncased();
        assert_eq!(got.vocab_size,              want.vocab_size);
        assert_eq!(got.hidden_size,             want.hidden_size);
        assert_eq!(got.num_hidden_layers,       want.num_hidden_layers);
        assert_eq!(got.num_attention_heads,     want.num_attention_heads);
        assert_eq!(got.intermediate_size,       want.intermediate_size);
        assert_eq!(got.max_position_embeddings, want.max_position_embeddings);
        assert_eq!(got.type_vocab_size,         want.type_vocab_size);
        assert_eq!(got.pad_token_id,            want.pad_token_id);
        assert!((got.layer_norm_eps               - want.layer_norm_eps).abs() < 1e-18);
        assert!((got.hidden_dropout_prob          - want.hidden_dropout_prob).abs() < 1e-9);
        assert!((got.attention_probs_dropout_prob - want.attention_probs_dropout_prob).abs() < 1e-9);
    }

    /// Missing a required integer field must surface a clear error that
    /// names the offending key — the whole point of the validator is to
    /// be loud about drift.
    #[test]
    fn bert_config_from_json_str_rejects_missing_field() {
        // No `hidden_size`.
        let json = r#"{
            "vocab_size": 30522,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "type_vocab_size": 2
        }"#;
        let err = BertConfig::from_json_str(json).unwrap_err().to_string();
        assert!(err.contains("hidden_size"),
            "error must name the missing field: {err}");
        assert!(err.contains("missing required integer field"),
            "error must explain the failure mode: {err}");
    }

    /// Explicit `"pad_token_id": null` and absent `pad_token_id` must both
    /// produce `None` — these are the two ways HF configs spell "no
    /// dedicated pad token" (e.g. GPT-2-style).
    #[test]
    fn bert_config_from_json_str_pad_token_id_nullable() {
        let required_fields = r#"
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "type_vocab_size": 2
        "#;
        let explicit_null = format!(r#"{{ {required_fields}, "pad_token_id": null }}"#);
        let absent        = format!(r#"{{ {required_fields} }}"#);
        let a = BertConfig::from_json_str(&explicit_null).unwrap();
        let b = BertConfig::from_json_str(&absent).unwrap();
        assert_eq!(a.pad_token_id, None);
        assert_eq!(b.pad_token_id, None);
    }

    /// `num_labels` + `id2label` parse correctly from a fine-tuned
    /// checkpoint's config. Labels are ordered by integer id so `Vec[k]`
    /// reads as `id2label[k]`, and `num_labels` tracks the entry count.
    #[test]
    fn bert_config_from_json_str_parses_task_head_metadata() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "num_labels": 3,
            "id2label": { "2": "JOY", "0": "ANGER", "1": "SADNESS" },
            "label2label": { "IGNORED": 1 }
        }"#;
        let c = BertConfig::from_json_str(json).unwrap();
        assert_eq!(c.num_labels, Some(3));
        assert_eq!(
            c.id2label,
            Some(vec!["ANGER".to_string(), "SADNESS".to_string(), "JOY".to_string()]),
        );
    }

    /// If only `id2label` is present (some older fine-tunes omit
    /// `num_labels`), `num_labels` is derived from the label count.
    #[test]
    fn bert_config_num_labels_derived_from_id2label() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "id2label": { "0": "NEGATIVE", "1": "POSITIVE" }
        }"#;
        let c = BertConfig::from_json_str(json).unwrap();
        assert_eq!(c.num_labels, Some(2));
        assert_eq!(c.id2label.unwrap(), vec!["NEGATIVE", "POSITIVE"]);
    }

    /// Base configs without any task-head metadata leave both fields as
    /// `None`. Task-head constructors pick sensible fallbacks from the
    /// runtime `num_labels` argument in that case.
    #[test]
    fn bert_config_without_task_metadata_is_none() {
        let c = BertConfig::bert_base_uncased();
        assert_eq!(c.num_labels, None);
        assert_eq!(c.id2label, None);
    }

    /// Non-contiguous label ids (gap, duplicate, or negative) must surface
    /// as a clear error. Silently reindexing would misalign class names
    /// with logits row indices on load.
    #[test]
    fn bert_config_rejects_non_contiguous_id2label() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "id2label": { "0": "A", "2": "C" }
        }"#;
        let err = BertConfig::from_json_str(json).unwrap_err().to_string();
        assert!(err.contains("contiguous"), "error must call out contiguity: {err}");
    }

    /// Optional dropout + layer-norm-eps fields fall back to BERT defaults
    /// when absent. Keeps configs for bare-metal / test-only checkpoints
    /// parseable without boilerplate.
    #[test]
    fn bert_config_from_json_str_uses_defaults_for_missing_optional_fields() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "type_vocab_size": 2
        }"#;
        let c = BertConfig::from_json_str(json).unwrap();
        assert!((c.layer_norm_eps               - 1e-12).abs() < 1e-18);
        assert!((c.hidden_dropout_prob          - 0.1).abs() < 1e-9);
        assert!((c.attention_probs_dropout_prob - 0.1).abs() < 1e-9);
    }

    /// Smoke test: construct a tiny BERT on CPU, run forward_multi with
    /// made-up ids + an all-attend mask, and verify the output shape.
    /// Catches obvious wiring breakage (residual mismatch, missing named
    /// input, transpose axis bug) without requiring real tokenized inputs.
    #[test]
    fn bert_forward_shape_smoke() {
        let config = tiny_bert_config();
        let dev = Device::CPU;
        let graph = BertModel::on_device(&config, dev).unwrap();
        graph.eval();

        let batch = 2;
        let seq = 4;
        let word_ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3, 4, 5, 6, 7, 0], &[batch, seq], dev).unwrap(),
            false,
        );
        let position_ids = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3, 0, 1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let token_type_ids = Variable::new(
            Tensor::from_i64(&[0, 0, 0, 0, 1, 1, 1, 1], &[batch, seq], dev).unwrap(),
            false,
        );
        let mask_flat = Tensor::ones(&[batch, seq], TensorOptions { dtype: DType::Float32, device: dev }).unwrap();
        let attention_mask = Variable::new(
            build_extended_attention_mask(&mask_flat).unwrap(),
            false,
        );

        let out = graph
            .forward_multi(&[word_ids, position_ids, token_type_ids, attention_mask])
            .unwrap();
        // Pooler reduces [batch, seq, hidden] → [batch, hidden]
        assert_eq!(out.shape(), vec![batch, config.hidden_size]);
    }

    /// `build_extended_attention_mask` turns a `[B, S]` 0/1 mask into a
    /// `[B, 1, 1, S]` additive f32 mask: attend positions → `0.0`, mask
    /// positions → `-1e4`. Pins the HF convention so Phase-2 checkpoint
    /// parity doesn't drift when we compare against PyTorch.
    #[test]
    fn extended_attention_mask_shape_and_values() {
        let dev = Device::CPU;
        // [2, 3] with batch-0 fully attending and batch-1 masking the
        // trailing position (e.g. padding after [CLS] tok tok).
        let raw = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0, 1.0, 0.0], &[2, 3], dev).unwrap();
        let additive = build_extended_attention_mask(&raw).unwrap();
        assert_eq!(additive.shape(), vec![2, 1, 1, 3]);

        let values: Vec<f32> = additive.reshape(&[6]).unwrap().to_f32_vec().unwrap();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], 0.0);
        assert_eq!(values[3], 0.0);
        assert_eq!(values[4], 0.0);
        assert!((values[5] - -1e4).abs() < 1e-3, "masked position should be ~-1e4, got {}", values[5]);
    }
    // ── Task head tests ──────────────────────────────────────────────────

    /// `BertModel::on_device_without_pooler` drops the last two parameter
    /// tensors (`bert.pooler.dense.{weight,bias}`). The remaining 197
    /// keys stay in lockstep with the pooled backbone — critical for
    /// checkpoint loading when task heads sit on top of
    /// `add_pooling_layer=False`.
    #[test]
    fn bert_without_pooler_drops_two_keys() {
        let config = BertConfig::bert_base_uncased();
        let graph = BertModel::on_device_without_pooler(&config, Device::CPU).unwrap();
        let expected = expected_from_graph(&graph);
        let keys: Vec<&str> = expected.iter().map(|p| p.key.as_str()).collect();

        assert_eq!(expected.len(), 197, "197 backbone keys expected");
        assert!(!keys.iter().any(|k| k.starts_with("bert.pooler.")));
    }

    /// `BertForSequenceClassification` adds exactly two classifier keys
    /// on top of a pooled backbone.
    #[test]
    fn sequence_classification_parameter_keys_match_hf() {
        let config = BertConfig::bert_base_uncased();
        let head = BertForSequenceClassification::on_device(&config, 3, Device::CPU).unwrap();
        let expected = expected_from_graph(head.graph());
        let mut head_keys: Vec<&str> = expected
            .iter()
            .map(|p| p.key.as_str())
            .filter(|k| !k.starts_with("bert."))
            .collect();
        head_keys.sort();
        assert_eq!(head_keys, vec!["classifier.bias", "classifier.weight"]);

        let by_key: std::collections::HashMap<&str, &[i64]> = expected
            .iter().map(|p| (p.key.as_str(), p.shape.as_slice())).collect();
        assert_eq!(by_key["classifier.weight"], &[3, 768]);
        assert_eq!(by_key["classifier.bias"],   &[3]);
    }

    /// The config's `id2label` (if present) flows through to `labels()`.
    /// Otherwise the `LABEL_k` fallback kicks in.
    #[test]
    fn sequence_classification_labels_from_config_or_fallback() {
        let mut cfg = BertConfig::bert_base_uncased();
        cfg.num_labels = Some(3);
        cfg.id2label = Some(vec!["A".into(), "B".into(), "C".into()]);
        let head = BertForSequenceClassification::on_device(&cfg, 3, Device::CPU).unwrap();
        assert_eq!(head.labels(), &["A".to_string(), "B".to_string(), "C".to_string()]);

        let bare = BertConfig::bert_base_uncased();
        let fallback = BertForSequenceClassification::on_device(&bare, 2, Device::CPU).unwrap();
        assert_eq!(fallback.labels(), &["LABEL_0".to_string(), "LABEL_1".to_string()]);
    }

    /// Smoke: forward through the full classification graph produces
    /// `[batch, num_labels]` logits.
    #[test]
    fn sequence_classification_forward_shape_smoke() {
        let config = tiny_bert_config();
        let dev = Device::CPU;
        let head = BertForSequenceClassification::on_device(&config, 5, dev).unwrap();
        head.graph().eval();

        let batch = 2;
        let seq = 4;
        let ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3, 4, 5, 6, 7, 0], &[batch, seq], dev).unwrap(),
            false,
        );
        let pos = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3, 0, 1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let tt = Variable::new(
            Tensor::from_i64(&[0; 8], &[batch, seq], dev).unwrap(),
            false,
        );
        let mask_flat = Tensor::ones(&[batch, seq], TensorOptions {
            dtype: DType::Float32, device: dev,
        }).unwrap();
        let mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

        let out = head.graph().forward_multi(&[ids, pos, tt, mask]).unwrap();
        assert_eq!(out.shape(), vec![batch, 5]);
    }

    /// `BertForTokenClassification` adds two classifier keys on top of
    /// an un-pooled backbone. Pooler must be absent.
    #[test]
    fn token_classification_parameter_keys_match_hf() {
        let config = BertConfig::bert_base_uncased();
        let head = BertForTokenClassification::on_device(&config, 9, Device::CPU).unwrap();
        let expected = expected_from_graph(head.graph());
        let keys: Vec<&str> = expected.iter().map(|p| p.key.as_str()).collect();
        assert!(!keys.iter().any(|k| k.starts_with("bert.pooler.")),
            "token classification must not carry pooler params");
        assert!(keys.contains(&"classifier.weight"));
        assert!(keys.contains(&"classifier.bias"));

        let by_key: std::collections::HashMap<&str, &[i64]> = expected
            .iter().map(|p| (p.key.as_str(), p.shape.as_slice())).collect();
        assert_eq!(by_key["classifier.weight"], &[9, 768]);
        assert_eq!(by_key["classifier.bias"],   &[9]);
    }

    /// Smoke: token classifier emits per-token logits of shape
    /// `[batch, seq, num_labels]`.
    #[test]
    fn token_classification_forward_shape_smoke() {
        let config = tiny_bert_config();
        let dev = Device::CPU;
        let head = BertForTokenClassification::on_device(&config, 7, dev).unwrap();
        head.graph().eval();

        let batch = 2;
        let seq = 4;
        let ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3, 4, 5, 6, 7, 0], &[batch, seq], dev).unwrap(),
            false,
        );
        let pos = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3, 0, 1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let tt = Variable::new(Tensor::from_i64(&[0; 8], &[batch, seq], dev).unwrap(), false);
        let mask_flat = Tensor::ones(&[batch, seq], TensorOptions {
            dtype: DType::Float32, device: dev,
        }).unwrap();
        let mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

        let out = head.graph().forward_multi(&[ids, pos, tt, mask]).unwrap();
        assert_eq!(out.shape(), vec![batch, seq, 7]);
    }

    /// `BertForQuestionAnswering` adds exactly `qa_outputs.{weight,bias}`
    /// of shape `[2, H]` / `[2]`.
    #[test]
    fn question_answering_parameter_keys_match_hf() {
        let config = BertConfig::bert_base_uncased();
        let head = BertForQuestionAnswering::on_device(&config, Device::CPU).unwrap();
        let expected = expected_from_graph(head.graph());
        let mut head_keys: Vec<&str> = expected
            .iter().map(|p| p.key.as_str()).filter(|k| !k.starts_with("bert.")).collect();
        head_keys.sort();
        assert_eq!(head_keys, vec!["qa_outputs.bias", "qa_outputs.weight"]);

        let by_key: std::collections::HashMap<&str, &[i64]> = expected
            .iter().map(|p| (p.key.as_str(), p.shape.as_slice())).collect();
        assert_eq!(by_key["qa_outputs.weight"], &[2, 768]);
        assert_eq!(by_key["qa_outputs.bias"],   &[2]);
    }

    /// Smoke: QA head emits `[batch, seq, 2]` (start, end logits).
    #[test]
    fn question_answering_forward_shape_smoke() {
        let config = tiny_bert_config();
        let dev = Device::CPU;
        let head = BertForQuestionAnswering::on_device(&config, dev).unwrap();
        head.graph().eval();

        let batch = 1;
        let seq = 4;
        let ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3, 4], &[batch, seq], dev).unwrap(),
            false,
        );
        let pos = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let tt = Variable::new(Tensor::from_i64(&[0; 4], &[batch, seq], dev).unwrap(), false);
        let mask_flat = Tensor::ones(&[batch, seq], TensorOptions {
            dtype: DType::Float32, device: dev,
        }).unwrap();
        let mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

        let out = head.graph().forward_multi(&[ids, pos, tt, mask]).unwrap();
        assert_eq!(out.shape(), vec![batch, seq, 2]);
    }

    /// Zero labels must error rather than produce a zero-width head
    /// that silently passes through later shape checks.
    #[test]
    fn task_heads_reject_zero_labels() {
        let config = BertConfig::bert_base_uncased();
        let dev = Device::CPU;
        assert!(BertForSequenceClassification::on_device(&config, 0, dev).is_err());
        assert!(BertForTokenClassification::on_device(&config, 0, dev).is_err());
    }
}
