//! RoBERTa encoder, compatible with HuggingFace `roberta-base` checkpoints.
//!
//! Same encoder shape as [BERT](crate::models::bert) — a stack of
//! pre-LN-free self-attention + feed-forward blocks — with four
//! load-bearing deltas from the BERT port:
//!
//! 1. **Position ids are computed internally** from `input_ids` using
//!    the RoBERTa convention: `padding_idx + cumsum(non-pad-mask) *
//!    non-pad-mask`. Real (non-pad) tokens start at `padding_idx + 1`,
//!    padding sits at `padding_idx`. Matches HF Python's
//!    `RobertaModel.forward`. Callers pass `[input_ids, token_type_ids,
//!    attention_mask]` into the graph's `forward_multi` — no position
//!    ids in the signature.
//! 2. **LayerNorm ε defaults to 1e-5** (BERT's is 1e-12).
//! 3. **`type_vocab_size = 1`** — `token_type_ids` is always zero, but
//!    the embedding table exists and is loaded from the checkpoint for
//!    fidelity with HF.
//! 4. **`RobertaForSequenceClassification` uses a custom 2-layer head**
//!    on the `<s>` hidden state (keys
//!    `classifier.dense.*` + `classifier.out_proj.*`), not the
//!    BERT-style `pooler → Dropout → Linear`.
//!
//! The encoder layer itself is shared with the BERT and DistilBERT
//! ports via
//! [`LayerNaming::BERT`](crate::models::transformer_layer::LayerNaming::BERT)
//! — BERT and RoBERTa use the same `attention.self.{query,key,value}`
//! / `attention.output.dense` / `intermediate.dense` / `output.dense`
//! layout, so one [`TransformerLayer`](crate::models::transformer_layer::TransformerLayer)
//! serves both families. Task heads remain here because RoBERTa's
//! sequence-classification head diverges from BERT's.

use std::collections::HashMap;

use flodl::nn::{Dropout, Embedding, LayerNorm, Linear, Module, NamedInputModule, Parameter};
use flodl::{
    DType, Device, FlowBuilder, Graph, Result, Tensor, TensorError, Variable,
};

use crate::models::bert::build_extended_attention_mask;
use crate::models::transformer_layer::{LayerNaming, TransformerLayer, TransformerLayerConfig};
use crate::path::{prefix_params, HfPath};

/// RoBERTa hyperparameters. Matches the fields of a HuggingFace
/// `RobertaConfig` JSON file that affect model shape.
///
/// Use [`RobertaConfig::roberta_base`] for the standard 12-layer /
/// 768-dim preset.
#[derive(Debug, Clone)]
pub struct RobertaConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub type_vocab_size: i64,
    /// Padding token index. RoBERTa uses `1` (for `<pad>`). Drives both
    /// the word-embedding padding row AND the position-id computation —
    /// real tokens start at `padding_idx + 1` in the position embedding
    /// lookup.
    pub pad_token_id: i64,
    pub layer_norm_eps: f64,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    /// See [`crate::models::bert::BertConfig::num_labels`].
    pub num_labels: Option<i64>,
    /// See [`crate::models::bert::BertConfig::id2label`].
    pub id2label: Option<Vec<String>>,
}

impl RobertaConfig {
    /// Preset matching `roberta-base` on the HuggingFace Hub.
    pub fn roberta_base() -> Self {
        RobertaConfig {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            // RoBERTa reserves position rows `[0 .. padding_idx]` so the
            // effective max sequence length is `514 - 2 = 512` real tokens.
            max_position_embeddings: 514,
            type_vocab_size: 1,
            pad_token_id: 1,
            layer_norm_eps: 1e-5,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            num_labels: None,
            id2label: None,
        }
    }

    /// Parse a HuggingFace-style `config.json` string into a [`RobertaConfig`].
    ///
    /// Reads the fields that affect model shape plus the task-head
    /// metadata (`num_labels`, `id2label`) used by
    /// [`RobertaForSequenceClassification`] /
    /// [`RobertaForTokenClassification`] /
    /// [`RobertaForQuestionAnswering`]. Unknown fields are ignored.
    ///
    /// Defaults mirror HF's `RobertaConfig`:
    /// `layer_norm_eps = 1e-5`, `type_vocab_size = 1`, `pad_token_id = 1`,
    /// dropout probabilities `0.1`.
    pub fn from_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| TensorError::new(&format!("config.json parse error: {e}")))?;
        let get_i64 = |key: &str| -> Result<i64> {
            v.get(key)
                .and_then(|x| x.as_i64())
                .ok_or_else(|| {
                    TensorError::new(&format!(
                        "config.json missing required integer field: {key}",
                    ))
                })
        };
        let get_i64_or = |key: &str, default: i64| -> i64 {
            v.get(key).and_then(|x| x.as_i64()).unwrap_or(default)
        };
        let get_f64_or = |key: &str, default: f64| -> f64 {
            v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
        };

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
        let num_labels = v
            .get("num_labels")
            .and_then(|x| x.as_i64())
            .or_else(|| id2label.as_ref().map(|v| v.len() as i64));

        Ok(RobertaConfig {
            vocab_size:              get_i64("vocab_size")?,
            hidden_size:             get_i64("hidden_size")?,
            num_hidden_layers:       get_i64("num_hidden_layers")?,
            num_attention_heads:     get_i64("num_attention_heads")?,
            intermediate_size:       get_i64("intermediate_size")?,
            max_position_embeddings: get_i64("max_position_embeddings")?,
            type_vocab_size:         get_i64_or("type_vocab_size", 1),
            pad_token_id:            get_i64_or("pad_token_id", 1),
            layer_norm_eps:               get_f64_or("layer_norm_eps", 1e-5),
            hidden_dropout_prob:          get_f64_or("hidden_dropout_prob", 0.1),
            attention_probs_dropout_prob: get_f64_or("attention_probs_dropout_prob", 0.1),
            num_labels,
            id2label,
        })
    }
}

// ── RobertaEmbeddings ────────────────────────────────────────────────────

/// Token + position + token-type embeddings with post-LN and Dropout.
///
/// Differs from [`BertEmbeddings`](crate::models::bert::BertEmbeddings)
/// in one crucial way: `position_ids` are computed internally from
/// `input_ids` rather than read from a graph input. The formula
/// matches HF Python's
/// `create_position_ids_from_input_ids`:
///
/// ```text
/// mask       = (input_ids != padding_idx).long()   // [B, S]
/// positions  = cumsum(mask, dim=1) * mask + padding_idx
/// ```
///
/// Real tokens land at indices `padding_idx + 1, padding_idx + 2, …`.
/// Padding slots stay at `padding_idx`. This mirrors how HF fine-tuned
/// every public RoBERTa checkpoint, so safetensors weights line up
/// directly.
pub struct RobertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    padding_idx: i64,
}

impl RobertaEmbeddings {
    pub fn on_device(config: &RobertaConfig, device: Device) -> Result<Self> {
        Ok(RobertaEmbeddings {
            word_embeddings: Embedding::on_device_with_padding_idx(
                config.vocab_size,
                config.hidden_size,
                Some(config.pad_token_id),
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
            padding_idx: config.pad_token_id,
        })
    }

    /// Compute RoBERTa-style position ids from `input_ids`.
    ///
    /// Runs entirely on raw `Tensor`s (no autograd) — position ids are
    /// integer indices into the embedding table, so they don't need to
    /// participate in backward.
    fn position_ids_from_input_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Float mask in {0.0, 1.0}; cumsum along seq dim stays integer-valued
        // in f32 for all realistic sequence lengths (f32 holds exact integers
        // up to 2^24, safely above any tokenizer truncation).
        let mask = input_ids.ne_scalar(self.padding_idx as f64)?;
        let cum = mask.cumsum(1)?;
        cum.mul(&mask)?
            .add_scalar(self.padding_idx as f64)?
            .to_dtype(DType::Int64)
    }
}

impl Module for RobertaEmbeddings {
    fn name(&self) -> &str { "roberta_embeddings" }

    /// Single-input forward: word ids only. Position ids are computed
    /// internally; token-type embedding adds a zero-row contribution
    /// (`type_vocab_size = 1` means all tokens map to row 0).
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let pos_ids = self.position_ids_from_input_ids(&input.data())?;
        let pos_var = Variable::new(pos_ids, false);
        let word = self.word_embeddings.forward(input)?;
        let pe = self.position_embeddings.forward(&pos_var)?;
        let summed = word.add(&pe)?;
        let ln = self.layer_norm.forward(&summed)?;
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

impl NamedInputModule for RobertaEmbeddings {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let pos_ids = self.position_ids_from_input_ids(&input.data())?;
        let pos_var = Variable::new(pos_ids, false);

        let word = self.word_embeddings.forward(input)?;
        let pe = self.position_embeddings.forward(&pos_var)?;
        let mut summed = word.add(&pe)?;
        if let Some(tt) = refs.get("token_type_ids") {
            let te = self.token_type_embeddings.forward(tt)?;
            summed = summed.add(&te)?;
        }
        let ln = self.layer_norm.forward(&summed)?;
        self.dropout.forward(&ln)
    }
}

// ── RobertaPooler ────────────────────────────────────────────────────────

/// Pooler: take the `<s>` (index 0) hidden state, project, then tanh.
///
/// Structurally identical to `BertPooler`. Most RoBERTa fine-tunes
/// don't use the pooler (the sequence-classification head does its own
/// two-layer projection on the `<s>` state), but the pooler weights
/// are still published with the base `roberta-base` checkpoint, so the
/// backbone must be able to emit a pooled output when asked.
pub struct RobertaPooler {
    dense: Linear,
}

impl RobertaPooler {
    pub fn on_device(config: &RobertaConfig, device: Device) -> Result<Self> {
        Ok(RobertaPooler {
            dense: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
        })
    }
}

impl Module for RobertaPooler {
    fn name(&self) -> &str { "roberta_pooler" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let cls = input.select(1, 0)?;
        let pooled = self.dense.forward(&cls)?;
        pooled.tanh()
    }

    fn parameters(&self) -> Vec<Parameter> {
        prefix_params("dense", self.dense.parameters())
    }
}

// ── RobertaModel ─────────────────────────────────────────────────────────

/// Translate a [`RobertaConfig`] into the subset [`TransformerLayer`]
/// consumes. BERT and RoBERTa share the encoder layer layout exactly;
/// only the embedding and task-head code differs.
fn roberta_layer_config(config: &RobertaConfig) -> TransformerLayerConfig {
    TransformerLayerConfig {
        hidden_size:                  config.hidden_size,
        num_attention_heads:          config.num_attention_heads,
        intermediate_size:            config.intermediate_size,
        hidden_dropout_prob:          config.hidden_dropout_prob,
        attention_probs_dropout_prob: config.attention_probs_dropout_prob,
        layer_norm_eps:               config.layer_norm_eps,
    }
}

/// Assemble the RoBERTa backbone onto a fresh [`FlowBuilder`], up to
/// and optionally including the pooler.
///
/// Graph signature is **3 named inputs** (no `position_ids`):
/// `input_ids` (implicit first), `token_type_ids`, `attention_mask`.
/// The embedding layer computes position ids internally from
/// `input_ids`.
///
/// Graph shape: `roberta.embeddings` →
/// `roberta.encoder.layer.{0..N-1}` → (`roberta.pooler`?).
fn roberta_backbone_flow(
    config: &RobertaConfig,
    device: Device,
    with_pooler: bool,
) -> Result<FlowBuilder> {
    let mut fb = FlowBuilder::new()
        .input(&["token_type_ids", "attention_mask"])
        .through(RobertaEmbeddings::on_device(config, device)?)
        .tag("roberta.embeddings")
        .using(&["token_type_ids"]);

    let layer_root = HfPath::new("roberta").sub("encoder").sub("layer");
    let layer_cfg = roberta_layer_config(config);
    for i in 0..config.num_hidden_layers {
        let tag = layer_root.sub(i).to_string();
        fb = fb
            .through(TransformerLayer::on_device(&layer_cfg, LayerNaming::BERT, device)?)
            .tag(&tag)
            .using(&["attention_mask"]);
    }
    if with_pooler {
        fb = fb
            .through(RobertaPooler::on_device(config, device)?)
            .tag("roberta.pooler");
    }
    Ok(fb)
}

/// Assembled RoBERTa graph.
///
/// The returned [`Graph`] accepts **three** inputs via `forward_multi`,
/// in declaration order:
///
/// 1. `input_ids` (i64, shape `[batch, seq_len]`)
/// 2. `token_type_ids` (i64, shape `[batch, seq_len]`, all zeros for
///    most RoBERTa checkpoints)
/// 3. `attention_mask` (f32, shape `[batch, 1, 1, seq_len]`, additive
///    — build with
///    [`build_extended_attention_mask`](crate::models::bert::build_extended_attention_mask)
///    from a plain `[batch, seq_len]` 0/1 mask)
///
/// Position ids are computed internally from `input_ids`.
///
/// The default Hub loader ([`RobertaModel::from_pretrained`]) builds
/// the graph *without* a pooler since `roberta-base` and most
/// fine-tunes don't ship pooler weights (RoBERTa pretraining drops
/// NSP). Use [`RobertaModel::on_device`] explicitly when a specific
/// checkpoint is known to carry its own pooler.
pub struct RobertaModel;

impl RobertaModel {
    /// Build a RoBERTa graph on CPU with a pooler node.
    ///
    /// Prefer [`RobertaModel::on_device_without_pooler`] for the
    /// common case: `roberta-base` and its fine-tunes skip the
    /// pooler, and the default [`from_pretrained`](Self::from_pretrained)
    /// path matches that convention.
    pub fn build(config: &RobertaConfig) -> Result<Graph> {
        Self::on_device(config, Device::CPU)
    }

    /// Build a RoBERTa graph on `device` with a pooler node. Emits
    /// `pooler_output` (`[batch, hidden]`).
    pub fn on_device(config: &RobertaConfig, device: Device) -> Result<Graph> {
        roberta_backbone_flow(config, device, true)?.build()
    }

    /// Build a RoBERTa graph on `device` *without* the pooler. Emits
    /// `last_hidden_state` (`[batch, seq_len, hidden]`) — the shape
    /// every task head in this module consumes, and the shape
    /// [`from_pretrained`](Self::from_pretrained) loads into.
    pub fn on_device_without_pooler(config: &RobertaConfig, device: Device) -> Result<Graph> {
        roberta_backbone_flow(config, device, false)?.build()
    }
}

// ── Task heads ───────────────────────────────────────────────────────────

fn default_labels(n: i64) -> Vec<String> {
    (0..n).map(|i| format!("LABEL_{i}")).collect()
}

fn check_num_labels(n: i64) -> Result<i64> {
    if n <= 0 {
        return Err(TensorError::new(&format!(
            "num_labels must be > 0, got {n}",
        )));
    }
    Ok(n)
}

// ── RobertaClassificationHead ────────────────────────────────────────────

/// The two-layer classification head HF Python calls
/// `RobertaClassificationHead`:
/// `Dropout → dense → tanh → Dropout → out_proj`, all applied to the
/// `<s>` hidden state (index 0 along the sequence axis).
///
/// Parameter keys (nested under the outer `classifier` tag):
/// - `classifier.dense.weight`    (`[hidden, hidden]`)
/// - `classifier.dense.bias`      (`[hidden]`)
/// - `classifier.out_proj.weight` (`[num_labels, hidden]`)
/// - `classifier.out_proj.bias`   (`[num_labels]`)
///
/// This replaces BERT's `pooler → Dropout → Linear(hidden, num_labels)`
/// pattern. RoBERTa fine-tunes typically ship with the pooler weights
/// too, but the SeqCls head ignores them — so when the backbone loads
/// here it runs *without* the pooler node.
pub struct RobertaClassificationHead {
    dropout: Dropout,
    dense: Linear,
    out_proj: Linear,
}

impl RobertaClassificationHead {
    pub fn on_device(
        config: &RobertaConfig,
        num_labels: i64,
        device: Device,
    ) -> Result<Self> {
        Ok(RobertaClassificationHead {
            dropout: Dropout::new(config.hidden_dropout_prob),
            dense: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            out_proj: Linear::on_device(config.hidden_size, num_labels, device)?,
        })
    }
}

impl Module for RobertaClassificationHead {
    fn name(&self) -> &str { "roberta_classification_head" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // input: [batch, seq_len, hidden] → take index 0 along seq axis.
        let cls = input.select(1, 0)?;          // [batch, hidden]
        let x = self.dropout.forward(&cls)?;
        let x = self.dense.forward(&x)?;
        let x = x.tanh()?;
        let x = self.dropout.forward(&x)?;
        self.out_proj.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params("dense",    self.dense.parameters()));
        out.extend(prefix_params("out_proj", self.out_proj.parameters()));
        out
    }

    fn set_training(&self, training: bool) {
        self.dropout.set_training(training);
    }
}

// ── RobertaForSequenceClassification ─────────────────────────────────────

/// RoBERTa with a sequence-classification head on the `<s>` hidden
/// state: a two-layer MLP with tanh activation and dropout on either
/// side (see [`RobertaClassificationHead`]).
///
/// Matches HF Python's `RobertaForSequenceClassification`.
/// Pre-trained checkpoints:
/// `cardiffnlp/twitter-roberta-base-sentiment-latest` (3-label),
/// `roberta-large-mnli`, `SamLowe/roberta-base-go_emotions`.
pub struct RobertaForSequenceClassification {
    graph: Graph,
    id2label: Vec<String>,
    #[cfg(feature = "tokenizer")]
    tokenizer: Option<crate::tokenizer::HfTokenizer>,
}

impl RobertaForSequenceClassification {
    /// Build the full graph (backbone without pooler + two-layer
    /// classification head) on `device` without loading any weights.
    pub fn on_device(
        config: &RobertaConfig,
        num_labels: i64,
        device: Device,
    ) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = roberta_backbone_flow(config, device, /*with_pooler=*/ false)?
            .through(RobertaClassificationHead::on_device(config, num_labels, device)?)
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

    pub(crate) fn num_labels_from_config(config: &RobertaConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "RobertaForSequenceClassification: config.json has no `num_labels` \
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

    /// One-shot text → label distribution.
    #[cfg(feature = "tokenizer")]
    pub fn predict(&self, texts: &[&str]) -> Result<Vec<Vec<(String, f32)>>> {
        let tok = self.tokenizer.as_ref().ok_or_else(|| {
            TensorError::new(
                "RobertaForSequenceClassification::predict requires a tokenizer; \
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
            enc.token_type_ids.clone(),
            mask,
        ])
    }
}

#[cfg(feature = "tokenizer")]
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

// ── RobertaForTokenClassification ────────────────────────────────────────

/// One labelled token inside a [`RobertaForTokenClassification`] prediction.
#[cfg(feature = "tokenizer")]
#[derive(Debug, Clone)]
pub struct TokenPrediction {
    pub token: String,
    pub label: String,
    pub score: f32,
    pub attends: bool,
}

/// RoBERTa with a per-token classification head: `last_hidden_state →
/// Dropout → Linear(hidden, num_labels)`. Structurally identical to
/// [`BertForTokenClassification`](crate::models::bert::BertForTokenClassification).
///
/// Matches HF Python's `RobertaForTokenClassification`. Pre-trained
/// checkpoints: `Jean-Baptiste/roberta-large-ner-english`,
/// `obi/deid_roberta_i2b2`.
pub struct RobertaForTokenClassification {
    graph: Graph,
    id2label: Vec<String>,
    #[cfg(feature = "tokenizer")]
    tokenizer: Option<crate::tokenizer::HfTokenizer>,
}

impl RobertaForTokenClassification {
    pub fn on_device(
        config: &RobertaConfig,
        num_labels: i64,
        device: Device,
    ) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = roberta_backbone_flow(config, device, /*with_pooler=*/ false)?
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

    pub(crate) fn num_labels_from_config(config: &RobertaConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "RobertaForTokenClassification: config.json has no `num_labels` \
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

    #[cfg(feature = "tokenizer")]
    pub fn tag(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
    ) -> Result<Vec<Vec<TokenPrediction>>> {
        let tok = self.tokenizer.as_ref().ok_or_else(|| {
            TensorError::new(
                "RobertaForTokenClassification::tag requires a tokenizer; \
                 attach one via .with_tokenizer(...) or from_pretrained",
            )
        })?;
        self.graph.eval();
        let mask_f32 = enc.attention_mask.data().to_dtype(DType::Float32)?;
        let mask = Variable::new(build_extended_attention_mask(&mask_f32)?, false);
        let logits = self.graph.forward_multi(&[
            enc.input_ids.clone(),
            enc.token_type_ids.clone(),
            mask,
        ])?;
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

    #[cfg(feature = "tokenizer")]
    pub fn predict(&self, texts: &[&str]) -> Result<Vec<Vec<TokenPrediction>>> {
        let tok = self.tokenizer.as_ref().ok_or_else(|| {
            TensorError::new(
                "RobertaForTokenClassification::predict requires a tokenizer; \
                 use from_pretrained or .with_tokenizer(...) first",
            )
        })?;
        let enc = tok.encode(texts)?;
        self.tag(&enc)
    }
}

// ── RobertaForQuestionAnswering ──────────────────────────────────────────

/// Extracted answer span from a [`RobertaForQuestionAnswering`] prediction.
#[cfg(feature = "tokenizer")]
#[derive(Debug, Clone)]
pub struct Answer {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub score: f32,
}

/// RoBERTa with an extractive question-answering head:
/// `last_hidden_state → Linear(hidden, 2)` splitting into
/// `start_logits` and `end_logits`.
///
/// Parameter keys for the head:
/// - `qa_outputs.weight` (`[2, hidden]`)
/// - `qa_outputs.bias`   (`[2]`)
///
/// Matches HF Python's `RobertaForQuestionAnswering`. Pre-trained
/// checkpoints: `deepset/roberta-base-squad2`,
/// `csarron/roberta-base-squad-v1`.
pub struct RobertaForQuestionAnswering {
    graph: Graph,
    #[cfg(feature = "tokenizer")]
    tokenizer: Option<crate::tokenizer::HfTokenizer>,
}

impl RobertaForQuestionAnswering {
    pub fn on_device(config: &RobertaConfig, device: Device) -> Result<Self> {
        let graph = roberta_backbone_flow(config, device, /*with_pooler=*/ false)?
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

    #[cfg(feature = "tokenizer")]
    pub fn answer(&self, question: &str, context: &str) -> Result<Answer> {
        let mut out = self.answer_batch(&[(question, context)])?;
        Ok(out.pop().expect("answer_batch returns one per input"))
    }

    #[cfg(feature = "tokenizer")]
    pub fn answer_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<Answer>> {
        let tok = self.tokenizer.as_ref().ok_or_else(|| {
            TensorError::new(
                "RobertaForQuestionAnswering::answer requires a tokenizer; \
                 use from_pretrained or .with_tokenizer(...) first",
            )
        })?;
        let enc = tok.encode_pairs(pairs)?;
        self.extract(&enc)
    }

    /// Run the graph on a pre-tokenised `(question, context)` batch
    /// and extract best spans. The tokenizer's `sequence_ids` mark the
    /// context region (`== 1`); scoring is restricted to that region.
    ///
    /// RoBERTa's `token_type_ids` are all zero so the BERT-style
    /// `tt == 1` filter doesn't work; `sequence_ids` is the
    /// model-agnostic signal HF uses in its QA pipeline.
    #[cfg(feature = "tokenizer")]
    pub fn extract(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
    ) -> Result<Vec<Answer>> {
        let tok = self.tokenizer.as_ref().ok_or_else(|| {
            TensorError::new(
                "RobertaForQuestionAnswering::extract requires a tokenizer; \
                 attach one via .with_tokenizer(...) or from_pretrained",
            )
        })?;
        self.graph.eval();
        let mask_f32 = enc.attention_mask.data().to_dtype(DType::Float32)?;
        let mask = Variable::new(build_extended_attention_mask(&mask_f32)?, false);
        let logits = self.graph.forward_multi(&[
            enc.input_ids.clone(),
            enc.token_type_ids.clone(),
            mask,
        ])?;
        let shape = logits.shape();
        assert_eq!(shape.len(), 3, "expected [B, S, 2], got {shape:?}");
        let batch = shape[0] as usize;
        let seq = shape[1] as usize;
        assert_eq!(shape[2], 2, "QA head must be 2-wide, got {}", shape[2]);

        let starts = logits.narrow(-1, 0, 1)?.softmax(1)?;
        let ends = logits.narrow(-1, 1, 1)?.softmax(1)?;
        let starts_flat = starts.data().to_f32_vec()?;
        let ends_flat = ends.data().to_f32_vec()?;
        let sequence_ids: Vec<i64> = enc.sequence_ids.data().to_i64_vec()?;
        let input_ids: Vec<i64> = enc.input_ids.data().to_i64_vec()?;

        let mut answers = Vec::with_capacity(batch);
        for b in 0..batch {
            let offset = b * seq;
            let valid: Vec<usize> = (0..seq)
                .filter(|&s| sequence_ids[offset + s] == 1)
                .collect();
            if valid.is_empty() {
                return Err(TensorError::new(
                    "QA extract: no context tokens (sequence_id == 1) found; \
                     tokenizer did not produce a pair encoding",
                ));
            }
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

    /// The 16 parameter keys every encoder layer exposes.
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
        suffixes.iter().map(|s| format!("roberta.encoder.layer.{i}.{s}")).collect()
    }

    /// Full key set (199 keys: 5 embeddings + 16×12 layers + 2 pooler)
    /// matches HF safetensors dotted form exactly.
    #[test]
    fn roberta_parameter_keys_match_hf_dotted_form() {
        let config = RobertaConfig::roberta_base();
        let graph = RobertaModel::build(&config).unwrap();
        let expected = expected_from_graph(&graph);

        let mut keys: Vec<String> = expected.iter().map(|p| p.key.clone()).collect();
        keys.sort();

        let mut want: Vec<String> = vec![
            "roberta.embeddings.LayerNorm.bias".into(),
            "roberta.embeddings.LayerNorm.weight".into(),
            "roberta.embeddings.position_embeddings.weight".into(),
            "roberta.embeddings.token_type_embeddings.weight".into(),
            "roberta.embeddings.word_embeddings.weight".into(),
        ];
        for i in 0..config.num_hidden_layers {
            want.extend(expected_layer_keys(i));
        }
        want.extend([
            "roberta.pooler.dense.bias".into(),
            "roberta.pooler.dense.weight".into(),
        ]);
        want.sort();

        assert_eq!(keys, want);
    }

    /// SeqCls head keys: the two-layer classifier ships
    /// `classifier.dense.*` + `classifier.out_proj.*`, NOT a single
    /// `classifier.{weight,bias}` like BERT.
    #[test]
    fn roberta_seqcls_head_has_two_layer_keys() {
        let config = RobertaConfig::roberta_base();
        let head = RobertaForSequenceClassification::on_device(
            &config, 3, Device::CPU,
        ).unwrap();
        let expected = expected_from_graph(head.graph());
        let keys: Vec<String> = expected.iter().map(|p| p.key.clone()).collect();
        assert!(keys.contains(&"classifier.dense.weight".to_string()));
        assert!(keys.contains(&"classifier.dense.bias".to_string()));
        assert!(keys.contains(&"classifier.out_proj.weight".to_string()));
        assert!(keys.contains(&"classifier.out_proj.bias".to_string()));
        // No single-linear `classifier.weight` at the top level.
        assert!(!keys.iter().any(|k| k == "classifier.weight"));
    }

    /// Position-id computation: real tokens start at `padding_idx + 1`
    /// (== 2 for `roberta-base`), padding slots stay at `padding_idx`.
    #[test]
    fn roberta_position_ids_follow_hf_convention() {
        let config = RobertaConfig::roberta_base();
        let emb = RobertaEmbeddings::on_device(&config, Device::CPU).unwrap();
        // [<s>=0, real, real, </s>=2, <pad>=1, <pad>=1]  —  using
        // arbitrary non-pad ids for the real tokens.
        let ids = Tensor::from_i64(&[0, 100, 200, 2, 1, 1], &[1, 6], Device::CPU).unwrap();
        let pos = emb.position_ids_from_input_ids(&ids).unwrap();
        let flat: Vec<i64> = pos.to_i64_vec().unwrap();
        // Expected:
        //   mask        = [1, 1, 1, 1, 0, 0]
        //   cumsum*mask = [1, 2, 3, 4, 0, 0]
        //   + pad_idx=1 = [2, 3, 4, 5, 1, 1]
        assert_eq!(flat, vec![2, 3, 4, 5, 1, 1]);
    }
}
