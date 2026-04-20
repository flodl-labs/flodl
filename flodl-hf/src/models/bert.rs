//! BERT encoder, compatible with HuggingFace `bert-base-uncased` checkpoints.
//!
//! Structure: [`BertEmbeddings`] (token + position + token-type embeddings
//! with LayerNorm and Dropout), a stack of [`BertLayer`] instances (each one
//! a self-attention block followed by a two-layer GELU feed-forward, both
//! wrapped with residual + LayerNorm), and [`BertPooler`] on the `[CLS]`
//! position. [`BertModel::build`] assembles all of this into a flat
//! [`Graph`].
//!
//! Self-attention uses separate Q/K/V projections and libtorch's fused
//! [`scaled_dot_product_attention`] kernel. Padding is handled via an
//! additive attention mask threaded into every encoder layer as a named
//! graph input (see [`build_extended_attention_mask`]).
//!
//! Parameter names are chosen so `Graph::named_parameters()` output, once
//! passed through [`hf_key_from_flodl_key`](crate::path::hf_key_from_flodl_key),
//! matches safetensors checkpoint keys exactly. No remapping needed at load
//! time.

use std::cell::Cell;
use std::collections::HashMap;

use flodl::nn::{Dropout, Embedding, GELU, LayerNorm, Linear, Module, NamedInputModule, Parameter};
use flodl::{scaled_dot_product_attention, DType, Device, FlowBuilder, Graph, Result, Tensor, TensorError, Variable};

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
        }
    }

    /// Parse a HuggingFace-style `config.json` string into a [`BertConfig`].
    ///
    /// Reads the fields that affect model shape
    /// (`vocab_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
    /// `intermediate_size`, `max_position_embeddings`, `type_vocab_size`,
    /// `pad_token_id`, `layer_norm_eps`, `hidden_dropout_prob`,
    /// `attention_probs_dropout_prob`). Unknown fields are ignored, so adding
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

// ── BertSelfAttention ────────────────────────────────────────────────────

/// Self-attention with separate Q/K/V projections and fused scaled-dot-
/// product attention.
///
/// Uses [`flodl::scaled_dot_product_attention`] (libtorch's
/// `at::scaled_dot_product_attention`) for the Q·Kᵀ/√d → softmax → ·V
/// chain — one kernel in most cases, dispatched to flash / mem-efficient
/// / math by libtorch at runtime.
///
/// Parameter layout matches HF BERT's `attention.self.{query,key,value}`
/// exactly. The forward takes an optional additive attention mask
/// (shape `[batch, 1, 1, seq_len]`, see [`build_extended_attention_mask`])
/// which the graph plumbs in as a named input on every encoder layer.
pub struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    /// Dropout probability for attention probabilities. Threaded into SDPA
    /// via the fused kernel's `dropout_p` parameter; zero at eval time.
    /// No learnable parameters here; cell carries the training flag.
    attn_dropout_prob: f64,
    training: Cell<bool>,
    num_heads: i64,
    head_dim: i64,
}

impl BertSelfAttention {
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        assert!(
            config.hidden_size % config.num_attention_heads == 0,
            "hidden_size ({}) must be divisible by num_attention_heads ({})",
            config.hidden_size, config.num_attention_heads,
        );
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(BertSelfAttention {
            query: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            key:   Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            value: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            attn_dropout_prob: config.attention_probs_dropout_prob,
            training: Cell::new(true),
            num_heads: config.num_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, input: &Variable, attention_mask: Option<&Variable>) -> Result<Variable> {
        let shape = input.shape();
        let batch = shape[0];
        let seq = shape[1];

        let q = self.query.forward(input)?
            .reshape(&[batch, seq, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = self.key.forward(input)?
            .reshape(&[batch, seq, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = self.value.forward(input)?
            .reshape(&[batch, seq, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        let dropout_p = if self.training.get() { self.attn_dropout_prob } else { 0.0 };
        let mask_data = attention_mask.map(|m| m.data());
        let context = scaled_dot_product_attention(
            &q, &k, &v,
            mask_data.as_ref(),
            dropout_p,
            /*is_causal=*/false,
            /*scale=*/None,
        )?;

        context.transpose(1, 2)?
            .reshape(&[batch, seq, self.num_heads * self.head_dim])
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params("query", self.query.parameters()));
        out.extend(prefix_params("key",   self.key.parameters()));
        out.extend(prefix_params("value", self.value.parameters()));
        out
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }
}

// ── BertSelfOutput ───────────────────────────────────────────────────────

/// Post-attention projection + dropout + residual + LayerNorm.
/// HF parameter layout: `dense.{weight,bias}`, `LayerNorm.{weight,bias}`.
pub struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertSelfOutput {
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(BertSelfOutput {
            dense: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            layer_norm: LayerNorm::on_device_with_eps(
                config.hidden_size, config.layer_norm_eps, device,
            )?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    /// `hidden_states` passes through dense + dropout, is added to the
    /// pre-attention `residual`, and then LayerNormed.
    fn forward(&self, hidden_states: &Variable, residual: &Variable) -> Result<Variable> {
        let d = self.dense.forward(hidden_states)?;
        let dr = self.dropout.forward(&d)?;
        self.layer_norm.forward(&dr.add(residual)?)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params("dense", self.dense.parameters()));
        out.extend(prefix_params("LayerNorm", self.layer_norm.parameters()));
        out
    }

    fn set_training(&self, training: bool) {
        self.dropout.set_training(training);
    }
}

// ── BertAttention ────────────────────────────────────────────────────────

/// Self-attention block: QKV + output projection with residual and LN.
/// HF parameter layout: `self.*`, `output.*`.
pub struct BertAttention {
    self_attn: BertSelfAttention,
    output: BertSelfOutput,
}

impl BertAttention {
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(BertAttention {
            self_attn: BertSelfAttention::on_device(config, device)?,
            output:    BertSelfOutput::on_device(config, device)?,
        })
    }

    fn forward(&self, input: &Variable, attention_mask: Option<&Variable>) -> Result<Variable> {
        let self_out = self.self_attn.forward(input, attention_mask)?;
        self.output.forward(&self_out, input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params("self",   self.self_attn.parameters()));
        out.extend(prefix_params("output", self.output.parameters()));
        out
    }

    fn set_training(&self, training: bool) {
        self.self_attn.set_training(training);
        self.output.set_training(training);
    }
}

// ── BertIntermediate ─────────────────────────────────────────────────────

/// Feed-forward up-projection + GELU. `hidden → intermediate`.
pub struct BertIntermediate {
    dense: Linear,
    activation: GELU,
}

impl BertIntermediate {
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(BertIntermediate {
            dense: Linear::on_device(config.hidden_size, config.intermediate_size, device)?,
            activation: GELU::new(),
        })
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.activation.forward(&self.dense.forward(input)?)
    }

    fn parameters(&self) -> Vec<Parameter> {
        prefix_params("dense", self.dense.parameters())
    }
}

// ── BertOutput ───────────────────────────────────────────────────────────

/// Feed-forward down-projection + dropout + residual + LayerNorm.
/// `intermediate → hidden`.
pub struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertOutput {
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(BertOutput {
            dense: Linear::on_device(config.intermediate_size, config.hidden_size, device)?,
            layer_norm: LayerNorm::on_device_with_eps(
                config.hidden_size, config.layer_norm_eps, device,
            )?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    /// `hidden_states` (intermediate-sized) passes through dense + dropout,
    /// is added to the pre-FFN `residual`, and then LayerNormed.
    fn forward(&self, hidden_states: &Variable, residual: &Variable) -> Result<Variable> {
        let d = self.dense.forward(hidden_states)?;
        let dr = self.dropout.forward(&d)?;
        self.layer_norm.forward(&dr.add(residual)?)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params("dense", self.dense.parameters()));
        out.extend(prefix_params("LayerNorm", self.layer_norm.parameters()));
        out
    }

    fn set_training(&self, training: bool) {
        self.dropout.set_training(training);
    }
}

// ── BertLayer ────────────────────────────────────────────────────────────

/// One transformer encoder layer: attention → intermediate → output.
/// HF parameter layout: `attention.*`, `intermediate.*`, `output.*`.
pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(BertLayer {
            attention:    BertAttention::on_device(config, device)?,
            intermediate: BertIntermediate::on_device(config, device)?,
            output:       BertOutput::on_device(config, device)?,
        })
    }

    fn forward_impl(
        &self,
        input: &Variable,
        attention_mask: Option<&Variable>,
    ) -> Result<Variable> {
        let attn_out = self.attention.forward(input, attention_mask)?;
        let intermediate_out = self.intermediate.forward(&attn_out)?;
        self.output.forward(&intermediate_out, &attn_out)
    }
}

impl Module for BertLayer {
    fn name(&self) -> &str { "bert_layer" }

    /// Single-input forward with no attention mask. The graph drives the
    /// masked path via `forward_named`; bare calls (tests, diagnostics)
    /// run unmasked and attend to every position.
    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.forward_impl(input, None)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params("attention",    self.attention.parameters()));
        out.extend(prefix_params("intermediate", self.intermediate.parameters()));
        out.extend(prefix_params("output",       self.output.parameters()));
        out
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }

    fn set_training(&self, training: bool) {
        self.attention.set_training(training);
        self.output.set_training(training);
    }
}

impl NamedInputModule for BertLayer {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        self.forward_impl(input, refs.get("attention_mask"))
    }
}

// ── BertModel ────────────────────────────────────────────────────────────

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

    /// Build a BERT graph on `device`.
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Graph> {
        let mut fb = FlowBuilder::new()
            .input(&["position_ids", "token_type_ids", "attention_mask"])
            .through(BertEmbeddings::on_device(config, device)?)
            .tag("bert.embeddings")
            .using(&["position_ids", "token_type_ids"]);

        // Encoder stack. One BertLayer per layer, tagged with its HF path.
        // Every layer pulls the same `attention_mask` input.
        let layer_root = HfPath::new("bert").sub("encoder").sub("layer");
        for i in 0..config.num_hidden_layers {
            let tag = layer_root.sub(i).to_string();
            fb = fb
                .through(BertLayer::on_device(config, device)?)
                .tag(&tag)
                .using(&["attention_mask"]);
        }

        fb.through(BertPooler::on_device(config, device)?)
            .tag("bert.pooler")
            .build()
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

    /// Mask threading sanity: for the internal `forward_impl` path, a
    /// zero-additive mask (all positions attend) must produce bitwise-
    /// identical output to the no-mask path. If the plumbing were wrong
    /// (e.g. shape broadcast mismatch, stale ref lookup), the two would
    /// diverge even though the mask is semantically a no-op.
    #[test]
    fn bert_layer_zero_additive_mask_matches_unmasked() {
        let config = tiny_bert_config();
        let dev = Device::CPU;
        let layer = BertLayer::on_device(&config, dev).unwrap();
        layer.set_training(false);

        let batch = 1;
        let seq = 3;
        let hidden = config.hidden_size;
        let x_data: Vec<f32> = (0..(batch * seq * hidden) as usize)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let x = Variable::new(
            Tensor::from_f32(&x_data, &[batch, seq, hidden], dev).unwrap(),
            false,
        );

        // `[B, 1, 1, S]` zeros → semantic no-op for SDPA.
        let zero_mask = Variable::new(
            Tensor::zeros(&[batch, 1, 1, seq], TensorOptions { dtype: DType::Float32, device: dev }).unwrap(),
            false,
        );

        let unmasked = layer.forward_impl(&x, None).unwrap();
        let with_zero = layer.forward_impl(&x, Some(&zero_mask)).unwrap();

        let a: Vec<f32> = unmasked.data().to_f32_vec().unwrap();
        let b: Vec<f32> = with_zero.data().to_f32_vec().unwrap();
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "diverged at {i}: unmasked={x}, zero-masked={y}",
            );
        }
    }

    /// Masking a position must change the encoder output at the remaining
    /// positions (because SDPA no longer mixes information from the masked
    /// key). A trivially-true check (some difference anywhere) is enough —
    /// if the mask were silently dropped, outputs would match bitwise.
    #[test]
    fn bert_layer_padding_mask_changes_output() {
        let config = tiny_bert_config();
        let dev = Device::CPU;
        let layer = BertLayer::on_device(&config, dev).unwrap();
        layer.set_training(false);

        let batch = 1;
        let seq = 4;
        let hidden = config.hidden_size;
        let x_data: Vec<f32> = (0..(batch * seq * hidden) as usize)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect();
        let x = Variable::new(
            Tensor::from_f32(&x_data, &[batch, seq, hidden], dev).unwrap(),
            false,
        );

        // Attend to positions 0..2, mask position 3.
        let raw = Tensor::from_f32(&[1.0, 1.0, 1.0, 0.0], &[batch, seq], dev).unwrap();
        let additive = build_extended_attention_mask(&raw).unwrap();
        let mask = Variable::new(additive, false);

        let unmasked = layer.forward_impl(&x, None).unwrap();
        let masked = layer.forward_impl(&x, Some(&mask)).unwrap();

        let a: Vec<f32> = unmasked.data().to_f32_vec().unwrap();
        let b: Vec<f32> = masked.data().to_f32_vec().unwrap();
        let max_diff = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "masking a position must change attention output; max_diff={max_diff}",
        );
    }
}
