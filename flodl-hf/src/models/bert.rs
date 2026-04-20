//! BERT encoder, compatible with HuggingFace `bert-base-uncased` checkpoints.
//!
//! Structure: [`BertEmbeddings`] (token + position + token-type embeddings
//! with LayerNorm and Dropout), a stack of [`BertLayer`] instances (each one
//! a self-attention block followed by a two-layer GELU feed-forward, both
//! wrapped with residual + LayerNorm), and [`BertPooler`] on the `[CLS]`
//! position. [`BertModel::build`] assembles all of this into a flat
//! [`Graph`].
//!
//! Self-attention is naive Q·Kᵀ / √d → softmax → ·V with separate Q/K/V
//! projections (matching HF's default `attn_implementation`). A fused SDPA
//! path is a future optimisation; kernel count today already matches the
//! PyTorch HF reference.
//!
//! Parameter names are chosen so `Graph::named_parameters()` output, once
//! passed through [`hf_key_from_flodl_key`](crate::path::hf_key_from_flodl_key),
//! matches safetensors checkpoint keys exactly. No remapping needed at load
//! time.

use std::collections::HashMap;

use flodl::nn::{Dropout, Embedding, GELU, LayerNorm, Linear, Module, NamedInputModule, Parameter};
use flodl::{Device, FlowBuilder, Graph, Result, Variable};

use crate::path::{prefix_params, HfPath};

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

/// Naive scaled-dot-product self-attention with separate Q/K/V projections.
///
/// Kernel count matches PyTorch HF default (no fused SDPA). Parameter
/// layout matches HF BERT's `attention.self.{query,key,value}` exactly.
/// The forward is unmasked; attention masks will be threaded in alongside
/// real checkpoint loading.
pub struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    /// Dropout applied to the attention probability matrix after softmax,
    /// per HF. No learnable parameters.
    probs_dropout: Dropout,
    num_heads: i64,
    head_dim: i64,
    scale: f64,
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
            probs_dropout: Dropout::new(config.attention_probs_dropout_prob),
            num_heads: config.num_attention_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
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

        let k_t = k.transpose(2, 3)?;
        let scores = q.matmul(&k_t)?.mul_scalar(self.scale)?;
        let probs = self.probs_dropout.forward(&scores.softmax(-1)?)?;
        let context = probs.matmul(&v)?
            .transpose(1, 2)?
            .reshape(&[batch, seq, self.num_heads * self.head_dim])?;
        Ok(context)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params("query", self.query.parameters()));
        out.extend(prefix_params("key",   self.key.parameters()));
        out.extend(prefix_params("value", self.value.parameters()));
        out
    }

    fn set_training(&self, training: bool) {
        self.probs_dropout.set_training(training);
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

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let self_out = self.self_attn.forward(input)?;
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
}

impl Module for BertLayer {
    fn name(&self) -> &str { "bert_layer" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let attn_out = self.attention.forward(input)?;
        let intermediate_out = self.intermediate.forward(&attn_out)?;
        self.output.forward(&intermediate_out, &attn_out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params("attention",    self.attention.parameters()));
        out.extend(prefix_params("intermediate", self.intermediate.parameters()));
        out.extend(prefix_params("output",       self.output.parameters()));
        out
    }

    fn set_training(&self, training: bool) {
        self.attention.set_training(training);
        self.output.set_training(training);
    }
}

// ── BertModel ────────────────────────────────────────────────────────────

/// Assembled BERT graph.
///
/// The returned [`Graph`] accepts three inputs via `forward_multi`, in
/// declaration order:
///
/// 1. `input_ids` (i64, shape `[batch, seq_len]`)
/// 2. `position_ids` (i64, shape `[batch, seq_len]`)
/// 3. `token_type_ids` (i64, shape `[batch, seq_len]`)
///
/// Graph layout: `bert.embeddings` → `bert.encoder.layer.{0..N-1}` →
/// `bert.pooler`, where `N = config.num_hidden_layers`.
pub struct BertModel;

impl BertModel {
    /// Build a BERT graph on CPU.
    pub fn build(config: &BertConfig) -> Result<Graph> {
        Self::on_device(config, Device::CPU)
    }

    /// Build a BERT graph on `device`.
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Graph> {
        let mut fb = FlowBuilder::new()
            .input(&["position_ids", "token_type_ids"])
            .through(BertEmbeddings::on_device(config, device)?)
            .tag("bert.embeddings")
            .using(&["position_ids", "token_type_ids"]);

        // Encoder stack. One BertLayer per layer, tagged with its HF path.
        let layer_root = HfPath::new("bert").sub("encoder").sub("layer");
        for i in 0..config.num_hidden_layers {
            let tag = layer_root.sub(i).to_string();
            fb = fb
                .through(BertLayer::on_device(config, device)?)
                .tag(&tag);
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

    /// Smoke test: construct a tiny BERT on CPU, run forward_multi with
    /// made-up ids, and verify the output shape. Catches obvious wiring
    /// breakage (residual mismatch, missing named input, transpose axis
    /// bug) without requiring real tokenized inputs.
    #[test]
    fn bert_forward_shape_smoke() {
        use flodl::Tensor;

        let config = BertConfig {
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
        };
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

        let out = graph
            .forward_multi(&[word_ids, position_ids, token_type_ids])
            .unwrap();
        // Pooler reduces [batch, seq, hidden] → [batch, hidden]
        assert_eq!(out.shape(), vec![batch, config.hidden_size]);
    }
}
