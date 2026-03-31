//! Transformer encoder benchmark: Embedding + N x (MHA + FFN + LayerNorm) + projection.
//!
//! Tests attention throughput — the dominant architecture in modern DL.
//! Uses cross-entropy loss on token predictions.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

const VOCAB: i64 = 8192;
const D_MODEL: i64 = 512;
const D_FF: i64 = 2048;
const HEADS: i64 = 8;
const LAYERS: usize = 4;
const SEQ_LEN: i64 = 128;

/// Single transformer encoder layer: MHA + residual + LayerNorm + FFN + residual + LayerNorm.
struct EncoderLayer {
    mha: MultiheadAttention,
    norm1: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    norm2: LayerNorm,
}

impl EncoderLayer {
    fn new(device: Device) -> Result<Self> {
        Ok(Self {
            mha: MultiheadAttention::on_device(D_MODEL, HEADS, device)?,
            norm1: LayerNorm::on_device(D_MODEL, device)?,
            ff1: Linear::on_device(D_MODEL, D_FF, device)?,
            ff2: Linear::on_device(D_FF, D_MODEL, device)?,
            norm2: LayerNorm::on_device(D_MODEL, device)?,
        })
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Self-attention + residual + norm (post-LN)
        let attn = self.mha.forward(input)?;
        let x = input.add(&attn)?;
        let x = self.norm1.forward(&x)?;

        // FFN + residual + norm
        let ff = self.ff1.forward(&x)?;
        let ff = ff.gelu()?;
        let ff = self.ff2.forward(&ff)?;
        let out = x.add(&ff)?;
        self.norm2.forward(&out)
    }

    fn parameters(&self) -> Vec<flodl::nn::parameter::Parameter> {
        let mut p = self.mha.parameters();
        p.extend(self.norm1.parameters());
        p.extend(self.ff1.parameters());
        p.extend(self.ff2.parameters());
        p.extend(self.norm2.parameters());
        p
    }
}

/// 4-layer transformer encoder with learned positional embeddings.
struct TransformerEncoder {
    embedding: Embedding,
    pos_embed: Variable,
    layers: Vec<EncoderLayer>,
    output: Linear,
}

impl TransformerEncoder {
    fn new(device: Device) -> Result<Self> {
        let opts = TensorOptions { dtype: DType::Float32, device };
        let pos_data = Tensor::randn(&[1, SEQ_LEN, D_MODEL], opts)?;

        let mut layers = Vec::with_capacity(LAYERS);
        for _ in 0..LAYERS {
            layers.push(EncoderLayer::new(device)?);
        }

        Ok(Self {
            embedding: Embedding::on_device(VOCAB, D_MODEL, device)?,
            pos_embed: Variable::new(pos_data, true),
            layers,
            output: Linear::on_device(D_MODEL, VOCAB, device)?,
        })
    }
}

impl Module for TransformerEncoder {
    fn name(&self) -> &str { "transformer" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // input: [B, seq_len] token ids
        let mut x = self.embedding.forward(input)?; // [B, seq, d_model]
        x = x.add(&self.pos_embed)?;                // + positional (broadcasts)

        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        self.output.forward(&x) // [B, seq, vocab]
    }

    fn parameters(&self) -> Vec<flodl::nn::parameter::Parameter> {
        let mut p = self.embedding.parameters();
        p.push(flodl::nn::parameter::Parameter {
            variable: self.pos_embed.clone(),
            name: "pos_embed".into(),
        });
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p.extend(self.output.parameters());
        p
    }
}

pub fn run(device: Device) -> Result<BenchResult> {
    let config = BenchConfig {
        name: "transformer".into(),
        batch_size: 32,
        batches_per_epoch: 50,
        ..Default::default()
    };

    let model = TransformerEncoder::new(device)?;
    let params = model.parameters();
    let param_count = params.iter().map(|p| p.variable.numel()).sum::<i64>() as usize;
    let mut optimizer = Adam::new(&params, 1e-4);

    let opts = TensorOptions { dtype: DType::Int64, device };

    // Synthetic token data: [B, seq_len] input → [B, seq_len] targets
    let batches: Vec<(Tensor, Tensor)> = (0..config.batches_per_epoch)
        .map(|_| {
            let x = Tensor::randint(0, VOCAB, &[config.batch_size as i64, SEQ_LEN], opts).unwrap();
            let y = Tensor::randint(0, VOCAB, &[config.batch_size as i64, SEQ_LEN], opts).unwrap();
            (x, y)
        })
        .collect();

    run_benchmark(&config, param_count, |_epoch, _warmup| {
        let mut total_loss = 0.0;
        for (x, y) in &batches {
            let input = Variable::new(x.clone(), false);
            let target = Variable::new(y.clone(), false);
            let logits = model.forward(&input)?; // [B, seq, vocab]

            // Flatten for cross-entropy: [B*seq, vocab] vs [B*seq]
            let b = logits.shape()[0];
            let s = logits.shape()[1];
            let logits_flat = logits.reshape(&[b * s, VOCAB])?;
            let target_flat = target.reshape(&[b * s])?;
            let loss = cross_entropy_loss(&logits_flat, &target_flat)?;

            optimizer.zero_grad();
            loss.backward()?;
            optimizer.step()?;

            total_loss += loss.item()?;
        }
        Ok(total_loss / batches.len() as f64)
    })
}
