//! 2-layer transformer encoder, 4 heads, d_model=256, CrossEntropy.
//!
//! Tests complex gradient graphs under async averaging.

use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::BatchDataSet;
use flodl::nn::Module;
use flodl::tensor::{Device, Result, Tensor};
use flodl::*;

use super::ModelDef;
use crate::config::ModelDefaults;
use crate::data::SyntheticDataSet;

const VOCAB: i64 = 1000;
const D_MODEL: i64 = 256;
const D_FF: i64 = 512;
const N_HEADS: i64 = 4;
const SEQ_LEN: i64 = 64;

pub fn def() -> ModelDef {
    ModelDef {
        name: "transformer",
        description: "4-layer transformer, tests complex gradient graphs",
        build: build_model,
        dataset: make_dataset,
        train_fn: train_step,
        defaults: ModelDefaults {
            epochs: 5,
            batches_per_epoch: 1000,
            batch_size: 128,
            lr: 0.0005,
        },
    }
}

fn build_model(device: Device) -> Result<Box<dyn Module>> {
    let mut builder = FlowBuilder::from(Embedding::on_device(VOCAB, D_MODEL, device)?);

    // 4 transformer encoder layers
    for _ in 0..4 {
        builder = builder
            .also(
                FlowBuilder::from(MultiheadAttention::on_device(D_MODEL, N_HEADS, device)?)
                    .build()?,
            )
            .through(LayerNorm::on_device(D_MODEL, device)?)
            .also(
                FlowBuilder::from(Linear::on_device(D_MODEL, D_FF, device)?)
                    .through(GELU)
                    .through(Linear::on_device(D_FF, D_MODEL, device)?)
                    .build()?,
            )
            .through(LayerNorm::on_device(D_MODEL, device)?);
    }

    let model = builder
        .through(Linear::on_device(D_MODEL, VOCAB, device)?)
        .build()?;
    Ok(Box::new(model))
}

fn make_dataset(seed: u64, total_samples: usize) -> Result<Arc<dyn BatchDataSet>> {
    SyntheticDataSet::token_sequence(seed, total_samples, SEQ_LEN, VOCAB)
}

fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].to_dtype(flodl::tensor::DType::Int64)?, false);
    let target = batch[1].to_dtype(flodl::tensor::DType::Int64)?;

    let pred = model.forward(&input)?;
    // Flatten [B, seq, vocab] -> [B*seq, vocab] for cross entropy
    let shape = pred.shape();
    let flat_pred = pred.reshape(&[-1, shape[shape.len() - 1]])?;
    let flat_target = Variable::new(target.reshape(&[-1])?, false);
    cross_entropy_loss(&flat_pred, &flat_target)
}
