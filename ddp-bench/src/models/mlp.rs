//! 6-layer MLP: Linear(4096->8192) x3 blocks, GELU + LayerNorm, MSE.
//!
//! Tests optimizer state synchronization (Adam moments).

use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::BatchDataSet;
use flodl::nn::Module;
use flodl::tensor::{Device, Result, Tensor};
use flodl::*;

use super::ModelDef;
use crate::config::ModelDefaults;
use crate::data::SyntheticDataSet;

const INPUT_DIM: i64 = 4096;
const HIDDEN: i64 = 8192;
const OUTPUT_DIM: i64 = 2048;

pub fn def() -> ModelDef {
    ModelDef {
        name: "mlp",
        description: "3-layer MLP, tests optimizer state sync",
        build: build_model,
        dataset: make_dataset,
        train_fn: train_step,
        defaults: ModelDefaults {
            epochs: 5,
            batches_per_epoch: 1000,
            batch_size: 256,
            lr: 0.001,
        },
    }
}

fn build_model(device: Device) -> Result<Box<dyn Module>> {
    let mut builder = FlowBuilder::from(Linear::on_device(INPUT_DIM, HIDDEN, device)?)
        .through(GELU)
        .through(LayerNorm::on_device(HIDDEN, device)?);
    // Two more down-up blocks for real compute load
    for _ in 0..2 {
        builder = builder
            .through(Linear::on_device(HIDDEN, INPUT_DIM, device)?)
            .through(GELU)
            .through(LayerNorm::on_device(INPUT_DIM, device)?)
            .through(Linear::on_device(INPUT_DIM, HIDDEN, device)?)
            .through(GELU)
            .through(LayerNorm::on_device(HIDDEN, device)?);
    }
    let model = builder
        .through(Linear::on_device(HIDDEN, OUTPUT_DIM, device)?)
        .build()?;
    Ok(Box::new(model))
}

fn make_dataset(seed: u64, virtual_len: usize, pool_size: usize) -> Result<Arc<dyn BatchDataSet>> {
    // 2-layer teacher: student has excess capacity to learn this exactly
    SyntheticDataSet::teacher_mlp(seed, virtual_len, pool_size, INPUT_DIM, OUTPUT_DIM, OUTPUT_DIM)
}

fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].clone(), false);
    let pred = model.forward(&input)?;
    mse_loss(&pred, &target)
}
