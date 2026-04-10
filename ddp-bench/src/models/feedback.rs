//! 24 iterations via `.loop_body().for_n()`, 2048D, MSE.
//!
//! Tests iterative refinement under DDP parameter averaging.

use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::BatchDataSet;
use flodl::nn::Module;
use flodl::tensor::{Device, Result, Tensor};
use flodl::*;

use super::ModelDef;
use crate::config::ModelDefaults;
use crate::data::SyntheticDataSet;

const DIM: i64 = 2048;
const OUTPUT_DIM: i64 = 512;
const N_ITERS: usize = 24;

pub fn def() -> ModelDef {
    ModelDef {
        name: "feedback",
        description: "6-iter feedback loop, tests iterative refinement + DDP",
        build: build_model,
        dataset: make_dataset,
        train_fn: train_step,
        defaults: ModelDefaults {
            epochs: 5,
            batches_per_epoch: 1000,
            batch_size: 512,
            lr: 0.001,
        },
    }
}

fn build_model(device: Device) -> Result<Box<dyn Module>> {
    let refine_block = FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
        .through(GELU)
        .through(LayerNorm::on_device(DIM, device)?)
        .build()?;

    let model = FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
        .through(GELU)
        .through(LayerNorm::on_device(DIM, device)?)
        .loop_body(refine_block)
        .for_n(N_ITERS)
        .through(Linear::on_device(DIM, OUTPUT_DIM, device)?)
        .build()?;
    Ok(Box::new(model))
}

fn make_dataset(seed: u64, total_samples: usize) -> Result<Arc<dyn BatchDataSet>> {
    SyntheticDataSet::regression(seed, total_samples, DIM, OUTPUT_DIM)
}

fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].clone(), false);
    let pred = model.forward(&input)?;
    mse_loss(&pred, &target)
}
