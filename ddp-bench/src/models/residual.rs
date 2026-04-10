//! 24 residual blocks via `.also()`, 2048D, MSE.
//!
//! Tests FlowBuilder graph skip connections under DDP.

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

pub fn def() -> ModelDef {
    ModelDef {
        name: "residual",
        description: "24 residual blocks via .also(), tests graph builder + DDP",
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
    let mut builder = FlowBuilder::from(Linear::on_device(DIM, DIM, device)?);
    for _ in 0..24 {
        builder = builder.also(
            FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
                .through(GELU)
                .through(LayerNorm::on_device(DIM, device)?)
                .build()?,
        );
    }
    let model = builder
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
