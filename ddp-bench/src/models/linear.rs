//! Linear regression canary: Linear(4096 -> 1024), MSE.
//!
//! The simplest possible model. If this doesn't converge, everything is broken.

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
const OUTPUT_DIM: i64 = 1024;

pub fn def() -> ModelDef {
    ModelDef {
        name: "linear",
        description: "Canary: Linear(4096->1024), MSE",
        build: build_model,
        dataset: make_dataset,
        train_fn: train_step,
        defaults: ModelDefaults {
            epochs: 5,
            batches_per_epoch: 1000,
            batch_size: 1024,
            lr: 0.001,
        },
    }
}

fn build_model(device: Device) -> Result<Box<dyn Module>> {
    let model = FlowBuilder::from(Linear::on_device(INPUT_DIM, OUTPUT_DIM, device)?).build()?;
    Ok(Box::new(model))
}

fn make_dataset(seed: u64, total_samples: usize) -> Result<Arc<dyn BatchDataSet>> {
    SyntheticDataSet::regression(seed, total_samples, INPUT_DIM, OUTPUT_DIM)
}

fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].clone(), false);
    let pred = model.forward(&input)?;
    mse_loss(&pred, &target)
}
