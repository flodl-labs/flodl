//! 8-expert Mixture of Experts via `.gate()`, 2048D, MSE.
//!
//! Tests soft-routing weights under async parameter averaging.

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
const N_EXPERTS: usize = 8;

pub fn def() -> ModelDef {
    ModelDef {
        name: "moe",
        description: "8-expert MoE via .gate(), tests routing under async",
        build: build_model,
        dataset: make_dataset,
        train_fn: train_step,
        defaults: ModelDefaults {
            epochs: 5,
            batches_per_epoch: 1000,
            batch_size: 128,
            lr: 0.001,
        },
    }
}

fn build_model(device: Device) -> Result<Box<dyn Module>> {
    let mut experts: Vec<Box<dyn Module>> = Vec::new();
    for _ in 0..N_EXPERTS {
        let expert = FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
            .through(GELU)
            .through(LayerNorm::on_device(DIM, device)?)
            .through(Linear::on_device(DIM, DIM, device)?)
            .through(GELU)
            .through(LayerNorm::on_device(DIM, device)?)
            .build()?;
        experts.push(Box::new(expert));
    }

    let model = FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
        .through(GELU)
        .gate(SoftmaxRouter::on_device(DIM, N_EXPERTS as i64, device)?, experts)
        .through(Linear::on_device(DIM, OUTPUT_DIM, device)?)
        .build()?;
    Ok(Box::new(model))
}

fn make_dataset(seed: u64, virtual_len: usize, pool_size: usize) -> Result<Arc<dyn BatchDataSet>> {
    // 8 clusters with per-cluster linear maps; router must learn cluster membership
    SyntheticDataSet::clustered_regression(
        seed, virtual_len, pool_size, DIM, OUTPUT_DIM, N_EXPERTS as i64,
    )
}

fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].clone(), false);
    let pred = model.forward(&input)?;
    mse_loss(&pred, &target)
}
