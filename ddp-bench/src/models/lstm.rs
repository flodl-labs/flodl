//! 2-layer LSTM, 32 timesteps, MSE on final hidden state.
//!
//! Tests RNN gradient accumulation under async averaging.

use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::BatchDataSet;
use flodl::nn::Module;
use flodl::tensor::{Device, Result, Tensor};
use flodl::*;

use super::ModelDef;
use crate::config::ModelDefaults;
use crate::data::SyntheticDataSet;

const INPUT_DIM: i64 = 128;
const HIDDEN_DIM: i64 = 256;
const OUTPUT_DIM: i64 = 64;
const SEQ_LEN: i64 = 32;

pub fn def() -> ModelDef {
    ModelDef {
        name: "lstm",
        description: "4-layer LSTM, tests RNN gradients",
        build: build_model,
        dataset: make_dataset,
        train_fn: train_step,
        defaults: ModelDefaults {
            epochs: 5,
            batches_per_epoch: 1000,
            batch_size: 64,
            lr: 0.001,
        },
    }
}

fn build_model(device: Device) -> Result<Box<dyn Module>> {
    let model = FlowBuilder::from(LSTM::on_device(INPUT_DIM, HIDDEN_DIM, 4, true, device)?)
        .through(Linear::on_device(HIDDEN_DIM, OUTPUT_DIM, device)?)
        .build()?;
    Ok(Box::new(model))
}

fn make_dataset(seed: u64, virtual_len: usize, pool_size: usize) -> Result<Arc<dyn BatchDataSet>> {
    // Cumulative sum requires temporal accumulation across timesteps
    SyntheticDataSet::cumulative_sequence(
        seed, virtual_len, pool_size, SEQ_LEN, INPUT_DIM, OUTPUT_DIM,
    )
}

fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].clone(), false);
    let pred = model.forward(&input)?;
    // LSTM output is [batch, seq_len, output_dim]; select last timestep
    let last = pred.select(1, SEQ_LEN - 1)?;
    mse_loss(&last, &target)
}
