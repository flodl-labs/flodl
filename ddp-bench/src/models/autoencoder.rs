//! Conv encoder-decoder on 64x64 images, MSE reconstruction.
//!
//! Tests symmetric gradient flow under parameter averaging.

use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::BatchDataSet;
use flodl::nn::Module;
use flodl::tensor::{Device, Result, Tensor};
use flodl::*;

use super::ModelDef;
use crate::config::ModelDefaults;
use crate::data::SyntheticDataSet;

pub fn def() -> ModelDef {
    ModelDef {
        name: "autoencoder",
        description: "Conv autoencoder, tests symmetric gradient flow",
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
    // Encoder: 3->64->128->256, stride 2 each, 64x64 -> 8x8
    // Decoder: 256->128->64->3, stride 2 each, 8x8 -> 64x64
    let model = FlowBuilder::from(
        Conv2d::configure(3, 64, 4)
            .with_stride(2)
            .with_padding(1)
            .on_device(device)
            .done()?,
    )
    .through(ReLU)
    .through(
        Conv2d::configure(64, 128, 4)
            .with_stride(2)
            .with_padding(1)
            .on_device(device)
            .done()?,
    )
    .through(ReLU)
    .through(
        Conv2d::configure(128, 256, 4)
            .with_stride(2)
            .with_padding(1)
            .on_device(device)
            .done()?,
    )
    .through(ReLU)
    // Decoder
    .through(
        ConvTranspose2d::configure(256, 128, 4)
            .with_stride(2)
            .with_padding(1)
            .on_device(device)
            .done()?,
    )
    .through(ReLU)
    .through(
        ConvTranspose2d::configure(128, 64, 4)
            .with_stride(2)
            .with_padding(1)
            .on_device(device)
            .done()?,
    )
    .through(ReLU)
    .through(
        ConvTranspose2d::configure(64, 3, 4)
            .with_stride(2)
            .with_padding(1)
            .on_device(device)
            .done()?,
    )
    .through(Tanh)
    .build()?;
    Ok(Box::new(model))
}

fn make_dataset(seed: u64, total_samples: usize) -> Result<Arc<dyn BatchDataSet>> {
    SyntheticDataSet::reconstruction(seed, total_samples, &[3, 64, 64])
}

fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].clone(), false);
    let pred = model.forward(&input)?;
    mse_loss(&pred, &target)
}
