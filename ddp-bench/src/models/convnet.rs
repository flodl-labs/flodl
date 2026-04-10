//! 3-block CNN + BatchNorm + MaxPool on 64x64 images, CrossEntropy.
//!
//! Tests BatchNorm buffer synchronization across GPUs.

use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::BatchDataSet;
use flodl::nn::Module;
use flodl::tensor::{Device, Result, Tensor};
use flodl::*;

use super::ModelDef;
use crate::config::ModelDefaults;
use crate::data::SyntheticDataSet;

const NUM_CLASSES: i64 = 10;

pub fn def() -> ModelDef {
    ModelDef {
        name: "convnet",
        description: "3-block CNN + BatchNorm, tests buffer sync",
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
    // 64x64 input -> 32x32 -> 16x16 -> 8x8 -> 4x4 after 4 pools
    let model = FlowBuilder::from(
        Conv2d::configure(3, 64, 3).with_padding(1).on_device(device).done()?,
    )
    .through(BatchNorm2d::on_device(64, device)?)
    .through(ReLU)
    .through(MaxPool2d::new(2))
    .through(Conv2d::configure(64, 128, 3).with_padding(1).on_device(device).done()?)
    .through(BatchNorm2d::on_device(128, device)?)
    .through(ReLU)
    .through(MaxPool2d::new(2))
    .through(Conv2d::configure(128, 256, 3).with_padding(1).on_device(device).done()?)
    .through(BatchNorm2d::on_device(256, device)?)
    .through(ReLU)
    .through(MaxPool2d::new(2))
    .through(Conv2d::configure(256, 512, 3).with_padding(1).on_device(device).done()?)
    .through(BatchNorm2d::on_device(512, device)?)
    .through(ReLU)
    .through(MaxPool2d::new(2))
    .through(Flatten::default())
    .through(Linear::on_device(512 * 4 * 4, NUM_CLASSES, device)?)
    .build()?;
    Ok(Box::new(model))
}

fn make_dataset(seed: u64, total_samples: usize) -> Result<Arc<dyn BatchDataSet>> {
    SyntheticDataSet::classification(seed, total_samples, &[3, 64, 64], NUM_CLASSES)
}

fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = batch[1].to_dtype(flodl::tensor::DType::Int64)?;
    let target = Variable::new(target, false);
    let pred = model.forward(&input)?;
    cross_entropy_loss(&pred, &target)
}
