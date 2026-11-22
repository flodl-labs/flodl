//! Optimizer-step benchmarks: the step alone, no forward and no backward.
//!
//! The tensors are ResNet-20's 65 parameters with a fixed gradient set
//! once, so a row prices what one `step()` costs for a model made of many
//! small tensors. That cost is all framework: the same handful of
//! elementwise updates, issued as a few batched kernels or as a loop over
//! tensors.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};
use crate::tier1::resnet20::ResNet20;

/// Steps per measured "epoch". The comparison reports time per step.
const STEPS: usize = 200;

/// The optimizers measured, with the settings both sides use.
pub const NAMES: &[&str] =
    &["optim_sgd", "optim_adam", "optim_adamw", "optim_rmsprop", "optim_adagrad", "optim_nadam", "optim_radam"];

fn build(name: &str, params: &[Parameter]) -> Box<dyn Optimizer> {
    // A small rate keeps the weights finite over tens of thousands of
    // steps on a constant gradient; the step's cost does not depend on it.
    let lr = 1e-6;
    match name {
        "optim_sgd" => Box::new(SGD::new(params, lr, 0.9).weight_decay(1e-4)),
        "optim_adam" => Box::new(Adam::new(params, lr)),
        "optim_adamw" => Box::new(AdamW::new(params, lr, 1e-2)),
        "optim_rmsprop" => Box::new(RMSprop::new(params, lr)),
        "optim_adagrad" => Box::new(Adagrad::new(params, lr)),
        "optim_nadam" => Box::new(NAdam::new(params, lr)),
        "optim_radam" => Box::new(RAdam::new(params, lr)),
        other => panic!("unknown optimizer benchmark `{other}`"),
    }
}

pub fn run(name: &'static str, device: Device) -> Result<BenchResult> {
    let config = BenchConfig {
        name: name.into(),
        batch_size: 1,
        batches_per_epoch: STEPS,
        ..Default::default()
    };

    let params = ResNet20::new(device)?.parameters();

    // One backward of sum(p * c) leaves grad(p) = c on every parameter.
    // The steps never zero it, so each one sees the same gradient.
    let opts = TensorOptions { dtype: DType::Float32, device };
    for p in &params {
        let c = Variable::new(Tensor::randn(&p.variable.shape(), opts)?, false);
        p.variable.mul(&c)?.sum()?.backward()?;
    }

    let mut optimizer = build(name, &params);
    run_benchmark(&config, &params, |_epoch, _warmup| {
        for _ in 0..STEPS {
            optimizer.step()?;
        }
        Ok(0.0)
    })
}
