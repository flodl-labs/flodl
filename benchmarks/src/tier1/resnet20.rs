//! ResNet-20 benchmark: CIFAR-shaped input, SGD with momentum and weight decay.
//!
//! The launch-heavy case: 65 small parameter tensors and a step of a few
//! milliseconds, so per-step framework overhead (the optimizer step, the
//! dispatch of many small kernels) is a visible share of the time. The
//! other tier-1 entries are dominated by a few large kernels and use Adam.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

fn conv(in_ch: i64, out_ch: i64, kernel: i64, stride: i64, device: Device) -> Result<Conv2d> {
    Conv2d::configure(in_ch, out_ch, kernel)
        .with_stride(stride)
        .with_padding(kernel / 2)
        .without_bias()
        .on_device(device)
        .done()
}

/// The residual branch of a basic block: conv, norm, ReLU, conv, norm.
fn branch(in_ch: i64, out_ch: i64, stride: i64, device: Device) -> Result<Graph> {
    FlowBuilder::from(conv(in_ch, out_ch, 3, stride, device)?)
        .through(BatchNorm2d::on_device(out_ch, device)?)
        .through(ReLU)
        .through(conv(out_ch, out_ch, 3, 1, device)?)
        .through(BatchNorm2d::on_device(out_ch, device)?)
        .build()
}

/// The projection a block's skip path takes when its shape changes.
fn projection(in_ch: i64, out_ch: i64, stride: i64, device: Device) -> Result<Graph> {
    FlowBuilder::from(conv(in_ch, out_ch, 1, stride, device)?)
        .through(BatchNorm2d::on_device(out_ch, device)?)
        .build()
}

/// ResNet-20 for 32x32 inputs (He et al. 2015) as a graph: a stem and
/// three stages of three basic blocks at 16, 32 and 64 channels. `also`
/// is the residual connection; `also_with` projects the skip path where
/// a stage changes shape.
pub fn resnet20(device: Device) -> Result<Graph> {
    let mut fb = FlowBuilder::from(conv(3, 16, 3, 1, device)?)
        .through(BatchNorm2d::on_device(16, device)?)
        .through(ReLU);
    for (in_ch, out_ch, stride) in [(16, 16, 1), (16, 32, 2), (32, 64, 2)] {
        fb = if in_ch == out_ch {
            fb.also(branch(in_ch, out_ch, stride, device)?)
        } else {
            fb.also_with(
                projection(in_ch, out_ch, stride, device)?,
                branch(in_ch, out_ch, stride, device)?,
            )
        }
        .through(ReLU);
        for _ in 1..3 {
            fb = fb.also(branch(out_ch, out_ch, 1, device)?).through(ReLU);
        }
    }
    fb.through(AdaptiveAvgPool2d::new([1, 1]))
        .through(Flatten::default())
        .through(Linear::on_device(64, 10, device)?)
        .build()
}

pub fn run(device: Device) -> Result<BenchResult> {
    let config = BenchConfig {
        name: "resnet20".into(),
        batch_size: 64,
        batches_per_epoch: 100,
        ..Default::default()
    };

    let model = resnet20(device)?;
    let params = model.parameters();
    let mut optimizer = SGD::new(&params, 0.1, 0.9).weight_decay(1e-4);
    model.train();

    let image_opts = TensorOptions { dtype: DType::Float32, device };
    let label_opts = TensorOptions { dtype: DType::Int64, device };
    let batches: Vec<(Tensor, Tensor)> = (0..config.batches_per_epoch)
        .map(|_| {
            let x = Tensor::randn(&[config.batch_size as i64, 3, 32, 32], image_opts).unwrap();
            let y = Tensor::randint(0, 10, &[config.batch_size as i64], label_opts).unwrap();
            (x, y)
        })
        .collect();

    run_benchmark(&config, &params, |_epoch, _warmup| {
        let mut total_loss = 0.0;
        for (x, y) in &batches {
            let input = Variable::new(x.clone(), false);
            let target = Variable::new(y.clone(), false);
            let loss = cross_entropy_loss(&model.forward(&input)?, &target)?;

            optimizer.zero_grad();
            loss.backward()?;
            optimizer.step()?;

            total_loss += loss.item()?;
        }
        Ok(total_loss / batches.len() as f64)
    })
}
