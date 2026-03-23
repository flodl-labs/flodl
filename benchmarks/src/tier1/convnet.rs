//! ConvNet benchmark: Conv2d → BatchNorm2d → ReLU → MaxPool2d → Linear.
//!
//! Tests convolutional pipeline throughput on image-shaped data.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

/// Classification head: adaptive_avg_pool2d → flatten → linear.
struct ClassifyHead {
    linear: Linear,
}

impl ClassifyHead {
    fn new(in_channels: i64, num_classes: i64, device: Device) -> Result<Self> {
        Ok(Self {
            linear: Linear::on_device(in_channels, num_classes, device)?,
        })
    }
}

impl Module for ClassifyHead {
    fn name(&self) -> &str { "classify_head" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let pooled = adaptive_avg_pool2d(input, [1, 1])?;
        let flat = pooled.flatten(1, -1)?;
        self.linear.forward(&flat)
    }

    fn parameters(&self) -> Vec<flodl::nn::parameter::Parameter> {
        self.linear.parameters()
    }
}

pub fn run(device: Device) -> Result<BenchResult> {
    let config = BenchConfig {
        name: "convnet".into(),
        batch_size: 128,
        batches_per_epoch: 50,
        ..Default::default()
    };

    let opts = TensorOptions { dtype: DType::Float32, device };

    // Deeper ConvNet on 64x64 images with wider channels
    // Conv → BN → ReLU → MaxPool, 5 blocks, then global avg pool → classify
    let model = FlowBuilder::from(
        Conv2d::configure(3, 64, 3).with_padding(1).on_device(device).done()?
    )
        .through(BatchNorm2d::on_device(64, device)?)
        .through(ReLU)
        .through(MaxPool2d::new(2))  // 64x64 → 32x32
        .through(
            Conv2d::configure(64, 128, 3).with_padding(1).on_device(device).done()?
        )
        .through(BatchNorm2d::on_device(128, device)?)
        .through(ReLU)
        .through(MaxPool2d::new(2))  // 32x32 → 16x16
        .through(
            Conv2d::configure(128, 256, 3).with_padding(1).on_device(device).done()?
        )
        .through(BatchNorm2d::on_device(256, device)?)
        .through(ReLU)
        .through(MaxPool2d::new(2))  // 16x16 → 8x8
        .through(
            Conv2d::configure(256, 512, 3).with_padding(1).on_device(device).done()?
        )
        .through(BatchNorm2d::on_device(512, device)?)
        .through(ReLU)
        .through(MaxPool2d::new(2))  // 8x8 → 4x4
        .through(
            Conv2d::configure(512, 512, 3).with_padding(1).on_device(device).done()?
        )
        .through(BatchNorm2d::on_device(512, device)?)
        .through(ReLU)
        .through(ClassifyHead::new(512, 100, device)?)
        .build()?;

    let params = model.parameters();
    let param_count = params.iter().map(|p| p.variable.numel()).sum::<i64>() as usize;
    let mut optimizer = Adam::new(&params, 1e-3);
    model.train();

    // Synthetic image data: [B, 3, 64, 64] → class labels [B, 100]
    let batches: Vec<(Tensor, Tensor)> = (0..config.batches_per_epoch)
        .map(|_| {
            let x = Tensor::randn(&[config.batch_size as i64, 3, 64, 64], opts).unwrap();
            let y = Tensor::randn(&[config.batch_size as i64, 100], opts).unwrap();
            (x, y)
        })
        .collect();

    run_benchmark(&config, param_count, |_epoch, _warmup| {
        let mut total_loss = 0.0;
        for (x, y) in &batches {
            let input = Variable::new(x.clone(), false);
            let target = Variable::new(y.clone(), false);
            let pred = model.forward(&input)?;
            let loss = mse_loss(&pred, &target)?;

            optimizer.zero_grad();
            loss.backward()?;
            optimizer.step()?;

            total_loss += loss.item()?;
        }
        Ok(total_loss / batches.len() as f64)
    })
}
