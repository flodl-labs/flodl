//! Convolutional autoencoder benchmark: Conv2d encoder + ConvTranspose2d decoder.
//!
//! Tests transposed convolution throughput for image reconstruction — the
//! standard architecture for autoencoders, super-resolution, and GANs.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

pub fn run(device: Device) -> Result<BenchResult> {
    let config = BenchConfig {
        name: "conv_autoenc".into(),
        batch_size: 64,
        batches_per_epoch: 50,
        ..Default::default()
    };

    let opts = TensorOptions { dtype: DType::Float32, device };

    // Encoder: Conv2d(stride=2) → BN → ReLU, 4 blocks (64x64 → 4x4)
    // Decoder: ConvTranspose2d(stride=2) → BN → ReLU, 3 blocks + final → Tanh (4x4 → 64x64)
    let model = FlowBuilder::from(
        Conv2d::configure(3, 64, 4).with_stride(2).with_padding(1).on_device(device).done()?
    )
        .through(BatchNorm2d::on_device(64, device)?)
        .through(ReLU)                                          // 64x64 → 32x32
        .through(Conv2d::configure(64, 128, 4).with_stride(2).with_padding(1).on_device(device).done()?)
        .through(BatchNorm2d::on_device(128, device)?)
        .through(ReLU)                                          // 32x32 → 16x16
        .through(Conv2d::configure(128, 256, 4).with_stride(2).with_padding(1).on_device(device).done()?)
        .through(BatchNorm2d::on_device(256, device)?)
        .through(ReLU)                                          // 16x16 → 8x8
        .through(Conv2d::configure(256, 512, 4).with_stride(2).with_padding(1).on_device(device).done()?)
        .through(BatchNorm2d::on_device(512, device)?)
        .through(ReLU)                                          // 8x8 → 4x4
        // Decoder
        .through(ConvTranspose2d::build(512, 256, 4, true, [2, 2], [1, 1], [0, 0], [1, 1], 1, device)?)
        .through(BatchNorm2d::on_device(256, device)?)
        .through(ReLU)                                          // 4x4 → 8x8
        .through(ConvTranspose2d::build(256, 128, 4, true, [2, 2], [1, 1], [0, 0], [1, 1], 1, device)?)
        .through(BatchNorm2d::on_device(128, device)?)
        .through(ReLU)                                          // 8x8 → 16x16
        .through(ConvTranspose2d::build(128, 64, 4, true, [2, 2], [1, 1], [0, 0], [1, 1], 1, device)?)
        .through(BatchNorm2d::on_device(64, device)?)
        .through(ReLU)                                          // 16x16 → 32x32
        .through(ConvTranspose2d::build(64, 3, 4, true, [2, 2], [1, 1], [0, 0], [1, 1], 1, device)?)
        .through(Tanh)                                          // 32x32 → 64x64
        .build()?;

    let params = model.parameters();
    let param_count = params.iter().map(|p| p.variable.numel()).sum::<i64>() as usize;
    let mut optimizer = Adam::new(&params, 1e-3);
    model.train();

    // Synthetic image data: [B, 3, 64, 64] → reconstruct same shape
    let batches: Vec<(Tensor, Tensor)> = (0..config.batches_per_epoch)
        .map(|_| {
            let x = Tensor::randn(&[config.batch_size as i64, 3, 64, 64], opts).unwrap();
            let y = x.clone(); // autoencoder target = input
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
