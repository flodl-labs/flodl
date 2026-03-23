//! MLP benchmark: Linear → GELU → LayerNorm, 5 layers.
//!
//! Tests raw matmul + activation throughput with realistic dimensions.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

const DIM: i64 = 1024;
const HIDDEN: i64 = 2048;

pub fn run(device: Device, vram_baseline: u64, vram_reserved_baseline: u64) -> Result<BenchResult> {
    let config = BenchConfig {
        name: "mlp".into(),
        batch_size: 256,
        batches_per_epoch: 50,
        vram_baseline,
        vram_reserved_baseline,
        ..Default::default()
    };

    let opts = TensorOptions { dtype: DType::Float32, device };

    // Model: 1024 → 2048 → 2048 → 2048 → 2048 → 1024
    let model = FlowBuilder::from(Linear::on_device(DIM, HIDDEN, device)?)
        .through(GELU)
        .through(LayerNorm::on_device(HIDDEN, device)?)
        .through(Linear::on_device(HIDDEN, HIDDEN, device)?)
        .through(GELU)
        .through(LayerNorm::on_device(HIDDEN, device)?)
        .through(Linear::on_device(HIDDEN, HIDDEN, device)?)
        .through(GELU)
        .through(LayerNorm::on_device(HIDDEN, device)?)
        .through(Linear::on_device(HIDDEN, HIDDEN, device)?)
        .through(GELU)
        .through(LayerNorm::on_device(HIDDEN, device)?)
        .through(Linear::on_device(HIDDEN, DIM, device)?)
        .build()?;

    let params = model.parameters();
    let param_count = params.iter().map(|p| p.variable.numel()).sum::<i64>() as usize;
    let mut optimizer = Adam::new(&params, 1e-3);
    model.train();

    // Pre-generate synthetic data
    let batches: Vec<(Tensor, Tensor)> = (0..config.batches_per_epoch)
        .map(|_| {
            let x = Tensor::randn(&[config.batch_size as i64, DIM], opts).unwrap();
            let y = Tensor::randn(&[config.batch_size as i64, DIM], opts).unwrap();
            (x, y)
        })
        .collect();

    run_benchmark(&config, param_count, |_epoch, _warmup| {
        let mut total_loss = 0.0;
        for (x, y) in &batches {
            let input = Variable::new(x.clone(), true);
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
