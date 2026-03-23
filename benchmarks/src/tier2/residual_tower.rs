//! Residual tower benchmark: 8 layers with skip connections via `.also()`.
//!
//! Tests graph builder overhead for deep residual networks.
//! PyTorch equivalent requires manual `x = x + block(x)` in forward().

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

const DIM: i64 = 1024;
const NUM_BLOCKS: usize = 12;

pub fn run(device: Device, vram_baseline: u64, vram_reserved_baseline: u64) -> Result<BenchResult> {
    let config = BenchConfig {
        name: "residual_tower".into(),
        batch_size: 256,
        batches_per_epoch: 50,
        vram_baseline,
        vram_reserved_baseline,
        ..Default::default()
    };

    let opts = TensorOptions { dtype: DType::Float32, device };

    // Build: projection → 8 residual blocks → output projection
    let mut builder = FlowBuilder::from(Linear::on_device(DIM, DIM, device)?);

    for _ in 0..NUM_BLOCKS {
        // Each residual block: also(Linear → GELU → LayerNorm)
        // x = x + LayerNorm(GELU(Linear(x)))
        builder = builder.also(
            FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
                .through(GELU)
                .through(LayerNorm::on_device(DIM, device)?)
                .build()?
        );
    }

    let model = builder
        .through(Linear::on_device(DIM, DIM, device)?)
        .build()?;

    let params = model.parameters();
    let param_count = params.iter().map(|p| p.variable.numel()).sum::<i64>() as usize;
    let mut optimizer = Adam::new(&params, 1e-3);
    model.train();

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
