//! Iterative refinement benchmark: encoder → refinement loop (for_n) → decoder.
//!
//! Tests `.loop_body().for_n()` — a fixed number of refinement passes.
//! PyTorch equivalent: Python for-loop in forward() calling the same submodule.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

const DIM: i64 = 1024;
const REFINE_STEPS: usize = 8;

pub fn run(device: Device) -> Result<BenchResult> {
    let config = BenchConfig {
        name: "iterative_refine".into(),
        batch_size: 256,
        batches_per_epoch: 50,
        ..Default::default()
    };

    let opts = TensorOptions { dtype: DType::Float32, device };

    // Refinement block: Linear → GELU → LayerNorm with residual
    let refine_block = FlowBuilder::new()
        .also(
            FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
                .through(GELU)
                .through(LayerNorm::on_device(DIM, device)?)
                .build()?
        )
        .build()?;

    // Full model: encoder → loop(refine, 5) → decoder
    let model = FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
        .through(GELU)
        .through(LayerNorm::on_device(DIM, device)?)
        .loop_body(refine_block)
        .for_n(REFINE_STEPS)
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
