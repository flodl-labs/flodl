//! Feedback loop benchmark: encoder → adaptive loop with learned halt → decoder.
//!
//! Tests `.loop_body().until_cond()` — the body runs repeatedly until a
//! learned halt condition fires, up to a maximum iteration count.
//! This is the FBRL-style pattern: process → evaluate → loop back.
//!
//! PyTorch equivalent: a while-loop in forward() with a halt network,
//! manually threading state and managing gradient flow through iterations.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

const DIM: i64 = 512;
const MAX_ITER: usize = 10;

pub fn run(device: Device, vram_baseline: u64, vram_reserved_baseline: u64) -> Result<BenchResult> {
    let config = BenchConfig {
        name: "feedback_loop".into(),
        batch_size: 128,
        batches_per_epoch: 50,
        vram_baseline,
        vram_reserved_baseline,
        ..Default::default()
    };

    let opts = TensorOptions { dtype: DType::Float32, device };

    // Loop body: residual refinement block
    let loop_body = FlowBuilder::new()
        .also(
            FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
                .through(GELU)
                .through(LayerNorm::on_device(DIM, device)?)
                .build()?
        )
        .build()?;

    // Halt condition: learned projection → scalar (positive = halt)
    let halt = LearnedHalt::on_device(DIM, device)?;

    // Full model: encoder → adaptive loop → decoder
    let model = FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
        .through(GELU)
        .through(LayerNorm::on_device(DIM, device)?)
        .loop_body(loop_body)
        .until_cond(halt, MAX_ITER)
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
