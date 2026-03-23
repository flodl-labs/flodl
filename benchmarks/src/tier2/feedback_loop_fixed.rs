//! Fixed-iteration feedback loop benchmark: encoder → loop(refine, N) → decoder.
//!
//! Same architecture as `feedback_loop` but uses `.for_n(MAX_ITER)` instead of
//! `.until_cond()`. Both sides always run exactly MAX_ITER iterations with no
//! halt check and no GPU→CPU sync per iteration. This isolates pure framework
//! overhead for looping from adaptive halt behavior.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

const DIM: i64 = 512;
const MAX_ITER: usize = 10;

pub fn run(device: Device) -> Result<BenchResult> {
    let config = BenchConfig {
        name: "feedback_fixed".into(),
        batch_size: 128,
        batches_per_epoch: 50,
        ..Default::default()
    };

    let opts = TensorOptions { dtype: DType::Float32, device };

    // Loop body: residual refinement block (identical to feedback_loop)
    let loop_body = FlowBuilder::new()
        .also(
            FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
                .through(GELU)
                .through(LayerNorm::on_device(DIM, device)?)
                .build()?
        )
        .build()?;

    // Full model: encoder → fixed loop (no halt) → decoder
    let model = FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
        .through(GELU)
        .through(LayerNorm::on_device(DIM, device)?)
        .loop_body(loop_body)
        .for_n(MAX_ITER)
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
