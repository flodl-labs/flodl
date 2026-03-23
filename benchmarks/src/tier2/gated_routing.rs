//! Gated routing (MoE) benchmark: soft routing with 4 expert MLPs.
//!
//! Tests `.gate()` overhead — all experts run, weighted by learned router.
//! PyTorch equivalent requires manual softmax gating + stacking expert outputs.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

const DIM: i64 = 512;
const NUM_EXPERTS: i64 = 8;

fn expert_block(device: Device) -> Result<Graph> {
    FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
        .through(GELU)
        .through(LayerNorm::on_device(DIM, device)?)
        .build()
}

pub fn run(device: Device) -> Result<BenchResult> {
    let config = BenchConfig {
        name: "gated_routing".into(),
        batch_size: 256,
        batches_per_epoch: 50,
        ..Default::default()
    };

    let opts = TensorOptions { dtype: DType::Float32, device };

    // Build: projection → gate(router, [expert x 4]) → output
    let experts: Vec<Box<dyn Module>> = (0..NUM_EXPERTS as usize)
        .map(|_| -> Box<dyn Module> { Box::new(expert_block(device).unwrap()) })
        .collect();

    let model = FlowBuilder::from(Linear::on_device(DIM, DIM, device)?)
        .through(GELU)
        .gate(
            SoftmaxRouter::on_device(DIM, NUM_EXPERTS, device)?,
            experts,
        )
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
