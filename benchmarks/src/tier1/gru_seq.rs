//! GRU sequence benchmark: GRUCell unrolled over timesteps + output projection.
//!
//! Tests recurrent cell overhead — sequential stepping with hidden state.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

const SEQ_LEN: usize = 50;
const INPUT_DIM: i64 = 256;
const HIDDEN_DIM: i64 = 512;

/// Unrolled GRU sequence model: step through timesteps, project final hidden.
struct GruSeqModel {
    gru: GRUCell,
    output: Linear,
}

impl GruSeqModel {
    fn new(device: Device) -> Result<Self> {
        Ok(Self {
            gru: GRUCell::on_device(INPUT_DIM, HIDDEN_DIM, device)?,
            output: Linear::on_device(HIDDEN_DIM, INPUT_DIM, device)?,
        })
    }
}

impl Module for GruSeqModel {
    fn name(&self) -> &str { "gru_seq" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // input: [B, seq_len, input_dim]
        // Unroll GRU over timesteps
        let mut h: Option<Variable> = None;
        for t in 0..SEQ_LEN as i64 {
            let x_t = input.select(1, t)?;
            h = Some(self.gru.forward_step(&x_t, h.as_ref())?);
        }

        // Project final hidden state
        self.output.forward(&h.unwrap())
    }

    fn parameters(&self) -> Vec<flodl::nn::parameter::Parameter> {
        let mut p = self.gru.parameters();
        p.extend(self.output.parameters());
        p
    }
}

pub fn run(device: Device) -> Result<BenchResult> {
    // Disable cuDNN benchmark for GRU: cell-level unrolling triggers
    // nondeterministic autotuning that causes massive run-to-run variance.
    flodl::set_cudnn_benchmark(false);

    let config = BenchConfig {
        name: "gru_seq".into(),
        batch_size: 128,
        batches_per_epoch: 50,
        ..Default::default()
    };

    let opts = TensorOptions { dtype: DType::Float32, device };

    let model = GruSeqModel::new(device)?;
    let params = model.parameters();
    let param_count = params.iter().map(|p| p.variable.numel()).sum::<i64>() as usize;
    let mut optimizer = Adam::new(&params, 1e-3);

    // Synthetic sequence data: [B, seq_len, input_dim] → [B, input_dim]
    let batches: Vec<(Tensor, Tensor)> = (0..config.batches_per_epoch)
        .map(|_| {
            let x = Tensor::randn(&[config.batch_size as i64, SEQ_LEN as i64, INPUT_DIM], opts).unwrap();
            let y = Tensor::randn(&[config.batch_size as i64, INPUT_DIM], opts).unwrap();
            (x, y)
        })
        .collect();

    let result = run_benchmark(&config, param_count, |_epoch, _warmup| {
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
    });

    flodl::set_cudnn_benchmark(true); // restore for subsequent benchmarks
    result
}
