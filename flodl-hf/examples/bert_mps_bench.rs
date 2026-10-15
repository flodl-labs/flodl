//! Compare CPU and Apple MPS full fine-tuning throughput on a small BERT.
//!
//! Run on an Apple Silicon Mac with the `mps` feature and macOS LibTorch:
//! `cargo run --release -p flodl-hf --example bert_mps_bench --features flodl/mps`
//!
//! Prints CSV rows to stdout. The first run downloads the 17.5 MB checkpoint.

use std::time::Instant;

use flodl::{AdamW, Device, Module, Result, Tensor, Variable, gpu_synchronize, mps_available};
use flodl_hf::models::bert::BertForSequenceClassification;
use flodl_hf::tokenizer::HfTokenizer;

const MODEL: &str = "jason-zhoou/tiny-bert-sst2";
const BATCH_SIZES: &[usize] = &[16, 64, 256, 512];
const WARMUP_STEPS: usize = 2;
const MEASURED_STEPS: usize = 10;
const ROUNDS: usize = 3;

fn main() -> Result<()> {
    if !mps_available() {
        return Err(flodl_error(
            "MPS is unavailable; run this example on an Apple Silicon Mac with --features flodl/mps",
        ));
    }

    let tokenizer = HfTokenizer::from_pretrained(MODEL)?;
    println!("device,batch_size,sequence_length,median_ms_per_step,examples_per_second");

    // Alternate the order at each batch size to reduce systematic bias from
    // thermal and background-load drift during this short comparison.
    let mut rows = Vec::new();
    for (i, &batch_size) in BATCH_SIZES.iter().enumerate() {
        let devices = if i % 2 == 0 {
            [Device::CPU, Device::MPS]
        } else {
            [Device::MPS, Device::CPU]
        };
        for device in devices {
            let head = BertForSequenceClassification::from_pretrained_on_device(MODEL, device)?;
            head.graph().set_optimizer(|p| AdamW::new(p, 2e-5, 0.01));
            head.graph().set_training(true);

            // Repeated one-token words create a fixed 128-token input. Labels
            // alternate, making this a controlled throughput workload rather
            // than an accuracy or convergence benchmark.
            let positive = format!("{}", "good ".repeat(126));
            let negative = format!("{}", "bad ".repeat(126));
            let texts: Vec<&str> = (0..batch_size)
                .map(|i| {
                    if i % 2 == 0 {
                        positive.as_str()
                    } else {
                        negative.as_str()
                    }
                })
                .collect();
            let enc = tokenizer.encode_on_device(&texts, device)?;
            let seq_len = enc.input_ids.data().shape()[1] as usize;
            let label_ids: Vec<i64> = (0..batch_size).map(|i| (i % 2) as i64).collect();
            let labels = Variable::new(
                Tensor::from_i64(&label_ids, &[batch_size as i64], device)?,
                false,
            );

            for _ in 0..WARMUP_STEPS {
                let loss = head.compute_loss(&enc, &labels)?;
                loss.backward()?;
                head.graph().step()?;
            }

            let mut samples = Vec::with_capacity(ROUNDS);
            for _ in 0..ROUNDS {
                gpu_synchronize(device.index());
                let start = Instant::now();
                for _ in 0..MEASURED_STEPS {
                    let loss = head.compute_loss(&enc, &labels)?;
                    loss.backward()?;
                    head.graph().step()?;
                }
                gpu_synchronize(device.index());
                samples.push(start.elapsed() / MEASURED_STEPS as u32);
            }
            samples.sort_unstable();
            let median = samples[1];
            let ms = median.as_secs_f64() * 1000.0;
            let throughput = batch_size as f64 / median.as_secs_f64();
            let row = format!("{device},{batch_size},{seq_len},{ms:.3},{throughput:.1}");
            println!("{row}");
            rows.push((batch_size, device, ms));
        }
    }

    // Avoid silently publishing an incomplete benchmark if one side did not
    // complete all four batch sizes.
    if rows.len() != BATCH_SIZES.len() * 2 {
        return Err(flodl_error("benchmark did not produce all CPU/MPS rows"));
    }
    Ok(())
}

fn flodl_error(message: &str) -> flodl::TensorError {
    flodl::TensorError::new(message)
}
