//! ddp-bench: DDP validation and benchmark suite for flodl.
//!
//! Tests every common training pattern across all ElChe modes to validate
//! correctness and measure convergence, throughput, and GPU utilization.

mod config;
mod data;
mod harness;
mod models;
mod report;

use config::{DdpMode, RunConfig};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> flodl::tensor::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut model_filter: Option<String> = None;
    let mut mode_filter: Option<String> = None;
    let mut epochs: Option<usize> = None;
    let mut batches: Option<usize> = None;
    let mut batch_size: Option<usize> = None;
    let mut output = "runs".to_string();
    let mut monitor_port: Option<u16> = None;
    let mut validate = false;
    let mut save_baseline = false;
    let mut baseline_path = "baselines/baseline.json".to_string();
    let mut tolerance: f64 = 0.15; // 15% relative tolerance
    let mut seed: u64 = 42;
    let mut list = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_filter = Some(args[i].clone());
            }
            "--mode" => {
                i += 1;
                mode_filter = Some(args[i].clone());
            }
            "--epochs" => {
                i += 1;
                epochs = Some(args[i].parse().expect("invalid --epochs"));
            }
            "--batches" => {
                i += 1;
                batches = Some(args[i].parse().expect("invalid --batches"));
            }
            "--batch-size" => {
                i += 1;
                batch_size = Some(args[i].parse().expect("invalid --batch-size"));
            }
            "--output" => {
                i += 1;
                output = args[i].clone();
            }
            "--monitor" => {
                i += 1;
                monitor_port = Some(args[i].parse().expect("invalid --monitor"));
            }
            "--validate" => {
                validate = true;
            }
            "--save-baseline" => {
                save_baseline = true;
            }
            "--baseline" => {
                i += 1;
                baseline_path = args[i].clone();
            }
            "--tolerance" => {
                i += 1;
                tolerance = args[i].parse().expect("invalid --tolerance");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("invalid --seed");
            }
            "--list" => {
                list = true;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => {
                eprintln!("unknown argument: {other}");
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if list {
        eprintln!("Models:");
        for name in models::model_names() {
            if let Some(m) = models::find_model(name) {
                eprintln!("  {:<16} {}", m.name, m.description);
            }
        }
        eprintln!("\nModes:");
        for name in DdpMode::all_names() {
            eprintln!("  {name}");
        }
        return Ok(());
    }

    // Resolve models
    let model_defs: Vec<models::ModelDef> = if let Some(ref name) = model_filter {
        if name == "all" {
            models::all_models()
        } else {
            vec![models::find_model(name).ok_or_else(|| {
                flodl::tensor::TensorError::new(&format!("unknown model: {name}"))
            })?]
        }
    } else {
        models::all_models()
    };

    // Resolve modes
    let modes: Vec<DdpMode> = if let Some(ref name) = mode_filter {
        if name == "all" {
            DdpMode::all_names()
                .iter()
                .filter_map(|n| DdpMode::parse(n))
                .collect()
        } else {
            vec![DdpMode::parse(name)
                .ok_or_else(|| flodl::tensor::TensorError::new(&format!("unknown mode: {name}")))?]
        }
    } else {
        DdpMode::all_names()
            .iter()
            .filter_map(|n| DdpMode::parse(n))
            .collect()
    };

    // Check GPU availability for multi-GPU modes
    #[cfg(feature = "cuda")]
    let gpu_count = flodl::tensor::cuda_device_count() as usize;
    #[cfg(not(feature = "cuda"))]
    let gpu_count = 0usize;

    eprintln!(
        "ddp-bench: {} models x {} modes, {} GPUs available",
        model_defs.len(),
        modes.len(),
        gpu_count
    );

    let mut all_results: Vec<Vec<harness::RunResult>> = Vec::new();

    for model_def in &model_defs {
        let defaults = &model_def.defaults;
        let mut model_results = Vec::new();

        for mode in &modes {
            // Skip multi-GPU modes if only 1 GPU
            if mode.requires_multi_gpu() && gpu_count < 2 {
                eprintln!("  skipping {} (requires 2+ GPUs, have {})", mode, gpu_count);
                continue;
            }

            let run_config = RunConfig {
                epochs: epochs.unwrap_or(defaults.epochs),
                batches_per_epoch: batches.unwrap_or(defaults.batches_per_epoch),
                batch_size: batch_size.unwrap_or(defaults.batch_size),
                lr: defaults.lr,
                seed,
                output_dir: output.clone(),
                monitor_port,
            };

            match harness::run_combo(model_def, mode, &run_config) {
                Ok(result) => model_results.push(result),
                Err(e) => {
                    eprintln!("  FAILED: {} / {}: {}", model_def.name, mode, e);
                }
            }
        }

        if !model_results.is_empty() {
            report::print_comparison(&model_results);
        }
        all_results.push(model_results);
    }

    if all_results.len() > 1 {
        report::print_matrix(&all_results);
    }

    // Flatten results for baseline operations
    let flat_results: Vec<&harness::RunResult> = all_results.iter().flat_map(|v| v.iter()).collect();

    // Save baseline
    if save_baseline {
        let bl_epochs = epochs.unwrap_or(5);
        let bl_batches = batches.unwrap_or(1000);
        let bl_batch_size = batch_size.unwrap_or(64);
        let baselines = report::results_to_baselines(
            &flat_results.iter().map(|r| (*r).clone()).collect::<Vec<_>>(),
            bl_epochs,
            bl_batches,
            bl_batch_size,
        );
        // Ensure directory exists
        if let Some(parent) = std::path::Path::new(&baseline_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match report::save_baselines(&baseline_path, &baselines) {
            Ok(()) => eprintln!("\nBaseline saved to {baseline_path} ({} entries)", baselines.len()),
            Err(e) => eprintln!("\nFailed to save baseline: {e}"),
        }
    }

    // Validate against baseline
    if validate {
        match report::load_baselines(&baseline_path) {
            Ok(baselines) => {
                let results_owned: Vec<harness::RunResult> =
                    flat_results.iter().map(|r| (*r).clone()).collect();
                let (pass, fail, msgs) = report::validate_results(&results_owned, &baselines, tolerance);
                eprintln!("\n=== Validation ({:.0}% tolerance) ===\n", tolerance * 100.0);
                for msg in &msgs {
                    eprintln!("{msg}");
                }
                eprintln!("\n{pass} passed, {fail} failed");
                if fail > 0 {
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("\nCannot validate: {e}");
                eprintln!("Run with --save-baseline first to generate baselines.");
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

fn print_help() {
    eprintln!("ddp-bench: DDP validation and benchmark suite for flodl\n");
    eprintln!("Usage: ddp-bench [OPTIONS]\n");
    eprintln!("Options:");
    eprintln!("  --model <NAME|all>   Run specific model (default: all)");
    eprintln!("  --mode <MODE|all>    Run specific DDP mode (default: all)");
    eprintln!("  --epochs <N>         Override epoch count");
    eprintln!("  --batches <N>        Override batches per epoch");
    eprintln!("  --batch-size <N>     Override batch size");
    eprintln!("  --output <DIR>       Output directory (default: runs/)");
    eprintln!("  --monitor <PORT>     Live dashboard port");
    eprintln!("  --validate           Check results against baselines");
    eprintln!("  --save-baseline      Save results as baseline");
    eprintln!("  --baseline <PATH>    Baseline file (default: baselines/baseline.json)");
    eprintln!("  --tolerance <F>      Validation tolerance, 0.0-1.0 (default: 0.15)");
    eprintln!("  --seed <N>           RNG seed (default: 42)");
    eprintln!("  --list               Show available models and modes");
    eprintln!("  --help               Show this help");
    eprintln!("\nModels:");
    for name in models::model_names() {
        eprintln!("  {name}");
    }
    eprintln!("\nModes:");
    for name in DdpMode::all_names() {
        eprintln!("  {name}");
    }
    eprintln!("\nExamples:");
    eprintln!("  ddp-bench --model linear --mode solo-0 --epochs 2");
    eprintln!("  ddp-bench --model convnet --mode nccl-cadence --monitor 3000");
    eprintln!("  ddp-bench --model all --mode all --validate");
}
