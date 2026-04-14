//! ddp-bench: DDP validation and benchmark suite for flodl.
//!
//! Tests every common training pattern across all ElChe modes to validate
//! correctness and measure convergence, throughput, and GPU utilization.

mod analyze;
mod config;
mod data;
mod download;
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
    let mut data_dir = std::path::PathBuf::from("data");
    let mut monitor_port: Option<u16> = None;
    let mut validate = false;
    let mut save_baseline = false;
    let mut baseline_path = "baselines/baseline.json".to_string();
    let mut tolerance: f64 = 0.15; // 15% relative tolerance
    let mut seed: u64 = 42;
    let mut lr_scale: Option<f64> = None;
    let mut list = false;
    let mut do_report = false;
    let mut report_file: Option<String> = None;

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
            "--data-dir" => {
                i += 1;
                data_dir = std::path::PathBuf::from(&args[i]);
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
            "--report" => {
                do_report = true;
                // Optional: next arg can be a file path (if it doesn't start with --)
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    report_file = Some(args[i].clone());
                }
            }
        "--lr-scale" => {
                i += 1;
                lr_scale = Some(args[i].parse().expect("invalid --lr-scale"));
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

    // Report mode: analyze saved timeline data from runs/
    if do_report {
        let runs = analyze::discover_runs(&output);

        // Apply filters
        let filtered: Vec<(String, String)> = runs
            .into_iter()
            .filter(|(m, _)| model_filter.as_ref().is_none_or(|f| f == "all" || f == m))
            .filter(|(_, mode)| mode_filter.as_ref().is_none_or(|f| f == "all" || f == mode))
            .collect();

        if filtered.is_empty() {
            eprintln!("no runs found in {output}/");
            return Ok(());
        }

        eprintln!("analyzing {} runs from {output}/", filtered.len());

        // Load and analyze
        let mut analyses: Vec<analyze::RunAnalysis> = Vec::new();
        let mut gpu_info: Vec<String> = Vec::new();
        for (model, mode) in &filtered {
            let run_dir = std::path::Path::new(&output).join(model).join(mode);
            let log_path = run_dir.join("training.log");
            let tl_path = run_dir.join("timeline.json");

            // Parse training log (required -- source of truth for loss/eval).
            let log = match analyze::parse_training_log(&log_path) {
                Ok(log) => log,
                Err(e) => {
                    eprintln!("  skip {model}/{mode}: {e}");
                    continue;
                }
            };

            // Capture GPU info from the first log that has it.
            if gpu_info.is_empty() && !log.gpu_info.is_empty() {
                gpu_info.clone_from(&log.gpu_info);
            }

            // Timeline is optional (provides GPU utilization, idle, sync data).
            let mut a = if let Ok(tl) = analyze::load_timeline(&tl_path) {
                analyze::analyze(model, mode, &tl)
            } else {
                analyze::empty_analysis(model, mode)
            };

            // Apply training log data (overrides timeline-derived loss/epochs).
            analyze::apply_training_log(&mut a, &log);

            analyses.push(a);
        }

        // Group by model
        let mut groups: Vec<(String, Vec<analyze::RunAnalysis>)> = Vec::new();
        for a in analyses {
            if let Some(g) = groups.iter_mut().find(|(m, _)| *m == a.model) {
                g.1.push(a);
            } else {
                let model = a.model.clone();
                groups.push((model, vec![a]));
            }
        }

        // Collect all known mode names for missing-run detection.
        let all_modes: Vec<String> = config::DdpMode::all_names()
            .iter().map(|s| s.to_string()).collect();

        let refs: std::collections::HashMap<String, report::ModelRef> =
            models::model_references().into_iter()
                .map(|(k, note, eval, hib)| (k.to_string(), report::ModelRef {
                    note: note.to_string(),
                    published_eval: eval,
                    higher_is_better: hib,
                }))
                .collect();
        let md = report::generate_report(&groups, &refs, &gpu_info, &all_modes);
        if let Some(ref path) = report_file {
            std::fs::write(path, &md)
                .map_err(|e| flodl::tensor::TensorError::new(&format!("cannot write {path}: {e}")))?;
            eprintln!("report saved to {path}");
        } else {
            print!("{md}");
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
    #[cfg(feature = "cuda")]
    for dev in flodl::tensor::cuda_devices() {
        eprintln!(
            "  gpu{}: {} ({}GB, sm_{}{})",
            dev.index, dev.name, dev.total_memory / (1024 * 1024 * 1024),
            dev.sm_major, dev.sm_minor,
        );
    }

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

            let run_epochs = epochs.unwrap_or(defaults.epochs);

            // LR scaling: explicit --lr-scale takes priority.
            // When unset and epochs > default, auto-scale LR down so
            // convergence stretches across the extra epochs.
            // E.g. 10 epochs with default 5 -> lr_scale = 0.5
            let effective_lr = if let Some(scale) = lr_scale {
                defaults.lr * scale
            } else if run_epochs > defaults.epochs {
                defaults.lr * (defaults.epochs as f64 / run_epochs as f64)
            } else {
                defaults.lr
            };

            let run_config = RunConfig {
                epochs: run_epochs,
                batches_per_epoch: batches.unwrap_or(defaults.batches_per_epoch),
                batch_size: batch_size.unwrap_or(defaults.batch_size),
                lr: effective_lr,
                seed,
                output_dir: output.clone(),
                data_dir: data_dir.clone(),
                monitor_port,
            };

            match harness::run_combo(model_def, mode, &run_config) {
                Ok(result) => model_results.push(result),
                Err(e) => {
                    eprintln!("  FAILED: {} / {}: {}", model_def.name, mode, e);
                }
            }
        }

        all_results.push(model_results);
    }

    // Flatten results for baseline operations
    let flat_results: Vec<&harness::RunResult> = all_results.iter().flat_map(|v| v.iter()).collect();

    // Save baseline
    if save_baseline {
        let baselines = report::results_to_baselines(
            &flat_results.iter().map(|r| (*r).clone()).collect::<Vec<_>>(),
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
    eprintln!("  --lr-scale <F>       Multiply default LR (auto-scales when epochs > default)");
    eprintln!("  --output <DIR>       Output directory (default: runs/)");
    eprintln!("  --data-dir <PATH>    Dataset cache directory (default: data/)");
    eprintln!("  --monitor <PORT>     Live dashboard port");
    eprintln!("  --validate           Check results against baselines");
    eprintln!("  --save-baseline      Save results as baseline");
    eprintln!("  --baseline <PATH>    Baseline file (default: baselines/baseline.json)");
    eprintln!("  --tolerance <F>      Validation tolerance, 0.0-1.0 (default: 0.15)");
    eprintln!("  --seed <N>           RNG seed (default: 42)");
    eprintln!("  --report [FILE]      Analyze runs/ and print report (or save to FILE)");
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
    eprintln!("  ddp-bench --report                                 # all runs");
    eprintln!("  ddp-bench --report --model linear                  # one model");
    eprintln!("  ddp-bench --report --model linear --mode nccl-cadence  # one case");
}
