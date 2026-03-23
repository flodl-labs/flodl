//! Benchmark harness: timing, warmup, VRAM sampling, JSON output.

use std::time::Instant;

use serde::Serialize;

/// Result of a single benchmark run.
#[derive(Debug, Serialize)]
pub struct BenchResult {
    pub name: String,
    pub device: String,
    pub warmup_epochs: usize,
    pub measured_epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
    pub param_count: usize,
    pub epoch_times_ms: Vec<f64>,
    pub median_epoch_ms: f64,
    pub mean_epoch_ms: f64,
    pub min_epoch_ms: f64,
    pub max_epoch_ms: f64,
    pub final_loss: f64,
    /// Active tensor bytes (matches torch.cuda.memory_allocated semantics).
    pub vram_mb: Option<f64>,
    /// Total allocator reservation (matches torch.cuda.memory_reserved semantics).
    pub vram_reserved_mb: Option<f64>,
    pub rss_mb: f64,
}

/// Configuration for running a benchmark.
pub struct BenchConfig {
    pub name: String,
    pub warmup_epochs: usize,
    pub measured_epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
    /// Active VRAM (bytes) before benchmark setup.
    pub vram_baseline: u64,
    /// Reserved VRAM (bytes) before benchmark setup.
    pub vram_reserved_baseline: u64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            warmup_epochs: 3,
            measured_epochs: 20,
            batches_per_epoch: 100,
            batch_size: 128,
            vram_baseline: 0,
            vram_reserved_baseline: 0,
        }
    }
}

/// Flush the caching allocator and snapshot VRAM baselines (active, reserved).
/// Returns `(0, 0)` on CPU.
pub fn vram_baseline() -> (u64, u64) {
    if flodl::cuda_available() {
        flodl::cuda_empty_cache();
        let active = flodl::cuda_active_bytes().unwrap_or(0);
        let reserved = flodl::cuda_allocated_bytes().unwrap_or(0);
        (active, reserved)
    } else {
        (0, 0)
    }
}

/// Run a benchmark with the given configuration.
///
/// `run_epoch` is called for each epoch and should return the average loss.
/// It receives `(epoch_index, is_warmup)`.
pub fn run_benchmark(
    config: &BenchConfig,
    param_count: usize,
    mut run_epoch: impl FnMut(usize, bool) -> flodl::Result<f64>,
) -> flodl::Result<BenchResult> {
    let device_name = if flodl::cuda_available() {
        format!("CUDA ({})", flodl::cuda_device_name().unwrap_or_default())
    } else {
        "CPU".to_string()
    };

    // Warmup
    for i in 0..config.warmup_epochs {
        run_epoch(i, true)?;
    }

    // Sync before measurement
    #[cfg(feature = "cuda")]
    flodl::cuda_synchronize(0);

    // Measured epochs
    let mut epoch_times = Vec::with_capacity(config.measured_epochs);
    let mut final_loss = 0.0;

    for i in 0..config.measured_epochs {
        let start = Instant::now();
        final_loss = run_epoch(i, false)?;

        #[cfg(feature = "cuda")]
        flodl::cuda_synchronize(0);

        epoch_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    // VRAM — active tensors (matches torch.cuda.memory_allocated)
    let vram_mb = flodl::cuda_active_bytes()
        .ok()
        .map(|used| {
            let delta = used.saturating_sub(config.vram_baseline);
            delta as f64 / (1024.0 * 1024.0)
        });

    // VRAM — allocator reservation (matches torch.cuda.memory_reserved)
    let vram_reserved_mb = flodl::cuda_allocated_bytes()
        .ok()
        .map(|used| {
            let delta = used.saturating_sub(config.vram_reserved_baseline);
            delta as f64 / (1024.0 * 1024.0)
        });

    // RSS
    let rss_mb = flodl::rss_kb() as f64 / 1024.0;

    // Stats
    let mut sorted = epoch_times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let mean = epoch_times.iter().sum::<f64>() / epoch_times.len() as f64;
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];

    Ok(BenchResult {
        name: config.name.clone(),
        device: device_name,
        warmup_epochs: config.warmup_epochs,
        measured_epochs: config.measured_epochs,
        batches_per_epoch: config.batches_per_epoch,
        batch_size: config.batch_size,
        param_count,
        epoch_times_ms: epoch_times,
        median_epoch_ms: median,
        mean_epoch_ms: mean,
        min_epoch_ms: min,
        max_epoch_ms: max,
        final_loss,
        vram_mb,
        vram_reserved_mb,
        rss_mb,
    })
}

/// Pretty-print a benchmark result to stderr.
pub fn print_result(r: &BenchResult) {
    eprintln!("  {}", r.name);
    eprintln!("    device:     {}", r.device);
    eprintln!("    params:     {}", format_count(r.param_count));
    eprintln!("    batches:    {} x {}", r.batches_per_epoch, r.batch_size);
    eprintln!("    epochs:     {} warmup + {} measured", r.warmup_epochs, r.measured_epochs);
    eprintln!("    median:     {:.1} ms/epoch", r.median_epoch_ms);
    eprintln!("    mean:       {:.1} ms/epoch", r.mean_epoch_ms);
    eprintln!("    range:      {:.1} - {:.1} ms", r.min_epoch_ms, r.max_epoch_ms);
    eprintln!("    final loss: {:.6}", r.final_loss);
    if let Some(vram) = r.vram_mb {
        eprintln!("    VRAM alloc: {:.0} MB", vram);
    }
    if let Some(vram) = r.vram_reserved_mb {
        eprintln!("    VRAM rsrvd: {:.0} MB", vram);
    }
    eprintln!("    RSS:        {:.0} MB", r.rss_mb);
    eprintln!();
}

fn format_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}
