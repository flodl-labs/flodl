//! Post-hoc timeline analysis: reads `runs/<model>/<mode>/timeline.json`,
//! detects GPU idle gaps, correlates with sync/epoch events, and produces
//! structured analysis data for report generation.

use std::path::Path;

// ---------------------------------------------------------------------------
// Timeline data (mirrors flodl::monitor::Timeline JSON format)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GpuSample {
    #[allow(dead_code)]
    pub device: u8,
    pub util: u8,
    /// CUDA caching allocator bytes (from "va" field).
    pub vram_allocated: u64,
    /// Physical VRAM used bytes (from "vu" field).
    #[allow(dead_code)]
    pub vram_used: u64,
    /// Total physical VRAM bytes (from "vt" field).
    pub vram_total: u64,
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub t: u64,
    pub gpus: Vec<GpuSample>,
}

#[derive(Debug, Clone)]
pub struct Event {
    pub t: u64,
    pub kind: EventKind,
}

#[derive(Debug, Clone)]
pub enum EventKind {
    EpochStart { epoch: usize },
    EpochEnd { epoch: usize, loss: f64 },
    SyncStart,
    SyncEnd { ms: f64 },
    CpuAvgStart,
    CpuAvgEnd { ms: f64 },
    Anchor { #[allow(dead_code)] from: usize, #[allow(dead_code)] to: usize },
    Throttle { #[allow(dead_code)] rank: usize },
}

/// Loaded timeline data for one run.
pub struct Timeline {
    pub samples: Vec<Sample>,
    pub events: Vec<Event>,
}

/// A detected GPU idle gap.
#[derive(Debug, Clone)]
pub struct IdleGap {
    pub device: u8,
    pub start_ms: u64,
    #[allow(dead_code)]
    pub end_ms: u64,
    pub duration_ms: u64,
    pub cause: IdleCause,
}

/// Classification of what caused an idle gap.
#[derive(Debug, Clone)]
pub enum IdleCause {
    /// Near an epoch boundary (epoch_end within window).
    EpochBoundary { epoch: usize },
    /// Overlaps with a sync event.
    Sync,
    /// Overlaps with CPU averaging.
    CpuAveraging,
    /// At the very start or end of training.
    Startup,
    /// No nearby event explains it.
    Unexplained,
}

impl std::fmt::Display for IdleCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IdleCause::EpochBoundary { epoch } => write!(f, "epoch-boundary({})", epoch),
            IdleCause::Sync => write!(f, "sync"),
            IdleCause::CpuAveraging => write!(f, "cpu-avg"),
            IdleCause::Startup => write!(f, "startup"),
            IdleCause::Unexplained => write!(f, "unexplained"),
        }
    }
}

/// Per-GPU VRAM statistics.
#[derive(Debug, Clone, Default)]
pub struct VramStats {
    #[allow(dead_code)]
    pub device: u8,
    /// Peak VRAM allocated (bytes) during the run.
    pub peak_allocated: u64,
    /// Mean VRAM allocated (bytes) during the run.
    pub mean_allocated: u64,
    /// Total VRAM on this device (bytes).
    pub total: u64,
}

/// Per-epoch convergence data.
#[derive(Debug, Clone)]
pub struct EpochData {
    #[allow(dead_code)]
    pub epoch: usize,
    /// Loss at end of this epoch (from the last EpochEnd event for this epoch).
    pub loss: f64,
    /// Wall-clock span for this epoch (first start to last end, ms).
    #[allow(dead_code)]
    pub wall_ms: f64,
}

/// Aggregate analysis of a single run.
#[derive(Debug, Clone)]
pub struct RunAnalysis {
    pub model: String,
    pub mode: String,
    pub total_ms: u64,
    #[allow(dead_code)]
    pub n_epochs: usize,
    pub final_loss: f64,
    /// Per-epoch convergence trajectory.
    pub epoch_data: Vec<EpochData>,
    /// Per-GPU active percentage.
    pub gpu_active_pct: Vec<f64>,
    /// Sync event count.
    pub sync_count: usize,
    /// Average sync duration (ms).
    pub avg_sync_ms: f64,
    /// Total sync time (ms).
    #[allow(dead_code)]
    pub total_sync_ms: f64,
    /// Average number of batches between syncs (total_batches / sync_count).
    pub batches_per_sync: f64,
    /// CPU averaging count and average.
    pub cpu_avg_count: usize,
    pub avg_cpu_avg_ms: f64,
    /// Anchor changes.
    pub anchor_changes: usize,
    /// Throttle events.
    pub throttle_count: usize,
    /// All detected idle gaps (multi-second focus).
    pub idle_gaps: Vec<IdleGap>,
    /// Total idle time per GPU by cause (ms).
    pub idle_by_cause: Vec<IdleByCause>,
    /// Per-GPU VRAM statistics.
    pub vram_stats: Vec<VramStats>,
    /// Total epoch overlap time (ms). Nonzero when streaming epochs overlap.
    pub epoch_overlap_ms: f64,
    /// Sync intervals: time between consecutive SyncEnd events (ms).
    pub sync_intervals: Vec<f64>,
}

/// Total idle time for one GPU broken down by cause.
#[derive(Debug, Clone, Default)]
pub struct IdleByCause {
    pub device: u8,
    pub epoch_boundary_ms: f64,
    pub sync_ms: f64,
    pub cpu_avg_ms: f64,
    pub startup_ms: f64,
    pub unexplained_ms: f64,
    pub total_ms: f64,
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load a timeline from a JSON file.
pub fn load_timeline(path: &Path) -> Result<Timeline, String> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let val: serde_json::Value = serde_json::from_str(&data)
        .map_err(|e| format!("invalid JSON in {}: {e}", path.display()))?;

    let samples = parse_samples(&val["samples"])?;
    let events = parse_events(&val["events"])?;

    Ok(Timeline { samples, events })
}

fn parse_samples(val: &serde_json::Value) -> Result<Vec<Sample>, String> {
    let arr = val.as_array().ok_or("samples is not an array")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let t = item["t"].as_u64().unwrap_or(0);
        let gpus = if let Some(gpu_arr) = item["gpus"].as_array() {
            gpu_arr
                .iter()
                .map(|g| GpuSample {
                    device: g["d"].as_u64().unwrap_or(0) as u8,
                    util: g["u"].as_u64().unwrap_or(0) as u8,
                    vram_allocated: g["va"].as_u64().unwrap_or(0),
                    vram_used: g["vu"].as_u64().unwrap_or(0),
                    vram_total: g["vt"].as_u64().unwrap_or(0),
                })
                .collect()
        } else {
            Vec::new()
        };
        out.push(Sample { t, gpus });
    }
    Ok(out)
}

fn parse_events(val: &serde_json::Value) -> Result<Vec<Event>, String> {
    let arr = val.as_array().ok_or("events is not an array")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let t = item["t"].as_u64().unwrap_or(0);
        let kind = match item["k"].as_str().unwrap_or("") {
            "epoch_start" => EventKind::EpochStart {
                epoch: item["epoch"].as_u64().unwrap_or(0) as usize,
            },
            "epoch_end" => EventKind::EpochEnd {
                epoch: item["epoch"].as_u64().unwrap_or(0) as usize,
                loss: item["loss"].as_f64().unwrap_or(0.0),
            },
            "sync_start" => EventKind::SyncStart,
            "sync_end" => EventKind::SyncEnd {
                ms: item["ms"].as_f64().unwrap_or(0.0),
            },
            "cpu_avg_start" => EventKind::CpuAvgStart,
            "cpu_avg_end" => EventKind::CpuAvgEnd {
                ms: item["ms"].as_f64().unwrap_or(0.0),
            },
            "anchor" => EventKind::Anchor {
                from: item["from"].as_u64().unwrap_or(0) as usize,
                to: item["to"].as_u64().unwrap_or(0) as usize,
            },
            "throttle" => EventKind::Throttle {
                rank: item["rank"].as_u64().unwrap_or(0) as usize,
            },
            _ => continue, // skip unknown
        };
        out.push(Event { t, kind });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/// Minimum idle gap duration to report (ms).
const MIN_GAP_MS: u64 = 500;

/// Window around an idle gap to search for correlated events (ms).
const CORRELATION_WINDOW_MS: u64 = 500;

/// Analyze a loaded timeline.
pub fn analyze(model: &str, mode: &str, tl: &Timeline) -> RunAnalysis {
    let total_ms = tl.samples.last().map(|s| s.t).unwrap_or(0);
    let n_gpus = tl.samples.first().map(|s| s.gpus.len()).unwrap_or(0);

    // GPU active %
    let sample_count = tl.samples.len();
    let mut gpu_active_pct = vec![0.0; n_gpus];
    if sample_count > 0 {
        for s in &tl.samples {
            for (i, g) in s.gpus.iter().enumerate() {
                if g.util >= 5 {
                    gpu_active_pct[i] += 1.0;
                }
            }
        }
        for v in &mut gpu_active_pct {
            *v = *v / sample_count as f64 * 100.0;
        }
    }

    // VRAM statistics per GPU
    let mut vram_stats: Vec<VramStats> = (0..n_gpus)
        .map(|i| VramStats { device: i as u8, ..Default::default() })
        .collect();
    if sample_count > 0 {
        let mut vram_sums: Vec<u64> = vec![0; n_gpus];
        for s in &tl.samples {
            for (i, g) in s.gpus.iter().enumerate() {
                if g.vram_allocated > vram_stats[i].peak_allocated {
                    vram_stats[i].peak_allocated = g.vram_allocated;
                }
                vram_sums[i] += g.vram_allocated;
                if g.vram_total > 0 {
                    vram_stats[i].total = g.vram_total;
                }
            }
        }
        for (i, sum) in vram_sums.iter().enumerate() {
            vram_stats[i].mean_allocated = sum / sample_count as u64;
        }
    }

    // Sync stats
    let mut sync_count = 0usize;
    let mut sync_total_ms = 0.0f64;
    let mut cpu_avg_count = 0usize;
    let mut cpu_avg_total_ms = 0.0f64;
    let mut anchor_changes = 0usize;
    let mut throttle_count = 0usize;

    // Track sync intervals (time between consecutive SyncEnd events)
    let mut sync_end_times: Vec<u64> = Vec::new();

    for e in &tl.events {
        match &e.kind {
            EventKind::SyncStart => sync_count += 1,
            EventKind::SyncEnd { ms } => {
                sync_total_ms += ms;
                sync_end_times.push(e.t);
            }
            EventKind::CpuAvgStart => cpu_avg_count += 1,
            EventKind::CpuAvgEnd { ms } => cpu_avg_total_ms += ms,
            EventKind::Anchor { .. } => anchor_changes += 1,
            EventKind::Throttle { .. } => throttle_count += 1,
            _ => {}
        }
    }

    let sync_intervals: Vec<f64> = sync_end_times.windows(2)
        .map(|w| (w[1] - w[0]) as f64)
        .collect();

    // Epoch info: collect all start/end events per epoch
    let mut epoch_ends: Vec<(usize, f64, u64)> = Vec::new(); // (epoch, loss, t)
    let mut epoch_starts: Vec<(usize, u64)> = Vec::new();
    for e in &tl.events {
        match &e.kind {
            EventKind::EpochEnd { epoch, loss } => epoch_ends.push((*epoch, *loss, e.t)),
            EventKind::EpochStart { epoch } => epoch_starts.push((*epoch, e.t)),
            _ => {}
        }
    }

    let final_loss = epoch_ends.last().map(|(_, l, _)| *l).unwrap_or(0.0);

    // Per-epoch data: wall time + loss trajectory
    let max_epoch = epoch_ends.iter().map(|(e, _, _)| *e).max().unwrap_or(0);
    let n_epochs = if epoch_ends.is_empty() { 0 } else { max_epoch + 1 };
    let mut epoch_data = Vec::with_capacity(n_epochs);

    // Collect epoch spans for overlap detection
    let mut epoch_spans: Vec<(u64, u64)> = Vec::with_capacity(n_epochs); // (min_start, max_end)

    for ep in 0..n_epochs {
        let starts: Vec<u64> = epoch_starts.iter()
            .filter(|(e, _)| *e == ep)
            .map(|(_, t)| *t)
            .collect();
        let ends: Vec<u64> = epoch_ends.iter()
            .filter(|(e, _, _)| *e == ep)
            .map(|(_, _, t)| *t)
            .collect();
        // Loss: use the last EpochEnd for this epoch (most complete)
        let loss = epoch_ends.iter()
            .filter(|(e, _, _)| *e == ep)
            .last()
            .map(|(_, l, _)| *l)
            .unwrap_or(0.0);

        let wall_ms = match (starts.iter().min(), ends.iter().max()) {
            (Some(&s), Some(&e)) => {
                epoch_spans.push((s, e));
                (e - s) as f64
            }
            _ => {
                epoch_spans.push((0, 0));
                0.0
            }
        };

        epoch_data.push(EpochData { epoch: ep, loss, wall_ms });
    }

    // Epoch overlap: sum of overlapping time between consecutive epoch spans
    let mut epoch_overlap_ms = 0.0f64;
    for pair in epoch_spans.windows(2) {
        let (_, prev_end) = pair[0];
        let (next_start, _) = pair[1];
        if prev_end > next_start {
            epoch_overlap_ms += (prev_end - next_start) as f64;
        }
    }

    // Estimate total batches from epoch events for batches-per-sync
    // Each epoch has total_samples / batch_size batches. We estimate from
    // the total run: n_epochs worth of batches across all ranks.
    // For now, use sync_count directly since we know the total from the run config.
    // batches_per_sync is computed externally where batch counts are known.
    // Here we store the raw sync_intervals for the report to use.

    // Idle gap detection per GPU
    let mut all_gaps: Vec<IdleGap> = Vec::new();
    let mut idle_by_cause: Vec<IdleByCause> = (0..n_gpus as u8)
        .map(|d| IdleByCause { device: d, ..Default::default() })
        .collect();

    // First training event timestamp (skip startup idle)
    let first_training_t = tl.events.first().map(|e| e.t).unwrap_or(0);

    for gpu_idx in 0..n_gpus {
        let device = gpu_idx as u8;
        let mut gap_start: Option<u64> = None;

        for s in &tl.samples {
            let util = s.gpus.get(gpu_idx).map(|g| g.util).unwrap_or(100);

            if util < 5 {
                if gap_start.is_none() {
                    gap_start = Some(s.t);
                }
            } else if let Some(start) = gap_start.take() {
                let duration = s.t.saturating_sub(start);
                if duration >= MIN_GAP_MS {
                    let cause = classify_gap(start, s.t, first_training_t, &tl.events);
                    accumulate_cause(&mut idle_by_cause[gpu_idx], &cause, duration as f64);
                    all_gaps.push(IdleGap {
                        device,
                        start_ms: start,
                        end_ms: s.t,
                        duration_ms: duration,
                        cause,
                    });
                }
            }
        }

        // Trailing gap
        if let Some(start) = gap_start {
            if let Some(last) = tl.samples.last() {
                let duration = last.t.saturating_sub(start);
                if duration >= MIN_GAP_MS {
                    let cause = classify_gap(start, last.t, first_training_t, &tl.events);
                    accumulate_cause(&mut idle_by_cause[gpu_idx], &cause, duration as f64);
                    all_gaps.push(IdleGap {
                        device,
                        start_ms: start,
                        end_ms: last.t,
                        duration_ms: duration,
                        cause,
                    });
                }
            }
        }

        // Compute total
        idle_by_cause[gpu_idx].total_ms = idle_by_cause[gpu_idx].epoch_boundary_ms
            + idle_by_cause[gpu_idx].sync_ms
            + idle_by_cause[gpu_idx].cpu_avg_ms
            + idle_by_cause[gpu_idx].startup_ms
            + idle_by_cause[gpu_idx].unexplained_ms;
    }

    // batches_per_sync: caller should set this from run config, but we can
    // approximate from sync intervals if needed. Default to 0 (unknown).
    let batches_per_sync = 0.0;

    RunAnalysis {
        model: model.to_string(),
        mode: mode.to_string(),
        total_ms,
        n_epochs,
        final_loss,
        epoch_data,
        gpu_active_pct,
        sync_count,
        avg_sync_ms: if sync_count > 0 { sync_total_ms / sync_count as f64 } else { 0.0 },
        total_sync_ms: sync_total_ms,
        batches_per_sync,
        cpu_avg_count,
        avg_cpu_avg_ms: if cpu_avg_count > 0 { cpu_avg_total_ms / cpu_avg_count as f64 } else { 0.0 },
        anchor_changes,
        throttle_count,
        idle_gaps: all_gaps,
        idle_by_cause,
        vram_stats,
        epoch_overlap_ms,
        sync_intervals,
    }
}

/// Post-process: set batches_per_sync from known batch counts.
pub fn set_batches_per_sync(analysis: &mut RunAnalysis, total_batches: usize) {
    if analysis.sync_count > 0 {
        analysis.batches_per_sync = total_batches as f64 / analysis.sync_count as f64;
    }
}

/// Classify an idle gap by the nearest event.
fn classify_gap(start: u64, end: u64, first_training_t: u64, events: &[Event]) -> IdleCause {
    // Startup: gap starts before first training event
    if start <= first_training_t {
        return IdleCause::Startup;
    }

    let window_start = start.saturating_sub(CORRELATION_WINDOW_MS);
    let window_end = end + CORRELATION_WINDOW_MS;

    // Check for epoch boundaries first (most interesting)
    for e in events {
        if e.t < window_start || e.t > window_end {
            continue;
        }
        if let EventKind::EpochEnd { epoch, .. } = &e.kind {
            return IdleCause::EpochBoundary { epoch: *epoch };
        }
    }

    // Check for CPU averaging overlap
    for e in events {
        if e.t < window_start || e.t > window_end {
            continue;
        }
        if matches!(e.kind, EventKind::CpuAvgStart | EventKind::CpuAvgEnd { .. }) {
            return IdleCause::CpuAveraging;
        }
    }

    // Check for sync overlap
    for e in events {
        if e.t < window_start || e.t > window_end {
            continue;
        }
        if matches!(e.kind, EventKind::SyncStart | EventKind::SyncEnd { .. }) {
            return IdleCause::Sync;
        }
    }

    IdleCause::Unexplained
}

fn accumulate_cause(by_cause: &mut IdleByCause, cause: &IdleCause, ms: f64) {
    match cause {
        IdleCause::EpochBoundary { .. } => by_cause.epoch_boundary_ms += ms,
        IdleCause::Sync => by_cause.sync_ms += ms,
        IdleCause::CpuAveraging => by_cause.cpu_avg_ms += ms,
        IdleCause::Startup => by_cause.startup_ms += ms,
        IdleCause::Unexplained => by_cause.unexplained_ms += ms,
    }
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Discover available runs in the output directory.
/// Returns (model, mode) pairs sorted by model then mode.
pub fn discover_runs(output_dir: &str) -> Vec<(String, String)> {
    let mut runs = Vec::new();
    let base = Path::new(output_dir);
    if !base.is_dir() {
        return runs;
    }

    if let Ok(models) = std::fs::read_dir(base) {
        for model_entry in models.flatten() {
            if !model_entry.path().is_dir() {
                continue;
            }
            let model = model_entry.file_name().to_string_lossy().to_string();
            if let Ok(modes) = std::fs::read_dir(model_entry.path()) {
                for mode_entry in modes.flatten() {
                    let timeline_path = mode_entry.path().join("timeline.json");
                    if timeline_path.exists() {
                        let mode = mode_entry.file_name().to_string_lossy().to_string();
                        runs.push((model.clone(), mode));
                    }
                }
            }
        }
    }

    runs.sort();
    runs
}
