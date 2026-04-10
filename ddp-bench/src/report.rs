//! Comparison report generation and baseline validation.

use std::collections::HashMap;

use crate::harness::RunResult;

// ---------------------------------------------------------------------------
// Baselines
// ---------------------------------------------------------------------------

/// A baseline entry: expected loss for a (model, mode) pair.
#[derive(Debug, Clone)]
pub struct Baseline {
    pub model: String,
    pub mode: String,
    pub loss: f64,
    pub epochs: usize,
    pub batches: usize,
    pub batch_size: usize,
}

/// Load baselines from a JSON file.
///
/// Format: `[{"model":"linear","mode":"solo-0","loss":1.23,"epochs":5,"batches":1000,"batch_size":64}, ...]`
pub fn load_baselines(path: &str) -> Result<Vec<Baseline>, String> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {path}: {e}"))?;
    // Minimal JSON parsing (no serde on Baseline to keep deps light)
    parse_baselines(&data)
}

/// Save baselines to a JSON file.
pub fn save_baselines(path: &str, baselines: &[Baseline]) -> Result<(), String> {
    let mut out = String::from("[\n");
    for (i, b) in baselines.iter().enumerate() {
        if i > 0 {
            out.push_str(",\n");
        }
        out.push_str(&format!(
            "  {{\"model\":\"{}\",\"mode\":\"{}\",\"loss\":{:.6},\"epochs\":{},\"batches\":{},\"batch_size\":{}}}",
            b.model, b.mode, b.loss, b.epochs, b.batches, b.batch_size,
        ));
    }
    out.push_str("\n]\n");
    std::fs::write(path, &out)
        .map_err(|e| format!("cannot write {path}: {e}"))
}

/// Build baselines from run results.
pub fn results_to_baselines(
    results: &[RunResult],
    epochs: usize,
    batches: usize,
    batch_size: usize,
) -> Vec<Baseline> {
    results
        .iter()
        .map(|r| Baseline {
            model: r.model_name.clone(),
            mode: r.mode.clone(),
            loss: r.final_loss,
            epochs,
            batches,
            batch_size,
        })
        .collect()
}

/// Validate run results against baselines. Returns (pass_count, fail_count, messages).
///
/// Matches by model name only (ignoring mode) so any DDP mode can be validated
/// against the solo-0 reference. If multiple baselines exist for a model, uses
/// the first one found.
///
/// A result passes if its final loss is within `tolerance` (relative) of the baseline.
/// Missing baselines are reported but not counted as failures.
pub fn validate_results(
    results: &[RunResult],
    baselines: &[Baseline],
    tolerance: f64,
) -> (usize, usize, Vec<String>) {
    let lookup: HashMap<&str, &Baseline> = baselines
        .iter()
        .map(|b| (b.model.as_str(), b))
        .collect();

    let mut pass = 0;
    let mut fail = 0;
    let mut msgs = Vec::new();

    for r in results {
        if let Some(b) = lookup.get(r.model_name.as_str()) {
            let rel_diff = if b.loss.abs() > 1e-10 {
                (r.final_loss - b.loss).abs() / b.loss.abs()
            } else {
                (r.final_loss - b.loss).abs()
            };

            if rel_diff <= tolerance {
                pass += 1;
                msgs.push(format!(
                    "  PASS  {:<16} {:<20} loss={:.6} (baseline={:.6}, diff={:.1}%)",
                    r.model_name, r.mode, r.final_loss, b.loss, rel_diff * 100.0,
                ));
            } else {
                fail += 1;
                msgs.push(format!(
                    "  FAIL  {:<16} {:<20} loss={:.6} (baseline={:.6}, diff={:.1}%)",
                    r.model_name, r.mode, r.final_loss, b.loss, rel_diff * 100.0,
                ));
            }
        } else {
            msgs.push(format!(
                "  SKIP  {:<16} {:<20} loss={:.6} (no baseline)",
                r.model_name, r.mode, r.final_loss,
            ));
        }
    }

    (pass, fail, msgs)
}

fn parse_baselines(json: &str) -> Result<Vec<Baseline>, String> {
    // Use serde_json since we already depend on it
    let val: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| format!("invalid JSON: {e}"))?;
    let arr = val.as_array().ok_or("expected JSON array")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let model = item["model"].as_str().ok_or("missing model")?.to_string();
        let mode = item["mode"].as_str().ok_or("missing mode")?.to_string();
        let loss = item["loss"].as_f64().ok_or("missing loss")?;
        let epochs = item["epochs"].as_u64().unwrap_or(0) as usize;
        let batches = item["batches"].as_u64().unwrap_or(0) as usize;
        let batch_size = item["batch_size"].as_u64().unwrap_or(0) as usize;
        out.push(Baseline { model, mode, loss, epochs, batches, batch_size });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Comparison tables
// ---------------------------------------------------------------------------

/// Print a comparison table for a set of results (typically one model, all modes).
pub fn print_comparison(results: &[RunResult]) {
    if results.is_empty() {
        return;
    }

    let model = &results[0].model_name;
    let n_epochs = results[0].epoch_times_ms.len();

    eprintln!("\n=== {} ({} epochs) ===\n", model, n_epochs);
    eprintln!(
        "  {:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "mode", "loss", "epoch_ms", "total_s", "syncs", "avg_sync", "idle%_0", "idle%_1",
    );
    eprintln!("  {}", "-".repeat(96));

    for r in results {
        let median_epoch = if r.epoch_times_ms.is_empty() {
            0.0
        } else {
            let mut sorted = r.epoch_times_ms.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };

        let s = &r.timeline_summary;
        let idle0 = s.gpu_idle_pct.first().copied().unwrap_or(0.0);
        let idle1 = s.gpu_idle_pct.get(1).copied().unwrap_or(0.0);

        eprintln!(
            "  {:<20} {:>10.6} {:>9.0}ms {:>9.1}s {:>10} {:>9.1}ms {:>7.1}% {:>7.1}%",
            r.mode,
            r.final_loss,
            median_epoch,
            r.total_ms / 1000.0,
            s.sync_count,
            s.avg_sync_ms,
            idle0,
            idle1,
        );
    }
    eprintln!();
}

/// Print a summary across all models.
pub fn print_matrix(all_results: &[Vec<RunResult>]) {
    eprintln!("\n=== Summary Matrix ===\n");
    for group in all_results {
        if group.is_empty() {
            continue;
        }
        print_comparison(group);
    }
}
