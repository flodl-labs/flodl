//! System resource sampling: CPU, RAM, GPU memory, GPU utilization.
//!
//! CPU and RAM are read from `/proc/stat` and `/proc/meminfo` (Linux only).
//! GPU metrics use the FFI layer (CUDA + NVML).

use std::fs;

/// A snapshot of system resource usage.
#[derive(Debug, Clone, Default)]
pub struct ResourceSample {
    /// CPU utilization percentage (0-100), None if unavailable.
    pub cpu_percent: Option<f32>,
    /// RAM used by the system in bytes.
    pub ram_used_bytes: Option<u64>,
    /// Total system RAM in bytes.
    pub ram_total_bytes: Option<u64>,
    /// GPU utilization percentage (0-100), None if NVML unavailable.
    pub gpu_util_percent: Option<f32>,
    /// VRAM used in bytes.
    pub vram_used_bytes: Option<u64>,
    /// Total VRAM in bytes.
    pub vram_total_bytes: Option<u64>,
}

impl ResourceSample {
    /// Format a compact resource summary string.
    ///
    /// Example: `"CPU: 45% | RAM: 3.2/7.8 GB | GPU: 82% | VRAM: 2.1/6.0 GB"`
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        if let Some(cpu) = self.cpu_percent {
            parts.push(format!("CPU: {:.0}%", cpu));
        }
        if let (Some(used), Some(total)) = (self.ram_used_bytes, self.ram_total_bytes) {
            parts.push(format!(
                "RAM: {}/{}",
                super::format::format_bytes(used),
                super::format::format_bytes(total),
            ));
        }
        if let Some(gpu) = self.gpu_util_percent {
            parts.push(format!("GPU: {:.0}%", gpu));
        }
        if let (Some(used), Some(total)) = (self.vram_used_bytes, self.vram_total_bytes) {
            parts.push(format!(
                "VRAM: {}/{}",
                super::format::format_bytes(used),
                super::format::format_bytes(total),
            ));
        }

        parts.join(" | ")
    }
}

/// Accumulated CPU jiffies from `/proc/stat`.
#[derive(Clone)]
struct CpuTimes {
    total: u64,
    idle: u64,
}

/// Stateful resource sampler. Maintains previous CPU reading for delta computation.
pub struct ResourceSampler {
    prev_cpu: Option<CpuTimes>,
}

impl Default for ResourceSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceSampler {
    pub fn new() -> Self {
        let prev_cpu = read_cpu_times();
        Self { prev_cpu }
    }

    /// Take a snapshot of current system resources.
    pub fn sample(&mut self) -> ResourceSample {
        let mut s = ResourceSample::default();

        // CPU utilization (delta between two readings)
        if let Some(current) = read_cpu_times() {
            if let Some(ref prev) = self.prev_cpu {
                let d_total = current.total.saturating_sub(prev.total);
                let d_idle = current.idle.saturating_sub(prev.idle);
                if d_total > 0 {
                    s.cpu_percent = Some(
                        (d_total.saturating_sub(d_idle) as f32 / d_total as f32) * 100.0,
                    );
                }
            }
            self.prev_cpu = Some(current);
        }

        // RAM from /proc/meminfo
        if let Some((used, total)) = read_meminfo() {
            s.ram_used_bytes = Some(used);
            s.ram_total_bytes = Some(total);
        }

        // GPU memory via FFI (CUDA)
        if let Ok((used, total)) = crate::tensor::cuda_memory_info() {
            s.vram_used_bytes = Some(used);
            s.vram_total_bytes = Some(total);
        }

        // GPU utilization via FFI (NVML)
        if let Some(util) = crate::tensor::cuda_utilization() {
            s.gpu_util_percent = Some(util as f32);
        }

        s
    }
}

/// Parse `/proc/stat` first line for CPU jiffies.
fn read_cpu_times() -> Option<CpuTimes> {
    let content = fs::read_to_string("/proc/stat").ok()?;
    let line = content.lines().next()?;
    if !line.starts_with("cpu ") {
        return None;
    }
    let fields: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    if fields.len() < 4 {
        return None;
    }
    // Fields: user, nice, system, idle, iowait, irq, softirq, steal, ...
    let total: u64 = fields.iter().sum();
    let idle = fields[3] + fields.get(4).copied().unwrap_or(0); // idle + iowait
    Some(CpuTimes { total, idle })
}

/// Parse `/proc/meminfo` for total and available memory.
fn read_meminfo() -> Option<(u64, u64)> {
    let content = fs::read_to_string("/proc/meminfo").ok()?;
    let mut total: Option<u64> = None;
    let mut available: Option<u64> = None;

    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = parse_kb_value(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available = parse_kb_value(rest);
        }
        if total.is_some() && available.is_some() {
            break;
        }
    }

    match (total, available) {
        (Some(t), Some(a)) => Some((t.saturating_sub(a), t)),
        _ => None,
    }
}

/// Parse a value like "  16384000 kB" into bytes.
fn parse_kb_value(s: &str) -> Option<u64> {
    let val: u64 = s.split_whitespace().next()?.parse().ok()?;
    Some(val * 1024) // kB to bytes
}
