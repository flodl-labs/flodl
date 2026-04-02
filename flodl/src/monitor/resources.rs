//! System resource sampling: CPU, RAM, GPU memory, GPU utilization.
//!
//! CPU and RAM are read from `/proc/stat` and `/proc/meminfo` (Linux only).
//! GPU metrics use the FFI layer (CUDA + NVML).

use std::fs;

/// Per-device GPU snapshot.
#[derive(Debug, Clone, Default)]
pub struct GpuSnapshot {
    /// CUDA device index.
    pub device_index: u8,
    /// Device name (e.g., "NVIDIA GeForce RTX 5060 Ti").
    pub name: String,
    /// GPU utilization percentage (0-100), None if NVML unavailable.
    pub util_percent: Option<f32>,
    /// Bytes reserved by the CUDA caching allocator.
    pub vram_allocated_bytes: Option<u64>,
    /// Total physical VRAM in bytes.
    pub vram_total_bytes: Option<u64>,
}

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
    /// Aggregate: device 0 for backward compat.
    pub gpu_util_percent: Option<f32>,
    /// Total physical VRAM in bytes (device 0).
    pub vram_total_bytes: Option<u64>,
    /// Bytes reserved by the CUDA caching allocator (device 0).
    pub vram_allocated_bytes: Option<u64>,
    /// Per-GPU snapshots (empty on CPU builds).
    pub gpus: Vec<GpuSnapshot>,
}

impl ResourceSample {
    /// Format a compact resource summary string.
    ///
    /// Example: `"CPU: 45% | RAM: 3.2/7.8 GB | GPU: 82% | VRAM: 2.1 GB / 0 KB"`
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

        if self.gpus.len() > 1 {
            // Multi-GPU: show per-device VRAM
            for gpu in &self.gpus {
                if let Some(alloc) = gpu.vram_allocated_bytes {
                    let spill = match gpu.vram_total_bytes {
                        Some(total) if alloc > total => alloc - total,
                        _ => 0,
                    };
                    let util = gpu.util_percent.map(|u| format!(" ({:.0}%)", u)).unwrap_or_default();
                    parts.push(format!(
                        "GPU{}: {} / {}{}",
                        gpu.device_index,
                        super::format::format_bytes(alloc),
                        super::format::format_bytes(spill),
                        util,
                    ));
                }
            }
        } else {
            // Single GPU or CPU
            if let Some(gpu) = self.gpu_util_percent {
                parts.push(format!("GPU: {:.0}%", gpu));
            }
            if let Some(alloc) = self.vram_allocated_bytes {
                let spill = match self.vram_total_bytes {
                    Some(total) if alloc > total => alloc - total,
                    _ => 0,
                };
                parts.push(format!(
                    "VRAM: {} / {}",
                    super::format::format_bytes(alloc),
                    super::format::format_bytes(spill),
                ));
            }
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
    /// Create a new sampler, capturing an initial CPU reading for the first delta.
    pub fn new() -> Self {
        let prev_cpu = read_cpu_times();
        Self { prev_cpu }
    }

    /// Take a snapshot of current system resources (CPU, RAM, GPU, VRAM).
    ///
    /// CPU utilization is computed as a delta since the previous call.
    /// Fields that cannot be read on this platform are `None`.
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

        // Per-GPU snapshots
        let n = crate::tensor::cuda_device_count();
        for i in 0..n {
            let mut gpu = GpuSnapshot {
                device_index: i as u8,
                ..Default::default()
            };
            if let Some(name) = crate::tensor::cuda_device_name_idx(i) {
                gpu.name = name;
            }
            if let Ok((_, total)) = crate::tensor::cuda_memory_info_idx(i) {
                gpu.vram_total_bytes = Some(total);
            }
            if let Ok(alloc) = crate::tensor::cuda_allocated_bytes_idx(i) {
                gpu.vram_allocated_bytes = Some(alloc);
            }
            if let Some(util) = crate::tensor::cuda_utilization_idx(i) {
                gpu.util_percent = Some(util as f32);
            }
            s.gpus.push(gpu);
        }

        // Aggregate fields from device 0 for backward compat
        if let Some(gpu0) = s.gpus.first() {
            s.vram_total_bytes = gpu0.vram_total_bytes;
            s.vram_allocated_bytes = gpu0.vram_allocated_bytes;
            s.gpu_util_percent = gpu0.util_percent;
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
