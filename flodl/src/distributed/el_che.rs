/// El Che: heterogeneous DDP cadence strategy.
///
/// The column marches at the slowest one's pace. The slow device
/// anchors the cadence (`anchor` batches per sync step), the fast
/// ones range ahead doing more work, and everyone rejoins at AllReduce.
/// No one waits, no one idles.
///
/// After each sync step, call [`report_timing`](ElChe::report_timing)
/// with measured wall times and AllReduce overhead. El Che refines
/// batch ratios and auto-tunes the anchor count to keep AllReduce overhead
/// below a configurable target (default 10%).
///
/// # Example
///
/// ```ignore
/// let ddp = Ddp::wrap(&[&model0, &model1], &devices)?;
/// let mut cadence = ElChe::new(2, 10);
///
/// loop {
///     let start_events = record_start_events(&devices)?;
///     for rank in 0..2 {
///         for _ in 0..cadence.batches(rank) {
///             forward_backward(rank)?;
///         }
///     }
///     let wall_ms = measure_elapsed(&start_events)?;
///
///     let sync_start = Instant::now();
///     ddp.weighted_all_reduce_gradients(cadence.batch_counts())?;
///     let sync_ms = sync_start.elapsed().as_secs_f64() * 1000.0;
///
///     cadence.report_timing(&wall_ms, cadence.batch_counts(), sync_ms);
/// }
/// ```
pub struct ElChe {
    world_size: usize,
    /// Anchor batch count (slow device processes this many per step).
    anchor: usize,
    /// Per-device batch counts for the current cadence step.
    batch_counts: Vec<usize>,
    /// Per-device milliseconds per batch (from last measurement).
    ms_per_batch: Vec<f64>,
    /// Whether at least one real measurement has been taken.
    calibrated: bool,
    /// Target: max AllReduce overhead as fraction of compute time.
    overhead_target: f64,
    /// Minimum anchor (never below initial value).
    min_anchor: usize,
    /// Maximum anchor (gradient staleness limit).
    max_anchor: usize,
    /// Maximum allowed batch difference between fastest and slowest worker.
    /// When set, workers that exceed this lead are throttled until the
    /// slowest catches up. `Some(0)` = strict lockstep (sync DDP behavior).
    max_batch_diff: Option<usize>,
}

impl ElChe {
    /// Create a new sync cadence.
    ///
    /// `world_size`: number of devices (must be >= 2).
    /// `anchor`: initial batch count for the slow device per sync step.
    ///
    /// The first step uses equal counts (`anchor` for every device).
    /// After [`report_timing`](ElChe::report_timing), ratios adapt
    /// to measured throughput.
    pub fn new(world_size: usize, anchor: usize) -> Self {
        assert!(world_size >= 2, "El Che requires at least 2 devices");
        assert!(anchor >= 1, "anchor must be >= 1");
        ElChe {
            world_size,
            anchor,
            batch_counts: vec![anchor; world_size],
            ms_per_batch: vec![0.0; world_size],
            calibrated: false,
            overhead_target: 0.10,
            min_anchor: anchor,
            max_anchor: 200,
            max_batch_diff: None,
        }
    }

    /// Set the target AllReduce overhead as a fraction of compute time.
    ///
    /// Default: 0.10 (10%). The anchor auto-tunes upward to keep overhead
    /// below this target. Lower values = fewer syncs = more gradient
    /// staleness.
    pub fn with_overhead_target(mut self, target: f64) -> Self {
        self.overhead_target = target.clamp(0.01, 0.50);
        self
    }

    /// Set the maximum anchor count (gradient staleness limit).
    ///
    /// Default: 200. Higher values allow fewer syncs but accumulate more
    /// batches of gradient before averaging. Set to 1 to sync after every
    /// slow-device batch (minimal accumulation, traditional DDP cadence).
    pub fn with_max_anchor(mut self, max: usize) -> Self {
        self.max_anchor = max.max(1);
        // Ensure min_anchor doesn't exceed max_anchor
        if self.min_anchor > self.max_anchor {
            self.min_anchor = self.max_anchor;
            self.anchor = self.anchor.clamp(self.min_anchor, self.max_anchor);
        }
        self
    }

    /// Set the maximum batch difference between fastest and slowest worker.
    ///
    /// When the fastest worker leads the slowest by more than this many
    /// batches, it is throttled (paused) until the gap closes. This prevents
    /// catastrophic divergence with large batches or extreme speed ratios.
    ///
    /// - `None` (default): no limit, workers run freely.
    /// - `Some(0)`: strict lockstep, equivalent to synchronous DDP.
    /// - `Some(n)`: fast workers may lead by at most `n` batches.
    pub fn with_max_batch_diff(mut self, max: usize) -> Self {
        self.max_batch_diff = Some(max);
        self
    }

    /// Current max batch diff setting.
    pub fn max_batch_diff(&self) -> Option<usize> {
        self.max_batch_diff
    }

    /// Set initial speed estimate before the first timing measurement.
    ///
    /// `slow_rank`: which device is slowest (receives `anchor` batches).
    /// `ratio`: how many times faster the fastest device is (e.g., 3.0
    /// means the fast GPU processes ~3x more batches per unit time).
    ///
    /// Default (without this call): all devices start equal (`anchor`
    /// batches each). After the first [`report_timing`](ElChe::report_timing),
    /// actual measurements replace this estimate, so even a wrong guess
    /// self-corrects in one step.
    ///
    /// ```ignore
    /// // RTX 5060 Ti (rank 0) is ~2.3x faster than GTX 1060 (rank 1)
    /// let che = ElChe::new(2, 10).with_speed_ratio(1, 2.3);
    /// // → rank 0: 23 batches, rank 1: 10 batches
    /// ```
    pub fn with_speed_ratio(mut self, slow_rank: usize, ratio: f64) -> Self {
        assert!(
            slow_rank < self.world_size,
            "slow_rank ({slow_rank}) out of bounds for world_size ({})",
            self.world_size,
        );
        let ratio = ratio.max(1.0);
        for rank in 0..self.world_size {
            if rank == slow_rank {
                self.batch_counts[rank] = self.anchor;
            } else {
                self.batch_counts[rank] =
                    (self.anchor as f64 * ratio).round().max(1.0) as usize;
            }
        }
        self
    }

    /// Batch count for the given device rank in the current cadence step.
    pub fn batches(&self, rank: usize) -> usize {
        self.batch_counts[rank]
    }

    /// Per-device batch counts (for `Ddp::weighted_all_reduce_gradients`).
    pub fn batch_counts(&self) -> &[usize] {
        &self.batch_counts
    }

    /// Total batches across all devices for this cadence step.
    pub fn total_batches(&self) -> usize {
        self.batch_counts.iter().sum()
    }

    /// Current anchor batch count (slow device batches per step).
    pub fn anchor(&self) -> usize {
        self.anchor
    }

    /// Target wall time (ms) for one sync interval.
    ///
    /// Returns `anchor * slowest_ms_per_batch`, the intended wall-clock
    /// duration between AllReduce events. Both GPUs should accumulate
    /// this much compute time before syncing. Returns 0 if not yet
    /// calibrated (no timing data).
    pub fn anchor_wall_ms(&self) -> f64 {
        if !self.calibrated {
            return 0.0;
        }
        let slow_ms = self.ms_per_batch.iter().copied().fold(0.0_f64, f64::max);
        self.anchor as f64 * slow_ms
    }

    /// Reduce the anchor by `factor` (e.g. 0.5 = halve).
    ///
    /// One-directional correction for parameter divergence: tightens sync
    /// cadence when replicas drift apart. Does NOT loosen; ElChe's overhead
    /// auto-tune handles upward adjustment.
    ///
    /// Bypasses `min_anchor` (clamped to 1) because divergence is a stronger
    /// signal than the overhead floor. The overhead auto-tune will recover
    /// the anchor upward once divergence subsides.
    pub fn nudge_anchor_down(&mut self, factor: f64) {
        let new = (self.anchor as f64 * factor.clamp(0.1, 1.0)).ceil() as usize;
        self.anchor = new.max(1).min(self.anchor);
        // Recompute batch counts from the reduced anchor.
        let slow_ms = self.ms_per_batch
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        if slow_ms > 0.0 {
            self.recompute_batch_counts(slow_ms);
        }
    }

    /// Whether at least one timing measurement has been reported.
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Whether a speed hint was applied (batch_counts are non-uniform).
    ///
    /// Used by the coordinator to decide if epoch 0 should use
    /// throughput-proportional partitions before calibration.
    pub fn has_speed_hint(&self) -> bool {
        self.batch_counts.windows(2).any(|w| w[0] != w[1])
    }

    /// Per-device milliseconds per batch from last measurement.
    pub fn ms_per_batch(&self) -> &[f64] {
        &self.ms_per_batch
    }

    /// Report timing after a cadence step completes.
    ///
    /// `wall_ms[rank]`: wall-clock time for all batches on that device (ms).
    /// `actual_batches[rank]`: number of batches each rank actually processed
    /// since the last sync (i.e., `steps_since_avg`). In Cadence mode the fast
    /// GPU may process more batches than its intended `batch_counts` while
    /// waiting for the slow GPU to reach the trigger threshold. Using the
    /// intended count as divisor would inflate the fast GPU's ms_per_batch,
    /// inverting the throughput ratio.
    /// `sync_ms`: AllReduce overhead for this step (ms).
    ///
    /// Updates batch ratios based on measured throughput. If AllReduce
    /// overhead exceeds the target, anchor auto-tunes upward.
    pub fn report_timing(&mut self, wall_ms: &[f64], actual_batches: &[usize], sync_ms: f64) {
        assert_eq!(
            wall_ms.len(),
            self.world_size,
            "wall_ms length must match world_size",
        );

        // Compute per-batch timing for each device with adaptive EMA.
        // Alpha scales with prediction error: small jitter (thermal noise)
        // gets nearly ignored, large shifts (throttle, workload change)
        // adapt within 1-2 reports. First measurement is taken raw.
        for (rank, &wall) in wall_ms.iter().enumerate() {
            let n = actual_batches.get(rank).copied().unwrap_or(0);
            if n > 0 && wall > 0.0 {
                let new_ms = wall / n as f64;
                self.ms_per_batch[rank] = if self.calibrated && self.ms_per_batch[rank] > 0.0 {
                    let error = (new_ms - self.ms_per_batch[rank]).abs()
                        / self.ms_per_batch[rank];
                    let alpha = error.clamp(0.1, 0.8);
                    alpha * new_ms + (1.0 - alpha) * self.ms_per_batch[rank]
                } else {
                    new_ms
                };
            }
        }

        // Find the slowest device (highest ms per batch).
        let slow_ms = self
            .ms_per_batch
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);

        if slow_ms <= 0.0 {
            return; // no valid timing
        }

        // Auto-tune anchor: increase aggressively if AllReduce overhead
        // exceeds target, decay slowly (one step at a time) when overhead
        // drops well below target. Asymmetric response prevents oscillation
        // while still recovering from over-correction.
        let compute_ms = wall_ms
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        if compute_ms > 0.0 && sync_ms > 0.0 {
            let overhead = sync_ms / compute_ms;
            if overhead > self.overhead_target {
                // Aggressive increase to reduce overhead.
                let scale = overhead / self.overhead_target;
                let new_anchor =
                    (self.anchor as f64 * scale).ceil() as usize;
                self.anchor =
                    new_anchor.clamp(self.min_anchor, self.max_anchor);
            } else if overhead < self.overhead_target * 0.5
                      && self.anchor > self.min_anchor {
                // Gradual decay: only when overhead is less than half the
                // target, and only one step at a time. Prevents anchor
                // from staying inflated after a transient overhead spike.
                self.anchor -= 1;
            }
        }

        // Recompute batch counts from (possibly updated) anchor.
        self.recompute_batch_counts(slow_ms);
        self.calibrated = true;
    }

    /// Clamp batch counts to a maximum total, preserving proportions.
    ///
    /// Returns a new batch-count vector. Use near epoch boundaries to
    /// avoid consuming more batches than remain.
    pub fn clamp_total(&self, max_total: usize) -> Vec<usize> {
        let current_total = self.total_batches();
        if current_total <= max_total {
            return self.batch_counts.clone();
        }
        let scale = max_total as f64 / current_total as f64;
        let mut clamped: Vec<usize> = self
            .batch_counts
            .iter()
            .map(|&n| (n as f64 * scale).floor().max(1.0) as usize)
            .collect();
        // Distribute remainder to stay exactly at max_total.
        let sum: usize = clamped.iter().sum();
        let mut remainder = max_total.saturating_sub(sum);
        for c in &mut clamped {
            if remainder == 0 {
                break;
            }
            *c += 1;
            remainder -= 1;
        }
        clamped
    }

    /// Recompute batch counts: slow device gets `anchor`, faster devices
    /// get proportionally more based on their ms_per_batch.
    ///
    /// Applies a dead zone: a rank's count only changes when the new value
    /// differs from the current by more than 10%. This prevents batch count
    /// oscillation from minor speed fluctuations (thermal jitter, OS noise)
    /// while still adapting to genuine throughput shifts within a few reports.
    fn recompute_batch_counts(&mut self, slow_ms: f64) {
        for rank in 0..self.world_size {
            let ms = self.ms_per_batch[rank];
            let target = if ms <= 0.0 || (ms - slow_ms).abs() < 1e-6 {
                self.anchor
            } else {
                let ratio = slow_ms / ms;
                (self.anchor as f64 * ratio).round().max(1.0) as usize
            };

            let current = self.batch_counts[rank];
            let diff = (target as f64 - current as f64).abs();
            // Dead zone: only update if change exceeds 10% of current count.
            // Always update on first calibration (current == anchor for all).
            if diff > current as f64 * 0.10 || !self.calibrated {
                // Clamp per-update change to max_batch_diff (if set).
                // Without this, a sudden speed change (thermal throttle, power
                // limit) can cause the batch count to jump far beyond the
                // intended limit in a single update, and the reactive throttle
                // in check_throttle() only catches it one tick later.
                let clamped = match self.max_batch_diff {
                    Some(max) if self.calibrated => {
                        if target > current {
                            current.saturating_add(max).min(target)
                        } else {
                            current.saturating_sub(max).max(target).max(1)
                        }
                    }
                    _ => target,
                };
                self.batch_counts[rank] = clamped;
            }
        }
    }
}

