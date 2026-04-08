//! Coordinator: lightweight scheduling thread for DDP run mode.

use std::sync::mpsc;
use std::time::Instant;

use crate::tensor::{Device, Result, Tensor, TensorError};

use std::collections::HashMap;

use super::{
    ApplyPolicy, AverageBackend, TimingMsg, MetricsMsg,
    ParamSnapshot, AveragedParams, ControlMsg, EpochPlan, TrainedState,
};

// ---------------------------------------------------------------------------
// CPU averaging state machine
// ---------------------------------------------------------------------------

/// State machine for non-blocking CPU averaging.
///
/// Instead of blocking the coordinator thread during snapshot collection and
/// averaging computation, the CPU path operates as a multi-tick state machine:
///
/// ```text
/// Idle -> Collecting -> Computing -> Idle
///          (try_recv      (thread
///           per tick)      join)
/// ```
///
/// This keeps [`Coordinator::check_throttle`] running every tick, even during
/// averaging. Counter snapshots taken at trigger time allow correct timing
/// attribution: batches during the averaging window carry into the next period.
enum CpuAvgState {
    /// No averaging in progress. `should_average()` may trigger a new cycle.
    Idle,
    /// Waiting for worker snapshots. `try_recv` on `param_rx` each tick.
    Collecting {
        snapshots: Vec<ParamSnapshot>,
        received: Vec<bool>,
        deadline: Instant,
        start: Instant,
        /// `steps_since_avg` at trigger time (for subtract-not-zero).
        steps_snapshot: Vec<usize>,
        /// `wall_ms_accum` at trigger time (for subtract-not-zero).
        wall_ms_snapshot: Vec<f64>,
    },
    /// `average_params` + divergence check running on a background thread.
    Computing {
        handle: std::thread::JoinHandle<Result<CpuAvgResult>>,
        start: Instant,
        steps_snapshot: Vec<usize>,
        wall_ms_snapshot: Vec<f64>,
    },
}

/// Result from the CPU averaging compute thread.
struct CpuAvgResult {
    averaged: AveragedParams,
    /// Relative divergence across replicas (for anchor correction).
    /// None if all norms are zero.
    divergence: Option<f64>,
}

// ---------------------------------------------------------------------------
// Chunk pool for progressive dispatch
// ---------------------------------------------------------------------------

/// Tracks remaining unassigned samples for one epoch during progressive dispatch.
///
/// Instead of sending the full partition at epoch start, the coordinator hands
/// out small chunks from this pool. Each `take_chunk` advances a monotonic
/// cursor, guaranteeing non-overlapping slices into the global permutation.
struct ChunkPool {
    epoch: usize,
    total_samples: usize,
    /// Next unassigned offset into the global permutation.
    cursor: usize,
    /// Per-rank: samples dispatched (sum of all chunk sizes sent).
    dispatched: Vec<usize>,
    /// Per-rank: samples completed (from MetricsMsg.samples_processed).
    completed: Vec<usize>,
    /// Per-rank: number of chunks sent.
    chunks_sent: Vec<usize>,
    /// Wall-clock start of this epoch (for EpochMetrics).
    epoch_start: Instant,
}

impl ChunkPool {
    fn new(epoch: usize, total_samples: usize, world_size: usize) -> Self {
        ChunkPool {
            epoch,
            total_samples,
            cursor: 0,
            dispatched: vec![0; world_size],
            completed: vec![0; world_size],
            chunks_sent: vec![0; world_size],
            epoch_start: Instant::now(),
        }
    }

    /// Take the next chunk of `size` samples from the pool.
    ///
    /// Returns `(offset, actual_size)` or `None` if the pool is exhausted.
    /// Actual size may be smaller than requested if near the end.
    fn take_chunk(&mut self, size: usize, rank: usize) -> Option<(usize, usize)> {
        if self.cursor >= self.total_samples {
            return None;
        }
        let actual = size.min(self.total_samples - self.cursor);
        let offset = self.cursor;
        self.cursor += actual;
        self.dispatched[rank] += actual;
        self.chunks_sent[rank] += 1;
        Some((offset, actual))
    }

    /// Samples not yet assigned to any rank.
    fn remaining(&self) -> usize {
        self.total_samples.saturating_sub(self.cursor)
    }

    /// Record that a rank completed processing some samples.
    fn mark_completed(&mut self, rank: usize, samples: usize) {
        self.completed[rank] += samples;
        debug_assert!(
            self.completed[rank] <= self.dispatched[rank],
            "rank {} completed {} samples but only {} were dispatched",
            rank,
            self.completed[rank],
            self.dispatched[rank],
        );
    }

    /// Samples dispatched but not yet completed for a given rank.
    fn in_flight(&self, rank: usize) -> usize {
        self.dispatched[rank].saturating_sub(self.completed[rank])
    }

    /// True when all samples have been dispatched AND all ranks have
    /// reported completion for everything dispatched to them.
    fn is_epoch_done(&self) -> bool {
        self.cursor >= self.total_samples
            && self.dispatched.iter().zip(&self.completed).all(|(d, c)| c >= d)
    }

    /// Epoch wall-clock time in milliseconds.
    fn epoch_elapsed_ms(&self) -> f64 {
        self.epoch_start.elapsed().as_secs_f64() * 1000.0
    }
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

/// Lightweight scheduling coordinator for DDP run mode.
///
/// NOT an optimizer. Each GPU runs its own Adam. The coordinator:
/// 1. Collects timing from workers (for ElChe throughput ratios)
/// 2. Triggers periodic parameter averaging (NCCL or CPU path)
/// 3. Monitors divergence to correct ElChe's anchor (tighten-only)
/// 4. Rebalances data partitions after ElChe calibrates
///
/// All fields are Send. Runs on a dedicated CPU thread.
pub struct Coordinator {
    // Channels
    timing_rx: mpsc::Receiver<TimingMsg>,
    metrics_rx: mpsc::Receiver<MetricsMsg>,
    /// Only used with [`AverageBackend::Cpu`].
    param_rx: mpsc::Receiver<ParamSnapshot>,
    /// Dedicated receivers for final snapshots from each worker.
    final_param_rxs: Vec<mpsc::Receiver<ParamSnapshot>>,
    control_txs: Vec<mpsc::Sender<ControlMsg>>,

    // Configuration
    policy: ApplyPolicy,
    backend: AverageBackend,
    world_size: usize,
    total_samples: usize,

    // Scheduling
    pub(super) el_che: crate::distributed::ddp::ElChe,
    version: u64,
    /// Per-rank steps since last averaging.
    steps_since_avg: Vec<usize>,

    // Divergence monitoring (correction mechanism)
    divergence_threshold: f64,
    /// Has ElChe been calibrated (first timing report received)?
    calibrated: bool,

    /// Number of workers still actively training. Decremented when the
    /// coordinator drains a [`TimingMsg::Exiting`] message. Single-writer
    /// (coordinator thread only), no race with NCCL collectives.
    pub(super) active_count: usize,

    // Timing accumulation for ElChe
    /// Accumulated wall-clock ms per rank since last averaging.
    /// Fed to ElChe::report_timing at each averaging event.
    pub(super) wall_ms_accum: Vec<f64>,
    /// Most recent batch_ms per rank (for display/monitoring).
    last_batch_ms: Vec<f64>,
    /// Most recent CPU averaging time (ms). Fed to ElChe as sync_ms so
    /// the overhead auto-tune works for the CPU backend too.
    last_avg_ms: f64,
    /// Per-rank throttle state. Prevents sending duplicate Throttle messages
    /// to workers that are already blocked. Reset after averaging.
    throttled: Vec<bool>,

    // Checkpointing
    /// Number of averaging events completed.
    avg_count: usize,
    /// Save a checkpoint every N global epochs. None = disabled.
    pub(super) checkpoint_every: Option<usize>,

    // Non-blocking CPU averaging
    /// State machine for CPU averaging (Idle/Collecting/Computing).
    avg_state: CpuAvgState,
    /// Timeout for snapshot collection (seconds).
    snapshot_timeout_secs: u64,
    /// Number of CPU averaging rounds aborted due to timeout.
    abort_count: usize,

    // Epoch metrics aggregation
    /// Channel to send aggregated epoch metrics to DdpHandle.
    epoch_metrics_tx: Option<mpsc::Sender<super::EpochMetrics>>,
    /// Buffer for collecting per-rank metrics before aggregation.
    epoch_buffer: HashMap<usize, Vec<MetricsMsg>>,
    /// CUDA device index per rank (for EpochMetrics GPU data).
    device_indices: Vec<u8>,

    // Global epoch management
    /// Total number of epochs to train.
    num_epochs: usize,
    /// What epoch each rank is currently working on (last dispatched).
    rank_epoch: Vec<usize>,
    /// True if rank finished its epoch but is blocked by lookahead (Auto mode).
    rank_waiting: Vec<bool>,
    /// Last globally-aggregated epoch (all ranks reported).
    /// None = no epoch aggregated yet.
    last_aggregated_epoch: Option<usize>,
    /// User-specified partition ratios (disables auto-rebalancing).
    partition_ratios: Option<Vec<f64>>,
    /// Cached epoch plans: computed once per epoch, consistent across ranks.
    epoch_plan_cache: HashMap<usize, Vec<EpochPlan>>,

    // Progressive chunk dispatch
    /// Whether progressive dispatch is enabled.
    progressive: bool,
    /// Active chunk pool (None between epochs or when progressive is off).
    chunk_pool: Option<ChunkPool>,
    /// Floor for chunk size (in batches). Default: 4.
    min_chunk_batches: usize,
    /// Batch size (samples per batch), needed for chunk sizing.
    batch_size: usize,
}

/// Builder for configuring a [`Coordinator`].
pub struct CoordinatorBuilder {
    timing_rx: mpsc::Receiver<TimingMsg>,
    metrics_rx: mpsc::Receiver<MetricsMsg>,
    param_rx: mpsc::Receiver<ParamSnapshot>,
    final_param_rxs: Vec<mpsc::Receiver<ParamSnapshot>>,
    control_txs: Vec<mpsc::Sender<ControlMsg>>,
    policy: ApplyPolicy,
    backend: AverageBackend,
    world_size: usize,
    total_samples: usize,
    el_che: crate::distributed::ddp::ElChe,
    divergence_threshold: f64,
    checkpoint_every: Option<usize>,
    snapshot_timeout_secs: u64,
    epoch_metrics_tx: Option<mpsc::Sender<super::EpochMetrics>>,
    device_indices: Vec<u8>,
    num_epochs: usize,
    partition_ratios: Option<Vec<f64>>,
    progressive: bool,
    batch_size: usize,
}

impl CoordinatorBuilder {
    /// Enable or disable progressive chunk dispatch.
    /// Default: true for Cadence/Async, false for Sync.
    pub fn progressive(mut self, enabled: bool) -> Self {
        self.progressive = enabled;
        self
    }

    /// Set the batch size (needed for chunk sizing in progressive mode).
    pub fn batch_size(mut self, bs: usize) -> Self {
        self.batch_size = bs;
        self
    }

    /// Set the divergence threshold for anchor correction.
    /// Default: 0.05 (5% relative norm difference nudges ElChe's anchor down).
    pub fn divergence_threshold(mut self, threshold: f64) -> Self {
        self.divergence_threshold = threshold;
        self
    }

    /// Set the AllReduce overhead target (fraction of compute time).
    /// Default: 0.10 (10%). Lower = more frequent sync, higher = less overhead.
    pub fn overhead_target(mut self, target: f64) -> Self {
        self.el_che = self.el_che.with_overhead_target(target);
        self
    }

    /// Set the maximum anchor (max batches between AllReduce).
    /// Default: 200. Controls gradient staleness bound.
    pub fn max_anchor(mut self, max: usize) -> Self {
        self.el_che = self.el_che.with_max_anchor(max);
        self
    }

    /// Set the checkpoint interval (global epochs between checkpoints).
    pub fn checkpoint_every(mut self, n: usize) -> Self {
        self.checkpoint_every = Some(n);
        self
    }

    /// Set the timeout for CPU averaging snapshot collection (seconds).
    pub fn snapshot_timeout_secs(mut self, secs: u64) -> Self {
        self.snapshot_timeout_secs = secs;
        self
    }

    /// Set the channel for forwarding aggregated epoch metrics to the main thread.
    pub fn epoch_metrics_tx(mut self, tx: mpsc::Sender<super::EpochMetrics>) -> Self {
        self.epoch_metrics_tx = Some(tx);
        self
    }

    /// Set the CUDA device indices (one per rank).
    pub fn device_indices(mut self, indices: Vec<u8>) -> Self {
        self.device_indices = indices;
        self
    }

    /// Set the total number of epochs to train.
    pub fn num_epochs(mut self, n: usize) -> Self {
        self.num_epochs = n;
        self
    }

    /// Set explicit per-rank partition ratios.
    pub fn partition_ratios(mut self, ratios: Option<Vec<f64>>) -> Self {
        self.partition_ratios = ratios;
        self
    }

    /// Build the coordinator.
    pub fn build(self) -> Coordinator {
        Coordinator {
            timing_rx: self.timing_rx,
            metrics_rx: self.metrics_rx,
            param_rx: self.param_rx,
            final_param_rxs: self.final_param_rxs,
            control_txs: self.control_txs,
            policy: self.policy,
            backend: self.backend,
            world_size: self.world_size,
            total_samples: self.total_samples,
            el_che: self.el_che,
            version: 0,
            steps_since_avg: vec![0; self.world_size],
            divergence_threshold: self.divergence_threshold,
            calibrated: false,
            active_count: self.world_size,
            wall_ms_accum: vec![0.0; self.world_size],
            last_batch_ms: vec![0.0; self.world_size],
            last_avg_ms: 0.0,
            throttled: vec![false; self.world_size],
            avg_count: 0,
            checkpoint_every: self.checkpoint_every,
            avg_state: CpuAvgState::Idle,
            snapshot_timeout_secs: self.snapshot_timeout_secs,
            abort_count: 0,
            epoch_metrics_tx: self.epoch_metrics_tx,
            epoch_buffer: HashMap::new(),
            device_indices: self.device_indices,
            num_epochs: self.num_epochs,
            rank_epoch: vec![0; self.world_size],
            rank_waiting: vec![false; self.world_size],
            last_aggregated_epoch: None,
            partition_ratios: self.partition_ratios,
            epoch_plan_cache: HashMap::new(),
            progressive: self.progressive,
            chunk_pool: None,
            min_chunk_batches: 4,
            batch_size: self.batch_size.max(1),
        }
    }
}

impl Coordinator {
    /// Create a coordinator builder.
    #[allow(clippy::too_many_arguments)]
    pub fn builder(
        timing_rx: mpsc::Receiver<TimingMsg>,
        metrics_rx: mpsc::Receiver<MetricsMsg>,
        param_rx: mpsc::Receiver<ParamSnapshot>,
        final_param_rxs: Vec<mpsc::Receiver<ParamSnapshot>>,
        control_txs: Vec<mpsc::Sender<ControlMsg>>,
        policy: ApplyPolicy,
        backend: AverageBackend,
        world_size: usize,
        total_samples: usize,
        el_che: crate::distributed::ddp::ElChe,
    ) -> CoordinatorBuilder {
        CoordinatorBuilder {
            timing_rx,
            metrics_rx,
            param_rx,
            final_param_rxs,
            control_txs,
            policy,
            backend,
            world_size,
            total_samples,
            el_che,
            divergence_threshold: 0.05,
            checkpoint_every: None,
            snapshot_timeout_secs: 5,
            epoch_metrics_tx: None,
            device_indices: (0..world_size as u8).collect(),
            num_epochs: 1,
            partition_ratios: None,
            progressive: !matches!(policy, ApplyPolicy::Sync),
            batch_size: 1,
        }
    }

    /// Current model version (bumped after each averaging).
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Whether ElChe has been calibrated.
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Per-rank steps since last averaging.
    pub fn steps_since_avg(&self) -> &[usize] {
        &self.steps_since_avg
    }

    /// Whether CPU averaging is currently in progress (Collecting or Computing).
    pub fn is_cpu_averaging(&self) -> bool {
        !matches!(self.avg_state, CpuAvgState::Idle)
    }

    /// Number of successful averaging events completed.
    pub fn avg_count(&self) -> usize {
        self.avg_count
    }

    /// Number of CPU averaging rounds aborted due to timeout.
    pub fn abort_count(&self) -> usize {
        self.abort_count
    }

    /// Most recent per-rank batch time (ms).
    pub fn last_batch_ms(&self) -> &[f64] {
        &self.last_batch_ms
    }

    /// Most recent CPU averaging time (ms). Zero for NCCL backend.
    pub fn last_avg_ms(&self) -> f64 {
        self.last_avg_ms
    }

    /// Whether all epochs have been aggregated (training is complete).
    pub fn all_epochs_done(&self) -> bool {
        self.last_aggregated_epoch.is_some_and(|e| e + 1 >= self.num_epochs)
    }

    // -----------------------------------------------------------------------
    // Global epoch management
    // -----------------------------------------------------------------------

    /// Compute partition sizes per rank based on policy and throughput.
    pub(super) fn compute_partition_sizes(&self) -> Vec<usize> {
        if let Some(ratios) = &self.partition_ratios {
            return ratio_to_sizes(ratios, self.total_samples);
        }
        match self.policy {
            ApplyPolicy::Sync => {
                equal_sizes(self.world_size, self.total_samples)
            }
            ApplyPolicy::Cadence | ApplyPolicy::Async => {
                if self.el_che.is_calibrated() || self.el_che.has_speed_hint() {
                    throughput_sizes(&self.el_che, self.total_samples)
                } else {
                    equal_sizes(self.world_size, self.total_samples)
                }
            }
        }
    }

    /// Get (or lazily compute) the epoch plans for a given epoch.
    ///
    /// Partition sizes are computed once per epoch and cached, ensuring
    /// consistent offsets across all ranks even when dispatched at different times.
    fn plans_for_epoch(&mut self, epoch: usize) -> Vec<EpochPlan> {
        if let Some(plans) = self.epoch_plan_cache.get(&epoch) {
            return plans.clone();
        }
        let sizes = self.compute_partition_sizes();
        let mut plans = Vec::with_capacity(self.world_size);
        let mut offset = 0;
        for &size in &sizes {
            plans.push(EpochPlan { epoch, partition_offset: offset, partition_size: size });
            offset += size;
        }
        eprintln!("  ddp: epoch {epoch} | partitions {sizes:?}");
        self.epoch_plan_cache.insert(epoch, plans.clone());
        plans
    }

    /// Send StartEpoch to all ranks (used for epoch 0 and Sync/Cadence dispatch).
    ///
    /// In progressive mode, delegates to `start_epoch_progressive`.
    pub fn send_all_plans(&mut self, epoch: usize) {
        if self.progressive {
            self.start_epoch_progressive(epoch);
            return;
        }
        let plans = self.plans_for_epoch(epoch);
        for (rank, plan) in plans.into_iter().enumerate() {
            self.rank_epoch[rank] = epoch;
            self.rank_waiting[rank] = false;
            let _ = self.control_txs[rank].send(ControlMsg::StartEpoch(plan));
        }
    }

    // -----------------------------------------------------------------------
    // Progressive chunk dispatch
    // -----------------------------------------------------------------------

    /// Start a new epoch in progressive mode: create a chunk pool and
    /// dispatch initial chunks to all ranks.
    fn start_epoch_progressive(&mut self, epoch: usize) {
        // Align pool total to batch boundary. Sub-batch remainders can't form
        // a full batch, so they're dropped (standard DataLoader behaviour).
        // Without this, is_epoch_done never fires when total % batch_size != 0.
        let batch_total = (self.total_samples / self.batch_size) * self.batch_size;
        let pool = ChunkPool::new(epoch, batch_total, self.world_size);
        self.chunk_pool = Some(pool);

        let sizes: Vec<usize> = (0..self.world_size)
            .map(|r| self.compute_chunk_batches(r))
            .collect();
        eprintln!(
            "  ddp: epoch {epoch} progressive | initial chunks (batches) {sizes:?}"
        );
        for (rank, &batch_count) in sizes.iter().enumerate() {
            self.dispatch_next_chunk_with_batches(rank, epoch, batch_count);
        }
    }

    /// Dispatch the next chunk to a rank from the active pool.
    ///
    /// Computes chunk size based on calibration state, takes from the pool,
    /// and sends a `StartEpoch` plan. Does nothing if the pool is exhausted.
    fn dispatch_next_chunk(&mut self, rank: usize) {
        let epoch = match &self.chunk_pool {
            Some(pool) => pool.epoch,
            None => return,
        };
        let batches = self.compute_chunk_batches(rank);
        let remaining = self.chunk_pool.as_ref().map_or(0, |p| p.remaining());
        eprintln!(
            "  ddp: chunk -> rank {rank} | {batches} batches | {remaining} samples left"
        );
        self.dispatch_next_chunk_with_batches(rank, epoch, batches);
    }

    fn dispatch_next_chunk_with_batches(&mut self, rank: usize, epoch: usize, batches: usize) {
        let samples = batches * self.batch_size;
        if samples == 0 {
            return;
        }
        let (offset, actual_size) = match self.chunk_pool.as_mut() {
            Some(pool) => match pool.take_chunk(samples, rank) {
                Some(v) => v,
                None => return,
            },
            None => return,
        };
        self.rank_epoch[rank] = epoch;
        self.rank_waiting[rank] = false;
        let _ = self.control_txs[rank].send(ControlMsg::StartEpoch(EpochPlan {
            epoch,
            partition_offset: offset,
            partition_size: actual_size,
        }));
    }

    /// Compute how many batches the next chunk for `rank` should contain.
    fn compute_chunk_batches(&self, rank: usize) -> usize {
        let pool = match &self.chunk_pool {
            Some(pool) => pool,
            None => return 0,
        };
        let remaining_samples = pool.remaining();
        let remaining_batches = remaining_samples / self.batch_size;
        if remaining_batches == 0 {
            return 0;
        }

        if !self.el_che.is_calibrated() && !self.el_che.has_speed_hint() {
            // Probe: small equal chunks for fast calibration.
            // ~10% of total per rank, min 4 batches. Enough for 5-6 averaging
            // events at anchor=10, giving ElChe's EMA time to stabilize.
            let probe = (self.total_samples / (self.world_size * 10 * self.batch_size)).max(4);
            return probe.min(remaining_batches);
        }

        // Calibrated: proportional to throughput
        let counts = self.el_che.batch_counts();
        let total_counts: usize = counts.iter().sum();
        if total_counts == 0 {
            return remaining_batches.min(self.min_chunk_batches);
        }
        let ratio = counts[rank] as f64 / total_counts as f64;
        let mut target = (remaining_batches as f64 * ratio).ceil() as usize;

        // Tail-balance: when remaining work won't fill a full round of
        // chunks, size this chunk to finish when the slowest in-flight
        // rank finishes, preventing fast-GPU idle at epoch end.
        // Works in samples (not batches) to avoid truncation from
        // non-batch-aligned tail chunks.
        if remaining_batches < target * self.world_size {
            let my_ms = self.last_batch_ms[rank];
            if my_ms > 0.0 {
                let ms_per_sample = my_ms / self.batch_size as f64;
                let max_other_ms = (0..self.world_size)
                    .filter(|&r| r != rank)
                    .map(|r| {
                        let in_flight = pool.in_flight(r);
                        let r_ms = if self.last_batch_ms[r] > 0.0 {
                            self.last_batch_ms[r] / self.batch_size as f64
                        } else {
                            ms_per_sample
                        };
                        in_flight as f64 * r_ms
                    })
                    .fold(0.0_f64, f64::max);

                // Only tail-balance when the slowest rank has more than
                // one batch worth of wall-time left — below that the
                // overhead of a smaller chunk isn't worth it.
                if max_other_ms > self.last_batch_ms[rank] {
                    let fill = (max_other_ms / ms_per_sample).ceil() as usize;
                    let fill_batches = fill.div_ceil(self.batch_size);
                    target = target.min(fill_batches);
                }
            }
        }

        target.max(self.min_chunk_batches).min(remaining_batches)
    }

    /// Send StartEpoch to a single rank (Auto per-rank dispatch).
    fn send_rank_plan(&mut self, rank: usize, epoch: usize) {
        let plans = self.plans_for_epoch(epoch);
        if let Some(plan) = plans.into_iter().nth(rank) {
            self.rank_epoch[rank] = epoch;
            self.rank_waiting[rank] = false;
            let _ = self.control_txs[rank].send(ControlMsg::StartEpoch(plan));
        }
    }

    /// Called per-message when a rank's MetricsMsg arrives (epoch done for that rank).
    ///
    /// In Auto mode, immediately dispatches the next epoch if within lookahead.
    fn on_rank_done(&mut self, rank: usize, finished_epoch: usize) {
        if !matches!(self.policy, ApplyPolicy::Async) {
            return;
        }
        let next = finished_epoch + 1;
        if next >= self.num_epochs {
            return;
        }
        let within_lookahead = match self.last_aggregated_epoch {
            // Before any aggregation: allow epoch 0 and 1.
            // Epoch 0 was sent at startup; epoch 1 is the first lookahead.
            None => next <= 1,
            Some(agg) => next.saturating_sub(agg) <= 1,
        };
        if within_lookahead {
            self.send_rank_plan(rank, next);
        } else {
            self.rank_waiting[rank] = true;
        }
    }

    /// Called when all ranks have reported for an epoch (aggregation complete).
    ///
    /// Dispatches next epoch or sends Shutdown based on policy.
    pub(super) fn on_epoch_aggregated(&mut self, epoch: usize) {
        self.last_aggregated_epoch = Some(epoch);
        self.epoch_plan_cache.remove(&epoch);

        // Checkpoint on global epoch boundaries (1-based for file naming).
        if let Some(every) = self.checkpoint_every {
            if every > 0 && (epoch + 1) % every == 0 {
                if let Some(tx) = self.control_txs.first() {
                    let _ = tx.send(ControlMsg::Checkpoint { version: (epoch + 1) as u64 });
                }
            }
        }

        let next_global = epoch + 1;
        if next_global >= self.num_epochs {
            // All epochs done: tell all workers to exit.
            for tx in &self.control_txs {
                let _ = tx.send(ControlMsg::Shutdown);
            }
            return;
        }

        match self.policy {
            ApplyPolicy::Sync | ApplyPolicy::Cadence => {
                self.send_all_plans(next_global);
            }
            ApplyPolicy::Async => {
                if self.progressive {
                    // Progressive handles chunk dispatch; start the next
                    // epoch's pool the same way Sync/Cadence does.
                    self.send_all_plans(next_global);
                } else {
                    // Legacy per-rank dispatch already happened in on_rank_done.
                    // Unblock ranks that were waiting due to lookahead.
                    for rank in 0..self.world_size {
                        if self.rank_waiting[rank] {
                            let next = self.rank_epoch[rank] + 1;
                            if next < self.num_epochs {
                                self.send_rank_plan(rank, next);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Process a single timing message. Shared by [`Self::drain_timing`] and
    /// [`Self::drain_timing_blocking`].
    fn process_timing_msg(&mut self, msg: TimingMsg) {
        match msg {
            TimingMsg::Batch { rank, batch_ms, .. } => {
                self.steps_since_avg[rank] = self.steps_since_avg[rank].saturating_add(1);
                self.wall_ms_accum[rank] += batch_ms;
                self.last_batch_ms[rank] = batch_ms;
            }
            TimingMsg::Exiting { .. } => {
                self.active_count = self.active_count.saturating_sub(1);
            }
        }
    }

    /// Process all pending timing messages (non-blocking drain).
    ///
    /// Updates per-rank step counts and accumulates wall-clock time for ElChe.
    /// When a worker sends [`TimingMsg::Exiting`], decrements `active_count`
    /// so [`should_average`](Self::should_average) stops triggering collectives.
    pub fn drain_timing(&mut self) {
        while let Ok(msg) = self.timing_rx.try_recv() {
            self.process_timing_msg(msg);
        }
    }

    /// Block until a timing message arrives or `timeout` elapses, then drain
    /// all remaining messages non-blocking.
    ///
    /// Returns `false` if the channel is disconnected (all senders dropped),
    /// meaning all workers have exited. The caller should break its loop.
    pub fn drain_timing_blocking(&mut self, timeout: std::time::Duration) -> bool {
        match self.timing_rx.recv_timeout(timeout) {
            Ok(msg) => self.process_timing_msg(msg),
            Err(mpsc::RecvTimeoutError::Timeout) => return true,
            Err(mpsc::RecvTimeoutError::Disconnected) => return false,
        }
        // Drain remaining messages non-blocking
        while let Ok(msg) = self.timing_rx.try_recv() {
            self.process_timing_msg(msg);
        }
        true
    }

    /// Process all pending metrics messages (non-blocking drain).
    ///
    /// Returns collected metrics for logging/monitoring. Also buffers
    /// per-rank messages and aggregates into [`EpochMetrics`](super::EpochMetrics)
    /// when all active ranks have reported for the same epoch.
    pub fn drain_metrics(&mut self) -> Vec<MetricsMsg> {
        let mut msgs = Vec::new();
        while let Ok(msg) = self.metrics_rx.try_recv() {
            if self.progressive {
                // Progressive: accumulate into pool, dispatch next chunk
                if let Some(ref mut pool) = self.chunk_pool {
                    pool.mark_completed(msg.rank, msg.samples_processed);
                }
                self.epoch_buffer.entry(msg.epoch).or_default().push(msg.clone());
                // Dispatch next chunk to this rank (if pool has work)
                self.dispatch_next_chunk(msg.rank);
            } else {
                // Legacy: per-rank Auto dispatch before aggregation
                self.on_rank_done(msg.rank, msg.epoch);
                self.epoch_buffer.entry(msg.epoch).or_default().push(msg.clone());
            }
            msgs.push(msg);
        }
        // Global aggregation + dispatch
        self.try_aggregate_epochs();
        msgs
    }

    /// Check if any buffered epoch has reports from all active ranks.
    /// If so, aggregate, send metrics, and trigger epoch transitions.
    fn try_aggregate_epochs(&mut self) {
        if self.progressive {
            self.try_aggregate_epochs_progressive();
        } else {
            self.try_aggregate_epochs_legacy();
        }
    }

    /// Legacy aggregation: one MetricsMsg per rank per epoch.
    fn try_aggregate_epochs_legacy(&mut self) {
        let expected = self.active_count;
        let mut complete: Vec<usize> = self.epoch_buffer.iter()
            .filter(|(_, msgs)| msgs.len() >= expected)
            .map(|(epoch, _)| *epoch)
            .collect();
        complete.sort_unstable();
        for epoch in complete {
            if let Some(msgs) = self.epoch_buffer.remove(&epoch) {
                if let Some(tx) = &self.epoch_metrics_tx {
                    let metrics = aggregate_epoch_metrics(epoch, &msgs, &self.device_indices);
                    let _ = tx.send(metrics);
                }
                self.on_epoch_aggregated(epoch);
            }
        }
    }

    /// Progressive aggregation: epoch is done when the chunk pool says so.
    fn try_aggregate_epochs_progressive(&mut self) {
        let pool_done = self.chunk_pool.as_ref().is_some_and(|p| p.is_epoch_done());
        if !pool_done {
            return;
        }

        let epoch = self.chunk_pool.as_ref().unwrap().epoch;
        let epoch_ms = self.chunk_pool.as_ref().unwrap().epoch_elapsed_ms();

        // Remove pool before processing (allows on_epoch_aggregated to create the next one)
        self.chunk_pool = None;

        if let Some(msgs) = self.epoch_buffer.remove(&epoch) {
            if let Some(tx) = &self.epoch_metrics_tx {
                let mut metrics = aggregate_epoch_metrics(epoch, &msgs, &self.device_indices);
                // Override epoch_ms with the pool's wall-clock (not per-chunk times)
                metrics.epoch_ms = epoch_ms;
                let _ = tx.send(metrics);
            }
            eprintln!(
                "  ddp: epoch {epoch} progressive complete | {:.0}ms",
                epoch_ms,
            );
            self.on_epoch_aggregated(epoch);
        }
    }

    /// Check if averaging should be triggered based on the current policy.
    pub fn should_average(&self) -> bool {
        // Don't re-trigger while a CPU averaging cycle is in progress.
        if !matches!(self.avg_state, CpuAvgState::Idle) {
            return false;
        }
        // Training complete: workers received Shutdown, skip stale averaging.
        if self.all_epochs_done() {
            return false;
        }
        // Collectives require all ranks. If any worker has exited,
        // skip averaging to prevent NCCL deadlock or channel disconnect.
        if self.active_count < self.world_size {
            return false;
        }
        match self.policy {
            ApplyPolicy::Sync => {
                self.steps_since_avg.iter().all(|&s| s >= 1)
            }
            ApplyPolicy::Cadence => {
                // Wall-time trigger: fire when the slowest rank has
                // accumulated enough compute time since the last sync.
                // This avoids estimation errors from batch-count prediction
                // (EMA lag, dead zone, rounding) that cause the fast GPU
                // to idle at AllReduce boundaries.
                let target = self.el_che.anchor_wall_ms();
                if target > 0.0 {
                    let min_wall = self.wall_ms_accum.iter()
                        .copied()
                        .fold(f64::MAX, f64::min);
                    return min_wall >= target;
                }
                // Fallback (uncalibrated): batch-count trigger with equal
                // counts until first timing measurement arrives.
                let counts = self.el_che.batch_counts();
                self.steps_since_avg.iter().enumerate()
                    .all(|(r, &s)| s >= counts[r])
            }
            ApplyPolicy::Async => {
                // Batch-count trigger: proportional to throughput.
                // Async benefits from overshooting — the divergence between
                // replicas provides implicit regularization (Local SGD).
                // Wall-time matching kills this by constraining the fast GPU.
                let counts = self.el_che.batch_counts();
                self.steps_since_avg.iter().enumerate()
                    .all(|(r, &s)| s >= counts[r])
            }
        }
    }

    /// Trigger parameter averaging based on the configured backend.
    ///
    /// For NCCL: sends `SyncNow` to all workers, then runs the common tail
    /// (ElChe report, version bump, counter reset) synchronously.
    ///
    /// For CPU: sends `RequestParams`, snapshots the current counters, and
    /// enters the `CpuAvgState::Collecting` state. The actual collection
    /// and averaging happen over subsequent [`Self::poll_cpu_averaging`] ticks,
    /// keeping [`Self::check_throttle`] active throughout.
    pub fn trigger_averaging(&mut self) -> Result<()> {
        match self.backend {
            AverageBackend::Nccl => {
                for tx in &self.control_txs {
                    let _ = tx.send(ControlMsg::SyncNow);
                }
                // NCCL: workers block in AllReduce, no new batches happen,
                // so zeroing counters is correct.
                self.finish_averaging_nccl();
            }
            AverageBackend::Cpu => {
                // Snapshot counters at trigger time. Batches that arrive
                // during collection/computing carry into the next period.
                let steps_snapshot = self.steps_since_avg.clone();
                let wall_ms_snapshot = self.wall_ms_accum.clone();

                // Send RequestParams to all workers
                for tx in &self.control_txs {
                    let _ = tx.send(ControlMsg::RequestParams);
                }

                let timeout_secs = self.snapshot_timeout_secs;
                self.avg_state = CpuAvgState::Collecting {
                    snapshots: Vec::with_capacity(self.world_size),
                    received: vec![false; self.world_size],
                    deadline: Instant::now()
                        + std::time::Duration::from_secs(timeout_secs),
                    start: Instant::now(),
                    steps_snapshot,
                    wall_ms_snapshot,
                };
                // Return immediately; poll_cpu_averaging drives the rest.
            }
        }
        Ok(())
    }

    /// Common tail for NCCL averaging: report to ElChe, bump version, zero counters.
    ///
    /// NCCL workers block in AllReduce so no new batches arrive during the
    /// collective; zeroing is correct.
    fn finish_averaging_nccl(&mut self) {
        if self.wall_ms_accum.iter().any(|&ms| ms > 0.0) {
            self.el_che.report_timing(&self.wall_ms_accum, &self.steps_since_avg, self.last_avg_ms);
            self.last_avg_ms = 0.0;
            if !self.calibrated && self.el_che.is_calibrated() {
                self.calibrated = true;
            }
        }

        self.version += 1;
        self.avg_count += 1;

        eprintln!(
            "  ddp: NCCL averaging #{} complete (v{})",
            self.avg_count, self.version
        );

        for s in &mut self.steps_since_avg {
            *s = 0;
        }
        for a in &mut self.wall_ms_accum {
            *a = 0.0;
        }
        for t in &mut self.throttled {
            *t = false;
        }
    }

    /// Common tail for CPU averaging: report snapshot counters to ElChe,
    /// apply divergence correction, subtract snapshots from current counters
    /// (preserve during-averaging batches).
    pub(super) fn finish_averaging_cpu(
        &mut self,
        avg_ms: f64,
        steps_snapshot: &[usize],
        wall_ms_snapshot: &[f64],
        divergence: Option<f64>,
    ) {
        self.last_avg_ms = avg_ms;

        // Report the snapshot values to ElChe (accurate for the period
        // that triggered averaging, not inflated by during-averaging batches).
        if wall_ms_snapshot.iter().any(|&ms| ms > 0.0) {
            self.el_che.report_timing(wall_ms_snapshot, steps_snapshot, self.last_avg_ms);
            self.last_avg_ms = 0.0;
            if !self.calibrated && self.el_che.is_calibrated() {
                self.calibrated = true;
            }
        }

        // Divergence correction: if replicas drifted apart, nudge ElChe's
        // anchor down (tighter sync). One-directional pressure only; ElChe's
        // overhead auto-tune handles loosening.
        if let Some(div) = divergence {
            if div > self.divergence_threshold {
                let old_anchor = self.el_che.anchor();
                self.el_che.nudge_anchor_down(0.5);
                eprintln!(
                    "  ddp: divergence {div:.4} > {:.4}, anchor {} -> {}",
                    self.divergence_threshold, old_anchor, self.el_che.anchor()
                );
            }
        }

        self.version += 1;
        self.avg_count += 1;

        eprintln!(
            "  ddp: CPU averaging #{} complete (v{}, {:.1}ms)",
            self.avg_count, self.version, avg_ms
        );

        // Subtract snapshot from current counters. Residual = batches
        // that happened during the averaging window, carried forward.
        for (i, s) in self.steps_since_avg.iter_mut().enumerate() {
            *s = s.saturating_sub(steps_snapshot[i]);
        }
        for (i, a) in self.wall_ms_accum.iter_mut().enumerate() {
            *a = (*a - wall_ms_snapshot[i]).max(0.0);
        }
        for t in &mut self.throttled {
            *t = false;
        }
    }

    /// Abort an in-progress CPU averaging cycle and return to Idle.
    ///
    /// Drains stale snapshots from `param_rx` so the next cycle starts clean.
    fn abort_cpu_averaging(&mut self) {
        self.avg_state = CpuAvgState::Idle;
        // Drain any in-flight snapshots from the aborted round.
        while self.param_rx.try_recv().is_ok() {}
    }

    /// Cleanly shut down any in-progress CPU averaging before the coordinator exits.
    ///
    /// If in Computing state, joins the background thread (waits for it to finish)
    /// so no detached thread holds GPU resources when the coordinator returns.
    /// If in Collecting state, drains stale snapshots.
    pub fn drain_avg_state(&mut self) {
        let state = std::mem::replace(&mut self.avg_state, CpuAvgState::Idle);
        match state {
            CpuAvgState::Idle => {}
            CpuAvgState::Collecting { snapshots, .. } => {
                eprintln!(
                    "  ddp: discarding in-progress CPU averaging \
                     (Collecting, {}/{} snapshots received)",
                    snapshots.len(), self.world_size
                );
                while self.param_rx.try_recv().is_ok() {}
            }
            CpuAvgState::Computing { handle, .. } => {
                // Join the compute thread so no detached thread holds GPU resources.
                let result = handle.join();
                let status = match &result {
                    Ok(Ok(_)) => "completed result discarded",
                    Ok(Err(e)) => {
                        eprintln!("  ddp: CPU averaging compute error: {e}");
                        "errored"
                    }
                    Err(_) => "panicked",
                };
                eprintln!(
                    "  ddp: discarding in-progress CPU averaging (Computing, {status})"
                );
                // Drain any snapshots it might have sent before we discard the result.
                while self.param_rx.try_recv().is_ok() {}
            }
        }
    }

    /// Drive the CPU averaging state machine one tick.
    ///
    /// Called every coordinator loop iteration. Handles three states:
    ///
    /// - **Idle**: no-op.
    /// - **Collecting**: `try_recv` on `param_rx` for pending snapshots.
    ///   Transitions to Computing when all ranks have responded, or soft-aborts
    ///   on timeout (drains stale snapshots, returns to Idle, logs warning).
    /// - **Computing**: checks if the background thread has finished. When done,
    ///   sends `Update` to all workers and runs `finish_averaging_cpu`.
    ///
    /// Returns `Ok(())` on normal progress (including soft abort).
    /// Returns `Err` only on unrecoverable errors (compute thread panic).
    pub fn poll_cpu_averaging(&mut self) -> Result<()> {
        // Take ownership of the state to avoid borrow issues.
        let state = std::mem::replace(&mut self.avg_state, CpuAvgState::Idle);

        match state {
            CpuAvgState::Idle => {
                self.avg_state = CpuAvgState::Idle;
            }
            CpuAvgState::Collecting {
                mut snapshots,
                mut received,
                deadline,
                start,
                steps_snapshot,
                wall_ms_snapshot,
            } => {
                // Drain all available snapshots (non-blocking).
                while let Ok(snap) = self.param_rx.try_recv() {
                    if snap.rank < self.world_size && !received[snap.rank] {
                        received[snap.rank] = true;
                        snapshots.push(snap);
                    }
                    // Ignore duplicates or out-of-range ranks.
                }

                if snapshots.len() >= self.world_size {
                    // All snapshots collected. Spawn compute thread.
                    let version = self.version + 1;

                    let handle = std::thread::Builder::new()
                        .name("cpu-avg-compute".into())
                        .spawn(move || {
                            Self::compute_average_and_divergence(
                                snapshots, version,
                            )
                        })
                        .map_err(|e| TensorError::new(
                            &format!("failed to spawn CPU averaging thread: {e}")
                        ))?;

                    self.avg_state = CpuAvgState::Computing {
                        handle,
                        start,
                        steps_snapshot,
                        wall_ms_snapshot,
                    };
                } else if Instant::now() >= deadline {
                    // Timeout: soft abort.
                    let missing: Vec<usize> = received.iter().enumerate()
                        .filter(|(_, got)| !**got)
                        .map(|(r, _)| r)
                        .collect();
                    self.abort_count += 1;
                    eprintln!(
                        "  ddp: CPU averaging timeout, missing ranks: {missing:?} \
                         (abort #{}, will retry)", self.abort_count
                    );
                    self.abort_cpu_averaging();
                } else {
                    // Still waiting. Put state back.
                    self.avg_state = CpuAvgState::Collecting {
                        snapshots,
                        received,
                        deadline,
                        start,
                        steps_snapshot,
                        wall_ms_snapshot,
                    };
                }
            }
            CpuAvgState::Computing {
                handle,
                start,
                steps_snapshot,
                wall_ms_snapshot,
            } => {
                if handle.is_finished() {
                    let result = handle.join()
                        .map_err(|_| TensorError::new(
                            "CPU averaging compute thread panicked"
                        ))??;

                    let avg_ms = start.elapsed().as_secs_f64() * 1000.0;

                    // Send averaged params to all workers.
                    for (rank, tx) in self.control_txs.iter().enumerate() {
                        if tx.send(ControlMsg::Update(result.averaged.clone())).is_err() {
                            eprintln!(
                                "  ddp: failed to deliver Update to rank {rank} \
                                 (worker channel dead)"
                            );
                        }
                    }

                    self.finish_averaging_cpu(
                        avg_ms,
                        &steps_snapshot,
                        &wall_ms_snapshot,
                        result.divergence,
                    );
                    // avg_state is already Idle from the mem::replace.
                } else {
                    // Still computing. Put state back.
                    self.avg_state = CpuAvgState::Computing {
                        handle,
                        start,
                        steps_snapshot,
                        wall_ms_snapshot,
                    };
                }
            }
        }

        Ok(())
    }

    /// Compute weighted average of parameter snapshots.
    ///
    /// Weight each rank's contribution by its `batch_count` (number of batches
    /// since last averaging). This ensures faster GPUs contribute more.
    /// Buffers (e.g. BatchNorm running stats) are averaged with equal weight.
    pub(super) fn average_params(snapshots: &[ParamSnapshot], version: u64) -> Result<AveragedParams> {
        if snapshots.is_empty() {
            return Err(TensorError::new("average_params: no snapshots"));
        }

        let n_params = snapshots[0].params.len();
        let n_buffers = snapshots[0].buffers.len();

        // Validate all snapshots have the same structure.
        for (i, snap) in snapshots.iter().enumerate() {
            if snap.params.len() != n_params {
                return Err(TensorError::new(&format!(
                    "average_params: rank {} has {} params, expected {}",
                    i, snap.params.len(), n_params
                )));
            }
            if snap.buffers.len() != n_buffers {
                return Err(TensorError::new(&format!(
                    "average_params: rank {} has {} buffers, expected {}",
                    i, snap.buffers.len(), n_buffers
                )));
            }
        }

        let total_batches: usize = snapshots.iter().map(|s| s.batch_count.max(1)).sum();

        // Weighted average of parameters on CPU (snapshots may be on different devices)
        let mut avg_params = Vec::with_capacity(n_params);
        for pi in 0..n_params {
            let first = snapshots[0].params[pi].to_device(Device::CPU)?;
            let acc = Tensor::zeros_like(&first)?;
            for snap in snapshots {
                let weight = snap.batch_count.max(1) as f64 / total_batches as f64;
                let cpu_param = snap.params[pi].to_device(Device::CPU)?;
                let scaled = cpu_param.mul_scalar(weight)?;
                acc.add_(&scaled)?;
            }
            avg_params.push(acc);
        }

        // Average buffers (equal weight, e.g. BatchNorm running mean/var).
        let inv_n = 1.0 / snapshots.len() as f64;
        let mut avg_buffers = Vec::with_capacity(n_buffers);
        for bi in 0..n_buffers {
            let first = snapshots[0].buffers[bi].to_device(Device::CPU)?;
            let acc = Tensor::zeros_like(&first)?;
            for snap in snapshots {
                let cpu_buf = snap.buffers[bi].to_device(Device::CPU)?;
                let scaled = cpu_buf.mul_scalar(inv_n)?;
                acc.add_(&scaled)?;
            }
            avg_buffers.push(acc);
        }

        Ok(AveragedParams {
            params: avg_params,
            buffers: avg_buffers,
            version,
        })
    }

    /// Compute weighted average and divergence on a background thread.
    ///
    /// Static method: captures only the data it needs, no `&self` borrow.
    /// Returns `CpuAvgResult` with the averaged params and the relative
    /// divergence across replicas (for anchor correction by ElChe).
    fn compute_average_and_divergence(
        snapshots: Vec<ParamSnapshot>,
        version: u64,
    ) -> Result<CpuAvgResult> {
        let averaged = Self::average_params(&snapshots, version)?;

        // Compute per-rank parameter L2 norm for divergence monitoring.
        let mut norms = Vec::with_capacity(snapshots.len());
        for snap in &snapshots {
            let mut total_norm_sq = 0.0f64;
            for p in &snap.params {
                let n: f64 = p.norm()?.item()?;
                total_norm_sq += n * n;
            }
            norms.push(total_norm_sq.sqrt());
        }

        let mean_norm: f64 = norms.iter().sum::<f64>() / norms.len() as f64;
        let divergence = if mean_norm < 1e-10 {
            None // all zeros, no meaningful divergence
        } else {
            Some(norms.iter()
                .map(|n| (n - mean_norm).abs())
                .sum::<f64>() / mean_norm)
        };

        Ok(CpuAvgResult { averaged, divergence })
    }

    /// Throttle workers that have run too far ahead of the slowest rank.
    ///
    /// Sends [`ControlMsg::Throttle`] to any worker whose `steps_since_avg`
    /// exceeds the slowest rank's by more than `max_batch_diff`. The worker
    /// blocks until the next real command (averaging or shutdown).
    ///
    /// Tracks which ranks are already throttled to avoid sending duplicate
    /// Throttle messages (which would nest blocking loops in the worker).
    pub fn check_throttle(&mut self) {
        let max_diff = match self.el_che.max_batch_diff() {
            Some(d) => d,
            None => return,
        };

        if self.active_count < self.world_size {
            return; // some worker exited, don't throttle
        }

        let min_steps = self.steps_since_avg.iter().copied().min().unwrap_or(0);

        for (rank, &steps) in self.steps_since_avg.iter().enumerate() {
            let should_throttle = steps > min_steps + max_diff;
            if should_throttle && !self.throttled[rank] {
                let _ = self.control_txs[rank].send(ControlMsg::Throttle);
                self.throttled[rank] = true;
            }
        }
    }

    /// Run one tick of the coordinator loop.
    ///
    /// Drains timing/metrics, throttles fast workers, checks averaging.
    /// Returns collected metrics (if any) for external logging.
    pub fn tick(&mut self) -> Result<Vec<MetricsMsg>> {
        self.drain_timing();
        self.check_throttle();
        self.poll_cpu_averaging()?;
        let metrics = self.drain_metrics();

        if self.should_average() {
            // Final drain to catch last-second Exiting messages before
            // sending SyncNow (prevents AllReduce with a dead worker).
            self.drain_timing();
            if self.should_average() {
                self.trigger_averaging()?;
            }
        }

        Ok(metrics)
    }

    /// Collect final parameter snapshots from all workers after the main loop exits.
    ///
    /// Blocks on each dedicated `final_param_rx` channel with a timeout.
    /// Returns a [`TrainedState`] averaged from whatever snapshots arrived
    /// (partial failure: survivors' params are returned). Returns `None` if
    /// zero snapshots were collected.
    pub fn collect_final_state(&self) -> Option<TrainedState> {
        let timeout = std::time::Duration::from_secs(10);
        let mut snapshots = Vec::new();
        for (rank, rx) in self.final_param_rxs.iter().enumerate() {
            match rx.recv_timeout(timeout) {
                Ok(snap) => snapshots.push(snap),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    eprintln!("  ddp: timeout waiting for final snapshot from rank {rank}");
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    eprintln!("  ddp: rank {rank} channel disconnected (worker errored)");
                }
            }
        }
        if snapshots.is_empty() {
            return None;
        }
        // Average the final snapshots (reuse the existing averaging logic)
        match Self::average_params(&snapshots, self.version) {
            Ok(averaged) => Some(TrainedState {
                params: averaged.params,
                buffers: averaged.buffers,
            }),
            Err(_) => {
                // Fallback: return the first snapshot's tensors on CPU
                let snap = &snapshots[0];
                let params = snap.params.iter()
                    .filter_map(|t| t.to_device(Device::CPU).ok())
                    .collect();
                let buffers = snap.buffers.iter()
                    .filter_map(|t| t.to_device(Device::CPU).ok())
                    .collect();
                Some(TrainedState { params, buffers })
            }
        }
    }

    /// Send Shutdown to all workers so they can exit their
    /// `drain_until_shutdown` loop.
    pub fn shutdown_workers(&self) {
        for tx in &self.control_txs {
            let _ = tx.send(ControlMsg::Shutdown);
        }
    }
}

// ---------------------------------------------------------------------------
// Partition sizing helpers
// ---------------------------------------------------------------------------

/// Equal partition sizes with remainder distributed to the first ranks.
fn equal_sizes(world_size: usize, total: usize) -> Vec<usize> {
    let base = total / world_size;
    let remainder = total % world_size;
    (0..world_size)
        .map(|r| base + if r < remainder { 1 } else { 0 })
        .collect()
}

/// Throughput-proportional partition sizes from ElChe ms_per_batch.
///
/// Faster ranks (lower ms/batch) get more samples. Remainder distributed
/// to the fastest ranks.
fn throughput_sizes(el_che: &crate::distributed::ddp::ElChe, total: usize) -> Vec<usize> {
    let ms = el_che.ms_per_batch();
    // Inverse of ms_per_batch = throughput (batches/ms). Guard against zero.
    let throughputs: Vec<f64> = ms.iter().map(|&m| 1.0 / m.max(0.001)).collect();
    let total_tp: f64 = throughputs.iter().sum();
    if total_tp <= 0.0 {
        return equal_sizes(ms.len(), total);
    }
    let mut sizes: Vec<usize> = throughputs.iter()
        .map(|t| ((t / total_tp) * total as f64).floor() as usize)
        .collect();
    // Distribute remainder to fastest ranks (highest throughput first).
    let assigned: usize = sizes.iter().sum();
    let mut remaining = total.saturating_sub(assigned);
    if remaining > 0 {
        // Sort rank indices by throughput descending.
        let mut rank_order: Vec<usize> = (0..ms.len()).collect();
        rank_order.sort_by(|&a, &b| throughputs[b].partial_cmp(&throughputs[a]).unwrap_or(std::cmp::Ordering::Equal));
        for &rank in &rank_order {
            if remaining == 0 { break; }
            sizes[rank] += 1;
            remaining -= 1;
        }
    }
    sizes
}

/// Convert user-specified ratios to absolute partition sizes.
///
/// Ratios are normalized to sum to 1.0. Remainder distributed to the
/// ranks with the largest ratios.
fn ratio_to_sizes(ratios: &[f64], total: usize) -> Vec<usize> {
    let sum: f64 = ratios.iter().sum();
    let norm: Vec<f64> = if sum > 0.0 {
        ratios.iter().map(|r| r / sum).collect()
    } else {
        vec![1.0 / ratios.len() as f64; ratios.len()]
    };
    let mut sizes: Vec<usize> = norm.iter()
        .map(|r| (r * total as f64).floor() as usize)
        .collect();
    let assigned: usize = sizes.iter().sum();
    let mut remaining = total.saturating_sub(assigned);
    if remaining > 0 {
        let mut rank_order: Vec<usize> = (0..ratios.len()).collect();
        rank_order.sort_by(|&a, &b| norm[b].partial_cmp(&norm[a]).unwrap_or(std::cmp::Ordering::Equal));
        for &rank in &rank_order {
            if remaining == 0 { break; }
            sizes[rank] += 1;
            remaining -= 1;
        }
    }
    sizes
}

/// Aggregate per-rank `MetricsMsg` into a single `EpochMetrics`.
///
/// Loss and scalars are averaged weighted by batch count (proportional
/// to each rank's contribution). Epoch time is the max across ranks.
///
/// In progressive dispatch mode, each rank sends one [`MetricsMsg`] per
/// chunk (not per epoch), so there may be many more messages than ranks.
/// This function aggregates by rank first so the output always has
/// exactly `world_size` entries per vector.
pub(super) fn aggregate_epoch_metrics(epoch: usize, msgs: &[MetricsMsg], device_indices: &[u8]) -> super::EpochMetrics {
    let world_size = device_indices.len();

    // --- Step 1: Aggregate per-chunk messages by rank ---
    let mut rank_batches: Vec<usize> = vec![0; world_size];
    let mut rank_samples: Vec<usize> = vec![0; world_size];
    let mut rank_loss_sum: Vec<f64> = vec![0.0; world_size];
    let mut rank_time_ms: Vec<f64> = vec![0.0; world_size];
    // Per-rank scalar accumulators: (sum, count) per key
    let mut rank_scalars: Vec<std::collections::HashMap<String, (f64, usize)>> =
        (0..world_size).map(|_| std::collections::HashMap::new()).collect();

    for m in msgs {
        let r = m.rank.min(world_size - 1);
        rank_batches[r] += m.batches_processed;
        rank_samples[r] += m.samples_processed;
        rank_loss_sum[r] += m.avg_loss * m.batches_processed as f64;
        // Max time across chunks (sequential within a rank)
        rank_time_ms[r] = rank_time_ms[r].max(m.epoch_ms);
        for (k, (sum, count)) in &m.scalars {
            let entry = rank_scalars[r].entry(k.clone()).or_insert((0.0, 0));
            entry.0 += sum;
            entry.1 += count;
        }
    }

    // --- Step 2: Compute aggregated metrics ---
    let total_batches: usize = rank_batches.iter().sum();

    // Batch-weighted average loss
    let avg_loss = if total_batches > 0 {
        rank_loss_sum.iter().sum::<f64>() / total_batches as f64
    } else {
        0.0
    };

    // Max epoch_ms across ranks
    let epoch_ms = rank_time_ms.iter().copied().fold(0.0_f64, f64::max);

    // Per-rank scalar means (each rank's sum/count)
    let per_rank: Vec<std::collections::HashMap<String, f64>> = rank_scalars
        .iter()
        .map(|scalars| {
            scalars
                .iter()
                .map(|(k, (sum, count))| {
                    (k.clone(), if *count > 0 { sum / *count as f64 } else { 0.0 })
                })
                .collect()
        })
        .collect();

    // Weighted-average scalars across ranks
    let mut scalars: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut weights: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for (r, rank_sc) in rank_scalars.iter().enumerate() {
        let w = rank_batches[r] as f64;
        for (k, (sum, count)) in rank_sc {
            if *count > 0 {
                let mean = sum / *count as f64;
                *scalars.entry(k.clone()).or_default() += mean * w;
                *weights.entry(k.clone()).or_default() += w;
            }
        }
    }
    for (k, v) in &mut scalars {
        if let Some(w) = weights.get(k) {
            if *w > 0.0 {
                *v /= *w;
            }
        }
    }

    // Per-rank throughput (samples/ms) and batch share
    let total_samples: usize = rank_samples.iter().sum();
    let per_rank_throughput: Vec<f64> = (0..world_size).map(|r| {
        if rank_time_ms[r] > 0.0 { rank_samples[r] as f64 / rank_time_ms[r] } else { 0.0 }
    }).collect();
    let per_rank_batch_share: Vec<f64> = (0..world_size).map(|r| {
        if total_samples > 0 { rank_samples[r] as f64 / total_samples as f64 } else { 0.0 }
    }).collect();

    super::EpochMetrics {
        epoch, scalars, per_rank, avg_loss, epoch_ms,
        per_rank_throughput, per_rank_batch_share, device_indices: device_indices.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_pool_basic() {
        let mut pool = ChunkPool::new(0, 1000, 2);
        assert_eq!(pool.remaining(), 1000);
        assert!(!pool.is_epoch_done());

        // Take a chunk for rank 0
        let (off, size) = pool.take_chunk(300, 0).unwrap();
        assert_eq!(off, 0);
        assert_eq!(size, 300);
        assert_eq!(pool.remaining(), 700);

        // Take a chunk for rank 1
        let (off, size) = pool.take_chunk(200, 1).unwrap();
        assert_eq!(off, 300);
        assert_eq!(size, 200);
        assert_eq!(pool.remaining(), 500);

        // Not done yet (nothing completed)
        assert!(!pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_exhaustion() {
        let mut pool = ChunkPool::new(0, 100, 2);

        // Take more than available: clamped
        let (off, size) = pool.take_chunk(80, 0).unwrap();
        assert_eq!((off, size), (0, 80));

        let (off, size) = pool.take_chunk(50, 1).unwrap();
        assert_eq!((off, size), (80, 20)); // only 20 left

        // Pool exhausted
        assert!(pool.take_chunk(10, 0).is_none());
        assert_eq!(pool.remaining(), 0);
    }

    #[test]
    fn chunk_pool_is_epoch_done() {
        let mut pool = ChunkPool::new(0, 100, 2);

        pool.take_chunk(60, 0).unwrap();
        pool.take_chunk(40, 1).unwrap();
        assert!(pool.take_chunk(1, 0).is_none()); // exhausted

        // All dispatched but nothing completed
        assert!(!pool.is_epoch_done());

        // Rank 0 completes
        pool.mark_completed(0, 60);
        assert!(!pool.is_epoch_done()); // rank 1 still pending

        // Rank 1 completes
        pool.mark_completed(1, 40);
        assert!(pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_incremental_completion() {
        let mut pool = ChunkPool::new(0, 200, 2);

        // Two chunks for rank 0
        pool.take_chunk(50, 0).unwrap();
        pool.take_chunk(50, 1).unwrap();
        pool.take_chunk(60, 0).unwrap();
        pool.take_chunk(40, 1).unwrap();
        assert_eq!(pool.remaining(), 0);

        // Complete in stages
        pool.mark_completed(0, 50); // first chunk
        pool.mark_completed(1, 50);
        assert!(!pool.is_epoch_done()); // rank 0 dispatched 110, only 50 done

        pool.mark_completed(0, 60); // second chunk
        pool.mark_completed(1, 40);
        assert!(pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_no_overlap() {
        let mut pool = ChunkPool::new(0, 500, 3);
        let mut all_offsets = Vec::new();

        while pool.remaining() > 0 {
            for rank in 0..3 {
                if let Some((off, size)) = pool.take_chunk(60, rank) {
                    // Verify no overlap with previous chunks
                    for &(prev_off, prev_size) in &all_offsets {
                        let prev_end: usize = prev_off + prev_size;
                        let this_end = off + size;
                        assert!(off >= prev_end || this_end <= prev_off,
                            "overlap: ({off}, {size}) vs ({prev_off}, {prev_size})");
                    }
                    all_offsets.push((off, size));
                }
            }
        }

        // Total coverage = total_samples
        let total: usize = all_offsets.iter().map(|(_, s)| s).sum();
        assert_eq!(total, 500);
    }

    #[test]
    fn chunk_pool_epoch_elapsed() {
        let pool = ChunkPool::new(0, 100, 2);
        // Just verify it returns something reasonable (not zero, not huge)
        std::thread::sleep(std::time::Duration::from_millis(5));
        let ms = pool.epoch_elapsed_ms();
        assert!((4.0..1000.0).contains(&ms), "elapsed {ms}ms");
    }

    // -----------------------------------------------------------------------
    // ChunkPool edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_pool_zero_total_samples() {
        let mut pool = ChunkPool::new(0, 0, 2);
        assert_eq!(pool.remaining(), 0);
        assert!(pool.take_chunk(10, 0).is_none());
        // All dispatched (0) == all completed (0), so epoch is trivially done.
        assert!(pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_single_rank() {
        let mut pool = ChunkPool::new(0, 50, 1);
        let (off, size) = pool.take_chunk(50, 0).unwrap();
        assert_eq!((off, size), (0, 50));
        assert_eq!(pool.remaining(), 0);
        assert!(!pool.is_epoch_done());
        pool.mark_completed(0, 50);
        assert!(pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_take_chunk_size_zero() {
        let mut pool = ChunkPool::new(0, 100, 2);
        // take_chunk with size=0 should return (cursor, 0) since min(0, remaining)=0
        // Actually, 0.min(100) = 0, cursor doesn't move, dispatched stays 0.
        // But cursor == 0 < total_samples == 100, so it enters the body,
        // actual = 0.min(100-0) = 0. Returns Some((0, 0)).
        let result = pool.take_chunk(0, 0);
        assert_eq!(result, Some((0, 0)));
        // Cursor should not have advanced.
        assert_eq!(pool.remaining(), 100);
    }

    #[test]
    fn chunk_pool_in_flight_tracking() {
        let mut pool = ChunkPool::new(0, 100, 2);
        pool.take_chunk(40, 0).unwrap();
        pool.take_chunk(30, 1).unwrap();
        assert_eq!(pool.in_flight(0), 40);
        assert_eq!(pool.in_flight(1), 30);

        pool.mark_completed(0, 20);
        assert_eq!(pool.in_flight(0), 20);
        assert_eq!(pool.in_flight(1), 30);

        pool.mark_completed(0, 20);
        assert_eq!(pool.in_flight(0), 0);
    }

    // -----------------------------------------------------------------------
    // Helper: create a Coordinator without CUDA or real GPU threads.
    // -----------------------------------------------------------------------

    fn make_test_coordinator(
        world_size: usize,
        policy: ApplyPolicy,
        total_samples: usize,
    ) -> Coordinator {
        let (_timing_tx, timing_rx) = mpsc::channel();
        let (_metrics_tx, metrics_rx) = mpsc::channel();
        let (_param_tx, param_rx) = mpsc::channel();
        let mut final_rxs = Vec::new();
        let mut control_txs = Vec::new();
        for _ in 0..world_size {
            let (_ftx, frx) = mpsc::channel::<ParamSnapshot>();
            final_rxs.push(frx);
            let (ctx, _crx) = mpsc::channel::<ControlMsg>();
            control_txs.push(ctx);
        }
        let el_che = crate::distributed::ddp::ElChe::new(world_size, 10);
        Coordinator::builder(
            timing_rx, metrics_rx, param_rx, final_rxs, control_txs,
            policy, AverageBackend::Nccl, world_size, total_samples, el_che,
        )
        .num_epochs(10)
        .build()
    }

    /// Variant that returns the timing sender so tests can inject TimingMsg.
    fn make_test_coordinator_with_channels(
        world_size: usize,
        policy: ApplyPolicy,
        total_samples: usize,
    ) -> (
        Coordinator,
        mpsc::Sender<TimingMsg>,
        Vec<mpsc::Receiver<ControlMsg>>,
    ) {
        let (timing_tx, timing_rx) = mpsc::channel();
        let (_metrics_tx, metrics_rx) = mpsc::channel();
        let (_param_tx, param_rx) = mpsc::channel();
        let mut final_rxs = Vec::new();
        let mut control_txs = Vec::new();
        let mut control_rxs = Vec::new();
        for _ in 0..world_size {
            let (_ftx, frx) = mpsc::channel::<ParamSnapshot>();
            final_rxs.push(frx);
            let (ctx, crx) = mpsc::channel::<ControlMsg>();
            control_txs.push(ctx);
            control_rxs.push(crx);
        }
        let el_che = crate::distributed::ddp::ElChe::new(world_size, 10);
        let coord = Coordinator::builder(
            timing_rx, metrics_rx, param_rx, final_rxs, control_txs,
            policy, AverageBackend::Nccl, world_size, total_samples, el_che,
        )
        .num_epochs(10)
        .build();
        (coord, timing_tx, control_rxs)
    }

    // -----------------------------------------------------------------------
    // compute_partition_sizes
    // -----------------------------------------------------------------------

    #[test]
    fn partition_sync_equal_sizes() {
        let coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        let sizes = coord.compute_partition_sizes();
        assert_eq!(sizes, vec![500, 500]);
    }

    #[test]
    fn partition_sync_uneven_remainder() {
        let coord = make_test_coordinator(3, ApplyPolicy::Sync, 100);
        let sizes = coord.compute_partition_sizes();
        // 100 / 3 = 33 remainder 1. First rank gets the extra.
        assert_eq!(sizes, vec![34, 33, 33]);
        assert_eq!(sizes.iter().sum::<usize>(), 100);
    }

    #[test]
    fn partition_cadence_uncalibrated_falls_back_to_equal() {
        // Before calibration, Cadence mode uses equal sizes.
        let coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        assert!(!coord.el_che.is_calibrated());
        assert!(!coord.el_che.has_speed_hint());
        let sizes = coord.compute_partition_sizes();
        assert_eq!(sizes, vec![500, 500]);
    }

    #[test]
    fn partition_cadence_calibrated_proportional() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        // Simulate calibration: rank 0 is 2x faster (5ms/batch vs 10ms/batch).
        // report_timing needs wall_ms and actual_batches.
        coord.el_che.report_timing(&[50.0, 100.0], &[10, 10], 5.0);
        assert!(coord.el_che.is_calibrated());
        let sizes = coord.compute_partition_sizes();
        // Rank 0 has throughput 1/5=0.2, rank 1 has throughput 1/10=0.1.
        // Proportions: 0.2/0.3 = 0.667, 0.1/0.3 = 0.333.
        // ~667 and ~333.
        assert!(sizes[0] > sizes[1], "fast rank should get more: {:?}", sizes);
        assert_eq!(sizes.iter().sum::<usize>(), 1000);
    }

    #[test]
    fn partition_with_user_ratios() {
        let (_timing_tx, timing_rx) = mpsc::channel();
        let (_metrics_tx, metrics_rx) = mpsc::channel();
        let (_param_tx, param_rx) = mpsc::channel();
        let mut final_rxs = Vec::new();
        let mut control_txs = Vec::new();
        for _ in 0..3 {
            let (_ftx, frx) = mpsc::channel::<ParamSnapshot>();
            final_rxs.push(frx);
            let (ctx, _crx) = mpsc::channel::<ControlMsg>();
            control_txs.push(ctx);
        }
        let el_che = crate::distributed::ddp::ElChe::new(3, 10);
        let coord = Coordinator::builder(
            timing_rx, metrics_rx, param_rx, final_rxs, control_txs,
            ApplyPolicy::Cadence, AverageBackend::Nccl, 3, 900, el_che,
        )
        .partition_ratios(Some(vec![3.0, 2.0, 1.0]))
        .num_epochs(5)
        .build();

        let sizes = coord.compute_partition_sizes();
        // Ratios 3:2:1 normalized -> 0.5, 0.333, 0.167. On 900 samples:
        // floor: 450, 300, 150 = 900. No remainder.
        assert_eq!(sizes, vec![450, 300, 150]);
        assert_eq!(sizes.iter().sum::<usize>(), 900);
    }

    // -----------------------------------------------------------------------
    // Free function edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn equal_sizes_single_worker() {
        let sizes = equal_sizes(1, 100);
        assert_eq!(sizes, vec![100]);
    }

    #[test]
    fn ratio_to_sizes_zero_sum() {
        // All zero ratios: falls back to equal.
        let sizes = ratio_to_sizes(&[0.0, 0.0], 100);
        assert_eq!(sizes, vec![50, 50]);
    }

    // -----------------------------------------------------------------------
    // should_average
    // -----------------------------------------------------------------------

    #[test]
    fn should_average_sync_triggers_when_all_have_one_step() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        // Initially no steps: should not average.
        assert!(!coord.should_average());

        // Only rank 0 has a step.
        coord.steps_since_avg[0] = 1;
        assert!(!coord.should_average());

        // Both ranks have a step: should trigger.
        coord.steps_since_avg[1] = 1;
        assert!(coord.should_average());
    }

    #[test]
    fn should_average_cadence_wall_time_trigger() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        // Calibrate ElChe so anchor_wall_ms() returns > 0.
        coord.el_che.report_timing(&[100.0, 200.0], &[10, 10], 5.0);
        assert!(coord.el_che.is_calibrated());
        let target = coord.el_che.anchor_wall_ms();
        assert!(target > 0.0, "anchor_wall_ms should be positive after calibration");

        // Not enough wall time accumulated.
        coord.wall_ms_accum[0] = target * 0.5;
        coord.wall_ms_accum[1] = target * 0.5;
        assert!(!coord.should_average());

        // Both ranks reach the target.
        coord.wall_ms_accum[0] = target;
        coord.wall_ms_accum[1] = target;
        assert!(coord.should_average());
    }

    #[test]
    fn should_average_cadence_min_wall_governs() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        coord.el_che.report_timing(&[100.0, 200.0], &[10, 10], 5.0);
        let target = coord.el_che.anchor_wall_ms();

        // Rank 0 has enough, rank 1 does not: min < target.
        coord.wall_ms_accum[0] = target * 2.0;
        coord.wall_ms_accum[1] = target * 0.3;
        assert!(!coord.should_average());
    }

    #[test]
    fn should_average_async_batch_count_trigger() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Async, 1000);
        // Uncalibrated ElChe: batch_counts are [10, 10] (equal, anchor=10).
        let counts = coord.el_che.batch_counts().to_vec();
        assert_eq!(counts, vec![10, 10]);

        // Rank 0 at 10, rank 1 at 9: not enough.
        coord.steps_since_avg[0] = 10;
        coord.steps_since_avg[1] = 9;
        assert!(!coord.should_average());

        // Both at target.
        coord.steps_since_avg[1] = 10;
        assert!(coord.should_average());
    }

    #[test]
    fn should_average_blocked_when_worker_exited() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        coord.steps_since_avg[0] = 1;
        coord.steps_since_avg[1] = 1;
        assert!(coord.should_average());

        // One worker exits.
        coord.active_count = 1;
        assert!(!coord.should_average(), "must not average with fewer active workers");
    }

    #[test]
    fn should_average_blocked_when_all_epochs_done() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        coord.steps_since_avg[0] = 1;
        coord.steps_since_avg[1] = 1;
        assert!(coord.should_average());

        // Mark all epochs aggregated (num_epochs=10, so epoch 9 is the last).
        coord.last_aggregated_epoch = Some(9);
        assert!(coord.all_epochs_done());
        assert!(!coord.should_average());
    }

    #[test]
    fn should_average_blocked_during_cpu_averaging() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        coord.steps_since_avg[0] = 1;
        coord.steps_since_avg[1] = 1;
        assert!(coord.should_average());

        // Simulate CPU averaging in progress.
        coord.avg_state = CpuAvgState::Collecting {
            snapshots: Vec::new(),
            received: vec![false; 2],
            deadline: Instant::now() + std::time::Duration::from_secs(5),
            start: Instant::now(),
            steps_snapshot: vec![0; 2],
            wall_ms_snapshot: vec![0.0; 2],
        };
        assert!(!coord.should_average());
    }

    // -----------------------------------------------------------------------
    // process_timing_msg
    // -----------------------------------------------------------------------

    #[test]
    fn process_timing_msg_batch_accumulates() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        assert_eq!(coord.steps_since_avg[0], 0);
        assert_eq!(coord.wall_ms_accum[0], 0.0);

        coord.process_timing_msg(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 });
        assert_eq!(coord.steps_since_avg[0], 1);
        assert!((coord.wall_ms_accum[0] - 10.0).abs() < 1e-9);
        assert!((coord.last_batch_ms[0] - 10.0).abs() < 1e-9);

        // Second message accumulates.
        coord.process_timing_msg(TimingMsg::Batch { rank: 0, batch_ms: 15.0, step_count: 2 });
        assert_eq!(coord.steps_since_avg[0], 2);
        assert!((coord.wall_ms_accum[0] - 25.0).abs() < 1e-9);
        assert!((coord.last_batch_ms[0] - 15.0).abs() < 1e-9);

        // Rank 1 is independent.
        assert_eq!(coord.steps_since_avg[1], 0);
        assert_eq!(coord.wall_ms_accum[1], 0.0);
    }

    #[test]
    fn process_timing_msg_exiting_decrements_active() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        assert_eq!(coord.active_count, 2);

        coord.process_timing_msg(TimingMsg::Exiting { rank: 0 });
        assert_eq!(coord.active_count, 1);

        // Saturating: double exit won't underflow.
        coord.process_timing_msg(TimingMsg::Exiting { rank: 0 });
        assert_eq!(coord.active_count, 0);
    }

    // -----------------------------------------------------------------------
    // on_rank_done / on_epoch_aggregated
    // -----------------------------------------------------------------------

    #[test]
    fn on_rank_done_async_dispatches_within_lookahead() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Async, 1000);
        // Disable progressive so on_rank_done does the per-rank dispatch.
        coord.progressive = false;

        // Rank 0 finishes epoch 0. Next is epoch 1. No aggregation yet,
        // so lookahead allows epoch 0 and 1 (next <= 1).
        coord.on_rank_done(0, 0);
        // Should have sent StartEpoch(1) to rank 0.
        let msg = control_rxs[0].try_recv();
        assert!(
            matches!(msg, Ok(ControlMsg::StartEpoch(plan)) if plan.epoch == 1),
            "expected StartEpoch(1)"
        );
    }

    #[test]
    fn on_rank_done_async_blocks_beyond_lookahead() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Async, 1000);
        coord.progressive = false;

        // Rank 0 finishes epoch 1 but no aggregation yet.
        // next = 2, within_lookahead = (2 <= 1) = false. Should NOT dispatch.
        coord.on_rank_done(0, 1);
        assert!(
            control_rxs[0].try_recv().is_err(),
            "should NOT dispatch epoch 2 beyond lookahead"
        );
        assert!(coord.rank_waiting[0], "rank should be marked waiting");
    }

    #[test]
    fn on_rank_done_sync_is_noop() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Sync, 1000);
        // Sync policy: on_rank_done does nothing (early return for non-Async).
        coord.on_rank_done(0, 0);
        assert!(control_rxs[0].try_recv().is_err());
    }

    #[test]
    fn on_rank_done_last_epoch_is_noop() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Async, 1000);
        coord.progressive = false;
        // num_epochs = 10, so finishing epoch 9 means next = 10 >= 10.
        coord.on_rank_done(0, 9);
        assert!(control_rxs[0].try_recv().is_err());
    }

    #[test]
    fn on_epoch_aggregated_sends_shutdown_after_last_epoch() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Sync, 1000);
        // Aggregate the last epoch (num_epochs=10, so epoch 9).
        coord.on_epoch_aggregated(9);
        assert!(coord.all_epochs_done());
        // Both ranks should receive Shutdown.
        for rx in &control_rxs {
            let msg = rx.try_recv();
            assert!(
                matches!(msg, Ok(ControlMsg::Shutdown)),
                "expected Shutdown"
            );
        }
    }

    #[test]
    fn on_epoch_aggregated_dispatches_next_sync() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Sync, 100);
        // Disable progressive for Sync.
        coord.progressive = false;

        // Aggregate epoch 0: should dispatch epoch 1 to both ranks.
        coord.on_epoch_aggregated(0);
        assert_eq!(coord.last_aggregated_epoch, Some(0));
        for (rank, rx) in control_rxs.iter().enumerate() {
            let msg = rx.try_recv();
            assert!(
                matches!(msg, Ok(ControlMsg::StartEpoch(plan)) if plan.epoch == 1),
                "rank {rank}: expected StartEpoch(1)"
            );
        }
    }

    #[test]
    fn on_epoch_aggregated_unblocks_waiting_ranks_async() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Async, 1000);
        coord.progressive = false;

        // Simulate: rank 0 finished epoch 1 and is waiting (beyond lookahead).
        coord.rank_epoch[0] = 1;
        coord.rank_waiting[0] = true;
        // Rank 1 is also stuck.
        coord.rank_epoch[1] = 1;
        coord.rank_waiting[1] = true;

        // Aggregate epoch 0: should unblock waiting ranks, dispatching epoch 2.
        coord.on_epoch_aggregated(0);
        for (rank, rx) in control_rxs.iter().enumerate() {
            let msg = rx.try_recv();
            assert!(
                matches!(msg, Ok(ControlMsg::StartEpoch(plan)) if plan.epoch == 2),
                "rank {rank}: expected StartEpoch(2)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Coordinator state accessors
    // -----------------------------------------------------------------------

    #[test]
    fn all_epochs_done_boundary() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 100);
        // num_epochs = 10. Not done until epoch 9 is aggregated.
        assert!(!coord.all_epochs_done());
        coord.last_aggregated_epoch = Some(8);
        assert!(!coord.all_epochs_done());
        coord.last_aggregated_epoch = Some(9);
        assert!(coord.all_epochs_done());
    }

    #[test]
    fn finish_averaging_nccl_resets_counters() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        coord.steps_since_avg[0] = 5;
        coord.steps_since_avg[1] = 3;
        coord.wall_ms_accum[0] = 50.0;
        coord.wall_ms_accum[1] = 30.0;
        coord.throttled[0] = true;

        coord.finish_averaging_nccl();

        assert_eq!(coord.steps_since_avg, vec![0, 0]);
        assert_eq!(coord.wall_ms_accum, vec![0.0, 0.0]);
        assert!(!coord.throttled[0]);
        assert_eq!(coord.version, 1);
        assert_eq!(coord.avg_count, 1);
    }

    #[test]
    fn finish_averaging_cpu_subtracts_snapshots() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        // Simulate: at trigger time, steps were [5, 3], wall was [50, 30].
        // During averaging, rank 0 did 2 more batches (20ms), rank 1 did 1 (10ms).
        coord.steps_since_avg = vec![7, 4];
        coord.wall_ms_accum = vec![70.0, 40.0];

        let steps_snap = vec![5, 3];
        let wall_snap = vec![50.0, 30.0];
        coord.finish_averaging_cpu(12.5, &steps_snap, &wall_snap, None);

        // Residual = current - snapshot.
        assert_eq!(coord.steps_since_avg, vec![2, 1]);
        assert!((coord.wall_ms_accum[0] - 20.0).abs() < 1e-9);
        assert!((coord.wall_ms_accum[1] - 10.0).abs() < 1e-9);
        assert_eq!(coord.version, 1);
        assert_eq!(coord.avg_count, 1);
    }
}
