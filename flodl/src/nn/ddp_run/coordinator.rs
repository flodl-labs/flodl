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
    /// For Async policy: new averaging interval and param norms.
    cadence_update: Option<(usize, Vec<f64>)>,
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

/// Lightweight scheduling coordinator for DDP run mode.
///
/// NOT an optimizer. Each GPU runs its own Adam. The coordinator:
/// 1. Collects timing from workers (for ElChe throughput ratios)
/// 2. Triggers periodic parameter averaging (NCCL or CPU path)
/// 3. Monitors divergence to auto-tune averaging interval
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
    pub(super) el_che: crate::nn::ddp::ElChe,
    version: u64,
    /// Per-rank steps since last averaging.
    steps_since_avg: Vec<usize>,

    // Divergence monitoring (Async mode)
    last_param_norms: Vec<f64>,
    divergence_threshold: f64,
    /// Current adaptive averaging interval (Async mode).
    pub(super) avg_interval: usize,
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
    /// Save a checkpoint every N averaging events. None = disabled.
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
    el_che: crate::nn::ddp::ElChe,
    divergence_threshold: f64,
    checkpoint_every: Option<usize>,
    snapshot_timeout_secs: u64,
    epoch_metrics_tx: Option<mpsc::Sender<super::EpochMetrics>>,
    device_indices: Vec<u8>,
    num_epochs: usize,
    partition_ratios: Option<Vec<f64>>,
}

impl CoordinatorBuilder {
    /// Set the divergence threshold for adaptive averaging interval (Async mode).
    /// Default: 0.05 (5% relative norm difference triggers tightening).
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

    /// Set the checkpoint interval (averaging events between checkpoints).
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
            last_param_norms: vec![0.0; self.world_size],
            divergence_threshold: self.divergence_threshold,
            avg_interval: 1, // start conservative
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
        el_che: crate::nn::ddp::ElChe,
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
        }
    }

    /// Current model version (bumped after each averaging).
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Current averaging interval (K).
    pub fn avg_interval(&self) -> usize {
        self.avg_interval
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
    pub fn send_all_plans(&mut self, epoch: usize) {
        let plans = self.plans_for_epoch(epoch);
        for (rank, plan) in plans.into_iter().enumerate() {
            self.rank_epoch[rank] = epoch;
            self.rank_waiting[rank] = false;
            let _ = self.control_txs[rank].send(ControlMsg::StartEpoch(plan));
        }
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
    fn on_epoch_aggregated(&mut self, epoch: usize) {
        self.last_aggregated_epoch = Some(epoch);
        self.epoch_plan_cache.remove(&epoch);

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
                // Per-rank dispatch already happened in on_rank_done.
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

    /// Process a single timing message. Shared by [`drain_timing`] and
    /// [`drain_timing_blocking`].
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
            // Auto dispatch: per-rank, before aggregation
            self.on_rank_done(msg.rank, msg.epoch);
            self.epoch_buffer.entry(msg.epoch).or_default().push(msg.clone());
            msgs.push(msg);
        }
        // Global aggregation + Sync/Cadence dispatch
        self.try_aggregate_epochs();
        msgs
    }

    /// Check if any buffered epoch has reports from all active ranks.
    /// If so, aggregate, send metrics, and trigger epoch transitions.
    fn try_aggregate_epochs(&mut self) {
        let expected = self.active_count;
        let mut complete: Vec<usize> = self.epoch_buffer.iter()
            .filter(|(_, msgs)| msgs.len() >= expected)
            .map(|(epoch, _)| *epoch)
            .collect();
        // Process in order: dispatch/shutdown logic depends on sequence.
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

    /// Check if averaging should be triggered based on the current policy.
    pub fn should_average(&self) -> bool {
        // Don't re-trigger while a CPU averaging cycle is in progress.
        if !matches!(self.avg_state, CpuAvgState::Idle) {
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
                let counts = self.el_che.batch_counts();
                self.steps_since_avg.iter().enumerate()
                    .all(|(r, &s)| s >= counts[r])
            }
            ApplyPolicy::Async => {
                self.steps_since_avg.iter().all(|&s| s >= self.avg_interval)
            }
        }
    }

    /// Trigger parameter averaging based on the configured backend.
    ///
    /// For NCCL: sends `SyncNow` to all workers, then runs the common tail
    /// (ElChe report, version bump, counter reset) synchronously.
    ///
    /// For CPU: sends `RequestParams`, snapshots the current counters, and
    /// enters the [`CpuAvgState::Collecting`] state. The actual collection
    /// and averaging happen over subsequent [`poll_cpu_averaging`] ticks,
    /// keeping [`check_throttle`] active throughout.
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

        if let Some(every) = self.checkpoint_every {
            if every > 0 && self.avg_count % every == 0 {
                if let Some(tx) = self.control_txs.first() {
                    let _ = tx.send(ControlMsg::Checkpoint { version: self.version });
                }
            }
        }

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
    /// subtract snapshots from current counters (preserve during-averaging batches).
    fn finish_averaging_cpu(
        &mut self,
        avg_ms: f64,
        steps_snapshot: &[usize],
        wall_ms_snapshot: &[f64],
        cadence_update: Option<(usize, Vec<f64>)>,
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

        // Apply cadence update from the compute thread (Async policy).
        if let Some((new_interval, norms)) = cadence_update {
            self.avg_interval = new_interval;
            self.last_param_norms = norms;
        }

        self.version += 1;
        self.avg_count += 1;

        eprintln!(
            "  ddp: CPU averaging #{} complete (v{}, {:.1}ms)",
            self.avg_count, self.version, avg_ms
        );

        if let Some(every) = self.checkpoint_every {
            if every > 0 && self.avg_count % every == 0 {
                if let Some(tx) = self.control_txs.first() {
                    let _ = tx.send(ControlMsg::Checkpoint { version: self.version });
                }
            }
        }

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
                    let policy = self.policy;
                    let avg_interval = self.avg_interval;
                    let div_threshold = self.divergence_threshold;

                    let handle = std::thread::Builder::new()
                        .name("cpu-avg-compute".into())
                        .spawn(move || {
                            Self::compute_average_and_cadence(
                                snapshots, version, policy,
                                avg_interval, div_threshold,
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
                        result.cadence_update,
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

    /// Monitor parameter divergence across replicas and adjust averaging interval.
    ///
    /// Computes relative norm difference: `sum(|norm_i - mean|) / mean`.
    /// If divergence > threshold, halve the interval (more frequent averaging).
    /// If divergence < threshold/2, double the interval (less frequent averaging).
    ///
    /// Note: the production path uses [`compute_average_and_cadence`] on a
    /// background thread. This method is kept for direct testing.
    #[cfg(test)]
    pub(super) fn update_cadence(&mut self, snapshots: &[ParamSnapshot]) -> Result<()> {
        // Compute per-rank parameter L2 norm (sum of all param norms)
        let mut norms = Vec::with_capacity(snapshots.len());
        for snap in snapshots {
            let mut total_norm_sq = 0.0f64;
            for p in &snap.params {
                let n: f64 = p.norm()?.item()?;
                total_norm_sq += n * n;
            }
            norms.push(total_norm_sq.sqrt());
        }

        let mean_norm: f64 = norms.iter().sum::<f64>() / norms.len() as f64;
        if mean_norm < 1e-10 {
            return Ok(()); // all zeros, nothing to do
        }

        let divergence: f64 = norms.iter()
            .map(|n| (n - mean_norm).abs())
            .sum::<f64>() / mean_norm;

        if divergence > self.divergence_threshold {
            // Tighten: halve the interval (min 1)
            self.avg_interval = (self.avg_interval / 2).max(1);
        } else if divergence < self.divergence_threshold / 2.0 {
            // Back off: double the interval
            self.avg_interval = (self.avg_interval * 2).min(1000);
        }

        self.last_param_norms = norms;
        Ok(())
    }

    /// Compute weighted average and (optionally) cadence update on a background thread.
    ///
    /// Static method: captures only the data it needs, no `&self` borrow.
    /// Returns `CpuAvgResult` with the averaged params and an optional
    /// `(new_interval, norms)` tuple for Async policy.
    fn compute_average_and_cadence(
        snapshots: Vec<ParamSnapshot>,
        version: u64,
        policy: ApplyPolicy,
        avg_interval: usize,
        divergence_threshold: f64,
    ) -> Result<CpuAvgResult> {
        let averaged = Self::average_params(&snapshots, version)?;

        let cadence_update = if policy == ApplyPolicy::Async {
            // Inline the divergence computation (same logic as update_cadence
            // but without &mut self, returns the result instead of mutating).
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
            let new_interval = if mean_norm < 1e-10 {
                avg_interval // all zeros, no change
            } else {
                let divergence: f64 = norms.iter()
                    .map(|n| (n - mean_norm).abs())
                    .sum::<f64>() / mean_norm;
                if divergence > divergence_threshold {
                    (avg_interval / 2).max(1)
                } else if divergence < divergence_threshold / 2.0 {
                    (avg_interval * 2).min(1000)
                } else {
                    avg_interval
                }
            };
            Some((new_interval, norms))
        } else {
            None
        };

        Ok(CpuAvgResult { averaged, cadence_update })
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
    /// Drains the dedicated `final_param_rx` channels. Returns a [`TrainedState`]
    /// averaged from whatever snapshots arrived (partial failure: survivors' params
    /// are returned). Returns `None` if zero snapshots were collected.
    pub fn collect_final_state(&self) -> Option<TrainedState> {
        let mut snapshots = Vec::new();
        for rx in &self.final_param_rxs {
            if let Ok(snap) = rx.try_recv() {
                snapshots.push(snap);
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
fn throughput_sizes(el_che: &crate::nn::ddp::ElChe, total: usize) -> Vec<usize> {
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

/// Aggregate per-rank [`MetricsMsg`] into a single [`EpochMetrics`].
///
/// Loss and scalars are averaged weighted by batch count (proportional
/// to each rank's contribution). Epoch time is the max across ranks.
pub(super) fn aggregate_epoch_metrics(epoch: usize, msgs: &[MetricsMsg], device_indices: &[u8]) -> super::EpochMetrics {
    let total_batches: usize = msgs.iter().map(|m| m.batches_processed).sum();

    // Batch-weighted average loss
    let avg_loss = if total_batches > 0 {
        msgs.iter()
            .map(|m| m.avg_loss * m.batches_processed as f64)
            .sum::<f64>()
            / total_batches as f64
    } else {
        0.0
    };

    // Max epoch_ms across ranks
    let epoch_ms = msgs.iter().map(|m| m.epoch_ms).fold(0.0_f64, f64::max);

    // Per-rank scalar means
    let per_rank: Vec<std::collections::HashMap<String, f64>> = msgs
        .iter()
        .map(|m| {
            m.scalars
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
    for m in msgs {
        let w = m.batches_processed as f64;
        for (k, (sum, count)) in &m.scalars {
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
    let total_samples: usize = msgs.iter().map(|m| m.samples_processed).sum();
    let per_rank_throughput: Vec<f64> = msgs.iter().map(|m| {
        if m.epoch_ms > 0.0 { m.samples_processed as f64 / m.epoch_ms } else { 0.0 }
    }).collect();
    let per_rank_batch_share: Vec<f64> = msgs.iter().map(|m| {
        if total_samples > 0 { m.samples_processed as f64 / total_samples as f64 } else { 0.0 }
    }).collect();

    let dev_indices = if device_indices.len() >= msgs.len() {
        device_indices[..msgs.len()].to_vec()
    } else {
        (0..msgs.len() as u8).collect()
    };

    super::EpochMetrics {
        epoch, scalars, per_rank, avg_loss, epoch_ms,
        per_rank_throughput, per_rank_batch_share, device_indices: dev_indices,
    }
}
