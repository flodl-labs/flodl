//! GPU worker: thread-local training loop bound to a single device.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Instant;

use crate::autograd::{Variable, NoGradGuard};
use crate::data::BatchDataSet;
use crate::nn::buffer::Buffer;
use crate::distributed::cuda_event::{CudaEvent, CudaEventFlags};
use crate::distributed::cuda_stream::{CudaStream, StreamGuard};
use crate::distributed::nccl::{NcclRankComm, ReduceOp};
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{Device, Result, Tensor, TensorError};

use super::{
    CheckpointFn, WorkerConfig, TimingMsg, MetricsMsg,
    ParamSnapshot, AveragedParams, ControlMsg, EpochPlan, make_partition,
};

// ---------------------------------------------------------------------------
// GpuWorker
// ---------------------------------------------------------------------------

/// A training worker bound to a single GPU device.
///
/// Generic over the model type `M` so training closures can access concrete
/// model methods (graph tags, traces, etc.) beyond the [`Module`] trait.
///
/// NOT Send: contains `Rc<RefCell<...>>` types (Variable, Buffer, Module).
/// Must be constructed *inside* the spawned thread from Send ingredients.
///
/// Each worker runs an independent training loop with its own optimizer.
/// Communication with the coordinator is via channels (all messages are Send).
pub struct GpuWorker<M: Module> {
    // -- Thread-local model state (non-Send) --
    model: M,
    optimizer: Box<dyn Optimizer>,
    /// Ordered parameter variables for gradient extraction and param loading.
    pub(super) param_vars: Vec<Variable>,
    /// Ordered buffers for buffer synchronization.
    buffer_list: Vec<Buffer>,

    // -- Worker identity --
    rank: usize,
    device: Device,

    // -- CUDA streams for overlap (None on CPU) --
    compute_stream: Option<CudaStream>,
    comm_stream: Option<CudaStream>,
    /// Recorded on comm_stream after param copy/AllReduce.
    /// compute_stream waits on this before each forward.
    copy_done: Option<CudaEvent>,

    // -- NCCL per-rank communicator (None for CPU averaging or CPU device) --
    nccl_comm: Option<NcclRankComm>,

    // -- Channels --
    timing_tx: mpsc::Sender<TimingMsg>,
    metrics_tx: mpsc::Sender<MetricsMsg>,
    /// Used only with AverageBackend::Cpu.
    param_tx: mpsc::Sender<ParamSnapshot>,
    /// Dedicated channel for the final snapshot (avoids race with CPU averaging param_tx).
    final_param_tx: mpsc::Sender<ParamSnapshot>,
    control_rx: mpsc::Receiver<ControlMsg>,

    // -- Data iteration --
    dataset: Arc<dyn BatchDataSet>,
    /// Sample indices for this worker's current partition.
    pub(super) partition: Vec<usize>,
    batch_size: usize,
    base_seed: u64,

    // -- Training state --
    local_step: usize,
    /// Batches since last averaging (for snapshot weighting).
    steps_since_avg: usize,
    current_version: u64,
    pub(super) current_epoch: usize,
    /// Queued epoch plan from coordinator (set if StartEpoch arrives during run_epoch_plan).
    pub(super) pending_plan: Option<EpochPlan>,

    // -- Checkpoint --
    /// Called on rank 0 after averaging events. Log-and-continue on error.
    pub(super) checkpoint_fn: Option<CheckpointFn<M>>,

    // -- Async prefetch (VRAM gauge) --
    /// Background prefetch worker for async H2D transfers (None on CPU).
    prefetch: Option<crate::data::prefetch::PrefetchWorker>,
    /// Bytes per sample (for VRAM gauge depth calculation).
    per_sample_bytes: usize,
    /// Measured activation peak (activations + gradients) from training.
    /// Used as a reserve in the VRAM gauge so prefetch doesn't fill
    /// memory that forward/backward will need. Zero = not yet measured;
    /// first chunk runs sync to calibrate.
    activation_peak_bytes: usize,
    /// Maximum gradient norm for clipping (None = no clipping).
    max_grad_norm: Option<f64>,
}

/// Channels bundle returned by [`GpuWorker::channels`] for wiring into the coordinator.
pub struct WorkerChannels {
    /// Receives timing reports from this worker.
    pub timing_rx: mpsc::Receiver<TimingMsg>,
    /// Receives epoch-end metrics from this worker.
    pub metrics_rx: mpsc::Receiver<MetricsMsg>,
    /// Receives parameter snapshots from this worker (CPU averaging path).
    pub param_rx: mpsc::Receiver<ParamSnapshot>,
    /// Receives the final parameter snapshot from this worker (sent before exit).
    pub final_param_rx: mpsc::Receiver<ParamSnapshot>,
    /// Sends control messages to this worker.
    pub control_tx: mpsc::Sender<ControlMsg>,
}

/// Worker-side channel endpoints for passing into [`GpuWorker::new`].
#[allow(clippy::type_complexity)]
pub type WorkerEndpoints = (
    mpsc::Sender<TimingMsg>,
    mpsc::Sender<MetricsMsg>,
    mpsc::Sender<ParamSnapshot>,
    mpsc::Sender<ParamSnapshot>,  // final_param_tx
    mpsc::Receiver<ControlMsg>,
);

impl<M: Module> GpuWorker<M> {
    /// Create the channel pairs for one worker.
    ///
    /// Returns (worker-side senders/receiver, coordinator-side receivers/sender).
    /// Call this on the main thread, then pass the worker-side halves into
    /// [`GpuWorker::new`] inside the spawned thread.
    pub fn channels() -> (WorkerEndpoints, WorkerChannels) {
        let (timing_tx, timing_rx) = mpsc::channel();
        let (metrics_tx, metrics_rx) = mpsc::channel();
        let (param_tx, param_rx) = mpsc::channel();
        let (final_param_tx, final_param_rx) = mpsc::channel();
        let (control_tx, control_rx) = mpsc::channel();
        (
            (timing_tx, metrics_tx, param_tx, final_param_tx, control_rx),
            WorkerChannels { timing_rx, metrics_rx, param_rx, final_param_rx, control_tx },
        )
    }

    /// Build a GpuWorker inside a spawned thread.
    ///
    /// `model_factory` creates the model on `config.device` (thread-local, Rc-based).
    /// `optim_factory` creates the optimizer for the model's parameters.
    /// `initial_params`/`initial_buffers` from `WorkerConfig` are copied into the
    /// model's Variables to synchronize all workers to the same starting state.
    #[allow(clippy::too_many_arguments)]
    pub fn new<F, G, O>(
        config: &WorkerConfig,
        model_factory: F,
        optim_factory: G,
        dataset: Arc<dyn BatchDataSet>,
        nccl_comm: Option<NcclRankComm>,
        checkpoint_fn: Option<CheckpointFn<M>>,
        timing_tx: mpsc::Sender<TimingMsg>,
        metrics_tx: mpsc::Sender<MetricsMsg>,
        param_tx: mpsc::Sender<ParamSnapshot>,
        final_param_tx: mpsc::Sender<ParamSnapshot>,
        control_rx: mpsc::Receiver<ControlMsg>,
    ) -> Result<Self>
    where
        F: FnOnce(Device) -> Result<M>,
        G: FnOnce(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        // Create model on the target device
        let model = model_factory(config.device)?;
        let params = model.parameters();
        let buffers = model.buffers();

        // Copy initial params into model variables (no_grad: leaf tensors with requires_grad)
        if params.len() != config.initial_params.len() {
            return Err(TensorError::new(&format!(
                "GpuWorker rank {}: model has {} params but config has {}",
                config.rank, params.len(), config.initial_params.len()
            )));
        }
        {
            let _no_grad = NoGradGuard::new();
            for (p, src) in params.iter().zip(&config.initial_params) {
                p.variable.data().copy_(src, false)?;
            }
        }

        // Copy initial buffers into model buffers
        if buffers.len() != config.initial_buffers.len() {
            return Err(TensorError::new(&format!(
                "GpuWorker rank {}: model has {} buffers but config has {}",
                config.rank, buffers.len(), config.initial_buffers.len()
            )));
        }
        for (b, src) in buffers.iter().zip(&config.initial_buffers) {
            b.get().copy_(src, false)?;
        }

        // Create optimizer for this replica's parameters
        let optimizer = optim_factory(&params);

        // Extract variable handles (for snapshot/load)
        let param_vars: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        let buffer_list = buffers;

        // Create CUDA streams if on GPU
        let (compute_stream, comm_stream, copy_done) = if config.device.is_cuda() {
            let cs = CudaStream::new(config.device, false)?;
            let ms = CudaStream::new(config.device, false)?;
            let ev = CudaEvent::new(CudaEventFlags::DisableTiming)?;
            // Record initial event so first wait_event is a no-op
            ev.record_on(&ms)?;
            (Some(cs), Some(ms), Some(ev))
        } else {
            (None, None, None)
        };

        // Create prefetch worker for async H2D (VRAM gauge).
        // Cap depth at 512 to avoid huge channel allocations when
        // batch_bytes is tiny (e.g. toy test datasets).
        // Depth 0 = skip prefetch entirely (sync fallback for tight VRAM).
        // Skip entirely when the dataset fits in a single batch (nothing to
        // prefetch ahead of).
        //
        // Note: activation_reserve=0 here because we haven't measured the
        // training activation peak yet. The first run_epoch_plan() will
        // force depth=0 (sync) to calibrate, then adjust on subsequent chunks.
        let total_batches = dataset.len() / config.batch_size.max(1);
        let (prefetch, per_sample_bytes) = if config.device.is_cuda() && total_batches > 1 {
            let sample = dataset.get_batch(&[0])?;
            let psb: usize = sample.iter().map(|t| t.nbytes()).sum();
            drop(sample);
            let depth = crate::data::prefetch_depth_from_vram(
                psb, config.batch_size, config.device, 0.90, 0,
            ).min(512);
            // Reset peak stats so first run_epoch_plan gets a clean baseline.
            crate::tensor::cuda_reset_peak_stats_idx(config.device.index() as i32);
            if depth > 0 {
                let pw = crate::data::prefetch::PrefetchWorker::new(
                    Arc::clone(&dataset), config.device, depth,
                );
                (Some(pw), psb)
            } else {
                (None, psb)
            }
        } else {
            (None, 0)
        };

        Ok(GpuWorker {
            model,
            optimizer: Box::new(optimizer),
            param_vars,
            buffer_list,
            rank: config.rank,
            device: config.device,
            compute_stream,
            comm_stream,
            copy_done,
            nccl_comm,
            timing_tx,
            metrics_tx,
            param_tx,
            final_param_tx,
            control_rx,
            dataset,
            partition: Vec::new(), // filled by first StartEpoch from coordinator
            batch_size: config.batch_size,
            base_seed: config.seed,
            local_step: 0,
            steps_since_avg: 0,
            current_version: 0,
            current_epoch: 0,
            pending_plan: None,
            checkpoint_fn,
            prefetch,
            per_sample_bytes,
            activation_peak_bytes: 0,
            max_grad_norm: config.max_grad_norm,
        })
    }

    /// This worker's rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// This worker's device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Current local step count.
    pub fn local_step(&self) -> usize {
        self.local_step
    }

    /// Current model version (updated after loading averaged params).
    pub fn current_version(&self) -> u64 {
        self.current_version
    }

    /// Current epoch number (0-based).
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Set the learning rate on this worker's optimizer.
    pub fn set_lr(&mut self, lr: f64) {
        self.optimizer.set_lr(lr);
    }

    /// A reference to the concrete model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Extract current parameter values as a [`ParamSnapshot`].
    ///
    /// Tensors are copied to CPU so that the coordinator's compute thread
    /// never needs CUDA access (avoiding slow CUDA context init on the
    /// compute thread, which can deadlock with `drain_avg_state`).
    ///
    /// Synchronizes comm_stream before reading, so Update + RequestParams
    /// processed in the same `handle_control()` call cannot read mid-copy data.
    pub fn snapshot_params(&self) -> ParamSnapshot {
        // Wait for any pending load_averaged() non-blocking copy to finish.
        if let Some(stream) = &self.comm_stream {
            let _ = stream.synchronize();
        }
        let params = self.param_vars.iter()
            .map(|v| {
                let t = v.data();
                if t.device() == Device::CPU { t } else { t.to_device(Device::CPU).unwrap_or(t) }
            })
            .collect();
        let buffers = self.buffer_list.iter()
            .map(|b| {
                let t = b.get();
                if t.device() == Device::CPU { t } else { t.to_device(Device::CPU).unwrap_or(t) }
            })
            .collect();
        ParamSnapshot {
            rank: self.rank,
            params,
            buffers,
            batch_count: self.steps_since_avg.max(1),
        }
    }

    /// Load averaged parameters from the coordinator (CPU averaging path).
    ///
    /// Uses `copy_(non_blocking=true)` on the comm stream for GPU overlap.
    /// Records a `CudaEvent` so the compute stream waits before the next forward.
    pub fn load_averaged(&mut self, update: &AveragedParams) -> Result<()> {
        if update.params.len() != self.param_vars.len() {
            return Err(TensorError::new(&format!(
                "load_averaged: expected {} params, got {}",
                self.param_vars.len(), update.params.len()
            )));
        }

        let non_blocking = self.comm_stream.is_some();
        // Set comm stream as current if available (copy_ respects current stream)
        let _guard = self.comm_stream.as_ref().map(StreamGuard::new);

        // no_grad: parameters are leaf tensors with requires_grad=true
        {
            let _no_grad = NoGradGuard::new();
            for (var, src) in self.param_vars.iter().zip(&update.params) {
                var.data().copy_(src, non_blocking)?;
            }
        }
        for (buf, src) in self.buffer_list.iter().zip(&update.buffers) {
            buf.get().copy_(src, non_blocking)?;
        }

        // Record event on comm_stream so compute_stream can wait
        if let (Some(ev), Some(stream)) = (&self.copy_done, &self.comm_stream) {
            ev.record_on(stream)?;
        }

        self.current_version = update.version;
        Ok(())
    }

    /// Perform in-place NCCL AllReduce(Avg) on this rank's parameters.
    ///
    /// All ranks must process SyncNow concurrently for the collective to complete.
    /// Runs on `comm_stream` and records `copy_done` so the compute stream waits
    /// before the next forward.
    fn sync_now_nccl(&self) -> Result<()> {
        let comm = match &self.nccl_comm {
            Some(c) => c,
            None => return Ok(()), // No NCCL comm (CPU backend or single-GPU): no-op
        };

        // Collect param data tensors (raw storage, no grad graph)
        let param_tensors: Vec<_> = self.param_vars.iter().map(|v| v.data()).collect();
        let param_refs: Vec<&Tensor> = param_tensors.iter().collect();

        if let Some(stream) = &self.comm_stream {
            // AllReduce on comm_stream (non-blocking on host)
            comm.all_reduce_on_stream(&param_refs, ReduceOp::Avg, stream)?;
            // Record event so compute_stream waits before next forward
            if let Some(ev) = &self.copy_done {
                ev.record_on(stream)?;
            }
        } else {
            // CPU fallback (should not happen for NCCL backend, but handle gracefully)
            comm.all_reduce(&param_refs, ReduceOp::Avg)?;
        }

        Ok(())
    }

    /// Wait for any pending parameter copy to complete on the compute stream.
    ///
    /// Must be called before each forward pass to prevent reading mid-copy params.
    /// No-op on CPU (no streams).
    fn sync_before_forward(&self) -> Result<()> {
        if let (Some(ev), Some(stream)) = (&self.copy_done, &self.compute_stream) {
            stream.wait_event(ev)?;
        }
        Ok(())
    }

    /// Run one forward + backward + optimizer step.
    ///
    /// `train_fn` receives a reference to the concrete model `M` and the batch
    /// tensors, and must return the scalar loss [`Variable`]. The worker handles
    /// stream sync, backward, optimizer step, and zero_grad.
    ///
    /// Returns `(loss_value, wall_ms)`.
    pub fn train_step(
        &mut self,
        batch: &[Tensor],
        train_fn: &impl Fn(&M, &[Tensor]) -> Result<Variable>,
    ) -> Result<(f64, f64)> {
        self.sync_before_forward()?;

        let start = Instant::now();

        // User-provided forward + loss computation
        let loss = train_fn(&self.model, batch)?;
        let loss_val: f64 = loss.data().item()?;

        // Backward
        loss.backward()?;

        // Per-worker gradient clipping (before optimizer step).
        if let Some(max_norm) = self.max_grad_norm {
            let params: Vec<Tensor> = self.model.parameters()
                .iter()
                .filter(|p| p.variable.grad().is_some())
                .map(|p| p.variable.data())
                .collect();
            if !params.is_empty() {
                Tensor::clip_grad_norm_fused(&params, max_norm)?;
            }
        }

        // Optimizer step (GPU-local Adam, ~0.1ms fused kernel)
        self.optimizer.step()?;
        self.optimizer.zero_grad();

        self.local_step += 1;
        self.steps_since_avg += 1;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok((loss_val, elapsed_ms))
    }

    /// Process pending control messages (non-blocking).
    ///
    /// Returns `true` if a Shutdown was received.
    pub fn handle_control(&mut self) -> Result<bool> {
        while let Ok(msg) = self.control_rx.try_recv() {
            if self.dispatch_control(msg)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Block on control messages until Shutdown or channel disconnect.
    ///
    /// Called after training is done and `report_exiting()` has been sent.
    /// Skips NCCL collectives (SyncNow): since this worker has reported
    /// Exiting, the coordinator may not send SyncNow to our peers, but
    /// if it was already in-flight, calling AllReduce here would deadlock
    /// if the peer has also exited or errored.
    pub fn drain_until_shutdown(&mut self) {
        while let Ok(msg) = self.control_rx.recv() {
            match msg {
                ControlMsg::SyncNow => {
                    // Skip: peer may be dead, AllReduce would deadlock.
                    // The coordinator will stop triggering collectives
                    // once it processes our Exiting message.
                }
                ControlMsg::Shutdown => break,
                other => {
                    if self.dispatch_control(other).unwrap_or(true) {
                        break;
                    }
                }
            }
        }
    }

    /// Handle a single control message. Returns `true` on Shutdown.
    fn dispatch_control(&mut self, msg: ControlMsg) -> Result<bool> {
        match msg {
            ControlMsg::RequestParams => {
                let _ = self.param_tx.send(self.snapshot_params());
            }
            ControlMsg::Update(avg) => {
                self.load_averaged(&avg)?;
                self.steps_since_avg = 0;
            }
            ControlMsg::SyncNow => {
                self.sync_now_nccl()?;
                self.steps_since_avg = 0;
            }
            ControlMsg::StartEpoch(plan) => {
                self.pending_plan = Some(plan);
            }
            ControlMsg::Throttle => {
                // Worker is ahead of the slowest rank: block until averaging
                // completes (SyncNow/Update) or Shutdown. Intermediate messages
                // (RequestParams, StartEpoch) are handled but don't release
                // the throttle. Duplicate Throttle messages are ignored.
                loop {
                    match self.control_rx.recv() {
                        Ok(ControlMsg::Throttle) => continue, // already throttled
                        Ok(msg) => {
                            let releases = matches!(
                                &msg,
                                ControlMsg::SyncNow
                                    | ControlMsg::Update(_)
                                    | ControlMsg::Shutdown
                            );
                            let shutdown = self.dispatch_control(msg)?;
                            if shutdown || releases {
                                return Ok(shutdown);
                            }
                        }
                        Err(_) => return Ok(true), // channel dead
                    }
                }
            }
            ControlMsg::Checkpoint { version } => {
                if let Some(ref f) = self.checkpoint_fn {
                    if let Err(e) = f(version, &self.model) {
                        eprintln!("  ddp: checkpoint failed (v{version}): {e}");
                    }
                }
            }
            ControlMsg::Shutdown => return Ok(true),
        }
        Ok(false)
    }

    /// Send a timing report to the coordinator.
    pub fn report_timing(&self, batch_ms: f64) -> Result<()> {
        self.timing_tx.send(TimingMsg::Batch {
            rank: self.rank,
            batch_ms,
            step_count: self.local_step,
        }).map_err(|_| TensorError::new("timing channel disconnected"))
    }

    /// Send the final parameter snapshot on the dedicated channel before exiting.
    ///
    /// This uses `final_param_tx` (not `param_tx`) to avoid racing with
    /// CPU averaging snapshot collection on the same channel.
    pub fn send_final_snapshot(&self) {
        let _ = self.final_param_tx.send(self.snapshot_params());
    }

    /// Abort the NCCL communicator, unblocking any stuck collective.
    ///
    /// Must be called before [`Self::send_final_snapshot`] when the training loop
    /// exits due to shutdown. A pending AllReduce on `comm_stream` (from a
    /// SyncNow whose peer died) would block `to_device(CPU)` in snapshot_params
    /// because the CUDA default stream synchronizes with all other streams.
    pub fn abort_nccl(&mut self) {
        if let Some(comm) = self.nccl_comm.take() {
            let _ = comm.abort_handle().abort();
        }
    }

    /// Notify the coordinator that this worker is about to exit.
    ///
    /// Must be called before the thread terminates so the coordinator
    /// stops including this rank in NCCL collectives.
    pub fn report_exiting(&self) {
        let _ = self.timing_tx.send(TimingMsg::Exiting { rank: self.rank });
    }

    /// Send epoch-end metrics to the coordinator.
    ///
    /// Drains the thread-local scalar accumulator populated by
    /// [`record_scalar()`](super::record_scalar) calls during this epoch.
    pub fn report_epoch(&self, avg_loss: f64, batches: usize, epoch_ms: f64) -> Result<()> {
        let scalars = super::drain_scalars();
        self.metrics_tx.send(MetricsMsg {
            rank: self.rank,
            epoch: self.current_epoch,
            avg_loss,
            batches_processed: batches,
            epoch_ms,
            samples_processed: batches * self.batch_size,
            scalars,
        }).map_err(|_| TensorError::new("metrics channel disconnected"))
    }

    /// Block until the coordinator sends a StartEpoch or Shutdown.
    ///
    /// Handles intermediate control messages (SyncNow, RequestParams, etc.)
    /// to prevent NCCL deadlock while waiting between epochs.
    /// Returns `Some(plan)` for the next epoch, or `None` on Shutdown/disconnect.
    pub fn wait_for_epoch_plan(&mut self) -> Result<Option<EpochPlan>> {
        loop {
            // Check if a plan was queued by dispatch_control (e.g. StartEpoch
            // arrived during Throttle handler). Must be checked each iteration,
            // not just at entry, because dispatch_control may set it mid-loop.
            if let Some(plan) = self.pending_plan.take() {
                return Ok(Some(plan));
            }
            match self.control_rx.recv() {
                Ok(ControlMsg::StartEpoch(plan)) => return Ok(Some(plan)),
                Ok(ControlMsg::Shutdown) => return Ok(None),
                Ok(msg) => {
                    if self.dispatch_control(msg)? {
                        return Ok(None); // Shutdown consumed by handler (e.g. Throttle)
                    }
                }
                Err(_) => return Ok(None), // disconnected
            }
        }
    }

    /// Process one partition (or chunk) from the coordinator's plan.
    ///
    /// Generates sample indices from the plan's offset and size using the
    /// same deterministic shuffle as all other ranks. Reports metrics at
    /// the end so the coordinator can track completion.
    ///
    /// On CUDA, batches are prefetched asynchronously via a background
    /// worker thread with a VRAM-sized buffer (gauge model). On CPU,
    /// batches are loaded synchronously.
    ///
    /// Returns `true` if a Shutdown was received mid-plan.
    pub fn run_epoch_plan(
        &mut self,
        plan: &EpochPlan,
        train_fn: &impl Fn(&M, &[Tensor]) -> Result<Variable>,
    ) -> Result<bool> {
        self.current_epoch = plan.epoch;
        self.partition = make_partition(
            plan.partition_offset, plan.partition_size,
            self.dataset.len(), plan.epoch, self.base_seed,
        );

        let num_batches = self.partition.len() / self.batch_size;
        if num_batches == 0 {
            // Still report so coordinator gets the "done" signal.
            let _ = self.report_epoch(0.0, 0, 0.0);
            return Ok(false);
        }

        // Defragment CUDA allocator between chunks. The caching allocator
        // holds freed blocks from the previous chunk's activations and NCCL
        // buffers, which fragment VRAM. Without this, follow-up chunks can
        // OOM even though enough total VRAM exists (just not contiguous).
        if self.device.is_cuda() {
            crate::tensor::cuda_empty_cache();
        }

        // Update activation peak from the previous chunk's high-water mark.
        // Uses max() so the budget never grows beyond the worst observed peak.
        // Sync first so all async work (NCCL on comm_stream, etc.) completes
        // and deferred frees are processed before reading the baseline.
        if self.device.is_cuda() && self.activation_peak_bytes > 0 {
            let idx = self.device.index() as i32;
            crate::tensor::cuda_synchronize(idx as u8);
            if let Ok(peak) = crate::tensor::cuda_peak_active_bytes_idx(idx) {
                if let Ok(baseline) = crate::tensor::cuda_active_bytes_idx(idx) {
                    let overhead = (peak as usize).saturating_sub(baseline as usize);
                    let batch_bytes = self.per_sample_bytes * self.batch_size;
                    let activation = overhead.saturating_sub(batch_bytes);
                    self.activation_peak_bytes = self.activation_peak_bytes.max(activation);
                }
            }
            crate::tensor::cuda_reset_peak_stats_idx(idx);
        }

        // Recalculate prefetch depth at each plan boundary (VRAM may vary).
        // Cap at num_batches: no point buffering more than the chunk contains.
        // Depth 0 means VRAM is too tight for any prefetch buffer.
        //
        // If activation peak hasn't been measured yet, force depth=0 (sync
        // fallback) so the first chunk can calibrate safely.
        let use_prefetch = if let Some(ref mut pw) = self.prefetch {
            if self.activation_peak_bytes == 0 && self.device.is_cuda() {
                pw.set_prefetch_depth(0);
                false
            } else {
                let vram_depth = crate::data::prefetch_depth_from_vram(
                    self.per_sample_bytes, self.batch_size, self.device, 0.90,
                    self.activation_peak_bytes,
                );
                let depth = vram_depth.min(num_batches);
                pw.set_prefetch_depth(depth);
                depth > 0
            }
        } else {
            false
        };

        let epoch_start = Instant::now();
        let mut total_loss = 0.0;

        if use_prefetch {
            // CUDA async path: prefetch with VRAM gauge.
            let prefetch = self.prefetch.as_ref().unwrap();
            // start_distributed_epoch creates a fresh bounded channel whose
            // capacity equals the prefetch depth (VRAM budget). The prefetch
            // thread fills it; SyncSender blocks when VRAM is full.
            let batch_rx = prefetch.start_distributed_epoch();

            // Submit all batch indices for async H2D transfer
            for batch_idx in 0..num_batches {
                let start = batch_idx * self.batch_size;
                let end = start + self.batch_size;
                prefetch.load_batch(self.partition[start..end].to_vec());
            }

            // Consume prefetched batches as they become ready
            for _ in 0..num_batches {
                let prefetched = batch_rx.recv()
                    .map_err(|_| TensorError::new("prefetch channel closed"))??;

                // Ensure compute stream waits for async H2D copy to finish
                #[cfg(feature = "cuda")]
                if let Some(ref event) = prefetched.ready_event {
                    if let Some(ref stream) = self.compute_stream {
                        stream.wait_event(event)?;
                    }
                }

                let (loss, ms) = self.train_step(&prefetched.tensors, train_fn)?;
                total_loss += loss;
                let _ = self.report_timing(ms);
                if self.handle_control()? {
                    return Ok(true); // Shutdown
                }
            }
        } else {
            // Sync path: load one batch at a time, move to device if needed.
            // Used for CPU devices, or CUDA when VRAM is too tight for prefetch.
            let measuring_peak = self.activation_peak_bytes == 0 && self.device.is_cuda();

            for batch_idx in 0..num_batches {
                let start = batch_idx * self.batch_size;
                let end = start + self.batch_size;
                let indices = &self.partition[start..end];
                let batch = self.dataset.get_batch(indices)?;

                let batch: Vec<Tensor> = if self.device.is_cuda() {
                    batch.into_iter()
                        .map(|t| t.to_device(self.device))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    batch
                };

                let (loss, ms) = self.train_step(&batch, train_fn)?;
                total_loss += loss;

                // After first batch: measure activation peak from CUDA stats.
                // The peak includes model + batch + activations + gradients.
                // Subtract baseline (model/optimizer/NCCL) and one batch to
                // isolate the activation + gradient overhead. This is the
                // reserve that prefetch_depth_from_vram must account for.
                if measuring_peak && batch_idx == 0 {
                    let idx = self.device.index() as i32;
                    crate::tensor::cuda_synchronize(idx as u8);
                    if let Ok(peak) = crate::tensor::cuda_peak_active_bytes_idx(idx) {
                        if let Ok(current) = crate::tensor::cuda_active_bytes_idx(idx) {
                            let overhead = (peak as usize).saturating_sub(current as usize);
                            let batch_bytes = self.per_sample_bytes * self.batch_size;
                            self.activation_peak_bytes = overhead.saturating_sub(batch_bytes);
                        }
                    }
                    // Reset for ongoing monitoring in subsequent chunks.
                    crate::tensor::cuda_reset_peak_stats_idx(idx);
                }

                let _ = self.report_timing(ms);
                if self.handle_control()? {
                    return Ok(true); // Shutdown
                }
            }
        }

        let epoch_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;
        let _ = self.report_epoch(
            total_loss / num_batches as f64,
            num_batches,
            epoch_ms,
        );

        Ok(false)
    }
}
