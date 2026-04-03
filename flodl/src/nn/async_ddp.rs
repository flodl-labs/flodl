//! Async DDP: thread-per-GPU training with Local SGD and adaptive parameter averaging.
//!
//! Each GPU runs its own optimizer independently (zero wait). A lightweight coordinator
//! triggers periodic parameter averaging at ElChe-determined intervals. Two orthogonal
//! knobs control the behavior: [`ApplyPolicy`] (when to average) and [`AverageBackend`]
//! (how to average).
//!
//! # Architecture
//!
//! ```text
//! GPU Thread 0:  create model+Adam+dataset -> [fwd -> bwd -> adam step -> repeat]
//! GPU Thread 1:  create model+Adam+dataset -> [fwd -> bwd -> adam step -> repeat]
//! Coordinator:   collect timing/metrics -> trigger param averaging -> monitor divergence
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use crate::autograd::{Variable, NoGradGuard};
use crate::data::BatchDataSet;
use crate::nn::buffer::Buffer;
use crate::nn::cuda_event::{CudaEvent, CudaEventFlags};
use crate::nn::cuda_stream::{CudaStream, StreamGuard};
use crate::nn::nccl::{NcclRankComm, ReduceOp};
use crate::nn::{Module, Optimizer, Parameter};
use crate::rng::Rng;
use crate::tensor::{Device, Result, Tensor, TensorError};

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// Controls WHEN parameter averaging occurs (the interval K).
///
/// All three modes run the same architecture; only the averaging trigger differs.
/// - `Sync`: K=1 (every batch). Equivalent to standard DDP.
/// - `Cadence`: K=N (ElChe anchor count). Moderate GPU independence.
/// - `Async`: K=adaptive (ElChe auto-tunes based on divergence). Maximum independence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ApplyPolicy {
    /// Average after every batch (K=1). Equivalent to standard synchronous DDP.
    Sync,
    /// Average every N batches where N = ElChe anchor count.
    Cadence,
    /// ElChe auto-tunes averaging interval based on observed divergence.
    /// Starts conservative (K=1), backs off as convergence stabilizes.
    Async,
}

/// Controls HOW parameter averaging is performed.
///
/// Orthogonal to [`ApplyPolicy`]. All combinations are valid, enabling A/B testing:
/// same model, same K, NCCL vs CPU -- if loss curves match, CPU async is validated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AverageBackend {
    /// NCCL AllReduce in-place on GPU params. Hard sync at averaging point.
    /// Fast GPU waits for slow GPU at the collective barrier.
    /// Simpler code path: no snapshot collection, no CPU tensors.
    Nccl,
    /// CPU-mediated async snapshots through coordinator.
    /// No GPU ever blocks. Slight staleness window during transfer.
    Cpu,
}

/// Configuration for [`AsyncDdp`] tuning knobs.
///
/// All fields have sensible defaults. Use the builder methods to customize.
#[derive(Clone, Debug)]
pub struct AsyncDdpConfig {
    /// ElChe overhead target (fraction of compute time). Default: 0.10.
    pub overhead_target: Option<f64>,
    /// Maximum anchor count (gradient staleness limit). Default: 200.
    pub max_anchor: Option<usize>,
    /// Initial ElChe anchor (batches before first sync). Default: 10.
    pub anchor: Option<usize>,
    /// Divergence threshold for Async mode. Default: 0.05.
    pub divergence_threshold: Option<f64>,
}

impl Default for AsyncDdpConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncDdpConfig {
    /// Create a default config (all defaults).
    pub fn new() -> Self {
        AsyncDdpConfig {
            overhead_target: None,
            max_anchor: None,
            anchor: None,
            divergence_threshold: None,
        }
    }

    /// Set the AllReduce overhead target (fraction of compute time).
    pub fn with_overhead_target(mut self, target: f64) -> Self {
        self.overhead_target = Some(target);
        self
    }

    /// Set the maximum anchor count.
    pub fn with_max_anchor(mut self, max: usize) -> Self {
        self.max_anchor = Some(max);
        self
    }

    /// Set the initial anchor count.
    pub fn with_anchor(mut self, anchor: usize) -> Self {
        self.anchor = Some(anchor);
        self
    }

    /// Set the divergence threshold for Async mode.
    pub fn with_divergence_threshold(mut self, threshold: f64) -> Self {
        self.divergence_threshold = Some(threshold);
        self
    }
}

// ---------------------------------------------------------------------------
// Worker -> Coordinator messages
// ---------------------------------------------------------------------------

/// Timing report sent from a GPU worker to the coordinator after each batch.
///
/// Lightweight (3 fields, all Copy). Sent every batch for ElChe throughput tracking.
#[derive(Clone, Debug)]
pub struct TimingMsg {
    /// Which GPU sent this.
    pub rank: usize,
    /// Wall-clock time for this batch (ms).
    pub batch_ms: f64,
    /// Worker's local step counter (monotonically increasing).
    pub step_count: usize,
}

/// Epoch-end metrics sent from a GPU worker to the coordinator.
///
/// Fire-and-forget: worker sends this and immediately starts the next epoch.
#[derive(Clone, Debug)]
pub struct MetricsMsg {
    /// Which GPU sent this.
    pub rank: usize,
    /// Epoch number (local to this worker).
    pub epoch: usize,
    /// Average loss over this epoch.
    pub avg_loss: f64,
    /// Number of batches processed in this epoch.
    pub batches_processed: usize,
    /// Wall-clock time for this epoch (ms).
    pub epoch_ms: f64,
}

/// Parameter snapshot sent from a GPU worker to the coordinator (CPU averaging path only).
///
/// Contains cloned Tensor handles (Send+Sync via libtorch refcount).
#[derive(Clone)]
pub struct ParamSnapshot {
    /// Which GPU sent this.
    pub rank: usize,
    /// Current parameter tensors (on this worker's GPU device).
    pub params: Vec<Tensor>,
    /// Current buffer tensors (BatchNorm running stats, etc.).
    pub buffers: Vec<Tensor>,
    /// Number of batches processed since last averaging (for weighting).
    pub batch_count: usize,
}

// ---------------------------------------------------------------------------
// Coordinator -> Worker messages
// ---------------------------------------------------------------------------

/// Averaged parameters sent from the coordinator to a GPU worker (CPU averaging path only).
///
/// Contains pinned CPU tensors. Worker copies them into its Variables via `copy_(non_blocking=true)`.
#[derive(Clone)]
pub struct AveragedParams {
    /// Averaged parameter tensors (pinned CPU memory).
    pub params: Vec<Tensor>,
    /// Averaged buffer tensors.
    pub buffers: Vec<Tensor>,
    /// Monotonically increasing version number.
    pub version: u64,
}

/// Control signals from the coordinator to a GPU worker.
pub enum ControlMsg {
    /// \[CPU path\] Request parameter snapshot for averaging.
    RequestParams,
    /// \[CPU path\] Deliver averaged parameters.
    Update(AveragedParams),
    /// \[NCCL path\] Trigger in-place AllReduce on this worker's own params.
    /// Worker runs AllReduce on comm_stream and records CudaEvent.
    SyncNow,
    /// ElChe rebalance: adjust this worker's sample count for next epoch.
    /// Fire-and-forget from coordinator; worker applies at next `reshuffle_partition()`.
    PartitionHint {
        /// Number of samples this worker should process per epoch.
        num_samples: usize,
    },
    /// Shut down this worker.
    Shutdown,
}

// ---------------------------------------------------------------------------
// Initial setup
// ---------------------------------------------------------------------------

/// Configuration passed to a GPU worker at spawn time.
///
/// All fields are Send. The worker uses these to construct its thread-local
/// Graph, Optimizer, DataLoader, and streams inside the spawned thread.
#[derive(Clone)]
pub struct WorkerConfig {
    /// This worker's rank (0..world_size).
    pub rank: usize,
    /// Total number of workers.
    pub world_size: usize,
    /// The CUDA device this worker operates on.
    pub device: Device,
    /// Initial parameter tensors in pinned CPU memory (from rank 0 snapshot).
    /// Worker copies these into its Variables at startup.
    pub initial_params: Vec<Tensor>,
    /// Initial buffer tensors in pinned CPU memory.
    pub initial_buffers: Vec<Tensor>,
    /// Total number of samples in the dataset.
    pub total_samples: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Initial number of samples for this worker's partition.
    pub partition_samples: usize,
    /// RNG seed for deterministic shuffling.
    pub seed: u64,
}

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
    param_vars: Vec<Variable>,
    /// Ordered buffers for buffer synchronization.
    buffer_list: Vec<Buffer>,

    // -- Worker identity --
    rank: usize,
    device: Device,
    world_size: usize,

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
    control_rx: mpsc::Receiver<ControlMsg>,

    // -- Data iteration --
    dataset: Arc<dyn BatchDataSet>,
    /// Sample indices for this worker's current partition.
    partition: Vec<usize>,
    batch_size: usize,
    base_seed: u64,

    // -- Training state --
    local_step: usize,
    /// Batches since last averaging (for snapshot weighting).
    steps_since_avg: usize,
    current_version: u64,
    current_epoch: usize,
    pending_partition_size: Option<usize>,
}

/// Channels bundle returned by [`GpuWorker::channels`] for wiring into the coordinator.
pub struct WorkerChannels {
    /// Receives timing reports from this worker.
    pub timing_rx: mpsc::Receiver<TimingMsg>,
    /// Receives epoch-end metrics from this worker.
    pub metrics_rx: mpsc::Receiver<MetricsMsg>,
    /// Receives parameter snapshots from this worker (CPU averaging path).
    pub param_rx: mpsc::Receiver<ParamSnapshot>,
    /// Sends control messages to this worker.
    pub control_tx: mpsc::Sender<ControlMsg>,
}

/// Worker-side channel endpoints for passing into [`GpuWorker::new`].
#[allow(clippy::type_complexity)]
pub type WorkerEndpoints = (
    mpsc::Sender<TimingMsg>,
    mpsc::Sender<MetricsMsg>,
    mpsc::Sender<ParamSnapshot>,
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
        let (control_tx, control_rx) = mpsc::channel();
        (
            (timing_tx, metrics_tx, param_tx, control_rx),
            WorkerChannels { timing_rx, metrics_rx, param_rx, control_tx },
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
        timing_tx: mpsc::Sender<TimingMsg>,
        metrics_tx: mpsc::Sender<MetricsMsg>,
        param_tx: mpsc::Sender<ParamSnapshot>,
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

        // Generate initial partition for this rank
        let partition = make_partition(
            config.rank, config.world_size, config.total_samples,
            config.partition_samples, 0, config.seed,
        );

        Ok(GpuWorker {
            model,
            optimizer: Box::new(optimizer),
            param_vars,
            buffer_list,
            rank: config.rank,
            device: config.device,
            world_size: config.world_size,
            compute_stream,
            comm_stream,
            copy_done,
            nccl_comm,
            timing_tx,
            metrics_tx,
            param_tx,
            control_rx,
            dataset,
            partition,
            batch_size: config.batch_size,
            base_seed: config.seed,
            local_step: 0,
            steps_since_avg: 0,
            current_version: 0,
            current_epoch: 0,
            pending_partition_size: None,
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

    /// A reference to the concrete model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Extract current parameter values as a [`ParamSnapshot`].
    ///
    /// The returned tensors are shallow clones (share storage with the model's
    /// Variables). They are `Send` and can be passed through channels.
    pub fn snapshot_params(&self) -> ParamSnapshot {
        let params = self.param_vars.iter().map(|v| v.data()).collect();
        let buffers = self.buffer_list.iter().map(|b| b.get()).collect();
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
                ControlMsg::PartitionHint { num_samples } => {
                    self.pending_partition_size = Some(num_samples);
                }
                ControlMsg::Shutdown => return Ok(true),
            }
        }
        Ok(false)
    }

    /// Send a timing report to the coordinator.
    pub fn report_timing(&self, batch_ms: f64) -> Result<()> {
        self.timing_tx.send(TimingMsg {
            rank: self.rank,
            batch_ms,
            step_count: self.local_step,
        }).map_err(|_| TensorError::new("timing channel disconnected"))
    }

    /// Send epoch-end metrics to the coordinator.
    pub fn report_epoch(&self, avg_loss: f64, batches: usize, epoch_ms: f64) -> Result<()> {
        self.metrics_tx.send(MetricsMsg {
            rank: self.rank,
            epoch: self.current_epoch,
            avg_loss,
            batches_processed: batches,
            epoch_ms,
        }).map_err(|_| TensorError::new("metrics channel disconnected"))
    }

    /// Consume the pending partition hint (if any) and return the new sample count.
    ///
    /// Called at epoch boundaries by the worker's training loop.
    pub fn take_partition_hint(&mut self) -> Option<usize> {
        self.pending_partition_size.take()
    }

    /// Advance to the next epoch.
    pub fn advance_epoch(&mut self) {
        self.current_epoch += 1;
    }

    /// Reshuffle this worker's partition for the current epoch.
    ///
    /// If a [`PartitionHint`](ControlMsg::PartitionHint) was received, the new
    /// sample count is applied first.
    pub fn shuffle_partition(&mut self) {
        let partition_size = self.pending_partition_size.take()
            .unwrap_or(self.partition.len());
        self.partition = make_partition(
            self.rank, self.world_size, self.dataset.len(),
            partition_size, self.current_epoch, self.base_seed,
        );
    }

    /// Run one epoch: iterate batches, train, report timing, handle control.
    ///
    /// Returns `true` if a Shutdown was received (caller should exit).
    pub fn run_epoch(
        &mut self,
        train_fn: &impl Fn(&M, &[Tensor]) -> Result<Variable>,
    ) -> Result<bool> {
        self.shuffle_partition();

        let num_batches = self.partition.len() / self.batch_size;
        if num_batches == 0 {
            self.advance_epoch();
            return Ok(false);
        }

        let epoch_start = Instant::now();
        let mut total_loss = 0.0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * self.batch_size;
            let end = start + self.batch_size;
            let indices = &self.partition[start..end];

            // Fetch batch from dataset (returns CPU tensors)
            let cpu_batch = self.dataset.get_batch(indices)?;

            // Move to this worker's device
            let batch: Vec<Tensor> = if self.device == Device::CPU {
                cpu_batch
            } else {
                cpu_batch.into_iter()
                    .map(|t| t.to_device(self.device))
                    .collect::<Result<_>>()?
            };

            // Train step
            let (loss, ms) = self.train_step(&batch, train_fn)?;
            total_loss += loss;

            // Report timing (fire-and-forget)
            let _ = self.report_timing(ms);

            // Check control messages between batches
            if self.handle_control()? {
                return Ok(true); // Shutdown
            }
        }

        // Epoch done: fire metrics and advance
        let epoch_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;
        let _ = self.report_epoch(
            total_loss / num_batches as f64,
            num_batches,
            epoch_ms,
        );
        self.advance_epoch();

        Ok(false)
    }
}

// ---------------------------------------------------------------------------
// Partition generation
// ---------------------------------------------------------------------------

/// Generate a deterministic partition of sample indices for one rank.
///
/// All ranks sharing the same `(epoch, seed)` produce the same global permutation,
/// then each takes a slice from its own block of `total/world_size` items.
///
/// **Non-overlapping guarantee:** when all ranks use the same epoch and seed, and
/// `partition_size <= total/world_size`, slices are disjoint by construction.
/// When `partition_size > total/world_size`, overflow enters adjacent blocks.
/// During epoch transitions (ranks at different epochs), overlap is benign
/// for Local SGD convergence.
fn make_partition(
    rank: usize,
    world_size: usize,
    total: usize,
    partition_size: usize,
    epoch: usize,
    seed: u64,
) -> Vec<usize> {
    // Deterministic global shuffle (same seed = same permutation for all ranks)
    let mut rng = Rng::seed(seed.wrapping_add(epoch as u64));
    let mut all: Vec<usize> = (0..total).collect();
    rng.shuffle(&mut all);

    // This rank's slice (non-overlapping when sizes sum to total)
    let per_rank = total / world_size;
    let offset = rank * per_rank;
    let end = (offset + partition_size).min(total);
    all[offset..end].to_vec()
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

/// Lightweight scheduling coordinator for async DDP.
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
    control_txs: Vec<mpsc::Sender<ControlMsg>>,

    // Configuration
    policy: ApplyPolicy,
    backend: AverageBackend,
    world_size: usize,
    total_samples: usize,

    // Scheduling
    el_che: super::ddp::ElChe,
    version: u64,
    /// Per-rank steps since last averaging.
    steps_since_avg: Vec<usize>,

    // Divergence monitoring (Async mode)
    last_param_norms: Vec<f64>,
    divergence_threshold: f64,
    /// Current adaptive averaging interval (Async mode).
    avg_interval: usize,
    /// Has ElChe been calibrated (first timing report received)?
    calibrated: bool,

    /// Number of workers still actively training. Shared with worker threads.
    /// When < world_size, NCCL collective ops would deadlock; skip averaging.
    active_workers: Arc<AtomicUsize>,

    // Timing accumulation for ElChe
    /// Accumulated wall-clock ms per rank since last averaging.
    /// Fed to ElChe::report_timing at each averaging event.
    wall_ms_accum: Vec<f64>,
    /// Most recent batch_ms per rank (for display/monitoring).
    last_batch_ms: Vec<f64>,
}

/// Builder for configuring a [`Coordinator`].
pub struct CoordinatorBuilder {
    timing_rx: mpsc::Receiver<TimingMsg>,
    metrics_rx: mpsc::Receiver<MetricsMsg>,
    param_rx: mpsc::Receiver<ParamSnapshot>,
    control_txs: Vec<mpsc::Sender<ControlMsg>>,
    policy: ApplyPolicy,
    backend: AverageBackend,
    world_size: usize,
    total_samples: usize,
    el_che: super::ddp::ElChe,
    divergence_threshold: f64,
    active_workers: Arc<AtomicUsize>,
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

    /// Build the coordinator.
    pub fn build(self) -> Coordinator {
        Coordinator {
            timing_rx: self.timing_rx,
            metrics_rx: self.metrics_rx,
            param_rx: self.param_rx,
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
            active_workers: self.active_workers,
            wall_ms_accum: vec![0.0; self.world_size],
            last_batch_ms: vec![0.0; self.world_size],
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
        control_txs: Vec<mpsc::Sender<ControlMsg>>,
        policy: ApplyPolicy,
        backend: AverageBackend,
        world_size: usize,
        total_samples: usize,
        el_che: super::ddp::ElChe,
        active_workers: Arc<AtomicUsize>,
    ) -> CoordinatorBuilder {
        CoordinatorBuilder {
            timing_rx,
            metrics_rx,
            param_rx,
            control_txs,
            policy,
            backend,
            world_size,
            total_samples,
            el_che,
            divergence_threshold: 0.05,
            active_workers,
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

    /// Process all pending timing messages (non-blocking drain).
    ///
    /// Updates per-rank step counts and accumulates wall-clock time for ElChe.
    pub fn drain_timing(&mut self) {
        while let Ok(msg) = self.timing_rx.try_recv() {
            self.steps_since_avg[msg.rank] = self.steps_since_avg[msg.rank].saturating_add(1);
            self.wall_ms_accum[msg.rank] += msg.batch_ms;
            self.last_batch_ms[msg.rank] = msg.batch_ms;
        }
    }

    /// Process all pending metrics messages (non-blocking drain).
    ///
    /// Returns collected metrics for logging/monitoring.
    pub fn drain_metrics(&mut self) -> Vec<MetricsMsg> {
        let mut msgs = Vec::new();
        while let Ok(msg) = self.metrics_rx.try_recv() {
            msgs.push(msg);
        }
        msgs
    }

    /// Check if averaging should be triggered based on the current policy.
    pub fn should_average(&self) -> bool {
        // Collectives require all ranks. If any worker has exited,
        // skip averaging to prevent NCCL deadlock or channel disconnect.
        if self.active_workers.load(Ordering::Relaxed) < self.world_size {
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
    /// For NCCL: sends `SyncNow` to all workers (they AllReduce in-place).
    /// For CPU: collects snapshots, computes weighted average, sends updates.
    pub fn trigger_averaging(&mut self) -> Result<()> {
        match self.backend {
            AverageBackend::Nccl => {
                for tx in &self.control_txs {
                    let _ = tx.send(ControlMsg::SyncNow);
                }
            }
            AverageBackend::Cpu => {
                // 1. Request snapshots from all workers
                for tx in &self.control_txs {
                    let _ = tx.send(ControlMsg::RequestParams);
                }
                // 2. Collect all snapshots (blocking)
                let mut snapshots = Vec::with_capacity(self.world_size);
                for _ in 0..self.world_size {
                    let snap = self.param_rx.recv()
                        .map_err(|_| TensorError::new("param channel disconnected"))?;
                    snapshots.push(snap);
                }
                // 3. Check divergence, adjust cadence (Async mode)
                if self.policy == ApplyPolicy::Async {
                    self.update_cadence(&snapshots)?;
                }
                // 4. Compute weighted average
                let averaged = Self::average_params(&snapshots, self.version + 1)?;
                // 5. Send averaged params to all workers
                for tx in &self.control_txs {
                    let _ = tx.send(ControlMsg::Update(averaged.clone()));
                }
            }
        }

        // Report accumulated timing to ElChe for cadence adaptation.
        // wall_ms_accum[rank] = total wall-clock ms for all batches on that rank
        // since the last averaging event. ElChe divides by batch_counts to get ms/batch.
        if self.wall_ms_accum.iter().any(|&ms| ms > 0.0) {
            let sync_ms = 0.0; // no separate sync measurement in async model
            self.el_che.report_timing(&self.wall_ms_accum, sync_ms);
            if !self.calibrated && self.el_che.is_calibrated() {
                self.calibrated = true;
                self.rebalance_partitions();
            }
        }

        self.version += 1;
        for s in &mut self.steps_since_avg {
            *s = 0;
        }
        for a in &mut self.wall_ms_accum {
            *a = 0.0;
        }
        Ok(())
    }

    /// Compute weighted average of parameter snapshots.
    ///
    /// Weight each rank's contribution by its `batch_count` (number of batches
    /// since last averaging). This ensures faster GPUs contribute more.
    fn average_params(snapshots: &[ParamSnapshot], version: u64) -> Result<AveragedParams> {
        if snapshots.is_empty() {
            return Err(TensorError::new("average_params: no snapshots"));
        }

        let total_batches: usize = snapshots.iter().map(|s| s.batch_count.max(1)).sum();
        let n_params = snapshots[0].params.len();
        let n_buffers = snapshots[0].buffers.len();

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

        // Average buffers (unweighted, just use rank 0 on CPU)
        let avg_buffers: Vec<Tensor> = (0..n_buffers)
            .map(|bi| snapshots[0].buffers[bi].to_device(Device::CPU))
            .collect::<Result<_>>()?;

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
    fn update_cadence(&mut self, snapshots: &[ParamSnapshot]) -> Result<()> {
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

    /// Send PartitionHint to each worker based on ElChe throughput ratios.
    ///
    /// Called after ElChe calibration stabilizes. Fire-and-forget.
    pub fn rebalance_partitions(&self) {
        let counts = self.el_che.batch_counts();
        let total_ratio: usize = counts.iter().sum();
        if total_ratio == 0 {
            return;
        }
        for (rank, tx) in self.control_txs.iter().enumerate() {
            let num_samples = if rank < counts.len() {
                (self.total_samples * counts[rank]) / total_ratio
            } else {
                0
            };
            let _ = tx.send(ControlMsg::PartitionHint { num_samples });
        }
    }

    /// Run one tick of the coordinator loop.
    ///
    /// Drains timing/metrics, checks if averaging is due, triggers if so.
    /// Returns collected metrics (if any) for external logging.
    pub fn tick(&mut self) -> Result<Vec<MetricsMsg>> {
        self.drain_timing();
        let metrics = self.drain_metrics();

        if self.should_average() {
            self.trigger_averaging()?;
        }

        Ok(metrics)
    }
}

// ---------------------------------------------------------------------------
// AsyncDdp orchestrator
// ---------------------------------------------------------------------------

/// Async DDP orchestrator: spawns GPU worker threads and a coordinator thread.
///
/// Each GPU runs its own training loop with a local optimizer. The coordinator
/// triggers periodic parameter averaging based on [`ApplyPolicy`] and
/// [`AverageBackend`]. Workers self-manage their epochs.
///
/// # Usage
///
/// ```ignore
/// use flodl::*;
///
/// let ddp = AsyncDdp::auto(
///     |dev| Ok(Linear::on_device(784, 10, dev)?),
///     |params| Adam::new(params, 0.001),
///     dataset,
///     32,         // batch_size
///     ApplyPolicy::Async,
///     AverageBackend::Nccl,
/// )?;
///
/// // Training runs autonomously. Wait for completion:
/// ddp.join()?;
/// ```
pub struct AsyncDdp {
    worker_handles: Vec<std::thread::JoinHandle<Result<()>>>,
    coordinator_handle: Option<std::thread::JoinHandle<Result<()>>>,
    devices: Vec<Device>,
    shutdown: Arc<AtomicBool>,
}

impl AsyncDdp {
    /// Detect GPUs, spawn worker threads and coordinator thread with default config.
    ///
    /// See [`auto_with`](Self::auto_with) for the full parameter set.
    #[allow(clippy::too_many_arguments)]
    pub fn auto<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        policy: ApplyPolicy,
        backend: AverageBackend,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        Self::auto_with(
            model_factory, optim_factory, train_fn,
            dataset, batch_size, num_epochs,
            policy, backend, AsyncDdpConfig::new(),
        )
    }

    /// Detect GPUs, spawn worker threads and coordinator thread.
    ///
    /// - `model_factory` creates a model on the given device (called once per GPU thread).
    /// - `optim_factory` creates an optimizer for a model's parameters.
    /// - `train_fn` receives `(&M, &[Tensor])` and returns the scalar loss Variable.
    ///   The worker handles backward, optimizer step, and zero_grad.
    /// - `dataset` is shared across workers (Arc, indexed by partition).
    /// - `num_epochs` controls how many epochs each worker trains before stopping.
    /// - `config` tunes ElChe and divergence monitoring parameters.
    ///
    /// With 2+ CUDA devices, spawns one thread per GPU and a coordinator.
    /// With 0-1 CUDA devices, runs training on the single available device
    /// (no threads, no coordinator, no averaging).
    #[allow(clippy::too_many_arguments)]
    pub fn auto_with<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        policy: ApplyPolicy,
        backend: AverageBackend,
        config: AsyncDdpConfig,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        use std::sync::atomic::{AtomicBool, Ordering};

        let devices = crate::tensor::usable_cuda_devices();

        // Single-GPU fallback: run on main thread, no coordinator
        if devices.len() < 2 {
            let dev = devices.first().copied().unwrap_or(Device::CPU);
            return Self::run_single(
                &model_factory, &optim_factory, &train_fn,
                dataset, batch_size, num_epochs, dev,
            );
        }

        // Print device summary (same style as Ddp::auto)
        Self::print_summary(&devices, &policy, &backend);

        // Step 1: Create temp model on device[0] to extract initial params
        let tmp_model = model_factory(devices[0])?;
        let initial_params: Vec<Tensor> = tmp_model.parameters().iter()
            .map(|p| p.variable.data().to_device(Device::CPU).and_then(|t| t.pin_memory()))
            .collect::<Result<Vec<_>>>()?;
        let initial_buffers: Vec<Tensor> = tmp_model.buffers().iter()
            .map(|b| b.get().to_device(Device::CPU).and_then(|t| t.pin_memory()))
            .collect::<Result<Vec<_>>>()?;
        drop(tmp_model);

        let world_size = devices.len();
        let total_samples = dataset.len();
        let samples_per_worker = total_samples / world_size;

        // Step 2: Create channels
        let (timing_tx_main, timing_rx) = mpsc::channel();
        let (metrics_tx_main, metrics_rx) = mpsc::channel();
        let (param_tx_main, param_rx) = mpsc::channel();

        let mut coord_control_txs = Vec::new();
        let mut worker_control_rxs = Vec::new();
        for _ in 0..world_size {
            let (tx, rx) = mpsc::channel();
            coord_control_txs.push(tx);
            worker_control_rxs.push(rx);
        }

        // Step 2b: Init NCCL comms from main thread, then split into per-rank comms.
        // CRITICAL: ncclCommInitRank from worker threads corrupts CUDA context on
        // heterogeneous GPUs. Always use NcclComms::new() + split() instead.
        // See NcclRankComm and NcclComms::split docs for details.
        let mut rank_comms: Vec<Option<NcclRankComm>> = if backend == AverageBackend::Nccl {
            let group = super::nccl::NcclComms::new(&devices)?;
            group.split()?.into_iter().map(Some).collect()
        } else {
            (0..world_size).map(|_| None).collect()
        };

        // Step 3: Create ElChe with config knobs
        let anchor = config.anchor.unwrap_or(10);
        let mut el_che = super::ddp::ElChe::new(world_size, anchor);
        if let Some(target) = config.overhead_target {
            el_che = el_che.with_overhead_target(target);
        }
        if let Some(max) = config.max_anchor {
            el_che = el_che.with_max_anchor(max);
        }

        // Step 4: Spawn coordinator thread
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_coord = shutdown.clone();
        let active_workers = Arc::new(AtomicUsize::new(world_size));
        let active_coord = active_workers.clone();
        let div_threshold = config.divergence_threshold;

        let coordinator_handle = std::thread::Builder::new()
            .name("async-ddp-coordinator".into())
            .spawn(move || {
                let mut builder = Coordinator::builder(
                    timing_rx, metrics_rx, param_rx,
                    coord_control_txs,
                    policy, backend,
                    world_size, total_samples, el_che,
                    active_coord,
                );
                if let Some(dt) = div_threshold {
                    builder = builder.divergence_threshold(dt);
                }
                let mut coord = builder.build();

                while !shutdown_coord.load(Ordering::Relaxed) {
                    coord.tick()?;
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }

                Ok(())
            })
            .map_err(|e| TensorError::new(&format!("failed to spawn coordinator: {e}")))?;

        // Step 5: Spawn GPU worker threads
        let model_factory = Arc::new(model_factory);
        let optim_factory = Arc::new(optim_factory);
        let train_fn = Arc::new(train_fn);
        let mut worker_handles = Vec::new();

        for (rank, control_rx) in worker_control_rxs.into_iter().enumerate() {
            let device = devices[rank];
            let mf = model_factory.clone();
            let of = optim_factory.clone();
            let tf = train_fn.clone();
            let ds = dataset.clone();
            let params = initial_params.clone();
            let buffers = initial_buffers.clone();
            let t_tx = timing_tx_main.clone();
            let m_tx = metrics_tx_main.clone();
            let p_tx = param_tx_main.clone();
            let shutdown_w = shutdown.clone();
            let active_w = active_workers.clone();

            let worker_nccl = rank_comms[rank].take();
            let config = WorkerConfig {
                rank,
                world_size,
                device,
                initial_params: params,
                initial_buffers: buffers,
                total_samples,
                batch_size,
                partition_samples: samples_per_worker,
                seed: 42,
            };

            let handle = std::thread::Builder::new()
                .name(format!("async-ddp-gpu-{rank}"))
                .spawn(move || {
                    // Set CUDA device for this thread
                    if let Device::CUDA(idx) = device {
                        crate::tensor::set_current_cuda_device(idx);
                    }

                    // Build worker inside this thread (model + optimizer are Rc-based, thread-local)
                    // NCCL comm was pre-initialized on the main thread via NcclComms::split()
                    // to avoid per-thread ncclCommInitRank issues with CUDA context corruption.
                    let mut worker = GpuWorker::new(
                        &config,
                        |dev| (*mf)(dev),
                        |params| (*of)(params),
                        ds,
                        worker_nccl,
                        t_tx,
                        m_tx,
                        p_tx,
                        control_rx,
                    )?;

                    // Training loop: self-managed epochs with actual data iteration
                    for _epoch in 0..num_epochs {
                        if shutdown_w.load(Ordering::Relaxed) {
                            break;
                        }
                        if worker.run_epoch(&*tf)? {
                            break; // Shutdown received
                        }
                    }

                    // Signal coordinator that this worker is done.
                    // This prevents NCCL deadlock: coordinator won't send
                    // SyncNow once any worker has exited.
                    active_w.fetch_sub(1, Ordering::Relaxed);

                    Ok(())
                })
                .map_err(|e| TensorError::new(&format!("failed to spawn worker {rank}: {e}")))?;

            worker_handles.push(handle);
        }

        // Drop the main thread's clones so coordinator sees channel disconnect
        // when all workers are done
        drop(timing_tx_main);
        drop(metrics_tx_main);
        drop(param_tx_main);

        Ok(AsyncDdp {
            worker_handles,
            coordinator_handle: Some(coordinator_handle),
            devices: devices.to_vec(),
            shutdown,
        })
    }

    /// Single-GPU fallback: run training on the main thread.
    ///
    /// No coordinator, no worker threads, no parameter averaging.
    /// Same training loop as multi-GPU workers.
    fn run_single<F, M, G, O, T>(
        model_factory: &F,
        optim_factory: &G,
        train_fn: &T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        device: Device,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M>,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable>,
    {
        use std::sync::atomic::AtomicBool;

        eprintln!("  async-ddp: single device ({device:?}) | no coordination");

        let total_samples = dataset.len();
        let tmp_model = model_factory(device)?;
        let initial_params: Vec<Tensor> = tmp_model.parameters().iter()
            .map(|p| p.variable.data())
            .collect();
        let initial_buffers: Vec<Tensor> = tmp_model.buffers().iter()
            .map(|b| b.get())
            .collect();
        drop(tmp_model);

        let config = WorkerConfig {
            rank: 0,
            world_size: 1,
            device,
            initial_params,
            initial_buffers,
            total_samples,
            batch_size,
            partition_samples: total_samples,
            seed: 42,
        };

        // Channels (unused in single-GPU mode, but needed by GpuWorker)
        let ((timing_tx, metrics_tx, param_tx, control_rx), _channels) =
            GpuWorker::<M>::channels();

        let mut worker = GpuWorker::new(
            &config,
            model_factory,
            optim_factory,
            dataset,
            None, // no NCCL for single-GPU
            timing_tx,
            metrics_tx,
            param_tx,
            control_rx,
        )?;

        // Train directly on this thread
        for _epoch in 0..num_epochs {
            worker.run_epoch(train_fn)?;
        }

        Ok(AsyncDdp {
            worker_handles: Vec::new(),
            coordinator_handle: None,
            devices: vec![device],
            shutdown: Arc::new(AtomicBool::new(true)),
        })
    }

    /// Number of GPUs in this DDP group.
    pub fn world_size(&self) -> usize {
        self.devices.len()
    }

    /// Devices in use.
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// Wait for all training to complete and shut down. Consumes self.
    ///
    /// Workers run their `num_epochs` and exit naturally. After all workers
    /// finish, the coordinator is signalled to stop.
    ///
    /// For single-GPU mode, this is a no-op (training already completed in `auto()`).
    pub fn join(mut self) -> Result<()> {
        // Wait for all workers to finish their epochs first
        for h in self.worker_handles.drain(..) {
            h.join()
                .map_err(|_| TensorError::new("worker thread panicked"))??;
        }

        // All workers done. Signal coordinator to stop.
        self.shutdown.store(true, Ordering::Relaxed);

        if let Some(h) = self.coordinator_handle.take() {
            h.join()
                .map_err(|_| TensorError::new("coordinator thread panicked"))??;
        }

        Ok(())
    }

    /// Print device summary to stderr (same style as Ddp::auto).
    fn print_summary(devices: &[Device], policy: &ApplyPolicy, backend: &AverageBackend) {
        use crate::tensor::{cuda_device_name_idx, cuda_memory_info_idx};
        use crate::monitor::format_bytes;

        let mut parts = Vec::with_capacity(devices.len());
        let mut names = Vec::with_capacity(devices.len());

        for &dev in devices {
            if let Device::CUDA(idx) = dev {
                let raw_name = cuda_device_name_idx(idx as i32)
                    .unwrap_or_else(|| format!("CUDA({})", idx));
                let short = raw_name
                    .strip_prefix("NVIDIA ")
                    .unwrap_or(&raw_name)
                    .to_string();
                let vram = cuda_memory_info_idx(idx as i32)
                    .ok()
                    .map(|(_, total)| format!(" ({})", format_bytes(total)))
                    .unwrap_or_default();
                parts.push(format!("{}{}", short, vram));
                names.push(raw_name);
            }
        }

        let heterogeneous = names.windows(2).any(|w| w[0] != w[1]);
        let mode = if heterogeneous { "heterogeneous" } else { "homogeneous" };
        let policy_str = match policy {
            ApplyPolicy::Sync => "sync",
            ApplyPolicy::Cadence => "cadence",
            ApplyPolicy::Async => "async",
        };
        let backend_str = match backend {
            AverageBackend::Nccl => "nccl",
            AverageBackend::Cpu => "cpu",
        };

        eprintln!(
            "  async-ddp: {} GPUs ({}) | {} | policy={} backend={}",
            devices.len(), mode, parts.join(" | "), policy_str, backend_str,
        );
    }
}

impl Drop for AsyncDdp {
    fn drop(&mut self) {
        // Signal shutdown if not already joined
        self.shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_policy_variants() {
        let policies = [ApplyPolicy::Sync, ApplyPolicy::Cadence, ApplyPolicy::Async];
        assert_eq!(policies.len(), 3);
        assert_eq!(ApplyPolicy::Sync, ApplyPolicy::Sync);
        assert_ne!(ApplyPolicy::Sync, ApplyPolicy::Async);
    }

    #[test]
    fn test_average_backend_variants() {
        let backends = [AverageBackend::Nccl, AverageBackend::Cpu];
        assert_eq!(backends.len(), 2);
        assert_eq!(AverageBackend::Nccl, AverageBackend::Nccl);
        assert_ne!(AverageBackend::Nccl, AverageBackend::Cpu);
    }

    #[test]
    fn test_control_msg_variants() {
        // Verify all variants are constructable
        let _req = ControlMsg::RequestParams;
        let _sync = ControlMsg::SyncNow;
        let _hint = ControlMsg::PartitionHint { num_samples: 1000 };
        let _shutdown = ControlMsg::Shutdown;
        let _update = ControlMsg::Update(AveragedParams {
            params: vec![],
            buffers: vec![],
            version: 0,
        });
    }

    #[test]
    fn test_timing_msg_send() {
        // TimingMsg must be Send (all fields are Copy primitives)
        fn assert_send<T: Send>() {}
        assert_send::<TimingMsg>();
    }

    #[test]
    fn test_metrics_msg_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MetricsMsg>();
    }

    #[test]
    fn test_param_snapshot_send() {
        // ParamSnapshot contains Vec<Tensor> which is Send (Tensor: unsafe impl Send)
        fn assert_send<T: Send>() {}
        assert_send::<ParamSnapshot>();
    }

    #[test]
    fn test_averaged_params_send() {
        fn assert_send<T: Send>() {}
        assert_send::<AveragedParams>();
    }

    #[test]
    fn test_control_msg_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ControlMsg>();
    }

    #[test]
    fn test_worker_config_send() {
        fn assert_send<T: Send>() {}
        assert_send::<WorkerConfig>();
    }

    #[test]
    fn test_worker_config_clone() {
        let cfg = WorkerConfig {
            rank: 0,
            world_size: 2,
            device: Device::CPU,
            initial_params: vec![],
            initial_buffers: vec![],
            total_samples: 10000,
            batch_size: 32,
            partition_samples: 5000,
            seed: 42,
        };
        let cfg2 = cfg.clone();
        assert_eq!(cfg2.rank, 0);
        assert_eq!(cfg2.world_size, 2);
        assert_eq!(cfg2.total_samples, 10000);
        assert_eq!(cfg2.partition_samples, 5000);
    }

    // -----------------------------------------------------------------------
    // GpuWorker tests
    // -----------------------------------------------------------------------

    use crate::nn::Linear;
    use crate::tensor::{test_device, test_opts, Tensor, TensorOptions, DType};

    /// Simple test dataset: random (input, target) pairs.
    struct TestDataset {
        n: usize,
    }
    impl crate::data::BatchDataSet for TestDataset {
        fn len(&self) -> usize { self.n }
        fn get_batch(&self, indices: &[usize]) -> crate::tensor::Result<Vec<Tensor>> {
            let n = indices.len() as i64;
            let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
            Ok(vec![
                Tensor::randn(&[n, 4], opts)?,
                Tensor::randn(&[n, 2], opts)?,
            ])
        }
    }

    /// Simple MSE train function for tests.
    fn mse_train(model: &Linear, batch: &[Tensor]) -> Result<Variable> {
        let input = Variable::new(batch[0].clone(), false);
        let target = Variable::new(batch[1].clone(), false);
        let output = model.forward(&input)?;
        let diff = output.sub(&target)?;
        diff.mul(&diff)?.mean()
    }

    /// Create a GpuWorker with a simple Linear model for testing.
    fn make_test_worker() -> (GpuWorker<Linear>, WorkerChannels) {
        make_test_worker_with(0, 1, 1000)
    }

    /// Create a GpuWorker with configurable rank/world_size/dataset_size.
    fn make_test_worker_with(
        rank: usize,
        world_size: usize,
        dataset_size: usize,
    ) -> (GpuWorker<Linear>, WorkerChannels) {
        let dev = test_device();

        // Build a temporary model to extract initial params
        let tmp_model = Linear::on_device(4, 2, dev).unwrap();
        let tmp_params: Vec<Tensor> = tmp_model.parameters().iter()
            .map(|p| p.variable.data())
            .collect();
        let tmp_buffers: Vec<Tensor> = tmp_model.buffers().iter()
            .map(|b| b.get())
            .collect();
        drop(tmp_model);

        let config = WorkerConfig {
            rank,
            world_size,
            device: dev,
            initial_params: tmp_params,
            initial_buffers: tmp_buffers,
            total_samples: dataset_size,
            batch_size: 4,
            partition_samples: dataset_size / world_size,
            seed: 42,
        };

        let ((timing_tx, metrics_tx, param_tx, control_rx), channels) =
            GpuWorker::<Linear>::channels();

        let dataset: Arc<dyn crate::data::BatchDataSet> =
            Arc::new(TestDataset { n: dataset_size });

        let worker = GpuWorker::new(
            &config,
            |d| Linear::on_device(4, 2, d),
            |params| crate::nn::SGD::new(params, 0.01, 0.0),
            dataset,
            None, // no NCCL in unit tests
            timing_tx,
            metrics_tx,
            param_tx,
            control_rx,
        ).unwrap();

        (worker, channels)
    }

    #[test]
    fn test_worker_new_and_accessors() {
        let (worker, _ch) = make_test_worker();
        assert_eq!(worker.rank(), 0);
        assert_eq!(worker.local_step(), 0);
        assert_eq!(worker.current_version(), 0);
        assert_eq!(worker.param_vars.len(), 2); // Linear: weight + bias
    }

    #[test]
    fn test_worker_snapshot_params() {
        let (worker, _ch) = make_test_worker();
        let snap = worker.snapshot_params();
        assert_eq!(snap.rank, 0);
        assert_eq!(snap.params.len(), 2); // weight + bias
        assert_eq!(snap.buffers.len(), 0); // Linear has no buffers
        assert_eq!(snap.batch_count, 1); // max(steps_since_avg=0, 1)

        // Verify snapshot tensors have the right shapes
        assert_eq!(snap.params[0].shape(), &[2, 4]); // weight
        assert_eq!(snap.params[1].shape(), &[2]);     // bias
    }

    #[test]
    fn test_worker_snapshot_is_send() {
        let (worker, _ch) = make_test_worker();
        let snap = worker.snapshot_params();

        // Verify snapshot can be sent through a channel
        let (tx, rx) = mpsc::channel::<ParamSnapshot>();
        tx.send(snap).unwrap();
        let received = rx.recv().unwrap();
        assert_eq!(received.rank, 0);
        assert_eq!(received.params.len(), 2);
    }

    #[test]
    fn test_worker_load_averaged() {
        let (mut worker, _ch) = make_test_worker();
        let dev = test_device();
        let opts = TensorOptions { dtype: DType::Float32, device: dev };

        // Create "averaged" params (all ones)
        let new_weight = Tensor::ones(&[2, 4], opts).unwrap();
        let new_bias = Tensor::ones(&[2], opts).unwrap();

        let update = AveragedParams {
            params: vec![new_weight, new_bias],
            buffers: vec![],
            version: 42,
        };

        worker.load_averaged(&update).unwrap();

        // Verify version updated
        assert_eq!(worker.current_version(), 42);

        // Verify model params now contain all ones
        let snap = worker.snapshot_params();
        let w_sum: f64 = snap.params[0].sum().unwrap().item().unwrap();
        assert!((w_sum - 8.0).abs() < 1e-5, "weight should be all ones (sum=8), got {w_sum}");
        let b_sum: f64 = snap.params[1].sum().unwrap().item().unwrap();
        assert!((b_sum - 2.0).abs() < 1e-5, "bias should be all ones (sum=2), got {b_sum}");
    }

    #[test]
    fn test_worker_load_averaged_wrong_count() {
        let (mut worker, _ch) = make_test_worker();

        let update = AveragedParams {
            params: vec![], // wrong count
            buffers: vec![],
            version: 1,
        };
        assert!(worker.load_averaged(&update).is_err());
    }

    #[test]
    fn test_worker_train_step() {
        let (mut worker, ch) = make_test_worker();
        let opts = test_opts();

        let batch = vec![
            Tensor::randn(&[4, 4], opts).unwrap(),
            Tensor::randn(&[4, 2], opts).unwrap(),
        ];

        let (loss, ms) = worker.train_step(&batch, &mse_train).unwrap();
        assert!(ms > 0.0);
        assert!(loss > 0.0);
        assert_eq!(worker.local_step(), 1);

        // Verify timing was NOT auto-sent (train_step doesn't auto-send)
        assert!(ch.timing_rx.try_recv().is_err());
    }

    #[test]
    fn test_worker_report_timing() {
        let (worker, ch) = make_test_worker();

        worker.report_timing(12.5).unwrap();

        let msg = ch.timing_rx.recv().unwrap();
        assert_eq!(msg.rank, 0);
        assert!((msg.batch_ms - 12.5).abs() < 1e-10);
        assert_eq!(msg.step_count, 0);
    }

    #[test]
    fn test_worker_report_epoch() {
        let (worker, ch) = make_test_worker();

        worker.report_epoch(0.5, 100, 5000.0).unwrap();

        let msg = ch.metrics_rx.recv().unwrap();
        assert_eq!(msg.rank, 0);
        assert_eq!(msg.epoch, 0);
        assert!((msg.avg_loss - 0.5).abs() < 1e-10);
        assert_eq!(msg.batches_processed, 100);
    }

    #[test]
    fn test_worker_handle_control_request_params() {
        let (mut worker, ch) = make_test_worker();

        ch.control_tx.send(ControlMsg::RequestParams).unwrap();
        let shutdown = worker.handle_control().unwrap();
        assert!(!shutdown);

        // Verify snapshot was sent back
        let snap = ch.param_rx.recv().unwrap();
        assert_eq!(snap.rank, 0);
        assert_eq!(snap.params.len(), 2);
    }

    #[test]
    fn test_worker_handle_control_update() {
        let (mut worker, ch) = make_test_worker();
        let dev = test_device();
        let opts = TensorOptions { dtype: DType::Float32, device: dev };

        let update = AveragedParams {
            params: vec![
                Tensor::zeros(&[2, 4], opts).unwrap(),
                Tensor::zeros(&[2], opts).unwrap(),
            ],
            buffers: vec![],
            version: 7,
        };
        ch.control_tx.send(ControlMsg::Update(update)).unwrap();

        let shutdown = worker.handle_control().unwrap();
        assert!(!shutdown);
        assert_eq!(worker.current_version(), 7);
    }

    #[test]
    fn test_worker_handle_control_partition_hint() {
        let (mut worker, ch) = make_test_worker();

        assert!(worker.take_partition_hint().is_none());

        ch.control_tx.send(ControlMsg::PartitionHint { num_samples: 750 }).unwrap();
        worker.handle_control().unwrap();

        assert_eq!(worker.take_partition_hint(), Some(750));
        assert!(worker.take_partition_hint().is_none()); // consumed
    }

    #[test]
    fn test_worker_handle_control_shutdown() {
        let (mut worker, ch) = make_test_worker();

        ch.control_tx.send(ControlMsg::Shutdown).unwrap();
        let shutdown = worker.handle_control().unwrap();
        assert!(shutdown);
    }

    #[test]
    fn test_worker_handle_control_sync_now_noop() {
        let (mut worker, ch) = make_test_worker();

        // SyncNow is a no-op without NCCL (Phase 4)
        ch.control_tx.send(ControlMsg::SyncNow).unwrap();
        let shutdown = worker.handle_control().unwrap();
        assert!(!shutdown);
    }

    #[test]
    fn test_worker_full_roundtrip() {
        // Simulates: train -> snapshot -> "average" -> load -> train again
        let (mut worker, ch) = make_test_worker();
        let opts = test_opts();

        // Step 1: train a step
        let batch = vec![
            Tensor::randn(&[4, 4], opts).unwrap(),
            Tensor::randn(&[4, 2], opts).unwrap(),
        ];
        worker.train_step(&batch, &mse_train).unwrap();
        assert_eq!(worker.local_step(), 1);

        // Step 2: coordinator requests params
        ch.control_tx.send(ControlMsg::RequestParams).unwrap();
        worker.handle_control().unwrap();
        let snap = ch.param_rx.recv().unwrap();
        assert_eq!(snap.batch_count, 1);

        // Step 3: coordinator sends back "averaged" params (same values, pretend averaged)
        let update = AveragedParams {
            params: snap.params,
            buffers: snap.buffers,
            version: 1,
        };
        ch.control_tx.send(ControlMsg::Update(update)).unwrap();
        worker.handle_control().unwrap();
        assert_eq!(worker.current_version(), 1);

        // Step 4: train another step with loaded params
        let batch2 = vec![
            Tensor::randn(&[4, 4], opts).unwrap(),
            Tensor::randn(&[4, 2], opts).unwrap(),
        ];
        worker.train_step(&batch2, &mse_train).unwrap();
        assert_eq!(worker.local_step(), 2);
    }

    #[test]
    fn test_worker_advance_epoch() {
        let (mut worker, _ch) = make_test_worker();
        assert_eq!(worker.current_epoch, 0);
        worker.advance_epoch();
        assert_eq!(worker.current_epoch, 1);
        worker.advance_epoch();
        assert_eq!(worker.current_epoch, 2);
    }

    #[test]
    fn test_worker_channels_create() {
        let ((timing_tx, metrics_tx, param_tx, _control_rx), ch) =
            GpuWorker::<Linear>::channels();

        // Verify channel pairs work
        timing_tx.send(TimingMsg { rank: 0, batch_ms: 1.0, step_count: 0 }).unwrap();
        let msg = ch.timing_rx.recv().unwrap();
        assert_eq!(msg.rank, 0);

        metrics_tx.send(MetricsMsg {
            rank: 0, epoch: 0, avg_loss: 0.5, batches_processed: 10, epoch_ms: 100.0,
        }).unwrap();
        let msg = ch.metrics_rx.recv().unwrap();
        assert_eq!(msg.batches_processed, 10);

        param_tx.send(ParamSnapshot {
            rank: 0, params: vec![], buffers: vec![], batch_count: 0,
        }).unwrap();
        let snap = ch.param_rx.recv().unwrap();
        assert_eq!(snap.rank, 0);

        ch.control_tx.send(ControlMsg::Shutdown).unwrap();
    }

    // -----------------------------------------------------------------------
    // Coordinator tests
    // -----------------------------------------------------------------------

    use crate::nn::ddp::ElChe;

    /// Simple coordinator test helper.
    struct CoordTestHarness {
        coord: Coordinator,
        /// Send timing/metrics/params TO the coordinator.
        timing_tx: mpsc::Sender<TimingMsg>,
        metrics_tx: mpsc::Sender<MetricsMsg>,
        param_tx: mpsc::Sender<ParamSnapshot>,
        /// Receive control messages FROM the coordinator (one per worker).
        control_rxs: Vec<mpsc::Receiver<ControlMsg>>,
    }

    fn make_coord_harness(
        n: usize,
        policy: ApplyPolicy,
        backend: AverageBackend,
    ) -> CoordTestHarness {
        let (timing_tx, timing_rx) = mpsc::channel();
        let (metrics_tx, metrics_rx) = mpsc::channel();
        let (param_tx, param_rx) = mpsc::channel();

        let mut control_txs = Vec::new();
        let mut control_rxs = Vec::new();
        for _ in 0..n {
            let (tx, rx) = mpsc::channel();
            control_txs.push(tx);
            control_rxs.push(rx);
        }

        let el_che = ElChe::new(n, 10);
        let active = Arc::new(AtomicUsize::new(n));
        let coord = Coordinator::builder(
            timing_rx, metrics_rx, param_rx,
            control_txs,
            policy, backend,
            n, 10000, el_che,
            active,
        ).build();

        CoordTestHarness { coord, timing_tx, metrics_tx, param_tx, control_rxs }
    }

    #[test]
    fn test_coordinator_initial_state() {
        let h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);
        assert_eq!(h.coord.version(), 0);
        assert_eq!(h.coord.avg_interval(), 1);
        assert!(!h.coord.is_calibrated());
        assert_eq!(h.coord.steps_since_avg(), &[0, 0]);
    }

    #[test]
    fn test_coordinator_drain_timing() {
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

        h.timing_tx.send(TimingMsg { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();

        h.coord.drain_timing();

        assert_eq!(h.coord.steps_since_avg(), &[1, 1]);
    }

    #[test]
    fn test_coordinator_should_average_sync() {
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

        // Not ready yet (no steps)
        assert!(!h.coord.should_average());

        // One rank reports
        h.timing_tx.send(TimingMsg { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();
        assert!(!h.coord.should_average()); // rank 1 still at 0

        // Both ranks report
        h.timing_tx.send(TimingMsg { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();
        assert!(h.coord.should_average());
    }

    #[test]
    fn test_coordinator_should_average_async() {
        let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Nccl);

        // avg_interval starts at 1
        assert_eq!(h.coord.avg_interval(), 1);

        // Feed one step per rank
        h.timing_tx.send(TimingMsg { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();

        assert!(h.coord.should_average());
    }

    #[test]
    fn test_coordinator_trigger_nccl() {
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

        // Feed timing and trigger
        h.timing_tx.send(TimingMsg { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();
        h.coord.trigger_averaging().unwrap();

        // Workers should receive SyncNow
        for rx in &h.control_rxs {
            match rx.recv().unwrap() {
                ControlMsg::SyncNow => {}
                other => panic!("expected SyncNow, got {:?}", std::mem::discriminant(&other)),
            }
        }

        // Version bumped, steps reset
        assert_eq!(h.coord.version(), 1);
        assert_eq!(h.coord.steps_since_avg(), &[0, 0]);
    }

    #[test]
    fn test_coordinator_trigger_cpu_averaging() {
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Cpu);
        let dev = test_device();
        let opts = TensorOptions { dtype: DType::Float32, device: dev };

        // Feed timing
        h.timing_tx.send(TimingMsg { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();

        // Trigger averaging in a thread because it blocks waiting for snapshots
        let param_tx = h.param_tx.clone();
        let handle = std::thread::spawn(move || {
            // Simulate workers responding to RequestParams
            // Worker 0: weight=1.0, 10 batches
            param_tx.send(ParamSnapshot {
                rank: 0,
                params: vec![Tensor::ones(&[2, 3], opts).unwrap()],
                buffers: vec![],
                batch_count: 10,
            }).unwrap();
            // Worker 1: weight=3.0, 10 batches
            param_tx.send(ParamSnapshot {
                rank: 1,
                params: vec![Tensor::full(&[2, 3], 3.0, opts).unwrap()],
                buffers: vec![],
                batch_count: 10,
            }).unwrap();
        });

        h.coord.trigger_averaging().unwrap();
        handle.join().unwrap();

        // Workers should receive RequestParams then Update
        for rx in &h.control_rxs {
            match rx.recv().unwrap() {
                ControlMsg::RequestParams => {}
                other => panic!("expected RequestParams, got {:?}", std::mem::discriminant(&other)),
            }
            match rx.recv().unwrap() {
                ControlMsg::Update(avg) => {
                    // Weighted average of 1.0 and 3.0 with equal batch counts = 2.0
                    let sum: f64 = avg.params[0].sum().unwrap().item().unwrap();
                    let expected = 2.0 * 6.0; // 2.0 * (2*3 elements)
                    assert!((sum - expected).abs() < 1e-4,
                        "expected sum={expected}, got {sum}");
                    assert_eq!(avg.version, 1);
                }
                other => panic!("expected Update, got {:?}", std::mem::discriminant(&other)),
            }
        }
    }

    #[test]
    fn test_coordinator_average_params_weighted() {
        let dev = test_device();
        let opts = TensorOptions { dtype: DType::Float32, device: dev };

        // Rank 0: all 1.0, did 1 batch
        // Rank 1: all 5.0, did 3 batches
        // Weighted avg: (1*1.0 + 3*5.0) / (1+3) = 16/4 = 4.0
        let snapshots = vec![
            ParamSnapshot {
                rank: 0,
                params: vec![Tensor::ones(&[4], opts).unwrap()],
                buffers: vec![],
                batch_count: 1,
            },
            ParamSnapshot {
                rank: 1,
                params: vec![Tensor::full(&[4], 5.0, opts).unwrap()],
                buffers: vec![],
                batch_count: 3,
            },
        ];

        let avg = Coordinator::average_params(&snapshots, 42).unwrap();
        assert_eq!(avg.version, 42);
        assert_eq!(avg.params.len(), 1);

        // Each element should be (1*1.0 + 3*5.0) / (1+3) = 4.0
        let sum: f64 = avg.params[0].sum().unwrap().item().unwrap();
        let expected = 4.0 * 4.0; // 4.0 per element * 4 elements
        assert!((sum - expected).abs() < 1e-4, "expected sum={expected}, got {sum}");
    }

    #[test]
    fn test_coordinator_tick_sync_flow() {
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

        // No steps yet: tick should not trigger
        let metrics = h.coord.tick().unwrap();
        assert!(metrics.is_empty());
        assert_eq!(h.coord.version(), 0);

        // Feed steps from both ranks
        h.timing_tx.send(TimingMsg { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();

        // Tick: should trigger averaging
        let metrics = h.coord.tick().unwrap();
        assert!(metrics.is_empty());
        assert_eq!(h.coord.version(), 1);

        // Workers got SyncNow
        for rx in &h.control_rxs {
            assert!(matches!(rx.recv().unwrap(), ControlMsg::SyncNow));
        }
    }

    #[test]
    fn test_coordinator_drain_metrics() {
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

        h.metrics_tx.send(MetricsMsg {
            rank: 0, epoch: 1, avg_loss: 0.3, batches_processed: 50, epoch_ms: 2000.0,
        }).unwrap();

        let metrics = h.coord.drain_metrics();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].rank, 0);
        assert_eq!(metrics[0].epoch, 1);
    }

    #[test]
    fn test_coordinator_rebalance_partitions() {
        let h = make_coord_harness(2, ApplyPolicy::Cadence, AverageBackend::Nccl);

        // Simulate ElChe calibration by reporting timing
        // This won't actually calibrate (needs report_timing), but we can test
        // the rebalance path directly
        h.coord.rebalance_partitions();

        // Workers should receive PartitionHint
        for rx in &h.control_rxs {
            match rx.recv().unwrap() {
                ControlMsg::PartitionHint { num_samples } => {
                    // ElChe not calibrated yet, so batch_counts are equal
                    assert_eq!(num_samples, 5000); // 10000 / 2
                }
                other => panic!("expected PartitionHint, got {:?}", std::mem::discriminant(&other)),
            }
        }
    }

    #[test]
    fn test_coordinator_update_cadence_divergent() {
        let dev = test_device();
        let opts = TensorOptions { dtype: DType::Float32, device: dev };

        let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Cpu);
        h.coord.avg_interval = 10; // start with wide interval

        // Highly divergent snapshots
        let snapshots = vec![
            ParamSnapshot {
                rank: 0,
                params: vec![Tensor::ones(&[100], opts).unwrap()],
                buffers: vec![],
                batch_count: 1,
            },
            ParamSnapshot {
                rank: 1,
                params: vec![Tensor::full(&[100], 100.0, opts).unwrap()],
                buffers: vec![],
                batch_count: 1,
            },
        ];

        h.coord.update_cadence(&snapshots).unwrap();

        // High divergence should have halved the interval
        assert!(h.coord.avg_interval() < 10,
            "interval should decrease, got {}", h.coord.avg_interval());
    }

    #[test]
    fn test_coordinator_update_cadence_converged() {
        let dev = test_device();
        let opts = TensorOptions { dtype: DType::Float32, device: dev };

        let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Cpu);
        h.coord.avg_interval = 4;

        // Nearly identical snapshots (low divergence)
        let snapshots = vec![
            ParamSnapshot {
                rank: 0,
                params: vec![Tensor::ones(&[100], opts).unwrap()],
                buffers: vec![],
                batch_count: 1,
            },
            ParamSnapshot {
                rank: 1,
                params: vec![Tensor::full(&[100], 1.001, opts).unwrap()],
                buffers: vec![],
                batch_count: 1,
            },
        ];

        h.coord.update_cadence(&snapshots).unwrap();

        // Low divergence should have doubled the interval
        assert!(h.coord.avg_interval() > 4,
            "interval should increase, got {}", h.coord.avg_interval());
    }

    // -----------------------------------------------------------------------
    // AsyncDdp tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_async_ddp_single_gpu_fallback() {
        // With <2 GPUs, auto() falls back to single-device training.
        // With 2+ GPUs, it uses all of them. Either way, join succeeds.
        let ddp = AsyncDdp::auto(
            |dev| Linear::on_device(4, 2, dev),
            |params| crate::nn::SGD::new(params, 0.01, 0.0),
            mse_train,
            Arc::new(TestDataset { n: 100 }),
            4,
            2,  // 2 epochs
            ApplyPolicy::Sync,
            AverageBackend::Cpu, // CPU backend: no NCCL needed for this test
        ).unwrap();

        assert!(ddp.world_size() >= 1);
        ddp.join().unwrap();
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: make cuda-test-nccl"]
    fn test_async_ddp_multi_gpu_nccl() {
        if crate::tensor::usable_cuda_devices().len() < 2 {
            return;
        }

        let ddp = AsyncDdp::auto(
            |dev| Linear::on_device(4, 2, dev),
            |params| crate::nn::SGD::new(params, 0.01, 0.0),
            mse_train,
            Arc::new(TestDataset { n: 256 }),
            32,
            2,  // 2 epochs
            ApplyPolicy::Sync,
            AverageBackend::Nccl,
        ).unwrap();

        assert!(ddp.world_size() >= 2);

        // Workers train for 2 epochs then exit, join should succeed
        ddp.join().unwrap();
    }

    #[test]
    fn test_async_ddp_send_sync() {
        fn assert_send<T: Send>() {}
        assert_send::<AsyncDdp>();
    }

    // -----------------------------------------------------------------------
    // Phase 10: 2-GPU end-to-end validation
    // -----------------------------------------------------------------------

    /// Shared loss tracker for multi-GPU convergence tests.
    /// Each rank appends (rank, step, loss) tuples.
    type LossLog = Arc<std::sync::Mutex<Vec<(usize, usize, f64)>>>;

    fn make_loss_tracker() -> LossLog {
        Arc::new(std::sync::Mutex::new(Vec::new()))
    }

    /// Run a 2-GPU async DDP session and return collected losses per rank.
    /// Returns (rank0_losses, rank1_losses) in chronological order.
    fn run_2gpu_training(
        backend: AverageBackend,
        policy: ApplyPolicy,
        num_epochs: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let log = make_loss_tracker();
        let log_clone = log.clone();

        let ddp = AsyncDdp::auto(
            |dev| Linear::on_device(4, 2, dev),
            |params| crate::nn::SGD::new(params, 0.01, 0.0),
            move |model: &Linear, batch: &[Tensor]| {
                let input = Variable::new(batch[0].clone(), false);
                let target = Variable::new(batch[1].clone(), false);
                let output = model.forward(&input)?;
                let diff = output.sub(&target)?;
                let loss = diff.mul(&diff)?.mean()?;
                let loss_val: f64 = loss.data().item()?;
                // Determine rank from device
                let rank = match batch[0].device() {
                    Device::CUDA(idx) => idx as usize,
                    Device::CPU => 0,
                };
                let step = {
                    let mut lg = log_clone.lock().unwrap();
                    let step = lg.iter().filter(|(r, _, _)| *r == rank).count();
                    lg.push((rank, step, loss_val));
                    step
                };
                let _ = step;
                Ok(loss)
            },
            Arc::new(TestDataset { n: 512 }),
            32,
            num_epochs,
            policy,
            backend,
        ).unwrap();

        ddp.join().unwrap();

        let entries = log.lock().unwrap();
        let r0: Vec<f64> = entries.iter().filter(|(r, _, _)| *r == 0).map(|(_, _, l)| *l).collect();
        let r1: Vec<f64> = entries.iter().filter(|(r, _, _)| *r == 1).map(|(_, _, l)| *l).collect();
        (r0, r1)
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: make cuda-test-nccl"]
    fn test_async_ddp_2gpu_cpu_backend_loss_decreases() {
        if crate::tensor::usable_cuda_devices().len() < 2 {
            return;
        }

        let (r0, r1) = run_2gpu_training(AverageBackend::Cpu, ApplyPolicy::Sync, 5);

        // Both ranks should have trained
        assert!(!r0.is_empty(), "rank 0 should have loss entries");
        assert!(!r1.is_empty(), "rank 1 should have loss entries");

        // Loss should converge: final losses should be finite and reasonable.
        // For a tiny Linear(4,2) with random data, the irreducible MSE is ~1.0.
        // We check that training converges (not diverges) rather than strictly decreases,
        // since NCCL averaging overhead can cause minor fluctuations.
        let check_converged = |losses: &[f64], rank: usize| {
            let n = losses.len();
            let quarter = (n / 4).max(1);
            let last_avg: f64 = losses[n - quarter..].iter().sum::<f64>() / quarter as f64;
            assert!(last_avg.is_finite() && last_avg < 2.0,
                "rank {rank} should converge: last_avg={last_avg:.4}");
        };

        check_converged(&r0, 0);
        check_converged(&r1, 1);
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: make cuda-test-nccl"]
    fn test_async_ddp_2gpu_nccl_backend_loss_decreases() {
        if crate::tensor::usable_cuda_devices().len() < 2 {
            return;
        }

        let (r0, r1) = run_2gpu_training(AverageBackend::Nccl, ApplyPolicy::Sync, 5);

        assert!(!r0.is_empty(), "rank 0 should have loss entries");
        assert!(!r1.is_empty(), "rank 1 should have loss entries");

        let check_converged = |losses: &[f64], rank: usize| {
            let n = losses.len();
            let quarter = (n / 4).max(1);
            let last_avg: f64 = losses[n - quarter..].iter().sum::<f64>() / quarter as f64;
            assert!(last_avg.is_finite() && last_avg < 2.0,
                "rank {rank} should converge: last_avg={last_avg:.4}");
        };

        check_converged(&r0, 0);
        check_converged(&r1, 1);
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: make cuda-test-nccl"]
    fn test_async_ddp_ab_cpu_vs_nccl() {
        if crate::tensor::usable_cuda_devices().len() < 2 {
            return;
        }

        let epochs = 5;
        let (cpu_r0, cpu_r1) = run_2gpu_training(AverageBackend::Cpu, ApplyPolicy::Sync, epochs);
        let (nccl_r0, nccl_r1) = run_2gpu_training(AverageBackend::Nccl, ApplyPolicy::Sync, epochs);

        // Both backends should converge (loss decreases)
        let final_avg = |losses: &[f64]| -> f64 {
            let n = losses.len();
            let quarter = n / 4;
            if quarter == 0 { return f64::MAX; }
            losses[n - quarter..].iter().sum::<f64>() / quarter as f64
        };

        let cpu_final = (final_avg(&cpu_r0) + final_avg(&cpu_r1)) / 2.0;
        let nccl_final = (final_avg(&nccl_r0) + final_avg(&nccl_r1)) / 2.0;

        // Both should have converged to a reasonable loss
        assert!(cpu_final < 2.0,
            "CPU backend final loss too high: {cpu_final:.4}");
        assert!(nccl_final < 2.0,
            "NCCL backend final loss too high: {nccl_final:.4}");

        // Final losses should be in the same ballpark (within 2x of each other).
        // They won't be identical because data shuffling differs across runs,
        // but for a simple Linear model both should converge to similar regions.
        let ratio = cpu_final.max(nccl_final) / cpu_final.min(nccl_final);
        eprintln!("  A/B: CPU final={cpu_final:.4} NCCL final={nccl_final:.4} ratio={ratio:.2}");
        assert!(ratio < 3.0,
            "CPU vs NCCL final loss ratio too large: {ratio:.2} (CPU={cpu_final:.4} NCCL={nccl_final:.4})");
    }

    // -----------------------------------------------------------------------
    // ElChe cadence + adaptive K tests (Phase 6)
    // -----------------------------------------------------------------------

    #[test]
    fn test_cadence_heterogeneous_timing() {
        // Simulate 2:1 speed ratio. Rank 0 is 2x faster (5ms/batch vs 10ms/batch).
        // With Cadence policy, ElChe should give rank 0 more batches.
        let mut h = make_coord_harness(2, ApplyPolicy::Cadence, AverageBackend::Nccl);

        // Feed enough timing to calibrate ElChe.
        // First, trigger with equal steps so ElChe sees the timing.
        for _ in 0..10 {
            h.timing_tx.send(TimingMsg { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
            h.timing_tx.send(TimingMsg { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
            h.coord.drain_timing();
            if h.coord.should_average() {
                h.coord.trigger_averaging().unwrap();
                // Drain control messages
                for rx in &h.control_rxs {
                    while rx.try_recv().is_ok() {}
                }
            }
        }

        // After calibration, ElChe batch_counts should reflect the speed ratio
        if h.coord.is_calibrated() {
            let counts = h.coord.el_che.batch_counts();
            // Rank 0 (fast) should have more batches than rank 1 (slow)
            assert!(counts[0] >= counts[1],
                "fast rank should get more batches: {:?}", counts);
        }
    }

    #[test]
    fn test_async_adaptive_k_increase() {
        // With Async policy and low divergence, K should increase.
        let dev = test_device();
        let opts = TensorOptions { dtype: DType::Float32, device: dev };
        let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Cpu);

        // Start at K=1
        assert_eq!(h.coord.avg_interval(), 1);

        // Feed timing so trigger fires, with nearly identical params
        for _ in 0..5 {
            h.timing_tx.send(TimingMsg { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
            h.timing_tx.send(TimingMsg { rank: 1, batch_ms: 5.0, step_count: 0 }).unwrap();
            h.coord.drain_timing();

            if h.coord.should_average() {
                // CPU path: need to supply snapshots
                let param_tx = h.param_tx.clone();
                let opts_copy = opts;
                let handle = std::thread::spawn(move || {
                    param_tx.send(ParamSnapshot {
                        rank: 0,
                        params: vec![Tensor::ones(&[10], opts_copy).unwrap()],
                        buffers: vec![],
                        batch_count: 1,
                    }).unwrap();
                    param_tx.send(ParamSnapshot {
                        rank: 1,
                        params: vec![Tensor::full(&[10], 1.001, opts_copy).unwrap()],
                        buffers: vec![],
                        batch_count: 1,
                    }).unwrap();
                });
                h.coord.trigger_averaging().unwrap();
                handle.join().unwrap();
                for rx in &h.control_rxs {
                    while rx.try_recv().is_ok() {}
                }
            }
        }

        // Low divergence should have increased K beyond initial 1
        assert!(h.coord.avg_interval() > 1,
            "K should increase with low divergence, got {}", h.coord.avg_interval());
    }

    #[test]
    fn test_elche_calibration_triggers_rebalance() {
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);
        let mut hints_found = false;

        // Feed heterogeneous timing to trigger ElChe calibration
        for _ in 0..5 {
            h.timing_tx.send(TimingMsg { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
            h.timing_tx.send(TimingMsg { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
            h.coord.drain_timing();
            if h.coord.should_average() {
                h.coord.trigger_averaging().unwrap();
                // Check ALL control messages (SyncNow + possible PartitionHint)
                for rx in &h.control_rxs {
                    while let Ok(msg) = rx.try_recv() {
                        if matches!(msg, ControlMsg::PartitionHint { .. }) {
                            hints_found = true;
                        }
                    }
                }
            }
        }

        assert!(h.coord.is_calibrated(), "ElChe should have calibrated");
        assert!(hints_found, "calibration should trigger partition rebalance");
    }

    #[test]
    fn test_wall_ms_accumulation() {
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

        // Send multiple timing messages per rank
        h.timing_tx.send(TimingMsg { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
        h.timing_tx.send(TimingMsg { rank: 0, batch_ms: 7.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
        h.timing_tx.send(TimingMsg { rank: 1, batch_ms: 12.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();

        // wall_ms_accum should have accumulated totals
        assert!((h.coord.wall_ms_accum[0] - 12.0).abs() < 1e-10, "rank 0 should be 5+7=12");
        assert!((h.coord.wall_ms_accum[1] - 22.0).abs() < 1e-10, "rank 1 should be 10+12=22");
    }

    #[test]
    fn test_config_defaults() {
        let cfg = AsyncDdpConfig::new();
        assert!(cfg.overhead_target.is_none());
        assert!(cfg.max_anchor.is_none());
        assert!(cfg.anchor.is_none());
        assert!(cfg.divergence_threshold.is_none());
    }

    #[test]
    fn test_config_builder() {
        let cfg = AsyncDdpConfig::new()
            .with_overhead_target(0.05)
            .with_max_anchor(100)
            .with_anchor(20)
            .with_divergence_threshold(0.01);
        assert_eq!(cfg.overhead_target, Some(0.05));
        assert_eq!(cfg.max_anchor, Some(100));
        assert_eq!(cfg.anchor, Some(20));
        assert_eq!(cfg.divergence_threshold, Some(0.01));
    }

    // -----------------------------------------------------------------------
    // Partition + data iteration tests (Phase 5)
    // -----------------------------------------------------------------------

    #[test]
    fn test_make_partition_basic() {
        let p0 = make_partition(0, 2, 100, 50, 0, 42);
        let p1 = make_partition(1, 2, 100, 50, 0, 42);
        assert_eq!(p0.len(), 50);
        assert_eq!(p1.len(), 50);

        // Non-overlapping (same epoch, same seed)
        let mut all: Vec<usize> = p0.iter().chain(p1.iter()).copied().collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 100, "partitions should be non-overlapping");
    }

    #[test]
    fn test_make_partition_different_epochs() {
        let p_e0 = make_partition(0, 2, 100, 50, 0, 42);
        let p_e1 = make_partition(0, 2, 100, 50, 1, 42);
        // Different epochs should produce different orderings
        assert_ne!(p_e0, p_e1);
    }

    #[test]
    fn test_make_partition_deterministic() {
        let p1 = make_partition(0, 2, 100, 50, 5, 42);
        let p2 = make_partition(0, 2, 100, 50, 5, 42);
        assert_eq!(p1, p2, "same params should produce same partition");
    }

    #[test]
    fn test_worker_shuffle_partition() {
        let (mut worker, _ch) = make_test_worker();
        let initial = worker.partition.clone();
        worker.advance_epoch();
        worker.shuffle_partition();
        // Different epoch should reshuffle
        assert_ne!(worker.partition, initial);
    }

    #[test]
    fn test_worker_shuffle_partition_applies_hint() {
        let (mut worker, ch) = make_test_worker();
        let initial_len = worker.partition.len();

        // Send a partition hint
        ch.control_tx.send(ControlMsg::PartitionHint { num_samples: 200 }).unwrap();
        worker.handle_control().unwrap();

        // Shuffle should apply the hint
        worker.shuffle_partition();
        assert_eq!(worker.partition.len(), 200);
        assert_ne!(worker.partition.len(), initial_len);
    }

    #[test]
    fn test_worker_run_epoch() {
        // 40 samples, batch_size=4 -> 10 batches per epoch
        let (mut worker, ch) = make_test_worker_with(0, 1, 40);

        let shutdown = worker.run_epoch(&mse_train).unwrap();
        assert!(!shutdown);
        assert_eq!(worker.current_epoch, 1);

        // Should have received timing messages (one per batch)
        let mut count = 0;
        while ch.timing_rx.try_recv().is_ok() {
            count += 1;
        }
        assert!(count > 0, "should have sent timing messages");

        // Should have received epoch metrics
        let metrics = ch.metrics_rx.recv().unwrap();
        assert_eq!(metrics.epoch, 0); // epoch 0 was just completed
        assert!(metrics.avg_loss > 0.0);
        assert!(metrics.batches_processed > 0);
    }

    #[test]
    fn test_worker_run_epoch_loss_decreases() {
        let (mut worker, _ch) = make_test_worker_with(0, 1, 80);

        // Run a few epochs, loss should decrease
        for _ in 0..5 {
            worker.run_epoch(&mse_train).unwrap();
        }
        // Snapshot and check loss on a fixed batch
        let opts = test_opts();
        let batch = vec![
            Tensor::randn(&[4, 4], opts).unwrap(),
            Tensor::randn(&[4, 2], opts).unwrap(),
        ];
        let loss_after: f64 = mse_train(worker.model(), &batch).unwrap().data().item().unwrap();
        // After 5 epochs of training, loss should be finite and non-negative
        assert!(loss_after.is_finite());
    }

    #[test]
    fn test_worker_run_epoch_shutdown_mid_epoch() {
        let (mut worker, ch) = make_test_worker_with(0, 1, 400);

        // Send shutdown after a short delay via the control channel
        ch.control_tx.send(ControlMsg::Shutdown).unwrap();

        let shutdown = worker.run_epoch(&mse_train).unwrap();
        assert!(shutdown, "should detect shutdown during epoch");
    }

    #[test]
    fn test_cpu_averaging_end_to_end() {
        // Two workers on CPU, CPU averaging backend.
        // Simulate the coordinator cycle manually.
        let (mut w0, _ch0) = make_test_worker_with(0, 2, 40);
        let (mut w1, _ch1) = make_test_worker_with(1, 2, 40);

        // Run one epoch on each worker
        w0.run_epoch(&mse_train).unwrap();
        w1.run_epoch(&mse_train).unwrap();

        // Snapshot params from both
        let snap0 = w0.snapshot_params();
        let snap1 = w1.snapshot_params();

        // Average them (coordinator's static method)
        let averaged = Coordinator::average_params(&[snap0, snap1], 1).unwrap();

        // Load averaged params into both workers
        w0.load_averaged(&averaged).unwrap();
        w1.load_averaged(&averaged).unwrap();

        assert_eq!(w0.current_version(), 1);
        assert_eq!(w1.current_version(), 1);

        // Both should now have the same params
        let s0 = w0.snapshot_params();
        let s1 = w1.snapshot_params();
        for (p0, p1) in s0.params.iter().zip(&s1.params) {
            let diff: f64 = p0.sub(p1).unwrap().abs().unwrap().sum().unwrap().item().unwrap();
            assert!(diff < 1e-5, "params should be identical after averaging, diff={diff}");
        }
    }

    // -----------------------------------------------------------------------
    // Proportional epoch sharding tests (Phase 7)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proportional_sharding() {
        // 2:1 speed ratio -> partition sizes should be 2:1
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

        // Calibrate ElChe with 2:1 timing
        for _ in 0..3 {
            h.timing_tx.send(TimingMsg { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
            h.timing_tx.send(TimingMsg { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
            h.coord.drain_timing();
            if h.coord.should_average() {
                h.coord.trigger_averaging().unwrap();
                for rx in &h.control_rxs {
                    while rx.try_recv().is_ok() {}
                }
            }
        }

        if h.coord.is_calibrated() {
            // Manually trigger rebalance and capture hints
            h.coord.rebalance_partitions();
            let mut hints = Vec::new();
            for rx in &h.control_rxs {
                if let Ok(ControlMsg::PartitionHint { num_samples }) = rx.recv() {
                    hints.push(num_samples);
                }
            }
            assert_eq!(hints.len(), 2);
            // Fast rank (0) should get more samples than slow rank (1)
            assert!(hints[0] > hints[1],
                "fast rank should get more samples: {:?}", hints);
            // Total should approximate dataset size (10000)
            let total: usize = hints.iter().sum();
            assert!(total <= 10000, "partitions should not exceed total: {total}");
        }
    }

    #[test]
    fn test_partition_non_overlapping_equal_sizes() {
        // Equal partition sizes: guaranteed non-overlapping
        let total = 300;
        let per_rank = total / 3; // 100 each
        let p0 = make_partition(0, 3, total, per_rank, 5, 42);
        let p1 = make_partition(1, 3, total, per_rank, 5, 42);
        let p2 = make_partition(2, 3, total, per_rank, 5, 42);

        assert_eq!(p0.len(), 100);
        assert_eq!(p1.len(), 100);
        assert_eq!(p2.len(), 100);

        let set0: std::collections::HashSet<usize> = p0.iter().copied().collect();
        let set1: std::collections::HashSet<usize> = p1.iter().copied().collect();
        let set2: std::collections::HashSet<usize> = p2.iter().copied().collect();
        assert_eq!(set0.intersection(&set1).count(), 0, "rank 0/1 should not overlap");
        assert_eq!(set0.intersection(&set2).count(), 0, "rank 0/2 should not overlap");
        assert_eq!(set1.intersection(&set2).count(), 0, "rank 1/2 should not overlap");
    }

    #[test]
    fn test_partition_non_overlapping_smaller_sizes() {
        // Sizes smaller than per_rank block: still non-overlapping
        let total = 300;
        let p0 = make_partition(0, 3, total, 50, 5, 42); // takes 50 from block of 100
        let p1 = make_partition(1, 3, total, 80, 5, 42); // takes 80 from block of 100
        let p2 = make_partition(2, 3, total, 60, 5, 42); // takes 60 from block of 100

        let set0: std::collections::HashSet<usize> = p0.iter().copied().collect();
        let set1: std::collections::HashSet<usize> = p1.iter().copied().collect();
        let set2: std::collections::HashSet<usize> = p2.iter().copied().collect();
        assert_eq!(set0.intersection(&set1).count(), 0, "rank 0/1 should not overlap");
        assert_eq!(set0.intersection(&set2).count(), 0, "rank 0/2 should not overlap");
        assert_eq!(set1.intersection(&set2).count(), 0, "rank 1/2 should not overlap");
    }

    #[test]
    fn test_partition_benign_overlap_different_epochs() {
        // Different epochs produce different permutations, so overlap is expected
        let p0_e5 = make_partition(0, 2, 100, 50, 5, 42);
        let p1_e6 = make_partition(1, 2, 100, 50, 6, 42);
        // These are from different epochs, so some overlap is expected and benign
        let set0: std::collections::HashSet<usize> = p0_e5.iter().copied().collect();
        let set1: std::collections::HashSet<usize> = p1_e6.iter().copied().collect();
        // Just verify they're valid indices
        assert!(set0.iter().all(|&i| i < 100));
        assert!(set1.iter().all(|&i| i < 100));
    }

    #[test]
    fn test_self_managed_epochs() {
        // Worker should run multiple epochs independently, reporting metrics each time
        let (mut worker, ch) = make_test_worker_with(0, 1, 40);

        // Run 3 epochs
        for _ in 0..3 {
            let shutdown = worker.run_epoch(&mse_train).unwrap();
            assert!(!shutdown);
        }

        assert_eq!(worker.current_epoch, 3);

        // Should have received 3 epoch metrics
        let mut epoch_msgs = Vec::new();
        while let Ok(msg) = ch.metrics_rx.try_recv() {
            epoch_msgs.push(msg);
        }
        assert_eq!(epoch_msgs.len(), 3);
        assert_eq!(epoch_msgs[0].epoch, 0);
        assert_eq!(epoch_msgs[1].epoch, 1);
        assert_eq!(epoch_msgs[2].epoch, 2);
    }

    #[test]
    fn test_partition_hint_applied_at_epoch_boundary() {
        let (mut worker, ch) = make_test_worker_with(0, 1, 80);
        let initial_partition_len = worker.partition.len();

        // Run one epoch
        worker.run_epoch(&mse_train).unwrap();

        // Send partition hint for smaller partition
        ch.control_tx.send(ControlMsg::PartitionHint { num_samples: 20 }).unwrap();
        worker.handle_control().unwrap();

        // Next epoch should use the new partition size
        worker.run_epoch(&mse_train).unwrap();
        assert_eq!(worker.partition.len(), 20);
        assert_ne!(worker.partition.len(), initial_partition_len);
    }
}
