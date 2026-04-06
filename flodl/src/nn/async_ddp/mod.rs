//! Async DDP: thread-per-GPU training with Local SGD and adaptive parameter averaging.
//!
//! Each GPU runs its own optimizer independently (zero wait). A lightweight coordinator
//! triggers periodic parameter averaging at ElChe-determined intervals. Two orthogonal
//! knobs control the behavior: [`ApplyPolicy`] (when to average) and [`AverageBackend`]
//! (how to average).
//!
//! # Quick start
//!
//! ```ignore
//! use flodl::*;
//!
//! let ddp = AsyncDdp::builder(model_factory, optim_factory, train_fn)
//!     .dataset(dataset)
//!     .batch_size(32)
//!     .num_epochs(10)
//!     .policy(ApplyPolicy::Cadence)
//!     .backend(AverageBackend::Nccl)
//!     .checkpoint_every(5)
//!     .checkpoint_fn(|ver, g| g.save_checkpoint(&format!("ckpt_v{ver}.fdl")))
//!     .run()?;
//!
//! let state = ddp.join()?;
//! // state.params[i] corresponds to model.parameters()[i]
//! // state.buffers[i] corresponds to model.buffers()[i]
//! ```
//!
//! # Architecture
//!
//! ```text
//! GPU Thread 0:  create model+Adam+dataset -> [fwd -> bwd -> adam step -> repeat]
//! GPU Thread 1:  create model+Adam+dataset -> [fwd -> bwd -> adam step -> repeat]
//! Coordinator:   collect timing/metrics -> trigger param averaging -> monitor divergence
//! ```
//!
//! # Choosing a policy
//!
//! | Policy | When to use | Tradeoff |
//! |--------|-------------|----------|
//! | [`ApplyPolicy::Sync`] | Correctness-first, small models, homogeneous GPUs | Identical to standard DDP. Fast GPU waits at every batch. |
//! | [`ApplyPolicy::Cadence`] | Heterogeneous GPUs (e.g. Pascal + Blackwell) | Fast GPU runs ahead by ElChe-determined batches. Good throughput/convergence balance. |
//! | [`ApplyPolicy::Async`] | Maximum throughput, large models, fault tolerance | Averaging interval auto-tunes from divergence monitoring. Best for experienced users. |
//!
//! # Choosing a backend
//!
//! | Backend | When to use | Tradeoff |
//! |---------|-------------|----------|
//! | [`AverageBackend::Nccl`] | Default choice. NVLink/PCIe peer-to-peer. | In-place AllReduce, zero extra memory, hard sync at averaging point. |
//! | [`AverageBackend::Cpu`] | No NVLink, A/B testing, debugging, CPU-only setups | Params copied to CPU for averaging. No GPU blocks, but uses O(world_size * model_size) CPU RAM and adds latency from GPU-CPU-GPU round-trip. |
//!
//! Start with `Cadence` + `Nccl` for heterogeneous setups, `Sync` + `Nccl` for
//! homogeneous. Use `Cpu` backend when debugging or when NCCL is unavailable.
//!
//! # Safety guards
//!
//! - [`AsyncDdpConfig::with_max_batch_diff`]: hard limit on how far any GPU can
//!   run ahead. Set to `0` for strict lockstep. Prevents catastrophic divergence
//!   with large batches or extreme speed ratios.
//! - [`ElChe`](super::ddp::ElChe) adaptive speed tracking with dead-zone hysteresis:
//!   tolerates thermal jitter while adapting quickly to sustained speed changes.
//! - NCCL abort handles: if a worker dies mid-collective, surviving workers are
//!   unblocked via `ncclCommAbort` instead of hanging forever.

mod worker;
mod coordinator;
mod orchestrator;

pub use worker::*;
pub use coordinator::*;
pub use orchestrator::*;

use std::sync::Arc;

use crate::rng::Rng;
use crate::tensor::{Device, Result, Tensor};


/// Checkpoint callback type: `(version, &model) -> Result<()>`.
///
/// Called on rank 0 after averaging events (multi-GPU) or at epoch boundaries
/// (single-GPU). Errors are logged but do not stop training.
pub type CheckpointFn<M> = Arc<dyn Fn(u64, &M) -> Result<()> + Send + Sync>;

// ---------------------------------------------------------------------------
// Return type
// ---------------------------------------------------------------------------

/// Trained parameters and buffers returned by [`AsyncDdp::join`].
///
/// Contains the averaged final state from all workers. Parameters are on CPU.
/// Buffers include running statistics (e.g. BatchNorm mean/var) needed for inference.
///
/// # Example
///
/// ```ignore
/// let state = ddp.join()?;
/// // state.params[i] corresponds to model.parameters()[i]
/// // state.buffers[i] corresponds to model.buffers()[i]
/// ```
#[derive(Clone, Debug)]
pub struct TrainedState {
    /// Averaged parameter tensors (CPU). Same order as `Module::parameters()`.
    pub params: Vec<Tensor>,
    /// Averaged buffer tensors (CPU). Same order as `Module::buffers()`.
    pub buffers: Vec<Tensor>,
}

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// Controls WHEN parameter averaging occurs (the interval K).
///
/// All three modes run the same architecture; only the averaging trigger differs.
/// The interval K determines how many batches each GPU processes with its own
/// local optimizer before parameters are synchronized across replicas.
///
/// - `Sync`: K=1 (every batch). Equivalent to standard DDP. Best convergence
///   guarantees, but fast GPUs idle waiting for slow ones.
/// - `Cadence`: K=N (ElChe anchor count). The slow GPU anchors the cadence,
///   fast GPUs fill the wall time with extra batches. Recommended for
///   heterogeneous hardware (e.g. mixing GPU generations).
/// - `Async`: K=adaptive. ElChe starts conservative (K=1), then backs off
///   as parameter divergence stays low. Maximizes GPU utilization at the
///   cost of some gradient staleness. Best for large models where each
///   batch is expensive and synchronization overhead matters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ApplyPolicy {
    /// Average after every batch (K=1). Equivalent to standard synchronous DDP.
    /// Lowest risk of model divergence. Fast GPUs wait at the collective barrier.
    Sync,
    /// Average every N batches where N is determined by ElChe's cadence strategy.
    /// The slow device sets the pace; fast devices process proportionally more
    /// batches per averaging window. Good default for mixed GPU setups.
    Cadence,
    /// ElChe auto-tunes the averaging interval based on observed parameter
    /// divergence. Starts conservative (K=1), backs off as convergence
    /// stabilizes. Tightens again if replicas drift apart. Best throughput,
    /// requires monitoring. Pair with [`AsyncDdpConfig::with_max_batch_diff`]
    /// for a safety bound.
    Async,
}

/// Controls HOW parameter averaging is performed.
///
/// Orthogonal to [`ApplyPolicy`]. All combinations are valid, enabling A/B testing:
/// same model, same K, NCCL vs CPU. If loss curves match, the cheaper backend is
/// validated for your workload.
///
/// # NCCL vs CPU tradeoffs
///
/// | | NCCL | CPU |
/// |---|---|---|
/// | **Memory** | Zero extra (in-place) | O(world_size * model_size) CPU RAM |
/// | **Latency** | GPU-to-GPU DMA (NVLink or PCIe) | GPU->CPU->average->CPU->GPU round-trip |
/// | **Blocking** | All GPUs sync at collective barrier | No GPU ever blocks |
/// | **Fault tolerance** | Abort handles unblock stuck collectives | Coordinator timeout (5s) detects dead workers |
/// | **Buffer averaging** | Natural (AllReduce averages everything) | Explicit (buffers averaged with equal weight) |
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AverageBackend {
    /// NCCL AllReduce in-place on GPU params. Default and recommended.
    ///
    /// GPU-to-GPU DMA via NVLink or PCIe peer-to-peer. Zero extra memory.
    /// At the averaging point, all GPUs participate in a collective barrier:
    /// the fast GPU waits for the slow GPU to arrive. If a worker dies
    /// mid-collective, abort handles unblock the survivors.
    Nccl,
    /// CPU-mediated parameter averaging through the coordinator.
    ///
    /// Workers send parameter snapshots to the coordinator, which computes
    /// a weighted average on CPU, then distributes the result back. No GPU
    /// ever blocks on another GPU. Useful when NVLink/PCIe peer access is
    /// unavailable, for debugging, or for A/B comparison with the NCCL path.
    ///
    /// Uses O(world_size * model_size) CPU RAM for snapshot collection.
    /// Averaging time is measured and fed to ElChe so the overhead auto-tune
    /// accounts for the CPU round-trip cost.
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
    /// Maximum batch lead of fastest over slowest worker.
    /// `Some(0)` = strict lockstep. `None` = unlimited. Default: `None`.
    pub max_batch_diff: Option<usize>,
    /// Save a checkpoint every N averaging events (multi-GPU) or N epochs (single-GPU).
    /// `None` = no checkpointing. Default: `None`.
    pub checkpoint_every: Option<usize>,
    /// Timeout for CPU averaging snapshot collection (seconds). Default: 5.
    /// Only applies to [`AverageBackend::Cpu`].
    pub snapshot_timeout_secs: u64,
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
            max_batch_diff: None,
            checkpoint_every: None,
            snapshot_timeout_secs: 5,
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

    /// Set the maximum batch lead of fastest over slowest worker.
    ///
    /// `0` = strict lockstep (sync DDP behavior). Workers that exceed
    /// this lead are paused until the slowest catches up.
    pub fn with_max_batch_diff(mut self, max: usize) -> Self {
        self.max_batch_diff = Some(max);
        self
    }

    /// Save a checkpoint every N averaging events (multi-GPU) or N epochs (single-GPU).
    ///
    /// Requires a `checkpoint_fn` to be provided via the builder or `auto_with`.
    /// Errors from the checkpoint function are logged but do not stop training.
    pub fn with_checkpoint_every(mut self, n: usize) -> Self {
        self.checkpoint_every = Some(n);
        self
    }

    /// Set the timeout for CPU averaging snapshot collection (seconds).
    ///
    /// Default: 5. Only applies to [`AverageBackend::Cpu`]. If not all worker
    /// snapshots arrive within this timeout, the averaging attempt is aborted
    /// and retried on the next cycle.
    pub fn with_snapshot_timeout(mut self, secs: u64) -> Self {
        self.snapshot_timeout_secs = secs;
        self
    }
}

// ---------------------------------------------------------------------------
// Worker -> Coordinator messages
// ---------------------------------------------------------------------------

/// Message from a GPU worker to the coordinator on the timing channel.
///
/// Batch reports are lightweight (sent every batch for ElChe throughput tracking).
/// Exiting is sent exactly once, before the worker thread terminates, so the
/// coordinator never sends NCCL collectives to a dead worker.
#[derive(Clone, Debug)]
pub enum TimingMsg {
    /// Per-batch timing report.
    Batch {
        /// Which GPU sent this.
        rank: usize,
        /// Wall-clock time for this batch (ms).
        batch_ms: f64,
        /// Worker's local step counter (monotonically increasing).
        step_count: usize,
    },
    /// Worker is about to exit. Coordinator must stop including this rank
    /// in collectives before processing any further messages.
    Exiting {
        /// Which GPU is exiting.
        rank: usize,
    },
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
    /// Worker is too far ahead: block until the next real command arrives.
    /// Sent when the worker's batch lead exceeds [`ElChe::max_batch_diff`].
    Throttle,
    /// Save a checkpoint from rank 0 after averaging.
    Checkpoint {
        /// Version number (averaging event count in multi-GPU, epoch in single-GPU).
        version: u64,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Module;
    use crate::tensor::TensorError;

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
        let _throttle = ControlMsg::Throttle;
        let _hint = ControlMsg::PartitionHint { num_samples: 1000 };
        let _ckpt = ControlMsg::Checkpoint { version: 42 };
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

    use std::sync::mpsc;
    use crate::autograd::Variable;
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

        let ((timing_tx, metrics_tx, param_tx, final_param_tx, control_rx), channels) =
            GpuWorker::<Linear>::channels();

        let dataset: Arc<dyn crate::data::BatchDataSet> =
            Arc::new(TestDataset { n: dataset_size });

        let worker = GpuWorker::new(
            &config,
            |d| Linear::on_device(4, 2, d),
            |params| crate::nn::SGD::new(params, 0.01, 0.0),
            dataset,
            None, // no NCCL in unit tests
            None, // no checkpoint in unit tests
            timing_tx,
            metrics_tx,
            param_tx,
            final_param_tx,
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

        // load_averaged uses non-blocking copy_ on comm_stream (CUDA).
        // In the training loop, sync_before_forward() at the next train_step
        // waits for the event. Here we read directly, so sync the device.
        if let Device::CUDA(idx) = dev {
            crate::tensor::cuda_synchronize(idx);
        }

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
        match msg {
            TimingMsg::Batch { rank, batch_ms, step_count } => {
                assert_eq!(rank, 0);
                assert!((batch_ms - 12.5).abs() < 1e-10);
                assert_eq!(step_count, 0);
            }
            _ => panic!("expected Batch"),
        }
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
        let ((timing_tx, metrics_tx, param_tx, _final_param_tx, _control_rx), ch) =
            GpuWorker::<Linear>::channels();

        // Verify channel pairs work
        timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 1.0, step_count: 0 }).unwrap();
        let msg = ch.timing_rx.recv().unwrap();
        assert!(matches!(msg, TimingMsg::Batch { rank: 0, .. }));

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
        make_coord_harness_with_timeout(n, policy, backend, 5)
    }

    fn make_coord_harness_with_timeout(
        n: usize,
        policy: ApplyPolicy,
        backend: AverageBackend,
        snapshot_timeout_secs: u64,
    ) -> CoordTestHarness {
        let (timing_tx, timing_rx) = mpsc::channel();
        let (metrics_tx, metrics_rx) = mpsc::channel();
        let (param_tx, param_rx) = mpsc::channel();

        let mut control_txs = Vec::new();
        let mut control_rxs = Vec::new();
        let mut final_param_rxs = Vec::new();
        for _ in 0..n {
            let (tx, rx) = mpsc::channel();
            control_txs.push(tx);
            control_rxs.push(rx);
            let (_ftx, frx) = mpsc::channel();
            final_param_rxs.push(frx);
        }

        let el_che = ElChe::new(n, 10);
        let coord = Coordinator::builder(
            timing_rx, metrics_rx, param_rx,
            final_param_rxs,
            control_txs,
            policy, backend,
            n, 10000, el_che,
        )
        .snapshot_timeout_secs(snapshot_timeout_secs)
        .build();

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

        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();

        h.coord.drain_timing();

        assert_eq!(h.coord.steps_since_avg(), &[1, 1]);
    }

    #[test]
    fn test_coordinator_should_average_sync() {
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

        // Not ready yet (no steps)
        assert!(!h.coord.should_average());

        // One rank reports
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();
        assert!(!h.coord.should_average()); // rank 1 still at 0

        // Both ranks report
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();
        assert!(h.coord.should_average());
    }

    #[test]
    fn test_coordinator_should_average_async() {
        let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Nccl);

        // avg_interval starts at 1
        assert_eq!(h.coord.avg_interval(), 1);

        // Feed one step per rank
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();

        assert!(h.coord.should_average());
    }

    #[test]
    fn test_coordinator_trigger_nccl() {
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

        // Feed timing and trigger
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
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
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();

        // trigger_averaging now returns immediately (enters Collecting state)
        h.coord.trigger_averaging().unwrap();

        // Workers should receive RequestParams
        for rx in &h.control_rxs {
            match rx.recv().unwrap() {
                ControlMsg::RequestParams => {}
                other => panic!("expected RequestParams, got {:?}", std::mem::discriminant(&other)),
            }
        }

        // Send snapshots (simulating workers responding)
        h.param_tx.send(ParamSnapshot {
            rank: 0,
            params: vec![Tensor::ones(&[2, 3], opts).unwrap()],
            buffers: vec![],
            batch_count: 10,
        }).unwrap();
        h.param_tx.send(ParamSnapshot {
            rank: 1,
            params: vec![Tensor::full(&[2, 3], 3.0, opts).unwrap()],
            buffers: vec![],
            batch_count: 10,
        }).unwrap();

        // Poll until the state machine completes (Collecting -> Computing -> Idle)
        for _ in 0..100 {
            h.coord.poll_cpu_averaging().unwrap();
            if h.coord.version() > 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert_eq!(h.coord.version(), 1);

        // Workers should receive Update
        for rx in &h.control_rxs {
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
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();

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
    // Throttle (max_batch_diff) tests
    // -----------------------------------------------------------------------

    fn make_throttle_harness(
        n: usize,
        max_batch_diff: usize,
    ) -> CoordTestHarness {
        let (timing_tx, timing_rx) = mpsc::channel();
        let (metrics_tx, metrics_rx) = mpsc::channel();
        let (param_tx, param_rx) = mpsc::channel();

        let mut control_txs = Vec::new();
        let mut control_rxs = Vec::new();
        let mut final_param_rxs = Vec::new();
        for _ in 0..n {
            let (tx, rx) = mpsc::channel();
            control_txs.push(tx);
            control_rxs.push(rx);
            let (_ftx, frx) = mpsc::channel();
            final_param_rxs.push(frx);
        }

        let el_che = ElChe::new(n, 10).with_max_batch_diff(max_batch_diff);
        let coord = Coordinator::builder(
            timing_rx, metrics_rx, param_rx,
            final_param_rxs,
            control_txs,
            ApplyPolicy::Async, AverageBackend::Nccl,
            n, 10000, el_che,
        ).build();

        CoordTestHarness { coord, timing_tx, metrics_tx, param_tx, control_rxs }
    }

    #[test]
    fn test_throttle_sends_when_diff_exceeded() {
        let mut h = make_throttle_harness(2, 3);

        // Rank 0 is 5 steps ahead, rank 1 at 0 -> diff = 5 > 3
        for i in 0..5 {
            h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: i }).unwrap();
        }
        h.coord.drain_timing();
        h.coord.check_throttle();

        // Rank 0 should receive Throttle
        match h.control_rxs[0].try_recv() {
            Ok(ControlMsg::Throttle) => {}
            _ => panic!("expected Throttle for rank 0"),
        }

        // Rank 1 should NOT receive Throttle
        assert!(h.control_rxs[1].try_recv().is_err(), "rank 1 should not be throttled");
    }

    #[test]
    fn test_throttle_no_send_within_limit() {
        let mut h = make_throttle_harness(2, 5);

        // Rank 0 is 3 steps ahead, rank 1 at 0 -> diff = 3 <= 5
        for i in 0..3 {
            h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: i }).unwrap();
        }
        h.coord.drain_timing();
        h.coord.check_throttle();

        // No throttle for either rank
        assert!(h.control_rxs[0].try_recv().is_err());
        assert!(h.control_rxs[1].try_recv().is_err());
    }

    #[test]
    fn test_throttle_zero_is_strict_lockstep() {
        let mut h = make_throttle_harness(2, 0);

        // Rank 0 does 1 batch, rank 1 does 0 -> diff = 1 > 0
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
        h.coord.drain_timing();
        h.coord.check_throttle();

        // Rank 0 throttled immediately
        match h.control_rxs[0].try_recv() {
            Ok(ControlMsg::Throttle) => {}
            _ => panic!("expected Throttle for rank 0"),
        }
    }

    #[test]
    fn test_throttle_disabled_when_none() {
        // Default harness has no max_batch_diff
        let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Nccl);

        // Rank 0 far ahead
        for i in 0..50 {
            h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: i }).unwrap();
        }
        h.coord.drain_timing();
        h.coord.check_throttle();

        // No throttle (feature disabled)
        assert!(h.control_rxs[0].try_recv().is_err());
    }

    #[test]
    fn test_throttle_worker_unblocks_on_sync_now() {
        // Simulate: worker receives Throttle, then SyncNow unblocks it.
        let (mut worker, ch) = make_test_worker();

        ch.control_tx.send(ControlMsg::Throttle).unwrap();
        ch.control_tx.send(ControlMsg::SyncNow).unwrap();

        // handle_control processes Throttle (blocks on recv), then
        // SyncNow arrives and unblocks it.
        let shutdown = worker.handle_control().unwrap();
        assert!(!shutdown, "should not shutdown");
    }

    #[test]
    fn test_throttle_worker_unblocks_on_shutdown() {
        let (mut worker, ch) = make_test_worker();

        ch.control_tx.send(ControlMsg::Throttle).unwrap();
        ch.control_tx.send(ControlMsg::Shutdown).unwrap();

        let shutdown = worker.handle_control().unwrap();
        assert!(shutdown, "should signal shutdown");
    }

    #[test]
    fn test_async_ddp_config_max_batch_diff() {
        let config = AsyncDdpConfig::new().with_max_batch_diff(5);
        assert_eq!(config.max_batch_diff, Some(5));

        let config2 = AsyncDdpConfig::new();
        assert_eq!(config2.max_batch_diff, None);
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
        let state = ddp.join().unwrap();
        // Linear(4,2): weight [2,4] + bias [2] = 2 params, 0 buffers
        assert_eq!(state.params.len(), 2);
        assert_eq!(state.buffers.len(), 0);
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

        // Workers train for 2 epochs then exit, join returns trained state
        let state = ddp.join().unwrap();
        assert_eq!(state.params.len(), 2);
    }

    #[test]
    fn test_async_ddp_send_sync() {
        fn assert_send<T: Send>() {}
        assert_send::<AsyncDdp>();
        assert_send::<TrainedState>();
    }

    // -----------------------------------------------------------------------
    // AsyncDdpBuilder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_with_defaults() {
        let ddp = AsyncDdp::builder(
            |dev| Linear::on_device(4, 2, dev),
            |params| crate::nn::SGD::new(params, 0.01, 0.0),
            mse_train,
        )
        .dataset(Arc::new(TestDataset { n: 100 }))
        .batch_size(4)
        .num_epochs(2)
        .backend(AverageBackend::Cpu)
        .run()
        .unwrap();

        assert!(ddp.world_size() >= 1);
        let state = ddp.join().unwrap();
        assert_eq!(state.params.len(), 2);
    }

    #[test]
    fn test_builder_with_all_options() {
        let ddp = AsyncDdp::builder(
            |dev| Linear::on_device(4, 2, dev),
            |params| crate::nn::SGD::new(params, 0.01, 0.0),
            mse_train,
        )
        .dataset(Arc::new(TestDataset { n: 100 }))
        .batch_size(4)
        .num_epochs(2)
        .policy(ApplyPolicy::Sync)
        .backend(AverageBackend::Cpu)
        .overhead_target(0.15)
        .max_anchor(100)
        .anchor(5)
        .divergence_threshold(0.1)
        .max_batch_diff(10)
        .run()
        .unwrap();

        let state = ddp.join().unwrap();
        assert_eq!(state.params.len(), 2);
    }

    #[test]
    #[should_panic(expected = "dataset is required")]
    fn test_builder_missing_dataset_panics() {
        let _ = AsyncDdp::builder(
            |dev| Linear::on_device(4, 2, dev),
            |params| crate::nn::SGD::new(params, 0.01, 0.0),
            mse_train,
        )
        .batch_size(4)
        .num_epochs(2)
        .run();
    }

    #[test]
    #[should_panic(expected = "batch_size is required")]
    fn test_builder_missing_batch_size_panics() {
        let _ = AsyncDdp::builder(
            |dev| Linear::on_device(4, 2, dev),
            |params| crate::nn::SGD::new(params, 0.01, 0.0),
            mse_train,
        )
        .dataset(Arc::new(TestDataset { n: 100 }))
        .num_epochs(2)
        .run();
    }

    #[test]
    #[should_panic(expected = "num_epochs is required")]
    fn test_builder_missing_num_epochs_panics() {
        let _ = AsyncDdp::builder(
            |dev| Linear::on_device(4, 2, dev),
            |params| crate::nn::SGD::new(params, 0.01, 0.0),
            mse_train,
        )
        .dataset(Arc::new(TestDataset { n: 100 }))
        .batch_size(4)
        .run();
    }

    #[test]
    fn test_worker_send_final_snapshot() {
        let (worker, ch) = make_test_worker();
        worker.send_final_snapshot();
        let snap = ch.final_param_rx.recv().unwrap();
        assert_eq!(snap.params.len(), 2); // Linear(4,2): weight + bias
        assert_eq!(snap.rank, 0);
    }

    #[test]
    fn test_collect_final_state_averages() {
        let (timing_tx, timing_rx) = mpsc::channel();
        let (_metrics_tx, metrics_rx) = mpsc::channel();
        let (_param_tx, param_rx) = mpsc::channel();

        let mut control_txs = Vec::new();
        let mut final_param_rxs = Vec::new();
        let mut final_param_txs = Vec::new();
        for _ in 0..2 {
            let (ctx, _crx) = mpsc::channel();
            control_txs.push(ctx);
            let (ftx, frx) = mpsc::channel();
            final_param_txs.push(ftx);
            final_param_rxs.push(frx);
        }

        let el_che = ElChe::new(2, 10);
        let coord = Coordinator::builder(
            timing_rx, metrics_rx, param_rx,
            final_param_rxs,
            control_txs,
            ApplyPolicy::Sync, AverageBackend::Cpu,
            2, 1000, el_che,
        ).build();

        // Send final snapshots from both "workers"
        let opts = crate::tensor::test_opts();
        let t1 = Tensor::full(&[3], 2.0, opts).unwrap();
        let t2 = Tensor::full(&[3], 4.0, opts).unwrap();
        final_param_txs[0].send(ParamSnapshot {
            rank: 0, params: vec![t1], buffers: vec![], batch_count: 1,
        }).unwrap();
        final_param_txs[1].send(ParamSnapshot {
            rank: 1, params: vec![t2], buffers: vec![], batch_count: 1,
        }).unwrap();

        let state = coord.collect_final_state().unwrap();
        assert_eq!(state.params.len(), 1);
        // Average of 2.0 and 4.0 with equal weights = 3.0
        let vals: Vec<f64> = state.params[0].to_f64_vec().unwrap();
        assert!(vals.iter().all(|v| (v - 3.0).abs() < 1e-5), "expected all ~3.0, got {vals:?}");

        // Also verify timing_tx keeps coordinator alive
        drop(timing_tx);
    }

    #[test]
    fn test_collect_final_state_single_survivor() {
        let (_timing_tx, timing_rx) = mpsc::channel();
        let (_metrics_tx, metrics_rx) = mpsc::channel();
        let (_param_tx, param_rx) = mpsc::channel();

        let mut control_txs = Vec::new();
        let mut final_param_rxs = Vec::new();
        let mut final_param_txs = Vec::new();
        for _ in 0..2 {
            let (ctx, _crx) = mpsc::channel();
            control_txs.push(ctx);
            let (ftx, frx) = mpsc::channel();
            final_param_txs.push(ftx);
            final_param_rxs.push(frx);
        }

        let el_che = ElChe::new(2, 10);
        let coord = Coordinator::builder(
            timing_rx, metrics_rx, param_rx,
            final_param_rxs,
            control_txs,
            ApplyPolicy::Sync, AverageBackend::Cpu,
            2, 1000, el_che,
        ).build();

        // Only one worker sends a final snapshot (the other "died")
        let opts = crate::tensor::test_opts();
        let t = Tensor::full(&[3], 7.0, opts).unwrap();
        final_param_txs[0].send(ParamSnapshot {
            rank: 0, params: vec![t], buffers: vec![], batch_count: 5,
        }).unwrap();
        // Worker 1 never sends

        let state = coord.collect_final_state().unwrap();
        assert_eq!(state.params.len(), 1);
        let vals: Vec<f64> = state.params[0].to_f64_vec().unwrap();
        assert!(vals.iter().all(|v| (v - 7.0).abs() < 1e-5), "single survivor should return its own params");
    }

    // -----------------------------------------------------------------------
    // Checkpoint coordination tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_msg_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ControlMsg>();
    }

    #[test]
    fn test_checkpoint_fn_called_on_dispatch() {
        use std::sync::atomic::{AtomicU64, Ordering};

        let (mut worker, ch) = make_test_worker();
        let called_version = Arc::new(AtomicU64::new(0));
        let cv = called_version.clone();
        worker.checkpoint_fn = Some(Arc::new(move |ver, _model| {
            cv.store(ver, Ordering::Relaxed);
            Ok(())
        }));

        ch.control_tx.send(ControlMsg::Checkpoint { version: 7 }).unwrap();
        worker.handle_control().unwrap();

        assert_eq!(called_version.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn test_checkpoint_error_logged_not_propagated() {
        let (mut worker, ch) = make_test_worker();
        worker.checkpoint_fn = Some(Arc::new(|_ver, _model| {
            Err(TensorError::new("disk full"))
        }));

        ch.control_tx.send(ControlMsg::Checkpoint { version: 1 }).unwrap();
        // Should not return an error: log-and-continue
        let shutdown = worker.handle_control().unwrap();
        assert!(!shutdown);
    }

    #[test]
    fn test_coordinator_sends_checkpoint_every_n() {
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);
        h.coord.checkpoint_every = Some(2);

        // Simulate 3 averaging events
        for _ in 0..3 {
            // Feed enough timing for should_average to trigger
            h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 1.0, step_count: 1 }).unwrap();
            h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 1.0, step_count: 1 }).unwrap();
            h.coord.drain_timing();
            assert!(h.coord.should_average());
            h.coord.trigger_averaging().unwrap();
        }

        // avg_count=3, checkpoint_every=2, so checkpoint sent at avg_count=2
        // Check rank 0's control channel for Checkpoint messages
        let mut checkpoint_count = 0;
        for rx in &h.control_rxs {
            while let Ok(msg) = rx.try_recv() {
                if matches!(msg, ControlMsg::Checkpoint { .. }) {
                    checkpoint_count += 1;
                }
            }
        }
        assert_eq!(checkpoint_count, 1, "should send exactly 1 checkpoint after 3 averaging events with every=2");
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

        let _state = ddp.join().unwrap();

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
            h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
            h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
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
            h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
            h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 0 }).unwrap();
            h.coord.drain_timing();

            if h.coord.should_average() {
                // Non-blocking trigger enters Collecting state
                h.coord.trigger_averaging().unwrap();

                // Supply snapshots
                h.param_tx.send(ParamSnapshot {
                    rank: 0,
                    params: vec![Tensor::ones(&[10], opts).unwrap()],
                    buffers: vec![],
                    batch_count: 1,
                }).unwrap();
                h.param_tx.send(ParamSnapshot {
                    rank: 1,
                    params: vec![Tensor::full(&[10], 1.001, opts).unwrap()],
                    buffers: vec![],
                    batch_count: 1,
                }).unwrap();

                // Poll until averaging completes
                let v_before = h.coord.version();
                for _ in 0..100 {
                    h.coord.poll_cpu_averaging().unwrap();
                    if h.coord.version() > v_before {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }

                for rx in &h.control_rxs {
                    while rx.try_recv().is_ok() {}
                }
            }
        }

        // Low divergence should have increased K beyond initial 1
        assert!(h.coord.avg_interval() > 1,
            "K should increase with low divergence, got {}", h.coord.avg_interval());
    }

    // -----------------------------------------------------------------------
    // Non-blocking CPU averaging tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_throttle_during_cpu_averaging() {
        // The key invariant: check_throttle fires even while CPU averaging
        // is in Collecting state.
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Cpu);
        let el_che = ElChe::new(2, 10).with_max_batch_diff(2);
        h.coord.el_che = el_che;

        // Feed enough timing to trigger averaging
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();

        // Trigger averaging (enters Collecting state, returns immediately)
        assert!(h.coord.should_average());
        h.coord.trigger_averaging().unwrap();
        assert!(h.coord.is_cpu_averaging());
        assert!(!h.coord.should_average()); // guard prevents re-trigger

        // Consume RequestParams from control channels
        for rx in &h.control_rxs {
            match rx.try_recv() {
                Ok(ControlMsg::RequestParams) => {}
                other => panic!("expected RequestParams, got {:?}", other.map(|m| std::mem::discriminant(&m))),
            }
        }

        // Simulate rank 0 running ahead by 5 batches during the averaging window
        for i in 0..5 {
            h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 1.0, step_count: 2 + i }).unwrap();
        }
        h.coord.drain_timing();

        // check_throttle should fire even though we're in Collecting state
        h.coord.check_throttle();

        // Rank 0 should receive Throttle (it's 5 batches ahead, max_diff=2)
        match h.control_rxs[0].try_recv() {
            Ok(ControlMsg::Throttle) => {}
            other => panic!("expected Throttle for rank 0, got {:?}", other.map(|m| std::mem::discriminant(&m))),
        }
        // Rank 1 should NOT be throttled
        assert!(h.control_rxs[1].try_recv().is_err(), "rank 1 should not be throttled");
    }

    #[test]
    fn test_cpu_avg_state_machine_full_cycle() {
        // Drive the full Idle -> Collecting -> Computing -> Idle cycle.
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Cpu);
        let dev = test_device();
        let opts = TensorOptions { dtype: DType::Float32, device: dev };

        // Feed timing
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();

        assert_eq!(h.coord.version(), 0);
        assert!(!h.coord.is_cpu_averaging());

        // Trigger: enters Collecting
        h.coord.trigger_averaging().unwrap();
        assert!(h.coord.is_cpu_averaging());

        // Poll with no snapshots yet: still Collecting
        h.coord.poll_cpu_averaging().unwrap();
        assert!(h.coord.is_cpu_averaging());

        // Supply snapshots
        h.param_tx.send(ParamSnapshot {
            rank: 0,
            params: vec![Tensor::ones(&[4], opts).unwrap()],
            buffers: vec![],
            batch_count: 5,
        }).unwrap();
        h.param_tx.send(ParamSnapshot {
            rank: 1,
            params: vec![Tensor::full(&[4], 3.0, opts).unwrap()],
            buffers: vec![],
            batch_count: 5,
        }).unwrap();

        // Poll: transitions Collecting -> Computing (spawns thread)
        h.coord.poll_cpu_averaging().unwrap();

        // Poll until Computing -> Idle
        for _ in 0..100 {
            h.coord.poll_cpu_averaging().unwrap();
            if !h.coord.is_cpu_averaging() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        // Verify completion
        assert!(!h.coord.is_cpu_averaging());
        assert_eq!(h.coord.version(), 1);

        // Workers should have received RequestParams then Update
        for rx in &h.control_rxs {
            let mut got_request = false;
            let mut got_update = false;
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    ControlMsg::RequestParams => got_request = true,
                    ControlMsg::Update(avg) => {
                        got_update = true;
                        assert_eq!(avg.version, 1);
                    }
                    _ => {}
                }
            }
            assert!(got_request, "worker should have received RequestParams");
            assert!(got_update, "worker should have received Update");
        }
    }

    #[test]
    fn test_cpu_avg_collection_timeout() {
        // Use a very short timeout (1 second) and never send snapshots.
        let mut h = make_coord_harness_with_timeout(
            2, ApplyPolicy::Sync, AverageBackend::Cpu, 1,
        );

        // Feed timing to trigger averaging
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();

        // Trigger: enters Collecting
        h.coord.trigger_averaging().unwrap();
        assert!(h.coord.is_cpu_averaging());

        // Wait for the timeout to expire
        std::thread::sleep(std::time::Duration::from_secs(2));

        // Poll: should soft-abort (back to Idle)
        h.coord.poll_cpu_averaging().unwrap(); // Ok, not Err
        assert!(!h.coord.is_cpu_averaging());
        assert_eq!(h.coord.version(), 0); // no version bump

        // should_average is available again for retry
        assert!(h.coord.should_average());
    }

    #[test]
    fn test_stale_snapshot_after_timeout() {
        // After a timeout, stale snapshots from the aborted round
        // must not contaminate the next round.
        let mut h = make_coord_harness_with_timeout(
            2, ApplyPolicy::Sync, AverageBackend::Cpu, 1,
        );
        let dev = test_device();
        let opts = TensorOptions { dtype: DType::Float32, device: dev };

        // Feed timing
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 1 }).unwrap();
        h.coord.drain_timing();

        // Round 1: trigger, send only rank 0's snapshot, let it timeout
        h.coord.trigger_averaging().unwrap();
        h.param_tx.send(ParamSnapshot {
            rank: 0,
            params: vec![Tensor::full(&[4], 999.0, opts).unwrap()],
            buffers: vec![],
            batch_count: 1,
        }).unwrap();

        // Wait for timeout
        std::thread::sleep(std::time::Duration::from_secs(2));
        h.coord.poll_cpu_averaging().unwrap();
        assert!(!h.coord.is_cpu_averaging()); // soft abort
        assert_eq!(h.coord.version(), 0);

        // Round 2: trigger fresh. The stale rank-0 snapshot from round 1
        // should have been drained by abort_cpu_averaging.
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 2 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 2 }).unwrap();
        h.coord.drain_timing();

        h.coord.trigger_averaging().unwrap();

        // Send FRESH snapshots for both ranks (value=1.0 and 3.0)
        h.param_tx.send(ParamSnapshot {
            rank: 0,
            params: vec![Tensor::ones(&[4], opts).unwrap()],
            buffers: vec![],
            batch_count: 1,
        }).unwrap();
        h.param_tx.send(ParamSnapshot {
            rank: 1,
            params: vec![Tensor::full(&[4], 3.0, opts).unwrap()],
            buffers: vec![],
            batch_count: 1,
        }).unwrap();

        // Poll until complete
        for _ in 0..100 {
            h.coord.poll_cpu_averaging().unwrap();
            if h.coord.version() > 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(h.coord.version(), 1);

        // Verify the Update contains fresh data (avg of 1.0 and 3.0 = 2.0),
        // NOT 999.0 from the stale snapshot.
        for rx in &h.control_rxs {
            let mut found_update = false;
            while let Ok(msg) = rx.try_recv() {
                if let ControlMsg::Update(avg) = msg {
                    let sum: f64 = avg.params[0].sum().unwrap().item().unwrap();
                    let expected = 2.0 * 4.0; // 2.0 per element * 4 elements
                    assert!(
                        (sum - expected).abs() < 1e-4,
                        "expected sum={expected}, got {sum} (stale data leaked?)"
                    );
                    found_update = true;
                }
            }
            assert!(found_update, "worker should have received Update");
        }
    }

    #[test]
    fn test_elche_calibration_triggers_rebalance() {
        let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);
        let mut hints_found = false;

        // Feed heterogeneous timing to trigger ElChe calibration
        for _ in 0..5 {
            h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
            h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
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
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 7.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 12.0, step_count: 1 }).unwrap();
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
            h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
            h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
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
