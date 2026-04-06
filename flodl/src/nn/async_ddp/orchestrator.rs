//! AsyncDdp orchestrator: spawns GPU worker threads and a coordinator thread.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use crate::autograd::Variable;
use crate::data::BatchDataSet;
use crate::nn::nccl;
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{Device, Result, Tensor, TensorError};

use super::{
    ApplyPolicy, AverageBackend, AsyncDdpConfig, CheckpointFn,
    TrainedState, TimingMsg, WorkerConfig,
};
use super::worker::GpuWorker;
use super::coordinator::Coordinator;

// ---------------------------------------------------------------------------
// AsyncDdp orchestrator
// ---------------------------------------------------------------------------

/// Async DDP orchestrator: spawns GPU worker threads and a coordinator thread.
///
/// Each GPU runs its own training loop with a local optimizer. The coordinator
/// triggers periodic parameter averaging based on [`ApplyPolicy`] and
/// [`AverageBackend`]. Workers self-manage their epochs.
///
/// Use [`AsyncDdp::builder`] for the full configuration API, or [`AsyncDdp::auto`]
/// for a quick start with defaults.
///
/// # Quick start
///
/// ```ignore
/// use flodl::*;
///
/// let ddp = AsyncDdp::builder(model_factory, optim_factory, train_fn)
///     .dataset(dataset)
///     .batch_size(32)
///     .num_epochs(10)
///     .run()?;                // non-blocking: spawns threads, returns immediately
///
/// let state = ddp.join()?;   // blocks until training completes
/// // state.params / state.buffers contain the averaged trained tensors (CPU)
/// ```
///
/// # Recommended configurations
///
/// **Homogeneous GPUs (same model, same VRAM):**
/// `Sync` + `Nccl`. Equivalent to PyTorch DDP. Simplest, best convergence.
///
/// **Heterogeneous GPUs (mixed generations, different VRAM):**
/// `Cadence` + `Nccl`. ElChe assigns more batches to the fast GPU. Consider
/// [`with_max_batch_diff`](AsyncDdpConfig::with_max_batch_diff) as a safety guard.
///
/// **Maximum throughput (large models, expensive batches):**
/// `Async` + `Nccl`. Auto-tunes averaging interval. Monitor loss curves.
///
/// **Debugging / no NCCL:**
/// Any policy + `Cpu`. Works everywhere, logs averaging time for comparison.
///
/// # Single-GPU fallback
///
/// With fewer than 2 CUDA devices, training runs on the main thread with no
/// coordinator or averaging. The API is identical; [`join`](Self::join) returns
/// a [`TrainedState`] in both cases.
pub struct AsyncDdp {
    worker_handles: Vec<std::thread::JoinHandle<Result<()>>>,
    coordinator_handle: Option<std::thread::JoinHandle<Result<TrainedState>>>,
    devices: Vec<Device>,
    shutdown: Arc<AtomicBool>,
    /// Abort handles for NCCL communicators. Calling abort unblocks any
    /// worker stuck in an NCCL collective (e.g. AllReduce for a dead rank).
    nccl_abort_handles: Vec<Arc<nccl::NcclAbortHandle>>,
    /// For single-GPU mode: final state captured inline during run_single().
    final_state: Option<TrainedState>,
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
        Self::launch(
            model_factory, optim_factory, train_fn,
            dataset, batch_size, num_epochs,
            policy, backend, config, None,
        )
    }

    /// Internal launcher shared by `auto_with` and the builder.
    #[allow(clippy::too_many_arguments)]
    fn launch<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        policy: ApplyPolicy,
        backend: AverageBackend,
        config: AsyncDdpConfig,
        checkpoint_fn: Option<CheckpointFn<M>>,
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
                checkpoint_fn.as_ref().cloned(),
                config.checkpoint_every,
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
        let mut worker_final_txs = Vec::new();
        let mut coord_final_rxs = Vec::new();
        for _ in 0..world_size {
            let (tx, rx) = mpsc::channel();
            coord_control_txs.push(tx);
            worker_control_rxs.push(rx);
            let (ftx, frx) = mpsc::channel();
            worker_final_txs.push(ftx);
            coord_final_rxs.push(frx);
        }

        // Step 2b: Init NCCL comms from main thread, then split into per-rank comms.
        // CRITICAL: ncclCommInitRank from worker threads corrupts CUDA context on
        // heterogeneous GPUs. Always use NcclComms::new() + split() instead.
        // See NcclRankComm and NcclComms::split docs for details.
        let (mut rank_comms, nccl_abort_handles): (Vec<Option<_>>, Vec<_>) =
            if backend == AverageBackend::Nccl {
                let group = nccl::NcclComms::new(&devices)?;
                let comms = group.split()?;
                let aborts = comms.iter().map(|c| c.abort_handle()).collect();
                (comms.into_iter().map(Some).collect(), aborts)
            } else {
                ((0..world_size).map(|_| None).collect(), Vec::new())
            };

        // Step 3: Create ElChe with config knobs
        let anchor = config.anchor.unwrap_or(10);
        let mut el_che = crate::nn::ddp::ElChe::new(world_size, anchor);
        if let Some(target) = config.overhead_target {
            el_che = el_che.with_overhead_target(target);
        }
        if let Some(max) = config.max_anchor {
            el_che = el_che.with_max_anchor(max);
        }
        if let Some(diff) = config.max_batch_diff {
            el_che = el_che.with_max_batch_diff(diff);
        }

        // Step 4: Spawn coordinator thread
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_coord = shutdown.clone();
        let div_threshold = config.divergence_threshold;
        let ckpt_every = config.checkpoint_every;
        let snap_timeout = config.snapshot_timeout_secs;

        let coordinator_handle = std::thread::Builder::new()
            .name("async-ddp-coordinator".into())
            .spawn(move || -> Result<TrainedState> {
                let mut builder = Coordinator::builder(
                    timing_rx, metrics_rx, param_rx,
                    coord_final_rxs,
                    coord_control_txs,
                    policy, backend,
                    world_size, total_samples, el_che,
                )
                .snapshot_timeout_secs(snap_timeout);
                if let Some(dt) = div_threshold {
                    builder = builder.divergence_threshold(dt);
                }
                if let Some(n) = ckpt_every {
                    builder = builder.checkpoint_every(n);
                }
                let mut coord = builder.build();

                let poll_timeout = std::time::Duration::from_micros(100);
                let loop_err = loop {
                    if shutdown_coord.load(Ordering::Relaxed) {
                        break None;
                    }
                    // Block until a timing message arrives or timeout.
                    // Returns false when all senders disconnect (workers exited).
                    if !coord.drain_timing_blocking(poll_timeout) {
                        break None; // channel disconnected: all workers done
                    }
                    // All workers sent Exiting: no more collectives possible.
                    if coord.active_count == 0 {
                        break None;
                    }
                    coord.check_throttle();
                    if let Err(e) = coord.poll_cpu_averaging() {
                        shutdown_coord.store(true, Ordering::Relaxed);
                        break Some(e);
                    }
                    coord.drain_metrics();
                    if coord.should_average() {
                        // Final drain to catch last-second Exiting messages.
                        // Without this, a fast worker can send Exiting and
                        // destroy its NcclRankComm between our drain and the
                        // SyncNow send, leaving the slow worker's AllReduce
                        // with no partner.
                        coord.drain_timing();
                        if coord.should_average() {
                            if let Err(e) = coord.trigger_averaging() {
                                shutdown_coord.store(true, Ordering::Relaxed);
                                break Some(e);
                            }
                        }
                    }
                };

                // Ensure any in-progress CPU averaging is fully cleaned up
                // before we return. This joins the compute thread (if any)
                // so no detached thread holds GPU resources that could
                // interfere with subsequent NCCL init.
                coord.drain_avg_state();

                // Tell workers to exit their drain_until_shutdown loop.
                // Workers that finished training are blocked on recv()
                // waiting for this signal before dropping their NcclRankComm.
                coord.shutdown_workers();

                // Workers may take a moment to send final snapshots after
                // their last timing message. Short grace period.
                std::thread::sleep(std::time::Duration::from_millis(50));
                match coord.collect_final_state() {
                    Some(state) => Ok(state),
                    None => match loop_err {
                        Some(e) => Err(e),
                        None => Err(TensorError::new(
                            "coordinator: no final snapshots received from workers"
                        )),
                    },
                }
            })
            .map_err(|e| TensorError::new(&format!("failed to spawn coordinator: {e}")))?;

        // Step 5: Spawn GPU worker threads
        let model_factory = Arc::new(model_factory);
        let optim_factory = Arc::new(optim_factory);
        let train_fn = Arc::new(train_fn);
        // checkpoint_fn is already Arc, clone for each worker
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
            let t_tx_err = timing_tx_main.clone();
            let m_tx = metrics_tx_main.clone();
            let p_tx = param_tx_main.clone();
            let fp_tx = worker_final_txs.remove(0);
            let ckpt_fn = checkpoint_fn.clone();
            let shutdown_w = shutdown.clone();

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

                    // Inner closure so we can always run cleanup on exit.
                    let result = (|| -> Result<()> {
                        // Build worker inside this thread (model + optimizer are
                        // Rc-based, thread-local). NCCL comm was pre-initialized
                        // on the main thread via NcclComms::split() to avoid
                        // per-thread ncclCommInitRank CUDA context corruption.
                        let mut worker = GpuWorker::new(
                            &config,
                            |dev| (*mf)(dev),
                            |params| (*of)(params),
                            ds,
                            worker_nccl,
                            ckpt_fn,
                            t_tx,
                            m_tx,
                            p_tx,
                            fp_tx,
                            control_rx,
                        )?;

                        // Training loop: self-managed epochs with actual data
                        for _epoch in 0..num_epochs {
                            if shutdown_w.load(Ordering::Relaxed) {
                                break;
                            }
                            if worker.run_epoch(&*tf)? {
                                break; // Shutdown received
                            }
                        }

                        // Send final snapshot on the dedicated channel before exiting.
                        // Uses final_param_tx (not param_tx) to avoid racing with
                        // CPU averaging snapshot collection.
                        worker.send_final_snapshot();
                        worker.report_exiting();

                        // Keep handling control messages (especially SyncNow)
                        // until the coordinator sends Shutdown or closes the
                        // channel. Without this, the coordinator may send
                        // SyncNow after our Exiting but before processing it,
                        // leaving partner workers stuck in AllReduce.
                        worker.drain_until_shutdown();
                        Ok(())
                    })();

                    if result.is_err() {
                        // Ensure coordinator knows this rank is gone even on
                        // error (prevents NCCL deadlock on surviving workers).
                        // Send directly on the raw channel since the worker
                        // may not exist (e.g. GpuWorker::new failed).
                        let _ = t_tx_err.send(TimingMsg::Exiting { rank });
                        // Signal all siblings to stop so they don't block in
                        // an NCCL collective waiting for this dead rank.
                        shutdown_w.store(true, Ordering::Relaxed);
                    }

                    result
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
            nccl_abort_handles,
            final_state: None,
        })
    }

    /// Single-GPU fallback: run training on the main thread.
    ///
    /// No coordinator, no worker threads, no parameter averaging.
    /// Same training loop as multi-GPU workers.
    #[allow(clippy::too_many_arguments)]
    fn run_single<F, M, G, O, T>(
        model_factory: &F,
        optim_factory: &G,
        train_fn: &T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        device: Device,
        checkpoint_fn: Option<CheckpointFn<M>>,
        checkpoint_every: Option<usize>,
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
        let ((timing_tx, metrics_tx, param_tx, final_param_tx, control_rx), _channels) =
            GpuWorker::<M>::channels();

        let mut worker = GpuWorker::new(
            &config,
            model_factory,
            optim_factory,
            dataset,
            None, // no NCCL for single-GPU
            checkpoint_fn.clone(),
            timing_tx,
            metrics_tx,
            param_tx,
            final_param_tx,
            control_rx,
        )?;

        // Train directly on this thread
        for epoch in 0..num_epochs {
            worker.run_epoch(train_fn)?;
            // Single-GPU checkpoint: version = epoch number (monotonic)
            if let (Some(every), Some(f)) = (checkpoint_every, &checkpoint_fn) {
                if every > 0 && (epoch + 1) % every == 0 {
                    if let Err(e) = f((epoch + 1) as u64, worker.model()) {
                        eprintln!("  async-ddp: checkpoint failed (epoch {}): {e}", epoch + 1);
                    }
                }
            }
        }

        // Capture final state before dropping the worker
        let snap = worker.snapshot_params();
        let final_state = TrainedState {
            params: snap.params.iter()
                .map(|t| t.to_device(Device::CPU))
                .collect::<Result<Vec<_>>>()?,
            buffers: snap.buffers.iter()
                .map(|t| t.to_device(Device::CPU))
                .collect::<Result<Vec<_>>>()?,
        };

        Ok(AsyncDdp {
            worker_handles: Vec::new(),
            coordinator_handle: None,
            devices: vec![device],
            shutdown: Arc::new(AtomicBool::new(true)),
            nccl_abort_handles: Vec::new(),
            final_state: Some(final_state),
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

    /// Abort all NCCL communicators, unblocking any stuck collective ops.
    ///
    /// Called on error/shutdown to ensure no worker thread hangs forever
    /// in an AllReduce waiting for a dead rank.
    fn abort_nccl(&self) {
        for h in &self.nccl_abort_handles {
            let _ = h.abort();
        }
    }

    /// Wait for all training to complete and return the trained state.
    ///
    /// Workers run their `num_epochs` and exit naturally. Each sends a final
    /// parameter snapshot before terminating. The coordinator collects and
    /// averages these into a [`TrainedState`] (CPU tensors).
    ///
    /// For single-GPU mode, the state was captured inline during training.
    ///
    /// On partial failure (some workers died), returns the average of
    /// surviving workers' final snapshots. Returns an error only if
    /// all workers failed.
    pub fn join(mut self) -> Result<TrainedState> {
        // Single-GPU: state was captured in run_single()
        if let Some(state) = self.final_state.take() {
            return Ok(state);
        }

        // Join ALL workers, even if some fail. A failed worker already
        // set shutdown=true (see the error path in the spawn closure),
        // but we set it again on first error to cover panics.
        let mut first_err: Option<TensorError> = None;
        let handles: Vec<_> = self.worker_handles.drain(..).collect();

        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    self.shutdown.store(true, Ordering::Relaxed);
                    self.abort_nccl();
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
                Err(_) => {
                    self.shutdown.store(true, Ordering::Relaxed);
                    self.abort_nccl();
                    if first_err.is_none() {
                        first_err = Some(TensorError::new("worker thread panicked"));
                    }
                }
            }
        }

        // All workers done (or failed). Shut down coordinator.
        self.shutdown.store(true, Ordering::Relaxed);

        if let Some(h) = self.coordinator_handle.take() {
            match h.join() {
                Ok(Ok(state)) => return Ok(state),
                Ok(Err(e)) if first_err.is_none() => first_err = Some(e),
                Err(_) if first_err.is_none() => {
                    first_err = Some(TensorError::new("coordinator thread panicked"));
                }
                _ => {}
            }
        }

        Err(first_err.unwrap_or_else(|| TensorError::new("join: no trained state available")))
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

// ---------------------------------------------------------------------------
// AsyncDdpBuilder
// ---------------------------------------------------------------------------

/// Builder for configuring and launching async DDP training.
///
/// Created via [`AsyncDdp::builder`]. Required fields must be set before
/// calling [`run`](Self::run); missing fields produce a clear panic message.
///
/// # Example
///
/// ```ignore
/// use flodl::*;
///
/// let ddp = AsyncDdp::builder(
///     |dev| model_factory(dev),
///     |params| Adam::new(params, 0.001),
///     |model, batch| { /* return loss Variable */ },
/// )
/// .dataset(dataset)
/// .batch_size(32)
/// .num_epochs(10)
/// .policy(ApplyPolicy::Cadence)
/// .backend(AverageBackend::Nccl)
/// .run()?;
///
/// let state = ddp.join()?; // blocks until training completes
/// ```
pub struct AsyncDdpBuilder<F, M, G, O, T>
where
    F: Fn(Device) -> Result<M> + Send + Sync + 'static,
    M: Module + 'static,
    G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
    O: Optimizer + 'static,
    T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
{
    model_factory: F,
    optim_factory: G,
    train_fn: T,
    dataset: Option<Arc<dyn BatchDataSet>>,
    batch_size: Option<usize>,
    num_epochs: Option<usize>,
    policy: ApplyPolicy,
    backend: AverageBackend,
    config: AsyncDdpConfig,
    checkpoint_fn: Option<CheckpointFn<M>>,
    _phantom: PhantomData<(M, O)>,
}

impl<F, M, G, O, T> AsyncDdpBuilder<F, M, G, O, T>
where
    F: Fn(Device) -> Result<M> + Send + Sync + 'static,
    M: Module + 'static,
    G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
    O: Optimizer + 'static,
    T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
{
    /// Set the training dataset (required).
    pub fn dataset(mut self, dataset: Arc<dyn BatchDataSet>) -> Self {
        self.dataset = Some(dataset);
        self
    }

    /// Set the batch size (required).
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    /// Set the number of epochs (required).
    pub fn num_epochs(mut self, n: usize) -> Self {
        self.num_epochs = Some(n);
        self
    }

    /// Set the averaging policy. Default: [`ApplyPolicy::Cadence`].
    pub fn policy(mut self, policy: ApplyPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set the averaging backend. Default: [`AverageBackend::Nccl`].
    pub fn backend(mut self, backend: AverageBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the AllReduce overhead target (fraction of compute time).
    pub fn overhead_target(mut self, target: f64) -> Self {
        self.config = self.config.with_overhead_target(target);
        self
    }

    /// Set the maximum anchor count.
    pub fn max_anchor(mut self, max: usize) -> Self {
        self.config = self.config.with_max_anchor(max);
        self
    }

    /// Set the initial anchor count.
    pub fn anchor(mut self, anchor: usize) -> Self {
        self.config = self.config.with_anchor(anchor);
        self
    }

    /// Set the divergence threshold for Async mode.
    pub fn divergence_threshold(mut self, threshold: f64) -> Self {
        self.config = self.config.with_divergence_threshold(threshold);
        self
    }

    /// Set the maximum batch lead of fastest over slowest worker.
    /// `0` = strict lockstep.
    pub fn max_batch_diff(mut self, max: usize) -> Self {
        self.config = self.config.with_max_batch_diff(max);
        self
    }

    /// Save a checkpoint every N averaging events (multi-GPU) or N epochs (single-GPU).
    pub fn checkpoint_every(mut self, n: usize) -> Self {
        self.config = self.config.with_checkpoint_every(n);
        self
    }

    /// Set the checkpoint function called on rank 0 after averaging.
    ///
    /// Receives `(version, &model)`. Errors are logged but do not stop training.
    ///
    /// For Graph models: `.checkpoint_fn(|ver, g| g.save_checkpoint(&format!("ckpt_v{ver}.fdl")))`
    pub fn checkpoint_fn<C>(mut self, f: C) -> Self
    where
        C: Fn(u64, &M) -> Result<()> + Send + Sync + 'static,
    {
        self.checkpoint_fn = Some(Arc::new(f));
        self
    }

    /// Launch training. Non-blocking: spawns threads and returns immediately.
    ///
    /// Call [`AsyncDdp::join`] to block until training completes and retrieve
    /// the trained parameters and buffers.
    ///
    /// # Panics
    ///
    /// Panics if `dataset`, `batch_size`, or `num_epochs` were not set.
    pub fn run(self) -> Result<AsyncDdp> {
        let dataset = self.dataset.expect("AsyncDdpBuilder: dataset is required");
        let batch_size = self.batch_size.expect("AsyncDdpBuilder: batch_size is required");
        let num_epochs = self.num_epochs.expect("AsyncDdpBuilder: num_epochs is required");

        AsyncDdp::launch(
            self.model_factory,
            self.optim_factory,
            self.train_fn,
            dataset,
            batch_size,
            num_epochs,
            self.policy,
            self.backend,
            self.config,
            self.checkpoint_fn,
        )
    }
}

impl AsyncDdp {
    /// Create a builder for configuring async DDP training.
    ///
    /// The three required closures are provided here. Dataset, batch size,
    /// and epoch count must be set on the builder before calling [`run`](AsyncDdpBuilder::run).
    pub fn builder<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
    ) -> AsyncDdpBuilder<F, M, G, O, T>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        AsyncDdpBuilder {
            model_factory,
            optim_factory,
            train_fn,
            dataset: None,
            batch_size: None,
            num_epochs: None,
            policy: ApplyPolicy::Cadence,
            backend: AverageBackend::Nccl,
            config: AsyncDdpConfig::new(),
            checkpoint_fn: None,
            _phantom: PhantomData,
        }
    }
}

impl Drop for AsyncDdp {
    fn drop(&mut self) {
        // Signal shutdown if not already joined
        self.shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
        // Abort NCCL comms so workers stuck in collectives can unblock.
        self.abort_nccl();
    }
}
