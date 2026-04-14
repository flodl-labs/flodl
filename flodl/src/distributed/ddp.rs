//! Distributed Data Parallel (DDP) for transparent multi-GPU training.
//!
//! `Ddp` is the single entry point for all multi-GPU training modes:
//!
//! **Setup mode** ([`Ddp::setup()`]): Distributes a Graph across GPUs. You write
//! the training loop. Works transparently with 1 or N GPUs.
//!
//! **Builder mode** ([`Ddp::builder()`]): Framework-managed training. Provide
//! factories and a train function, the framework handles threads, data pipeline,
//! epochs, and parameter averaging. Returns a [`DdpHandle`] to join.
//!
//! **Manual mode** ([`Ddp::wrap()`]): Low-level explicit control over gradient
//! sync and parameter broadcast for complex patterns (GAN, RL, progressive).
//!
//! # Setup mode (user owns the loop)
//!
//! ```ignore
//! Ddp::setup(&model, |dev| build_model(dev), |p| Adam::new(p, 0.001))?;
//!
//! // Training loop is identical for 1 or N GPUs:
//! for (x, y) in &train_loader {
//!     let out = model.forward(&x)?;
//!     let loss = cross_entropy_loss(&out, &y)?;
//!     loss.backward()?;
//!     model.step()?;
//! }
//! ```
//!
//! # Builder mode (framework owns the loop)
//!
//! ```ignore
//! let handle = Ddp::builder(model_factory, optim_factory, train_fn)
//!     .dataset(dataset)
//!     .batch_size(32)
//!     .num_epochs(10)
//!     .run()?;
//!
//! let state = handle.join()?;
//! ```
//!
//! # Manual mode
//!
//! ```ignore
//! let ddp = Ddp::wrap(&[&model0, &model1], &devices)?;
//! ddp.sync_params()?;
//! // ... custom forward/backward ...
//! ddp.all_reduce_gradients()?;
//! ```

use crate::autograd::Variable;
use crate::graph::Graph;
use crate::nn::{Buffer, Module, Optimizer, Parameter};
use super::cuda_event::CudaEvent;
use super::nccl::{NcclComms, ReduceOp};
use super::ddp_run::{DdpBuilder, DdpHandle};
pub use super::el_che::ElChe;
use crate::tensor::{Device, Result, Tensor, TensorError};


/// Shared lock for serializing NCCL communicator creation across test modules.
/// NCCL init is a collective operation that deadlocks if two tests try to
/// create communicators simultaneously.
#[cfg(test)]
pub(crate) static NCCL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Default number of steps before the first rebalance.
pub(crate) const DEFAULT_CALIBRATION_STEPS: usize = 10;

/// How often to re-evaluate chunk ratios after calibration.
pub(crate) const DEFAULT_REBALANCE_INTERVAL: usize = 50;

/// EMA smoothing factor for throughput tracking (higher = more reactive).
const EMA_ALPHA: f64 = 0.3;

/// Minimum ratio any device can receive (prevents starving a GPU entirely).
const MIN_CHUNK_RATIO: f64 = 0.05;

// ---------------------------------------------------------------------------
// Internal distributed state (held by Graph)
// ---------------------------------------------------------------------------

/// Internal distributed state held by Graph when `distribute()` is called.
pub(crate) struct DistributedState {
    /// Model replicas for ranks 1..N (rank 0 is the Graph itself).
    pub replicas: Vec<Box<dyn Module>>,
    /// NCCL communicators (one per device).
    pub comms: NcclComms,
    /// All devices including rank 0.
    pub devices: Vec<Device>,
    /// Per-replica optimizers indexed by rank (including rank 0).
    pub optimizers: Vec<Box<dyn Optimizer>>,
    /// Chunk ratios for auto-balancing (sum = 1.0). Default: equal.
    pub chunk_ratios: Vec<f64>,
    /// Parameters matched across replicas: param_groups\[param_idx\]\[rank\].
    pub param_groups: Vec<Vec<Variable>>,
    /// Buffers matched across replicas: buffer_groups\[buf_idx\]\[rank\].
    pub buffer_groups: Vec<Vec<Buffer>>,

    // -- Auto-balancer state --

    /// Per-rank forward timing events from last forward pass: (start, end).
    /// Set by forward_distributed(), read by step().
    pub last_timing: Option<Vec<(CudaEvent, CudaEvent)>>,
    /// Shard sizes from last forward pass (for throughput calculation).
    pub last_shard_sizes: Vec<i64>,
    /// EMA throughput per rank (samples/ms). Zero until first measurement.
    pub ema_throughput: Vec<f64>,
    /// Number of completed training steps.
    pub step_count: usize,
    /// Steps of equal-split calibration before first rebalance.
    pub calibration_steps: usize,
    /// Steps between ratio recalculations after calibration.
    pub rebalance_interval: usize,

    // -- El Che cadence (heterogeneous DDP) --

    /// El Che cadence strategy. When Some, Graph uses per-device multi-batch
    /// forward instead of per-batch scatter. When None, existing scatter path.
    pub el_che: Option<ElChe>,
    /// Per-rank batch counts from the last El Che forward pass.
    /// Set by forward_distributed_el_che(), read by step().
    pub last_el_che_counts: Vec<usize>,
    /// Wall-clock time at end of last El Che AllReduce.
    pub last_el_che_sync: Option<std::time::Instant>,
    /// Maximum gradient norm for per-rank clipping in El Che mode.
    pub max_grad_norm: Option<f64>,
    /// Optional system timeline for high-frequency profiling.
    pub timeline: Option<std::sync::Arc<crate::monitor::Timeline>>,
}

impl DistributedState {
    /// AllReduce-average gradients across all replicas.
    pub fn all_reduce_gradients(&self) -> Result<()> {
        for group in &self.param_groups {
            // Skip frozen parameters (no gradient on rank 0)
            if group[0].grad().is_none() {
                continue;
            }
            let grads: Vec<Tensor> = group
                .iter()
                .map(|v| v.grad().expect("gradient missing on replica"))
                .collect();
            let refs: Vec<&Tensor> = grads.iter().collect();
            self.comms.all_reduce(&refs, ReduceOp::Avg)?;
        }
        Ok(())
    }

    /// Broadcast buffers from rank 0 to all replicas (BatchNorm stats etc).
    pub fn sync_buffers(&self) -> Result<()> {
        for group in &self.buffer_groups {
            let tensors: Vec<Tensor> = group.iter().map(|b| b.get()).collect();
            let refs: Vec<&Tensor> = tensors.iter().collect();
            self.comms.broadcast(&refs, 0)?;
        }
        Ok(())
    }

    /// Broadcast parameters and buffers from rank 0 to all replicas.
    pub fn sync_params(&self) -> Result<()> {
        for group in &self.param_groups {
            let tensors: Vec<Tensor> = group.iter().map(|v| v.data()).collect();
            let refs: Vec<&Tensor> = tensors.iter().collect();
            self.comms.broadcast(&refs, 0)?;
        }
        self.sync_buffers()
    }

    /// Compute shard sizes from chunk ratios, guaranteeing they sum to batch_size.
    pub fn compute_shard_sizes(&self, batch_size: i64) -> Vec<i64> {
        let n = self.devices.len();
        let mut sizes = Vec::with_capacity(n);
        let mut remaining = batch_size;

        for i in 0..n {
            if i == n - 1 {
                // Last device gets whatever is left
                sizes.push(remaining);
            } else {
                let s = (batch_size as f64 * self.chunk_ratios[i]).round() as i64;
                let s = s.max(1).min(remaining - (n - i - 1) as i64); // leave at least 1 per remaining device
                sizes.push(s);
                remaining -= s;
            }
        }

        sizes
    }

    /// Number of devices.
    pub fn world_size(&self) -> usize {
        self.devices.len()
    }

    /// Whether chunk ratios are meaningfully unequal (need weighted gradients).
    pub fn is_balanced(&self) -> bool {
        let first = self.chunk_ratios[0];
        self.chunk_ratios.iter().all(|r| (r - first).abs() < 1e-6)
    }

    /// AllReduce gradients with weighted averaging for unequal shard sizes.
    ///
    /// Each replica's gradient is scaled by `(shard_size / batch_size)` before
    /// AllReduce Sum, which produces the correct mean gradient regardless of
    /// how the batch was split.
    pub fn weighted_all_reduce_gradients(&self, batch_size: i64) -> Result<()> {
        for group in &self.param_groups {
            if group[0].grad().is_none() {
                continue;
            }
            let grads: Vec<Tensor> = group
                .iter()
                .enumerate()
                .map(|(rank, v)| {
                    let g = v.grad().expect("gradient missing on replica");
                    let weight = self.last_shard_sizes[rank] as f64 / batch_size as f64;
                    g.mul_scalar_(weight).ok();
                    g
                })
                .collect();
            let refs: Vec<&Tensor> = grads.iter().collect();
            self.comms.all_reduce(&refs, ReduceOp::Sum)?;
        }
        Ok(())
    }

    /// Read timing from last forward pass, update EMA throughput, and
    /// rebalance chunk ratios if it's time.
    ///
    /// Called from Graph::step() after gradient sync. Returns true if
    /// chunk ratios were updated this step.
    pub fn update_balance(&mut self) -> Result<bool> {
        self.step_count += 1;

        // Read timing events (set by forward_distributed)
        if let Some(timing) = self.last_timing.take() {
            for (rank, (start, end)) in timing.iter().enumerate() {
                let ms = CudaEvent::elapsed_time(start, end)?;
                if ms > 0.0 && self.last_shard_sizes[rank] > 0 {
                    let throughput = self.last_shard_sizes[rank] as f64 / ms as f64;
                    if self.ema_throughput[rank] == 0.0 {
                        // First measurement: initialize directly
                        self.ema_throughput[rank] = throughput;
                    } else {
                        self.ema_throughput[rank] =
                            EMA_ALPHA * throughput + (1.0 - EMA_ALPHA) * self.ema_throughput[rank];
                    }
                }
            }
        }

        // Check if it's time to rebalance
        let should_rebalance = if self.step_count == self.calibration_steps {
            true
        } else if self.step_count > self.calibration_steps {
            (self.step_count - self.calibration_steps) % self.rebalance_interval == 0
        } else {
            false
        };

        if should_rebalance {
            self.rebalance();
            return Ok(true);
        }

        Ok(false)
    }

    /// Recompute chunk_ratios proportional to EMA throughput.
    fn rebalance(&mut self) {
        let total: f64 = self.ema_throughput.iter().sum();
        if total <= 0.0 {
            return; // no data yet
        }

        let n = self.devices.len();
        let min_total = MIN_CHUNK_RATIO * n as f64;

        // Compute raw proportional ratios
        let mut ratios: Vec<f64> = self.ema_throughput.iter().map(|t| t / total).collect();

        // Clamp: no device below MIN_CHUNK_RATIO
        let mut deficit = 0.0;
        let mut unclamped = 0;
        for r in &mut ratios {
            if *r < MIN_CHUNK_RATIO {
                deficit += MIN_CHUNK_RATIO - *r;
                *r = MIN_CHUNK_RATIO;
            } else {
                unclamped += 1;
            }
        }

        // Redistribute deficit from unclamped devices proportionally
        if deficit > 0.0 && unclamped > 0 {
            let unclamped_total: f64 = ratios
                .iter()
                .filter(|&&r| r > MIN_CHUNK_RATIO + 1e-9)
                .sum();
            if unclamped_total > min_total {
                for r in &mut ratios {
                    if *r > MIN_CHUNK_RATIO + 1e-9 {
                        *r -= deficit * (*r / unclamped_total);
                        *r = r.max(MIN_CHUNK_RATIO);
                    }
                }
            }
        }

        // Normalize to sum exactly to 1.0
        let sum: f64 = ratios.iter().sum();
        if sum > 0.0 {
            for r in &mut ratios {
                *r /= sum;
            }
        }

        self.chunk_ratios = ratios;
    }

    /// Configure El Che cadence from a [`DdpConfig`].
    ///
    /// Creates an internal ElChe when enabled (max_anchor != Some(0)),
    /// seeds chunk_ratios from speed_hint if provided.
    pub(crate) fn configure_el_che(&mut self, config: &DdpConfig) {
        let n = self.devices.len();
        if n < 2 {
            return;
        }

        // max_anchor = Some(0) → disabled (traditional DDP)
        if config.max_anchor == Some(0) {
            self.el_che = None;
            return;
        }

        // Build ElChe with sensible defaults
        let anchor = 10; // initial anchor, auto-tunes from timing
        let mut el_che = ElChe::new(n, anchor);

        if let Some(target) = config.overhead_target {
            el_che = el_che.with_overhead_target(target);
        }
        if let Some(max) = config.max_anchor {
            el_che = el_che.with_max_anchor(max);
        }
        if let Some((slow_rank, ratio)) = config.speed_hint {
            el_che = el_che.with_speed_ratio(slow_rank, ratio);
            // Also seed chunk_ratios for the existing auto-balancer
            self.apply_speed_hint(slow_rank, ratio);
        }

        self.el_che = Some(el_che);
        self.max_grad_norm = config.max_grad_norm;
    }

    /// Seed chunk_ratios from a speed hint.
    fn apply_speed_hint(&mut self, slow_rank: usize, ratio: f64) {
        let n = self.devices.len();
        if slow_rank >= n {
            return;
        }
        let ratio = ratio.max(1.0);
        let mut weights = vec![ratio; n];
        weights[slow_rank] = 1.0;
        let total: f64 = weights.iter().sum();
        self.chunk_ratios = weights.iter().map(|w| w / total).collect();
    }
}

// ---------------------------------------------------------------------------
// Manual DDP coordinator
// ---------------------------------------------------------------------------

/// Manual DDP coordinator for multi-GPU gradient sync.
///
/// For complex training patterns (GAN, RL, progressive) where transparent
/// Graph-level DDP doesn't fit. Provides explicit control over parameter
/// broadcast and gradient averaging.
///
/// For standard training, use [`crate::graph::Graph::distribute`] instead.
pub struct Ddp {
    comms: NcclComms,
    devices: Vec<Device>,
    param_groups: Vec<Vec<Variable>>,
    buffer_groups: Vec<Vec<Buffer>>,
}

impl Ddp {
    /// Wrap pre-created model replicas for manual DDP control.
    ///
    /// Models must have identical architecture (same parameter count/shapes).
    /// Each model should already reside on its target device.
    pub fn wrap(models: &[&dyn Module], devices: &[Device]) -> Result<Self> {
        if models.len() < 2 {
            return Err(TensorError::new("Ddp::wrap requires at least 2 models"));
        }
        if models.len() != devices.len() {
            return Err(TensorError::new(
                "Ddp::wrap: model count must match device count",
            ));
        }

        let comms = NcclComms::new(devices)?;

        // Match parameters across models
        let all_params: Vec<Vec<Parameter>> =
            models.iter().map(|m| m.parameters()).collect();
        let n_params = all_params[0].len();
        for (rank, params) in all_params.iter().enumerate().skip(1) {
            if params.len() != n_params {
                return Err(TensorError::new(&format!(
                    "Ddp: replica {} has {} parameters, expected {}",
                    rank,
                    params.len(),
                    n_params
                )));
            }
        }

        let mut param_groups = Vec::with_capacity(n_params);
        for pi in 0..n_params {
            let group: Vec<Variable> =
                all_params.iter().map(|p| p[pi].variable.clone()).collect();
            param_groups.push(group);
        }

        // Match buffers
        let all_buffers: Vec<Vec<Buffer>> =
            models.iter().map(|m| m.buffers()).collect();
        let n_buffers = all_buffers[0].len();
        let mut buffer_groups = Vec::with_capacity(n_buffers);
        for bi in 0..n_buffers {
            let group: Vec<Buffer> =
                all_buffers.iter().map(|b| b[bi].clone()).collect();
            buffer_groups.push(group);
        }

        Ok(Ddp {
            comms,
            devices: devices.to_vec(),
            param_groups,
            buffer_groups,
        })
    }

    /// Broadcast all parameters and buffers from rank 0 to all replicas.
    pub fn sync_params(&self) -> Result<()> {
        for group in &self.param_groups {
            let tensors: Vec<Tensor> = group.iter().map(|v| v.data()).collect();
            let refs: Vec<&Tensor> = tensors.iter().collect();
            self.comms.broadcast(&refs, 0)?;
        }
        for group in &self.buffer_groups {
            let tensors: Vec<Tensor> = group.iter().map(|b| b.get()).collect();
            let refs: Vec<&Tensor> = tensors.iter().collect();
            self.comms.broadcast(&refs, 0)?;
        }
        Ok(())
    }

    /// AllReduce-average gradients across all replicas.
    /// Call after backward(), before optimizer.step().
    pub fn all_reduce_gradients(&self) -> Result<()> {
        for group in &self.param_groups {
            if group[0].grad().is_none() {
                continue;
            }
            let grads: Vec<Tensor> = group
                .iter()
                .map(|v| v.grad().expect("gradient missing on replica"))
                .collect();
            let refs: Vec<&Tensor> = grads.iter().collect();
            self.comms.all_reduce(&refs, ReduceOp::Avg)?;
        }
        Ok(())
    }

    /// Broadcast buffers from rank 0 (BatchNorm running stats etc).
    pub fn sync_buffers(&self) -> Result<()> {
        for group in &self.buffer_groups {
            let tensors: Vec<Tensor> = group.iter().map(|b| b.get()).collect();
            let refs: Vec<&Tensor> = tensors.iter().collect();
            self.comms.broadcast(&refs, 0)?;
        }
        Ok(())
    }

    /// AllReduce gradients weighted by per-device batch contribution.
    ///
    /// For heterogeneous DDP where devices process different numbers of
    /// batches per sync step. Each replica's gradient is scaled by
    /// `(batch_counts[rank] / total)` before AllReduce Sum, producing
    /// the correct mean gradient.
    ///
    /// Use with [`ElChe::batch_counts`] for automatic weighting
    /// (see [`ElChe`] for the full heterogeneous DDP strategy):
    ///
    /// ```ignore
    /// ddp.weighted_all_reduce_gradients(cadence.batch_counts())?;
    /// ```
    pub fn weighted_all_reduce_gradients(&self, batch_counts: &[usize]) -> Result<()> {
        if batch_counts.len() != self.devices.len() {
            return Err(TensorError::new(&format!(
                "weighted_all_reduce: batch_counts len ({}) != device count ({})",
                batch_counts.len(),
                self.devices.len(),
            )));
        }
        let total: usize = batch_counts.iter().sum();
        if total == 0 {
            return Err(TensorError::new("weighted_all_reduce: total batch count is 0"));
        }
        for group in &self.param_groups {
            if group[0].grad().is_none() {
                continue;
            }
            let grads: Vec<Tensor> = group
                .iter()
                .enumerate()
                .map(|(rank, v)| {
                    let g = v.grad().expect("gradient missing on replica");
                    let weight = batch_counts[rank] as f64 / total as f64;
                    g.mul_scalar_(weight).ok();
                    g
                })
                .collect();
            let refs: Vec<&Tensor> = grads.iter().collect();
            self.comms.all_reduce(&refs, ReduceOp::Sum)?;
        }
        Ok(())
    }

    /// Number of devices.
    pub fn world_size(&self) -> usize {
        self.devices.len()
    }

    /// Devices in use.
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    // --- One-liner DDP setup (operates on Graph) ---

    /// One-call setup: auto-detect GPUs, distribute the model, set the
    /// optimizer, and enable training mode.
    ///
    /// - **Multi-GPU** (2+ usable CUDA devices): replicates via
    ///   [`Graph::distribute`], creates per-replica optimizers, enables training.
    /// - **Single-GPU / CPU**: sets optimizer and training mode only (no DDP
    ///   overhead).
    ///
    /// Always prints a diagnostic summary to stderr showing detected hardware.
    ///
    /// ```ignore
    /// Ddp::setup(&model, |dev| build_model(dev), |p| Adam::new(p, 0.001))?;
    ///
    /// // Training loop is identical for 1 or N GPUs:
    /// for batch in model.epoch(epoch).activate() {
    ///     let out = model.forward_batch(&batch?)?;
    ///     loss.backward()?;
    ///     model.step()?;
    /// }
    /// ```
    pub fn setup<F, M, G, O>(
        model: &Graph,
        builder: F,
        optimizer: G,
    ) -> Result<()>
    where
        F: Fn(Device) -> Result<M>,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        Self::print_device_summary();
        model.distribute(builder)?;
        model.set_optimizer(optimizer);
        model.set_training(true);

        // Auto-enable El Che for heterogeneous GPU setups
        if Self::is_heterogeneous() {
            model.configure_el_che(&DdpConfig::new());
        }

        Ok(())
    }

    /// One-call setup with explicit configuration.
    ///
    /// Like [`setup()`](Self::setup) but accepts a [`DdpConfig`] for
    /// controlling El Che cadence, speed hints, and overhead targets.
    ///
    /// ```ignore
    /// Ddp::setup_with(&model, builder, optimizer,
    ///     DdpConfig::new().speed_hint(1, 2.3))?;
    /// ```
    pub fn setup_with<F, M, G, O>(
        model: &Graph,
        builder: F,
        optimizer: G,
        config: DdpConfig,
    ) -> Result<()>
    where
        F: Fn(Device) -> Result<M>,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        Self::print_device_summary();
        model.distribute(builder)?;
        model.set_optimizer(optimizer);
        model.set_training(true);
        model.configure_el_che(&config);
        // Pass timeline to distributed state for event injection in step().
        if let Some(tl) = config.timeline {
            if let Some(ref mut state) = *model.distributed.borrow_mut() {
                state.timeline = Some(tl);
            }
        }
        Ok(())
    }

    /// Deprecated: renamed to [`setup()`](Self::setup).
    #[deprecated(since = "0.3.0", note = "Renamed to Ddp::setup()")]
    pub fn auto<F, M, G, O>(
        model: &Graph,
        builder: F,
        optimizer: G,
    ) -> Result<()>
    where
        F: Fn(Device) -> Result<M>,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        Self::setup(model, builder, optimizer)
    }

    /// Deprecated: renamed to [`setup_with()`](Self::setup_with).
    #[deprecated(since = "0.3.0", note = "Renamed to Ddp::setup_with()")]
    pub fn auto_with<F, M, G, O>(
        model: &Graph,
        builder: F,
        optimizer: G,
        config: DdpConfig,
    ) -> Result<()>
    where
        F: Fn(Device) -> Result<M>,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        Self::setup_with(model, builder, optimizer, config)
    }

    // -------------------------------------------------------------------
    // Builder mode: framework-managed training
    // -------------------------------------------------------------------

    /// Create a builder for framework-managed multi-GPU training.
    ///
    /// The framework owns the training loop, data pipeline, and epoch management.
    /// Each GPU gets its own model replica and optimizer. A coordinator triggers
    /// periodic parameter averaging based on the configured `ApplyPolicy` and
    /// `AverageBackend`.
    ///
    /// Returns a [`DdpBuilder`] for fluent configuration. Call `.run()` to
    /// spawn training threads, then `.join()` on the returned [`DdpHandle`]
    /// to block until completion.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use flodl::*;
    ///
    /// let handle = Ddp::builder(
    ///     |dev| model_factory(dev),
    ///     |params| Adam::new(params, 0.001),
    ///     |model, batch| { /* forward + loss */ },
    /// )
    /// .dataset(dataset)
    /// .batch_size(32)
    /// .num_epochs(10)
    /// .policy(ApplyPolicy::Cadence)
    /// .backend(AverageBackend::Nccl)
    /// .run()?;
    ///
    /// let state = handle.join()?;
    /// ```
    ///
    /// With fewer than 2 CUDA devices, training runs on the main thread
    /// with no coordination. The API is identical in both cases.
    pub fn builder<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
    ) -> DdpBuilder<F, M, G, O, T>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        DdpHandle::new_builder(model_factory, optim_factory, train_fn)
    }

    /// Detect whether the current CUDA setup has different GPU models.
    fn is_heterogeneous() -> bool {
        use crate::tensor::{cuda_available, cuda_device_count, cuda_device_name_idx};
        if !cuda_available() || cuda_device_count() < 2 {
            return false;
        }
        let n = cuda_device_count();
        let names: Vec<Option<String>> = (0..n)
            .map(cuda_device_name_idx)
            .collect();
        names.windows(2).any(|w| w[0] != w[1])
    }

    /// Print a diagnostic summary of detected CUDA devices to stderr.
    fn print_device_summary() {
        use crate::tensor::{
            cuda_available, cuda_device_count,
            cuda_device_name_idx, cuda_memory_info_idx,
        };
        use crate::monitor::format_bytes;

        if !cuda_available() || cuda_device_count() == 0 {
            crate::verbose!("  ddp: no CUDA available | CPU mode");
            return;
        }

        let n = cuda_device_count();
        let mut names = Vec::with_capacity(n as usize);
        let mut parts = Vec::with_capacity(n as usize);

        for i in 0..n {
            let raw_name = cuda_device_name_idx(i)
                .unwrap_or_else(|| format!("CUDA({})", i));
            let short = raw_name
                .strip_prefix("NVIDIA ")
                .unwrap_or(&raw_name)
                .to_string();
            let vram = cuda_memory_info_idx(i)
                .map(|(_, total)| format!(" ({})", format_bytes(total)))
                .unwrap_or_default();
            parts.push(format!("{}{}", short, vram));
            names.push(raw_name);
        }

        let heterogeneous = names.windows(2).any(|w| w[0] != w[1]);

        if n == 1 {
            crate::verbose!("  ddp: 1 GPU | {} | single-device mode", parts[0]);
        } else if heterogeneous {
            crate::verbose!(
                "  ddp: {} GPUs (heterogeneous) | {}",
                n,
                parts.join(" | "),
            );
        } else {
            crate::verbose!("  ddp: {} GPUs | {}", n, parts.join(" | "));
        }
    }
}

// ---------------------------------------------------------------------------
// DDP configuration
// ---------------------------------------------------------------------------

/// Configuration for [`Ddp::setup_with()`].
///
/// Controls El Che cadence behavior for heterogeneous multi-GPU training.
/// Use [`DdpConfig::new()`] for defaults or build with method chaining.
///
/// ```ignore
/// Ddp::setup_with(&model, builder, optimizer,
///     DdpConfig::new()
///         .speed_hint(1, 2.3)     // rank 1 is slow, 2.3x ratio
///         .overhead_target(0.08)  // tune to 8% overhead
/// )?;
/// ```
#[derive(Debug, Clone)]
pub struct DdpConfig {
    /// Initial speed ratio hint: (slow_rank, fast_to_slow_ratio).
    /// Applied before the first timing measurement.
    pub speed_hint: Option<(usize, f64)>,
    /// AllReduce overhead target for anchor auto-tune (default: 0.10).
    pub overhead_target: Option<f64>,
    /// Max batches on slow device before AllReduce.
    /// - `None` = auto (El Che decides, default).
    /// - `Some(0)` = disabled (traditional per-batch DDP, no El Che).
    /// - `Some(n)` = fixed anchor at n.
    pub max_anchor: Option<usize>,
    /// Maximum gradient norm for per-rank clipping in El Che mode.
    ///
    /// When set, each rank's accumulated gradients are clipped (L2 norm)
    /// before the normalize-by-count and weighted AllReduce steps. This
    /// ensures replica gradients (which the caller cannot reach) are bounded
    /// identically to rank 0.
    ///
    /// Standard DDP does not need this because the caller clips rank 0's
    /// gradients and AllReduce averages them.
    pub max_grad_norm: Option<f64>,
    /// Optional system timeline for high-frequency profiling.
    pub timeline: Option<std::sync::Arc<crate::monitor::Timeline>>,
}

impl DdpConfig {
    /// Default configuration: El Che auto-enabled for heterogeneous GPUs.
    pub fn new() -> Self {
        DdpConfig {
            speed_hint: None,
            overhead_target: None,
            max_anchor: None,
            max_grad_norm: None,
            timeline: None,
        }
    }

    /// Set initial speed ratio hint.
    ///
    /// `slow_rank`: which device is slowest.
    /// `ratio`: how many times faster the fastest device is (e.g., 2.3).
    ///
    /// After the first AllReduce, El Che discovers actual speeds and
    /// self-corrects even a wrong guess.
    pub fn speed_hint(mut self, slow_rank: usize, ratio: f64) -> Self {
        self.speed_hint = Some((slow_rank, ratio));
        self
    }

    /// Set AllReduce overhead target (fraction of compute time).
    ///
    /// Default: 0.10 (10%). Lower values = fewer AllReduces = more
    /// gradient accumulation. El Che auto-tunes the anchor to stay
    /// below this target.
    pub fn overhead_target(mut self, target: f64) -> Self {
        self.overhead_target = Some(target.clamp(0.01, 0.50));
        self
    }

    /// Set max batches on slow device before AllReduce.
    ///
    /// - `None` (default): El Che auto-tunes from overhead measurement.
    /// - `Some(0)`: disable El Che entirely (traditional per-batch sync).
    /// - `Some(n)`: fixed anchor at n (fast device gets proportionally more).
    pub fn max_anchor(mut self, max: Option<usize>) -> Self {
        self.max_anchor = max;
        self
    }

    /// Set maximum gradient norm for per-rank clipping in El Che mode.
    ///
    /// When set, each rank's accumulated gradients are clipped to this L2
    /// norm before normalize-by-count and AllReduce. Essential for
    /// heterogeneous DDP where replica gradients are otherwise unreachable
    /// by the caller.
    pub fn max_grad_norm(mut self, max_norm: f64) -> Self {
        self.max_grad_norm = Some(max_norm);
        self
    }

    /// Attach a system timeline for high-frequency profiling.
    pub fn timeline(mut self, tl: std::sync::Arc<crate::monitor::Timeline>) -> Self {
        self.timeline = Some(tl);
        self
    }
}

impl Default for DdpConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "ddp_tests.rs"]
mod tests;
