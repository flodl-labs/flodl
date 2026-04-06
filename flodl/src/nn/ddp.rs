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
//! Ddp::setup(&model, |dev| build_model(dev), |p| Adam::new(&p, 0.001))?;
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
use crate::nn::cuda_event::CudaEvent;
use crate::nn::nccl::{NcclComms, ReduceOp};
use crate::nn::ddp_run::{DdpBuilder, DdpHandle};
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
    /// Ddp::setup(&model, |dev| build_model(dev), |p| Adam::new(&p, 0.001))?;
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
        G: Fn(Vec<Parameter>) -> O,
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
        G: Fn(Vec<Parameter>) -> O,
        O: Optimizer + 'static,
    {
        Self::print_device_summary();
        model.distribute(builder)?;
        model.set_optimizer(optimizer);
        model.set_training(true);
        model.configure_el_che(&config);
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
        G: Fn(Vec<Parameter>) -> O,
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
        G: Fn(Vec<Parameter>) -> O,
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
    /// periodic parameter averaging based on the configured [`ApplyPolicy`] and
    /// [`AverageBackend`].
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
            eprintln!("  ddp: no CUDA available | CPU mode");
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
            eprintln!("  ddp: 1 GPU | {} | single-device mode", parts[0]);
        } else if heterogeneous {
            eprintln!(
                "  ddp: {} GPUs (heterogeneous) | {}",
                n,
                parts.join(" | "),
            );
        } else {
            eprintln!("  ddp: {} GPUs | {}", n, parts.join(" | "));
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
}

impl DdpConfig {
    /// Default configuration: El Che auto-enabled for heterogeneous GPUs.
    pub fn new() -> Self {
        DdpConfig {
            speed_hint: None,
            overhead_target: None,
            max_anchor: None,
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
}

impl Default for DdpConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// El Che -- heterogeneous DDP cadence strategy
// ---------------------------------------------------------------------------

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

    /// Per-device batch counts (for [`Ddp::weighted_all_reduce_gradients`]).
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{
        cuda_device_count, cuda_synchronize, test_device, DType, TensorOptions,
    };
    use super::NCCL_LOCK;

    fn require_multi_gpu() -> bool {
        if !test_device().is_cuda() || cuda_device_count() < 2 {
            return false;
        }
        for i in 0..2 {
            let opts = TensorOptions {
                dtype: DType::Float32,
                device: Device::CUDA(i),
            };
            if Tensor::zeros(&[1], opts).is_err() {
                eprintln!(
                    "Device CUDA({i}) cannot run compute kernels, skipping multi-GPU test"
                );
                return false;
            }
        }
        true
    }

    // -- CPU validation tests -----------------------------------------------

    #[test]
    fn test_ddp_requires_two_models() {
        // Can't construct Ddp with 1 model (NCCL needs 2+ CUDA devices).
        // Just verify the validation logic.
        let result = Ddp::wrap(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ddp_model_device_mismatch() {
        // Model count must match device count
        let result = Ddp::wrap(
            &[],
            &[Device::CUDA(0), Device::CUDA(1)],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_shard_sizes_equal() {
        let ratios = vec![0.5, 0.5];
        let state = mock_state(&ratios);
        assert_eq!(state.compute_shard_sizes(10), vec![5, 5]);
        assert_eq!(state.compute_shard_sizes(11), vec![6, 5]);
        assert_eq!(state.compute_shard_sizes(3), vec![2, 1]);
    }

    #[test]
    fn test_shard_sizes_unequal() {
        let ratios = vec![0.7, 0.3];
        let state = mock_state(&ratios);
        assert_eq!(state.compute_shard_sizes(10), vec![7, 3]);
        assert_eq!(state.compute_shard_sizes(100), vec![70, 30]);
    }

    #[test]
    fn test_shard_sizes_three_devices() {
        let ratios = vec![0.5, 0.3, 0.2];
        let state = mock_state(&ratios);
        let sizes = state.compute_shard_sizes(10);
        assert_eq!(sizes.iter().sum::<i64>(), 10);
        assert_eq!(sizes, vec![5, 3, 2]);
    }

    /// Helper: create a minimal DistributedState for unit tests.
    fn mock_state(ratios: &[f64]) -> DistributedState {
        let n = ratios.len();
        DistributedState {
            replicas: Vec::new(),
            // Safety: we never use comms in shard/balance tests. Build a dummy.
            comms: unsafe { mock_nccl_comms(n) },
            devices: (0..n as u8)
                .map(Device::CUDA)
                .collect(),
            optimizers: Vec::new(),
            chunk_ratios: ratios.to_vec(),
            param_groups: Vec::new(),
            buffer_groups: Vec::new(),
            last_timing: None,
            last_shard_sizes: vec![0; n],
            ema_throughput: vec![0.0; n],
            step_count: 0,
            calibration_steps: DEFAULT_CALIBRATION_STEPS,
            rebalance_interval: DEFAULT_REBALANCE_INTERVAL,
            el_che: None,
            last_el_che_counts: Vec::new(),
            last_el_che_sync: None,
        }
    }

    /// Create a NcclComms with a null handle for shard-size unit tests only.
    /// Never call any actual NCCL operations on this.
    unsafe fn mock_nccl_comms(n: usize) -> NcclComms {
        let devices: Vec<Device> = (0..n as u8).map(Device::CUDA).collect();
        // Drop on a null handle is a no-op.
        unsafe { NcclComms::from_raw(std::ptr::null_mut(), devices) }
    }

    // -- Auto-balancer unit tests (CPU, no NCCL needed) ---------------------

    #[test]
    fn test_is_balanced_equal() {
        let state = mock_state(&[0.5, 0.5]);
        assert!(state.is_balanced());
    }

    #[test]
    fn test_is_balanced_unequal() {
        let state = mock_state(&[0.7, 0.3]);
        assert!(!state.is_balanced());
    }

    #[test]
    fn test_rebalance_proportional() {
        let mut state = mock_state(&[0.5, 0.5]);
        // GPU 0 is 3x faster than GPU 1
        state.ema_throughput = vec![30.0, 10.0];
        state.rebalance();
        // Expect ~75/25 split
        assert!((state.chunk_ratios[0] - 0.75).abs() < 0.01,
            "fast GPU should get ~75%, got {}", state.chunk_ratios[0]);
        assert!((state.chunk_ratios[1] - 0.25).abs() < 0.01,
            "slow GPU should get ~25%, got {}", state.chunk_ratios[1]);
        // Must sum to 1.0
        let sum: f64 = state.chunk_ratios.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "ratios must sum to 1.0, got {sum}");
    }

    #[test]
    fn test_rebalance_three_devices() {
        let mut state = mock_state(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        // Throughput: 50, 30, 20 (total 100)
        state.ema_throughput = vec![50.0, 30.0, 20.0];
        state.rebalance();
        assert!((state.chunk_ratios[0] - 0.50).abs() < 0.01);
        assert!((state.chunk_ratios[1] - 0.30).abs() < 0.01);
        assert!((state.chunk_ratios[2] - 0.20).abs() < 0.01);
        let sum: f64 = state.chunk_ratios.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_rebalance_respects_min_ratio() {
        let mut state = mock_state(&[0.5, 0.5]);
        // GPU 1 is extremely slow (would get <5% without clamping)
        state.ema_throughput = vec![100.0, 1.0];
        state.rebalance();
        assert!(state.chunk_ratios[1] >= MIN_CHUNK_RATIO,
            "slow GPU should get at least MIN_CHUNK_RATIO, got {}", state.chunk_ratios[1]);
        let sum: f64 = state.chunk_ratios.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_rebalance_no_data() {
        let mut state = mock_state(&[0.5, 0.5]);
        state.ema_throughput = vec![0.0, 0.0];
        state.rebalance();
        // Should not change ratios when no data
        assert_eq!(state.chunk_ratios, vec![0.5, 0.5]);
    }

    #[test]
    fn test_update_balance_calibration_timing() {
        let mut state = mock_state(&[0.5, 0.5]);
        // Simulate steps without timing (no CudaEvents on CPU)
        for _ in 0..DEFAULT_CALIBRATION_STEPS - 1 {
            let rebalanced = state.update_balance().unwrap();
            assert!(!rebalanced, "should not rebalance during calibration");
        }
        // Step at calibration boundary triggers rebalance (but no-op without data)
        let rebalanced = state.update_balance().unwrap();
        assert!(rebalanced, "should rebalance at calibration boundary");
    }

    #[test]
    fn test_update_balance_interval() {
        let mut state = mock_state(&[0.5, 0.5]);
        // Skip past calibration
        state.step_count = DEFAULT_CALIBRATION_STEPS;
        // Steps up to next interval should not rebalance
        for _ in 0..DEFAULT_REBALANCE_INTERVAL - 1 {
            let rebalanced = state.update_balance().unwrap();
            assert!(!rebalanced);
        }
        // At interval boundary: rebalance
        let rebalanced = state.update_balance().unwrap();
        assert!(rebalanced);
    }

    #[test]
    fn test_ema_throughput_init() {
        let mut state = mock_state(&[0.5, 0.5]);
        // First measurement initializes directly (not blended)
        state.ema_throughput = vec![0.0, 0.0];
        // Manually set what update_throughput would compute from timing
        let throughput_0 = 10.0;
        state.ema_throughput[0] = throughput_0; // simulates first measurement
        assert_eq!(state.ema_throughput[0], 10.0);
    }

    #[test]
    fn test_ema_throughput_smoothing() {
        let mut state = mock_state(&[0.5, 0.5]);
        state.ema_throughput = vec![10.0, 5.0];
        // Simulate what update_balance does with a new measurement
        let new_measurement = 20.0;
        state.ema_throughput[0] =
            EMA_ALPHA * new_measurement + (1.0 - EMA_ALPHA) * state.ema_throughput[0];
        // EMA: 0.3 * 20 + 0.7 * 10 = 6 + 7 = 13
        assert!((state.ema_throughput[0] - 13.0).abs() < 1e-9);
    }

    #[test]
    fn test_shard_sizes_after_rebalance() {
        let mut state = mock_state(&[0.5, 0.5]);
        // Rebalance to 70/30
        state.ema_throughput = vec![70.0, 30.0];
        state.rebalance();
        // Verify shard computation uses new ratios
        let sizes = state.compute_shard_sizes(100);
        assert_eq!(sizes.iter().sum::<i64>(), 100);
        assert_eq!(sizes[0], 70);
        assert_eq!(sizes[1], 30);
    }

    // -- Cross-device autograd verification ---------------------------------

    #[test]
    fn test_cross_device_autograd_gradient_flow() {
        if !require_multi_gpu() {
            return;
        }

        let opts0 = TensorOptions {
            dtype: DType::Float32,
            device: Device::CUDA(0),
        };
        let opts1 = TensorOptions {
            dtype: DType::Float32,
            device: Device::CUDA(1),
        };

        // Parameters on two different devices
        let w0 = Variable::new(Tensor::ones(&[4, 3], opts0).unwrap(), true);
        let w1 = Variable::new(Tensor::ones(&[4, 3], opts1).unwrap(), true);

        // Input on device 0 (no requires_grad, like training data)
        let input = Variable::new(
            Tensor::ones(&[4, 4], opts0).unwrap(),
            false,
        );

        // Chunk along batch dim: 2 shards of size 2
        let chunks = input.chunk(2, 0).unwrap();
        assert_eq!(chunks.len(), 2);

        // Shard 0: forward on device 0
        let out0 = chunks[0].matmul(&w0).unwrap(); // [2, 3] on dev0

        // Shard 1: move to device 1, forward there, move output back to device 0
        let shard1_dev1 = chunks[1].to_device(Device::CUDA(1)).unwrap();
        let out1_dev1 = shard1_dev1.matmul(&w1).unwrap(); // [2, 3] on dev1
        let out1_dev0 = out1_dev1.to_device(Device::CUDA(0)).unwrap(); // [2, 3] on dev0

        // Gather: cat outputs on device 0
        let gathered = Variable::cat_many(&[&out0, &out1_dev0], 0).unwrap(); // [4, 3]

        // Compute scalar loss
        let loss = gathered.sum().unwrap();

        // Backward
        loss.backward().unwrap();

        // Verify: both parameters received gradients on their own devices
        let grad0 = w0.grad();
        let grad1 = w1.grad();
        assert!(
            grad0.is_some(),
            "w0 on device 0 should have gradient after backward"
        );
        assert!(
            grad1.is_some(),
            "w1 on device 1 should have gradient after backward"
        );

        // Verify gradients are on the correct devices
        let g0 = grad0.unwrap();
        let g1 = grad1.unwrap();
        assert_eq!(g0.device(), Device::CUDA(0), "w0 gradient should be on device 0");
        assert_eq!(g1.device(), Device::CUDA(1), "w1 gradient should be on device 1");

        // Verify gradient values are non-zero
        let g0_sum = g0.sum().unwrap().item().unwrap();
        let g1_sum = g1.sum().unwrap().item().unwrap();
        assert!(
            g0_sum.abs() > 1e-6,
            "w0 gradient should be non-zero, got {g0_sum}"
        );
        assert!(
            g1_sum.abs() > 1e-6,
            "w1 gradient should be non-zero, got {g1_sum}"
        );

        cuda_synchronize(0);
        cuda_synchronize(1);
    }

    #[test]
    fn test_cross_device_autograd_values() {
        // Verify that cross-device backward produces the SAME gradients
        // as single-device backward (correctness check).
        if !require_multi_gpu() {
            return;
        }

        // Use deterministic values
        let w_data = Tensor::from_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[4, 2],
            Device::CUDA(0),
        )
        .unwrap();

        // Single-device reference: forward all on device 0
        let w_ref = Variable::new(w_data.clone(), true);
        let x = Tensor::from_f32(
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            &[4, 4],
            Device::CUDA(0),
        )
        .unwrap();
        let x_var = Variable::new(x.clone(), false);
        let out_ref = x_var.matmul(&w_ref).unwrap();
        let loss_ref = out_ref.sum().unwrap();
        loss_ref.backward().unwrap();
        let grad_ref = w_ref.grad().unwrap();
        let grad_ref_vals = grad_ref.to_f32_vec().unwrap();

        // Cross-device: split batch across 2 devices.
        // Create w0 and w1 from fresh tensors (not clones of w_data,
        // which was tainted by set_requires_grad through w_ref's shallow clone).
        let w0 = Variable::new(
            Tensor::from_f32(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[4, 2],
                Device::CUDA(0),
            )
            .unwrap(),
            true,
        );
        let w1 = Variable::new(
            Tensor::from_f32(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[4, 2],
                Device::CUDA(1),
            )
            .unwrap(),
            true,
        );
        let x_var2 = Variable::new(x, false);

        let chunks = x_var2.chunk(2, 0).unwrap();

        let out0 = chunks[0].matmul(&w0).unwrap();
        let shard1 = chunks[1].to_device(Device::CUDA(1)).unwrap();
        let out1_dev1 = shard1.matmul(&w1).unwrap();
        let out1_dev0 = out1_dev1.to_device(Device::CUDA(0)).unwrap();
        let gathered = Variable::cat_many(&[&out0, &out1_dev0], 0).unwrap();
        let loss = gathered.sum().unwrap();
        loss.backward().unwrap();

        // Sum of cross-device gradients should equal single-device gradient
        let g0 = w0.grad().unwrap().to_f32_vec().unwrap();
        let g1 = w1.grad().unwrap().to_f32_vec().unwrap();

        for i in 0..g0.len() {
            let cross_sum = g0[i] + g1[i];
            let diff = (cross_sum - grad_ref_vals[i]).abs();
            assert!(
                diff < 1e-5,
                "gradient mismatch at index {i}: cross-device sum {cross_sum} vs reference {}",
                grad_ref_vals[i]
            );
        }

        cuda_synchronize(0);
        cuda_synchronize(1);
    }

    // -- Graph integration tests (CPU, single-GPU fallback) -----------------

    #[test]
    fn test_graph_set_optimizer_and_step() {
        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear, ReLU, mse_loss};

        let model = FlowBuilder::from(Linear::new(4, 8).unwrap())
            .through(ReLU::new())
            .through(Linear::new(8, 2).unwrap())
            .build()
            .unwrap();

        model.set_optimizer(|p| Adam::new(&p, 0.01));
        model.set_training(true);

        // Snapshot initial params
        let params_before: Vec<f32> = model
            .parameters()
            .iter()
            .flat_map(|p| p.variable.data().to_f32_vec().unwrap())
            .collect();

        // One training step
        let x = Variable::new(
            Tensor::randn(&[4, 4], Default::default()).unwrap(),
            false,
        );
        let target = Variable::new(
            Tensor::randn(&[4, 2], Default::default()).unwrap(),
            false,
        );
        let out = model.forward(&x).unwrap();
        let loss = mse_loss(&out, &target).unwrap();
        loss.backward().unwrap();
        model.step().unwrap();

        // Params should have changed
        let params_after: Vec<f32> = model
            .parameters()
            .iter()
            .flat_map(|p| p.variable.data().to_f32_vec().unwrap())
            .collect();

        let changed = params_before
            .iter()
            .zip(&params_after)
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(changed, "parameters should change after step()");
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: make cuda-test-all"]
    fn test_graph_distribute_adapts_to_hardware() {
        use crate::graph::FlowBuilder;
        use crate::nn::Linear;
        use crate::tensor::usable_cuda_devices;

        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        let result = model.distribute(|dev| {
            FlowBuilder::from(Linear::on_device(4, 2, dev)?).build()
        });
        assert!(result.is_ok());

        let usable = usable_cuda_devices();
        if usable.len() >= 2 {
            // Multi-GPU: should be distributed
            assert!(model.is_distributed());
            assert_eq!(model.world_size(), usable.len());
        } else {
            // Single GPU or CPU: no-op
            assert!(!model.is_distributed());
            assert_eq!(model.world_size(), 1);
        }
    }

    #[test]
    fn test_ddp_auto_single_gpu() {
        // On multi-GPU hardware Ddp::setup would initialize NCCL,
        // which poisons CUBLAS for concurrent tests. Skip here;
        // multi-GPU path is validated in test_ddp_auto_multi_gpu.
        if cuda_device_count() >= 2 {
            return;
        }

        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear, ReLU, mse_loss};

        let model = FlowBuilder::from(Linear::new(4, 8).unwrap())
            .through(ReLU::new())
            .through(Linear::new(8, 2).unwrap())
            .build()
            .unwrap();

        Ddp::setup(
            &model,
            |dev| {
                FlowBuilder::from(Linear::on_device(4, 8, dev)?)
                    .through(ReLU::new())
                    .through(Linear::on_device(8, 2, dev)?)
                    .build()
            },
            |p| Adam::new(&p, 0.001),
        )
        .unwrap();

        // Optimizer should be set: step() works
        let x = Variable::new(
            Tensor::randn(&[4, 4], Default::default()).unwrap(),
            false,
        );
        let target = Variable::new(
            Tensor::randn(&[4, 2], Default::default()).unwrap(),
            false,
        );
        let out = model.forward(&x).unwrap();
        let loss = mse_loss(&out, &target).unwrap();
        loss.backward().unwrap();
        model.step().unwrap();

        assert!(!model.is_distributed());
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: make cuda-test-nccl"]
    fn test_ddp_auto_multi_gpu() {
        if !require_multi_gpu() {
            return;
        }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear, ReLU, mse_loss};

        let model = FlowBuilder::from(
            Linear::on_device(4, 8, Device::CUDA(0)).unwrap(),
        )
        .through(ReLU::new())
        .through(Linear::on_device(8, 2, Device::CUDA(0)).unwrap())
        .build()
        .unwrap();

        Ddp::setup(
            &model,
            |dev| {
                FlowBuilder::from(Linear::on_device(4, 8, dev)?)
                    .through(ReLU::new())
                    .through(Linear::on_device(8, 2, dev)?)
                    .build()
            },
            |p| Adam::new(&p, 0.001),
        )
        .unwrap();

        assert!(model.is_distributed());
        assert_eq!(model.world_size(), 2);

        // Full training step
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: Device::CUDA(0),
        };
        let x = Variable::new(
            Tensor::randn(&[8, 4], opts).unwrap(),
            false,
        );
        let target = Variable::new(
            Tensor::randn(&[8, 2], opts).unwrap(),
            false,
        );
        let out = model.forward(&x).unwrap();
        let loss = mse_loss(&out, &target).unwrap();
        loss.backward().unwrap();
        model.step().unwrap();

        cuda_synchronize(0);
        cuda_synchronize(1);
    }

    #[test]
    fn test_graph_step_without_optimizer() {
        use crate::graph::FlowBuilder;
        use crate::nn::Linear;

        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        // step() without set_optimizer() should be a no-op, not a crash
        let result = model.step();
        assert!(result.is_ok());
    }

    #[test]
    fn test_graph_set_lr() {
        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear};

        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        model.set_optimizer(|p| Adam::new(&p, 0.01));
        // Should not panic
        model.set_lr(0.001);
    }

    // -- El Che unit tests (CPU, no NCCL needed) ----------------------------

    #[test]
    fn test_cadence_initial_equal() {
        let c = ElChe::new(2, 10);
        assert_eq!(c.batches(0), 10);
        assert_eq!(c.batches(1), 10);
        assert_eq!(c.total_batches(), 20);
        assert_eq!(c.anchor(), 10);
        assert!(!c.is_calibrated());
    }

    #[test]
    fn test_cadence_initial_three_devices() {
        let c = ElChe::new(3, 15);
        assert_eq!(c.batches(0), 15);
        assert_eq!(c.batches(1), 15);
        assert_eq!(c.batches(2), 15);
        assert_eq!(c.total_batches(), 45);
    }

    #[test]
    fn test_cadence_ratio_discovery_2x() {
        // Device 0 is 2x faster than device 1.
        // Equal counts (10:10), device 0 finishes in 500ms, device 1 in 1000ms.
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.50); // high target to avoid anchor auto-tune
        let bc = c.batch_counts().to_vec(); c.report_timing(&[500.0, 1000.0], &bc, 10.0);

        assert!(c.is_calibrated());
        // Slow device (rank 1) keeps anchor=10, fast device (rank 0) gets ~20.
        assert_eq!(c.batches(1), 10);
        assert_eq!(c.batches(0), 20);
    }

    #[test]
    fn test_cadence_ratio_discovery_fbrl_like() {
        // Simulates RTX 5060 Ti vs GTX 1060 (~2.3:1 speed ratio).
        // Anchor=10 on slow device, equal initial counts.
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.50); // no auto-tune

        // Both ran 10 batches; fast took 730ms (73ms/batch), slow took 1640ms (164ms/batch).
        let bc = c.batch_counts().to_vec(); c.report_timing(&[730.0, 1640.0], &bc, 50.0);

        assert!(c.is_calibrated());
        assert_eq!(c.batches(1), 10); // slow device: anchor
        // Fast device: 164/73 * 10 ≈ 22.5, rounds to 22 or 23
        let fast = c.batches(0);
        assert!(
            (22..=23).contains(&fast),
            "expected ~22-23, got {fast}"
        );
    }

    #[test]
    fn test_cadence_anchor_auto_tune() {
        // High AllReduce overhead should trigger anchor increase.
        // 10% target: compute 1000ms, sync 500ms => overhead 50% >> 10%.
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.10);

        // Both devices equal speed, anchor=10.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[1000.0, 1000.0], &bc, 500.0);

        // overhead = 500/1000 = 0.50, target = 0.10
        // scale = 0.50/0.10 = 5.0 => new anchor = ceil(10 * 5) = 50
        assert_eq!(c.anchor(), 50);
        assert_eq!(c.batches(0), 50);
        assert_eq!(c.batches(1), 50);
    }

    #[test]
    fn test_cadence_anchor_auto_tune_with_speed_ratio() {
        // Heterogeneous: fast device 2x, high sync overhead.
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.10);

        // Fast=500ms, slow=1000ms (equal initial counts), sync=400ms.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[500.0, 1000.0], &bc, 400.0);

        // overhead = 400/1000 = 0.40, target = 0.10, scale = 4.0
        // new anchor = ceil(10 * 4) = 40
        assert_eq!(c.anchor(), 40);
        assert_eq!(c.batches(1), 40); // slow device
        // fast device: 100ms/batch vs 50ms/batch => 2x ratio => 80
        assert_eq!(c.batches(0), 80);
    }

    #[test]
    fn test_cadence_anchor_capped_at_max() {
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.01)
            .with_max_anchor(30);

        // Extreme overhead: sync dominates.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[100.0, 100.0], &bc, 500.0);

        // Would want anchor=500 but capped at 30.
        assert_eq!(c.anchor(), 30);
        assert_eq!(c.batches(0), 30);
    }

    #[test]
    fn test_cadence_stable_when_overhead_low() {
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.10);

        // sync=5ms on 1000ms compute => 0.5% overhead, well below 10%.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[1000.0, 1000.0], &bc, 5.0);

        assert_eq!(c.anchor(), 10); // no change
    }

    #[test]
    fn test_cadence_three_devices_mixed_speed() {
        let mut c = ElChe::new(3, 10)
            .with_overhead_target(0.50); // no auto-tune

        // Device 0: 3x fast (333ms), device 1: 2x fast (500ms), device 2: slow (1000ms).
        let bc = c.batch_counts().to_vec(); c.report_timing(&[333.0, 500.0, 1000.0], &bc, 10.0);

        assert_eq!(c.batches(2), 10); // slow: anchor
        // Device 1: 100ms/batch vs 33.3ms/batch for device 0
        // Device 0: ratio 100/33.3 = 3.0 => 30
        // Device 1: ratio 100/50 = 2.0 => 20
        assert_eq!(c.batches(0), 30);
        assert_eq!(c.batches(1), 20);
    }

    #[test]
    fn test_cadence_successive_reports_refine() {
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.50);

        // First report: 2x speed ratio.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[500.0, 1000.0], &bc, 10.0);
        assert_eq!(c.batches(0), 20);
        assert_eq!(c.batches(1), 10);

        // Second report: new counts, faster device did 20 in 1000ms (50ms/batch),
        // slow did 10 in 1000ms (100ms/batch). Ratio stays 2:1.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[1000.0, 1000.0], &bc, 10.0);
        assert_eq!(c.batches(0), 20);
        assert_eq!(c.batches(1), 10);
    }

    #[test]
    fn test_cadence_clamp_total() {
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.50);

        // Fast device gets 20, slow gets 10. Total = 30.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[500.0, 1000.0], &bc, 10.0);

        // Only 15 batches remain in the epoch.
        let clamped = c.clamp_total(15);
        assert_eq!(clamped.iter().sum::<usize>(), 15);
        // Proportions roughly preserved (2:1).
        assert!(clamped[0] >= clamped[1], "fast device should still get more");
    }

    #[test]
    fn test_cadence_clamp_total_no_op_when_within() {
        let c = ElChe::new(2, 10);
        // Total is 20, max is 30 => no clamping needed.
        let clamped = c.clamp_total(30);
        assert_eq!(clamped, vec![10, 10]);
    }

    #[test]
    fn test_cadence_builders() {
        let c = ElChe::new(2, 10)
            .with_overhead_target(0.20)
            .with_max_anchor(100);
        assert_eq!(c.anchor(), 10);
        assert!(!c.is_calibrated());

        // Overhead target clamped to valid range
        let c2 = ElChe::new(2, 5)
            .with_overhead_target(0.001); // below min 0.01
        // Would be clamped to 0.01 internally
        let _ = c2;
    }

    #[test]
    fn test_cadence_max_batch_diff() {
        let c = ElChe::new(2, 10).with_max_batch_diff(5);
        assert_eq!(c.max_batch_diff(), Some(5));

        let c2 = ElChe::new(2, 10);
        assert_eq!(c2.max_batch_diff(), None);
    }

    #[test]
    fn test_batch_count_clamped_to_max_diff() {
        // Setup: 2 GPUs, anchor=10, max_batch_diff=3.
        let mut c = ElChe::new(2, 10).with_max_batch_diff(3);

        // First report (calibration): GPU 0 slow (10ms/batch), GPU 1 fast (2ms/batch).
        // batch_counts are [10, 10] initially, so wall = ms_per_batch * count.
        // GPU 0: 10 batches * 10ms = 100ms. GPU 1: 10 batches * 2ms = 20ms.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[100.0, 20.0], &bc, 0.0);
        assert!(c.is_calibrated());
        // Calibration pass: no clamping. GPU 1 gets 50 batches (ratio 10/2 * 10).
        let counts_after_cal = c.batch_counts().to_vec();
        assert_eq!(counts_after_cal[0], 10);
        assert_eq!(counts_after_cal[1], 50);

        // Second report: GPU 1 suddenly slows to near GPU 0 speed.
        // batch_counts now [10, 50]. GPU 0: 10*10ms=100ms. GPU 1: 50*9ms=450ms.
        // ms_per_batch[1] EMA: alpha=clamp(|9-2|/2, 0.1, 0.8)=0.8, new=0.8*9+0.2*2=7.6
        // slow_ms = max(10, 7.6) = 10. target[1] = 10*(10/7.6)=13.
        // Without clamping: 50 -> 13 (drop of 37). With max_batch_diff=3: 50 -> 47.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[100.0, 450.0], &bc, 0.0);
        let counts = c.batch_counts();
        assert!(counts[1] >= counts_after_cal[1] - 3,
            "batch count drop should be clamped to 3, was {} now {}",
            counts_after_cal[1], counts[1]);
    }

    #[test]
    fn test_cadence_weighted_allreduce_validation() {
        // Validates that Ddp::weighted_all_reduce_gradients rejects
        // mismatched batch_counts length (tested indirectly via the
        // assertion in ElChe that world_size >= 2).
        let c = ElChe::new(2, 10);
        assert_eq!(c.batch_counts().len(), 2);
    }

    #[test]
    #[should_panic(expected = "El Che requires at least 2 devices")]
    fn test_cadence_requires_two_devices() {
        ElChe::new(1, 10);
    }

    #[test]
    #[should_panic(expected = "anchor must be >= 1")]
    fn test_cadence_requires_positive_anchor() {
        ElChe::new(2, 0);
    }

    #[test]
    fn test_cadence_speed_ratio_2x() {
        // Rank 1 is slow, rank 0 is 2x faster
        let c = ElChe::new(2, 10).with_speed_ratio(1, 2.0);
        assert_eq!(c.batches(0), 20);
        assert_eq!(c.batches(1), 10);
    }

    #[test]
    fn test_cadence_speed_ratio_fbrl() {
        // RTX 5060 Ti (rank 0) ~2.3x faster than GTX 1060 (rank 1)
        let c = ElChe::new(2, 10).with_speed_ratio(1, 2.3);
        assert_eq!(c.batches(0), 23);
        assert_eq!(c.batches(1), 10);
    }

    #[test]
    fn test_cadence_speed_ratio_slow_rank_0() {
        // Rank 0 is the slow one (unusual but valid)
        let c = ElChe::new(2, 10).with_speed_ratio(0, 3.0);
        assert_eq!(c.batches(0), 10);
        assert_eq!(c.batches(1), 30);
    }

    #[test]
    fn test_cadence_speed_ratio_equal() {
        let c = ElChe::new(2, 10).with_speed_ratio(1, 1.0);
        assert_eq!(c.batches(0), 10);
        assert_eq!(c.batches(1), 10);
    }

    #[test]
    fn test_cadence_speed_ratio_three_devices() {
        // Rank 2 is slow, others are 3x faster
        let c = ElChe::new(3, 10).with_speed_ratio(2, 3.0);
        assert_eq!(c.batches(0), 30);
        assert_eq!(c.batches(1), 30);
        assert_eq!(c.batches(2), 10);
    }

    #[test]
    fn test_cadence_speed_ratio_three_devices_mid_slow() {
        // Rank 1 is slow, 0 and 2 are fast
        let c = ElChe::new(3, 10).with_speed_ratio(1, 2.0);
        assert_eq!(c.batches(0), 20);
        assert_eq!(c.batches(1), 10);
        assert_eq!(c.batches(2), 20);
    }

    #[test]
    fn test_cadence_max_anchor_one() {
        // max_anchor=1: minimal cadence, sync after every slow-device batch
        let mut c = ElChe::new(2, 1)
            .with_max_anchor(1)
            .with_speed_ratio(1, 2.0);

        assert_eq!(c.batches(0), 2);
        assert_eq!(c.batches(1), 1);

        // High overhead won't increase anchor past 1
        let bc = c.batch_counts().to_vec(); c.report_timing(&[100.0, 200.0], &bc, 500.0);
        assert_eq!(c.anchor(), 1);
    }

    #[test]
    fn test_cadence_speed_ratio_self_corrects() {
        // Start with wrong guess: say rank 0 is slow, but it's actually fast
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.50)
            .with_speed_ratio(0, 2.0);

        // Wrong: rank 0 gets 10, rank 1 gets 20
        assert_eq!(c.batches(0), 10);
        assert_eq!(c.batches(1), 20);

        // After timing: rank 0 is actually 2x faster (500ms for 10 vs 2000ms for 20)
        let bc = c.batch_counts().to_vec(); c.report_timing(&[500.0, 2000.0], &bc, 10.0);

        // Self-corrected: rank 1 is slow (anchor), rank 0 gets more
        assert_eq!(c.batches(1), c.anchor());
        assert!(c.batches(0) > c.batches(1), "fast device should get more batches");
    }

    // -- DdpConfig tests ------------------------------------------------------

    #[test]
    fn test_ddp_config_defaults() {
        let c = DdpConfig::new();
        assert!(c.speed_hint.is_none());
        assert!(c.overhead_target.is_none());
        assert!(c.max_anchor.is_none());
    }

    #[test]
    fn test_ddp_config_builder() {
        let c = DdpConfig::new()
            .speed_hint(1, 2.5)
            .overhead_target(0.05)
            .max_anchor(Some(20));
        assert_eq!(c.speed_hint, Some((1, 2.5)));
        assert_eq!(c.overhead_target, Some(0.05));
        assert_eq!(c.max_anchor, Some(20));
    }

    #[test]
    fn test_ddp_config_disable_el_che() {
        let c = DdpConfig::new().max_anchor(Some(0));
        assert_eq!(c.max_anchor, Some(0));
    }

    #[test]
    fn test_configure_el_che_creates_from_config() {
        let mut state = mock_state(&[0.5, 0.5]);

        let config = DdpConfig::new().speed_hint(1, 2.0).overhead_target(0.15);
        state.configure_el_che(&config);

        assert!(state.el_che.is_some());
        let el = state.el_che.as_ref().unwrap();
        // Slow rank gets anchor, fast gets more
        assert_eq!(el.batches(1), el.anchor());
        assert!(el.batches(0) > el.batches(1));
    }

    #[test]
    fn test_configure_el_che_disabled() {
        let mut state = mock_state(&[0.5, 0.5]);

        let config = DdpConfig::new().max_anchor(Some(0));
        state.configure_el_che(&config);

        assert!(state.el_che.is_none());
    }

    #[test]
    fn test_configure_el_che_single_device_noop() {
        let mut state = mock_state(&[1.0]);

        let config = DdpConfig::new();
        state.configure_el_che(&config);

        // Single device -- El Che not created
        assert!(state.el_che.is_none());
    }

    // -- El Che CUDA integration tests (multi-GPU, NCCL) ----------------------

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: make cuda-test-nccl"]
    fn test_el_che_full_training_loop() {
        if !require_multi_gpu() {
            return;
        }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear, ReLU, mse_loss};
        use crate::data::{DataLoader, DataSet};

        // Simple dataset: 200 samples, 4 features, 2 targets
        struct TinyData;
        impl DataSet for TinyData {
            fn len(&self) -> usize { 200 }
            fn get(&self, index: usize) -> crate::tensor::Result<Vec<Tensor>> {
                let x = Tensor::from_f32(
                    &[index as f32; 4], &[4], Device::CPU,
                )?;
                let y = Tensor::from_f32(
                    &[(index as f32) * 0.1; 2], &[2], Device::CPU,
                )?;
                Ok(vec![x, y])
            }
        }

        let model = FlowBuilder::from(
            Linear::on_device(4, 8, Device::CUDA(0)).unwrap(),
        )
        .through(ReLU::new())
        .through(Linear::on_device(8, 2, Device::CUDA(0)).unwrap())
        .build()
        .unwrap();

        Ddp::setup_with(
            &model,
            |dev| {
                FlowBuilder::from(Linear::on_device(4, 8, dev)?)
                    .through(ReLU::new())
                    .through(Linear::on_device(8, 2, dev)?)
                    .build()
            },
            |p| Adam::new(&p, 0.001),
            DdpConfig::new().speed_hint(1, 2.0).max_anchor(Some(3)),
        )
        .unwrap();

        assert!(model.is_distributed());
        assert!(model.has_el_che());
        assert_eq!(model.world_size(), 2);

        // Set up DataLoader
        let loader = DataLoader::from_dataset(TinyData)
            .batch_size(10)
            .names(&["input", "target"])
            .build()
            .unwrap();

        model.set_data_loader(loader, "input").unwrap();

        // Run 1 epoch
        let mut step_count = 0;
        for batch in model.epoch(0).activate() {
            let b = batch.unwrap();
            let out = model.forward_batch(&b).unwrap();
            let target = Variable::new(b["target"].clone(), false);
            let loss = mse_loss(&out, &target).unwrap();
            loss.backward().unwrap();
            model.step().unwrap();
            step_count += 1;
        }

        // With anchor=3 and ratio=2.0: ~5 batches per El Che step (3 + 2*3=6, total ~5-6)
        // 200 samples / 10 batch_size = 20 batches total
        // ~20 / 5 = ~4 El Che iterations
        assert!(step_count > 0, "should have trained at least one step");
        assert!(step_count <= 20, "should not have more steps than batches");

        cuda_synchronize(0);
        cuda_synchronize(1);
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: make cuda-test-nccl"]
    fn test_el_che_tagged_outputs_gathered() {
        if !require_multi_gpu() {
            return;
        }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear, ReLU, mse_loss};
        use crate::data::{DataLoader, DataSet};

        struct TinyData;
        impl DataSet for TinyData {
            fn len(&self) -> usize { 100 }
            fn get(&self, index: usize) -> crate::tensor::Result<Vec<Tensor>> {
                let x = Tensor::from_f32(
                    &[index as f32; 4], &[4], Device::CPU,
                )?;
                let y = Tensor::from_f32(
                    &[(index as f32) * 0.1; 2], &[2], Device::CPU,
                )?;
                Ok(vec![x, y])
            }
        }

        // Build model with a tagged intermediate
        let model = FlowBuilder::from(
            Linear::on_device(4, 8, Device::CUDA(0)).unwrap(),
        )
        .through(ReLU::new())
        .tag("hidden")
        .through(Linear::on_device(8, 2, Device::CUDA(0)).unwrap())
        .build()
        .unwrap();

        Ddp::setup_with(
            &model,
            |dev| {
                FlowBuilder::from(Linear::on_device(4, 8, dev)?)
                    .through(ReLU::new())
                    .tag("hidden")
                    .through(Linear::on_device(8, 2, dev)?)
                    .build()
            },
            |p| Adam::new(&p, 0.001),
            DdpConfig::new().max_anchor(Some(2)),
        )
        .unwrap();

        let loader = DataLoader::from_dataset(TinyData)
            .batch_size(10)
            .names(&["input", "target"])
            .build()
            .unwrap();

        model.set_data_loader(loader, "input").unwrap();

        // Run one iteration and check tagged output
        let mut iter = model.epoch(0).activate();
        if let Some(batch) = iter.next() {
            let b = batch.unwrap();
            let out = model.forward_batch(&b).unwrap();

            // Tagged output should exist and have gathered batch dimension
            let hidden = model.tagged("hidden");
            assert!(hidden.is_some(), "tagged output should be gathered");
            let h = hidden.unwrap();
            // hidden shape: [total_samples_across_devices, 8]
            assert_eq!(h.shape()[1], 8);
            // Total samples should be > batch_size (multiple batches gathered)
            assert!(h.shape()[0] >= 10, "gathered hidden should span multiple batches");

            let target = Variable::new(b["target"].clone(), false);
            let loss = mse_loss(&out, &target).unwrap();
            loss.backward().unwrap();
            model.step().unwrap();
        }

        cuda_synchronize(0);
        cuda_synchronize(1);
    }
}
