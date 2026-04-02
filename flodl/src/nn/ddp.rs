//! Distributed Data Parallel (DDP) for transparent multi-GPU training.
//!
//! Three usage levels:
//!
//! **One-liner (recommended)**: [`Ddp::auto()`] detects GPUs, distributes the
//! model, sets the optimizer, and enables training mode in a single call.
//! Works transparently with 1 or N GPUs.
//!
//! **Transparent (via Graph)**: Call `model.distribute()`, `set_optimizer()`,
//! and `set_training()` separately for finer control over setup.
//!
//! **Manual (via Ddp)**: For complex training patterns (GAN, RL, progressive).
//! Explicit control over gradient sync and parameter broadcast.
//!
//! # One-liner setup
//!
//! ```ignore
//! Ddp::auto(&model, |dev| build_model(dev), |p| Adam::new(&p, 0.001))?;
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
//! # Transparent mode (separate calls)
//!
//! ```ignore
//! model.distribute(|dev| build_model(dev))?;
//! model.set_optimizer(|p| Adam::new(&p, 0.001));
//! model.set_training(true);
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
    /// Ddp::auto(&model, |dev| build_model(dev), |p| Adam::new(&p, 0.001))?;
    ///
    /// // Training loop is identical for 1 or N GPUs:
    /// for batch in model.epoch(epoch).activate() {
    ///     let out = model.forward_batch(&batch?)?;
    ///     loss.backward()?;
    ///     model.step()?;
    /// }
    /// ```
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
        Self::print_device_summary();
        model.distribute(builder)?;
        model.set_optimizer(optimizer);
        model.set_training(true);
        Ok(())
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
        // On multi-GPU hardware Ddp::auto would initialize NCCL,
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

        Ddp::auto(
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

        Ddp::auto(
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
}
