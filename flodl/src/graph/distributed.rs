use std::collections::HashMap;

use crate::autograd::Variable;
use crate::data::Batch;
use crate::nn::Module;
use crate::tensor::{Result, Tensor, TensorError};

use super::node::DEFAULT_INPUT;
use super::graph::{Graph, DataLoaderBinding, GraphEpochIterator};
use super::LossContext;

// ---------------------------------------------------------------------------
// Distributed Data Parallel + optimizer integration
// ---------------------------------------------------------------------------

impl Graph {
    /// Enable multi-GPU training. Detects usable CUDA devices, creates model
    /// replicas via the factory closure, and broadcasts parameters from rank 0.
    ///
    /// If only one usable GPU is found, this is a no-op (single-GPU mode).
    /// The factory receives a `Device` and must return a model with the same
    /// architecture as `self` (same parameter count and shapes).
    ///
    /// After this call, [`forward`](Module::forward) automatically scatters
    /// input across devices and gathers output. [`step`](Graph::step) handles
    /// AllReduce + optimizer step + zero_grad.
    ///
    /// For a one-liner that also sets the optimizer and training mode, see
    /// [`Ddp::setup()`](crate::distributed::Ddp::setup).
    ///
    /// ```ignore
    /// model.distribute(|dev| build_model(dev))?;
    /// ```
    pub fn distribute<F, M>(&self, factory: F) -> Result<()>
    where
        F: Fn(crate::tensor::Device) -> Result<M>,
        M: crate::nn::Module + 'static,
    {
        use crate::distributed::ddp::DistributedState;
        use crate::distributed::nccl::NcclComms;

        let devices = crate::tensor::usable_cuda_devices();
        if devices.len() < 2 {
            // Single GPU or no GPU: no-op, step() still works with local optimizer
            return Ok(());
        }

        // Create replicas for ranks 1..N
        let mut replicas: Vec<Box<dyn crate::nn::Module>> = Vec::new();
        for &dev in &devices[1..] {
            let model = factory(dev)?;
            replicas.push(Box::new(model));
        }

        // Init NCCL communicators
        let comms = NcclComms::new(&devices)?;

        // Match parameters across replicas
        let rank0_params = self.parameters();
        let n_params = rank0_params.len();
        let mut param_groups = Vec::with_capacity(n_params);
        for (pi, p) in rank0_params.iter().enumerate() {
            let mut group = vec![p.variable.clone()];
            for replica in &replicas {
                let rp = replica.parameters();
                if rp.len() != n_params {
                    return Err(TensorError::new(&format!(
                        "distribute: replica has {} parameters, expected {}",
                        rp.len(),
                        n_params
                    )));
                }
                group.push(rp[pi].variable.clone());
            }
            param_groups.push(group);
        }

        // Match buffers
        let rank0_buffers = self.buffers();
        let n_buffers = rank0_buffers.len();
        let mut buffer_groups = Vec::with_capacity(n_buffers);
        for (bi, b) in rank0_buffers.iter().enumerate() {
            let mut group = vec![b.clone()];
            for replica in &replicas {
                let rb = replica.buffers();
                if rb.len() != n_buffers {
                    return Err(TensorError::new(&format!(
                        "distribute: replica has {} buffers, expected {}",
                        rb.len(),
                        n_buffers
                    )));
                }
                group.push(rb[bi].clone());
            }
            buffer_groups.push(group);
        }

        let n = devices.len();
        let equal_ratio = 1.0 / n as f64;

        let state = DistributedState {
            replicas,
            comms,
            devices,
            optimizers: Vec::new(),
            chunk_ratios: vec![equal_ratio; n],
            param_groups,
            buffer_groups,
            last_timing: None,
            last_shard_sizes: vec![0; n],
            ema_throughput: vec![0.0; n],
            step_count: 0,
            calibration_steps: crate::distributed::ddp::DEFAULT_CALIBRATION_STEPS,
            rebalance_interval: crate::distributed::ddp::DEFAULT_REBALANCE_INTERVAL,
            el_che: None,
            last_el_che_counts: Vec::new(),
            last_el_che_sync: None,
            max_grad_norm: None,
            timeline: None,
        };

        // Broadcast params from rank 0 to all replicas
        state.sync_params()?;

        *self.distributed.borrow_mut() = Some(state);
        Ok(())
    }

    /// Auto-detect usable CUDA devices and distribute the model.
    ///
    /// The builder closure receives a Device and must return a fresh model.
    /// No-op if fewer than 2 usable GPUs are found.
    ///
    /// ```ignore
    /// model.auto_distribute(|dev| build_model(dev))?;
    /// ```
    pub fn auto_distribute<F>(&self, builder: F) -> Result<()>
    where
        F: Fn(crate::tensor::Device) -> Result<Graph>,
    {
        let devices = crate::tensor::usable_cuda_devices();
        if devices.len() < 2 {
            return Ok(());
        }
        self.distribute(builder)
    }

    /// Set the optimizer for training. When distributed, creates one optimizer
    /// per replica. When single-GPU, creates a single optimizer.
    ///
    /// The factory receives the parameter list and returns an optimizer.
    ///
    /// ```ignore
    /// model.set_optimizer(|p| Adam::new(p, 0.001));
    /// ```
    pub fn set_optimizer<F, O>(&self, factory: F)
    where
        F: Fn(&[crate::nn::Parameter]) -> O,
        O: crate::nn::Optimizer + 'static,
    {
        let mut dist = self.distributed.borrow_mut();
        if let Some(ref mut state) = *dist {
            // Distributed: one optimizer per replica
            let mut optimizers: Vec<Box<dyn crate::nn::Optimizer>> = Vec::new();

            // Rank 0 optimizer (uses self's parameters)
            let rank0_opt = factory(&self.parameters());
            optimizers.push(Box::new(rank0_opt));

            // Replicas
            for replica in &state.replicas {
                let opt = factory(&replica.parameters());
                optimizers.push(Box::new(opt));
            }

            state.optimizers = optimizers;
        } else {
            // Single GPU: one optimizer
            let opt = factory(&self.parameters());
            *self.optimizer.borrow_mut() = Some(Box::new(opt));
        }
    }

    /// Attach a per-batch LR scheduler.
    ///
    /// When set, `step()` updates every optimizer's learning rate to
    /// `scheduler.lr(training_step) * lr_scale` before the optimizer step.
    /// The internal `training_step` counter increments once per `step()`
    /// call and is independent of the recurrent-state `step_count`.
    ///
    /// Works for both single-GPU and distributed graphs.
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// let sched: Arc<dyn Scheduler> = Arc::new(MultiStepLR::new(0.1, &[100, 150], 0.1));
    /// graph.set_scheduler(sched);
    /// ```
    pub fn set_scheduler(&self, scheduler: std::sync::Arc<dyn crate::nn::Scheduler>) {
        *self.scheduler.borrow_mut() = Some(scheduler);
    }

    /// Set the DDP linear-scaling factor (Goyal et al., 2017) applied to the
    /// attached scheduler's output every batch. Defaults to 1.0 (no scaling).
    ///
    /// Has no effect if no scheduler is attached; bake the scaling into the
    /// optimizer's base LR instead for that case.
    pub fn set_lr_scale(&self, scale: f64) {
        self.lr_scale.set(scale);
    }

    /// Current training step (increments once per `step()` call). Used by the
    /// attached scheduler, if any.
    pub fn training_step(&self) -> usize {
        self.training_step.get()
    }

    /// Compute the scheduled LR for the current training step, if a
    /// scheduler is attached. Returns `None` when no scheduler is set so
    /// the caller can leave the optimizer LR alone.
    fn scheduled_lr(&self) -> Option<f64> {
        let sched = self.scheduler.borrow();
        sched.as_ref()
            .map(|s| s.lr(self.training_step.get()) * self.lr_scale.get())
    }

    /// Perform one training step.
    ///
    /// When distributed: AllReduce gradients (weighted if auto-balanced),
    /// sync buffers, step all optimizers, zero grad, update auto-balancer.
    /// When single-GPU: step optimizer, zero grad.
    ///
    /// This is the single call that replaces `opt.step(); opt.zero_grad();`
    /// and makes multi-GPU training transparent.
    ///
    /// When a scheduler is attached via [`Self::set_scheduler`], every
    /// optimizer's LR is updated from `scheduler.lr(training_step) *
    /// lr_scale` before the step, and `training_step` increments by one
    /// after.
    pub fn step(&self) -> Result<()> {
        let mut dist = self.distributed.borrow_mut();
        if let Some(ref mut state) = *dist {
            // Auto-detect external forward/backward: El Che needs forward_batch()
            // to populate batch counts. If counts are empty on the first step,
            // the caller is using manual forward + backward. Disable El Che and
            // fall through to standard AllReduce.
            if state.el_che.is_some() && state.last_el_che_counts.iter().sum::<usize>() == 0 {
                crate::verbose!(
                    "  ddp: El Che disabled (external forward/backward detected). \
                     Using standard AllReduce. To silence: DdpConfig::new().max_anchor(Some(0))"
                );
                state.el_che = None;
            }

            if state.el_che.is_some() {
                // El Che path: weighted AllReduce by actual per-device batch counts.
                // Gradients were accumulated in forward_distributed_el_che().
                use crate::distributed::cuda_event::CudaEvent;
                use crate::distributed::nccl::ReduceOp;

                let counts = state.last_el_che_counts.clone();
                let total: usize = counts.iter().sum();

                if total > 0 {
                    // Compute wall time since last sync (= compute time for this cadence step)
                    let compute_ms = state.last_el_che_sync
                        .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);

                    // Per-rank gradient clipping (before normalize-by-count).
                    // Bounds accumulated gradients on all ranks identically,
                    // including replicas the caller cannot reach.
                    if let Some(max_norm) = state.max_grad_norm {
                        for rank in 0..state.devices.len() {
                            if counts[rank] == 0 {
                                continue;
                            }
                            let params: Vec<Tensor> = state.param_groups
                                .iter()
                                .filter(|group| group[rank].grad().is_some())
                                .map(|group| group[rank].data())
                                .collect();
                            if !params.is_empty() {
                                Tensor::clip_grad_norm_fused(&params, max_norm)?;
                            }
                        }
                    }

                    // Normalize accumulated gradients: each rank accumulated
                    // counts[rank] backward passes, scale by 1/count so the
                    // optimizer sees the mean gradient regardless of batch count.
                    for group in &state.param_groups {
                        if group[0].grad().is_none() {
                            continue;
                        }
                        for (rank, var) in group.iter().enumerate() {
                            if counts[rank] > 1 {
                                if let Some(g) = var.grad() {
                                    let _ = g.mul_scalar_(1.0 / counts[rank] as f64);
                                }
                            }
                        }
                    }

                    // Weighted AllReduce: scale by batch contribution, then Sum.
                    // Ranks with 0 batches (epoch-end clamping) have no gradients;
                    // zero them so AllReduce still produces the correct mean.
                    if let Some(ref tl) = state.timeline {
                        tl.event(crate::monitor::EventKind::SyncStart);
                    }
                    let sync_start = std::time::Instant::now();
                    for group in &state.param_groups {
                        if group[0].grad().is_none() && counts[0] > 0 {
                            continue;
                        }
                        let grads: Vec<Tensor> = group
                            .iter()
                            .enumerate()
                            .map(|(rank, v)| {
                                let weight = counts[rank] as f64 / total as f64;
                                match v.grad() {
                                    Some(g) => {
                                        g.mul_scalar_(weight).ok();
                                        g
                                    }
                                    None => {
                                        // No gradient on this rank (0 batches). Use zeros.
                                        let data = v.data();
                                        let opts = crate::tensor::TensorOptions {
                                            dtype: data.dtype(),
                                            device: data.device(),
                                        };
                                        Tensor::zeros(&data.shape(), opts)
                                            .expect("failed to create zero gradient")
                                    }
                                }
                            })
                            .collect();
                        let refs: Vec<&Tensor> = grads.iter().collect();
                        state.comms.all_reduce(&refs, ReduceOp::Sum)?;
                    }
                    state.sync_buffers()?;
                    let sync_ms = sync_start.elapsed().as_secs_f64() * 1000.0;
                    if let Some(ref tl) = state.timeline {
                        tl.event(crate::monitor::EventKind::SyncEnd { duration_ms: sync_ms });
                    }

                    if let Some(lr) = self.scheduled_lr() {
                        for opt in &mut state.optimizers {
                            opt.set_lr(lr);
                        }
                    }
                    for opt in &mut state.optimizers {
                        opt.step()?;
                    }
                    for opt in &state.optimizers {
                        opt.zero_grad();
                    }

                    // Report timing to El Che for ratio + anchor adaptation.
                    // Use per-device wall times from CudaEvents if available,
                    // otherwise estimate from total wall time.
                    let wall_ms: Vec<f64> = if let Some(ref timing) = state.last_timing {
                        timing.iter().map(|(start, end)| {
                            CudaEvent::elapsed_time(start, end).unwrap_or(0.0) as f64
                        }).collect()
                    } else {
                        vec![compute_ms; state.devices.len()]
                    };

                    let old_anchor = state.el_che.as_ref().map(|e| e.anchor());
                    let updated_counts = if let Some(ref mut el_che) = state.el_che {
                        if !wall_ms.is_empty() {
                            el_che.report_timing(&wall_ms, &counts, sync_ms);
                        }
                        Some(el_che.batch_counts().to_vec())
                    } else {
                        None
                    };
                    if let (Some(tl), Some(old), Some(el_che)) =
                        (&state.timeline, old_anchor, &state.el_che)
                    {
                        let new = el_che.anchor();
                        if new != old {
                            tl.event(crate::monitor::EventKind::AnchorChanged {
                                from: old,
                                to: new,
                            });
                        }
                    }

                    state.last_timing = None;
                    state.last_el_che_sync = Some(std::time::Instant::now());

                    // Must drop the distributed borrow before accessing data_binding
                    drop(dist);

                    // Feed updated batch counts back to the loader for the next iteration
                    if let Some(counts) = updated_counts {
                        let binding = self.data_binding.borrow();
                        if let Some(ref b) = *binding {
                            b.loader.set_el_che_counts(counts);
                        }
                    }
                }
            } else {
                // Standard DDP path: per-batch scatter + AllReduce
                if let Some(ref tl) = state.timeline {
                    tl.event(crate::monitor::EventKind::SyncStart);
                }
                let ddp_sync_start = std::time::Instant::now();
                if state.is_balanced() {
                    state.all_reduce_gradients()?;
                } else {
                    let batch_size: i64 = state.last_shard_sizes.iter().sum();
                    state.weighted_all_reduce_gradients(batch_size)?;
                }
                state.sync_buffers()?;
                if let Some(ref tl) = state.timeline {
                    let dur = ddp_sync_start.elapsed().as_secs_f64() * 1000.0;
                    tl.event(crate::monitor::EventKind::SyncEnd { duration_ms: dur });
                }

                if let Some(lr) = self.scheduled_lr() {
                    for opt in &mut state.optimizers {
                        opt.set_lr(lr);
                    }
                }
                for opt in &mut state.optimizers {
                    opt.step()?;
                }
                for opt in &state.optimizers {
                    opt.zero_grad();
                }

                // Update throughput tracking and potentially rebalance
                state.update_balance()?;
            }
        } else {
            // Single GPU
            let scheduled = self.scheduled_lr();
            let mut opt = self.optimizer.borrow_mut();
            if let Some(ref mut optimizer) = *opt {
                if let Some(lr) = scheduled {
                    optimizer.set_lr(lr);
                }
                optimizer.step()?;
                optimizer.zero_grad();
            }
        }
        self.training_step.set(self.training_step.get() + 1);
        Ok(())
    }

    /// Number of devices in use (1 if not distributed).
    pub fn world_size(&self) -> usize {
        self.distributed
            .borrow()
            .as_ref()
            .map_or(1, |d| d.world_size())
    }

    /// Whether this graph is running in distributed mode.
    pub fn is_distributed(&self) -> bool {
        self.distributed.borrow().is_some()
    }

    /// Whether El Che cadence is active (heterogeneous DDP).
    pub fn has_el_che(&self) -> bool {
        self.distributed
            .borrow()
            .as_ref()
            .is_some_and(|d| d.el_che.is_some())
    }

    /// Configure El Che cadence for distributed training.
    ///
    /// Called by `Ddp::setup_with()` after [`distribute`](Graph::distribute).
    /// No-op if not in distributed mode.
    pub(crate) fn configure_el_che(&self, config: &crate::distributed::ddp::DdpConfig) {
        let mut dist = self.distributed.borrow_mut();
        if let Some(ref mut state) = *dist {
            state.configure_el_che(config);
        }
    }

    /// Current batch distribution ratios across devices.
    ///
    /// Returns a vector of fractions summing to 1.0. Empty if not distributed.
    /// Changes over time as the auto-balancer adapts to measured throughput.
    pub fn chunk_ratios(&self) -> Vec<f64> {
        self.distributed
            .borrow()
            .as_ref()
            .map_or_else(Vec::new, |d| d.chunk_ratios.clone())
    }

    /// Per-device throughput (samples/ms) measured by the auto-balancer.
    ///
    /// Returns EMA-smoothed values. Empty if not distributed or no
    /// measurements yet.
    pub fn throughput(&self) -> Vec<f64> {
        self.distributed
            .borrow()
            .as_ref()
            .map_or_else(Vec::new, |d| d.ema_throughput.clone())
    }

    /// Per-device shard sizes from the last forward pass.
    ///
    /// Returns the actual number of samples each device processed.
    /// Empty if not distributed.
    pub fn shard_sizes(&self) -> Vec<i64> {
        self.distributed
            .borrow()
            .as_ref()
            .map_or_else(Vec::new, |d| d.last_shard_sizes.clone())
    }

    /// Devices used for distributed training. Empty if not distributed.
    pub fn devices(&self) -> Vec<crate::tensor::Device> {
        self.distributed
            .borrow()
            .as_ref()
            .map_or_else(Vec::new, |d| d.devices.clone())
    }

    /// Set learning rate on all optimizers (distributed and single-GPU).
    pub fn set_lr(&self, lr: f64) {
        let mut dist = self.distributed.borrow_mut();
        if let Some(ref mut state) = *dist {
            for opt in &mut state.optimizers {
                opt.set_lr(lr);
            }
        } else {
            let mut opt = self.optimizer.borrow_mut();
            if let Some(ref mut optimizer) = *opt {
                optimizer.set_lr(lr);
            }
        }
    }

    // -- DataLoader integration -----------------------------------------------

    /// Attach a DataLoader for integrated training.
    ///
    /// When distributed: upgrades the loader to per-device backends (resident
    /// or streaming per device based on VRAM). Enables `model.epoch()` for
    /// zero-transfer iteration and `model.forward(&batch)` for auto-wired
    /// forward passes.
    ///
    /// When single-GPU: stores the loader as-is. `model.epoch()` delegates
    /// to `loader.epoch()` directly.
    ///
    /// The `forward_input` parameter names the batch field used as the primary
    /// model input (e.g., "image"). Other batch fields that match graph
    /// `.input()` ports are auto-wired as auxiliary inputs. All remaining
    /// batch fields are treated as targets (available in the user-facing
    /// Batch for loss computation).
    ///
    /// ```ignore
    /// model.set_data_loader(loader, "image")?;
    /// ```
    pub fn set_data_loader(
        &self,
        mut loader: crate::data::DataLoader,
        forward_input: &str,
    ) -> Result<()> {
        let loader_names: Vec<String> = loader.names().to_vec();

        // Validate forward_input exists in loader names
        if !loader_names.iter().any(|n| n == forward_input) {
            return Err(TensorError::new(&format!(
                "set_data_loader: forward_input '{}' not found in loader names [{}]",
                forward_input,
                loader_names.join(", ")
            )));
        }

        // If distributed, upgrade the loader to per-device backends
        let dist = self.distributed.borrow();
        if let Some(ref state) = *dist {
            let devices = state.devices.clone();
            // We need the dataset Arc. Get it by reading from the loader's internals
            // before upgrading. The upgrade_distributed method handles this.
            drop(dist); // drop borrow before mutating loader
            // Create a temporary dataset from the loader's existing data.
            // upgrade_distributed will load it onto all devices.
            loader.upgrade_distributed(
                &devices,
                // The dataset Arc is extracted inside upgrade_distributed
                // from the existing loader inner. We pass a dummy that
                // upgrade_distributed replaces. Actually, let me read the
                // loader's dataset.
                // Problem: the dataset is inside the loader. We need to
                // extract it. Let me add a method.
                loader.dataset_arc()?,
            )?;
        } else {
            drop(dist);
        }

        // Match batch names to graph Input ports
        let graph_input_names: Vec<String> = self.inputs.iter().map(|i| i.name.clone()).collect();
        let mut graph_inputs: Vec<(String, String)> = Vec::new();
        let mut target_names: Vec<String> = Vec::new();

        for name in &loader_names {
            if name == forward_input {
                continue; // primary input, handled separately
            }
            if graph_input_names.contains(name) {
                graph_inputs.push((name.clone(), name.clone()));
            } else {
                target_names.push(name.clone());
            }
        }

        // Build shard_input_map: graph input index -> loader tensor position.
        // self.inputs[0] is the entry (forward_input), self.inputs[1..] are .input() ports.
        let mut shard_input_map: Vec<usize> = Vec::with_capacity(self.inputs.len());
        for port in &self.inputs {
            let lookup_name = if port.name == DEFAULT_INPUT {
                forward_input
            } else {
                &port.name
            };
            match loader_names.iter().position(|n| n == lookup_name) {
                Some(idx) => shard_input_map.push(idx),
                None => {
                    return Err(TensorError::new(&format!(
                        "set_data_loader: graph input '{}' not found in loader names [{}]",
                        lookup_name,
                        loader_names.join(", ")
                    )));
                }
            }
        }

        // Get chunk_ratios
        let chunk_ratios = {
            let dist = self.distributed.borrow();
            dist.as_ref()
                .map(|d| d.chunk_ratios.clone())
                .unwrap_or_default()
        };

        *self.data_binding.borrow_mut() = Some(DataLoaderBinding {
            batch_names: loader_names.clone(),
            loader,
            forward_input: forward_input.to_string(),
            graph_inputs,
            target_names,
            shard_input_map,
            chunk_ratios,
        });

        Ok(())
    }

    /// Register a per-batch loss function for El Che distributed training.
    ///
    /// When set, `forward_distributed_el_che` runs forward + loss + backward
    /// per batch internally, keeping only ONE forward graph in VRAM at a time.
    /// Without this, all forward graphs are held simultaneously (VRAM scales
    /// with anchor * devices), which caps the practical anchor at 1.
    ///
    /// The closure receives a [`LossContext`] with live autograd on all fields.
    /// It must return a scalar loss `Variable`.
    ///
    /// `forward_batch()` returns detached gathered outputs when a loss function
    /// is registered. Tags and traces on the graph are gathered (detached) for
    /// metrics. Calling `.backward()` on the returned Variable is a no-op.
    ///
    /// ```ignore
    /// model.set_loss_fn(|ctx: &LossContext| {
    ///     let cls  = cross_entropy_loss(&ctx.tags["head"], &ctx.batch["label"])?;
    ///     let rec  = mse_loss(&ctx.tags["recon"], &ctx.batch["image"])?;
    ///     Ok(cls + rec)
    /// });
    ///
    /// for batch in model.epoch(epoch).activate() {
    ///     let _metrics = model.forward_batch(&batch?)?;
    ///     model.step()?;
    /// }
    /// ```
    pub fn set_loss_fn<F>(&self, f: F)
    where
        F: Fn(&LossContext) -> Result<Variable> + 'static,
    {
        *self.loss_fn.borrow_mut() = Some(Box::new(f));
    }

    /// Whether a per-batch loss function is registered.
    pub fn has_loss_fn(&self) -> bool {
        self.loss_fn.borrow().is_some()
    }

    /// Get an epoch iterator for integrated training.
    ///
    /// When distributed: returns a `DistributedEpochIterator` that produces
    /// per-rank shards and a user-facing Batch with targets on the gather device.
    /// When single-GPU: delegates to the DataLoader's epoch iterator.
    ///
    /// ```ignore
    /// for batch in model.epoch(epoch) {
    ///     let b = batch?;
    ///     let out = model.forward(&b)?;
    ///     let loss = mse_loss(&out, &b["letter"])?;
    ///     loss.backward()?;
    ///     model.step()?;
    /// }
    /// ```
    pub fn epoch(&self, epoch: usize) -> GraphEpochIterator<'_> {
        // Update chunk_ratios and seed El Che counts from distributed state
        {
            let dist = self.distributed.borrow();
            let mut binding = self.data_binding.borrow_mut();
            if let (Some(d), Some(ref mut b)) = (dist.as_ref(), binding.as_mut()) {
                b.chunk_ratios = d.chunk_ratios.clone();

                // Seed El Che batch counts for the epoch iterator
                if let Some(ref el_che) = d.el_che {
                    b.loader.set_el_che_counts(el_che.batch_counts().to_vec());
                }
            }
        }

        let binding = self.data_binding.borrow();
        if binding.is_none() {
            panic!("Graph::epoch() requires set_data_loader() first");
        }

        let is_distributed = {
            let b = self.data_binding.borrow();
            b.as_ref().unwrap().loader.is_distributed()
        };

        if is_distributed {
            GraphEpochIterator::Distributed(self, epoch)
        } else {
            GraphEpochIterator::Single(self, epoch)
        }
    }

    /// Number of batches per epoch (delegates to the attached DataLoader).
    pub fn data_num_batches(&self) -> usize {
        self.data_binding
            .borrow()
            .as_ref()
            .expect("call set_data_loader first")
            .loader
            .num_batches()
    }

    /// Batch size (delegates to the attached DataLoader).
    pub fn data_batch_size(&self) -> usize {
        self.data_binding
            .borrow()
            .as_ref()
            .expect("call set_data_loader first")
            .loader
            .batch_size()
    }

    /// Distributed forward: scatter input, parallel forward on replicas, gather output.
    /// Records CudaEvent timing per rank for auto-balancing.
    pub(crate) fn forward_distributed_scatter(&self, input: &Variable) -> Result<Variable> {
        use crate::distributed::cuda_event::{CudaEvent, CudaEventFlags};
        use crate::tensor::set_current_cuda_device;

        // Read config without holding borrow during forward calls
        let (n, devices, shard_sizes) = {
            let dist = self.distributed.borrow();
            let dist = dist.as_ref().unwrap();
            let batch_size = input.shape()[0];
            let n = dist.devices.len();
            let shard_sizes = dist.compute_shard_sizes(batch_size);
            let devices = dist.devices.clone();
            (n, devices, shard_sizes)
        }; // borrow dropped

        let mut offset = 0i64;
        let mut outputs: Vec<Variable> = Vec::with_capacity(n);
        let mut timing: Vec<(CudaEvent, CudaEvent)> = Vec::with_capacity(n);

        for (rank, &shard_size) in shard_sizes.iter().enumerate() {
            if shard_size == 0 {
                continue;
            }

            let shard = input.narrow(0, offset, shard_size)?;
            offset += shard_size;

            // Record start event on this device's default stream
            let device_idx = match devices[rank] {
                crate::tensor::Device::CUDA(i) => i,
                _ => 0,
            };
            set_current_cuda_device(device_idx);
            let start = CudaEvent::new(CudaEventFlags::Default)?;
            start.record()?;

            if rank == 0 {
                let dev_shard = shard.to_device(devices[0])?;
                let out = self.forward_impl(std::slice::from_ref(&dev_shard))?;
                outputs.push(out);
            } else {
                let dev_shard = shard.to_device(devices[rank])?;
                let out = {
                    let dist = self.distributed.borrow();
                    let dist = dist.as_ref().unwrap();
                    dist.replicas[rank - 1].forward(&dev_shard)?
                };
                let out_rank0 = out.to_device(devices[0])?;
                outputs.push(out_rank0);
            }

            // Record end event on same device's stream
            set_current_cuda_device(device_idx);
            let end = CudaEvent::new(CudaEventFlags::Default)?;
            end.record()?;
            timing.push((start, end));
        }

        // Store timing and shard sizes for step() to consume
        {
            let mut dist = self.distributed.borrow_mut();
            let dist = dist.as_mut().unwrap();
            dist.last_timing = Some(timing);
            dist.last_shard_sizes = shard_sizes;
        }

        if outputs.len() == 1 {
            return Ok(outputs.into_iter().next().unwrap());
        }

        let refs: Vec<&Variable> = outputs.iter().collect();
        Variable::cat_many(&refs, 0)
    }

    /// Presharded distributed forward: per-rank data already on each device.
    /// Consumes shards from the DataLoader, forwards on each replica, gathers output.
    pub(crate) fn forward_distributed_presharded(&self) -> Result<Variable> {
        use crate::distributed::cuda_event::{CudaEvent, CudaEventFlags};
        use crate::tensor::set_current_cuda_device;

        // Take per-rank shards and input mapping from the DataLoader
        let (per_rank_shards, shard_input_map) = {
            let binding = self.data_binding.borrow();
            let binding = binding.as_ref().unwrap();
            let shards = binding.loader.take_shards()
                .expect("forward_distributed_presharded: no shards pending");
            let map = binding.shard_input_map.clone();
            (shards, map)
        };

        let (n, devices, gather_device) = {
            let dist = self.distributed.borrow();
            let dist = dist.as_ref().unwrap();
            let n = dist.devices.len();
            let devices = dist.devices.clone();
            let gather_device = self.data_binding.borrow()
                .as_ref()
                .map(|b| b.loader.device())
                .unwrap_or(devices[0]);
            (n, devices, gather_device)
        };

        let mut outputs: Vec<Variable> = Vec::with_capacity(n);
        let mut timing: Vec<(CudaEvent, CudaEvent)> = Vec::with_capacity(n);
        let mut shard_sizes: Vec<i64> = Vec::with_capacity(n);

        for (rank, shard_data) in per_rank_shards.iter().enumerate() {
            if shard_data.is_empty() || shard_data[0].shape()[0] == 0 {
                shard_sizes.push(0);
                continue;
            }

            let shard_size = shard_data[0].shape()[0];
            shard_sizes.push(shard_size);

            // Record start event on this device's default stream
            let device_idx = match devices[rank] {
                crate::tensor::Device::CUDA(i) => i,
                _ => 0,
            };
            set_current_cuda_device(device_idx);
            let start = CudaEvent::new(CudaEventFlags::Default)?;
            start.record()?;

            // Build full input vector: map graph inputs to shard positions
            let graph_inputs: Vec<Variable> = shard_input_map.iter()
                .map(|&idx| Variable::new(shard_data[idx].clone(), false))
                .collect();

            if rank == 0 {
                let out = self.forward_impl(&graph_inputs)?;
                outputs.push(out);
            } else {
                let out = {
                    let dist = self.distributed.borrow();
                    let dist = dist.as_ref().unwrap();
                    let replica = &dist.replicas[rank - 1];
                    match replica.as_graph() {
                        Some(g) => g.forward_impl(&graph_inputs)?,
                        None => replica.forward(&graph_inputs[0])?,
                    }
                };
                // Gather output to gather device
                let out_gathered = if out.data().device() != gather_device {
                    out.to_device(gather_device)?
                } else {
                    out
                };
                outputs.push(out_gathered);
            }

            // Record end event
            set_current_cuda_device(device_idx);
            let end = CudaEvent::new(CudaEventFlags::Default)?;
            end.record()?;
            timing.push((start, end));
        }

        // Store timing and shard sizes for step()
        {
            let mut dist = self.distributed.borrow_mut();
            let dist = dist.as_mut().unwrap();
            dist.last_timing = Some(timing);
            dist.last_shard_sizes = shard_sizes;
        }

        if outputs.len() == 1 {
            return Ok(outputs.into_iter().next().unwrap());
        }

        let refs: Vec<&Variable> = outputs.iter().collect();
        Variable::cat_many(&refs, 0)
    }

    /// Gather tagged outputs and loop traces from a graph into accumulators.
    /// Used by forward_distributed_el_che for both the main graph and replicas.
    fn gather_tags_and_traces(
        g: &Graph,
        gather_device: crate::tensor::Device,
        has_tags: bool,
        has_traces: bool,
        gathered_tags: &mut HashMap<String, Vec<Variable>>,
        gathered_traces: &mut HashMap<(String, usize), Vec<Variable>>,
    ) -> Result<()> {
        if has_tags {
            let tagged = g.tagged_outputs.borrow();
            for (name, var) in tagged.iter() {
                let moved = if var.data().device() != gather_device {
                    var.to_device(gather_device)?
                } else {
                    var.clone()
                };
                gathered_tags.entry(name.clone()).or_default().push(moved);
            }
        }
        if has_traces {
            for tag_name in g.tag_names() {
                if let Some(step_traces) = g.traces(&tag_name) {
                    for (step_idx, trace_var) in step_traces.iter().enumerate() {
                        let moved = if trace_var.data().device() != gather_device {
                            trace_var.to_device(gather_device)?
                        } else {
                            trace_var.clone()
                        };
                        gathered_traces
                            .entry((tag_name.clone(), step_idx))
                            .or_default()
                            .push(moved);
                    }
                }
            }
        }
        Ok(())
    }

    /// El Che distributed forward: multiple complete batches per device.
    ///
    /// Each device processes its `batch_counts[rank]` batches independently.
    /// Tagged outputs are gathered across all batches and all devices.
    ///
    /// **Per-batch backward** (when `set_loss_fn` is registered): each batch
    /// runs forward -> loss -> backward immediately, freeing the forward graph.
    /// Only ONE activation graph is alive at any time, regardless of anchor.
    /// Gradients accumulate across batches. Returns detached gathered outputs.
    ///
    /// **Legacy path** (no loss_fn): all forward graphs are held in VRAM
    /// simultaneously. The user calls backward on the gathered output.
    ///
    /// Called by `forward_batch()` when El Che batches are pending.
    fn forward_distributed_el_che(&self) -> Result<Variable> {
        // Take per-device batches, input mapping, and batch field names
        let (per_device_batches, shard_input_map, batch_names) = {
            let binding = self.data_binding.borrow();
            let binding = binding.as_ref().unwrap();
            let batches = binding.loader.take_el_che_batches()
                .expect("forward_distributed_el_che: no El Che batches pending");
            let map = binding.shard_input_map.clone();
            let names = binding.batch_names.clone();
            (batches, map, names)
        };

        // Take loss_fn out of RefCell to avoid borrow conflicts with &self
        let loss_fn = self.loss_fn.borrow_mut().take();

        let result = if loss_fn.is_some() {
            self.el_che_per_batch_backward(
                &per_device_batches,
                &shard_input_map,
                &batch_names,
                loss_fn.as_deref().unwrap(),
            )
        } else {
            self.el_che_legacy_forward(
                &per_device_batches,
                &shard_input_map,
            )
        };

        // Put loss_fn back
        *self.loss_fn.borrow_mut() = loss_fn;

        result
    }

    /// Per-batch backward El Che path: forward -> loss -> backward per batch.
    ///
    /// Only one forward graph alive at a time. Gradients accumulate across
    /// batches. Returns detached gathered outputs for metrics.
    ///
    /// Round-robin submission: batches are interleaved across devices
    /// (batch 0 on each device, then batch 1, ...) so GPU streams overlap
    /// and VRAM peaks are distributed evenly.
    fn el_che_per_batch_backward(
        &self,
        per_device_batches: &[Vec<Vec<Tensor>>],
        shard_input_map: &[usize],
        batch_names: &[String],
        loss_fn: &dyn Fn(&LossContext) -> Result<Variable>,
    ) -> Result<Variable> {
        use crate::distributed::cuda_event::CudaEventFlags;
        use crate::tensor::set_current_cuda_device;

        let (_n, devices, gather_device) = self.el_che_read_config()?;
        let has_tags = !self.tag_capture.is_empty();
        let has_traces = self.nodes.iter().any(|nd| nd.trace_buf.is_some());
        let device_indices = Self::cuda_device_indices(&devices);

        let batch_counts: Vec<usize> = per_device_batches.iter()
            .map(|b| b.len()).collect();
        let max_batches = batch_counts.iter().copied().max().unwrap_or(0);

        let mut all_outputs: Vec<Variable> = Vec::new();
        let mut gathered_tags: HashMap<String, Vec<Variable>> = HashMap::new();
        let mut gathered_traces: HashMap<(String, usize), Vec<Variable>> = HashMap::new();

        // Record start events on all device streams
        let timing_starts = Self::record_events_all(&device_indices, CudaEventFlags::Default)?;

        // Round-robin: one batch per device at a time
        for batch_idx in 0..max_batches {
            for (rank, device_batches) in per_device_batches.iter().enumerate() {
                if batch_idx >= device_batches.len() {
                    continue;
                }
                let batch_tensors = &device_batches[batch_idx];
                set_current_cuda_device(device_indices[rank]);

                let graph_inputs: Vec<Variable> = shard_input_map.iter()
                    .map(|&idx| Variable::new(batch_tensors[idx].clone(), false))
                    .collect();

                // Forward
                let out = self.el_che_forward_on_rank(rank, &graph_inputs)?;

                // Snapshot tags and traces (live autograd) for the loss closure
                let tags = self.el_che_snapshot_tags(rank, has_tags)?;
                let traces = self.el_che_snapshot_traces(rank, has_traces);

                // Reconstruct Batch with all fields (inputs + targets)
                let batch = Batch::new(batch_tensors.clone(), batch_names.to_vec());

                // Call loss closure and backward (frees forward graph)
                let ctx = LossContext {
                    output: &out,
                    batch: &batch,
                    tags: &tags,
                    traces: &traces,
                };
                let loss = loss_fn(&ctx)?;
                loss.backward()?;

                // Gather detached output for metrics
                let detached = out.detach();
                all_outputs.push(Self::move_to(detached, gather_device)?);

                if has_tags || has_traces {
                    Self::gather_detached_tags(
                        &tags, gather_device, &mut gathered_tags,
                    )?;
                    Self::gather_detached_traces(
                        &traces, gather_device, &mut gathered_traces,
                    )?;
                }
            }
        }

        // Record end events on all device streams
        let timing_ends = Self::record_events_all(&device_indices, CudaEventFlags::Default)?;
        let timing = Self::zip_timing(timing_starts, timing_ends);

        self.el_che_store_timing(batch_counts, timing);
        self.el_che_set_gathered_tags(has_tags, &gathered_tags)?;
        self.el_che_set_gathered_traces(&gathered_traces)?;
        Self::cat_outputs(all_outputs)
    }

    /// Legacy El Che path: all forward graphs held simultaneously.
    /// User calls backward on the gathered output.
    ///
    /// Round-robin submission for GPU stream overlap.
    fn el_che_legacy_forward(
        &self,
        per_device_batches: &[Vec<Vec<Tensor>>],
        shard_input_map: &[usize],
    ) -> Result<Variable> {
        use crate::distributed::cuda_event::CudaEventFlags;
        use crate::tensor::set_current_cuda_device;

        let (_n, devices, gather_device) = self.el_che_read_config()?;
        let has_tags = !self.tag_capture.is_empty();
        let has_traces = self.nodes.iter().any(|nd| nd.trace_buf.is_some());
        let device_indices = Self::cuda_device_indices(&devices);

        let batch_counts: Vec<usize> = per_device_batches.iter()
            .map(|b| b.len()).collect();
        let max_batches = batch_counts.iter().copied().max().unwrap_or(0);

        let mut all_outputs: Vec<Variable> = Vec::new();
        let mut gathered_tags: HashMap<String, Vec<Variable>> = HashMap::new();
        let mut gathered_traces: HashMap<(String, usize), Vec<Variable>> = HashMap::new();

        // Record start events on all device streams
        let timing_starts = Self::record_events_all(&device_indices, CudaEventFlags::Default)?;

        // Round-robin: one batch per device at a time
        for batch_idx in 0..max_batches {
            for (rank, device_batches) in per_device_batches.iter().enumerate() {
                if batch_idx >= device_batches.len() {
                    continue;
                }
                let batch_tensors = &device_batches[batch_idx];
                set_current_cuda_device(device_indices[rank]);

                let graph_inputs: Vec<Variable> = shard_input_map.iter()
                    .map(|&idx| Variable::new(batch_tensors[idx].clone(), false))
                    .collect();

                let out = self.el_che_forward_on_rank(rank, &graph_inputs)?;

                all_outputs.push(Self::move_to(out, gather_device)?);

                if has_tags || has_traces {
                    if rank == 0 {
                        Self::gather_tags_and_traces(
                            self, gather_device, has_tags, has_traces,
                            &mut gathered_tags, &mut gathered_traces,
                        )?;
                    } else {
                        let dist = self.distributed.borrow();
                        let dist = dist.as_ref().unwrap();
                        if let Some(g) = dist.replicas[rank - 1].as_graph() {
                            Self::gather_tags_and_traces(
                                g, gather_device, has_tags, has_traces,
                                &mut gathered_tags, &mut gathered_traces,
                            )?;
                        }
                    }
                }
            }
        }

        // Record end events on all device streams
        let timing_ends = Self::record_events_all(&device_indices, CudaEventFlags::Default)?;
        let timing = Self::zip_timing(timing_starts, timing_ends);

        self.el_che_store_timing(batch_counts, timing);
        self.el_che_set_gathered_tags(has_tags, &gathered_tags)?;
        self.el_che_set_gathered_traces(&gathered_traces)?;
        Self::cat_outputs(all_outputs)
    }

    // -- El Che helpers -------------------------------------------------------

    /// Forward on a specific rank (rank 0 = self, rank > 0 = replica).
    fn el_che_forward_on_rank(&self, rank: usize, graph_inputs: &[Variable]) -> Result<Variable> {
        if rank == 0 {
            self.forward_impl(graph_inputs)
        } else {
            let dist = self.distributed.borrow();
            let dist = dist.as_ref().unwrap();
            let replica = &dist.replicas[rank - 1];
            match replica.as_graph() {
                Some(g) => g.forward_impl(graph_inputs),
                None => replica.forward(&graph_inputs[0]),
            }
        }
    }

    /// Move a Variable to the target device, or return it unchanged if already there.
    fn move_to(var: Variable, target: crate::tensor::Device) -> Result<Variable> {
        if var.data().device() != target {
            var.to_device(target)
        } else {
            Ok(var)
        }
    }

    /// Extract CUDA device indices (0 for CPU devices).
    fn cuda_device_indices(devices: &[crate::tensor::Device]) -> Vec<u8> {
        devices.iter().map(|d| match d {
            crate::tensor::Device::CUDA(i) => *i,
            _ => 0,
        }).collect()
    }

    /// Record a CudaEvent on each device stream.
    fn record_events_all(
        device_indices: &[u8],
        flags: crate::distributed::cuda_event::CudaEventFlags,
    ) -> Result<Vec<crate::distributed::cuda_event::CudaEvent>> {
        use crate::distributed::cuda_event::CudaEvent;
        use crate::tensor::set_current_cuda_device;
        let mut events = Vec::with_capacity(device_indices.len());
        for &idx in device_indices {
            set_current_cuda_device(idx);
            let ev = CudaEvent::new(flags)?;
            ev.record()?;
            events.push(ev);
        }
        Ok(events)
    }

    /// Zip start/end event Vecs into timing pairs.
    fn zip_timing(
        starts: Vec<crate::distributed::cuda_event::CudaEvent>,
        ends: Vec<crate::distributed::cuda_event::CudaEvent>,
    ) -> Vec<(crate::distributed::cuda_event::CudaEvent, crate::distributed::cuda_event::CudaEvent)> {
        starts.into_iter().zip(ends).collect()
    }

    /// Read distributed config for El Che forward paths.
    fn el_che_read_config(&self) -> Result<(usize, Vec<crate::tensor::Device>, crate::tensor::Device)> {
        let dist = self.distributed.borrow();
        let dist = dist.as_ref().unwrap();
        let n = dist.devices.len();
        let devices = dist.devices.clone();
        let gather_device = self.data_binding.borrow()
            .as_ref()
            .map(|b| b.loader.device())
            .unwrap_or(devices[0]);
        Ok((n, devices, gather_device))
    }

    /// Snapshot tagged outputs from the graph that ran forward (rank 0 = self).
    fn el_che_snapshot_tags(
        &self,
        rank: usize,
        has_tags: bool,
    ) -> Result<HashMap<String, Variable>> {
        if !has_tags {
            return Ok(HashMap::new());
        }
        if rank == 0 {
            Ok(self.tagged_outputs.borrow().clone())
        } else {
            let dist = self.distributed.borrow();
            let dist = dist.as_ref().unwrap();
            match dist.replicas[rank - 1].as_graph() {
                Some(g) => Ok(g.tagged_outputs.borrow().clone()),
                None => Ok(HashMap::new()),
            }
        }
    }

    /// Snapshot loop traces from the graph that ran forward (rank 0 = self).
    fn el_che_snapshot_traces(
        &self,
        rank: usize,
        has_traces: bool,
    ) -> HashMap<String, Vec<Variable>> {
        let mut result = HashMap::new();
        if !has_traces {
            return result;
        }
        let collect_from = |g: &Graph| -> HashMap<String, Vec<Variable>> {
            let mut r = HashMap::new();
            for tag_name in g.tag_names() {
                if let Some(traces) = g.traces(&tag_name) {
                    r.insert(tag_name, traces);
                }
            }
            r
        };
        if rank == 0 {
            result = collect_from(self);
        } else {
            let dist = self.distributed.borrow();
            let dist = dist.as_ref().unwrap();
            if let Some(g) = dist.replicas[rank - 1].as_graph() {
                result = collect_from(g);
            }
        }
        result
    }

    /// Gather detached tags into the accumulator (for per-batch backward path).
    fn gather_detached_tags(
        tags: &HashMap<String, Variable>,
        gather_device: crate::tensor::Device,
        gathered: &mut HashMap<String, Vec<Variable>>,
    ) -> Result<()> {
        for (name, var) in tags {
            let detached = var.detach();
            let moved = if detached.data().device() != gather_device {
                detached.to_device(gather_device)?
            } else {
                detached
            };
            gathered.entry(name.clone()).or_default().push(moved);
        }
        Ok(())
    }

    /// Gather detached traces into the accumulator (for per-batch backward path).
    fn gather_detached_traces(
        traces: &HashMap<String, Vec<Variable>>,
        gather_device: crate::tensor::Device,
        gathered: &mut HashMap<(String, usize), Vec<Variable>>,
    ) -> Result<()> {
        for (tag_name, step_vars) in traces {
            for (step_idx, var) in step_vars.iter().enumerate() {
                let detached = var.detach();
                let moved = if detached.data().device() != gather_device {
                    detached.to_device(gather_device)?
                } else {
                    detached
                };
                gathered
                    .entry((tag_name.clone(), step_idx))
                    .or_default()
                    .push(moved);
            }
        }
        Ok(())
    }

    /// Store batch counts and timing on DistributedState for step().
    fn el_che_store_timing(
        &self,
        batch_counts: Vec<usize>,
        timing: Vec<(crate::distributed::cuda_event::CudaEvent, crate::distributed::cuda_event::CudaEvent)>,
    ) {
        let mut dist = self.distributed.borrow_mut();
        let dist = dist.as_mut().unwrap();
        dist.last_el_che_counts = batch_counts;
        dist.last_timing = Some(timing);
    }

    /// Set gathered tagged outputs on the main graph (catted across batches/devices).
    fn el_che_set_gathered_tags(
        &self,
        has_tags: bool,
        gathered_tags: &HashMap<String, Vec<Variable>>,
    ) -> Result<()> {
        if has_tags && !gathered_tags.is_empty() {
            let mut tagged = self.tagged_outputs.borrow_mut();
            tagged.clear();
            for (name, vars) in gathered_tags {
                if vars.len() == 1 {
                    tagged.insert(name.clone(), vars[0].clone());
                } else {
                    let refs: Vec<&Variable> = vars.iter().collect();
                    tagged.insert(name.clone(), Variable::cat_many(&refs, 0)?);
                }
            }
        }
        Ok(())
    }

    /// Set gathered loop traces on the main graph (catted per step across batches/devices).
    fn el_che_set_gathered_traces(
        &self,
        gathered_traces: &HashMap<(String, usize), Vec<Variable>>,
    ) -> Result<()> {
        if !gathered_traces.is_empty() {
            let mut by_tag: HashMap<String, Vec<(usize, Variable)>> = HashMap::new();
            for ((tag_name, step_idx), vars) in gathered_traces {
                let catted = if vars.len() == 1 {
                    vars[0].clone()
                } else {
                    let refs: Vec<&Variable> = vars.iter().collect();
                    Variable::cat_many(&refs, 0)?
                };
                by_tag.entry(tag_name.clone()).or_default().push((*step_idx, catted));
            }
            for (tag_name, mut steps) in by_tag {
                steps.sort_by_key(|(idx, _)| *idx);
                let ordered: Vec<Variable> = steps.into_iter().map(|(_, v)| v).collect();
                self.set_traces(&tag_name, ordered);
            }
        }
        Ok(())
    }

    /// Cat output Variables along dim 0, or return the single one.
    fn cat_outputs(outputs: Vec<Variable>) -> Result<Variable> {
        if outputs.len() == 1 {
            return Ok(outputs.into_iter().next().unwrap());
        }
        let refs: Vec<&Variable> = outputs.iter().collect();
        Variable::cat_many(&refs, 0)
    }

    /// Batch-aware forward pass.
    ///
    /// Extracts the primary input and auxiliary graph inputs from the named
    /// Batch, handles DDP presharding and El Che transparently.
    ///
    /// ```ignore
    /// let out = model.forward_batch(&b)?;
    /// let loss = mse_loss(&out, &b["letter"])?;
    /// ```
    pub fn forward_batch(&self, batch: &crate::data::Batch) -> Result<Variable> {
        // Scope the borrow so it is released before calling methods that re-borrow.
        let (has_shards, has_el_che_batches, forward_input_name, shard_input_map) = {
            let guard = self.data_binding.borrow();
            let binding = guard.as_ref()
                .expect("call set_data_loader before forward_batch");

            let is_dist = self.distributed.borrow().is_some();
            let has_shards = is_dist && binding.loader.has_shards();
            let has_el_che = is_dist && binding.loader.has_el_che_batches();
            let name = binding.forward_input.clone();
            let map = binding.shard_input_map.clone();
            (has_shards, has_el_che, name, map)
        };

        // El Che path: multi-batch per device
        if has_el_che_batches {
            return self.forward_distributed_el_che();
        }

        // Standard presharded path
        if has_shards {
            return self.forward_distributed_presharded();
        }

        // Build full input vector from batch using shard_input_map
        let batch_names = batch.names();
        let graph_inputs: Vec<Variable> = shard_input_map.iter()
            .map(|&idx| Variable::new(batch[batch_names[idx].as_str()].clone(), false))
            .collect();

        if graph_inputs.is_empty() {
            return Err(TensorError::new(&format!(
                "forward_batch: batch missing forward input '{}'",
                forward_input_name,
            )));
        }

        if self.distributed.borrow().is_some() {
            self.forward_distributed_scatter(&graph_inputs[0])
        } else {
            self.forward_impl(&graph_inputs)
        }
    }
}
