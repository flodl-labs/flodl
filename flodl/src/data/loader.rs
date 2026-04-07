//! DataLoader: async data pipeline with automatic prefetching.
//!
//! Manages a background pipeline that keeps GPU(s) fed with data.
//! Supports two modes:
//!
//! - **Resident**: entire dataset fits in VRAM. Loaded once, reshuffled per epoch.
//! - **Streaming**: prefetch ring buffer with async H2D transfers.
//!
//! Mode is auto-detected at build time based on available VRAM.
//!
//! Build the model first, then the loader, so VRAM probing reflects
//! actual free memory after model allocation.

use std::sync::Arc;

use super::prefetch::PrefetchWorker;
use super::sampler::{RandomSampler, Sampler, SequentialSampler};
use super::{Batch, BatchDataSet, DataSet, DataSetAdapter};
use crate::tensor::{Device, Result, Tensor, TensorError};

/// Default fraction of total VRAM to use. Reserves 10% for activations,
/// gradients, and CUDA allocator overhead.
const VRAM_MAX_USAGE: f64 = 0.90;

/// Check whether the full dataset fits in VRAM (or RAM for CPU).
///
/// For CPU targets, always returns true (RAM is plentiful).
/// For CUDA targets, probes free VRAM and checks if the dataset fits
/// within the headroom budget.
fn can_fit_resident(n: usize, per_sample_bytes: usize, device: Device) -> bool {
    if !device.is_cuda() {
        return true;
    }

    let total_bytes = per_sample_bytes as u64 * n as u64;
    let idx = device.index() as i32;

    match crate::tensor::cuda_memory_info_idx(idx) {
        Ok((free, total)) => {
            let used = total.saturating_sub(free);
            let cap = (total as f64 * VRAM_MAX_USAGE) as u64;
            let budget = cap.saturating_sub(used);
            total_bytes < budget
        }
        Err(_) => false, // can't probe -> assume won't fit
    }
}

/// Bootstrap prefetch depth: small buffer for the period between
/// `build()` and the first `epoch()` call. The real depth is computed
/// at `epoch()` time when free VRAM reflects actual model allocation.
const BOOTSTRAP_PREFETCH: usize = 4;

/// Compute prefetch depth from VRAM usage cap.
///
/// `max_usage` is the fraction of **total** VRAM to use (default 0.90).
/// The prefetch budget is the gap between current usage and the cap,
/// minus `activation_reserve` bytes reserved for forward/backward
/// activation memory and gradients.
///
/// Called at each `epoch()` boundary. By that point the model, optimizer,
/// and any other allocations are done, so current usage is the real baseline.
pub(crate) fn prefetch_depth_from_vram(
    per_sample_bytes: usize,
    batch_size: usize,
    device: Device,
    max_usage: f64,
    activation_reserve: usize,
) -> usize {
    if !device.is_cuda() {
        return 2; // CPU: just double-buffer
    }

    let batch_bytes = per_sample_bytes * batch_size;
    if batch_bytes == 0 {
        return 2;
    }

    let idx = device.index() as i32;
    let (free, total) = crate::tensor::cuda_memory_info_idx(idx)
        .unwrap_or((0, 0));

    let used = (total as usize).saturating_sub(free as usize);
    let cap = (total as f64 * max_usage.clamp(0.5, 0.99)) as usize;
    let budget = cap.saturating_sub(used + activation_reserve);

    budget / batch_bytes
}

// ---------------------------------------------------------------------------
// DataLoaderBuilder
// ---------------------------------------------------------------------------

/// Builder for [`DataLoader`]. Constructed via
/// [`DataLoader::from_dataset`] or [`DataLoader::from_batch_dataset`].
pub struct DataLoaderBuilder {
    dataset: Box<dyn BatchDataSet>,
    batch_size: usize,
    device: Device,
    sampler: Option<Box<dyn Sampler>>,
    prefetch_depth: Option<usize>,
    seed: u64,
    drop_last: bool,
    force_streaming: bool,
    names: Option<Vec<String>>,
    vram_max_usage: f64,
}

impl DataLoaderBuilder {
    fn new(dataset: Box<dyn BatchDataSet>) -> Self {
        DataLoaderBuilder {
            dataset,
            batch_size: 0,
            device: Device::CPU,
            sampler: None,
            prefetch_depth: None,
            seed: 42,
            drop_last: true,
            force_streaming: false,
            names: None,
            vram_max_usage: 0.90,
        }
    }

    /// Set the batch size. Required.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Target device for loaded batches. Default: `Device::CPU`.
    ///
    /// For single-GPU training, set to `Device::CUDA(0)`. Data arrives
    /// on the GPU ready for forward pass.
    ///
    /// For DDP training, leave as `Device::CPU` -- data arrives in pinned
    /// memory and `forward_distributed` scatters to devices efficiently.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set the RNG seed for shuffling. Default: 42.
    ///
    /// Each epoch derives its permutation from `seed + epoch`, so different
    /// epochs produce different orderings but the same seed is always
    /// reproducible.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable or disable shuffling. Default: `true` (uses [`RandomSampler`]).
    ///
    /// When `false`, uses [`SequentialSampler`] (indices in order every epoch).
    /// This is overridden if [`sampler`](DataLoaderBuilder::sampler) is called.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        if !shuffle {
            let n = self.dataset.len();
            self.sampler = Some(Box::new(SequentialSampler::new(n)));
        }
        self
    }

    /// Custom sampler. Overrides the [`shuffle`](DataLoaderBuilder::shuffle) setting.
    pub fn sampler(mut self, sampler: Box<dyn Sampler>) -> Self {
        self.sampler = Some(sampler);
        self
    }

    /// Override auto-detected prefetch depth (streaming mode only).
    ///
    /// Auto-detection fills `(1 - margin)` of free VRAM at build time.
    /// Use this to set a specific depth instead. Disables automatic
    /// per-epoch adaptation.
    ///
    /// Set to 0 for synchronous loading (no background thread).
    pub fn prefetch(mut self, depth: usize) -> Self {
        self.prefetch_depth = Some(depth);
        self
    }

    /// Maximum fraction of total VRAM to use for prefetch (streaming mode).
    ///
    /// Default: 0.90 (use up to 90% of total VRAM). At each `epoch()` call,
    /// the loader probes current VRAM usage and fills the gap between that
    /// usage and the cap with prefetch batches. The remaining headroom covers
    /// activation memory, gradients, and CUDA allocator overhead.
    ///
    /// The budget is computed at `epoch()` time (not `build()`), so the model
    /// can be loaded in any order. Clamped to `[0.50, 0.99]`.
    pub fn vram_max_usage(mut self, max_usage: f64) -> Self {
        self.vram_max_usage = max_usage.clamp(0.50, 0.99);
        self
    }

    /// Force streaming mode even when the dataset fits in memory.
    ///
    /// Useful for preserving VRAM headroom, testing the prefetch pipeline,
    /// or benchmarking resident vs streaming performance.
    pub fn streaming(mut self) -> Self {
        self.force_streaming = true;
        self
    }

    /// Name the tensor positions in each batch.
    ///
    /// Names enable `batch["image"]` access alongside positional `batch[0]`.
    /// The number of names must match the number of tensors returned by the
    /// dataset's `get()` / `get_batch()`.
    ///
    /// If not called, auto-generated positional names ("0", "1", ...) are used.
    ///
    /// ```ignore
    /// let loader = DataLoader::from_dataset(data)
    ///     .batch_size(64)
    ///     .names(&["image", "letter", "case", "origin"])
    ///     .build()?;
    ///
    /// let b = loader.epoch(0).next().unwrap()?;
    /// let images = &b["image"];
    /// ```
    pub fn names(mut self, names: &[&str]) -> Self {
        self.names = Some(names.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Drop the last incomplete batch if dataset size is not divisible
    /// by batch_size. Default: `true`.
    ///
    /// Defaulting to `true` avoids the well-known BatchNorm footgun:
    /// a final batch of size 1 produces NaN variance. Set to `false`
    /// for evaluation or inference where every sample matters.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Build the DataLoader.
    ///
    /// Performs auto-detection of resident vs streaming mode based on
    /// available VRAM. For resident mode, loads the entire dataset into
    /// GPU memory at this point.
    ///
    /// Build the model first, then the loader, so VRAM probing reflects
    /// actual free memory after model allocation.
    pub fn build(self) -> Result<DataLoader> {
        if self.dataset.is_empty() {
            return Err(TensorError::new("DataLoader: empty dataset"));
        }
        if self.batch_size == 0 {
            return Err(TensorError::new("DataLoader: batch_size must be > 0"));
        }

        // Destructure to avoid partial-move issues
        let DataLoaderBuilder {
            dataset,
            batch_size,
            device,
            sampler,
            prefetch_depth,
            seed,
            drop_last,
            force_streaming,
            names,
            vram_max_usage,
        } = self;

        let n = dataset.len();

        // Probe dataset size for mode decision
        let sample = dataset.get_batch(&[0])?;
        if sample.is_empty() {
            return Err(TensorError::new(
                "DataLoader: dataset returned empty tensor list",
            ));
        }
        let num_tensors = sample.len();
        let per_sample_bytes: usize = sample.iter().map(|t| t.nbytes()).sum();
        drop(sample);

        // Resolve names: validate if provided, auto-generate if not
        let names = match names {
            Some(ref n) if n.len() != num_tensors => {
                return Err(TensorError::new(&format!(
                    "DataLoader: names count ({}) does not match dataset tensor count ({})",
                    n.len(),
                    num_tensors,
                )));
            }
            Some(n) => n,
            None => (0..num_tensors).map(|i| i.to_string()).collect(),
        };

        let use_resident = !force_streaming && can_fit_resident(n, per_sample_bytes, device);

        // Wrap in Arc early so both paths can share it, and OOM fallback
        // from resident to streaming keeps the dataset alive.
        let dataset: Arc<dyn BatchDataSet> = Arc::from(dataset);
        let shuffle = sampler.is_none();

        let sampler = sampler.unwrap_or_else(|| {
            Box::new(RandomSampler::new(n, seed))
        });

        let user_set_depth = prefetch_depth.is_some();
        // Bootstrap depth: small buffer to start. The real depth is
        // computed at epoch() time when free VRAM reflects the actual
        // model allocation. User override skips adaptive sizing.
        let streaming_depth = prefetch_depth.unwrap_or(BOOTSTRAP_PREFETCH);
        if use_resident {
            match build_resident(Arc::clone(&dataset), batch_size, device, sampler, drop_last, names.clone()) {
                Ok(loader) => Ok(loader),
                Err(e) if device.is_cuda() && e.is_cuda_oom() => {
                    // VRAM estimate was wrong, fall back to streaming.
                    // Recreate sampler since build_resident consumed it.
                    let sampler: Box<dyn Sampler> = if shuffle {
                        Box::new(RandomSampler::new(n, seed))
                    } else {
                        Box::new(SequentialSampler::new(n))
                    };
                    crate::tensor::cuda_empty_cache();
                    build_streaming(dataset, batch_size, device, sampler, drop_last, streaming_depth, per_sample_bytes, vram_max_usage, user_set_depth, names)
                }
                Err(e) => Err(e),
            }
        } else {
            build_streaming(dataset, batch_size, device, sampler, drop_last, streaming_depth, per_sample_bytes, vram_max_usage, user_set_depth, names)
        }
    }
}

fn build_resident(
    dataset: Arc<dyn BatchDataSet>,
    batch_size: usize,
    device: Device,
    sampler: Box<dyn Sampler>,
    drop_last: bool,
    names: Vec<String>,
) -> Result<DataLoader> {
    let n = dataset.len();
    let all_indices: Vec<usize> = (0..n).collect();
    let tensors = dataset.get_batch(&all_indices)?;

    if tensors.is_empty() {
        return Err(TensorError::new(
            "DataLoader: dataset returned empty tensor list",
        ));
    }

    let gpu_data = if device.is_cuda() {
        let mut on_device = Vec::with_capacity(tensors.len());
        for t in &tensors {
            let pinned = t.pin_memory()?;
            on_device.push(pinned.to_device(device)?);
        }
        on_device
    } else {
        tensors
    };

    Ok(DataLoader {
        inner: LoaderInner::Resident(ResidentLoader {
            gpu_data,
            _dataset: dataset,
            device,
            batch_size,
            sampler,
            drop_last,
            names,
        }),
    })
}

#[allow(clippy::too_many_arguments)]
fn build_streaming(
    dataset: Arc<dyn BatchDataSet>,
    batch_size: usize,
    device: Device,
    sampler: Box<dyn Sampler>,
    drop_last: bool,
    prefetch_depth: usize,
    per_sample_bytes: usize,
    vram_max_usage: f64,
    user_set_depth: bool,
    names: Vec<String>,
) -> Result<DataLoader> {
    let worker = PrefetchWorker::new(Arc::clone(&dataset), device, prefetch_depth);

    Ok(DataLoader {
        inner: LoaderInner::Streaming(StreamingLoader {
            _dataset: dataset,
            batch_size,
            device,
            sampler,
            drop_last,
            worker,
            names,
            per_sample_bytes,
            vram_max_usage,
            user_set_depth,
        }),
    })
}

// ---------------------------------------------------------------------------
// DataLoader
// ---------------------------------------------------------------------------

/// Async data loader with automatic prefetching and device transfer.
///
/// Manages a background pipeline that keeps GPU(s) fed with data.
///
/// # Construction
///
/// ```ignore
/// let loader = DataLoader::from_dataset(my_data)
///     .batch_size(64)
///     .device(Device::CUDA(0))
///     .build()?;
/// ```
///
/// # Training loop
///
/// ```ignore
/// for epoch in 0..100 {
///     for batch in loader.epoch(epoch) {
///         let b = batch?;
///         let input = Variable::new(b[0].clone(), true);
///         let target = Variable::new(b[1].clone(), false);
///         let pred = model.forward(&input)?;
///         let loss = mse_loss(&pred, &target)?;
///         loss.backward()?;
///         model.step()?;
///     }
/// }
/// ```
pub struct DataLoader {
    pub(crate) inner: LoaderInner,
}

pub(crate) enum LoaderInner {
    Resident(ResidentLoader),
    Streaming(StreamingLoader),
    Distributed(DistributedLoader),
}

impl DataLoader {
    /// Access the internal loader variant (for Graph integration).
    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &LoaderInner {
        &self.inner
    }
}

impl DataLoader {
    /// Create a DataLoader from a per-item [`DataSet`].
    ///
    /// Items are automatically stacked into batches.
    pub fn from_dataset<D: DataSet + 'static>(dataset: D) -> DataLoaderBuilder {
        DataLoaderBuilder::new(Box::new(DataSetAdapter { inner: dataset }))
    }

    /// Create a DataLoader from a per-batch [`BatchDataSet`].
    ///
    /// The dataset is responsible for returning properly batched tensors.
    pub fn from_batch_dataset<D: BatchDataSet + 'static>(dataset: D) -> DataLoaderBuilder {
        DataLoaderBuilder::new(Box::new(dataset))
    }

    /// Get an epoch iterator.
    ///
    /// Each call reshuffles the data (if using a random sampler) and
    /// returns an iterator over batches. Each batch is a [`Batch`]
    /// containing tensors already on the target device.
    ///
    /// The epoch number is passed to the sampler for deterministic
    /// reproducibility.
    ///
    /// For distributed loaders, use `Graph::epoch()` instead (which
    /// provides chunk_ratios from the auto-balancer).
    pub fn epoch(&mut self, epoch: usize) -> EpochIterator<'_> {
        match &mut self.inner {
            LoaderInner::Resident(loader) => loader.epoch(epoch),
            LoaderInner::Streaming(loader) => loader.epoch(epoch),
            LoaderInner::Distributed(_) => {
                panic!("DataLoader: distributed mode requires Graph::epoch(), not direct epoch()")
            }
        }
    }

    /// Number of samples in the dataset.
    pub fn len(&self) -> usize {
        match &self.inner {
            LoaderInner::Resident(l) => l.sampler.len(),
            LoaderInner::Streaming(l) => l.sampler.len(),
            LoaderInner::Distributed(l) => l.sampler.borrow().len(),
        }
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of batches per epoch.
    pub fn num_batches(&self) -> usize {
        let (n, bs, dl) = match &self.inner {
            LoaderInner::Resident(l) => (l.sampler.len(), l.batch_size, l.drop_last),
            LoaderInner::Streaming(l) => (l.sampler.len(), l.batch_size, l.drop_last),
            LoaderInner::Distributed(l) => (l.sampler.borrow().len(), l.batch_size, l.drop_last),
        };
        if dl { n / bs } else { n.div_ceil(bs) }
    }

    /// Batch size.
    pub fn batch_size(&self) -> usize {
        match &self.inner {
            LoaderInner::Resident(l) => l.batch_size,
            LoaderInner::Streaming(l) => l.batch_size,
            LoaderInner::Distributed(l) => l.batch_size,
        }
    }

    /// Target device (for single-device loaders) or gather device (for distributed).
    pub fn device(&self) -> Device {
        match &self.inner {
            LoaderInner::Resident(l) => l.device,
            LoaderInner::Streaming(l) => l.device,
            LoaderInner::Distributed(l) => l.gather_device,
        }
    }

    /// Whether the loader is in resident mode (full dataset in memory on one device).
    pub fn is_resident(&self) -> bool {
        matches!(&self.inner, LoaderInner::Resident(_))
    }

    /// Tensor names for each batch position.
    pub fn names(&self) -> &[String] {
        match &self.inner {
            LoaderInner::Resident(l) => &l.names,
            LoaderInner::Streaming(l) => &l.names,
            LoaderInner::Distributed(l) => &l.names,
        }
    }

    /// Whether the loader is in distributed mode (multi-device backends).
    pub fn is_distributed(&self) -> bool {
        matches!(&self.inner, LoaderInner::Distributed(_))
    }

    /// Current prefetch depth (streaming mode). Returns 0 for resident loaders.
    pub fn prefetch_depth(&self) -> usize {
        match &self.inner {
            LoaderInner::Resident(_) => 0,
            LoaderInner::Streaming(l) => l.worker.prefetch_depth(),
            LoaderInner::Distributed(l) => {
                l.backends.iter().filter_map(|b| match b {
                    DeviceBackend::Streaming { worker, .. } => Some(worker.prefetch_depth()),
                    _ => None,
                }).max().unwrap_or(0)
            }
        }
    }

    /// Set prefetch depth for streaming backends. Takes effect on the next epoch.
    ///
    /// Disables automatic resize (the loader won't override your setting).
    /// No-op for resident loaders (the entire dataset is already in VRAM).
    pub fn set_prefetch_depth(&mut self, depth: usize) {
        match &mut self.inner {
            LoaderInner::Resident(_) => {}
            LoaderInner::Streaming(l) => {
                l.worker.set_prefetch_depth(depth);
                l.user_set_depth = true;
            }
            LoaderInner::Distributed(l) => {
                for backend in &mut l.backends {
                    if let DeviceBackend::Streaming { worker, .. } = backend {
                        worker.set_prefetch_depth(depth);
                    }
                }
            }
        }
    }

    /// Measure free VRAM and resize prefetch buffers to fill available space.
    ///
    /// **This happens automatically** at every epoch boundary (epoch 1+).
    /// The loader re-probes free VRAM each epoch and fills 90% of it.
    /// You only need to call this manually if you want to resize at a
    /// different point (e.g., mid-epoch during an AllReduce window).
    ///
    /// Calling this (or [`set_prefetch_depth`](DataLoader::set_prefetch_depth))
    /// disables automatic adaptation -- the loader assumes you're managing
    /// depth yourself.
    ///
    /// The data is static across epochs, so a deeper buffer means more of
    /// the dataset stays in VRAM and fewer H2D transfers are needed. If the
    /// buffer covers the entire epoch, performance converges to resident mode.
    ///
    /// Returns the new prefetch depth (0 for resident loaders).
    pub fn auto_resize(&mut self) -> usize {
        match &mut self.inner {
            LoaderInner::Resident(_) => 0,
            LoaderInner::Streaming(l) => {
                let depth = prefetch_depth_from_vram(l.per_sample_bytes, l.batch_size, l.device, l.vram_max_usage, 0);
                l.worker.set_prefetch_depth(depth);
                l.user_set_depth = true;
                depth
            }
            LoaderInner::Distributed(l) => {
                let bs = l.batch_size;
                let mut max_depth = 0;
                for backend in &mut l.backends {
                    if let DeviceBackend::Streaming { worker, device, per_sample_bytes } = backend {
                        let depth = prefetch_depth_from_vram(*per_sample_bytes, bs, *device, VRAM_MAX_USAGE, 0);
                        worker.set_prefetch_depth(depth);
                        max_depth = max_depth.max(depth);
                    }
                }
                max_depth
            }
        }
    }

    /// Get the shared dataset Arc (for upgrade_distributed to load onto devices).
    pub(crate) fn dataset_arc(&self) -> Result<Arc<dyn BatchDataSet>> {
        match &self.inner {
            LoaderInner::Resident(l) => Ok(Arc::clone(&l._dataset)),
            LoaderInner::Streaming(l) => Ok(Arc::clone(&l._dataset)),
            LoaderInner::Distributed(l) => Ok(Arc::clone(&l.dataset)),
        }
    }

    /// Upgrade this loader to distributed mode with per-device backends.
    ///
    /// Called by `Graph::set_data_loader()`. Replaces the inner loader with
    /// a `DistributedLoader` that has one backend per device (resident or
    /// streaming, chosen per device based on VRAM).
    pub(crate) fn upgrade_distributed(
        &mut self,
        devices: &[Device],
        dataset: Arc<dyn BatchDataSet>,
    ) -> Result<()> {
        // Extract config from current inner
        let (batch_size, sampler_len, drop_last, names, seed) = match &self.inner {
            LoaderInner::Resident(l) => (l.batch_size, l.sampler.len(), l.drop_last, l.names.clone(), 42u64),
            LoaderInner::Streaming(l) => (l.batch_size, l.sampler.len(), l.drop_last, l.names.clone(), 42u64),
            LoaderInner::Distributed(_) => {
                return Err(TensorError::new("DataLoader: already in distributed mode"));
            }
        };

        let prefetch_depth = BOOTSTRAP_PREFETCH;
        let (backends, gather_device, gather_resident_idx) =
            build_distributed_backends(&dataset, devices, prefetch_depth)?;

        let sampler: Box<dyn Sampler> = Box::new(
            super::sampler::RandomSampler::new(sampler_len, seed),
        );

        self.inner = LoaderInner::Distributed(DistributedLoader {
            backends,
            dataset,
            sampler: std::cell::RefCell::new(sampler),
            batch_size,
            drop_last,
            names,
            pending_shards: std::cell::Cell::new(None),
            el_che_counts: std::cell::Cell::new(None),
            pending_el_che_batches: std::cell::Cell::new(None),
            gather_device,
            gather_resident_idx,
            seed,
        });

        Ok(())
    }

    /// Consume and return pre-placed per-rank shards (for `forward_distributed_presharded`).
    pub(crate) fn take_shards(&self) -> Option<Vec<Vec<Tensor>>> {
        match &self.inner {
            LoaderInner::Distributed(l) => l.take_shards(),
            _ => None,
        }
    }

    /// Whether per-rank shards are pending.
    pub(crate) fn has_shards(&self) -> bool {
        match &self.inner {
            LoaderInner::Distributed(l) => l.has_shards(),
            _ => false,
        }
    }

    /// Set El Che per-device batch counts (called by Graph::step).
    pub(crate) fn set_el_che_counts(&self, counts: Vec<usize>) {
        if let LoaderInner::Distributed(l) = &self.inner {
            l.set_el_che_counts(counts);
        }
    }

    /// Consume per-device El Che batches (for forward_distributed_el_che).
    pub(crate) fn take_el_che_batches(&self) -> Option<Vec<Vec<Vec<Tensor>>>> {
        match &self.inner {
            LoaderInner::Distributed(l) => l.take_el_che_batches(),
            _ => None,
        }
    }

    /// Whether El Che batches are pending.
    pub(crate) fn has_el_che_batches(&self) -> bool {
        match &self.inner {
            LoaderInner::Distributed(l) => l.has_el_che_batches(),
            _ => false,
        }
    }

    /// Start a distributed epoch. Returns a `DistributedEpochIterator`.
    #[allow(dead_code)]
    pub(crate) fn epoch_distributed<'a>(
        &'a mut self,
        epoch: usize,
        chunk_ratios: &'a [f64],
    ) -> Result<DistributedEpochIterator<'a>> {
        match &self.inner {
            LoaderInner::Distributed(l) => Ok(DistributedEpochIterator::new(l, epoch, chunk_ratios)),
            _ => Err(TensorError::new("DataLoader: not in distributed mode")),
        }
    }
}

// ---------------------------------------------------------------------------
// ResidentLoader
// ---------------------------------------------------------------------------

pub(crate) struct ResidentLoader {
    /// Full dataset tensors on target device, one per position.
    gpu_data: Vec<Tensor>,
    /// Original dataset (kept for upgrade_distributed).
    _dataset: Arc<dyn BatchDataSet>,
    device: Device,
    batch_size: usize,
    sampler: Box<dyn Sampler>,
    drop_last: bool,
    names: Vec<String>,
}

impl ResidentLoader {
    fn epoch(&mut self, epoch: usize) -> EpochIterator<'_> {
        let indices = self.sampler.indices(epoch);
        let n = indices.len();
        let bs = self.batch_size;

        // Compute batch boundaries
        let mut batch_ranges = Vec::new();
        let mut start = 0;
        while start < n {
            let end = (start + bs).min(n);
            if self.drop_last && (end - start) < bs {
                break;
            }
            batch_ranges.push((start, end - start));
            start = end;
        }

        // Build index tensor on the target device (i64 for index_select)
        let i64_indices: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
        let perm = Tensor::from_i64(
            &i64_indices,
            &[i64_indices.len() as i64],
            self.device,
        )
        .expect("failed to create permutation tensor");

        EpochIterator {
            inner: EpochIteratorInner::Resident(ResidentEpochIter {
                data: &self.gpu_data,
                perm,
                batch_ranges,
                pos: 0,
                names: &self.names,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingLoader
// ---------------------------------------------------------------------------

pub(crate) struct StreamingLoader {
    /// Dataset shared with the worker thread.
    _dataset: Arc<dyn BatchDataSet>,
    batch_size: usize,
    device: Device,
    sampler: Box<dyn Sampler>,
    drop_last: bool,
    worker: PrefetchWorker,
    names: Vec<String>,
    /// Per-sample bytes (for adaptive resize depth calculation).
    per_sample_bytes: usize,
    /// Maximum fraction of total VRAM to use for prefetch.
    vram_max_usage: f64,
    /// True when the user explicitly set depth (`.prefetch()` or `set_prefetch_depth()`).
    /// Skips automatic adaptation so we don't override the user's choice.
    user_set_depth: bool,
}

impl StreamingLoader {
    fn epoch(&mut self, epoch: usize) -> EpochIterator<'_> {
        // Probe VRAM usage and size the prefetch buffer to fill up to cap.
        // At epoch 0 this is the real signal: model is loaded, VRAM is known.
        // At epoch N>0: re-probe in case conditions changed.
        if !self.user_set_depth {
            let depth = prefetch_depth_from_vram(
                self.per_sample_bytes, self.batch_size, self.device, self.vram_max_usage, 0,
            );
            self.worker.set_prefetch_depth(depth);
        }

        let indices = self.sampler.indices(epoch);
        let n = indices.len();
        let bs = self.batch_size;

        // Count batches
        let num_batches = if self.drop_last {
            n / bs
        } else {
            n.div_ceil(bs)
        };

        // Start the epoch: gets a fresh per-epoch batch channel.
        // If the previous epoch was dropped mid-way, the old channel is already
        // closed (old batch_tx dropped by the worker when send fails or epoch ends).
        let batch_rx = self.worker.start_epoch(indices, bs, self.drop_last);

        EpochIterator {
            inner: EpochIteratorInner::Streaming(StreamingEpochIter {
                batch_rx,
                remaining: num_batches,
                names: &self.names,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// EpochIterator
// ---------------------------------------------------------------------------

/// Iterator over batches for one training epoch.
///
/// Created by [`DataLoader::epoch`]. Each element is a
/// `Result<`[`Batch`]`>` containing tensors already on the target device.
///
/// Dropping the iterator mid-epoch is safe and cancels any outstanding
/// prefetch work (in streaming mode).
pub struct EpochIterator<'a> {
    inner: EpochIteratorInner<'a>,
}

enum EpochIteratorInner<'a> {
    Resident(ResidentEpochIter<'a>),
    Streaming(StreamingEpochIter<'a>),
}

struct ResidentEpochIter<'a> {
    data: &'a [Tensor],
    perm: Tensor,
    /// (start_in_perm, batch_len)
    batch_ranges: Vec<(usize, usize)>,
    pos: usize,
    names: &'a [String],
}

struct StreamingEpochIter<'a> {
    batch_rx: std::sync::mpsc::Receiver<Result<super::prefetch::PrefetchedBatch>>,
    remaining: usize,
    names: &'a [String],
}

impl<'a> Iterator for EpochIterator<'a> {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            EpochIteratorInner::Resident(iter) => iter.next(),
            EpochIteratorInner::Streaming(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            EpochIteratorInner::Resident(iter) => {
                let remaining = iter.batch_ranges.len() - iter.pos;
                (remaining, Some(remaining))
            }
            EpochIteratorInner::Streaming(iter) => {
                (iter.remaining, Some(iter.remaining))
            }
        }
    }
}

impl ExactSizeIterator for EpochIterator<'_> {}

impl<'a> ResidentEpochIter<'a> {
    fn next(&mut self) -> Option<Result<Batch>> {
        if self.pos >= self.batch_ranges.len() {
            return None;
        }
        let (start, len) = self.batch_ranges[self.pos];
        self.pos += 1;

        // Slice the permutation tensor for this batch
        let batch_perm = match self.perm.narrow(0, start as i64, len as i64) {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };

        // index_select each tensor position
        let mut tensors = Vec::with_capacity(self.data.len());
        for t in self.data {
            match t.index_select(0, &batch_perm) {
                Ok(selected) => tensors.push(selected),
                Err(e) => return Some(Err(e)),
            }
        }

        Some(Ok(Batch::new(tensors, self.names.to_vec())))
    }
}

impl StreamingEpochIter<'_> {
    fn next(&mut self) -> Option<Result<Batch>> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;

        // Receive the next ready batch from the worker
        match self.batch_rx.recv() {
            Ok(Ok(batch)) => {
                // Wait for async H2D copy to complete (typically instant
                // since the batch was submitted prefetch_depth steps ago)
                #[cfg(feature = "cuda")]
                if let Some(ref event) = batch.ready_event {
                    if let Err(e) = event.synchronize() {
                        return Some(Err(e));
                    }
                }
                Some(Ok(Batch::new(batch.tensors, self.names.to_vec())))
            }
            Ok(Err(e)) => Some(Err(e)),
            Err(_) => {
                // Channel closed (worker stopped or panicked)
                self.remaining = 0;
                Some(Err(TensorError::new(
                    "DataLoader: prefetch worker stopped unexpectedly",
                )))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DistributedLoader (DDP-aware, per-device backends)
// ---------------------------------------------------------------------------

/// Per-device data backend: resident (full dataset in VRAM) or streaming
/// (prefetch worker with async H2D transfers).
///
/// Each device independently chooses its mode based on available VRAM.
pub(crate) enum DeviceBackend {
    Resident {
        gpu_data: Vec<Tensor>,
        device: Device,
    },
    Streaming {
        worker: PrefetchWorker,
        device: Device,
        /// Per-sample bytes (for adaptive resize depth calculation).
        per_sample_bytes: usize,
    },
}

impl DeviceBackend {
    fn device(&self) -> Device {
        match self {
            DeviceBackend::Resident { device, .. } | DeviceBackend::Streaming { device, .. } => *device,
        }
    }

    #[allow(dead_code)]
    fn is_resident(&self) -> bool {
        matches!(self, DeviceBackend::Resident { .. })
    }
}

/// Distributed data loader with per-device backends.
///
/// Created by [`DataLoader::upgrade_distributed`] when `Graph::set_data_loader()`
/// detects a multi-GPU topology. Each device gets its own backend (resident
/// if the dataset fits in its VRAM, streaming otherwise).
pub(crate) struct DistributedLoader {
    /// One backend per device, indexed by rank.
    pub backends: Vec<DeviceBackend>,
    /// Shared dataset (used by streaming backends and for gather fallback).
    pub dataset: Arc<dyn BatchDataSet>,
    /// Epoch shuffling (RefCell for interior mutability via shared references).
    pub sampler: std::cell::RefCell<Box<dyn Sampler>>,
    pub batch_size: usize,
    pub drop_last: bool,
    pub names: Vec<String>,
    /// Pre-computed per-rank shards from last epoch iterator advance.
    /// `pending_shards[rank]` = `Vec<Tensor>` (all tensor positions) on `devices[rank]`.
    /// Set by `DistributedEpochIterator::next()`, consumed by `Graph::forward_distributed_presharded()`.
    pub pending_shards: std::cell::Cell<Option<Vec<Vec<Tensor>>>>,
    /// El Che: per-device batch counts for the current cadence step.
    /// Set by `Graph::step()` after `ElChe::report_timing()`, read by `DistributedEpochIterator::next()`.
    /// `None` means El Che is inactive (standard sharding path).
    pub el_che_counts: std::cell::Cell<Option<Vec<usize>>>,
    /// El Che: per-device complete batches from the last epoch iterator advance.
    /// `[rank][batch_idx][tensor_position]` -- each batch is a complete, unsharded batch on that device.
    /// Set by `DistributedEpochIterator::next()`, consumed by `Graph::forward_distributed_el_che()`.
    pub pending_el_che_batches: std::cell::Cell<Option<Vec<Vec<Vec<Tensor>>>>>,
    /// Device for the user-facing batch (loss computation).
    pub gather_device: Device,
    /// If gather_device is a resident backend, its index. None if gather is CPU.
    pub gather_resident_idx: Option<usize>,
    #[allow(dead_code)]
    pub seed: u64,
}

impl DistributedLoader {
    /// Consume and return the pre-placed per-rank shards.
    /// Returns None if no shards are pending (forward called without epoch advance).
    pub fn take_shards(&self) -> Option<Vec<Vec<Tensor>>> {
        self.pending_shards.take()
    }

    /// Whether shards are pending from the last epoch iterator advance.
    pub fn has_shards(&self) -> bool {
        // Cell<Option<T>> doesn't have a peek, but we can check via take+put
        let val = self.pending_shards.take();
        let has = val.is_some();
        self.pending_shards.set(val);
        has
    }

    /// Set El Che per-device batch counts (called by Graph::step after report_timing).
    pub fn set_el_che_counts(&self, counts: Vec<usize>) {
        self.el_che_counts.set(Some(counts));
    }

    /// Take El Che batch counts (consumed by the epoch iterator each iteration).
    pub fn take_el_che_counts(&self) -> Option<Vec<usize>> {
        self.el_che_counts.take()
    }

    /// Peek whether El Che counts are set.
    pub fn has_el_che_counts(&self) -> bool {
        let val = self.el_che_counts.take();
        let has = val.is_some();
        self.el_che_counts.set(val);
        has
    }

    /// Consume per-device El Che batches (for forward_distributed_el_che).
    pub fn take_el_che_batches(&self) -> Option<Vec<Vec<Vec<Tensor>>>> {
        self.pending_el_che_batches.take()
    }

    /// Whether El Che batches are pending.
    pub fn has_el_che_batches(&self) -> bool {
        let val = self.pending_el_che_batches.take();
        let has = val.is_some();
        self.pending_el_che_batches.set(val);
        has
    }
}

/// Build per-device backends for a distributed loader.
///
/// For each device: probe VRAM, attempt resident loading, fallback to streaming
/// on OOM. Returns the backends and gather device info.
fn build_distributed_backends(
    dataset: &Arc<dyn BatchDataSet>,
    devices: &[Device],
    prefetch_depth: usize,
) -> Result<(Vec<DeviceBackend>, Device, Option<usize>)> {
    let n = dataset.len();
    let all_indices: Vec<usize> = (0..n).collect();

    // Load full dataset to CPU once (shared across all device loads)
    let cpu_tensors = dataset.get_batch(&all_indices)?;
    if cpu_tensors.is_empty() {
        return Err(TensorError::new(
            "DataLoader: dataset returned empty tensor list",
        ));
    }

    let per_sample_bytes: usize = cpu_tensors.iter().map(|t| t.nbytes()).sum();
    let mut backends = Vec::with_capacity(devices.len());

    for &dev in devices {
        if can_fit_resident(n, per_sample_bytes, dev) {
            // Try resident: pin + transfer
            match load_resident_tensors(&cpu_tensors, dev) {
                Ok(gpu_data) => {
                    backends.push(DeviceBackend::Resident { gpu_data, device: dev });
                    continue;
                }
                Err(e) if dev.is_cuda() && e.is_cuda_oom() => {
                    // VRAM estimate wrong, fall back to streaming
                    crate::tensor::cuda_empty_cache();
                }
                Err(e) => return Err(e),
            }
        }
        // Streaming fallback
        let worker = PrefetchWorker::new(Arc::clone(dataset), dev, prefetch_depth);
        backends.push(DeviceBackend::Streaming {
            worker,
            device: dev,
            per_sample_bytes,
        });
    }

    // Select gather device: prefer resident backend with most free VRAM
    let (gather_device, gather_idx) = select_gather_device(&backends);

    Ok((backends, gather_device, gather_idx))
}

/// Transfer CPU tensors to a device via pin_memory.
fn load_resident_tensors(cpu_tensors: &[Tensor], device: Device) -> Result<Vec<Tensor>> {
    let mut gpu_data = Vec::with_capacity(cpu_tensors.len());
    for t in cpu_tensors {
        let pinned = t.pin_memory()?;
        gpu_data.push(pinned.to_device(device)?);
    }
    Ok(gpu_data)
}

/// Pick the gather device: resident backend with most free VRAM,
/// or the primary device when all backends are streaming.
fn select_gather_device(backends: &[DeviceBackend]) -> (Device, Option<usize>) {
    let mut best_idx: Option<usize> = None;
    let mut best_free: u64 = 0;

    for (i, backend) in backends.iter().enumerate() {
        if let DeviceBackend::Resident { device: Device::CUDA(idx), .. } = backend {
            let free = crate::tensor::cuda_memory_info_idx(*idx as i32)
                .map(|(f, _)| f)
                .unwrap_or(0);
            if free > best_free {
                best_free = free;
                best_idx = Some(i);
            }
        }
    }

    match best_idx {
        Some(idx) => (backends[idx].device(), Some(idx)),
        // All streaming: gather on the primary device so targets stay
        // on the same CUDA device as model weights.
        None => (backends[0].device(), None),
    }
}

/// Epoch iterator for distributed training.
///
/// Yields `Result<Batch>` containing target tensors on the gather device.
/// Simultaneously stores per-rank input shards in the `DistributedLoader`
/// for `forward_distributed_presharded()` to consume.
pub struct DistributedEpochIterator<'a> {
    loader: &'a DistributedLoader,
    /// Global permutation for this epoch.
    permutation: Vec<usize>,
    /// Current position in the permutation (sample index, not batch index).
    cursor: usize,
    /// Number of batches remaining.
    remaining: usize,
    /// Per-rank chunk ratios (read from Graph's DistributedState each batch).
    chunk_ratios: &'a [f64],
    /// Streaming batch receivers, one per streaming backend (indexed by rank).
    /// None for resident backends.
    streaming_rx: Vec<Option<std::sync::mpsc::Receiver<Result<super::prefetch::PrefetchedBatch>>>>,
}

impl<'a> DistributedEpochIterator<'a> {
    pub(crate) fn new(
        loader: &'a DistributedLoader,
        epoch: usize,
        chunk_ratios: &'a [f64],
    ) -> Self {
        let permutation = loader.sampler.borrow_mut().indices(epoch);
        let n = permutation.len();
        let bs = loader.batch_size;
        let num_batches = if loader.drop_last { n / bs } else { n.div_ceil(bs) };

        // Open one persistent channel per streaming backend for the entire epoch.
        let streaming_rx: Vec<Option<std::sync::mpsc::Receiver<Result<super::prefetch::PrefetchedBatch>>>> =
            loader.backends.iter().map(|backend| {
                match backend {
                    DeviceBackend::Streaming { worker, .. } => {
                        Some(worker.start_distributed_epoch())
                    }
                    DeviceBackend::Resident { .. } => None,
                }
            }).collect();

        DistributedEpochIterator {
            loader,
            permutation,
            cursor: 0,
            remaining: num_batches,
            chunk_ratios,
            streaming_rx,
        }
    }
}

impl Iterator for DistributedEpochIterator<'_> {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // El Che path: pull complete batches per device
        if self.loader.has_el_che_counts() {
            return self.next_el_che();
        }

        // Standard sharding path
        self.remaining -= 1;

        let bs = self.loader.batch_size;
        let n = self.permutation.len();
        let end = (self.cursor + bs).min(n);
        if self.loader.drop_last && (end - self.cursor) < bs {
            self.remaining = 0;
            return None;
        }

        // Global batch indices from the permutation
        let batch_indices: Vec<usize> = self.permutation[self.cursor..end].to_vec();
        let batch_len = batch_indices.len() as i64;
        self.cursor = end;

        // Compute per-rank shard sizes
        let shard_sizes = compute_shard_sizes_from_ratios(batch_len, self.chunk_ratios);

        // Split batch indices into per-rank slices
        let mut per_rank_shards: Vec<Vec<Tensor>> = Vec::with_capacity(self.loader.backends.len());
        let mut offset = 0usize;

        for (rank, backend) in self.loader.backends.iter().enumerate() {
            let shard_len = shard_sizes[rank] as usize;
            let shard_indices = &batch_indices[offset..offset + shard_len];
            offset += shard_len;

            match backend {
                DeviceBackend::Resident { gpu_data, device } => {
                    // Build index tensor on device, index_select each position
                    let idx_i64: Vec<i64> = shard_indices.iter().map(|&i| i as i64).collect();
                    let idx_tensor = match Tensor::from_i64(
                        &idx_i64,
                        &[idx_i64.len() as i64],
                        *device,
                    ) {
                        Ok(t) => t,
                        Err(e) => return Some(Err(e)),
                    };

                    let mut shard_tensors = Vec::with_capacity(gpu_data.len());
                    for t in gpu_data {
                        match t.index_select(0, &idx_tensor) {
                            Ok(selected) => shard_tensors.push(selected),
                            Err(e) => return Some(Err(e)),
                        }
                    }
                    per_rank_shards.push(shard_tensors);
                }
                DeviceBackend::Streaming { worker, .. } => {
                    // Send shard indices; result arrives on persistent epoch channel.
                    worker.load_batch(shard_indices.to_vec());

                    let rx = self.streaming_rx[rank].as_ref().unwrap();
                    match rx.recv() {
                        Ok(Ok(batch)) => {
                            #[cfg(feature = "cuda")]
                            if let Some(ref event) = batch.ready_event {
                                if let Err(e) = event.synchronize() {
                                    return Some(Err(e));
                                }
                            }
                            per_rank_shards.push(batch.tensors);
                        }
                        Ok(Err(e)) => return Some(Err(e)),
                        Err(_) => {
                            return Some(Err(TensorError::new(
                                "DataLoader: streaming worker stopped unexpectedly",
                            )));
                        }
                    }
                }
            }
        }

        // Build user-facing Batch with targets on the gather device.
        // Targets are all tensor positions (for now). In Step 5/6,
        // forward(&Batch) will filter to target-only fields.
        let user_batch = match self.build_gather_batch(&batch_indices, &per_rank_shards) {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        // Store per-rank shards for forward_distributed_presharded()
        self.loader.pending_shards.set(Some(per_rank_shards));

        Some(Ok(user_batch))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for DistributedEpochIterator<'_> {}

impl DistributedEpochIterator<'_> {
    /// El Che iteration: pull complete batches per device, not shards.
    ///
    /// Each device gets `counts[rank]` complete batches of `batch_size` samples.
    /// Data is loaded to each device independently. The user-facing Batch
    /// contains all targets concatenated (for loss computation on gathered output).
    fn next_el_che(&mut self) -> Option<Result<Batch>> {
        let counts = self.loader.take_el_che_counts().unwrap_or_default();
        let n_devices = counts.len();
        let total_batches: usize = counts.iter().sum();
        if total_batches == 0 {
            self.remaining = 0;
            return None;
        }

        // Every rank must process at least 1 batch per sync point.
        // If fewer batches remain than devices, end the epoch.
        if self.remaining < n_devices {
            self.remaining = 0;
            return None;
        }

        // Clamp if near epoch end, ensuring minimum 1 per rank
        let actual_counts = if total_batches > self.remaining {
            // Scale proportionally to fit remaining batches
            let scale = self.remaining as f64 / total_batches as f64;
            let mut clamped: Vec<usize> = counts.iter()
                .map(|&c| ((c as f64 * scale).floor() as usize).max(1))
                .collect();
            // Trim if we overshot remaining (from the .max(1) floors)
            let mut clamped_total: usize = clamped.iter().sum();
            while clamped_total > self.remaining {
                // Reduce the largest count
                if let Some(max_idx) = clamped.iter().enumerate()
                    .filter(|&(_, &c)| c > 1)
                    .max_by_key(|&(_, &c)| c)
                    .map(|(i, _)| i)
                {
                    clamped[max_idx] -= 1;
                    clamped_total -= 1;
                } else {
                    break; // all at 1, can't reduce further
                }
            }
            // Distribute any remaining deficit
            let mut deficit = self.remaining.saturating_sub(clamped_total);
            for c in &mut clamped {
                if deficit == 0 { break; }
                *c += 1;
                deficit -= 1;
            }
            clamped
        } else {
            counts
        };

        let actual_total: usize = actual_counts.iter().sum();
        if actual_total == 0 {
            self.remaining = 0;
            return None;
        }

        let bs = self.loader.batch_size;
        let n = self.permutation.len();

        // Pull total_batches * batch_size samples from the permutation
        let total_samples = actual_total * bs;
        let avail = n - self.cursor;
        let take_samples = total_samples.min(avail);
        let all_indices: Vec<usize> = self.permutation[self.cursor..self.cursor + take_samples].to_vec();
        self.cursor += take_samples;

        // Route complete batches to each device
        let mut per_device_batches: Vec<Vec<Vec<Tensor>>> = Vec::with_capacity(actual_counts.len());
        let mut sample_offset = 0usize;

        for (rank, &count) in actual_counts.iter().enumerate() {
            let backend = &self.loader.backends[rank];
            let mut device_batches: Vec<Vec<Tensor>> = Vec::with_capacity(count);

            for _ in 0..count {
                let batch_end = (sample_offset + bs).min(all_indices.len());
                if batch_end <= sample_offset {
                    break;
                }
                let batch_indices = &all_indices[sample_offset..batch_end];
                sample_offset = batch_end;

                match self.load_batch_on_device(backend, batch_indices, rank) {
                    Ok(tensors) => device_batches.push(tensors),
                    Err(e) => return Some(Err(e)),
                }
            }

            per_device_batches.push(device_batches);
        }

        self.remaining = self.remaining.saturating_sub(actual_total);

        // Build gathered user batch with all targets concatenated
        let user_batch = match self.build_gather_batch(&all_indices[..take_samples.min(all_indices.len())], &[]) {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        // Store per-device batches for forward_distributed_el_che()
        self.loader.pending_el_che_batches.set(Some(per_device_batches));

        // Re-seed counts for next iteration (step() will overwrite with updated counts)
        self.loader.el_che_counts.set(Some(actual_counts));

        Some(Ok(user_batch))
    }

    /// Load a single batch on a specific device backend.
    fn load_batch_on_device(
        &self,
        backend: &DeviceBackend,
        batch_indices: &[usize],
        rank: usize,
    ) -> Result<Vec<Tensor>> {
        match backend {
            DeviceBackend::Resident { gpu_data, device } => {
                let idx_i64: Vec<i64> = batch_indices.iter().map(|&i| i as i64).collect();
                let idx_tensor = Tensor::from_i64(
                    &idx_i64,
                    &[idx_i64.len() as i64],
                    *device,
                )?;
                let mut tensors = Vec::with_capacity(gpu_data.len());
                for t in gpu_data {
                    tensors.push(t.index_select(0, &idx_tensor)?);
                }
                Ok(tensors)
            }
            DeviceBackend::Streaming { worker, .. } => {
                worker.load_batch(batch_indices.to_vec());
                let rx = self.streaming_rx[rank].as_ref().unwrap();
                match rx.recv() {
                    Ok(Ok(batch)) => {
                        #[cfg(feature = "cuda")]
                        if let Some(ref event) = batch.ready_event {
                            event.synchronize()?;
                        }
                        Ok(batch.tensors)
                    }
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(TensorError::new(
                        "DataLoader: streaming worker stopped unexpectedly",
                    )),
                }
            }
        }
    }

    /// Build the user-facing Batch on the gather device.
    fn build_gather_batch(
        &self,
        batch_indices: &[usize],
        _per_rank_shards: &[Vec<Tensor>],
    ) -> Result<Batch> {
        let names = self.loader.names.clone();

        match self.loader.gather_resident_idx {
            Some(gather_rank) => {
                // Gather from a resident backend: index_select all positions
                if let DeviceBackend::Resident { gpu_data, device } = &self.loader.backends[gather_rank] {
                    let idx_i64: Vec<i64> = batch_indices.iter().map(|&i| i as i64).collect();
                    let idx_tensor = Tensor::from_i64(
                        &idx_i64,
                        &[idx_i64.len() as i64],
                        *device,
                    )?;

                    let mut tensors = Vec::with_capacity(gpu_data.len());
                    for t in gpu_data {
                        tensors.push(t.index_select(0, &idx_tensor)?);
                    }
                    Ok(Batch::new(tensors, names))
                } else {
                    unreachable!("gather_resident_idx points to non-resident backend")
                }
            }
            None => {
                // All streaming: fetch targets from CPU dataset
                let tensors = self.loader.dataset.get_batch(batch_indices)?;
                Ok(Batch::new(tensors, names))
            }
        }
    }
}

/// Compute per-rank shard sizes from chunk ratios.
/// Same logic as DistributedState::compute_shard_sizes but standalone.
fn compute_shard_sizes_from_ratios(batch_size: i64, ratios: &[f64]) -> Vec<i64> {
    let n = ratios.len();
    let mut sizes = Vec::with_capacity(n);
    let mut remaining = batch_size;

    for (i, &ratio) in ratios.iter().enumerate().take(n) {
        if i == n - 1 {
            sizes.push(remaining);
        } else {
            let s = (batch_size as f64 * ratio).round() as i64;
            let s = s.max(1).min(remaining - (n - i - 1) as i64);
            sizes.push(s);
            remaining -= s;
        }
    }

    sizes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------


#[cfg(test)]
#[path = "loader_tests.rs"]
mod tests;
