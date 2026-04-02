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

/// VRAM headroom: fraction of free memory considered usable for resident data.
/// Reserves 25% for model parameters, gradients, activations, and CUDA overhead.
const VRAM_HEADROOM: f64 = 0.75;

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
        Ok((free, _total)) => {
            let usable = (free as f64 * VRAM_HEADROOM) as u64;
            total_bytes < usable
        }
        Err(_) => false, // can't probe -> assume won't fit
    }
}

/// Auto-detect prefetch depth for streaming mode.
///
/// The cap at 4 is conservative for local storage. For network/cloud
/// storage (NFS, S3-FUSE), use `.prefetch(n)` to override.
#[allow(dead_code)] // Phase 3
fn auto_prefetch_depth(
    per_sample_bytes: usize,
    batch_size: usize,
    device: Device,
) -> usize {
    if !device.is_cuda() {
        return 2; // CPU: just double-buffer
    }

    let batch_bytes = per_sample_bytes * batch_size;
    if batch_bytes == 0 {
        return 2;
    }

    let idx = device.index() as i32;
    let free = crate::tensor::cuda_memory_info_idx(idx)
        .map(|(free, _)| free)
        .unwrap_or(0) as usize;

    // Use at most 10% of free VRAM for prefetch buffer
    let budget = free / 10;
    (budget / batch_bytes).clamp(2, 4)
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
    /// The default auto-detection caps at 4 batches, which is sufficient
    /// for local storage. For high-latency scenarios (cloud/NFS/S3-FUSE),
    /// increase this to hide network round-trip times.
    ///
    /// Set to 0 for synchronous loading (no background thread).
    pub fn prefetch(mut self, depth: usize) -> Self {
        self.prefetch_depth = Some(depth);
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

        let streaming_depth = prefetch_depth.unwrap_or_else(|| {
            auto_prefetch_depth(per_sample_bytes, batch_size, device)
        });

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
                    build_streaming(dataset, batch_size, device, sampler, drop_last, streaming_depth, names)
                }
                Err(e) => Err(e),
            }
        } else {
            build_streaming(dataset, batch_size, device, sampler, drop_last, streaming_depth, names)
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

fn build_streaming(
    dataset: Arc<dyn BatchDataSet>,
    batch_size: usize,
    device: Device,
    sampler: Box<dyn Sampler>,
    drop_last: bool,
    prefetch_depth: usize,
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

        let per_sample_bytes: usize = {
            let sample = dataset.get_batch(&[0])?;
            sample.iter().map(|t| t.nbytes()).sum()
        };

        let prefetch_depth = auto_prefetch_depth(per_sample_bytes, batch_size, devices[0]);
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
}

impl StreamingLoader {
    fn epoch(&mut self, epoch: usize) -> EpochIterator<'_> {
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
    },
}

impl DeviceBackend {
    fn device(&self) -> Device {
        match self {
            DeviceBackend::Resident { device, .. } => *device,
            DeviceBackend::Streaming { device, .. } => *device,
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
        backends.push(DeviceBackend::Streaming { worker, device: dev });
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

/// Pick the gather device: resident backend with most free VRAM, or CPU.
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
        None => (Device::CPU, None),
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
                DeviceBackend::Streaming { worker, device: _ } => {
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
mod tests {
    use super::*;
    use crate::tensor::{DType, TensorOptions, test_device};

    struct SimpleData {
        x: Tensor,
        y: Tensor,
    }

    impl DataSet for SimpleData {
        fn len(&self) -> usize {
            self.x.shape()[0] as usize
        }
        fn get(&self, index: usize) -> Result<Vec<Tensor>> {
            Ok(vec![
                self.x.select(0, index as i64)?,
                self.y.select(0, index as i64)?,
            ])
        }
    }

    struct SequentialData {
        n: usize,
    }

    impl DataSet for SequentialData {
        fn len(&self) -> usize {
            self.n
        }
        fn get(&self, index: usize) -> Result<Vec<Tensor>> {
            Ok(vec![
                Tensor::from_f32(&[index as f32], &[1], Device::CPU)?,
            ])
        }
    }

    struct PairBatch {
        x: Tensor,
        y: Tensor,
    }

    impl BatchDataSet for PairBatch {
        fn len(&self) -> usize {
            self.x.shape()[0] as usize
        }
        fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
            let idx: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
            let idx_t = Tensor::from_i64(&idx, &[idx.len() as i64], Device::CPU)?;
            Ok(vec![
                self.x.index_select(0, &idx_t)?,
                self.y.index_select(0, &idx_t)?,
            ])
        }
    }

    fn make_data(n: usize) -> SimpleData {
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        SimpleData {
            x: Tensor::randn(&[n as i64, 4], opts).unwrap(),
            y: Tensor::randn(&[n as i64, 2], opts).unwrap(),
        }
    }

    fn make_cpu_data_for_device(n: usize) -> SimpleData {
        // DataSet contract: return CPU tensors. DataLoader handles device transfer.
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        SimpleData {
            x: Tensor::randn(&[n as i64, 4], opts).unwrap(),
            y: Tensor::randn(&[n as i64, 2], opts).unwrap(),
        }
    }

    #[test]
    fn test_basic_epoch_iteration() {
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 4); // 20 / 5 = 4
        for b in &batches {
            assert_eq!(b.len(), 2); // x and y
            assert_eq!(b[0].shape(), &[5, 4]);
            assert_eq!(b[1].shape(), &[5, 2]);
        }
    }

    #[test]
    fn test_drop_last_true() {
        let data = make_data(22);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .drop_last(true)
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 4); // 22 / 5 = 4, drop remainder of 2
    }

    #[test]
    fn test_drop_last_false() {
        let data = make_data(22);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .drop_last(false)
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 5); // 4 full + 1 partial
        assert_eq!(batches[4][0].shape(), &[2, 4]); // last batch has 2 samples
    }

    #[test]
    fn test_sequential_sampler() {
        let mut loader = DataLoader::from_dataset(SequentialData { n: 10 })
            .batch_size(3)
            .shuffle(false)
            .drop_last(false)
            .build()
            .unwrap();

        // Epoch 0 and epoch 1 should produce the same ordering
        let e0: Vec<f32> = loader
            .epoch(0)
            .flat_map(|b| {
                let b = b.unwrap();
                b[0].to_f32_vec().unwrap()
            })
            .collect();
        let e1: Vec<f32> = loader
            .epoch(1)
            .flat_map(|b| {
                let b = b.unwrap();
                b[0].to_f32_vec().unwrap()
            })
            .collect();
        assert_eq!(e0, e1);
        // And they should be in order
        assert_eq!(e0, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_shuffle_different_epochs() {
        let mut loader = DataLoader::from_dataset(SequentialData { n: 20 })
            .batch_size(20)
            .drop_last(false)
            .build()
            .unwrap();

        let e0: Vec<f32> = loader.epoch(0).next().unwrap().unwrap()[0]
            .to_f32_vec()
            .unwrap();
        let e1: Vec<f32> = loader.epoch(1).next().unwrap().unwrap()[0]
            .to_f32_vec()
            .unwrap();
        // Different epochs should yield different orderings (with overwhelming probability)
        assert_ne!(e0, e1);
    }

    #[test]
    fn test_shuffle_reproducible() {
        let data1 = SequentialData { n: 20 };
        let data2 = SequentialData { n: 20 };
        let mut l1 = DataLoader::from_dataset(data1)
            .batch_size(20)
            .seed(99)
            .drop_last(false)
            .build()
            .unwrap();
        let mut l2 = DataLoader::from_dataset(data2)
            .batch_size(20)
            .seed(99)
            .drop_last(false)
            .build()
            .unwrap();

        let e1: Vec<f32> = l1.epoch(3).next().unwrap().unwrap()[0]
            .to_f32_vec()
            .unwrap();
        let e2: Vec<f32> = l2.epoch(3).next().unwrap().unwrap()[0]
            .to_f32_vec()
            .unwrap();
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_all_samples_visited() {
        let mut loader = DataLoader::from_dataset(SequentialData { n: 10 })
            .batch_size(3)
            .drop_last(false)
            .build()
            .unwrap();

        let mut vals: Vec<f32> = loader
            .epoch(0)
            .flat_map(|b| {
                let b = b.unwrap();
                b[0].to_f32_vec().unwrap()
            })
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            vals,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_batch_dataset_path() {
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        let batch_ds = PairBatch {
            x: Tensor::randn(&[30, 8], opts).unwrap(),
            y: Tensor::randn(&[30, 3], opts).unwrap(),
        };
        let mut loader = DataLoader::from_batch_dataset(batch_ds)
            .batch_size(10)
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0][0].shape(), &[10, 8]);
        assert_eq!(batches[0][1].shape(), &[10, 3]);
    }

    #[test]
    fn test_exact_size_iterator() {
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .build()
            .unwrap();

        let iter = loader.epoch(0);
        assert_eq!(iter.len(), 4);
    }

    #[test]
    fn test_loader_metadata() {
        let data = make_data(50);
        let loader = DataLoader::from_dataset(data)
            .batch_size(8)
            .build()
            .unwrap();

        assert_eq!(loader.len(), 50);
        assert_eq!(loader.batch_size(), 8);
        assert_eq!(loader.num_batches(), 6); // 50/8 = 6 (drop_last=true)
        assert!(!loader.is_empty());
        assert!(loader.is_resident());
    }

    #[test]
    fn test_empty_dataset_errors() {
        struct Empty;
        impl DataSet for Empty {
            fn len(&self) -> usize { 0 }
            fn get(&self, _: usize) -> Result<Vec<Tensor>> { unreachable!() }
        }

        let result = DataLoader::from_dataset(Empty)
            .batch_size(10)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_batch_size_errors() {
        let data = make_data(10);
        let result = DataLoader::from_dataset(data)
            .batch_size(0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_size_larger_than_dataset() {
        let data = make_data(5);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(100)
            .drop_last(false)
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0][0].shape(), &[5, 4]);
    }

    #[test]
    fn test_batch_size_larger_than_dataset_drop_last() {
        let data = make_data(5);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(100)
            .drop_last(true)
            .build()
            .unwrap();

        // 5 < 100, so the only batch is incomplete -> dropped
        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 0);
    }

    #[test]
    fn test_device_aware_loading() {
        let data = make_cpu_data_for_device(20);
        let dev = test_device();
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .device(dev)
            .build()
            .unwrap();

        assert_eq!(loader.device(), dev);

        let b = loader.epoch(0).next().unwrap().unwrap();
        assert_eq!(b[0].device(), dev);
        assert_eq!(b[1].device(), dev);
    }

    #[test]
    fn test_multi_target_dataset() {
        struct FbrlLike {
            images: Tensor,
            letters: Tensor,
            cases: Tensor,
            origins: Tensor,
        }

        impl DataSet for FbrlLike {
            fn len(&self) -> usize { self.images.shape()[0] as usize }
            fn get(&self, i: usize) -> Result<Vec<Tensor>> {
                Ok(vec![
                    self.images.select(0, i as i64)?,
                    self.letters.select(0, i as i64)?,
                    self.cases.select(0, i as i64)?,
                    self.origins.select(0, i as i64)?,
                ])
            }
        }

        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        let data = FbrlLike {
            images: Tensor::randn(&[16, 3, 8, 8], opts).unwrap(),
            letters: Tensor::randn(&[16, 26], opts).unwrap(),
            cases: Tensor::randn(&[16, 2], opts).unwrap(),
            origins: Tensor::randn(&[16, 5], opts).unwrap(),
        };

        let mut loader = DataLoader::from_dataset(data)
            .batch_size(4)
            .build()
            .unwrap();

        let b = loader.epoch(0).next().unwrap().unwrap();
        assert_eq!(b.len(), 4);
        assert_eq!(b[0].shape(), &[4, 3, 8, 8]); // images
        assert_eq!(b[1].shape(), &[4, 26]);        // letters
        assert_eq!(b[2].shape(), &[4, 2]);          // cases
        assert_eq!(b[3].shape(), &[4, 5]);          // origins
    }

    // -- Streaming mode tests -------------------------------------------------

    #[test]
    fn test_streaming_basic_epoch() {
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .streaming()
            .build()
            .unwrap();

        assert!(!loader.is_resident());

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 4);
        for b in &batches {
            assert_eq!(b.len(), 2);
            assert_eq!(b[0].shape(), &[5, 4]);
            assert_eq!(b[1].shape(), &[5, 2]);
        }
    }

    #[test]
    fn test_streaming_drop_last() {
        let data = make_data(22);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .drop_last(true)
            .streaming()
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 4); // 22/5 = 4, drop 2
    }

    #[test]
    fn test_streaming_drop_last_false() {
        let data = make_data(22);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .drop_last(false)
            .streaming()
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 5); // 4 full + 1 partial
        assert_eq!(batches[4][0].shape(), &[2, 4]);
    }

    #[test]
    fn test_streaming_all_samples_visited() {
        let mut loader = DataLoader::from_dataset(SequentialData { n: 10 })
            .batch_size(3)
            .drop_last(false)
            .streaming()
            .build()
            .unwrap();

        let mut vals: Vec<f32> = loader
            .epoch(0)
            .flat_map(|b| {
                let b = b.unwrap();
                b[0].to_f32_vec().unwrap()
            })
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_streaming_multiple_epochs() {
        let mut loader = DataLoader::from_dataset(SequentialData { n: 20 })
            .batch_size(20)
            .drop_last(false)
            .streaming()
            .build()
            .unwrap();

        let e0: Vec<f32> = loader.epoch(0).next().unwrap().unwrap()[0]
            .to_f32_vec()
            .unwrap();
        let e1: Vec<f32> = loader.epoch(1).next().unwrap().unwrap()[0]
            .to_f32_vec()
            .unwrap();

        // Different epochs should produce different orderings
        assert_ne!(e0, e1);

        // But same number of samples
        assert_eq!(e0.len(), 20);
        assert_eq!(e1.len(), 20);
    }

    #[test]
    fn test_streaming_sequential() {
        let mut loader = DataLoader::from_dataset(SequentialData { n: 10 })
            .batch_size(3)
            .shuffle(false)
            .drop_last(false)
            .streaming()
            .build()
            .unwrap();

        let vals: Vec<f32> = loader
            .epoch(0)
            .flat_map(|b| b.unwrap()[0].to_f32_vec().unwrap())
            .collect();
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_streaming_multi_target() {
        struct Multi {
            a: Tensor,
            b: Tensor,
            c: Tensor,
        }
        impl DataSet for Multi {
            fn len(&self) -> usize { self.a.shape()[0] as usize }
            fn get(&self, i: usize) -> Result<Vec<Tensor>> {
                Ok(vec![
                    self.a.select(0, i as i64)?,
                    self.b.select(0, i as i64)?,
                    self.c.select(0, i as i64)?,
                ])
            }
        }

        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        let data = Multi {
            a: Tensor::randn(&[12, 4], opts).unwrap(),
            b: Tensor::randn(&[12, 8], opts).unwrap(),
            c: Tensor::randn(&[12, 2], opts).unwrap(),
        };

        let mut loader = DataLoader::from_dataset(data)
            .batch_size(4)
            .streaming()
            .build()
            .unwrap();

        let b = loader.epoch(0).next().unwrap().unwrap();
        assert_eq!(b.len(), 3);
        assert_eq!(b[0].shape(), &[4, 4]);
        assert_eq!(b[1].shape(), &[4, 8]);
        assert_eq!(b[2].shape(), &[4, 2]);
    }

    #[test]
    fn test_streaming_drop_mid_epoch() {
        let data = make_data(100);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(10)
            .streaming()
            .build()
            .unwrap();

        // Consume only 2 out of 10 batches, then drop the iterator
        {
            let mut iter = loader.epoch(0);
            let _ = iter.next().unwrap().unwrap();
            let _ = iter.next().unwrap().unwrap();
            // drop iter here
        }

        // Should be able to start a new epoch without issues
        let batches: Vec<Batch> = loader.epoch(1).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 10);
    }

    // -- Named Batch tests ---------------------------------------------------

    #[test]
    fn test_named_batch_via_loader() {
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .names(&["input", "target"])
            .build()
            .unwrap();

        let b = loader.epoch(0).next().unwrap().unwrap();
        assert_eq!(b.names(), &["input", "target"]);
        assert_eq!(b["input"].shape(), &[5, 4]);
        assert_eq!(b["target"].shape(), &[5, 2]);
        assert!(b.has("input"));
        assert!(b.has("target"));
        assert!(!b.has("missing"));
    }

    #[test]
    fn test_named_batch_streaming() {
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .names(&["x", "y"])
            .streaming()
            .build()
            .unwrap();

        let b = loader.epoch(0).next().unwrap().unwrap();
        assert_eq!(b.names(), &["x", "y"]);
        assert_eq!(b["x"].shape(), &[5, 4]);
        assert_eq!(b["y"].shape(), &[5, 2]);
    }

    #[test]
    fn test_names_count_mismatch_errors() {
        let data = make_data(10);
        let result = DataLoader::from_dataset(data)
            .batch_size(5)
            .names(&["only_one"])
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_names_when_unspecified() {
        let data = make_data(10);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .build()
            .unwrap();

        assert_eq!(loader.names(), &["0", "1"]);
        let b = loader.epoch(0).next().unwrap().unwrap();
        assert_eq!(b["0"].shape(), &[5, 4]);
        assert_eq!(b["1"].shape(), &[5, 2]);
    }

    // -- Graph + DataLoader integration tests --------------------------------

    #[test]
    fn test_graph_set_data_loader_single_gpu() {
        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear, Module, ReLU, mse_loss};

        let model = FlowBuilder::from(Linear::new(4, 8).unwrap())
            .through(ReLU::new())
            .through(Linear::new(8, 2).unwrap())
            .build()
            .unwrap();

        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        struct TrainData { x: Tensor, y: Tensor }
        impl super::DataSet for TrainData {
            fn len(&self) -> usize { self.x.shape()[0] as usize }
            fn get(&self, i: usize) -> Result<Vec<Tensor>> {
                Ok(vec![
                    self.x.select(0, i as i64)?,
                    self.y.select(0, i as i64)?,
                ])
            }
        }

        let data = TrainData {
            x: Tensor::randn(&[20, 4], opts).unwrap(),
            y: Tensor::randn(&[20, 2], opts).unwrap(),
        };

        let loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .names(&["input", "target"])
            .build()
            .unwrap();

        model.set_data_loader(loader, "input").unwrap();
        model.set_optimizer(|p| Adam::new(&p, 0.01));
        model.set_training(true);

        // Snapshot params before training
        let params_before: Vec<f32> = model
            .parameters()
            .iter()
            .flat_map(|p| p.variable.data().to_f32_vec().unwrap())
            .collect();

        // One epoch of training
        let iter = model.epoch(0);
        let mut active = iter.activate();
        let mut batch_count = 0;
        while let Some(batch_result) = active.next() {
            let b = batch_result.unwrap();
            assert!(b.has("input"));
            assert!(b.has("target"));
            let out = model.forward_batch(&b).unwrap();
            let target = crate::autograd::Variable::new(b["target"].clone(), false);
            let loss = mse_loss(&out, &target).unwrap();
            loss.backward().unwrap();
            model.step().unwrap();
            batch_count += 1;
        }

        assert_eq!(batch_count, 4); // 20 / 5 = 4

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
        assert!(changed, "parameters should change after training");
    }

    #[test]
    fn test_graph_data_num_batches() {
        use crate::graph::FlowBuilder;
        use crate::nn::Linear;

        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        let data = make_data(20);
        let loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .names(&["x", "y"])
            .build()
            .unwrap();

        model.set_data_loader(loader, "x").unwrap();
        assert_eq!(model.data_num_batches(), 4);
        assert_eq!(model.data_batch_size(), 5);
    }

    #[test]
    fn test_set_data_loader_invalid_input_name() {
        use crate::graph::FlowBuilder;
        use crate::nn::Linear;

        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        let data = make_data(10);
        let loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .names(&["x", "y"])
            .build()
            .unwrap();

        let result = model.set_data_loader(loader, "missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_scatter_fallback_without_data_loader() {
        // Module::forward(&Variable) still works without set_data_loader
        use crate::graph::FlowBuilder;
        use crate::nn::{Linear, Module};

        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        let x = crate::autograd::Variable::new(
            Tensor::randn(&[3, 4], Default::default()).unwrap(),
            false,
        );
        let out = model.forward(&x).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
    }
}
