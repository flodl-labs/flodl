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
        } = self;

        let n = dataset.len();

        // Probe dataset size for mode decision
        let sample = dataset.get_batch(&[0])?;
        if sample.is_empty() {
            return Err(TensorError::new(
                "DataLoader: dataset returned empty tensor list",
            ));
        }
        let per_sample_bytes: usize = sample.iter().map(|t| t.nbytes()).sum();
        drop(sample);

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
            match build_resident(Arc::clone(&dataset), batch_size, device, sampler, drop_last) {
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
                    build_streaming(dataset, batch_size, device, sampler, drop_last, streaming_depth)
                }
                Err(e) => Err(e),
            }
        } else {
            build_streaming(dataset, batch_size, device, sampler, drop_last, streaming_depth)
        }
    }
}

fn build_resident(
    dataset: Arc<dyn BatchDataSet>,
    batch_size: usize,
    device: Device,
    sampler: Box<dyn Sampler>,
    drop_last: bool,
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
            device,
            batch_size,
            sampler,
            drop_last,
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
    inner: LoaderInner,
}

enum LoaderInner {
    Resident(ResidentLoader),
    Streaming(StreamingLoader),
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
    pub fn epoch(&mut self, epoch: usize) -> EpochIterator<'_> {
        match &mut self.inner {
            LoaderInner::Resident(loader) => loader.epoch(epoch),
            LoaderInner::Streaming(loader) => loader.epoch(epoch),
        }
    }

    /// Number of samples in the dataset.
    pub fn len(&self) -> usize {
        match &self.inner {
            LoaderInner::Resident(l) => l.sampler.len(),
            LoaderInner::Streaming(l) => l.sampler.len(),
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
        };
        if dl { n / bs } else { n.div_ceil(bs) }
    }

    /// Batch size.
    pub fn batch_size(&self) -> usize {
        match &self.inner {
            LoaderInner::Resident(l) => l.batch_size,
            LoaderInner::Streaming(l) => l.batch_size,
        }
    }

    /// Target device.
    pub fn device(&self) -> Device {
        match &self.inner {
            LoaderInner::Resident(l) => l.device,
            LoaderInner::Streaming(l) => l.device,
        }
    }

    /// Whether the loader is in resident mode (full dataset in memory).
    pub fn is_resident(&self) -> bool {
        matches!(&self.inner, LoaderInner::Resident(_))
    }
}

// ---------------------------------------------------------------------------
// ResidentLoader
// ---------------------------------------------------------------------------

struct ResidentLoader {
    /// Full dataset tensors on target device, one per position.
    gpu_data: Vec<Tensor>,
    device: Device,
    batch_size: usize,
    sampler: Box<dyn Sampler>,
    drop_last: bool,
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
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingLoader
// ---------------------------------------------------------------------------

struct StreamingLoader {
    /// Dataset shared with the worker thread.
    _dataset: Arc<dyn BatchDataSet>,
    batch_size: usize,
    device: Device,
    sampler: Box<dyn Sampler>,
    drop_last: bool,
    worker: PrefetchWorker,
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
    Streaming(StreamingEpochIter),
}

struct ResidentEpochIter<'a> {
    data: &'a [Tensor],
    perm: Tensor,
    /// (start_in_perm, batch_len)
    batch_ranges: Vec<(usize, usize)>,
    pos: usize,
}

struct StreamingEpochIter {
    batch_rx: std::sync::mpsc::Receiver<Result<super::prefetch::PrefetchedBatch>>,
    remaining: usize,
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

        Some(Ok(Batch::new(tensors)))
    }
}

impl StreamingEpochIter {
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
                Some(Ok(Batch::new(batch.tensors)))
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, TensorOptions, test_device, test_opts};

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

    fn make_device_data(n: usize) -> SimpleData {
        let opts = test_opts();
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
        let data = make_device_data(20);
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
}
