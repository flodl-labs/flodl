//! Prefetch pipeline internals for streaming mode.
//!
//! Not part of the public API. Used by [`DataLoader`](super::DataLoader)
//! when the dataset does not fit in VRAM.

use std::sync::mpsc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crate::tensor::{Device, Result, Tensor};
use super::BatchDataSet;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single prefetched batch, ready on the target device.
pub(crate) struct PrefetchedBatch {
    pub tensors: Vec<Tensor>,
    /// Event recorded after async H2D copy. Consumer waits on this.
    #[cfg(feature = "cuda")]
    pub ready_event: Option<crate::distributed::cuda_event::CudaEvent>,
}

/// Commands sent to the persistent worker thread.
pub(crate) enum WorkerCmd {
    /// Start a new epoch. Includes a fresh batch channel for this epoch.
    StartEpoch {
        indices: Vec<usize>,
        batch_size: usize,
        drop_last: bool,
        /// Per-epoch batch sender. Dropped when the epoch is done or cancelled.
        batch_tx: mpsc::SyncSender<Result<PrefetchedBatch>>,
    },
    /// Open a distributed epoch: install the batch sender, then wait for
    /// `LoadBatch` commands. The channel stays open until the next
    /// `StartEpoch`/`StartDistributedEpoch`/`Stop`.
    StartDistributedEpoch {
        batch_tx: mpsc::SyncSender<Result<PrefetchedBatch>>,
    },
    /// Load a single batch (distributed mode). Worker sends the result on
    /// the channel from the preceding `StartDistributedEpoch`.
    LoadBatch {
        indices: Vec<usize>,
    },
    /// Shut down the worker.
    Stop,
}

// ---------------------------------------------------------------------------
// PrefetchWorker (persistent, lives for DataLoader lifetime)
// ---------------------------------------------------------------------------

/// Persistent background worker for streaming prefetch.
///
/// Created once at `DataLoader::build()`, lives until the DataLoader is
/// dropped. Keeps its dedicated CUDA stream alive across epochs.
///
/// Each epoch gets a fresh batch channel (created in `start_epoch()`),
/// so dropping an epoch iterator mid-epoch naturally cancels outstanding
/// work: the worker detects the closed channel and moves on.
pub(crate) struct PrefetchWorker {
    cmd_tx: mpsc::Sender<WorkerCmd>,
    handle: Option<JoinHandle<()>>,
    prefetch_depth: usize,
}

impl PrefetchWorker {
    /// Spawn the persistent worker thread.
    pub fn new(
        dataset: Arc<dyn BatchDataSet>,
        device: Device,
        prefetch_depth: usize,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCmd>();

        let handle = thread::spawn(move || {
            worker_loop(dataset, device, cmd_rx);
        });

        PrefetchWorker {
            cmd_tx,
            handle: Some(handle),
            prefetch_depth,
        }
    }

    /// Start a new epoch and return a receiver for the batches.
    pub fn start_epoch(
        &self,
        indices: Vec<usize>,
        batch_size: usize,
        drop_last: bool,
    ) -> mpsc::Receiver<Result<PrefetchedBatch>> {
        let (batch_tx, batch_rx) =
            mpsc::sync_channel::<Result<PrefetchedBatch>>(self.prefetch_depth);

        let _ = self.cmd_tx.send(WorkerCmd::StartEpoch {
            indices,
            batch_size,
            drop_last,
            batch_tx,
        });

        batch_rx
    }

    /// Open a distributed epoch: create one channel that persists across
    /// all batches. Follow with [`Self::load_batch()`] calls per batch.
    pub fn start_distributed_epoch(&self) -> mpsc::Receiver<Result<PrefetchedBatch>> {
        let (batch_tx, batch_rx) =
            mpsc::sync_channel::<Result<PrefetchedBatch>>(self.prefetch_depth);

        let _ = self.cmd_tx.send(WorkerCmd::StartDistributedEpoch { batch_tx });

        batch_rx
    }

    /// Send a single batch of indices for loading (distributed mode).
    /// The result arrives on the receiver from [`Self::start_distributed_epoch()`].
    pub fn load_batch(&self, indices: Vec<usize>) {
        let _ = self.cmd_tx.send(WorkerCmd::LoadBatch { indices });
    }

    /// Current prefetch depth (channel capacity for next epoch).
    pub fn prefetch_depth(&self) -> usize {
        self.prefetch_depth
    }

    /// Update prefetch depth. Takes effect on the next epoch (the channel
    /// is recreated with the new capacity in `start_epoch()`).
    pub fn set_prefetch_depth(&mut self, depth: usize) {
        self.prefetch_depth = depth;
    }
}

impl Drop for PrefetchWorker {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(WorkerCmd::Stop);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Worker loop
// ---------------------------------------------------------------------------

fn worker_loop(
    dataset: Arc<dyn BatchDataSet>,
    device: Device,
    cmd_rx: mpsc::Receiver<WorkerCmd>,
) {
    // Create a dedicated CUDA stream for H2D transfers (lives across epochs).
    #[cfg(feature = "cuda")]
    let copy_stream = if device.is_cuda() {
        crate::distributed::cuda_stream::CudaStream::new(device, false).ok()
    } else {
        None
    };

    // Distributed epoch channel, kept alive across LoadBatch commands.
    let mut dist_tx: Option<mpsc::SyncSender<Result<PrefetchedBatch>>> = None;

    for cmd in &cmd_rx {
        match cmd {
            WorkerCmd::StartEpoch {
                indices,
                batch_size,
                drop_last,
                batch_tx,
            } => {
                dist_tx = None; // close any distributed channel

                let n = indices.len();
                let mut start = 0;

                while start < n {
                    let end = (start + batch_size).min(n);
                    if drop_last && (end - start) < batch_size {
                        break;
                    }

                    let batch_indices = &indices[start..end];
                    start = end;

                    let result = fetch_and_transfer(
                        &*dataset,
                        batch_indices,
                        device,
                        #[cfg(feature = "cuda")]
                        copy_stream.as_ref(),
                    );

                    // If the consumer dropped (epoch iterator dropped mid-epoch),
                    // the send fails. We stop this epoch and wait for the next command.
                    if batch_tx.send(result).is_err() {
                        break;
                    }
                }
                // batch_tx is dropped here, closing the epoch's channel.
            }
            WorkerCmd::StartDistributedEpoch { batch_tx } => {
                dist_tx = Some(batch_tx);
            }
            WorkerCmd::LoadBatch { indices } => {
                if let Some(ref tx) = dist_tx {
                    let result = fetch_and_transfer(
                        &*dataset,
                        &indices,
                        device,
                        #[cfg(feature = "cuda")]
                        copy_stream.as_ref(),
                    );
                    if tx.send(result).is_err() {
                        dist_tx = None; // consumer dropped
                    }
                }
            }
            WorkerCmd::Stop => break,
        }
    }
}

/// Fetch a batch from the dataset and transfer to the target device.
fn fetch_and_transfer(
    dataset: &dyn BatchDataSet,
    indices: &[usize],
    device: Device,
    #[cfg(feature = "cuda")] copy_stream: Option<&crate::distributed::cuda_stream::CudaStream>,
) -> Result<PrefetchedBatch> {
    let tensors = dataset.get_batch(indices)?;

    if !device.is_cuda() {
        return Ok(PrefetchedBatch {
            tensors,
            #[cfg(feature = "cuda")]
            ready_event: None,
        });
    }

    // Pin memory and async-copy to GPU on dedicated stream
    #[cfg(feature = "cuda")]
    {
        use crate::distributed::cuda_event::{CudaEvent, CudaEventFlags};
        use crate::distributed::cuda_stream::StreamGuard;

        let mut on_device = Vec::with_capacity(tensors.len());

        if let Some(stream) = copy_stream {
            let _guard = StreamGuard::new(stream);
            for t in &tensors {
                let pinned = t.pin_memory()?;
                on_device.push(pinned.to_device_async(device)?);
            }

            // Record completion event on the copy stream
            let event = CudaEvent::new(CudaEventFlags::DisableTiming)?;
            event.record_on(stream)?;

            return Ok(PrefetchedBatch {
                tensors: on_device,
                ready_event: Some(event),
            });
        }

        // Fallback: synchronous transfer (no stream available)
        for t in &tensors {
            let pinned = t.pin_memory()?;
            on_device.push(pinned.to_device(device)?);
        }

        Ok(PrefetchedBatch {
            tensors: on_device,
            ready_event: None,
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Without CUDA feature, just return CPU tensors
        Ok(PrefetchedBatch { tensors })
    }
}
