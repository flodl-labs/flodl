//! Distributed Data Parallel (DDP) infrastructure for multi-GPU training.
//!
//! Three entry points, one type:
//!
//! - [`Ddp::setup()`] -- user owns the training loop (Graph-based, transparent)
//! - [`Ddp::builder()`] -- framework manages threads, data, epochs, averaging
//! - [`Ddp::wrap()`] -- manual gradient sync for advanced patterns (GAN, RL)
//!
//! Supporting infrastructure: NCCL bindings, CUDA events/streams, El Che
//! heterogeneous cadence strategy, and the async DDP runtime.

pub mod cuda_event;
pub mod cuda_stream;
pub mod nccl;
pub mod ddp;
pub mod ddp_run;
pub mod el_che;

pub use cuda_event::{CudaEvent, CudaEventFlags};
pub use cuda_stream::{CudaStream, StreamGuard};
pub use nccl::{NcclAbortHandle, NcclComms, NcclRankComm, NcclUniqueId, ReduceOp};
pub use ddp::{Ddp, DdpConfig};
pub use el_che::ElChe;
pub use ddp_run::{ApplyPolicy, DdpHandle, DdpBuilder, DdpRunConfig, AverageBackend, TrainedState, EpochMetrics, record_scalar, GpuWorker};
// Deprecated aliases
#[allow(deprecated)]
pub use ddp_run::{AsyncDdp, AsyncDdpBuilder, AsyncDdpConfig};
