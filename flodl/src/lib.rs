//! flodl — a deep learning framework built on libtorch, from Rust.
//!
//! Stack: `flodl-sys` (C++ shim FFI) → `tensor` → `autograd` → `nn` → `graph`.
//!
//! ```ignore
//! use flodl::*;
//!
//! // Build a model as a computation graph
//! let g = FlowBuilder::from(Linear::new(4, 8)?)
//!     .through(GELU::new())
//!     .through(Linear::new(8, 2)?)
//!     .build()?;
//!
//! // Forward pass
//! let x = Variable::new(Tensor::randn(&[1, 4], Default::default())?, false);
//! let y = g.forward(&x)?;
//!
//! // Backward + optimize
//! let loss = mse_loss(&y, &target)?;
//! loss.backward()?;
//! optimizer.step()?;
//! ```

pub mod tensor;
pub mod autograd;
pub mod nn;
pub mod graph;
pub mod monitor;

/// Shorthand for building `Vec<Box<dyn Module>>` from a list of modules.
/// Use with `split`, `gate`, and `switch` to avoid manual `Box::new()` wrapping.
///
/// ```ignore
/// .split(modules![read_head(H), read_head(H)])
/// .gate(router, modules![Linear::new(H, H)?, Linear::new(H, H)?])
/// ```
#[macro_export]
macro_rules! modules {
    ($($module:expr),* $(,)?) => {
        vec![$(Box::new($module) as Box<dyn $crate::Module>),*]
    };
}

pub use tensor::{cuda_available, cuda_device_count, cuda_memory_info, cuda_memory_info_idx, cuda_utilization, cuda_utilization_idx, cuda_device_name, cuda_device_name_idx, cuda_devices, DeviceInfo, set_current_cuda_device, current_cuda_device, cuda_synchronize, hardware_summary, set_cudnn_benchmark, malloc_trim, live_tensor_count, rss_kb, Device, DType, Result, Tensor, TensorOptions};
pub use autograd::{Variable, no_grad, is_grad_enabled, NoGradGuard, adaptive_avg_pool2d, grid_sample};
pub use nn::{
    Module, NamedInputModule,
    Parameter, Buffer, Linear, Optimizer, Stateful, SGD, SGDBuilder, Adam, AdamBuilder, AdamW, AdamWBuilder,
    save_checkpoint, load_checkpoint, save_checkpoint_file, load_checkpoint_file,
    LoadReport,
    GradScaler, cast_parameters,
    Identity, ReLU, Sigmoid, Tanh, GELU, SiLU,
    Dropout, Dropout2d, LayerNorm, Embedding, GRUCell, LSTMCell,
    Conv2d, ConvTranspose2d, BatchNorm, BatchNorm2d,
    mse_loss, cross_entropy_loss, bce_with_logits_loss, l1_loss, smooth_l1_loss, kl_div_loss,
    clip_grad_norm, clip_grad_value,
    Scheduler, StepDecay, CosineScheduler, WarmupScheduler, PlateauScheduler,
    xavier_uniform, xavier_normal,
    walk_modules, walk_modules_visited,
};
pub use graph::{
    FlowBuilder, MergeOp, Graph, MapBuilder, Trend, TrendGroup,
    Profile, NodeTiming, LevelTiming, format_duration,
    SoftmaxRouter, SigmoidRouter, FixedSelector, ArgmaxSelector,
    ThresholdHalt, LearnedHalt,
    Reshape, StateAdd, Reduce,
};
