//! Computation graph: fluent builder, parallel execution, observation, profiling,
//! visualization, and hierarchical composition.
//!
//! Build graphs with `FlowBuilder`, execute via the `Module` trait.
//! Label subgraphs for tree features: selective freeze/thaw, subgraph
//! checkpoints, cross-boundary observation, and per-subgraph optimizer groups.
//!
//! ```ignore
//! let encoder = FlowBuilder::from(Linear::new(4, 8)?)
//!     .through(GELU)
//!     .label("encoder")
//!     .build()?;
//!
//! let model = FlowBuilder::from(encoder)
//!     .through(Linear::new(8, 2)?)
//!     .build()?;
//!
//! let y = model.forward(&x)?;
//! model.freeze("encoder")?;  // freeze by label path
//! ```

pub mod node;
pub mod flow;
pub mod loop_node;
pub mod switch;
pub mod gate;
pub mod map;
pub mod observe;
pub mod trend;
pub mod profile;
pub mod dot;
pub mod plot;
pub mod router;
pub mod halt;
pub mod reshape;
pub mod state;
pub mod snapshot;
pub mod tree;
pub mod verbose;
#[allow(clippy::module_inception)]
mod graph;
mod distributed;

use std::collections::HashMap;

use crate::autograd::Variable;
use crate::data::Batch;

/// Context passed to the per-batch loss closure during El Che distributed
/// training. All fields carry live autograd graphs, so the returned loss
/// scalar can be backpropagated immediately.
///
/// ```ignore
/// model.set_loss_fn(|ctx: &LossContext| {
///     let cls  = cross_entropy_loss(&ctx.tags["head"], &ctx.batch["label"])?;
///     let rec  = mse_loss(&ctx.tags["recon"], &ctx.batch["image"])?;
///     Ok(cls + rec)
/// });
/// ```
pub struct LossContext<'a> {
    /// Forward output (live autograd).
    pub output: &'a Variable,
    /// The per-device batch with all named fields (inputs + targets).
    pub batch: &'a Batch,
    /// Tagged outputs from this forward pass (live autograd).
    pub tags: &'a HashMap<String, Variable>,
    /// Loop traces keyed by tag name (live autograd).
    pub traces: &'a HashMap<String, Vec<Variable>>,
}

pub use flow::FlowBuilder;
pub use loop_node::LoopBuilder;
pub use map::MapBuilder;
pub use trend::{Trend, TrendGroup};
pub use profile::{Profile, NodeTiming, LevelTiming};
pub use plot::format_duration;
pub use router::{SoftmaxRouter, SigmoidRouter, FixedSelector, ArgmaxSelector};
pub use halt::{ThresholdHalt, LearnedHalt};
pub use reshape::Reshape;
pub use state::StateAdd;
pub use observe::Reduce;
pub use tree::PathKind;
pub use snapshot::ModelSnapshot;
pub use graph::*;

/// Merge operation for combining split branches.
pub enum MergeOp {
    /// Element-wise sum of all branches.
    Add,
    /// Element-wise mean of all branches.
    Mean,
}
