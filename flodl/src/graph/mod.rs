//! Computation graph: fluent builder, parallel execution, observation, profiling, and visualization.
//!
//! Build graphs with [`FlowBuilder`], execute via the [`Module`] trait.
//!
//! ```ignore
//! let g = FlowBuilder::from(Linear::new(4, 8)?)
//!     .through(GELU::new())
//!     .through(Linear::new(8, 2)?)
//!     .build()?;
//!
//! let y = g.forward(&x)?;
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

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;
use std::time::Instant;

use indexmap::IndexMap;
use sha2::{Sha256, Digest};

use node::*;
use crate::autograd::Variable;
use crate::nn::{Buffer, Module, Parameter};
use crate::tensor::{Result, Tensor, TensorError};

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

/// Merge operation for combining split branches.
pub enum MergeOp {
    /// Element-wise sum of all branches.
    Add,
    /// Element-wise mean of all branches.
    Mean,
}

/// Forward-reference state buffer. Persists across `forward()` calls.
struct StateEntry {
    writer_ni: usize,
    value: Rc<RefCell<Option<Variable>>>,
}

/// An executable computation graph. Implements `Module` for composability.
///
/// Built via [`FlowBuilder`]. Supports parallel execution of independent nodes,
/// observation of tagged outputs, profiling, and DOT/SVG visualization.
///
/// ```ignore
/// let g = FlowBuilder::from(Linear::new(4, 8)?)
///     .through(GELU::new())
///     .through(Linear::new(8, 2)?)
///     .build()?;
///
/// // Forward pass (graph implements Module)
/// let y = g.forward(&x)?;
///
/// // Observation
/// g.end_step();
/// g.end_epoch();
/// let loss_trend = g.trend("loss");
///
/// // Visualization
/// let dot = g.dot();
/// g.svg(Some("graph.svg"))?;
/// ```
pub struct Graph {
    nodes: Vec<Node>,
    node_index: HashMap<String, usize>,
    levels: Vec<Vec<usize>>,
    edges: Vec<Edge>,
    edges_from: HashMap<usize, Vec<usize>>,
    inputs: Vec<ExposedPort>,
    outputs: Vec<ExposedPort>,
    order: Vec<usize>,
    state: Vec<StateEntry>,
    // State writer lookup: node_idx → [(state_entry_idx, output_port_idx)]
    state_writers: HashMap<usize, Vec<(usize, usize)>>,
    // Tag groups: group name → suffixed tag names
    tag_groups: HashMap<String, Vec<String>>,
    // Observation: tag mapping (immutable after build)
    tag_names: HashMap<String, (usize, usize)>,           // tag name → (node_idx, port_idx)
    tag_capture: HashMap<usize, Vec<(String, usize)>>,     // node_idx → [(tag_name, port_idx)]
    // Observation: mutable state (RefCell/Cell for &self methods)
    tagged_outputs: RefCell<HashMap<String, Variable>>,
    batch_buffer: RefCell<HashMap<String, Vec<f64>>>,
    epoch_history: RefCell<HashMap<String, Vec<f64>>>,
    flush_count: Cell<usize>,
    // Profiling
    profiling: Cell<bool>,
    last_profile: RefCell<Option<profile::Profile>>,
    timing_buffer: RefCell<HashMap<String, Vec<f64>>>,
    timing_history: RefCell<HashMap<String, Vec<f64>>>,
    // Flush timestamps (seconds since first forward — for ETA in write_log)
    flush_times: RefCell<Vec<f64>>,
    training_start: Cell<f64>,
    // Step/epoch counters
    step_count: Cell<usize>,
    epoch_count: Cell<usize>,
    // Identity: label + structural hash
    label: Option<String>,
    structural_hash_cache: OnceCell<String>,
}

impl Graph {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        mut node_map: IndexMap<String, Node>,
        edges: Vec<Edge>,
        inputs: Vec<ExposedPort>,
        outputs: Vec<ExposedPort>,
        tags: HashMap<String, NodeRef>,
        forward_refs: Vec<ForwardRefSpec>,
        tag_groups: HashMap<String, Vec<String>>,
        label: Option<String>,
    ) -> Result<Self> {
        // Set up forward-reference state buffers and wire state read nodes
        let mut state = Vec::with_capacity(forward_refs.len());
        for fr in &forward_refs {
            let value: Rc<RefCell<Option<Variable>>> = Rc::new(RefCell::new(None));
            let reader_value = value.clone();

            // Wire the state read node to return the buffer value
            if let Some(node) = node_map.get_mut(&fr.reader_id) {
                node.run = Box::new(move |_: &[Variable]| {
                    match reader_value.borrow().as_ref() {
                        Some(v) => Ok(vec![v.clone()]),
                        None => Ok(vec![]), // empty = no state yet
                    }
                });
            }

            state.push(StateEntry {
                writer_ni: 0, // resolved after node indexing
                value,
            });
        }

        // Convert to indexed storage
        let mut nodes = Vec::with_capacity(node_map.len());
        let mut node_index = HashMap::with_capacity(node_map.len());

        for (_key, node) in node_map {
            let idx = nodes.len();
            node_index.insert(node.id.clone(), idx);
            nodes.push(node);
        }

        // Validate edges
        for edge in &edges {
            if !node_index.contains_key(&edge.from_node) {
                return Err(TensorError::new(&format!(
                    "unknown source node: {}",
                    edge.from_node
                )));
            }
            if !node_index.contains_key(&edge.to_node) {
                return Err(TensorError::new(&format!(
                    "unknown target node: {}",
                    edge.to_node
                )));
            }
        }

        // Build edges_from lookup
        let mut edges_from: HashMap<usize, Vec<usize>> = HashMap::new();
        for (ei, edge) in edges.iter().enumerate() {
            let from_idx = node_index[&edge.from_node];
            edges_from.entry(from_idx).or_default().push(ei);
        }

        // Topological levels (Kahn's algorithm)
        let levels = topological_levels(&nodes, &node_index, &edges)?;
        let order: Vec<usize> = levels.iter().flat_map(|l| l.iter().copied()).collect();

        // Build tag capture indices for observation
        let mut tag_names_map: HashMap<String, (usize, usize)> = HashMap::new();
        let mut tag_capture: HashMap<usize, Vec<(String, usize)>> = HashMap::new();
        for (name, node_ref) in &tags {
            if let Some(&ni) = node_index.get(&node_ref.node_id) {
                let port_idx = nodes[ni]
                    .output_ports
                    .iter()
                    .position(|p| p == &node_ref.port)
                    .unwrap_or(0);
                tag_names_map.insert(name.clone(), (ni, port_idx));
                tag_capture
                    .entry(ni)
                    .or_default()
                    .push((name.clone(), port_idx));
            }
        }

        // Build state writer lookup: node_idx → [(state_entry_idx, port_idx)]
        // Also resolve writer_ni on each state entry for DOT rendering.
        let mut state_writers: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (si, fr) in forward_refs.iter().enumerate() {
            if let Some(&ni) = node_index.get(&fr.writer_id) {
                state[si].writer_ni = ni;
                let port_idx = nodes[ni]
                    .output_ports
                    .iter()
                    .position(|p| p == &fr.writer_port)
                    .unwrap_or(0);
                state_writers.entry(ni).or_default().push((si, port_idx));
            }
        }

        Ok(Graph {
            nodes,
            node_index,
            levels,
            edges,
            edges_from,
            inputs,
            outputs,
            order,
            state,
            state_writers,
            tag_groups,
            tag_names: tag_names_map,
            tag_capture,
            tagged_outputs: RefCell::new(HashMap::new()),
            batch_buffer: RefCell::new(HashMap::new()),
            epoch_history: RefCell::new(HashMap::new()),
            flush_count: Cell::new(0),
            profiling: Cell::new(false),
            last_profile: RefCell::new(None),
            timing_buffer: RefCell::new(HashMap::new()),
            timing_history: RefCell::new(HashMap::new()),
            flush_times: RefCell::new(Vec::new()),
            training_start: Cell::new(0.0),
            step_count: Cell::new(0),
            epoch_count: Cell::new(0),
            label,
            structural_hash_cache: OnceCell::new(),
        })
    }

    fn forward_impl(&self, graph_inputs: &[Variable]) -> Result<Variable> {
        if graph_inputs.len() != self.inputs.len() {
            return Err(TensorError::new(&format!(
                "expected {} inputs, got {}",
                self.inputs.len(),
                graph_inputs.len()
            )));
        }

        // Record training start on first forward (for ETA).
        if self.training_start.get() == 0.0 {
            self.training_start.set(instant_secs());
        }

        let is_profiling = self.profiling.get();
        let forward_start = if is_profiling { Some(Instant::now()) } else { None };
        let mut prof_nodes: Vec<profile::NodeTiming> = Vec::new();
        let mut prof_levels: Vec<profile::LevelTiming> = Vec::new();

        // Build reverse tag lookup for profiling: node_idx → first tag name
        let tags_by_node: HashMap<usize, String> = if is_profiling {
            let mut m = HashMap::new();
            for (name, &(ni, _)) in &self.tag_names {
                m.entry(ni).or_insert_with(|| name.clone());
            }
            m
        } else {
            HashMap::new()
        };

        let n = self.nodes.len();
        let mut input_slots: Vec<HashMap<String, Option<Variable>>> =
            (0..n).map(|_| HashMap::new()).collect();
        let mut output_values: Vec<Option<Vec<Variable>>> = vec![None; n];
        let has_tags = !self.tag_capture.is_empty();
        let mut tagged_outputs = if has_tags {
            Some(HashMap::new())
        } else {
            None
        };

        // Route graph inputs to node input ports
        for (i, ep) in self.inputs.iter().enumerate() {
            let ni = self.node_index[&ep.node_id];
            input_slots[ni].insert(ep.port.clone(), Some(graph_inputs[i].clone()));
        }

        // Execute levels sequentially
        for (level_idx, level) in self.levels.iter().enumerate() {
            let level_start = if is_profiling { Some(Instant::now()) } else { None };
            let mut level_sum_ns: u64 = 0;

            for &ni in level {
                let node = &self.nodes[ni];

                // Collect inputs in port order, zero-filling nil state refs
                let inputs: Vec<Variable> = node
                    .input_ports
                    .iter()
                    .enumerate()
                    .map(|(i, port)| {
                        match input_slots[ni].get(port).and_then(|v| v.as_ref()) {
                            Some(v) => v.clone(),
                            None if i > 0 => {
                                // Zero fill: create zeros matching first input shape
                                let first = input_slots[ni]
                                    .get(&node.input_ports[0])
                                    .and_then(|v| v.as_ref())
                                    .expect("missing primary input");
                                Variable::new(
                                    Tensor::zeros_like(&first.data()).unwrap(),
                                    false,
                                )
                            }
                            _ => panic!("missing input '{}' for node '{}'", port, node.id),
                        }
                    })
                    .collect();

                // Execute (with optional per-node timing)
                let node_start = if is_profiling { Some(Instant::now()) } else { None };
                let outputs = (node.run)(&inputs)?;
                if is_profiling {
                    let elapsed = node_start.unwrap().elapsed();
                    level_sum_ns += elapsed.as_nanos() as u64;
                    prof_nodes.push(profile::NodeTiming {
                        id: node.id.clone(),
                        tag: tags_by_node.get(&ni).cloned().unwrap_or_default(),
                        duration: elapsed,
                        level: level_idx,
                    });
                }
                output_values[ni] = Some(outputs);

                // Route outputs to downstream nodes
                if let Some(edge_indices) = self.edges_from.get(&ni) {
                    for &ei in edge_indices {
                        let edge = &self.edges[ei];
                        let from_port_idx = node
                            .output_ports
                            .iter()
                            .position(|p| p == &edge.from_port)
                            .expect("bad output port");
                        let to_ni = self.node_index[&edge.to_node];
                        let outs = output_values[ni].as_ref().unwrap();
                        // State read with no value returns empty vec → insert None
                        let value = if from_port_idx < outs.len() {
                            Some(outs[from_port_idx].clone())
                        } else {
                            None
                        };
                        input_slots[to_ni].insert(edge.to_port.clone(), value);
                    }
                }

                // Capture state: if this node is a state writer, store its output
                if let Some(writers) = self.state_writers.get(&ni)
                    && let Some(ref outs) = output_values[ni]
                {
                    for &(si, port_idx) in writers {
                        if port_idx < outs.len() {
                            *self.state[si].value.borrow_mut() = Some(outs[port_idx].clone());
                        }
                    }
                }

                // Capture tagged outputs for observation
                if let Some(ref mut tagged) = tagged_outputs
                    && let Some(captures) = self.tag_capture.get(&ni)
                    && let Some(ref outs) = output_values[ni]
                {
                    for (tag_name, port_idx) in captures {
                        if *port_idx < outs.len() {
                            tagged.insert(tag_name.clone(), outs[*port_idx].clone());
                        }
                    }
                }
            }

            // Record level timing
            if is_profiling {
                prof_levels.push(profile::LevelTiming {
                    index: level_idx,
                    wall_clock: level_start.unwrap().elapsed(),
                    sum_nodes: std::time::Duration::from_nanos(level_sum_ns),
                    num_nodes: level.len(),
                });
            }
        }

        // Store tagged outputs
        if let Some(tagged) = tagged_outputs {
            *self.tagged_outputs.borrow_mut() = tagged;
        }

        // Store profile
        if is_profiling {
            *self.last_profile.borrow_mut() = Some(profile::Profile {
                total: forward_start.unwrap().elapsed(),
                levels: prof_levels,
                nodes: prof_nodes,
            });
        }

        // Collect graph output
        let out = &self.outputs[0];
        let out_ni = self.node_index[&out.node_id];
        let out_port_idx = self.nodes[out_ni]
            .output_ports
            .iter()
            .position(|p| p == &out.port)
            .expect("bad output port");

        output_values[out_ni]
            .as_ref()
            .and_then(|o| o.get(out_port_idx).cloned())
            .ok_or_else(|| TensorError::new("graph produced no output"))
    }
}

impl Graph {
    /// Clear all forward-reference state buffers to None.
    /// Call when starting inference on a new sequence.
    pub fn reset_state(&self) {
        for entry in &self.state {
            *entry.value.borrow_mut() = None;
        }
    }

    /// Break gradient chain on forward-reference state buffers and module state.
    /// Call between training steps to prevent unbounded graph growth.
    pub fn detach_state(&self) {
        // Detach graph-level state buffers (forward references).
        for entry in &self.state {
            let mut val = entry.value.borrow_mut();
            if let Some(ref v) = *val {
                *val = Some(v.detach());
            }
        }
        // Detach tagged outputs — these hold Variables from the forward
        // pass whose grad_fn chains reference the C++ autograd graph.
        // Without this, the Node objects persist until the next forward
        // pass replaces tagged_outputs.
        {
            let mut tagged = self.tagged_outputs.borrow_mut();
            for var in tagged.values_mut() {
                *var = var.detach();
            }
        }
        // Propagate detach to modules that hold internal state.
        for node in &self.nodes {
            if let Some(ref module) = node.module {
                module.detach_state();
            }
        }
    }

    /// Returns true if this graph has forward-reference state.
    pub fn has_state(&self) -> bool {
        !self.state.is_empty()
    }

    /// End-of-step housekeeping: detach state (cut gradient chain but
    /// preserve values for the next forward), collect timings,
    /// increment step counter.
    ///
    /// For recurrent models this implements truncated BPTT — state carries
    /// over between steps but gradients don't flow across step boundaries.
    /// Call [`end_sequence`](Self::end_sequence) to fully wipe state
    /// when starting a new independent sequence.
    ///
    /// ```ignore
    /// for token in sequence {
    ///     let y = graph.forward(&token)?;
    ///     // ... backward, optimize ...
    ///     graph.end_step();       // keep state, cut gradients
    /// }
    /// graph.end_sequence();       // wipe state for next sequence
    /// ```
    pub fn end_step(&self) {
        self.detach_state();
        if self.profiling.get() {
            self.collect_timings(&[]);
        }
        self.step_count.set(self.step_count.get() + 1);
    }

    /// End-of-sequence housekeeping: fully reset state buffers to None.
    /// Call between independent sequences so the model starts fresh.
    ///
    /// For non-recurrent graphs (no forward refs) this is a no-op.
    pub fn end_sequence(&self) {
        self.reset_state();
    }

    /// End-of-epoch housekeeping: flush all observation and timing buffers,
    /// increment epoch counter.
    pub fn end_epoch(&self) {
        self.flush(&[]);
        if self.profiling.get() {
            self.flush_timings(&[]);
        }
        self.epoch_count.set(self.epoch_count.get() + 1);
    }

    /// Number of completed training steps.
    pub fn step_count(&self) -> usize {
        self.step_count.get()
    }

    /// Number of completed training epochs.
    pub fn epoch_count(&self) -> usize {
        self.epoch_count.get()
    }

    /// Get member tags of a tag group, or None if not registered.
    pub fn tag_group(&self, name: &str) -> Option<&[String]> {
        self.tag_groups.get(name).map(|v| v.as_slice())
    }

    /// Forward with multiple inputs (for graphs with Input ports).
    /// Inputs are in declaration order: From entry first, then each Input.
    pub fn forward_multi(&self, inputs: &[Variable]) -> Result<Variable> {
        self.forward_impl(inputs)
    }

    /// Move all parameters, state buffers, and module buffers to a device.
    pub fn set_device(&self, device: crate::tensor::Device) {
        // Move parameters — detach first so the moved tensor is a fresh leaf,
        // not a non-leaf with CopyBackward from native autograd.
        for p in self.parameters() {
            if p.variable.data().device() != device
                && let Ok(t) = p.variable.data().detach()
                    .and_then(|d| d.to_device(device))
            {
                p.variable.set_data(t);
            }
        }
        // Move state buffers
        for entry in &self.state {
            let mut val = entry.value.borrow_mut();
            if let Some(ref v) = *val
                && v.data().device() != device
                && let Ok(t) = v.data().to_device(device)
            {
                *val = Some(Variable::new(t, false));
            }
        }
        // Walk modules for move_to_device (BatchNorm running stats, etc.)
        let mut visited = HashSet::new();
        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                crate::nn::walk_modules_visited(
                    module.as_ref(),
                    &mut visited,
                    &mut |m: &dyn crate::nn::Module| m.move_to_device(device),
                );
            }
        }
    }

    /// Return parameters with qualified names: `"prefix/param_name"`.
    ///
    /// The prefix is the tag name if the node is tagged, otherwise the node ID
    /// (e.g. `"linear_1"`). When a node has multiple parameters with the same
    /// name, suffixes `_0`, `_1`, ... are appended to disambiguate.
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        // Build reverse map: node_idx → tag name
        let mut idx_to_tag: HashMap<usize, String> = HashMap::new();
        for (tag, &(ni, _)) in &self.tag_names {
            // First tag wins (deterministic because we only need one prefix)
            idx_to_tag.entry(ni).or_insert_with(|| tag.clone());
        }

        let mut result = Vec::new();
        let mut seen = HashSet::new();

        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                let prefix = idx_to_tag.get(&ni)
                    .cloned()
                    .unwrap_or_else(|| self.nodes[ni].id.clone());

                let params = module.parameters();
                // Check for duplicate param names within this node
                let mut name_counts: HashMap<String, usize> = HashMap::new();
                for p in &params {
                    *name_counts.entry(p.name.clone()).or_insert(0) += 1;
                }

                let mut name_idx: HashMap<String, usize> = HashMap::new();
                for p in params {
                    let ptr = Rc::as_ptr(&p.variable.inner) as usize;
                    if !seen.insert(ptr) {
                        continue;
                    }

                    let qualified = if name_counts[&p.name] > 1 {
                        let idx = name_idx.entry(p.name.clone()).or_insert(0);
                        let q = format!("{}/{}_{}", prefix, p.name, idx);
                        *idx += 1;
                        q
                    } else {
                        format!("{}/{}", prefix, p.name)
                    };

                    result.push((qualified, p));
                }
            }
        }

        result
    }

    /// Return buffers with qualified names, using the same prefix logic
    /// as `named_parameters()`.
    pub fn named_buffers(&self) -> Vec<(String, Buffer)> {
        let mut idx_to_tag: HashMap<usize, String> = HashMap::new();
        for (tag, &(ni, _)) in &self.tag_names {
            idx_to_tag.entry(ni).or_insert_with(|| tag.clone());
        }

        let mut result = Vec::new();
        let mut seen = HashSet::new();

        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                let prefix = idx_to_tag.get(&ni)
                    .cloned()
                    .unwrap_or_else(|| self.nodes[ni].id.clone());

                let bufs = module.buffers();
                let mut name_counts: HashMap<String, usize> = HashMap::new();
                for b in &bufs {
                    *name_counts.entry(b.name.clone()).or_insert(0) += 1;
                }

                let mut name_idx: HashMap<String, usize> = HashMap::new();
                for b in bufs {
                    let ptr = Rc::as_ptr(&b.inner) as usize;
                    if !seen.insert(ptr) {
                        continue;
                    }

                    let qualified = if name_counts[&b.name] > 1 {
                        let idx = name_idx.entry(b.name.clone()).or_insert(0);
                        let q = format!("{}/{}_{}", prefix, b.name, idx);
                        *idx += 1;
                        q
                    } else {
                        format!("{}/{}", prefix, b.name)
                    };

                    result.push((qualified, b));
                }
            }
        }

        result
    }

    /// Human-readable label set via `FlowBuilder::label()`.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Full 64-character hex structural hash (computed lazily, cached).
    pub fn structural_hash(&self) -> &str {
        self.structural_hash_cache.get_or_init(|| self.compute_structural_hash())
    }

    /// First 8 characters of the structural hash.
    pub fn short_hash(&self) -> &str {
        &self.structural_hash()[..8]
    }

    /// Save all parameters and buffers to a checkpoint file.
    ///
    /// Embeds the structural hash for architecture validation on load.
    /// Supports `.gz` extension for gzip compression.
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        let params = self.named_parameters();
        let buffers = self.named_buffers();
        let hash = self.structural_hash();
        crate::nn::save_checkpoint_file(path, &params, &buffers, Some(hash))
    }

    /// Load parameters and buffers from a checkpoint file.
    ///
    /// Validates the structural hash against this graph's architecture.
    /// Returns a [`LoadReport`](crate::nn::LoadReport) describing what was
    /// loaded, skipped, or missing.
    pub fn load_checkpoint(&self, path: &str) -> Result<crate::nn::LoadReport> {
        let params = self.named_parameters();
        let buffers = self.named_buffers();
        let hash = self.structural_hash();
        crate::nn::load_checkpoint_file(path, &params, &buffers, Some(hash))
    }

    fn compute_structural_hash(&self) -> String {
        let mut hasher = Sha256::new();

        // 1. Nodes in topological order
        for &ni in &self.order {
            let node = &self.nodes[ni];
            hasher.update(node.id.as_bytes());
            hasher.update(b"\0");

            if let Some(ref module) = node.module {
                hasher.update(module.name().as_bytes());
                hasher.update(b"\0");

                // Sorted parameters
                let mut params: Vec<_> = module.parameters().into_iter()
                    .map(|p| (p.name.clone(), p.variable.shape()))
                    .collect();
                params.sort_by(|a, b| a.0.cmp(&b.0));
                for (name, shape) in &params {
                    hasher.update(b"P");
                    hasher.update(name.as_bytes());
                    hasher.update(b"\0");
                    for &dim in shape {
                        hasher.update(dim.to_le_bytes());
                    }
                }

                // Sorted buffers
                let mut bufs: Vec<_> = module.buffers().into_iter()
                    .map(|b| (b.name.clone(), b.shape()))
                    .collect();
                bufs.sort_by(|a, b| a.0.cmp(&b.0));
                for (name, shape) in &bufs {
                    hasher.update(b"B");
                    hasher.update(name.as_bytes());
                    hasher.update(b"\0");
                    for &dim in shape {
                        hasher.update(dim.to_le_bytes());
                    }
                }

                // Nested graph hash
                if let Some(nested_hash) = module.structural_hash() {
                    hasher.update(b"G");
                    hasher.update(nested_hash.as_bytes());
                }
            }
        }

        // 2. Edges
        hasher.update(b"EDGES");
        for edge in &self.edges {
            hasher.update(edge.from_node.as_bytes());
            hasher.update(b"\0");
            hasher.update(edge.from_port.as_bytes());
            hasher.update(b"\0");
            hasher.update(edge.to_node.as_bytes());
            hasher.update(b"\0");
            hasher.update(edge.to_port.as_bytes());
            hasher.update(b"\0");
        }

        // 3. Tags (sorted)
        hasher.update(b"TAGS");
        let mut tags: Vec<_> = self.tag_names.iter().collect();
        tags.sort_by(|a, b| a.0.cmp(b.0));
        for (name, (node_idx, port_idx)) in &tags {
            hasher.update(name.as_bytes());
            hasher.update(b"\0");
            hasher.update((*node_idx as u64).to_le_bytes());
            hasher.update((*port_idx as u64).to_le_bytes());
        }

        // 4. Input/output ports
        hasher.update(b"INPUTS");
        for port in &self.inputs {
            hasher.update(port.name.as_bytes());
            hasher.update(b"\0");
            hasher.update(port.node_id.as_bytes());
            hasher.update(b"\0");
            hasher.update(port.port.as_bytes());
            hasher.update(b"\0");
        }
        hasher.update(b"OUTPUTS");
        for port in &self.outputs {
            hasher.update(port.name.as_bytes());
            hasher.update(b"\0");
            hasher.update(port.node_id.as_bytes());
            hasher.update(b"\0");
            hasher.update(port.port.as_bytes());
            hasher.update(b"\0");
        }

        format!("{:064x}", hasher.finalize())
    }
}

impl Module for Graph {
    fn name(&self) -> &str { "graph" }

    fn structural_hash(&self) -> Option<String> {
        Some(self.structural_hash().to_string())
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.forward_impl(std::slice::from_ref(input))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        let mut seen = HashSet::new();

        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                for p in module.parameters() {
                    let ptr = Rc::as_ptr(&p.variable.inner) as usize;
                    if seen.insert(ptr) {
                        params.push(p);
                    }
                }
            }
        }

        params
    }

    fn set_training(&self, training: bool) {
        let mut visited = HashSet::new();
        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                crate::nn::walk_modules_visited(
                    module.as_ref(),
                    &mut visited,
                    &mut |m: &dyn crate::nn::Module| m.set_training(training),
                );
            }
        }
    }

    fn move_to_device(&self, device: crate::tensor::Device) {
        self.set_device(device);
    }
}

/// Current time as seconds since epoch (monotonic approximation for ETA).
fn instant_secs() -> f64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Kahn's algorithm with level grouping for parallel execution.
fn topological_levels(
    nodes: &[Node],
    node_index: &HashMap<String, usize>,
    edges: &[Edge],
) -> Result<Vec<Vec<usize>>> {
    let n = nodes.len();

    // Build unique dependency sets (node-level, not edge-level).
    // dependents uses BTreeSet so iteration follows node index order,
    // making the topological sort deterministic across runs.
    let mut deps: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut dependents: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];

    for edge in edges {
        let from_ni = node_index[&edge.from_node];
        let to_ni = node_index[&edge.to_node];
        deps[to_ni].insert(from_ni);
        dependents[from_ni].insert(to_ni);
    }

    let mut in_degree: Vec<usize> = deps.iter().map(|d| d.len()).collect();

    // Seed with zero in-degree nodes
    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut levels = Vec::new();
    let mut visited = 0;

    while !queue.is_empty() {
        levels.push(queue.clone());
        visited += queue.len();

        let mut next_queue = Vec::new();
        for &ni in &queue {
            for &dep in &dependents[ni] {
                in_degree[dep] -= 1;
                if in_degree[dep] == 0 {
                    next_queue.push(dep);
                }
            }
        }
        queue = next_queue;
    }

    if visited != n {
        return Err(TensorError::new("cycle detected in graph"));
    }

    Ok(levels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Variable;
    use crate::nn::{Linear, NamedInputModule, ReLU, Sigmoid, mse_loss, Optimizer, SGD};
    use crate::tensor::Tensor;
    use std::collections::HashMap;

    fn from_f32(data: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_f32(data, shape, crate::tensor::test_device()).unwrap()
    }

    // --- Helper modules for testing ---

    /// Doubles the input: forward(x) = 2*x
    struct Doubler;
    impl Module for Doubler {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            input.add(input)
        }
    }

    /// Adds a learnable bias at each step (for gradient accumulation testing).
    struct BiasStep {
        bias: Parameter,
    }
    impl BiasStep {
        fn new(size: i64) -> Result<Self> {
            let data = Tensor::zeros(&[size], crate::tensor::test_opts())?;
            let var = Variable::new(data, true);
            Ok(BiasStep {
                bias: Parameter {
                    variable: var,
                    name: "loop_bias".to_string(),
                },
            })
        }
    }
    impl Module for BiasStep {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            input.add(&self.bias.variable)
        }
        fn parameters(&self) -> Vec<Parameter> {
            vec![self.bias.clone()]
        }
    }

    /// Module that adds a tagged ref to the stream (for Using tests).
    struct AddRefModule;
    impl Module for AddRefModule {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            Ok(input.clone())
        }
        fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }
    }
    impl NamedInputModule for AddRefModule {
        fn forward_named(
            &self,
            input: &Variable,
            refs: &HashMap<String, Variable>,
        ) -> Result<Variable> {
            if let Some(ctx) = refs.get("ctx") {
                input.add(ctx)
            } else {
                Ok(input.clone())
            }
        }
    }

    // --- Core graph tests (from before) ---

    #[test]
    fn test_single_module() {
        let l = Linear::on_device(3, 2, crate::tensor::test_device()).unwrap();
        let graph = FlowBuilder::from(l).build().unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn test_linear_chain() {
        let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .through(ReLU::new())
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn test_also_residual() {
        let l1 = Linear::on_device(3, 3, crate::tensor::test_device()).unwrap();
        l1.weight.variable.set_data(from_f32(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
        ));
        l1.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0, 0.0], &[3]));

        let l2 = Linear::on_device(3, 3, crate::tensor::test_device()).unwrap();
        l2.weight.variable.set_data(from_f32(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
        ));
        l2.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[1.0, 1.0, 1.0], &[3]));

        // l1(x) + l2(l1(x)) = x + (x + 1) = 2x + 1
        let graph = FlowBuilder::from(l1).also(l2).build().unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 5.0).abs() < 1e-5);
        assert!((data[2] - 7.0).abs() < 1e-5);
    }

    // --- Fork tests ---

    #[test]
    fn test_fork_basic() {
        // Fork runs a side module but main stream continues unchanged.
        // identity(x) → fork(linear) tagged "side" → through(ReLU)
        // Main stream: ReLU(identity(x)) = ReLU(x)
        // Side output: linear(x) accessible via tagged("side")
        let l = Linear::on_device(2, 3, crate::tensor::test_device()).unwrap();

        let graph = FlowBuilder::from(Identity)
            .fork(l)
            .tag("side")
            .through(ReLU::new())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, -2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();

        // Main stream went through ReLU(identity(x)) → shape [1, 2]
        assert_eq!(y.shape(), vec![1, 2]);
        let data = y.data().to_f32_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 0.0).abs() < 1e-5); // ReLU(-2) = 0

        // Side output is linear(x) → shape [1, 3]
        let side = graph.tagged("side").unwrap();
        assert_eq!(side.shape(), vec![1, 3]);
    }

    #[test]
    fn test_fork_multiple() {
        // Two forks from the same stream: letter_head and case_head pattern
        let head_a = Linear::on_device(4, 3, crate::tensor::test_device()).unwrap();
        let head_b = Linear::on_device(4, 2, crate::tensor::test_device()).unwrap();

        let graph = FlowBuilder::from(Linear::on_device(2, 4, crate::tensor::test_device()).unwrap())
            .tag("latent")
            .fork(head_a)
            .tag("head_a")
            .fork(head_b)
            .tag("head_b")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();

        // Main stream is still the linear(2→4) output
        assert_eq!(y.shape(), vec![1, 4]);

        // Both forks produced their outputs
        let a = graph.tagged("head_a").unwrap();
        assert_eq!(a.shape(), vec![1, 3]);
        let b = graph.tagged("head_b").unwrap();
        assert_eq!(b.shape(), vec![1, 2]);
    }

    #[test]
    fn test_fork_backward() {
        // Gradients flow through both forks and the main stream
        let graph = FlowBuilder::from(Linear::on_device(2, 4, crate::tensor::test_device()).unwrap())
            .fork(Linear::on_device(4, 3, crate::tensor::test_device()).unwrap())
            .tag("side")
            .through(Linear::on_device(4, 1, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();

        // Loss from main stream + side output
        let side = graph.tagged("side").unwrap();
        let loss = y.sum().unwrap().add(&side.sum().unwrap()).unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some(), "input should have gradient");
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    // --- Split/Merge tests ---

    #[test]
    fn test_split_merge_add() {
        let graph = FlowBuilder::from(Linear::on_device(3, 3, crate::tensor::test_device()).unwrap())
            .split(vec![Box::new(ReLU::new()), Box::new(Sigmoid::new())])
            .merge(MergeOp::Add)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, -1.0, 2.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 3]);
    }

    #[test]
    fn test_split_merge_mean() {
        let l = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
        l.weight
            .variable
            .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        l.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0], &[2]));

        let b1 = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
        b1.weight
            .variable
            .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        b1.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0], &[2]));
        let b2 = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
        b2.weight
            .variable
            .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        b2.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0], &[2]));

        let graph = FlowBuilder::from(l)
            .split(vec![Box::new(b1), Box::new(b2)])
            .merge(MergeOp::Mean)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[3.0, 7.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_parameters() {
        let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .through(ReLU::new())
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let params = graph.parameters();
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_graph_backward() {
        let l1 = Linear::on_device(3, 2, crate::tensor::test_device()).unwrap();
        let l2 = Linear::on_device(2, 1, crate::tensor::test_device()).unwrap();

        let graph = FlowBuilder::from(l1)
            .through(ReLU::new())
            .through(l2)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_graph_as_module() {
        let inner = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .through(ReLU::new())
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = outer.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
        assert_eq!(outer.parameters().len(), 4);
    }

    #[test]
    fn test_training_loop() {
        let graph = FlowBuilder::from(Linear::on_device(1, 1, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let params = graph.parameters();
        let mut optim = SGD::new(&params, 0.01, 0.0);

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1]), false);
        let target = Variable::new(from_f32(&[3.0, 5.0, 7.0, 9.0], &[4, 1]), false);

        let mut last_loss = f64::MAX;
        for _ in 0..800 {
            optim.zero_grad();
            let pred = graph.forward(&x).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();
            last_loss = loss.item().unwrap();
            loss.backward().unwrap();
            optim.step().unwrap();
        }

        assert!(last_loss < 0.01, "got loss={}", last_loss);
    }

    #[test]
    fn test_also_backward() {
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .also(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_split_merge_backward() {
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .split(vec![
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
            ])
            .merge(MergeOp::Add)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_build_error_open_streams() {
        let result = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .split(vec![Box::new(ReLU::new()), Box::new(Sigmoid::new())])
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_build_error_duplicate_tag() {
        let result = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .tag("features")
            .through(ReLU::new())
            .tag("features")
            .build();
        assert!(result.is_err());
    }

    // --- Using tests ---

    #[test]
    fn test_using_backward_ref() {
        // Tag a point, then use it downstream
        // Graph: linear(x) → tag("ctx") → through(AddRef).using("ctx")
        // AddRef adds ctx to stream: stream + ctx = 2 * linear(x)
        let l = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
        l.weight
            .variable
            .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        l.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0], &[2]));

        let graph = FlowBuilder::from(l)
            .tag("ctx")
            .through(AddRefModule)
            .using(&["ctx"])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[3.0, 5.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        // identity(x) = [3, 5], then AddRef adds ctx ([3, 5]) = [6, 10]
        assert!((data[0] - 6.0).abs() < 1e-5);
        assert!((data[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_using_backward_gradients() {
        let l = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
        let graph = FlowBuilder::from(l)
            .tag("ctx")
            .through(AddRefModule)
            .using(&["ctx"])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_using_error_plain_module() {
        // Using on a plain module (not NamedInputModule) should error
        let result = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .tag("ctx")
            .through(ReLU::new())
            .using(&["ctx"])
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_using_error_unknown_tag() {
        let result = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .through(AddRefModule)
            .using(&["nonexistent"])
            .build();
        assert!(result.is_err());
    }

    // --- Loop tests ---

    #[test]
    fn test_loop_for() {
        // Doubler × 3 iterations: [1, 2] → [8, 16]
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Doubler)
            .for_n(3)
            .build()
            .unwrap();

        // Set linear to identity
        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 8.0).abs() < 1e-5, "1*2^3=8, got {}", data[0]);
        assert!((data[1] - 16.0).abs() < 1e-5, "2*2^3=16, got {}", data[1]);
    }

    #[test]
    fn test_loop_for_backward() {
        // Loop with a learnable bias — gradient should accumulate across iterations
        let bias_step = BiasStep::new(2).unwrap();
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(bias_step)
            .for_n(3)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        // All parameters should have gradients
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }

        // The bias gradient should be 3 (accumulated from 3 iterations)
        // dL/db = 1 per iteration, 3 iterations → grad = [3, 3]
        // (because sum reduces to scalar, dL/d_each_element = 1, and bias contributes at each step)
        let all_params = graph.parameters();
        // Find the loop_bias parameter (from BiasStep, not Linear's "bias")
        let bias_param = all_params.iter().find(|p| p.name == "loop_bias").unwrap();
        let grad = bias_param.variable.grad().unwrap().to_f32_vec().unwrap();
        assert!(
            (grad[0] - 3.0).abs() < 1e-5,
            "bias grad should be 3, got {}",
            grad[0]
        );
    }

    #[test]
    fn test_loop_while() {
        // While max < 10: double. Input [1, 2] → double until max >= 10
        // Iter 0: check [1,2] max=2 < 10 → double → [2,4]
        // Iter 1: check [2,4] max=4 < 10 → double → [4,8]
        // Iter 2: check [4,8] max=8 < 10 → double → [8,16]
        // Iter 3: check [8,16] max=16 >= 10 → halt
        // Result: [8, 16]
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Doubler)
            .while_cond(ThresholdHalt::new(10.0), 20)
            .build()
            .unwrap();

        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 8.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 16.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_loop_while_immediate_halt() {
        // Threshold 0.5 — input [1, 2] max=2 > 0.5, halt immediately
        // While checks before body, so body never runs
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Doubler)
            .while_cond(ThresholdHalt::new(0.5), 20)
            .build()
            .unwrap();

        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        // Body never ran — output = input
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_loop_until() {
        // Until max > 10: double. Body runs at least once.
        // Input [1, 2]
        // Iter 0: double → [2, 4], check max=4 <= 10 → continue
        // Iter 1: double → [4, 8], check max=8 <= 10 → continue
        // Iter 2: double → [8, 16], check max=16 > 10 → halt
        // Result: [8, 16]
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Doubler)
            .until_cond(ThresholdHalt::new(10.0), 20)
            .build()
            .unwrap();

        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 8.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 16.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_loop_until_at_least_once() {
        // Until with threshold 0.5 — input [1, 2] would halt immediately in While,
        // but Until always runs body at least once
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Doubler)
            .until_cond(ThresholdHalt::new(0.5), 20)
            .build()
            .unwrap();

        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        // Body ran once: [2, 4]
        assert!((data[0] - 2.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 4.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_loop_parameters() {
        // Loop with learnable body — parameters should include body params
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .for_n(3)
            .build()
            .unwrap();

        let params = graph.parameters();
        // From module: weight + bias = 2, loop body Linear: weight + bias = 2
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_loop_while_parameters() {
        // While loop with body + condition — both contribute parameters
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .while_cond(Linear::on_device(2, 1, crate::tensor::test_device()).unwrap(), 10)
            .build()
            .unwrap();

        let params = graph.parameters();
        // From module: 2, loop body: 2, condition: 2 = 6
        assert_eq!(params.len(), 6);
    }

    #[test]
    fn test_loop_in_chain() {
        // Linear → Loop(ReLU) × 3 → Linear
        let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .loop_body(ReLU::new())
            .for_n(3)
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn test_loop_using_backward_ref() {
        // Tag a tensor, then use it inside a loop body via .using()
        // Graph: identity → tag("ctx") → loop_body(AddRefModule).for_n(3).using("ctx")
        // Each iteration: state = state + ctx
        // So after 3 iterations: state = x + 3*x = 4*x
        let graph = FlowBuilder::from(Identity)
            .tag("ctx")
            .loop_body(AddRefModule)
            .for_n(3)
            .using(&["ctx"])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[2.0, 3.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        // x = [2, 3], after 3 iterations of (state + ctx): [8, 12]
        assert!((data[0] - 8.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 12.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_loop_using_backward_gradients() {
        // Ensure gradients flow through loop+using
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .tag("ctx")
            .loop_body(AddRefModule)
            .for_n(2)
            .using(&["ctx"])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some(), "input should have gradient");
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    // --- Forward reference tests ---

    /// Nil-safe add: skips nil inputs, adds rest. For forward ref state accumulation.
    struct NilSafeAdd;
    impl Module for NilSafeAdd {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            Ok(input.clone())
        }
        fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }
    }
    impl NamedInputModule for NilSafeAdd {
        fn forward_named(
            &self,
            input: &Variable,
            refs: &HashMap<String, Variable>,
        ) -> Result<Variable> {
            if let Some(memory) = refs.get("memory") {
                input.add(memory)
            } else {
                Ok(input.clone())
            }
        }
    }

    use crate::nn::Identity;

    #[test]
    fn test_flowbuilder_new() {
        // FlowBuilder::new() starts with implicit Identity
        let graph = FlowBuilder::new()
            .tag("input")
            .through(Linear::on_device(3, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn test_forward_ref() {
        // Forward reference: using() before tag(). State carries between forward() calls.
        // Graph: entry → NilSafeAdd.Using("memory") → Identity.Tag("memory")
        // Pass 1: add gets [stream, zeros] (memory is nil/zeroed) → Identity → state captured
        // Pass 2: add gets [stream, prev_output] → sum → Identity → state captured
        let graph = FlowBuilder::from(Identity)
            .through(NilSafeAdd)
            .using(&["memory"])
            .through(Identity)
            .tag("memory")
            .build()
            .unwrap();

        assert!(graph.has_state());

        // Pass 1: [1,2] + zeros → [1,2]
        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y1 = graph.forward(&x).unwrap();
        let d1 = y1.data().to_f32_vec().unwrap();
        assert!((d1[0] - 1.0).abs() < 1e-5, "pass1[0]: got {}", d1[0]);
        assert!((d1[1] - 2.0).abs() < 1e-5, "pass1[1]: got {}", d1[1]);

        // Pass 2: [1,2] + [1,2] → [2,4]
        let y2 = graph.forward(&x).unwrap();
        let d2 = y2.data().to_f32_vec().unwrap();
        assert!((d2[0] - 2.0).abs() < 1e-5, "pass2[0]: got {}", d2[0]);
        assert!((d2[1] - 4.0).abs() < 1e-5, "pass2[1]: got {}", d2[1]);

        // Pass 3: [1,2] + [2,4] → [3,6]
        let y3 = graph.forward(&x).unwrap();
        let d3 = y3.data().to_f32_vec().unwrap();
        assert!((d3[0] - 3.0).abs() < 1e-5, "pass3[0]: got {}", d3[0]);
        assert!((d3[1] - 6.0).abs() < 1e-5, "pass3[1]: got {}", d3[1]);
    }

    #[test]
    fn test_forward_ref_reset_state() {
        let graph = FlowBuilder::from(Identity)
            .through(NilSafeAdd)
            .using(&["memory"])
            .through(Identity)
            .tag("memory")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);

        // Build up state
        graph.forward(&x).unwrap();
        graph.forward(&x).unwrap();
        let y_before = graph.forward(&x).unwrap();
        let d_before = y_before.data().to_f32_vec().unwrap();
        assert!((d_before[0] - 3.0).abs() < 1e-5);

        // Reset and verify state is cleared
        graph.reset_state();
        let y_after = graph.forward(&x).unwrap();
        let d_after = y_after.data().to_f32_vec().unwrap();
        assert!((d_after[0] - 1.0).abs() < 1e-5, "after reset: got {}", d_after[0]);
    }

    #[test]
    fn test_forward_ref_detach_state() {
        let graph = FlowBuilder::from(Identity)
            .through(NilSafeAdd)
            .using(&["memory"])
            .through(Identity)
            .tag("memory")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);

        // Run forward, accumulate state
        let y1 = graph.forward(&x).unwrap();
        let _ = y1.sum().unwrap();

        // Detach state — values preserved but gradient chain broken
        graph.detach_state();

        // State should still have values (not reset)
        let y2 = graph.forward(&x).unwrap();
        let d2 = y2.data().to_f32_vec().unwrap();
        assert!((d2[0] - 2.0).abs() < 1e-5, "detach preserves values: got {}", d2[0]);
    }

    #[test]
    fn test_forward_ref_backward() {
        // Gradients should flow through forward-ref connections
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .through(NilSafeAdd)
            .using(&["memory"])
            .through(Identity)
            .tag("memory")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some(), "input should have gradient");
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_forward_ref_unresolved_error() {
        // Using a tag that is never defined should error at build
        let result = FlowBuilder::from(Identity)
            .through(NilSafeAdd)
            .using(&["nonexistent"])
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_ref_mixed_refs() {
        // Mix backward ref (tag before using) and forward ref (using before tag)
        // "ctx" is backward (AddRefModule expects "ctx"), "memory" is forward (NilSafeAdd expects "memory")
        let graph = FlowBuilder::from(Identity)
            .tag("ctx")
            .through(AddRefModule)
            .using(&["ctx"])
            .through(NilSafeAdd)
            .using(&["memory"])
            .through(Identity)
            .tag("memory")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);

        // Pass 1: entry=[1,2], AddRef adds ctx=[1,2] → [2,4], NilSafeAdd +zeros → [2,4]
        let y1 = graph.forward(&x).unwrap();
        let d1 = y1.data().to_f32_vec().unwrap();
        assert!((d1[0] - 2.0).abs() < 1e-5, "mixed pass1[0]: got {}", d1[0]);

        // Pass 2: entry=[1,2], AddRef adds ctx=[1,2] → [2,4], NilSafeAdd +[2,4] → [4,8]
        let y2 = graph.forward(&x).unwrap();
        let d2 = y2.data().to_f32_vec().unwrap();
        assert!((d2[0] - 4.0).abs() < 1e-5, "mixed pass2[0]: got {}", d2[0]);
    }

    // --- Switch tests ---

    /// Triples input.
    struct Tripler;
    impl Module for Tripler {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            input.add(&input.add(input)?)
        }
        fn parameters(&self) -> Vec<Parameter> { vec![] }
    }

    #[test]
    fn test_switch_selects_branch() {
        // Branch 0: double, Branch 1: triple. Router selects branch 1.
        let graph = FlowBuilder::from(Identity)
            .switch(FixedSelector::new(1), vec![Box::new(Doubler), Box::new(Tripler)])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        assert!((data[0] - 3.0).abs() < 1e-5, "triple [1]=3, got {}", data[0]);
        assert!((data[1] - 6.0).abs() < 1e-5, "triple [2]=6, got {}", data[1]);
    }

    #[test]
    fn test_switch_branch0() {
        let graph = FlowBuilder::from(Identity)
            .switch(FixedSelector::new(0), vec![Box::new(Doubler), Box::new(Tripler)])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-5, "double [1]=2, got {}", data[0]);
        assert!((data[1] - 4.0).abs() < 1e-5, "double [2]=4, got {}", data[1]);
    }

    #[test]
    fn test_switch_backward() {
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .switch(FixedSelector::new(0), vec![
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
            ])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        // Only entry + selected branch params should have gradients
        // (router has no params, unselected branch wasn't executed)
    }

    #[test]
    fn test_switch_parameters() {
        let graph = FlowBuilder::from(Identity)
            .switch(
                Linear::on_device(2, 1, crate::tensor::test_device()).unwrap(),
                vec![
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                ],
            )
            .build()
            .unwrap();

        let params = graph.parameters();
        // Router: 2, Branch0: 2, Branch1: 2 = 6
        assert_eq!(params.len(), 6);
    }

    // --- Gate tests ---

    /// Router that outputs equal weights for all experts.
    struct EqualRouter(usize);
    impl Module for EqualRouter {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            let batch = input.shape()[0];
            let w = 1.0 / self.0 as f32;
            let data = vec![w; batch as usize * self.0];
            Ok(Variable::new(
                Tensor::from_f32(&data, &[batch, self.0 as i64], crate::tensor::test_device())?,
                false,
            ))
        }
        fn parameters(&self) -> Vec<Parameter> { vec![] }
    }

    #[test]
    fn test_gate_equal_weights() {
        // Equal weights: output = mean of expert outputs
        let graph = FlowBuilder::from(Identity)
            .gate(EqualRouter(2), vec![Box::new(Doubler), Box::new(Tripler)])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[2.0, 4.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // double=[4,8], triple=[6,12], mean = [5, 10]
        assert!((data[0] - 5.0).abs() < 1e-5, "gate[0]=5, got {}", data[0]);
        assert!((data[1] - 10.0).abs() < 1e-5, "gate[1]=10, got {}", data[1]);
    }

    #[test]
    fn test_gate_backward() {
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .gate(
                Linear::on_device(2, 2, crate::tensor::test_device()).unwrap(),
                vec![
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                ],
            )
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_gate_parameters() {
        let graph = FlowBuilder::from(Identity)
            .gate(
                Linear::on_device(2, 2, crate::tensor::test_device()).unwrap(),
                vec![
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                ],
            )
            .build()
            .unwrap();

        let params = graph.parameters();
        // Router: 2, Expert0: 2, Expert1: 2 = 6
        assert_eq!(params.len(), 6);
    }

    // --- Map tests ---

    #[test]
    fn test_map_each() {
        // Map doubler over 3 elements along dim 0
        let graph = FlowBuilder::from(Identity)
            .map(Doubler)
            .each()
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert_eq!(y.shape(), vec![3, 2]);
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[5] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_map_batched() {
        // Batched: pass full tensor, skip element-wise
        let graph = FlowBuilder::from(Identity)
            .map(Doubler)
            .batched()
            .each()
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_map_backward() {
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .map(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .each()
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    // --- Observation tests ---

    /// Scalar output module: sum all elements to a single value.
    struct ScalarSum;
    impl Module for ScalarSum {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            input.sum()
        }
    }

    #[test]
    fn test_tagged_capture() {
        // Tag intermediate output and retrieve it after forward
        let graph = FlowBuilder::from(Identity)
            .tag("features")
            .through(Doubler)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let _ = graph.forward(&x).unwrap();

        // Tagged value should be the identity output (before doubling)
        let features = graph.tagged("features").unwrap();
        let data = features.data().to_f32_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);

        assert!(graph.tagged("nonexistent").is_none());
    }

    #[test]
    fn test_tagged_updates_each_forward() {
        let graph = FlowBuilder::from(Doubler)
            .tag("doubled")
            .build()
            .unwrap();

        let x1 = Variable::new(from_f32(&[1.0], &[1, 1]), false);
        let _ = graph.forward(&x1).unwrap();
        let v1 = graph.tagged("doubled").unwrap().item().unwrap();
        assert!((v1 - 2.0).abs() < 1e-5);

        let x2 = Variable::new(from_f32(&[5.0], &[1, 1]), false);
        let _ = graph.forward(&x2).unwrap();
        let v2 = graph.tagged("doubled").unwrap().item().unwrap();
        assert!((v2 - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_tag_names() {
        let graph = FlowBuilder::from(Identity)
            .tag("a")
            .through(Identity)
            .tag("b")
            .build()
            .unwrap();

        let mut names = graph.tag_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_collect_flush_trend() {
        // Simulate a training loop with collect → flush → trend
        let graph = FlowBuilder::from(ScalarSum)
            .tag("loss")
            .build()
            .unwrap();

        // Epoch 1: 3 batches with different inputs
        for val in &[1.0f32, 2.0, 3.0] {
            let x = Variable::new(from_f32(&[*val], &[1, 1]), false);
            let _ = graph.forward(&x).unwrap();
            graph.collect(&["loss"]).unwrap();
        }
        // batch buffer should have [1, 2, 3]
        let collected = graph.collected("loss");
        assert_eq!(collected.len(), 3);

        graph.flush(&["loss"]);
        assert_eq!(graph.flush_count(), 1);

        // Epoch 2: 3 batches
        for val in &[0.5f32, 0.3, 0.2] {
            let x = Variable::new(from_f32(&[*val], &[1, 1]), false);
            let _ = graph.forward(&x).unwrap();
            graph.collect(&["loss"]).unwrap();
        }
        graph.flush(&["loss"]);
        assert_eq!(graph.flush_count(), 2);

        // Trend should show decrease: epoch1 mean=2.0, epoch2 mean≈0.333
        let trend = graph.trend("loss");
        assert_eq!(trend.len(), 2);
        assert!((trend.values()[0] - 2.0).abs() < 1e-5);
        assert!((trend.values()[1] - (1.0 / 3.0)).abs() < 1e-5);
        assert!(trend.improving(0));
    }

    #[test]
    fn test_record_external_values() {
        let graph = FlowBuilder::from(Identity).build().unwrap();

        graph.record("external_loss", &[0.5, 0.4, 0.3]);
        graph.flush(&["external_loss"]);

        graph.record("external_loss", &[0.1, 0.05]);
        graph.flush(&["external_loss"]);

        let trend = graph.trend("external_loss");
        assert_eq!(trend.len(), 2);
        assert!((trend.values()[0] - 0.4).abs() < 1e-5); // mean(0.5, 0.4, 0.3)
        assert!((trend.values()[1] - 0.075).abs() < 1e-5); // mean(0.1, 0.05)
        assert!(trend.improving(0));
    }

    #[test]
    fn test_flush_all() {
        let graph = FlowBuilder::from(Identity).build().unwrap();

        graph.record("a", &[1.0, 2.0]);
        graph.record("b", &[3.0, 4.0]);
        graph.flush(&[]); // flush all

        assert_eq!(graph.trend("a").len(), 1);
        assert_eq!(graph.trend("b").len(), 1);
    }

    #[test]
    fn test_reset_trend() {
        let graph = FlowBuilder::from(Identity).build().unwrap();

        graph.record("loss", &[1.0]);
        graph.flush(&[]);
        assert_eq!(graph.trend("loss").len(), 1);

        graph.reset_trend(&["loss"]);
        assert_eq!(graph.trend("loss").len(), 0);
    }

    #[test]
    fn test_trends_group() {
        let graph = FlowBuilder::from(Identity).build().unwrap();

        // Two decreasing series
        for epoch in &[10.0, 8.0, 6.0, 4.0] {
            graph.record("a", &[*epoch]);
            graph.record("b", &[*epoch * 0.5]);
            graph.flush(&[]);
        }

        let tg = graph.trends(&["a", "b"]);
        assert_eq!(tg.len(), 2);
        assert!(tg.all_improving(0));
    }

    // --- TagGroup tests ---

    #[test]
    fn test_tag_group() {
        // Split into 3 branches with tag_group, then merge
        let graph = FlowBuilder::from(Identity)
            .split(vec![
                Box::new(Doubler),
                Box::new(Tripler),
                Box::new(Identity),
            ])
            .tag_group("branch")
            .merge(MergeOp::Add)
            .build()
            .unwrap();

        // Check group registration
        let members = graph.tag_group("branch").unwrap();
        assert_eq!(members, &["branch_0", "branch_1", "branch_2"]);

        // Non-existent group returns None
        assert!(graph.tag_group("nonexistent").is_none());

        // Tags work for observation
        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let _ = graph.forward(&x).unwrap();

        let b0 = graph.tagged("branch_0").unwrap();
        let b0_data = b0.data().to_f32_vec().unwrap();
        assert!((b0_data[0] - 2.0).abs() < 1e-5, "doubler: got {}", b0_data[0]);

        let b1 = graph.tagged("branch_1").unwrap();
        let b1_data = b1.data().to_f32_vec().unwrap();
        assert!((b1_data[0] - 3.0).abs() < 1e-5, "tripler: got {}", b1_data[0]);
    }

    #[test]
    fn test_tag_group_observation() {
        // Tag group with collect/flush and trends expansion
        let graph = FlowBuilder::from(Identity)
            .split(vec![Box::new(ScalarSum), Box::new(ScalarSum)])
            .tag_group("head")
            .merge(MergeOp::Add)
            .build()
            .unwrap();

        // Run a few epochs
        for epoch in &[1.0f32, 2.0, 3.0] {
            let x = Variable::new(from_f32(&[*epoch], &[1, 1]), false);
            let _ = graph.forward(&x).unwrap();
            graph.collect(&["head_0", "head_1"]).unwrap();
            graph.flush(&["head_0", "head_1"]);
        }

        // Trends with group expansion
        let tg = graph.trends(&["head"]);
        assert_eq!(tg.len(), 2); // head_0 and head_1
    }

    #[test]
    fn test_tag_group_errors() {
        // tag_group on single stream should error
        let result = FlowBuilder::from(Identity)
            .tag_group("bad")
            .build();
        assert!(result.is_err());

        // Duplicate group name
        let result = FlowBuilder::from(Identity)
            .split(vec![Box::new(Doubler), Box::new(Tripler)])
            .tag_group("x")
            .merge(MergeOp::Add)
            .split(vec![Box::new(Doubler), Box::new(Tripler)])
            .tag_group("x")
            .merge(MergeOp::Add)
            .build();
        assert!(result.is_err());
    }

    // --- Input tests ---

    /// Module that adds all refs to input (for multi-input testing).
    struct SumRefs;
    impl Module for SumRefs {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            Ok(input.clone())
        }
        fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }
    }
    impl NamedInputModule for SumRefs {
        fn forward_named(
            &self,
            input: &Variable,
            refs: &HashMap<String, Variable>,
        ) -> Result<Variable> {
            let mut result = input.clone();
            for v in refs.values() {
                result = result.add(v)?;
            }
            Ok(result)
        }
    }

    #[test]
    fn test_input_auxiliary() {
        // Graph with auxiliary inputs: From(identity) + Input("ctx")
        // Downstream: through(SumRefs).using("ctx")
        let graph = FlowBuilder::from(Identity)
            .input(&["ctx"])
            .through(SumRefs)
            .using(&["ctx"])
            .build()
            .unwrap();

        let main = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let ctx = Variable::new(from_f32(&[10.0, 20.0], &[1, 2]), false);

        let y = graph.forward_multi(&[main, ctx]).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // SumRefs adds ctx to main: [1+10, 2+20] = [11, 22]
        assert!((data[0] - 11.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 22.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_input_multiple() {
        // Graph with two auxiliary inputs
        let graph = FlowBuilder::from(Identity)
            .input(&["a", "b"])
            .through(SumRefs)
            .using(&["a", "b"])
            .build()
            .unwrap();

        let main = Variable::new(from_f32(&[1.0], &[1, 1]), false);
        let a = Variable::new(from_f32(&[10.0], &[1, 1]), false);
        let b = Variable::new(from_f32(&[100.0], &[1, 1]), false);

        let y = graph.forward_multi(&[main, a, b]).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // 1 + 10 + 100 = 111
        assert!((data[0] - 111.0).abs() < 1e-5, "got {}", data[0]);
    }

    #[test]
    fn test_input_error_count_mismatch() {
        let graph = FlowBuilder::from(Identity)
            .input(&["ctx"])
            .build()
            .unwrap();

        // forward() with single input should fail (expects 2: main + ctx)
        let x = Variable::new(from_f32(&[1.0], &[1, 1]), false);
        assert!(graph.forward(&x).is_err());
    }

    // --- Graph set_training test ---

    #[test]
    fn test_graph_set_training() {
        use crate::nn::Dropout;

        let graph = FlowBuilder::from(Linear::on_device(3, 3, crate::tensor::test_device()).unwrap())
            .through(Dropout::new(0.5))
            .build()
            .unwrap();

        // Training mode: dropout is active
        let x = Variable::new(from_f32(&[1.0; 12], &[4, 3]), false);
        let y1 = graph.forward(&x).unwrap();
        assert_eq!(y1.shape(), vec![4, 3]);

        // Set eval via graph
        graph.set_training(false);
        let y2 = graph.forward(&x).unwrap();
        let y3 = graph.forward(&x).unwrap();
        assert_eq!(y2.shape(), vec![4, 3]);

        // In eval: dropout is identity, so repeated forward gives same output
        let d2 = y2.data().to_f32_vec().unwrap();
        let d3 = y3.data().to_f32_vec().unwrap();
        let same = d2.iter().zip(d3.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(same, "eval mode should be deterministic (no dropout)");
    }

    // --- walk_modules test ---

    #[test]
    fn test_walk_modules() {
        use crate::nn::walk_modules;

        let l1 = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
        let mut count = 0;
        walk_modules(&l1, &mut |_| count += 1);
        assert_eq!(count, 1); // leaf module, no children
    }

    // --- Profiling tests ---

    #[test]
    fn test_profiling_basic() {
        let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .tag("encoder")
            .through(ReLU::new())
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .tag("decoder")
            .build()
            .unwrap();

        // No profiling by default
        assert!(!graph.profiling());
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        graph.forward(&x).unwrap();
        assert!(graph.profile().is_none());

        // Enable profiling
        graph.enable_profiling();
        assert!(graph.profiling());
        graph.forward(&x).unwrap();

        let p = graph.profile().unwrap();
        assert!(p.total.as_nanos() > 0, "total should be nonzero");
        assert!(!p.nodes.is_empty(), "should have node timings");
        assert!(!p.levels.is_empty(), "should have level timings");

        // Tagged node timing
        let enc_dur = p.timing("encoder");
        assert!(enc_dur.as_nanos() > 0, "encoder timing should be nonzero");
        let dec_dur = p.timing("decoder");
        assert!(dec_dur.as_nanos() > 0, "decoder timing should be nonzero");
        assert!(p.timing("nonexistent").is_zero());

        // Graph-level timing shortcut
        assert!(graph.timing("encoder").as_nanos() > 0);

        // Display
        let s = p.to_string();
        assert!(s.contains("Forward:"));
        assert!(s.contains("Level"));

        // Disable
        graph.disable_profiling();
        assert!(!graph.profiling());
        graph.forward(&x).unwrap();
        assert!(graph.profile().is_none());
    }

    #[test]
    fn test_profiling_timing_trend() {
        let graph = FlowBuilder::from(ScalarSum)
            .tag("loss")
            .build()
            .unwrap();

        graph.enable_profiling();

        // Simulate 2 epochs, 3 batches each
        for _ in 0..2 {
            for val in &[1.0f32, 2.0, 3.0] {
                let x = Variable::new(from_f32(&[*val], &[1, 1]), false);
                graph.forward(&x).unwrap();
                graph.collect_timings(&["loss"]);
            }
            graph.flush_timings(&[]);
        }

        let trend = graph.timing_trend("loss");
        assert_eq!(trend.len(), 2, "2 epochs flushed");
        assert!(trend.values()[0] > 0.0, "timing values should be positive");

        // Reset
        graph.reset_timing_trend(&["loss"]);
        assert_eq!(graph.timing_trend("loss").len(), 0);
    }

    // --- DOT tests ---

    #[test]
    fn test_dot_basic() {
        let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .tag("enc")
            .through(ReLU::new())
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let dot = graph.dot();
        assert!(dot.contains("digraph G"));
        assert!(dot.contains("level 0"));
        assert!(dot.contains("#enc"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_dot_with_profile() {
        let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .tag("enc")
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);

        // Without profiling: dot_with_profile falls back to structural
        let dot1 = graph.dot_with_profile();
        assert!(dot1.contains("digraph G"));

        // With profiling: includes timing annotations
        graph.enable_profiling();
        graph.forward(&x).unwrap();
        let dot2 = graph.dot_with_profile();
        assert!(dot2.contains("digraph G"));
        assert!(dot2.contains("Forward:"));
    }

    // --- Traced tests ---

    /// A loop body that implements trace() — captures per-iteration side data.
    struct TracingDoubler {
        last_output: RefCell<Option<Variable>>,
    }
    impl TracingDoubler {
        fn new() -> Self {
            TracingDoubler {
                last_output: RefCell::new(None),
            }
        }
    }
    impl Module for TracingDoubler {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            let out = input.add(input)?;
            *self.last_output.borrow_mut() = Some(out.clone());
            Ok(out)
        }
        fn trace(&self) -> Option<Variable> {
            self.last_output.borrow().clone()
        }
    }

    #[test]
    fn test_loop_traces() {
        // Loop(TracingDoubler) × 3: [1,2] → [2,4] → [4,8] → [8,16]
        // traces should capture [2,4], [4,8], [8,16]
        let graph = FlowBuilder::from(Identity)
            .loop_body(TracingDoubler::new())
            .for_n(3)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        assert!((data[0] - 8.0).abs() < 1e-5);

        // Get traces — should find them on the loop node
        let traces = graph.traces("any").unwrap();
        assert_eq!(traces.len(), 3, "3 iterations = 3 traces");

        let t0 = traces[0].data().to_f32_vec().unwrap();
        assert!((t0[0] - 2.0).abs() < 1e-5, "iter0: [2,4], got {}", t0[0]);

        let t1 = traces[1].data().to_f32_vec().unwrap();
        assert!((t1[0] - 4.0).abs() < 1e-5, "iter1: [4,8], got {}", t1[0]);

        let t2 = traces[2].data().to_f32_vec().unwrap();
        assert!((t2[0] - 8.0).abs() < 1e-5, "iter2: [8,16], got {}", t2[0]);
    }

    #[test]
    fn test_loop_traces_cleared_each_forward() {
        let graph = FlowBuilder::from(Identity)
            .loop_body(TracingDoubler::new())
            .for_n(2)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0], &[1, 1]), false);
        graph.forward(&x).unwrap();
        let traces1 = graph.traces("any").unwrap();
        assert_eq!(traces1.len(), 2);

        // Second forward should clear and re-populate
        graph.forward(&x).unwrap();
        let traces2 = graph.traces("any").unwrap();
        assert_eq!(traces2.len(), 2);
    }

    #[test]
    fn test_loop_no_traces_without_trace_impl() {
        // Doubler doesn't implement trace() (returns None by default)
        let graph = FlowBuilder::from(Identity)
            .loop_body(Doubler)
            .for_n(3)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0], &[1, 1]), false);
        graph.forward(&x).unwrap();

        // No traces since Doubler's trace() returns None
        assert!(graph.traces("any").is_none());
    }

    // --- Router tests ---

    #[test]
    fn test_softmax_router_gate() {
        // SoftmaxRouter with 2 experts: double + triple, weights from learned router
        let graph = FlowBuilder::from(Identity)
            .gate(
                SoftmaxRouter::on_device(2, 2, crate::tensor::test_device()).unwrap(),
                vec![Box::new(Doubler), Box::new(Tripler)],
            )
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        // Output should be a weighted combination — just verify it runs and has correct shape
        assert_eq!(y.shape(), vec![1, 2]);
        // Router has 2 params (weight + bias), experts have 0
        let params = graph.parameters();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_softmax_router_backward() {
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .gate(
                SoftmaxRouter::on_device(2, 2, crate::tensor::test_device()).unwrap(),
                vec![
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                ],
            )
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} missing gradient", p.name);
        }
    }

    #[test]
    fn test_sigmoid_router_gate() {
        let graph = FlowBuilder::from(Identity)
            .gate(
                SigmoidRouter::on_device(2, 2, crate::tensor::test_device()).unwrap(),
                vec![Box::new(Doubler), Box::new(Tripler)],
            )
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn test_fixed_selector_switch() {
        // FixedSelector(1) always picks branch 1 (Tripler)
        let graph = FlowBuilder::from(Identity)
            .switch(FixedSelector::new(1), vec![Box::new(Doubler), Box::new(Tripler)])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[2.0, 3.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        assert!((data[0] - 6.0).abs() < 1e-5, "triple 2=6, got {}", data[0]);
        assert!((data[1] - 9.0).abs() < 1e-5, "triple 3=9, got {}", data[1]);
    }

    #[test]
    fn test_argmax_selector_switch() {
        let graph = FlowBuilder::from(Identity)
            .switch(
                ArgmaxSelector::on_device(2, 2, crate::tensor::test_device()).unwrap(),
                vec![Box::new(Doubler), Box::new(Tripler)],
            )
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        // Should select one branch — just verify it runs and has correct shape
        assert_eq!(y.shape(), vec![1, 2]);
        // ArgmaxSelector has params from its Linear projection
        assert_eq!(graph.parameters().len(), 2);
    }

    // --- Halt tests ---

    #[test]
    fn test_threshold_halt_while() {
        // body = Doubler, halt when max > 10
        // input [1,2] → iter1 [2,4] → iter2 [4,8] → iter3 [8,16] halt (16 > 10)
        let graph = FlowBuilder::from(Identity)
            .loop_body(Doubler)
            .while_cond(ThresholdHalt::new(10.0), 20)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // Should stop at [8, 16] (max=16 > 10)
        assert!((data[0] - 8.0).abs() < 1e-5, "expected 8, got {}", data[0]);
        assert!((data[1] - 16.0).abs() < 1e-5, "expected 16, got {}", data[1]);
    }

    #[test]
    fn test_threshold_halt_until() {
        // Until: body runs first, then check
        // input [1,2] → iter1 body [2,4] check (max=4 < 10 continue)
        //             → iter2 body [4,8] check (max=8 < 10 continue)
        //             → iter3 body [8,16] check (max=16 > 10 halt)
        let graph = FlowBuilder::from(Identity)
            .loop_body(Doubler)
            .until_cond(ThresholdHalt::new(10.0), 20)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // Should stop at [8, 16] (max=16 > 10)
        assert!((data[0] - 8.0).abs() < 1e-5, "expected 8, got {}", data[0]);
        assert!((data[1] - 16.0).abs() < 1e-5, "expected 16, got {}", data[1]);
    }

    #[test]
    fn test_threshold_halt_immediate() {
        // Threshold already exceeded: while should not iterate
        let graph = FlowBuilder::from(Identity)
            .loop_body(Doubler)
            .while_cond(ThresholdHalt::new(0.5), 20)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // max=2.0 > 0.5 → halt immediately, input passes through
        assert!((data[0] - 1.0).abs() < 1e-5, "expected 1, got {}", data[0]);
        assert!((data[1] - 2.0).abs() < 1e-5, "expected 2, got {}", data[1]);
    }

    #[test]
    fn test_learned_halt_parameters() {
        let graph = FlowBuilder::from(Identity)
            .loop_body(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .until_cond(LearnedHalt::on_device(2, crate::tensor::test_device()).unwrap(), 5)
            .build()
            .unwrap();

        // Body Linear: 2 params, LearnedHalt Linear(2→1): 2 params = 4
        let params = graph.parameters();
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_named_parameters_unique() {
        let graph = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(ReLU::new())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let named = graph.named_parameters();
        // Two Linear layers: 2 params each (weight + bias) = 4
        assert_eq!(named.len(), 4);

        // All names should be unique
        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(names.len(), unique.len(), "duplicate names: {:?}", names);
    }

    #[test]
    fn test_named_parameters_tagged_prefix() {
        let graph = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .tag("encoder")
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let named = graph.named_parameters();
        // First Linear is tagged "encoder", second is untagged
        let encoder_params: Vec<&str> = named.iter()
            .filter(|(n, _)| n.starts_with("encoder/"))
            .map(|(n, _)| n.as_str())
            .collect();
        assert_eq!(encoder_params.len(), 2, "tagged node should have 2 params with 'encoder/' prefix");

        // Untagged node uses its node_id (like "linear_2")
        let untagged: Vec<&str> = named.iter()
            .filter(|(n, _)| !n.starts_with("encoder/"))
            .map(|(n, _)| n.as_str())
            .collect();
        assert_eq!(untagged.len(), 2, "untagged node should have 2 params");
        assert!(untagged[0].contains('/'), "should have prefix/name format: {}", untagged[0]);
    }

    // --- Structural hash tests ---

    #[test]
    fn test_structural_hash_deterministic() {
        let g1 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(ReLU::new())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let g2 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(ReLU::new())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        assert_eq!(g1.structural_hash(), g2.structural_hash());
    }

    #[test]
    fn test_structural_hash_differs() {
        let g1 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        // Different architecture: different hidden size
        let g2 = FlowBuilder::from(Linear::on_device(4, 16, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(16, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        assert_ne!(g1.structural_hash(), g2.structural_hash());
    }

    #[test]
    fn test_short_hash_length() {
        let g = FlowBuilder::from(Linear::on_device(2, 3, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        assert_eq!(g.structural_hash().len(), 64);
        assert_eq!(g.short_hash().len(), 8);
        assert!(g.structural_hash().starts_with(g.short_hash()));
    }

    #[test]
    fn test_label_default_none() {
        let g = FlowBuilder::from(Linear::on_device(2, 3, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();
        assert!(g.label().is_none());
    }

    #[test]
    fn test_label_set() {
        let g = FlowBuilder::from(Linear::on_device(2, 3, crate::tensor::test_device()).unwrap())
            .label("my-model")
            .build()
            .unwrap();
        assert_eq!(g.label(), Some("my-model"));
    }

    #[test]
    fn test_label_does_not_affect_hash() {
        let g1 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let g2 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .label("different-label")
            .build()
            .unwrap();

        assert_eq!(g1.structural_hash(), g2.structural_hash());
    }

    #[test]
    fn test_graph_save_load_checkpoint() {
        let g = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .tag("enc")
            .through(ReLU::new())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .tag("dec")
            .build()
            .unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_graph_ckpt.fdl");
        let path_str = path.to_str().unwrap();

        // Save
        g.save_checkpoint(path_str).unwrap();

        // Build identical architecture, load into it
        let g2 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .tag("enc")
            .through(ReLU::new())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .tag("dec")
            .build()
            .unwrap();

        let report = g2.load_checkpoint(path_str).unwrap();
        assert_eq!(report.loaded.len(), 4); // 2 Linear × (weight + bias)
        assert!(report.skipped.is_empty());
        assert!(report.missing.is_empty());

        // Verify weights match
        for ((n1, p1), (n2, p2)) in g.named_parameters().iter().zip(g2.named_parameters().iter()) {
            assert_eq!(n1, n2);
            assert_eq!(p1.variable.data().to_f32_vec().unwrap(),
                       p2.variable.data().to_f32_vec().unwrap());
        }

        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn test_graph_checkpoint_hash_mismatch() {
        let g1 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_graph_ckpt_mismatch.fdl");
        let path_str = path.to_str().unwrap();

        g1.save_checkpoint(path_str).unwrap();

        // Different architecture
        let g2 = FlowBuilder::from(Linear::on_device(4, 16, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(16, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let result = g2.load_checkpoint(path_str);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("architecture mismatch"));

        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn test_graph_checkpoint_gz() {
        let g = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_graph_ckpt.fdl.gz");
        let path_str = path.to_str().unwrap();

        g.save_checkpoint(path_str).unwrap();

        let g2 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let report = g2.load_checkpoint(path_str).unwrap();
        assert_eq!(report.loaded.len(), 4);

        std::fs::remove_file(path_str).ok();
    }
}
