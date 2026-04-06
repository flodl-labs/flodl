use std::cell::{Cell, OnceCell, RefCell};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;
use std::time::Instant;

use indexmap::IndexMap;
use hmac_sha256::Hash as Sha256;

use super::node::*;
use super::profile;
use super::LossContext;
use crate::autograd::Variable;
use crate::data::Batch;
use crate::nn::{Buffer, Module, Parameter};
use crate::tensor::{Result, Tensor, TensorError};

/// Pre-computed route from one node's output port to another node's input port.
/// Replaces HashMap-based edge routing in forward_impl for O(1) access.
#[derive(Clone)]
pub(crate) struct Route {
    from_port_idx: usize,
    to_node_idx: usize,
    to_port_idx: usize,
}

/// Pre-computed graph input → node input slot mapping.
pub(crate) struct InputRoute {
    node_idx: usize,
    port_idx: usize,
}

/// Forward-reference state buffer. Persists across `forward()` calls.
pub(crate) struct StateEntry {
    pub(crate) writer_ni: usize,
    pub(crate) value: Rc<RefCell<Option<Variable>>>,
}

/// An executable computation graph. Implements `Module` for composability.
///
/// Built via [`FlowBuilder`]. Supports parallel execution of independent nodes,
/// observation of tagged outputs, profiling, and DOT/SVG visualization.
///
/// ```ignore
/// let g = FlowBuilder::from(Linear::new(4, 8)?)
///     .through(GELU)
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
    pub(crate) nodes: Vec<Node>,
    pub(crate) node_index: HashMap<String, usize>,
    pub(crate) levels: Vec<Vec<usize>>,
    pub(crate) edges: Vec<Edge>,
    #[allow(dead_code)] // kept for DOT/debug introspection
    pub(crate) edges_from: HashMap<usize, Vec<usize>>,
    pub(crate) inputs: Vec<ExposedPort>,
    pub(crate) outputs: Vec<ExposedPort>,
    pub(crate) order: Vec<usize>,
    pub(crate) state: Vec<StateEntry>,
    // State writer lookup: node_idx → [(state_entry_idx, output_port_idx)]
    pub(crate) state_writers: HashMap<usize, Vec<(usize, usize)>>,
    // Tag groups: group name → suffixed tag names
    pub(crate) tag_groups: HashMap<String, Vec<String>>,
    // Observation: tag mapping (immutable after build)
    pub(crate) tag_names: HashMap<String, (usize, usize)>,           // tag name → (node_idx, port_idx)
    pub(crate) tag_capture: HashMap<usize, Vec<(String, usize)>>,     // node_idx → [(tag_name, port_idx)]
    // Observation: mutable state (RefCell/Cell for &self methods)
    pub(crate) tagged_outputs: RefCell<HashMap<String, Variable>>,
    pub(crate) batch_buffer: RefCell<HashMap<String, Vec<f64>>>,
    pub(crate) epoch_history: RefCell<HashMap<String, Vec<f64>>>,
    pub(crate) metric_order: RefCell<Vec<String>>,
    pub(crate) flush_count: Cell<usize>,
    // Profiling
    pub(crate) profiling: Cell<bool>,
    pub(crate) last_profile: RefCell<Option<profile::Profile>>,
    pub(crate) timing_buffer: RefCell<HashMap<String, Vec<f64>>>,
    pub(crate) timing_history: RefCell<HashMap<String, Vec<f64>>>,
    // Flush timestamps (seconds since first forward — for ETA in write_log)
    pub(crate) flush_times: RefCell<Vec<f64>>,
    pub(crate) training_start: Cell<f64>,
    // Step/epoch counters
    pub(crate) step_count: Cell<usize>,
    pub(crate) epoch_count: Cell<usize>,
    // Identity: label + structural hash
    pub(crate) label: Option<String>,
    pub(crate) structural_hash_cache: OnceCell<String>,
    // Graph tree: hierarchical composition
    pub(crate) children: HashMap<String, usize>,
    pub(crate) composed: Cell<bool>,
    pub(crate) internal_tags: HashSet<String>,
    // Pre-computed execution plan (built once, used every forward call)
    pub(crate) routes_from: Vec<Vec<Route>>,
    pub(crate) input_routes: Vec<InputRoute>,
    pub(crate) output_node_idx: usize,
    pub(crate) output_port_idx: usize,
    pub(crate) node_input_count: Vec<usize>,
    // Cached execution buffers (reused across forward calls, avoids re-allocation)
    pub(crate) exec_slots: RefCell<Vec<Vec<Option<Variable>>>>,
    // Distributed Data Parallel state (set by distribute(), None for single-GPU)
    pub(crate) distributed: RefCell<Option<crate::nn::ddp::DistributedState>>,
    // Optimizer for step() (works for both single-GPU and distributed)
    pub(crate) optimizer: RefCell<Option<Box<dyn crate::nn::Optimizer>>>,
    // DataLoader binding for resident DDP (set by set_data_loader(), None by default)
    pub(crate) data_binding: RefCell<Option<DataLoaderBinding>>,
    // Per-batch loss closure for El Che (set by set_loss_fn(), None = legacy gather path)
    #[allow(clippy::type_complexity)]
    pub(crate) loss_fn: RefCell<Option<Box<dyn Fn(&LossContext) -> Result<Variable>>>>,
}

/// Binding between a [`DataLoader`] and a [`Graph`] for integrated training.
///
/// Created by [`Graph::set_data_loader`]. Maps batch tensor names to
/// graph inputs and stores the loader reference.
pub(crate) struct DataLoaderBinding {
    /// The DataLoader (possibly upgraded to distributed mode).
    pub loader: crate::data::DataLoader,
    /// Name of the batch field used as the primary forward input (e.g., "image").
    pub forward_input: String,
    /// Mappings from batch field names to graph Input port names.
    /// Only populated for graphs with `.input()` ports that match batch names.
    #[allow(dead_code)]
    pub graph_inputs: Vec<(String, String)>, // (batch_name, graph_input_name)
    /// Names of batch fields that are targets (for loss), not consumed by forward.
    #[allow(dead_code)]
    pub target_names: Vec<String>,
    /// Maps graph input index → shard/batch tensor position.
    /// `shard_input_map[i]` is the index into `per_rank_shards[rank]` or
    /// `Batch` that provides `self.inputs[i]`.
    pub shard_input_map: Vec<usize>,
    /// Chunk ratios for distributed training (updated by auto-balancer).
    /// Stored here so the epoch iterator can read them without borrowing DistributedState.
    pub chunk_ratios: Vec<f64>,
    /// Batch field names (from loader) for reconstructing Batch objects in
    /// forward_distributed_el_che's per-batch backward path.
    pub batch_names: Vec<String>,
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
        mut internal_tags: HashSet<String>,
        verbose: bool,
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

        // Detect child subgraphs: labeled Graphs become tree children
        let mut children: HashMap<String, usize> = HashMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            if let Some(ref module) = node.module {
                if let Some(child_graph) = module.as_graph() {
                    if let Some(child_label) = child_graph.label() {
                        if child_label.contains('.') {
                            return Err(TensorError::new(&format!(
                                "child graph label {:?} contains a dot — \
                                 dots are reserved for path separators",
                                child_label
                            )));
                        }
                        if children.contains_key(child_label) {
                            return Err(TensorError::new(&format!(
                                "duplicate child graph label {:?} at the same tree level",
                                child_label
                            )));
                        }
                        // Validate: label doesn't shadow a tag on a different node
                        if let Some(&(tag_ni, _)) = tag_names_map.get(child_label) {
                            if tag_ni != idx {
                                return Err(TensorError::new(&format!(
                                    "child graph label {:?} collides with a tag \
                                     on a different node",
                                    child_label
                                )));
                            }
                        }
                        children.insert(child_label.to_string(), idx);
                        child_graph.composed.set(true);
                    }
                    // Unlabeled graphs: not registered, no tree features, no error
                }
            }
        }

        // Auto-internal inference: underscore-prefixed tags
        for name in tag_names_map.keys() {
            if name.starts_with('_') {
                internal_tags.insert(name.clone());
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

        // Pre-compute routing table: flat Vec lookups replace HashMap edge routing
        let n = nodes.len();
        let mut routes_from: Vec<Vec<Route>> = vec![Vec::new(); n];
        for edge in &edges {
            let from_ni = node_index[&edge.from_node];
            let to_ni = node_index[&edge.to_node];
            let from_port_idx = nodes[from_ni]
                .output_ports
                .iter()
                .position(|p| p == &edge.from_port)
                .unwrap_or(0);
            let to_port_idx = nodes[to_ni]
                .input_ports
                .iter()
                .position(|p| p == &edge.to_port)
                .unwrap_or(0);
            routes_from[from_ni].push(Route {
                from_port_idx,
                to_node_idx: to_ni,
                to_port_idx,
            });
        }

        // Pre-compute graph input → slot mapping
        let input_routes: Vec<InputRoute> = inputs
            .iter()
            .map(|ep| {
                let ni = node_index[&ep.node_id];
                let port_idx = nodes[ni]
                    .input_ports
                    .iter()
                    .position(|p| p == &ep.port)
                    .unwrap_or(0);
                InputRoute {
                    node_idx: ni,
                    port_idx,
                }
            })
            .collect();

        // Pre-compute output location
        let output_node_idx = node_index[&outputs[0].node_id];
        let output_port_idx = nodes[output_node_idx]
            .output_ports
            .iter()
            .position(|p| p == &outputs[0].port)
            .unwrap_or(0);

        // Pre-compute input port counts and allocate execution buffers
        let node_input_count: Vec<usize> = nodes.iter().map(|nd| nd.input_ports.len()).collect();
        let exec_slots = RefCell::new(
            node_input_count.iter().map(|&c| vec![None; c]).collect(),
        );

        let graph = Ok(Graph {
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
            metric_order: RefCell::new(Vec::new()),
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
            children,
            composed: Cell::new(false),
            internal_tags,
            routes_from,
            input_routes,
            output_node_idx,
            output_port_idx,
            node_input_count,
            exec_slots,
            distributed: RefCell::new(None),
            optimizer: RefCell::new(None),
            data_binding: RefCell::new(None),
            loss_fn: RefCell::new(None),
        });

        if verbose {
            if let Ok(ref g) = graph {
                eprintln!("{}", g.tree_summary());
            }
        }

        graph
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

        let has_tags = !self.tag_capture.is_empty();

        // Reuse cached execution buffers (Vec-indexed, no HashMap overhead)
        let mut slots = self.exec_slots.borrow_mut();

        // Clear previous values (drops old Variables, reuses allocations)
        for node_slots in slots.iter_mut() {
            for slot in node_slots.iter_mut() {
                *slot = None;
            }
        }

        // Clear tagged outputs
        if has_tags {
            self.tagged_outputs.borrow_mut().clear();
        }

        // Route graph inputs via pre-computed index mapping
        for (i, route) in self.input_routes.iter().enumerate() {
            slots[route.node_idx][route.port_idx] = Some(graph_inputs[i].clone());
        }

        // Will hold the output node's results until we can extract the final value
        let mut final_output: Option<Vec<Variable>> = None;

        // Execute levels sequentially
        for (level_idx, level) in self.levels.iter().enumerate() {
            let level_start = if is_profiling { Some(Instant::now()) } else { None };
            let mut level_sum_ns: u64 = 0;

            for &ni in level {
                let node = &self.nodes[ni];
                let input_count = self.node_input_count[ni];

                // Collect inputs from pre-indexed slots (no HashMap lookups)
                let inputs: Vec<Variable> = (0..input_count)
                    .map(|i| {
                        match slots[ni][i].as_ref() {
                            Some(v) => Ok(v.clone()),
                            None if i > 0 => {
                                // Zero fill for unconnected ref ports (forward refs)
                                let first = slots[ni][0].as_ref().ok_or_else(|| {
                                    TensorError::new(&format!(
                                        "node '{}': ref port {} has no data and primary input \
                                         is also missing — check that all inputs are connected",
                                        node.id, i
                                    ))
                                })?;
                                Ok(Variable::new(
                                    Tensor::zeros_like(&first.data())?,
                                    false,
                                ))
                            }
                            _ => Err(TensorError::new(&format!(
                                "node '{}': missing primary input (port {}) — check that all \
                                 inputs to this node are connected in the graph builder",
                                node.id, i
                            ))),
                        }
                    })
                    .collect::<Result<Vec<Variable>>>()?;

                // Release input slots early (frees Rc references)
                for slot in slots[ni].iter_mut() {
                    *slot = None;
                }

                // Execute node (with optional per-node timing)
                let node_start = if is_profiling { Some(Instant::now()) } else { None };
                let node_outputs = (node.run)(&inputs)?;
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

                // Route outputs via pre-computed routing table (no HashMap, no String ops)
                for route in &self.routes_from[ni] {
                    let value = if route.from_port_idx < node_outputs.len() {
                        Some(node_outputs[route.from_port_idx].clone())
                    } else {
                        None
                    };
                    slots[route.to_node_idx][route.to_port_idx] = value;
                }

                // Capture state: if this node is a state writer, store its output
                if let Some(writers) = self.state_writers.get(&ni) {
                    for &(si, port_idx) in writers {
                        if port_idx < node_outputs.len() {
                            *self.state[si].value.borrow_mut() =
                                Some(node_outputs[port_idx].clone());
                        }
                    }
                }

                // Capture tagged outputs for observation
                if has_tags {
                    if let Some(captures) = self.tag_capture.get(&ni) {
                        let mut tagged = self.tagged_outputs.borrow_mut();
                        for (tag_name, port_idx) in captures {
                            if *port_idx < node_outputs.len() {
                                tagged.insert(
                                    tag_name.clone(),
                                    node_outputs[*port_idx].clone(),
                                );
                            }
                        }
                    }
                }

                // Keep output node's results; all others drop here (early release)
                if ni == self.output_node_idx {
                    final_output = Some(node_outputs);
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

        // Drop the borrow before storing profile (which also borrows RefCells)
        drop(slots);

        // Store profile
        if is_profiling {
            *self.last_profile.borrow_mut() = Some(profile::Profile {
                total: forward_start.unwrap().elapsed(),
                levels: prof_levels,
                nodes: prof_nodes,
            });
        }

        // Extract graph output
        final_output
            .and_then(|o| o.into_iter().nth(self.output_port_idx))
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

        hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl Module for Graph {
    fn name(&self) -> &str { "graph" }

    fn as_graph(&self) -> Option<&Graph> { Some(self) }

    fn structural_hash(&self) -> Option<String> {
        Some(self.structural_hash().to_string())
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        if self.distributed.borrow().is_some() {
            // Check if presharded data is available from the DataLoader
            let has_shards = self.data_binding.borrow()
                .as_ref()
                .is_some_and(|b| b.loader.has_shards());

            if has_shards {
                self.forward_distributed_presharded()
            } else {
                self.forward_distributed_scatter(input)
            }
        } else {
            self.forward_impl(std::slice::from_ref(input))
        }
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

// ---------------------------------------------------------------------------
// Distributed Data Parallel + optimizer integration
// ---------------------------------------------------------------------------

impl Graph {
    /// Enable multi-GPU training. Detects usable CUDA devices, creates model
    /// replicas via the factory closure, and broadcasts parameters from rank 0.
    ///
    /// If only one usable GPU is found, this is a no-op (single-GPU mode).
    /// The factory receives a [`Device`] and must return a model with the same
    /// architecture as `self` (same parameter count and shapes).
    ///
    /// After this call, [`forward`](Module::forward) automatically scatters
    /// input across devices and gathers output. [`step`](Graph::step) handles
    /// AllReduce + optimizer step + zero_grad.
    ///
    /// For a one-liner that also sets the optimizer and training mode, see
    /// [`Ddp::auto()`](crate::nn::Ddp::auto).
    ///
    /// ```ignore
    /// model.distribute(|dev| build_model(dev))?;
    /// ```
    pub fn distribute<F, M>(&self, factory: F) -> Result<()>
    where
        F: Fn(crate::tensor::Device) -> Result<M>,
        M: crate::nn::Module + 'static,
    {
        use crate::nn::ddp::DistributedState;
        use crate::nn::nccl::NcclComms;

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
            calibration_steps: crate::nn::ddp::DEFAULT_CALIBRATION_STEPS,
            rebalance_interval: crate::nn::ddp::DEFAULT_REBALANCE_INTERVAL,
            el_che: None,
            last_el_che_counts: Vec::new(),
            last_el_che_sync: None,
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
    /// model.set_optimizer(|p| Adam::new(&p, 0.001));
    /// ```
    pub fn set_optimizer<F, O>(&self, factory: F)
    where
        F: Fn(Vec<crate::nn::Parameter>) -> O,
        O: crate::nn::Optimizer + 'static,
    {
        let mut dist = self.distributed.borrow_mut();
        if let Some(ref mut state) = *dist {
            // Distributed: one optimizer per replica
            let mut optimizers: Vec<Box<dyn crate::nn::Optimizer>> = Vec::new();

            // Rank 0 optimizer (uses self's parameters)
            let rank0_opt = factory(self.parameters());
            optimizers.push(Box::new(rank0_opt));

            // Replicas
            for replica in &state.replicas {
                let opt = factory(replica.parameters());
                optimizers.push(Box::new(opt));
            }

            state.optimizers = optimizers;
        } else {
            // Single GPU: one optimizer
            let opt = factory(self.parameters());
            *self.optimizer.borrow_mut() = Some(Box::new(opt));
        }
    }

    /// Perform one training step.
    ///
    /// When distributed: AllReduce gradients (weighted if auto-balanced),
    /// sync buffers, step all optimizers, zero grad, update auto-balancer.
    /// When single-GPU: step optimizer, zero grad.
    ///
    /// This is the single call that replaces `opt.step(); opt.zero_grad();`
    /// and makes multi-GPU training transparent.
    pub fn step(&self) -> Result<()> {
        let mut dist = self.distributed.borrow_mut();
        if let Some(ref mut state) = *dist {
            if state.el_che.is_some() {
                // El Che path: weighted AllReduce by actual per-device batch counts.
                // Gradients were accumulated in forward_distributed_el_che().
                use crate::nn::cuda_event::CudaEvent;
                use crate::nn::nccl::ReduceOp;

                let counts = state.last_el_che_counts.clone();
                let total: usize = counts.iter().sum();

                if total > 0 {
                    // Compute wall time since last sync (= compute time for this cadence step)
                    let compute_ms = state.last_el_che_sync
                        .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);

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

                    let updated_counts = if let Some(ref mut el_che) = state.el_che {
                        if !wall_ms.is_empty() {
                            el_che.report_timing(&wall_ms, sync_ms);
                        }
                        Some(el_che.batch_counts().to_vec())
                    } else {
                        None
                    };

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
                if state.is_balanced() {
                    state.all_reduce_gradients()?;
                } else {
                    let batch_size: i64 = state.last_shard_sizes.iter().sum();
                    state.weighted_all_reduce_gradients(batch_size)?;
                }
                state.sync_buffers()?;

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
            let mut opt = self.optimizer.borrow_mut();
            if let Some(ref mut optimizer) = *opt {
                optimizer.step()?;
                optimizer.zero_grad();
            }
        }
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
    /// Called by [`Ddp::auto_with`] after [`distribute`](Graph::distribute).
    /// No-op if not in distributed mode.
    pub(crate) fn configure_el_che(&self, config: &crate::nn::ddp::DdpConfig) {
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

        // Build shard_input_map: graph input index → loader tensor position.
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
    fn forward_distributed_scatter(&self, input: &Variable) -> Result<Variable> {
        use crate::nn::cuda_event::{CudaEvent, CudaEventFlags};
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
    fn forward_distributed_presharded(&self) -> Result<Variable> {
        use crate::nn::cuda_event::{CudaEvent, CudaEventFlags};
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
    /// Each device processes its batch_counts[rank] batches independently.
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
        use crate::nn::cuda_event::CudaEventFlags;
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
        use crate::nn::cuda_event::CudaEventFlags;
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
        flags: crate::nn::cuda_event::CudaEventFlags,
    ) -> Result<Vec<crate::nn::cuda_event::CudaEvent>> {
        use crate::nn::cuda_event::CudaEvent;
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
        starts: Vec<crate::nn::cuda_event::CudaEvent>,
        ends: Vec<crate::nn::cuda_event::CudaEvent>,
    ) -> Vec<(crate::nn::cuda_event::CudaEvent, crate::nn::cuda_event::CudaEvent)> {
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
        timing: Vec<(crate::nn::cuda_event::CudaEvent, crate::nn::cuda_event::CudaEvent)>,
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

// ---------------------------------------------------------------------------
// GraphEpochIterator
// ---------------------------------------------------------------------------

/// Iterator over training batches, returned by [`Graph::epoch`].
///
/// Wraps either a single-GPU `EpochIterator` or a distributed
/// `DistributedEpochIterator`. Yields `Result<Batch>`.
pub enum GraphEpochIterator<'a> {
    /// Distributed mode: per-device backends, presharded data.
    Distributed(&'a Graph, usize),
    /// Single GPU: delegates to DataLoader's epoch iterator.
    Single(&'a Graph, usize),
}

/// Internal state once iteration starts (lazily initialized on first next()).
enum GraphEpochState<'a> {
    DistributedActive(crate::data::DistributedEpochIterator<'a>),
    SingleActive(crate::data::EpochIterator<'a>),
    Pending,
}

/// Active graph epoch iterator (initialized from GraphEpochIterator on first call).
pub struct ActiveGraphEpochIterator<'a> {
    state: GraphEpochState<'a>,
    #[allow(dead_code)]
    graph: &'a Graph,
}

impl<'a> GraphEpochIterator<'a> {
    /// Activate the iterator (must be called to start iteration).
    /// This resolves the borrow on the DataLoader binding.
    pub fn activate(self) -> ActiveGraphEpochIterator<'a> {
        match self {
            GraphEpochIterator::Distributed(graph, epoch) => {
                // Get chunk_ratios and create the distributed iterator
                let binding = graph.data_binding.borrow();
                let binding = binding.as_ref().unwrap();
                let chunk_ratios = &binding.chunk_ratios as *const Vec<f64>;

                // Safety: chunk_ratios lives in the DataLoaderBinding which is
                // behind a RefCell in the Graph. The Graph outlives the iterator.
                // We only read chunk_ratios, and they're only mutated between epochs
                // (in step()), not during iteration.
                let ratios_ref: &'a [f64] = unsafe { &*chunk_ratios };

                if let crate::data::loader::LoaderInner::Distributed(ref dist_loader) = binding.loader.inner {
                    let iter = crate::data::DistributedEpochIterator::new(
                        // Safety: same reasoning -- dist_loader lives in the DataLoaderBinding
                        unsafe { &*(dist_loader as *const _) },
                        epoch,
                        ratios_ref,
                    );
                    ActiveGraphEpochIterator {
                        state: GraphEpochState::DistributedActive(iter),
                        graph,
                    }
                } else {
                    ActiveGraphEpochIterator {
                        state: GraphEpochState::Pending,
                        graph,
                    }
                }
            }
            GraphEpochIterator::Single(graph, epoch) => {
                // For single GPU, we need a mutable borrow to call epoch()
                // on the inner DataLoader.
                let mut binding = graph.data_binding.borrow_mut();
                let binding = binding.as_mut().unwrap();
                let loader_ptr = &mut binding.loader as *mut crate::data::DataLoader;

                // Safety: the DataLoader lives in the Graph's DataLoaderBinding.
                // We create an EpochIterator that borrows from it. The Graph outlives
                // the iterator. No concurrent mutation occurs during iteration.
                let iter = unsafe { (*loader_ptr).epoch(epoch) };
                ActiveGraphEpochIterator {
                    state: GraphEpochState::SingleActive(iter),
                    graph,
                }
            }
        }
    }
}

impl Iterator for ActiveGraphEpochIterator<'_> {
    type Item = Result<crate::data::Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.state {
            GraphEpochState::DistributedActive(iter) => iter.next(),
            GraphEpochState::SingleActive(iter) => iter.next(),
            GraphEpochState::Pending => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.state {
            GraphEpochState::DistributedActive(iter) => iter.size_hint(),
            GraphEpochState::SingleActive(iter) => iter.size_hint(),
            GraphEpochState::Pending => (0, Some(0)),
        }
    }
}

impl ExactSizeIterator for ActiveGraphEpochIterator<'_> {}

impl Drop for ActiveGraphEpochIterator<'_> {
    fn drop(&mut self) {
        // El Che epoch-end flush: if forward_distributed_el_che() was called
        // but step() hasn't been called yet, gradients are accumulated and
        // un-synced. Force a step() to prevent silent gradient loss.
        if self.graph.has_el_che() {
            let needs_flush = self.graph.distributed.borrow()
                .as_ref()
                .is_some_and(|d| d.last_timing.is_some());

            if needs_flush {
                let _ = self.graph.step();
            }
        }
    }
}

/// Current time as seconds since epoch (monotonic approximation for ETA).
pub(crate) fn instant_secs() -> f64 {
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
    use crate::graph::{
        FlowBuilder, Reduce, MergeOp,
        SoftmaxRouter, SigmoidRouter, FixedSelector, ArgmaxSelector,
        ThresholdHalt, LearnedHalt,
    };
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

    // --- collect_with reduction tests ---

    #[test]
    fn test_collect_with_sum_reduction() {
        // Non-scalar tagged output reduced via Sum
        let graph = FlowBuilder::from(Identity)
            .tag("features")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect_with(&["features"], Reduce::Sum).unwrap();

        let collected = graph.collected("features");
        assert_eq!(collected.len(), 1);
        assert!((collected[0] - 6.0).abs() < 1e-5, "sum([1,2,3]) = 6, got {}", collected[0]);
    }

    #[test]
    fn test_collect_with_mean_reduction() {
        let graph = FlowBuilder::from(Identity)
            .tag("out")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[2.0, 4.0, 6.0], &[1, 3]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect_with(&["out"], Reduce::Mean).unwrap();

        let collected = graph.collected("out");
        assert!((collected[0] - 4.0).abs() < 1e-5, "mean([2,4,6]) = 4, got {}", collected[0]);
    }

    #[test]
    fn test_collect_with_max_reduction() {
        let graph = FlowBuilder::from(Identity)
            .tag("out")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 5.0, 3.0], &[1, 3]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect_with(&["out"], Reduce::Max).unwrap();

        let collected = graph.collected("out");
        assert!((collected[0] - 5.0).abs() < 1e-5, "max([1,5,3]) = 5, got {}", collected[0]);
    }

    #[test]
    fn test_collect_with_min_reduction() {
        let graph = FlowBuilder::from(Identity)
            .tag("out")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[-2.0, 0.0, 3.0], &[1, 3]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect_with(&["out"], Reduce::Min).unwrap();

        let collected = graph.collected("out");
        assert!((collected[0] - (-2.0)).abs() < 1e-5, "min([-2,0,3]) = -2, got {}", collected[0]);
    }

    #[test]
    fn test_collect_with_norm_reduction() {
        let graph = FlowBuilder::from(Identity)
            .tag("out")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[3.0, 4.0], &[1, 2]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect_with(&["out"], Reduce::Norm).unwrap();

        let collected = graph.collected("out");
        // L2 norm of [3, 4] = 5
        assert!((collected[0] - 5.0).abs() < 1e-4, "norm([3,4]) = 5, got {}", collected[0]);
    }

    #[test]
    fn test_collect_rejects_non_scalar() {
        // Plain collect() should reject non-scalar outputs
        let graph = FlowBuilder::from(Identity)
            .tag("out")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let _ = graph.forward(&x).unwrap();
        assert!(graph.collect(&["out"]).is_err());
    }

    #[test]
    fn test_collect_with_scalar_passthrough() {
        // collect_with on already-scalar output should work without reduction
        let graph = FlowBuilder::from(ScalarSum)
            .tag("loss")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[3.0, 7.0], &[1, 2]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect_with(&["loss"], Reduce::Max).unwrap();

        let collected = graph.collected("loss");
        // ScalarSum yields 10.0 (scalar), so it should pass through directly
        assert!((collected[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_collect_with_flush_trend_pipeline() {
        // Full pipeline: non-scalar → reduce → flush → trend
        let graph = FlowBuilder::from(Identity)
            .tag("h")
            .build()
            .unwrap();

        // Epoch 1: two batches with decreasing norms
        let x1 = Variable::new(from_f32(&[3.0, 4.0], &[1, 2]), false);
        let _ = graph.forward(&x1).unwrap();
        graph.collect_with(&["h"], Reduce::Norm).unwrap();

        let x2 = Variable::new(from_f32(&[1.0, 0.0], &[1, 2]), false);
        let _ = graph.forward(&x2).unwrap();
        graph.collect_with(&["h"], Reduce::Norm).unwrap();

        graph.flush(&["h"]);

        // Epoch 2
        let x3 = Variable::new(from_f32(&[0.5, 0.5], &[1, 2]), false);
        let _ = graph.forward(&x3).unwrap();
        graph.collect_with(&["h"], Reduce::Norm).unwrap();
        graph.flush(&["h"]);

        let trend = graph.trend("h");
        assert_eq!(trend.len(), 2);
        // Epoch 1 mean: (5.0 + 1.0) / 2 = 3.0
        assert!((trend.values()[0] - 3.0).abs() < 1e-4);
        assert!(trend.improving(0)); // norms should be decreasing
    }

    // --- Map.over and Map.slices tests ---

    #[test]
    fn test_map_over_tag() {
        // Tag a tensor, then map over it from a different stream position
        let graph = FlowBuilder::from(Identity)
            .tag("features")
            .through(Doubler)        // stream is now 2x
            .map(Doubler)
            .over("features")        // map over original (1x), not current stream (2x)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // .over("features") maps Doubler over the tagged value (original x)
        // Doubler: x + x = 2x, applied element-wise along dim 0
        assert_eq!(y.shape(), vec![2, 2]);
        assert!((data[0] - 2.0).abs() < 1e-5);  // 1.0 * 2
        assert!((data[1] - 4.0).abs() < 1e-5);  // 2.0 * 2
        assert!((data[2] - 6.0).abs() < 1e-5);  // 3.0 * 2
        assert!((data[3] - 8.0).abs() < 1e-5);  // 4.0 * 2
    }

    #[test]
    fn test_map_over_unknown_tag_error() {
        let result = FlowBuilder::from(Identity)
            .map(Doubler)
            .over("nonexistent")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_map_slices() {
        // Input [2, 4], slices(2): decompose → [4, 2], map Doubler, recompose → [2, 4]
        let graph = FlowBuilder::from(Identity)
            .map(Doubler)
            .slices(2)
            .build()
            .unwrap();

        let x = Variable::new(
            from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]),
            false,
        );
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        // Each element doubled
        assert_eq!(y.shape(), vec![2, 4]);
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[7] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_map_slices_batched() {
        // Same as above but with batched fast path
        let graph = FlowBuilder::from(Identity)
            .map(Doubler)
            .batched()
            .slices(2)
            .build()
            .unwrap();

        let x = Variable::new(
            from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]),
            false,
        );
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert_eq!(y.shape(), vec![2, 4]);
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[7] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_map_slices_gradient() {
        // Input [2, 4] → slices(2) decomposes to [4, 2] → Linear(2, 3) → [4, 3] → recompose [2, 6]
        let graph = FlowBuilder::from(Identity)
            .map(Linear::on_device(2, 3, crate::tensor::test_device()).unwrap())
            .slices(2)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]), true);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 6]); // 3 * 2 slices = 6
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_map_slices_not_divisible_error() {
        let graph = FlowBuilder::from(Identity)
            .map(Doubler)
            .slices(3)
            .build()
            .unwrap();

        // [2, 4] with slices(3) — 4 not divisible by 3
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]), false);
        assert!(graph.forward(&x).is_err());
    }
}

