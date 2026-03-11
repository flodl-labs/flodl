use std::collections::HashMap;
use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::{Module, NamedInputModule};
use crate::tensor::TensorError;

use super::loop_node::LoopBuilder;
use super::node::*;
use super::MergeOp;

pub struct FlowBuilder {
    pub(super) nodes: HashMap<String, Node>,
    pub(super) edges: Vec<Edge>,
    pub(super) inputs: Vec<ExposedPort>,
    pub(super) current: Vec<NodeRef>,
    pub(super) taps: HashMap<String, NodeRef>,
    pub(super) on_target: Option<NodeRef>,
    pub(super) counter: usize,
    pub(super) err: Option<String>,
    /// Pending forward references: Using("x") called before Tag("x").
    pub(super) pending: HashMap<String, Vec<PendingUsing>>,
    /// Resolved forward references ready for Graph::build.
    pub(super) forward_refs: Vec<ForwardRefSpec>,
}

impl FlowBuilder {
    /// Start a new graph flow with an entry module.
    pub fn from(module: impl Module + 'static) -> Self {
        let mut fb = FlowBuilder {
            nodes: HashMap::new(),
            edges: Vec::new(),
            inputs: Vec::new(),
            current: Vec::new(),
            taps: HashMap::new(),
            on_target: None,
            counter: 0,
            err: None,
            pending: HashMap::new(),
            forward_refs: Vec::new(),
        };

        let node_ref = fb.add_module(module);

        fb.inputs.push(ExposedPort {
            name: DEFAULT_INPUT.to_string(),
            node_id: node_ref.node_id.clone(),
            port: DEFAULT_INPUT.to_string(),
        });

        fb.on_target = Some(node_ref.clone());
        fb.current = vec![node_ref];
        fb
    }

    /// Chain a module sequentially: stream → module → stream.
    pub fn through(mut self, module: impl Module + 'static) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() != 1 {
            self.err = Some("through requires single stream".into());
            return self;
        }

        let prev = self.current[0].clone();
        let node_ref = self.add_module(module);

        self.edges.push(Edge {
            from_node: prev.node_id,
            from_port: prev.port,
            to_node: node_ref.node_id.clone(),
            to_port: DEFAULT_INPUT.into(),
        });

        self.on_target = Some(node_ref.clone());
        self.current = vec![node_ref];
        self
    }

    /// Chain a ref-capable module (implements NamedInputModule).
    /// Can receive additional inputs via [`using`](Self::using).
    pub fn through_ref(mut self, module: impl NamedInputModule + 'static) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() != 1 {
            self.err = Some("through_ref requires single stream".into());
            return self;
        }

        let prev = self.current[0].clone();
        let node_ref = self.add_named_module(module);

        self.edges.push(Edge {
            from_node: prev.node_id,
            from_port: prev.port,
            to_node: node_ref.node_id.clone(),
            to_port: DEFAULT_INPUT.into(),
        });

        self.on_target = Some(node_ref.clone());
        self.current = vec![node_ref];
        self
    }

    /// Residual connection: output = stream + module(stream).
    pub fn also(mut self, module: impl Module + 'static) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() != 1 {
            self.err = Some("also requires single stream".into());
            return self;
        }

        let prev = self.current[0].clone();
        let module_ref = self.add_module(module);
        let add_ref = self.add_add_node(2);

        // prev → module
        self.edges.push(Edge {
            from_node: prev.node_id.clone(),
            from_port: prev.port.clone(),
            to_node: module_ref.node_id.clone(),
            to_port: DEFAULT_INPUT.into(),
        });

        // prev → add input_0 (skip connection)
        self.edges.push(Edge {
            from_node: prev.node_id,
            from_port: prev.port,
            to_node: add_ref.node_id.clone(),
            to_port: "input_0".into(),
        });

        // module → add input_1
        self.edges.push(Edge {
            from_node: module_ref.node_id.clone(),
            from_port: module_ref.port.clone(),
            to_node: add_ref.node_id.clone(),
            to_port: "input_1".into(),
        });

        self.on_target = Some(module_ref);
        self.current = vec![add_ref];
        self
    }

    /// Name the current stream position for later reference via [`using`](Self::using).
    pub fn tag(mut self, name: &str) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() != 1 {
            self.err = Some("tag requires single stream".into());
            return self;
        }
        if self.taps.contains_key(name) {
            self.err = Some(format!("duplicate tag: {}", name));
            return self;
        }

        let cur = self.current[0].clone();
        self.taps.insert(name.to_string(), cur.clone());

        // Resolve any pending forward references to this tag
        if let Some(pending_list) = self.pending.remove(name) {
            for p in pending_list {
                self.forward_refs.push(ForwardRefSpec {
                    name: name.to_string(),
                    reader_id: p.reader_id,
                    writer_id: cur.node_id.clone(),
                    writer_port: cur.port.clone(),
                });
            }
        }

        self
    }

    /// Wire tagged outputs as extra inputs to the preceding module.
    /// The target module must have been added via [`through_ref`](Self::through_ref).
    /// Tags must already exist (backward refs only).
    pub fn using(mut self, refs: &[&str]) -> Self {
        if self.err.is_some() {
            return self;
        }
        if refs.is_empty() {
            return self;
        }

        // Determine target(s)
        let targets = if let Some(ref target) = self.on_target {
            vec![target.clone()]
        } else if self.current.len() > 1 {
            self.current.clone()
        } else {
            self.err = Some(
                "using requires a preceding through_ref, split, or merge".into(),
            );
            return self;
        };

        for target in &targets {
            if let Err(e) = self.wire_using(target, refs) {
                self.err = Some(e);
                return self;
            }
        }
        self
    }

    /// Fork the stream into parallel branches, one module per branch.
    pub fn split(mut self, modules: Vec<Box<dyn Module>>) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() != 1 {
            self.err = Some("split requires single stream".into());
            return self;
        }
        if modules.len() < 2 {
            self.err = Some("split requires at least 2 branches".into());
            return self;
        }

        let prev = self.current[0].clone();
        let mut branches = Vec::new();

        for module in modules {
            let node_ref = self.add_boxed_module(module);
            self.edges.push(Edge {
                from_node: prev.node_id.clone(),
                from_port: prev.port.clone(),
                to_node: node_ref.node_id.clone(),
                to_port: DEFAULT_INPUT.into(),
            });
            branches.push(node_ref);
        }

        self.on_target = None;
        self.current = branches;
        self
    }

    /// Recombine split branches using a merge operation.
    pub fn merge(mut self, op: MergeOp) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() < 2 {
            self.err = Some("merge requires multiple streams (after split)".into());
            return self;
        }

        let branches = self.current.clone();
        let n = branches.len();
        let merge_ref = self.add_merge_node(n, op);

        for (i, branch) in branches.iter().enumerate() {
            self.edges.push(Edge {
                from_node: branch.node_id.clone(),
                from_port: branch.port.clone(),
                to_node: merge_ref.node_id.clone(),
                to_port: format!("input_{}", i),
            });
        }

        self.on_target = Some(merge_ref.clone());
        self.current = vec![merge_ref];
        self
    }

    /// Start a loop construct. Chain with `.for_n(n)`, `.while_cond(cond, max)`,
    /// or `.until_cond(cond, max)` to finalize.
    pub fn loop_body(self, body: impl Module + 'static) -> LoopBuilder {
        LoopBuilder::new(self, Box::new(body))
    }

    /// Hard routing: router selects one branch, others are skipped.
    /// Router must output a scalar 0-based branch index.
    pub fn switch(
        self,
        router: impl Module + 'static,
        branches: Vec<Box<dyn Module>>,
    ) -> Self {
        super::switch::wire_switch(self, Box::new(router), branches)
    }

    /// Soft routing (mixture of experts): all experts execute, outputs
    /// combined via learned router weights. Router must output shape
    /// `[..., n_experts]`.
    pub fn gate(
        self,
        router: impl Module + 'static,
        experts: Vec<Box<dyn Module>>,
    ) -> Self {
        super::gate::wire_gate(self, Box::new(router), experts)
    }

    /// Start a Map construct. Chain with `.each()` or `.batched().each()`.
    pub fn map(self, body: impl Module + 'static) -> super::map::MapBuilder {
        super::map::MapBuilder::new(self, Box::new(body))
    }

    /// Finalize the flow into an executable Graph.
    pub fn build(self) -> crate::tensor::Result<super::Graph> {
        if let Some(err) = self.err {
            return Err(TensorError::new(&err));
        }
        // Check for unresolved forward refs
        if !self.pending.is_empty() {
            let names: Vec<&String> = self.pending.keys().collect();
            return Err(TensorError::new(&format!(
                "unresolved forward refs: {:?}",
                names
            )));
        }
        if self.current.len() != 1 {
            return Err(TensorError::new(
                "open streams: call merge before build",
            ));
        }

        let output = ExposedPort {
            name: DEFAULT_OUTPUT.to_string(),
            node_id: self.current[0].node_id.clone(),
            port: self.current[0].port.clone(),
        };

        super::Graph::build(
            self.nodes,
            self.edges,
            self.inputs,
            vec![output],
            self.taps,
            self.forward_refs,
        )
    }

    // --- Internal helpers ---

    pub(super) fn next_id(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{}_{}", prefix, self.counter)
    }

    fn add_module(&mut self, module: impl Module + 'static) -> NodeRef {
        self.add_boxed_module(Box::new(module))
    }

    fn add_boxed_module(&mut self, module: Box<dyn Module>) -> NodeRef {
        let id = self.next_id("module");
        let rc: Rc<dyn Module> = Rc::from(module);
        let run = wrap_module(rc.clone());

        self.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                input_ports: vec![DEFAULT_INPUT.into()],
                output_ports: vec![DEFAULT_OUTPUT.into()],
                run,
                module: Some(rc),
                ref_forward: None,
            },
        );

        NodeRef {
            node_id: id,
            port: DEFAULT_OUTPUT.into(),
        }
    }

    fn add_named_module(&mut self, module: impl NamedInputModule + 'static) -> NodeRef {
        let id = self.next_id("module");
        let rc = Rc::new(module);
        let module_dyn: Rc<dyn Module> = rc.clone();
        let run = wrap_module(module_dyn.clone());

        // Capture forward_named capability
        let named_clone = rc.clone();
        let ref_forward: RefForwardFn = Rc::new(move |input, refs| {
            named_clone.forward_named(input, refs)
        });

        self.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                input_ports: vec![DEFAULT_INPUT.into()],
                output_ports: vec![DEFAULT_OUTPUT.into()],
                run,
                module: Some(module_dyn),
                ref_forward: Some(ref_forward),
            },
        );

        NodeRef {
            node_id: id,
            port: DEFAULT_OUTPUT.into(),
        }
    }

    fn wire_using(
        &mut self,
        target: &NodeRef,
        refs: &[&str],
    ) -> std::result::Result<(), String> {
        // Check the target node supports refs
        {
            let node = self.nodes.get(&target.node_id).ok_or_else(|| {
                format!("unknown target node: {}", target.node_id)
            })?;
            if node.module.is_some() && node.ref_forward.is_none() {
                return Err(format!(
                    "module at '{}' does not support Using refs (use through_ref)",
                    target.node_id
                ));
            }
        }

        // Add ref ports and edges
        for ref_name in refs {
            let port_name = format!("ref_{}", ref_name);

            if let Some(tap) = self.taps.get(*ref_name).cloned() {
                // Backward reference: tag already exists, wire directly
                let node = self.nodes.get_mut(&target.node_id).unwrap();
                node.input_ports.push(port_name.clone());

                self.edges.push(Edge {
                    from_node: tap.node_id.clone(),
                    from_port: tap.port.clone(),
                    to_node: target.node_id.clone(),
                    to_port: port_name,
                });
            } else {
                // Forward reference: tag not set yet, create state reader node
                let reader_ref = self.add_state_read_node(ref_name);

                let node = self.nodes.get_mut(&target.node_id).unwrap();
                node.input_ports.push(port_name.clone());

                self.edges.push(Edge {
                    from_node: reader_ref.node_id.clone(),
                    from_port: reader_ref.port.clone(),
                    to_node: target.node_id.clone(),
                    to_port: port_name,
                });

                self.pending
                    .entry(ref_name.to_string())
                    .or_default()
                    .push(PendingUsing {
                        reader_id: reader_ref.node_id,
                    });
            }
        }

        // Rebuild run function with updated ports
        let node = self.nodes.get_mut(&target.node_id).unwrap();
        if let (Some(module), Some(ref_forward)) =
            (node.module.clone(), node.ref_forward.clone())
        {
            let ports = node.input_ports.clone();
            node.run = wrap_ref_module(module, ref_forward, ports);
        }

        Ok(())
    }

    fn add_add_node(&mut self, n: usize) -> NodeRef {
        let id = self.next_id("add");
        let input_ports: Vec<String> = (0..n).map(|i| format!("input_{}", i)).collect();

        self.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                input_ports,
                output_ports: vec![DEFAULT_OUTPUT.into()],
                run: Box::new(move |inputs: &[Variable]| {
                    let mut result = inputs[0].clone();
                    for i in 1..n {
                        result = result.add(&inputs[i])?;
                    }
                    Ok(vec![result])
                }),
                module: None,
                ref_forward: None,
            },
        );

        NodeRef {
            node_id: id,
            port: DEFAULT_OUTPUT.into(),
        }
    }

    fn add_state_read_node(&mut self, ref_name: &str) -> NodeRef {
        let id = self.next_id(&format!("state_read_{}", ref_name));

        // Placeholder run — will be replaced by Graph::build with the actual state buffer
        self.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                input_ports: vec![],
                output_ports: vec![DEFAULT_OUTPUT.into()],
                run: Box::new(|_| {
                    Err(crate::tensor::TensorError::new(
                        "state_read not wired (build bug)",
                    ))
                }),
                module: None,
                ref_forward: None,
            },
        );

        NodeRef {
            node_id: id,
            port: DEFAULT_OUTPUT.into(),
        }
    }

    fn add_merge_node(&mut self, n: usize, op: MergeOp) -> NodeRef {
        let id = self.next_id("merge");
        let input_ports: Vec<String> = (0..n).map(|i| format!("input_{}", i)).collect();

        let run: NodeFn = match op {
            MergeOp::Add => Box::new(move |inputs: &[Variable]| {
                let mut result = inputs[0].clone();
                for i in 1..n {
                    result = result.add(&inputs[i])?;
                }
                Ok(vec![result])
            }),
            MergeOp::Mean => Box::new(move |inputs: &[Variable]| {
                let mut result = inputs[0].clone();
                for i in 1..n {
                    result = result.add(&inputs[i])?;
                }
                result.mul_scalar(1.0 / n as f64).map(|v| vec![v])
            }),
        };

        self.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                input_ports,
                output_ports: vec![DEFAULT_OUTPUT.into()],
                run,
                module: None,
                ref_forward: None,
            },
        );

        NodeRef {
            node_id: id,
            port: DEFAULT_OUTPUT.into(),
        }
    }
}
