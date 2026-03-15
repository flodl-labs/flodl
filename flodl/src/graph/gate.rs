use std::collections::HashMap;
use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::Result;

use super::node::*;
use super::FlowBuilder;

/// Wire a Gate (soft routing / mixture of experts) into the graph.
/// All experts execute, outputs combined via learned router weights.
pub(super) fn wire_gate(
    mut fb: FlowBuilder,
    router: Box<dyn Module>,
    experts: Vec<Box<dyn Module>>,
) -> FlowBuilder {
    if fb.err.is_some() {
        return fb;
    }
    if fb.current.len() != 1 {
        fb.err = Some("gate requires single stream".into());
        return fb;
    }
    if experts.len() < 2 {
        fb.err = Some("gate requires at least 2 experts".into());
        return fb;
    }

    let cur = fb.current[0].clone();
    let id = fb.next_id("gate");

    let router: Rc<dyn Module> = Rc::from(router);
    let expert_modules: Vec<Rc<dyn Module>> = experts
        .into_iter()
        .map(|e| Rc::from(e) as Rc<dyn Module>)
        .collect();

    let composite: Rc<dyn Module> = Rc::new(GateComposite {
        router: router.clone(),
        experts: expert_modules.clone(),
    });

    let run = make_gate_func(router.clone(), expert_modules.clone());

    // Only enable ref_forward if the router actually implements NamedInputModule
    let ref_forward = if router.as_named_input().is_some() {
        Some(make_gate_ref_forward(router, expert_modules))
    } else {
        None
    };

    fb.nodes.insert(
        id.clone(),
        Node {
            id: id.clone(),
            input_ports: vec![DEFAULT_INPUT.into()],
            output_ports: vec![DEFAULT_OUTPUT.into()],
            run,
            module: Some(composite),
            ref_forward,
            trace_buf: None,
            loop_ports: None,
        },
    );

    fb.edges.push(Edge {
        from_node: cur.node_id,
        from_port: cur.port,
        to_node: id.clone(),
        to_port: DEFAULT_INPUT.into(),
    });

    let node_ref = NodeRef {
        node_id: id,
        port: DEFAULT_OUTPUT.into(),
    };
    fb.current = vec![node_ref.clone()];
    fb.on_target = Some(node_ref);
    fb
}

fn gate_route(
    router: &Rc<dyn Module>,
    stream: &Variable,
    refs: &HashMap<String, Variable>,
    experts: &[Rc<dyn Module>],
    n_experts: usize,
) -> Result<Variable> {
    // Try NamedInputModule if refs are available
    let weights = if !refs.is_empty() {
        if let Some(named) = router.as_named_input() {
            named.forward_named(stream, refs)?
        } else {
            router.forward(stream)?
        }
    } else {
        router.forward(stream)?
    };

    // Run all experts and combine with weights
    let mut result: Option<Variable> = None;
    for (i, expert) in experts.iter().enumerate() {
        let expert_out = expert.forward(stream)?;
        let last_dim = (weights.shape().len() - 1) as i32;
        let w_i = weights.narrow(last_dim, i as i64, 1)?;
        let weighted = expert_out.mul(&w_i)?;

        result = Some(match result {
            None => weighted,
            Some(acc) => acc.add(&weighted)?,
        });
    }

    Ok(result.unwrap_or_else(|| {
        panic!("gate: no experts (n={})", n_experts)
    }))
}

fn make_gate_func(
    router: Rc<dyn Module>,
    experts: Vec<Rc<dyn Module>>,
) -> NodeFn {
    let n_experts = experts.len();
    Box::new(move |inputs: &[Variable]| {
        let empty = HashMap::new();
        let output = gate_route(&router, &inputs[0], &empty, &experts, n_experts)?;
        Ok(vec![output])
    })
}

fn make_gate_ref_forward(
    router: Rc<dyn Module>,
    experts: Vec<Rc<dyn Module>>,
) -> RefForwardFn {
    let n_experts = experts.len();
    Rc::new(move |stream: &Variable, refs: &HashMap<String, Variable>| {
        gate_route(&router, stream, refs, &experts, n_experts)
    })
}

/// Bundles router + experts for parameter collection.
struct GateComposite {
    router: Rc<dyn Module>,
    experts: Vec<Rc<dyn Module>>,
}

impl Module for GateComposite {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.experts[0].forward(input)
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        let mut subs = vec![self.router.clone()];
        subs.extend(self.experts.iter().cloned());
        subs
    }

    fn move_to_device(&self, device: crate::tensor::Device) {
        self.router.move_to_device(device);
        for e in &self.experts {
            e.move_to_device(device);
        }
    }

    fn set_training(&self, training: bool) {
        self.router.set_training(training);
        for e in &self.experts {
            e.set_training(training);
        }
    }
}
