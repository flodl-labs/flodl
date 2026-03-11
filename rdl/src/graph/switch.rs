use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::{Module, Parameter};
use crate::tensor::Result;

use super::node::*;
use super::FlowBuilder;

/// Builder returned by [`FlowBuilder::switch`].
/// Finalized automatically — returns FlowBuilder ready for chaining.
pub(super) fn wire_switch(
    mut fb: FlowBuilder,
    router: Box<dyn Module>,
    branches: Vec<Box<dyn Module>>,
) -> FlowBuilder {
    if fb.err.is_some() {
        return fb;
    }
    if fb.current.len() != 1 {
        fb.err = Some("switch requires single stream".into());
        return fb;
    }
    if branches.len() < 2 {
        fb.err = Some("switch requires at least 2 branches".into());
        return fb;
    }

    let cur = fb.current[0].clone();
    let id = fb.next_id("switch");

    let router: Rc<dyn Module> = Rc::from(router);
    let branch_modules: Vec<Rc<dyn Module>> = branches
        .into_iter()
        .map(|b| Rc::from(b) as Rc<dyn Module>)
        .collect();

    let composite: Rc<dyn Module> = Rc::new(SwitchComposite {
        router: router.clone(),
        branches: branch_modules.clone(),
    });

    let run = make_switch_func(router, branch_modules);

    fb.nodes.insert(
        id.clone(),
        Node {
            id: id.clone(),
            input_ports: vec![DEFAULT_INPUT.into()],
            output_ports: vec![DEFAULT_OUTPUT.into()],
            run,
            module: Some(composite),
            ref_forward: None,
            trace_buf: None,
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

fn make_switch_func(
    router: Rc<dyn Module>,
    branches: Vec<Rc<dyn Module>>,
) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let stream = &inputs[0];

        // Router selects branch index (scalar output, 0-based)
        let route_out = router.forward(stream)?;
        let idx = route_out.data().item()? as usize;

        if idx >= branches.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "switch: router selected branch {} but only {} branches exist",
                idx,
                branches.len()
            )));
        }

        let output = branches[idx].forward(stream)?;
        Ok(vec![output])
    })
}

/// Bundles router + branches for parameter collection.
struct SwitchComposite {
    router: Rc<dyn Module>,
    branches: Vec<Rc<dyn Module>>,
}

impl Module for SwitchComposite {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Delegate to router + first branch as fallback
        self.branches[0].forward(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.router.parameters();
        for branch in &self.branches {
            params.extend(branch.parameters());
        }
        params
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        let mut subs = vec![self.router.clone()];
        subs.extend(self.branches.iter().cloned());
        subs
    }

    fn move_to_device(&self, device: crate::tensor::Device) {
        self.router.move_to_device(device);
        for b in &self.branches {
            b.move_to_device(device);
        }
    }

    fn set_training(&self, training: bool) {
        self.router.set_training(training);
        for b in &self.branches {
            b.set_training(training);
        }
    }
}
