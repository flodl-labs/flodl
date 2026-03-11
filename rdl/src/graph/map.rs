use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::Module;

use super::node::*;
use super::FlowBuilder;

/// Builder for Map constructs. Created by [`FlowBuilder::map`].
pub struct MapBuilder {
    fb: FlowBuilder,
    body: Box<dyn Module>,
    batched: bool,
}

impl MapBuilder {
    pub(super) fn new(fb: FlowBuilder, body: Box<dyn Module>) -> Self {
        MapBuilder {
            fb,
            body,
            batched: false,
        }
    }

    /// Fast path: pass full tensor to body in one call instead of element-wise.
    pub fn batched(mut self) -> Self {
        self.batched = true;
        self
    }

    /// Apply body to each element along dim 0 of the current stream.
    pub fn each(self) -> FlowBuilder {
        let mut fb = self.fb;
        if fb.err.is_some() {
            return fb;
        }
        if fb.current.len() != 1 {
            fb.err = Some("map requires single stream".into());
            return fb;
        }

        let cur = fb.current[0].clone();
        let body: Rc<dyn Module> = Rc::from(self.body);
        let composite: Rc<dyn Module> = body.clone();
        let batched = self.batched;
        let run = make_map_each_func(body, batched);
        let id = fb.next_id("map");

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
}

fn make_map_each_func(body: Rc<dyn Module>, batched: bool) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let stream = &inputs[0];

        if batched {
            // Fast path: pass full tensor to body
            let output = body.forward(stream)?;
            return Ok(vec![output]);
        }

        // Element-wise: narrow along dim 0, apply body, cat results
        let n = stream.shape()[0];
        if n == 0 {
            return Ok(vec![stream.clone()]);
        }

        let first = body.forward(&stream.narrow(0, 0, 1)?)?;
        let mut result = first;

        for i in 1..n {
            let element = stream.narrow(0, i, 1)?;
            let out = body.forward(&element)?;
            result = result.cat(&out, 0)?;
        }

        Ok(vec![result])
    })
}
