use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::Result;

use super::node::*;
use super::FlowBuilder;

/// Builder for loop constructs. Created by [`FlowBuilder::loop_body`].
pub struct LoopBuilder {
    fb: FlowBuilder,
    body: Box<dyn Module>,
}

impl LoopBuilder {
    pub(super) fn new(fb: FlowBuilder, body: Box<dyn Module>) -> Self {
        LoopBuilder { fb, body }
    }

    /// Fixed iteration count: repeat body N times.
    pub fn for_n(self, n: usize) -> FlowBuilder {
        let mut fb = self.fb;
        if fb.err.is_some() {
            return fb;
        }
        if fb.current.len() != 1 {
            fb.err = Some("loop requires single stream".into());
            return fb;
        }
        if n < 1 {
            fb.err = Some("loop requires at least 1 iteration".into());
            return fb;
        }

        let body: Rc<dyn Module> = Rc::from(self.body);
        let trace_buf: Rc<RefCell<Vec<Variable>>> = Rc::new(RefCell::new(Vec::new()));
        let ports: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(vec![DEFAULT_INPUT.into()]));
        let run = make_for_loop_func(body.clone(), n, trace_buf.clone(), ports.clone());
        let composite: Rc<dyn Module> = Rc::new(LoopComposite {
            body,
            cond: None,
        });
        wire_loop(fb, composite, run, trace_buf, ports)
    }

    /// Repeat while condition says "continue" (positive output = halt).
    /// Condition checked before each iteration — body may never run.
    pub fn while_cond(self, cond: impl Module + 'static, max_iter: usize) -> FlowBuilder {
        let mut fb = self.fb;
        if fb.err.is_some() {
            return fb;
        }
        if fb.current.len() != 1 {
            fb.err = Some("loop requires single stream".into());
            return fb;
        }
        if max_iter < 1 {
            fb.err = Some("loop requires max_iter >= 1".into());
            return fb;
        }

        let body: Rc<dyn Module> = Rc::from(self.body);
        let cond: Rc<dyn Module> = Rc::new(cond);
        let trace_buf: Rc<RefCell<Vec<Variable>>> = Rc::new(RefCell::new(Vec::new()));
        let ports: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(vec![DEFAULT_INPUT.into()]));
        let run = make_while_loop_func(body.clone(), cond.clone(), max_iter, trace_buf.clone(), ports.clone());
        let composite: Rc<dyn Module> = Rc::new(LoopComposite {
            body,
            cond: Some(cond),
        });
        wire_loop(fb, composite, run, trace_buf, ports)
    }

    /// Repeat until condition signals halt (positive output = halt).
    /// Body always runs at least once.
    pub fn until_cond(self, cond: impl Module + 'static, max_iter: usize) -> FlowBuilder {
        let mut fb = self.fb;
        if fb.err.is_some() {
            return fb;
        }
        if fb.current.len() != 1 {
            fb.err = Some("loop requires single stream".into());
            return fb;
        }
        if max_iter < 1 {
            fb.err = Some("loop requires max_iter >= 1".into());
            return fb;
        }

        let body: Rc<dyn Module> = Rc::from(self.body);
        let cond: Rc<dyn Module> = Rc::new(cond);
        let trace_buf: Rc<RefCell<Vec<Variable>>> = Rc::new(RefCell::new(Vec::new()));
        let ports: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(vec![DEFAULT_INPUT.into()]));
        let run = make_until_loop_func(body.clone(), cond.clone(), max_iter, trace_buf.clone(), ports.clone());
        let composite: Rc<dyn Module> = Rc::new(LoopComposite {
            body,
            cond: Some(cond),
        });
        wire_loop(fb, composite, run, trace_buf, ports)
    }
}

/// Wire a loop node into the graph and return the updated FlowBuilder.
fn wire_loop(
    mut fb: FlowBuilder,
    composite: Rc<dyn Module>,
    run: NodeFn,
    trace_buf: Rc<RefCell<Vec<Variable>>>,
    ports: Rc<RefCell<Vec<String>>>,
) -> FlowBuilder {
    let cur = fb.current[0].clone();
    let id = fb.next_id("loop");

    // If the body supports NamedInputModule, expose ref_forward on the loop node
    // so that .using() can be chained after the loop.
    let ref_forward = if composite.sub_modules().first()
        .and_then(|body| body.as_named_input())
        .is_some()
    {
        let body_rc = composite.sub_modules().into_iter().next().unwrap();
        let rf: RefForwardFn = Rc::new(move |input, refs| {
            body_rc.as_named_input().unwrap().forward_named(input, refs)
        });
        Some(rf)
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
            trace_buf: Some(trace_buf),
            loop_ports: Some(ports),
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

/// Execute body step: use forward_named if refs are available, plain forward otherwise.
fn body_step(
    body: &Rc<dyn Module>,
    state: &Variable,
    refs: &HashMap<String, Variable>,
) -> Result<Variable> {
    if !refs.is_empty()
        && let Some(named) = body.as_named_input()
    {
        return named.forward_named(state, refs);
    }
    body.forward(state)
}

fn make_for_loop_func(
    body: Rc<dyn Module>,
    count: usize,
    trace_buf: Rc<RefCell<Vec<Variable>>>,
    ports: Rc<RefCell<Vec<String>>>,
) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let mut state = inputs[0].clone();
        let refs = extract_refs(&ports.borrow(), inputs);
        trace_buf.borrow_mut().clear();
        body.reset();
        for i in 0..count {
            state = body_step(&body, &state, &refs).map_err(|e| {
                crate::tensor::TensorError::new(&format!("loop iteration {}: {}", i, e))
            })?;
            if let Some(t) = body.trace() {
                trace_buf.borrow_mut().push(t);
            }
        }
        Ok(vec![state])
    })
}

fn make_while_loop_func(
    body: Rc<dyn Module>,
    cond: Rc<dyn Module>,
    max_iter: usize,
    trace_buf: Rc<RefCell<Vec<Variable>>>,
    ports: Rc<RefCell<Vec<String>>>,
) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let mut state = inputs[0].clone();
        let refs = extract_refs(&ports.borrow(), inputs);
        trace_buf.borrow_mut().clear();
        body.reset();
        for i in 0..max_iter {
            let halt = cond.forward(&state)?;
            let halt_val = halt.data().to_f32_vec().map_err(|e| {
                crate::tensor::TensorError::new(&format!(
                    "loop condition at iteration {}: {}",
                    i, e
                ))
            })?;
            if !halt_val.is_empty() && halt_val[0] > 0.0 {
                break;
            }
            state = body_step(&body, &state, &refs).map_err(|e| {
                crate::tensor::TensorError::new(&format!("loop iteration {}: {}", i, e))
            })?;
            if let Some(t) = body.trace() {
                trace_buf.borrow_mut().push(t);
            }
        }
        Ok(vec![state])
    })
}

fn make_until_loop_func(
    body: Rc<dyn Module>,
    cond: Rc<dyn Module>,
    max_iter: usize,
    trace_buf: Rc<RefCell<Vec<Variable>>>,
    ports: Rc<RefCell<Vec<String>>>,
) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let mut state = inputs[0].clone();
        let refs = extract_refs(&ports.borrow(), inputs);
        trace_buf.borrow_mut().clear();
        body.reset();
        for i in 0..max_iter {
            state = body_step(&body, &state, &refs).map_err(|e| {
                crate::tensor::TensorError::new(&format!("loop iteration {}: {}", i, e))
            })?;
            if let Some(t) = body.trace() {
                trace_buf.borrow_mut().push(t);
            }
            // Skip condition check on last iteration
            if i < max_iter - 1 {
                let halt = cond.forward(&state)?;
                let halt_val = halt.data().to_f32_vec().map_err(|e| {
                    crate::tensor::TensorError::new(&format!(
                        "loop condition at iteration {}: {}",
                        i, e
                    ))
                })?;
                if !halt_val.is_empty() && halt_val[0] > 0.0 {
                    break;
                }
            }
        }
        Ok(vec![state])
    })
}

/// Bundles body + optional condition for parameter collection.
struct LoopComposite {
    body: Rc<dyn Module>,
    cond: Option<Rc<dyn Module>>,
}

impl Module for LoopComposite {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.body.forward(input)
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        let mut subs = vec![self.body.clone()];
        if let Some(ref cond) = self.cond {
            subs.push(cond.clone());
        }
        subs
    }

    fn move_to_device(&self, device: crate::tensor::Device) {
        self.body.move_to_device(device);
        if let Some(ref cond) = self.cond {
            cond.move_to_device(device);
        }
    }

    fn set_training(&self, training: bool) {
        self.body.set_training(training);
        if let Some(ref cond) = self.cond {
            cond.set_training(training);
        }
    }

    fn reset(&self) {
        self.body.reset();
    }

    fn detach_state(&self) {
        self.body.detach_state();
    }
}
