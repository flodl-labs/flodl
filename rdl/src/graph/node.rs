use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::Result;

pub(crate) const DEFAULT_INPUT: &str = "input";
pub(crate) const DEFAULT_OUTPUT: &str = "output";

pub(crate) type NodeFn = Box<dyn Fn(&[Variable]) -> Result<Vec<Variable>>>;

/// Closure type for modules that accept named refs via Using.
pub(crate) type RefForwardFn =
    Rc<dyn Fn(&Variable, &HashMap<String, Variable>) -> Result<Variable>>;

pub(crate) struct Node {
    pub id: String,
    pub input_ports: Vec<String>,
    pub output_ports: Vec<String>,
    pub run: NodeFn,
    pub module: Option<Rc<dyn Module>>,
    /// If set, this module can handle Using refs via forward_named.
    pub ref_forward: Option<RefForwardFn>,
    /// Trace buffer for loop nodes whose body implements Module::trace().
    pub trace_buf: Option<Rc<RefCell<Vec<Variable>>>>,
}

#[derive(Clone, Debug)]
pub(crate) struct NodeRef {
    pub node_id: String,
    pub port: String,
}

#[derive(Clone, Debug)]
pub(crate) struct Edge {
    pub from_node: String,
    pub from_port: String,
    pub to_node: String,
    pub to_port: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ExposedPort {
    #[allow(dead_code)]
    pub name: String,
    pub node_id: String,
    pub port: String,
}

/// Build-time forward reference: Using before Tag.
/// Resolved when Tag is called, converted to StateEntry at build.
pub(crate) struct ForwardRefSpec {
    #[allow(dead_code)]
    pub name: String,
    pub reader_id: String,
    pub writer_id: String,
    pub writer_port: String,
}

/// Pending forward reference awaiting Tag resolution.
pub(crate) struct PendingUsing {
    pub reader_id: String,
}

/// Extract ref_* ports into a name → Variable map.
pub(crate) fn extract_refs(
    ports: &[String],
    inputs: &[Variable],
) -> HashMap<String, Variable> {
    let mut refs = HashMap::new();
    for (i, port) in ports.iter().enumerate() {
        if let Some(name) = port.strip_prefix("ref_")
            && i < inputs.len()
        {
            refs.insert(name.to_string(), inputs[i].clone());
        }
    }
    refs
}

/// Wrap a Module into a NodeFn (single input → single output).
pub(crate) fn wrap_module(module: Rc<dyn Module>) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let output = module.forward(&inputs[0])?;
        Ok(vec![output])
    })
}

/// Wrap a ref-capable module. Checks ports at call time to extract refs.
pub(crate) fn wrap_ref_module(
    module: Rc<dyn Module>,
    ref_forward: RefForwardFn,
    ports: Vec<String>,
) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let refs = extract_refs(&ports, inputs);
        let output = if refs.is_empty() {
            module.forward(&inputs[0])?
        } else {
            ref_forward(&inputs[0], &refs)?
        };
        Ok(vec![output])
    })
}
