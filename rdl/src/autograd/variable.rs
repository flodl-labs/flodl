use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::tensor::{DType, Device, Result, Tensor};

/// Backward closure type: given upstream gradient, returns gradients for each input.
pub(crate) type GradApplyFn = Box<dyn Fn(&Tensor) -> Result<Vec<Tensor>>>;

/// The backward function for a graph node.
///
/// When the node is processed during backward, `apply` is called with the
/// incoming gradient and must return one gradient per input.
///
/// Saved tensors live inside the `apply` closure. When the GradFn is dropped
/// (set to None after backward processes the node), the closure is dropped,
/// which drops the saved tensors — deterministic VRAM release with zero
/// infrastructure. This replaces goDl's Retain/Release + Phase 1 + Phase 3.
pub(crate) struct GradFn {
    #[allow(dead_code)]
    pub name: &'static str,
    pub inputs: Vec<Variable>,
    pub apply: GradApplyFn,
}

pub(crate) struct VariableInner {
    pub data: Tensor,
    pub grad: Option<Tensor>,
    pub requires_grad: bool,
    pub grad_fn: Option<GradFn>,
    pub is_leaf: bool,
}

/// A differentiable variable wrapping a Tensor.
///
/// Variables track computation history for reverse-mode autodiff.
/// Leaf variables (created by user) accumulate gradients during backward.
/// Non-leaf variables (created by ops) hold a GradFn describing how to
/// compute input gradients.
///
/// Internally uses `Rc<RefCell<>>` for shared ownership — the backward
/// graph holds references to input variables via `GradFn.inputs`.
#[derive(Clone)]
pub struct Variable {
    pub(crate) inner: Rc<RefCell<VariableInner>>,
}

impl Variable {
    /// Create a leaf variable (parameter or input data).
    pub fn new(data: Tensor, requires_grad: bool) -> Self {
        Variable {
            inner: Rc::new(RefCell::new(VariableInner {
                data,
                grad: None,
                requires_grad,
                grad_fn: None,
                is_leaf: true,
            })),
        }
    }

    /// Create a non-leaf variable from an operation result.
    pub(crate) fn from_op(data: Tensor, grad_fn: GradFn) -> Self {
        Variable {
            inner: Rc::new(RefCell::new(VariableInner {
                data,
                grad: None,
                requires_grad: true,
                grad_fn: Some(grad_fn),
                is_leaf: false,
            })),
        }
    }

    /// Create a leaf with no gradient tracking (for op results when grad disabled).
    pub(crate) fn leaf(data: Tensor, requires_grad: bool) -> Self {
        Self::new(data, requires_grad)
    }

    /// Get the underlying tensor data (shallow clone).
    pub fn data(&self) -> Tensor {
        self.inner.borrow().data.clone()
    }

    /// Get the accumulated gradient, if any (shallow clone).
    pub fn grad(&self) -> Option<Tensor> {
        self.inner.borrow().grad.clone()
    }

    /// Replace the gradient tensor (for gradient clipping).
    pub fn set_grad(&self, grad: Tensor) {
        self.inner.borrow_mut().grad = Some(grad);
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    pub fn is_leaf(&self) -> bool {
        self.inner.borrow().is_leaf
    }

    pub fn shape(&self) -> Vec<i64> {
        self.inner.borrow().data.shape()
    }

    pub fn dtype(&self) -> DType {
        self.inner.borrow().data.dtype()
    }

    pub fn device(&self) -> Device {
        self.inner.borrow().data.device()
    }

    /// Extract a scalar value as f64.
    pub fn item(&self) -> Result<f64> {
        self.inner.borrow().data.item()
    }

    /// Zero out the accumulated gradient.
    pub fn zero_grad(&self) {
        self.inner.borrow_mut().grad = None;
    }

    /// Detach from the computation graph. Returns a new leaf variable
    /// sharing the same data tensor (shallow clone) with no gradient tracking.
    pub fn detach(&self) -> Variable {
        Variable::new(self.data(), false)
    }

    /// Move to a different device. Returns a new leaf variable.
    pub fn to_device(&self, device: Device) -> Result<Variable> {
        if self.device() == device {
            return Ok(self.clone());
        }
        let moved = self.inner.borrow().data.to_device(device)?;
        Ok(Variable::new(moved, self.requires_grad()))
    }

    /// Replace the underlying tensor data (used by optimizers).
    pub fn set_data(&self, data: Tensor) {
        self.inner.borrow_mut().data = data;
    }

    /// Number of elements in the data tensor.
    pub fn numel(&self) -> i64 {
        self.inner.borrow().data.numel()
    }

    /// Accumulate a gradient into this variable's grad field.
    pub(crate) fn accumulate_grad(&self, grad: &Tensor) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        match inner.grad.take() {
            None => {
                inner.grad = Some(grad.clone());
            }
            Some(existing) => {
                inner.grad = Some(existing.add(grad)?);
            }
        }
        Ok(())
    }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.borrow();
        write!(
            f,
            "Variable({:?}, {:?}, {:?}, requires_grad={}, is_leaf={})",
            inner.data.shape(),
            inner.data.dtype(),
            inner.data.device(),
            inner.requires_grad,
            inner.is_leaf,
        )
    }
}
