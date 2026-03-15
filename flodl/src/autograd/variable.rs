use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::tensor::{DType, Device, Result, Tensor};

pub(crate) struct VariableInner {
    pub data: Tensor,
}

/// A differentiable variable wrapping a Tensor.
///
/// Variables use libtorch's native autograd. When a tensor has
/// `requires_grad=true`, all standard operations build a C++ computation
/// graph automatically. Calling `backward()` runs libtorch's backward
/// engine — no Rust-side graph walking.
///
/// Internally uses `Rc<RefCell<>>` for shared ownership — the same
/// parameter can be referenced by both a Module and an Optimizer.
///
/// ```ignore
/// let w = Variable::new(Tensor::randn(&[3, 2], opts)?, true);
/// let x = Variable::new(Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], Device::CPU)?, false);
/// let loss = x.matmul(&w)?.sum()?;
/// loss.backward()?;
/// println!("{:?}", w.grad()); // gradient of loss w.r.t. w
/// ```
#[derive(Clone)]
pub struct Variable {
    pub(crate) inner: Rc<RefCell<VariableInner>>,
}

impl Variable {
    /// Create a leaf variable (parameter or input data).
    /// If `requires_grad` is true, libtorch will track operations for autodiff.
    pub fn new(data: Tensor, requires_grad: bool) -> Self {
        let data = if requires_grad {
            // Set requires_grad on the C++ tensor so libtorch tracks ops
            data.set_requires_grad(true).unwrap_or(data)
        } else {
            data
        };
        Variable {
            inner: Rc::new(RefCell::new(VariableInner { data })),
        }
    }

    /// Wrap a tensor that already has the correct requires_grad flag set.
    /// Used by ops to wrap libtorch output tensors (which inherit autograd
    /// metadata from their inputs automatically).
    pub(crate) fn wrap(data: Tensor) -> Self {
        Variable {
            inner: Rc::new(RefCell::new(VariableInner { data })),
        }
    }

    /// Get the underlying tensor data (shallow clone).
    pub fn data(&self) -> Tensor {
        self.inner.borrow().data.clone()
    }

    /// Get the accumulated gradient, if any.
    /// Reads from the C++ tensor's .grad() field.
    pub fn grad(&self) -> Option<Tensor> {
        self.inner.borrow().data.grad()
    }

    /// Replace the gradient tensor (for gradient clipping / unscaling).
    pub fn set_grad(&self, grad: Tensor) {
        let _ = self.inner.borrow().data.set_grad(&grad);
    }

    /// Whether this variable tracks gradients.
    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().data.requires_grad()
    }

    /// Change whether this variable tracks gradients.
    /// Replaces the inner data handle (the FFI returns a new handle sharing storage).
    /// All clones of this Variable share the same `Rc<RefCell>`, so the change
    /// is visible everywhere (module, optimizer, etc.).
    pub fn set_requires_grad(&self, requires_grad: bool) -> Result<()> {
        let data = self.inner.borrow().data.set_requires_grad(requires_grad)?;
        self.inner.borrow_mut().data = data;
        Ok(())
    }

    /// Whether this is a leaf variable (no grad_fn in libtorch).
    /// A leaf tensor is one created by the user, not by an operation.
    pub fn is_leaf(&self) -> bool {
        self.inner.borrow().data.is_leaf()
    }

    /// Count unique autograd nodes reachable from this variable's grad_fn.
    /// Returns 0 for leaf variables. Measures graph complexity — compare
    /// against Python's equivalent to detect decomposed-op bloat.
    pub fn autograd_node_count(&self) -> i64 {
        self.inner.borrow().data.autograd_node_count()
    }

    /// Shape of the underlying data tensor.
    pub fn shape(&self) -> Vec<i64> {
        self.inner.borrow().data.shape()
    }

    /// Data type of the underlying tensor.
    pub fn dtype(&self) -> DType {
        self.inner.borrow().data.dtype()
    }

    /// Device where the underlying tensor lives.
    pub fn device(&self) -> Device {
        self.inner.borrow().data.device()
    }

    /// Extract a scalar value as f64.
    pub fn item(&self) -> Result<f64> {
        self.inner.borrow().data.item()
    }

    /// Zero out the accumulated gradient.
    pub fn zero_grad(&self) {
        let _ = self.inner.borrow().data.zero_grad();
    }

    /// Null out the gradient instead of zeroing it. No CUDA kernel.
    pub fn zero_grad_set_to_none(&self) {
        self.inner.borrow().data.zero_grad_set_to_none();
    }

    /// Detach from the computation graph. Returns a new leaf variable
    /// sharing the same data tensor (detached) with no gradient tracking.
    pub fn detach(&self) -> Variable {
        let detached = self.inner.borrow().data.detach()
            .unwrap_or_else(|_| self.inner.borrow().data.clone());
        Variable::wrap(detached)
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
    /// Preserves the `requires_grad` flag from the current tensor.
    pub fn set_data(&self, data: Tensor) {
        let rg = self.requires_grad();
        let data = if rg {
            data.set_requires_grad(true).unwrap_or(data)
        } else {
            data
        };
        self.inner.borrow_mut().data = data;
    }

    /// Number of elements in the data tensor.
    pub fn numel(&self) -> i64 {
        self.inner.borrow().data.numel()
    }

    /// Run backward pass from this scalar variable.
    /// Populates .grad() on all leaf variables in the computation graph.
    ///
    /// After backward completes, the tensor is detached in-place to
    /// immediately release the C++ grad_fn chain. Without this, the
    /// autograd Node objects stay alive until the Variable is dropped.
    pub fn backward(&self) -> Result<()> {
        let inner = self.inner.borrow();
        inner.data.backward()?;
        inner.data.detach_()?;
        Ok(())
    }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.borrow();
        write!(
            f,
            "Variable({:?}, {:?}, {:?}, requires_grad={})",
            inner.data.shape(),
            inner.data.dtype(),
            inner.data.device(),
            inner.data.requires_grad(),
        )
    }
}
