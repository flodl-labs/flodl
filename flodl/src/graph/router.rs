use std::collections::HashMap;
use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::{Linear, Module, NamedInputModule};
use crate::tensor::{Device, Result, Tensor};

/// Softmax-normalized gate router for Mixture-of-Experts.
///
/// Projects input to `num_experts` logits, applies softmax so weights
/// sum to 1. When receiving multiple inputs (via Gate.Using), they are
/// summed element-wise before projection.
///
/// ```ignore
/// FlowBuilder::from(embed)
///     .tag("ctx")
///     .through(layer)
///     .gate(SoftmaxRouter::new(hidden, 3)?, vec![a, b, c])
///     .build()
/// ```
pub struct SoftmaxRouter {
    proj: Rc<Linear>,
}

impl SoftmaxRouter {
    pub fn new(input_dim: i64, num_experts: i64) -> Result<Self> {
        Self::on_device(input_dim, num_experts, Device::CPU)
    }

    pub fn on_device(input_dim: i64, num_experts: i64, device: Device) -> Result<Self> {
        Ok(SoftmaxRouter {
            proj: Rc::new(Linear::on_device(input_dim, num_experts, device)?),
        })
    }
}

impl Module for SoftmaxRouter {
    fn name(&self) -> &str { "softmax_router" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let out = self.proj.forward(input)?;
        let dim = out.data().ndim() as i32 - 1;
        out.softmax(dim)
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![self.proj.clone()]
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }
}

impl NamedInputModule for SoftmaxRouter {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let mut combined = input.clone();
        for v in refs.values() {
            combined = combined.add(v)?;
        }
        self.forward(&combined)
    }
}

/// Independent sigmoid gate router.
///
/// Projects input to `num_experts` logits, applies sigmoid independently.
/// Weights do NOT sum to 1 — each expert is gated between 0 and 1.
pub struct SigmoidRouter {
    proj: Rc<Linear>,
}

impl SigmoidRouter {
    pub fn new(input_dim: i64, num_experts: i64) -> Result<Self> {
        Self::on_device(input_dim, num_experts, Device::CPU)
    }

    pub fn on_device(input_dim: i64, num_experts: i64, device: Device) -> Result<Self> {
        Ok(SigmoidRouter {
            proj: Rc::new(Linear::on_device(input_dim, num_experts, device)?),
        })
    }
}

impl Module for SigmoidRouter {
    fn name(&self) -> &str { "sigmoid_router" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.proj.forward(input)?.sigmoid()
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![self.proj.clone()]
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }
}

impl NamedInputModule for SigmoidRouter {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let mut combined = input.clone();
        for v in refs.values() {
            combined = combined.add(v)?;
        }
        self.forward(&combined)
    }
}

/// Fixed branch selector for Switch — always picks the same branch.
///
/// Useful for testing, ablation studies, or static configurations.
pub struct FixedSelector {
    index: f32,
}

impl FixedSelector {
    pub fn new(index: usize) -> Self {
        FixedSelector { index: index as f32 }
    }
}

impl Module for FixedSelector {
    fn name(&self) -> &str { "fixed_selector" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        Ok(Variable::new(
            Tensor::from_f32(&[self.index], &[1], input.device())?,
            false,
        ))
    }
}

/// Learnable branch selector for Switch via argmax.
///
/// Projects input to `num_branches` logits and selects the one with
/// the highest value. Selection is non-differentiable — gradients flow
/// through the selected branch only. The projection parameters are
/// included in Parameters() for policy-gradient training.
pub struct ArgmaxSelector {
    proj: Rc<Linear>,
}

impl ArgmaxSelector {
    pub fn new(input_dim: i64, num_branches: i64) -> Result<Self> {
        Self::on_device(input_dim, num_branches, Device::CPU)
    }

    pub fn on_device(input_dim: i64, num_branches: i64, device: Device) -> Result<Self> {
        Ok(ArgmaxSelector {
            proj: Rc::new(Linear::on_device(input_dim, num_branches, device)?),
        })
    }
}

impl Module for ArgmaxSelector {
    fn name(&self) -> &str { "argmax_selector" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let logits = self.proj.forward(input)?;
        let data = logits.data().to_f32_vec()?;
        let best = data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(Variable::new(
            Tensor::from_f32(&[best as f32], &[1], input.device())?,
            false,
        ))
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![self.proj.clone()]
    }
}
