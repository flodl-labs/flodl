use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::{Linear, Module};
use crate::tensor::{Device, Result, Tensor};

/// Threshold-based halt condition for Loop.While / Loop.Until.
///
/// Signals halt (positive output) when the maximum element of the state
/// exceeds the threshold.
///
/// ```ignore
/// FlowBuilder::from(body)
///     .loop_body(body).until_cond(ThresholdHalt::new(50.0), 20)
/// ```
pub struct ThresholdHalt {
    threshold: f32,
}

impl ThresholdHalt {
    pub fn new(threshold: f32) -> Self {
        ThresholdHalt { threshold }
    }
}

impl Module for ThresholdHalt {
    fn name(&self) -> &str { "threshold_halt" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let data = input.data().to_f32_vec()?;
        let max_val = data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let val = max_val - self.threshold; // positive when exceeded → halt
        Ok(Variable::new(
            Tensor::from_f32(&[val], &[1], input.device())?,
            false,
        ))
    }
}

/// Learnable halt condition (Adaptive Computation Time / ACT pattern).
///
/// A linear probe projects the state to a scalar — iteration stops when
/// the output is positive. Fully differentiable.
///
/// ```ignore
/// FlowBuilder::from(body)
///     .loop_body(body).until_cond(LearnedHalt::new(hidden_dim)?, 20)
/// ```
pub struct LearnedHalt {
    proj: Rc<Linear>,
}

impl LearnedHalt {
    pub fn new(input_dim: i64) -> Result<Self> {
        Self::on_device(input_dim, Device::CPU)
    }

    pub fn on_device(input_dim: i64, device: Device) -> Result<Self> {
        Ok(LearnedHalt {
            proj: Rc::new(Linear::on_device(input_dim, 1, device)?),
        })
    }
}

impl Module for LearnedHalt {
    fn name(&self) -> &str { "learned_halt" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.proj.forward(input)
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![self.proj.clone()]
    }
}
