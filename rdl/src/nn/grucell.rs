use crate::autograd::Variable;
use crate::tensor::{Result, Tensor};

use super::linear::Linear;
use super::parameter::Parameter;
use super::Module;

/// Single-step GRU cell.
///
/// Takes `(x, h)` packed as input where h is optional (nil → zeros).
/// For use with graph forward refs: input = x, ref "hidden" = h.
///
/// Standalone usage: call `forward_step(x, h)` directly.
pub struct GRUCell {
    xr: Linear, xz: Linear, xn: Linear,
    hr: Linear, hz: Linear, hn: Linear,
    hidden_size: i64,
}

impl GRUCell {
    pub fn new(input_size: i64, hidden_size: i64) -> Result<Self> {
        Ok(GRUCell {
            xr: Linear::new(input_size, hidden_size)?,
            xz: Linear::new(input_size, hidden_size)?,
            xn: Linear::new(input_size, hidden_size)?,
            hr: Linear::no_bias(hidden_size, hidden_size)?,
            hz: Linear::no_bias(hidden_size, hidden_size)?,
            hn: Linear::no_bias(hidden_size, hidden_size)?,
            hidden_size,
        })
    }

    /// Single GRU step: returns new hidden state.
    pub fn forward_step(&self, x: &Variable, h: Option<&Variable>) -> Result<Variable> {
        let batch = x.shape()[0];

        // Default hidden to zeros if not provided
        let h = match h {
            Some(h) => h.clone(),
            None => Variable::new(
                Tensor::zeros(&[batch, self.hidden_size], Default::default())?,
                false,
            ),
        };

        // Gates
        let r = self.xr.forward(x)?.add(&self.hr.forward(&h)?)?.sigmoid()?;
        let z = self.xz.forward(x)?.add(&self.hz.forward(&h)?)?.sigmoid()?;
        let n = self.xn.forward(x)?.add(&r.mul(&self.hn.forward(&h)?)?)?.tanh_act()?;

        // h' = (1 - z) * n + z * h
        let one = Variable::new(Tensor::ones_like(&z.data())?, false);
        let one_minus_z = one.sub(&z)?;
        one_minus_z.mul(&n)?.add(&z.mul(&h)?)
    }
}

impl Module for GRUCell {
    /// Forward with hidden state as input.
    /// If input shape is `[batch, input_size]`, treats as x with no hidden state.
    /// Use `forward_step` for explicit hidden state control.
    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.forward_step(input, None)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for m in &[&self.xr, &self.xz, &self.xn, &self.hr, &self.hz, &self.hn] {
            params.extend(m.parameters());
        }
        params
    }
}
