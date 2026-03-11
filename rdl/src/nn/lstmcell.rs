use crate::autograd::Variable;
use crate::tensor::{Result, Tensor};

use super::linear::Linear;
use super::parameter::Parameter;
use super::Module;

/// Single-step LSTM cell.
///
/// State is packed as `cat(h, c, dim=1)` → shape `[batch, 2*hidden_size]`.
/// This allows single-Variable passthrough with graph forward refs.
///
/// Standalone usage: call `forward_step(x, state)` directly.
pub struct LSTMCell {
    xi: Linear, xf: Linear, xg: Linear, xo: Linear,
    hi: Linear, hf: Linear, hg: Linear, ho: Linear,
    hidden_size: i64,
}

impl LSTMCell {
    pub fn new(input_size: i64, hidden_size: i64) -> Result<Self> {
        Ok(LSTMCell {
            xi: Linear::new(input_size, hidden_size)?,
            xf: Linear::new(input_size, hidden_size)?,
            xg: Linear::new(input_size, hidden_size)?,
            xo: Linear::new(input_size, hidden_size)?,
            hi: Linear::no_bias(hidden_size, hidden_size)?,
            hf: Linear::no_bias(hidden_size, hidden_size)?,
            hg: Linear::no_bias(hidden_size, hidden_size)?,
            ho: Linear::no_bias(hidden_size, hidden_size)?,
            hidden_size,
        })
    }

    /// Single LSTM step. `state` is packed `cat(h, c, dim=1)` or None for zeros.
    /// Returns packed `cat(h', c', dim=1)`.
    pub fn forward_step(&self, x: &Variable, state: Option<&Variable>) -> Result<Variable> {
        let batch = x.shape()[0];
        let hs = self.hidden_size;

        let (h, c) = match state {
            Some(s) => {
                let h = s.narrow(1, 0, hs)?;
                let c = s.narrow(1, hs, hs)?;
                (h, c)
            }
            None => {
                let h = Variable::new(Tensor::zeros(&[batch, hs], Default::default())?, false);
                let c = Variable::new(Tensor::zeros(&[batch, hs], Default::default())?, false);
                (h, c)
            }
        };

        // Gates
        let i = self.xi.forward(x)?.add(&self.hi.forward(&h)?)?.sigmoid()?;
        let f = self.xf.forward(x)?.add(&self.hf.forward(&h)?)?.sigmoid()?;
        let g = self.xg.forward(x)?.add(&self.hg.forward(&h)?)?.tanh_act()?;
        let o = self.xo.forward(x)?.add(&self.ho.forward(&h)?)?.sigmoid()?;

        // Cell update: c' = f * c + i * g
        let c_new = f.mul(&c)?.add(&i.mul(&g)?)?;

        // Hidden: h' = o * tanh(c')
        let h_new = o.mul(&c_new.tanh_act()?)?;

        // Pack state: cat(h', c', dim=1)
        h_new.cat(&c_new, 1)
    }
}

impl Module for LSTMCell {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Extract just h from the packed state output
        let state = self.forward_step(input, None)?;
        state.narrow(1, 0, self.hidden_size)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for m in &[
            &self.xi, &self.xf, &self.xg, &self.xo,
            &self.hi, &self.hf, &self.hg, &self.ho,
        ] {
            params.extend(m.parameters());
        }
        params
    }
}
