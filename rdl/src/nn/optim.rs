use std::io::{Read, Write};

use crate::autograd::Variable;
use crate::tensor::Result;

use super::checkpoint::{
    write_tensor_state, read_tensor_state, write_f64_le, read_f64_le,
    write_u32_le, read_u32_le, write_i64_le, read_i64_le,
};
use super::parameter::Parameter;

/// Optimizer trait.
pub trait Optimizer {
    fn step(&mut self) -> Result<()>;
    fn zero_grad(&self);
    fn set_lr(&mut self, lr: f64);
}

/// Stateful trait for components that can save/load training state.
pub trait Stateful {
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()>;
    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()>;
}

/// SGD with optional momentum.
pub struct SGD {
    params: Vec<Variable>,
    lr: f64,
    momentum: f64,
    velocity: Vec<Option<crate::tensor::Tensor>>,
}

impl SGD {
    pub fn new(params: &[Parameter], lr: f64, momentum: f64) -> Self {
        let variables: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        let velocity = vec![None; variables.len()];
        SGD {
            params: variables,
            lr,
            momentum,
            velocity,
        }
    }

    pub fn lr(&self) -> f64 {
        self.lr
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

impl Optimizer for SGD {
    fn step(&mut self) -> Result<()> {
        for (i, param) in self.params.iter().enumerate() {
            if let Some(grad) = param.grad() {
                let update = if self.momentum > 0.0 {
                    let v = match self.velocity[i].take() {
                        Some(v) => v.mul_scalar(self.momentum)?.add(&grad)?,
                        None => grad.clone(),
                    };
                    self.velocity[i] = Some(v.clone());
                    v.mul_scalar(self.lr)?
                } else {
                    grad.mul_scalar(self.lr)?
                };

                let new_data = param.data().sub(&update)?;
                param.set_data(new_data);
            }
        }
        Ok(())
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

impl Stateful for SGD {
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, self.params.len() as u32)?;
        write_f64_le(w, self.lr)?;
        for v in &self.velocity {
            write_tensor_state(w, v.as_ref())?;
        }
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let count = read_u32_le(r)? as usize;
        if count != self.params.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "SGD: param count mismatch: checkpoint={} optimizer={}", count, self.params.len()
            )));
        }
        self.lr = read_f64_le(r)?;
        for (i, param) in self.params.iter().enumerate() {
            let dev = param.data().device();
            self.velocity[i] = read_tensor_state(r, dev)?;
        }
        Ok(())
    }
}

/// Adam optimizer (with bias correction).
pub struct Adam {
    params: Vec<Variable>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    m: Vec<Option<crate::tensor::Tensor>>,
    v: Vec<Option<crate::tensor::Tensor>>,
    t: usize,
}

impl Adam {
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        Adam {
            params: params.iter().map(|p| p.variable.clone()).collect(),
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: vec![None; n],
            v: vec![None; n],
            t: 0,
        }
    }

    pub fn lr(&self) -> f64 {
        self.lr
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> Result<()> {
        self.adam_update(0.0)
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

impl Adam {
    fn adam_update(&mut self, weight_decay: f64) -> Result<()> {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, param) in self.params.iter().enumerate() {
            if let Some(grad) = param.grad() {
                // First moment: m = β1*m + (1-β1)*grad
                let m_new = match self.m[i].take() {
                    Some(m) => m
                        .mul_scalar(self.beta1)?
                        .add(&grad.mul_scalar(1.0 - self.beta1)?)?,
                    None => grad.mul_scalar(1.0 - self.beta1)?,
                };

                // Second moment: v = β2*v + (1-β2)*grad²
                let grad_sq = grad.mul(&grad)?;
                let v_new = match self.v[i].take() {
                    Some(v) => v
                        .mul_scalar(self.beta2)?
                        .add(&grad_sq.mul_scalar(1.0 - self.beta2)?)?,
                    None => grad_sq.mul_scalar(1.0 - self.beta2)?,
                };

                self.m[i] = Some(m_new.clone());
                self.v[i] = Some(v_new.clone());

                // Bias-corrected estimates
                let m_hat = m_new.mul_scalar(1.0 / bc1)?;
                let v_hat = v_new.mul_scalar(1.0 / bc2)?;

                // param -= lr * m_hat / (sqrt(v_hat) + eps)
                let denom = v_hat.sqrt()?.add_scalar(self.eps)?;
                let update = m_hat.div(&denom)?.mul_scalar(self.lr)?;
                let mut new_data = param.data().sub(&update)?;

                // Decoupled weight decay: param -= lr * wd * param
                if weight_decay > 0.0 {
                    let decay = param.data().mul_scalar(self.lr * weight_decay)?;
                    new_data = new_data.sub(&decay)?;
                }

                param.set_data(new_data);
            }
        }
        Ok(())
    }
}

impl Stateful for Adam {
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, self.params.len() as u32)?;
        write_f64_le(w, self.lr)?;
        write_i64_le(w, self.t as i64)?;
        for i in 0..self.params.len() {
            write_tensor_state(w, self.m[i].as_ref())?;
            write_tensor_state(w, self.v[i].as_ref())?;
        }
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let count = read_u32_le(r)? as usize;
        if count != self.params.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "Adam: param count mismatch: checkpoint={} optimizer={}", count, self.params.len()
            )));
        }
        self.lr = read_f64_le(r)?;
        self.t = read_i64_le(r)? as usize;
        for i in 0..self.params.len() {
            let dev = self.params[i].data().device();
            self.m[i] = read_tensor_state(r, dev)?;
            self.v[i] = read_tensor_state(r, dev)?;
        }
        Ok(())
    }
}

/// AdamW optimizer (Adam with decoupled weight decay).
///
/// Unlike L2 regularization, weight decay is applied directly to parameters,
/// not to gradients.
pub struct AdamW {
    adam: Adam,
    weight_decay: f64,
}

impl AdamW {
    pub fn new(params: &[Parameter], lr: f64, weight_decay: f64) -> Self {
        AdamW {
            adam: Adam::new(params, lr),
            weight_decay,
        }
    }

    pub fn lr(&self) -> f64 {
        self.adam.lr
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) -> Result<()> {
        self.adam.adam_update(self.weight_decay)
    }

    fn zero_grad(&self) {
        self.adam.zero_grad()
    }

    fn set_lr(&mut self, lr: f64) {
        self.adam.lr = lr;
    }
}

impl Stateful for AdamW {
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        self.adam.save_state(w)
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        self.adam.load_state(r)
    }
}
