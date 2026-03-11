use crate::autograd::Variable;
use crate::tensor::Result;

use super::parameter::Parameter;

/// Optimizer trait.
pub trait Optimizer {
    fn step(&mut self) -> Result<()>;
    fn zero_grad(&self);
    fn set_lr(&mut self, lr: f64);
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
