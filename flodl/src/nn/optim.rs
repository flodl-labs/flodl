use std::io::{Read, Write};

use crate::autograd::{Variable, no_grad};
use crate::tensor::Result;

use super::checkpoint::{
    write_tensor_state, read_tensor_state, write_f64_le, read_f64_le,
    write_u32_le, read_u32_le, write_i64_le, read_i64_le,
};
use super::parameter::Parameter;

/// Optimizer trait: step, zero gradients, and adjust learning rate.
pub trait Optimizer {
    /// Perform a single optimization step using accumulated gradients.
    fn step(&mut self) -> Result<()>;
    /// Reset all parameter gradients to zero.
    fn zero_grad(&self);
    /// Update the learning rate (all groups if grouped).
    fn set_lr(&mut self, lr: f64);
    /// Set learning rate for a specific parameter group.
    /// Default: falls back to `set_lr` (single-group optimizers).
    fn set_group_lr(&mut self, _group: usize, lr: f64) {
        self.set_lr(lr);
    }
}

/// Per-group learning rate metadata.
struct GroupMeta {
    lr: f64,
    range: std::ops::Range<usize>,
}

/// Stateful trait for components that can save/load training state.
pub trait Stateful {
    /// Serialize optimizer state (lr, momentum buffers, etc.) to a writer.
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()>;
    /// Restore optimizer state from a reader.
    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()>;
}

/// SGD with optional momentum.
///
/// With momentum: `v = momentum * v + grad; param -= lr * v`.
/// Without momentum: `param -= lr * grad`.
///
/// ```ignore
/// let mut optim = SGD::new(&model.parameters(), 0.01, 0.9);
/// for batch in &data {
///     optim.zero_grad();
///     let loss = mse_loss(&model.forward(&batch.x)?, &batch.y)?;
///     loss.backward()?;
///     optim.step()?;
/// }
/// ```
pub struct SGD {
    params: Vec<Variable>,
    lr: f64,
    momentum: f64,
    velocity: Vec<Option<crate::tensor::Tensor>>,
    groups: Vec<GroupMeta>,
}

impl SGD {
    /// Create a new SGD optimizer. Set `momentum` to 0.0 for vanilla SGD.
    pub fn new(params: &[Parameter], lr: f64, momentum: f64) -> Self {
        let variables: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        let velocity = vec![None; variables.len()];
        SGD {
            params: variables,
            lr,
            momentum,
            velocity,
            groups: vec![],
        }
    }

    /// Create a builder for SGD with per-group learning rates.
    pub fn with_groups(momentum: f64) -> SGDBuilder {
        SGDBuilder { momentum, groups: vec![] }
    }

    /// Current learning rate (base LR, or first group's LR).
    pub fn lr(&self) -> f64 {
        self.lr
    }

    fn lr_for_param(&self, i: usize) -> f64 {
        for g in &self.groups {
            if g.range.contains(&i) {
                return g.lr;
            }
        }
        self.lr
    }
}

/// Builder for SGD with per-group learning rates.
pub struct SGDBuilder {
    momentum: f64,
    groups: Vec<(Vec<Variable>, f64)>,
}

impl SGDBuilder {
    /// Add a parameter group with its own learning rate.
    pub fn group(mut self, params: &[Parameter], lr: f64) -> Self {
        let vars: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        self.groups.push((vars, lr));
        self
    }

    /// Build the SGD optimizer.
    pub fn build(self) -> SGD {
        let mut all_params = Vec::new();
        let mut groups = Vec::new();
        let base_lr = self.groups.first().map(|(_, lr)| *lr).unwrap_or(0.01);

        for (vars, lr) in self.groups {
            let start = all_params.len();
            all_params.extend(vars);
            let end = all_params.len();
            groups.push(GroupMeta { lr, range: start..end });
        }

        let velocity = vec![None; all_params.len()];
        SGD {
            params: all_params,
            lr: base_lr,
            momentum: self.momentum,
            velocity,
            groups,
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (i, param) in self.params.iter().enumerate() {
                if let Some(grad) = param.grad() {
                    let lr = self.lr_for_param(i);
                    let data = param.data().detach()?;
                    if self.momentum > 0.0 {
                        let v = match self.velocity[i].take() {
                            Some(v) => {
                                v.mul_scalar_(self.momentum)?;
                                v.add_(&grad)?;
                                v
                            }
                            // mul_scalar creates a new tensor with independent storage,
                            // unlike clone() which shares storage with grad
                            None => grad.mul_scalar(1.0)?,
                        };
                        // data -= lr * v
                        let scaled = v.mul_scalar(lr)?;
                        data.sub_(&scaled)?;
                        self.velocity[i] = Some(v);
                    } else {
                        // data -= lr * grad
                        let scaled = grad.mul_scalar(lr)?;
                        data.sub_(&scaled)?;
                    }
                }
            }
            Ok(())
        })
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
        for g in &mut self.groups {
            g.lr = lr;
        }
    }

    fn set_group_lr(&mut self, group: usize, lr: f64) {
        if let Some(g) = self.groups.get_mut(group) {
            g.lr = lr;
        }
    }
}

impl Stateful for SGD {
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, self.params.len() as u32)?;
        write_f64_le(w, self.lr)?;
        for v in &self.velocity {
            write_tensor_state(w, v.as_ref())?;
        }
        // Groups
        write_u32_le(w, self.groups.len() as u32)?;
        for g in &self.groups {
            write_f64_le(w, g.lr)?;
            write_i64_le(w, g.range.start as i64)?;
            write_i64_le(w, g.range.end as i64)?;
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
        // Groups
        let ng = read_u32_le(r)? as usize;
        self.groups.clear();
        for _ in 0..ng {
            let lr = read_f64_le(r)?;
            let start = read_i64_le(r)? as usize;
            let end = read_i64_le(r)? as usize;
            self.groups.push(GroupMeta { lr, range: start..end });
        }
        Ok(())
    }
}

/// Adam optimizer with bias correction (Kingma & Ba, 2014).
///
/// Maintains per-parameter first and second moment estimates with
/// bias correction. Default betas: (0.9, 0.999), eps: 1e-8.
///
/// ```ignore
/// let mut optim = Adam::new(&model.parameters(), 0.001);
/// ```
pub struct Adam {
    params: Vec<Variable>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    m: Vec<Option<crate::tensor::Tensor>>,
    v: Vec<Option<crate::tensor::Tensor>>,
    t: usize,
    groups: Vec<GroupMeta>,
}

impl Adam {
    /// Create a new Adam optimizer with default betas (0.9, 0.999) and eps (1e-8).
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
            groups: vec![],
        }
    }

    /// Create a builder for Adam with per-group learning rates.
    pub fn with_groups() -> AdamBuilder {
        AdamBuilder { groups: vec![] }
    }

    /// Current learning rate (base LR, or first group's LR).
    pub fn lr(&self) -> f64 {
        self.lr
    }

    fn lr_for_param(&self, i: usize) -> f64 {
        for g in &self.groups {
            if g.range.contains(&i) {
                return g.lr;
            }
        }
        self.lr
    }
}

/// Builder for Adam with per-group learning rates.
pub struct AdamBuilder {
    groups: Vec<(Vec<Variable>, f64)>,
}

impl AdamBuilder {
    /// Add a parameter group with its own learning rate.
    pub fn group(mut self, params: &[Parameter], lr: f64) -> Self {
        let vars: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        self.groups.push((vars, lr));
        self
    }

    /// Build the Adam optimizer.
    pub fn build(self) -> Adam {
        let mut all_params = Vec::new();
        let mut groups = Vec::new();
        let base_lr = self.groups.first().map(|(_, lr)| *lr).unwrap_or(1e-3);

        for (vars, lr) in self.groups {
            let start = all_params.len();
            all_params.extend(vars);
            let end = all_params.len();
            groups.push(GroupMeta { lr, range: start..end });
        }

        let n = all_params.len();
        Adam {
            params: all_params,
            lr: base_lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: vec![None; n],
            v: vec![None; n],
            t: 0,
            groups,
        }
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
        for g in &mut self.groups {
            g.lr = lr;
        }
    }

    fn set_group_lr(&mut self, group: usize, lr: f64) {
        if let Some(g) = self.groups.get_mut(group) {
            g.lr = lr;
        }
    }
}

impl Adam {
    fn adam_update(&mut self, weight_decay: f64) -> Result<()> {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        no_grad(|| {
            for (i, param) in self.params.iter().enumerate() {
                if let Some(grad) = param.grad() {
                    let lr = self.lr_for_param(i);

                    // First moment: m = β1*m + (1-β1)*grad
                    let m = match self.m[i].take() {
                        Some(m) => {
                            m.mul_scalar_(self.beta1)?;
                            m.add_(&grad.mul_scalar(1.0 - self.beta1)?)?;
                            m
                        }
                        None => grad.mul_scalar(1.0 - self.beta1)?,
                    };

                    // Second moment: v = β2*v + (1-β2)*grad²
                    let grad_sq = grad.mul(&grad)?;
                    let v = match self.v[i].take() {
                        Some(v) => {
                            v.mul_scalar_(self.beta2)?;
                            v.add_(&grad_sq.mul_scalar(1.0 - self.beta2)?)?;
                            v
                        }
                        None => grad_sq.mul_scalar(1.0 - self.beta2)?,
                    };

                    // Bias-corrected estimates
                    let m_hat = m.mul_scalar(1.0 / bc1)?;
                    let v_hat = v.mul_scalar(1.0 / bc2)?;

                    self.m[i] = Some(m);
                    self.v[i] = Some(v);

                    // param -= lr * m_hat / (sqrt(v_hat) + eps)
                    let denom = v_hat.sqrt()?.add_scalar(self.eps)?;
                    let update = m_hat.div(&denom)?.mul_scalar(lr)?;
                    let data = param.data().detach()?;

                    // Decoupled weight decay: data *= (1 - lr * wd)
                    if weight_decay > 0.0 {
                        data.mul_scalar_(1.0 - lr * weight_decay)?;
                    }

                    data.sub_(&update)?;
                }
            }
            Ok(())
        })
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
        // Groups
        write_u32_le(w, self.groups.len() as u32)?;
        for g in &self.groups {
            write_f64_le(w, g.lr)?;
            write_i64_le(w, g.range.start as i64)?;
            write_i64_le(w, g.range.end as i64)?;
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
        // Groups
        let ng = read_u32_le(r)? as usize;
        self.groups.clear();
        for _ in 0..ng {
            let lr = read_f64_le(r)?;
            let start = read_i64_le(r)? as usize;
            let end = read_i64_le(r)? as usize;
            self.groups.push(GroupMeta { lr, range: start..end });
        }
        Ok(())
    }
}

/// AdamW optimizer — Adam with decoupled weight decay (Loshchilov & Hutter, 2017).
///
/// Unlike L2 regularization, weight decay is applied directly to parameters,
/// not to gradients. This distinction matters for adaptive optimizers and
/// generally improves generalization.
///
/// ```ignore
/// let mut optim = AdamW::new(&model.parameters(), 0.001, 0.01);
/// ```
pub struct AdamW {
    adam: Adam,
    weight_decay: f64,
}

impl AdamW {
    /// Create a new AdamW optimizer with decoupled weight decay.
    pub fn new(params: &[Parameter], lr: f64, weight_decay: f64) -> Self {
        AdamW {
            adam: Adam::new(params, lr),
            weight_decay,
        }
    }

    /// Create a builder for AdamW with per-group learning rates.
    pub fn with_groups(weight_decay: f64) -> AdamWBuilder {
        AdamWBuilder { weight_decay, groups: vec![] }
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 {
        self.adam.lr
    }
}

/// Builder for AdamW with per-group learning rates.
pub struct AdamWBuilder {
    weight_decay: f64,
    groups: Vec<(Vec<Variable>, f64)>,
}

impl AdamWBuilder {
    /// Add a parameter group with its own learning rate.
    pub fn group(mut self, params: &[Parameter], lr: f64) -> Self {
        let vars: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        self.groups.push((vars, lr));
        self
    }

    /// Build the AdamW optimizer.
    pub fn build(self) -> AdamW {
        let mut all_params = Vec::new();
        let mut groups = Vec::new();
        let base_lr = self.groups.first().map(|(_, lr)| *lr).unwrap_or(1e-3);

        for (vars, lr) in self.groups {
            let start = all_params.len();
            all_params.extend(vars);
            let end = all_params.len();
            groups.push(GroupMeta { lr, range: start..end });
        }

        let n = all_params.len();
        AdamW {
            adam: Adam {
                params: all_params,
                lr: base_lr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                m: vec![None; n],
                v: vec![None; n],
                t: 0,
                groups,
            },
            weight_decay: self.weight_decay,
        }
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
        self.adam.set_lr(lr);
    }

    fn set_group_lr(&mut self, group: usize, lr: f64) {
        self.adam.set_group_lr(group, lr);
    }
}

impl Stateful for AdamW {
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_f64_le(w, self.weight_decay)?;
        self.adam.save_state(w)
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        self.weight_decay = read_f64_le(r)?;
        self.adam.load_state(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Device, Tensor, TensorOptions};

    fn make_param(name: &str, shape: &[i64]) -> Parameter {
        let t = Tensor::randn(shape, TensorOptions {
            dtype: crate::tensor::DType::Float32,
            device: Device::CPU,
        }).unwrap();
        Parameter::new(t, name)
    }

    #[test]
    fn test_adam_backward_compat() {
        // Adam::new still works with a single LR
        let p = make_param("w", &[3, 2]);
        let mut opt = Adam::new(&[p.clone()], 0.01);

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], Device::CPU).unwrap(),
            false,
        );
        let y = x.matmul(&p.variable).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let before = p.variable.data().to_f32_vec().unwrap();
        opt.step().unwrap();
        let after = p.variable.data().to_f32_vec().unwrap();
        assert_ne!(before, after, "params should change after step");
    }

    #[test]
    fn test_adam_two_groups_different_lr() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);

        // Group 0: high LR, Group 1: very low LR
        let mut opt = Adam::with_groups()
            .group(&[p1.clone()], 0.1)
            .group(&[p2.clone()], 1e-10)
            .build();

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], Device::CPU).unwrap(),
            false,
        );
        // Both params participate
        let y1 = x.matmul(&p1.variable).unwrap();
        let y2 = x.matmul(&p2.variable).unwrap();
        let loss = y1.add(&y2).unwrap().sum().unwrap();
        loss.backward().unwrap();

        let p1_before = p1.variable.data().to_f32_vec().unwrap();
        let p2_before = p2.variable.data().to_f32_vec().unwrap();
        opt.step().unwrap();
        let p1_after = p1.variable.data().to_f32_vec().unwrap();
        let p2_after = p2.variable.data().to_f32_vec().unwrap();

        // p1 should change substantially (high LR), p2 barely moves (tiny LR)
        let p1_delta: f64 = p1_before.iter().zip(&p1_after)
            .map(|(a, b)| (a - b).abs() as f64).sum();
        let p2_delta: f64 = p2_before.iter().zip(&p2_after)
            .map(|(a, b)| (a - b).abs() as f64).sum();

        assert!(p1_delta > p2_delta * 1e6,
            "high-LR group should move much more: p1_delta={}, p2_delta={}", p1_delta, p2_delta);
    }

    #[test]
    fn test_set_group_lr_changes_one_group() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);

        let mut opt = Adam::with_groups()
            .group(&[p1.clone()], 0.01)
            .group(&[p2.clone()], 0.01)
            .build();

        opt.set_group_lr(1, 0.99);
        // Group 0 unchanged, group 1 updated
        assert!((opt.groups[0].lr - 0.01).abs() < 1e-12);
        assert!((opt.groups[1].lr - 0.99).abs() < 1e-12);
    }

    #[test]
    fn test_set_lr_changes_all_groups() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);

        let mut opt = Adam::with_groups()
            .group(&[p1.clone()], 0.01)
            .group(&[p2.clone()], 0.05)
            .build();

        opt.set_lr(0.42);
        assert!((opt.lr - 0.42).abs() < 1e-12);
        assert!((opt.groups[0].lr - 0.42).abs() < 1e-12);
        assert!((opt.groups[1].lr - 0.42).abs() < 1e-12);
    }

    #[test]
    fn test_frozen_params_in_group_no_crash() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);
        p1.freeze().unwrap();

        let mut opt = Adam::with_groups()
            .group(&[p1.clone(), p2.clone()], 0.01)
            .build();

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], Device::CPU).unwrap(),
            false,
        );
        let y = x.matmul(&p2.variable).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        // Should not crash even though p1 is frozen (no grad)
        opt.step().unwrap();
        opt.zero_grad();
    }

    #[test]
    fn test_adam_save_load_with_groups() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);

        let mut opt = Adam::with_groups()
            .group(&[p1.clone()], 0.01)
            .group(&[p2.clone()], 0.05)
            .build();

        // Do a step to populate moment buffers
        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], Device::CPU).unwrap(),
            false,
        );
        let y1 = x.matmul(&p1.variable).unwrap();
        let y2 = x.matmul(&p2.variable).unwrap();
        let loss = y1.add(&y2).unwrap().sum().unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();

        // Save
        let mut buf = Vec::new();
        opt.save_state(&mut buf).unwrap();

        // Load into fresh optimizer with same structure
        let mut opt2 = Adam::with_groups()
            .group(&[p1.clone()], 0.99)
            .group(&[p2.clone()], 0.99)
            .build();

        let mut cursor = std::io::Cursor::new(&buf);
        opt2.load_state(&mut cursor).unwrap();

        assert_eq!(opt2.t, opt.t);
        assert!((opt2.groups[0].lr - 0.01).abs() < 1e-12);
        assert!((opt2.groups[1].lr - 0.05).abs() < 1e-12);
    }
}
