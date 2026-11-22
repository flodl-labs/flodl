//! SGD with optional momentum, weight decay, and per-group learning rates.

use std::io::{Read, Write};

use crate::autograd::{Variable, no_grad};
use crate::tensor::Result;

use crate::nn::checkpoint::{
    read_f64_le, read_tensor_state, read_u32_le, write_f64_le, write_tensor_state, write_u32_le,
};
use crate::nn::parameter::Parameter;

use super::{GroupMeta, Optimizer, Stateful};

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
    weight_decay: f64,
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
            weight_decay: 0.0,
            velocity,
            groups: vec![],
        }
    }

    /// Set L2 weight decay (default 0.0). Applied as `grad += wd * param`
    /// before the momentum update, matching PyTorch's SGD behavior.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Create a builder for SGD with per-group learning rates.
    pub fn with_groups(momentum: f64) -> SGDBuilder {
        SGDBuilder {
            momentum,
            weight_decay: 0.0,
            groups: vec![],
        }
    }

    /// Current learning rate (base LR, or first group's LR).
    pub fn lr(&self) -> f64 {
        self.lr
    }

    /// The parameter ranges that share a learning rate: the declared
    /// groups, or every parameter at the base rate when none were declared
    /// ([`SGD::new`] declares none).
    fn spans(&self) -> Vec<(std::ops::Range<usize>, f64)> {
        if self.groups.is_empty() {
            vec![(0..self.params.len(), self.lr)]
        } else {
            self.groups
                .iter()
                .map(|g| (g.range.clone(), g.lr))
                .collect()
        }
    }
}

/// Builder for SGD with per-group learning rates.
pub struct SGDBuilder {
    momentum: f64,
    weight_decay: f64,
    groups: Vec<(Vec<Variable>, f64)>,
}

impl SGDBuilder {
    /// Add a parameter group with its own learning rate.
    pub fn group(mut self, params: &[Parameter], lr: f64) -> Self {
        let vars: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        self.groups.push((vars, lr));
        self
    }

    /// Set L2 weight decay (default 0.0).
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
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
            groups.push(GroupMeta {
                lr,
                range: start..end,
            });
        }

        let velocity = vec![None; all_params.len()];
        SGD {
            params: all_params,
            lr: base_lr,
            momentum: self.momentum,
            weight_decay: self.weight_decay,
            velocity,
            groups,
        }
    }
}

impl Optimizer for SGD {
    fn lr(&self) -> f64 {
        self.lr
    }
    /// One batched update per parameter group: a handful of multi-tensor
    /// kernels for the whole group, where a per-parameter loop costs
    /// several calls and temporaries for each tensor. The update is
    /// PyTorch's (`g += wd * p`, `v = m * v + g`, `p -= lr * v`), summed
    /// in an order that needs no temporary: `v = (m * v + g) + wd * p`.
    fn step(&mut self) -> Result<()> {
        use crate::tensor::Tensor;

        no_grad(|| {
            let (momentum, wd) = (self.momentum, self.weight_decay);
            for (range, lr) in self.spans() {
                // Parameters without a gradient this step are left alone,
                // velocity included.
                let mut data = Vec::new();
                let mut grads = Vec::new();
                let mut idx = Vec::new();
                for i in range {
                    if let Some(grad) = self.params[i].grad() {
                        data.push(self.params[i].data().detach()?);
                        grads.push(grad);
                        idx.push(i);
                    }
                }
                if data.is_empty() {
                    continue;
                }

                if momentum > 0.0 {
                    // Split by whether the velocity exists yet: a fresh one
                    // starts as a deep copy of the gradient (a shallow clone
                    // would share its storage and corrupt it in place).
                    let (mut seen_v, mut seen_g, mut seen_d) = (Vec::new(), Vec::new(), Vec::new());
                    let (mut new_v, mut new_d) = (Vec::new(), Vec::new());
                    let mut velocity = Vec::with_capacity(idx.len());
                    for (k, &i) in idx.iter().enumerate() {
                        match &self.velocity[i] {
                            Some(v) => {
                                seen_v.push(v.clone());
                                seen_g.push(grads[k].clone());
                                seen_d.push(data[k].clone());
                                velocity.push(v.clone());
                            }
                            None => {
                                let v = grads[k].copy()?;
                                new_v.push(v.clone());
                                new_d.push(data[k].clone());
                                velocity.push(v.clone());
                                self.velocity[i] = Some(v);
                            }
                        }
                    }
                    Tensor::foreach_mul_scalar_(&seen_v, momentum)?;
                    Tensor::foreach_add_list_(&seen_v, &seen_g, 1.0)?;
                    if wd > 0.0 {
                        Tensor::foreach_add_list_(&seen_v, &seen_d, wd)?;
                        Tensor::foreach_add_list_(&new_v, &new_d, wd)?;
                    }
                    Tensor::foreach_add_list_(&data, &velocity, -lr)?;
                } else {
                    // p -= lr * (g + wd * p), as p * (1 - lr * wd) - lr * g.
                    if wd > 0.0 {
                        Tensor::foreach_mul_scalar_(&data, 1.0 - lr * wd)?;
                    }
                    Tensor::foreach_add_list_(&data, &grads, -lr)?;
                }
            }
            Ok(())
        })
    }

    fn reset_state(&mut self) {
        // Momentum velocity buffers back to fresh (lazily re-allocated on the
        // next step, as at construction). Length preserved so the per-param
        // indexing stays valid.
        for slot in &mut self.velocity {
            *slot = None;
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad_set_to_none();
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

    fn save_state_to(&self, path: &str) -> Result<()> {
        <Self as Stateful>::save_state_file(self, path)
    }
}

impl Stateful for SGD {
    fn state_kind(&self) -> super::StateKind {
        super::StateKind::Sgd
    }

    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, self.params.len() as u32)?;
        write_f64_le(w, self.lr)?;
        write_f64_le(w, self.weight_decay)?;
        for v in &self.velocity {
            write_tensor_state(w, v.as_ref())?;
        }
        // Groups
        super::write_groups(w, &self.groups)?;
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let count = read_u32_le(r)? as usize;
        if count != self.params.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "SGD: param count mismatch: checkpoint={} optimizer={}",
                count,
                self.params.len()
            )));
        }
        self.lr = read_f64_le(r)?;
        self.weight_decay = read_f64_le(r)?;
        for (i, param) in self.params.iter().enumerate() {
            let dev = param.data().device();
            self.velocity[i] = read_tensor_state(r, dev)?;
        }
        // Groups
        self.groups = super::read_groups(r, self.params.len(), "SGD")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    /// Backward a loss whose gradient with respect to `p` is exactly `c`.
    fn set_grad(p: &Parameter, c: &[f32]) {
        let dev = crate::tensor::test_device();
        let c = Variable::new(
            Tensor::from_f32(c, &[1, c.len() as i64], dev).unwrap(),
            false,
        );
        p.variable
            .mul(&c)
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
    }

    fn values(p: &Parameter) -> Vec<f32> {
        p.variable.data().to_f32_vec().unwrap()
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn momentum_with_weight_decay_matches_the_closed_form() {
        // g += wd*p; v = m*v + g; p -= lr*v, with g = [1, 1] every step.
        let dev = crate::tensor::test_device();
        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[1, 2], dev).unwrap(), "p");
        let mut opt = SGD::new(std::slice::from_ref(&p), 0.5, 0.9).weight_decay(0.1);

        set_grad(&p, &[1.0, 1.0]);
        opt.step().unwrap();
        // v = [1.1, 1.2]
        assert_close(&values(&p), &[0.45, 1.4]);

        opt.zero_grad();
        set_grad(&p, &[1.0, 1.0]);
        opt.step().unwrap();
        // v = 0.9*[1.1, 1.2] + [1, 1] + 0.1*[0.45, 1.4] = [2.035, 2.22]
        assert_close(&values(&p), &[-0.5675, 0.29]);
    }

    #[test]
    fn plain_sgd_with_weight_decay_matches_the_closed_form() {
        let dev = crate::tensor::test_device();
        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[1, 2], dev).unwrap(), "p");
        let mut opt = SGD::new(std::slice::from_ref(&p), 0.5, 0.0).weight_decay(0.1);
        set_grad(&p, &[1.0, 1.0]);
        opt.step().unwrap();
        // p - 0.5 * (1 + 0.1*p)
        assert_close(&values(&p), &[0.45, 1.4]);
    }

    #[test]
    fn a_parameter_without_a_gradient_is_left_alone() {
        // Batching must not sweep in a parameter that took no part in the
        // loss: its value stays, and its velocity starts fresh the first
        // time it does get a gradient, not from a stale or zero buffer.
        let dev = crate::tensor::test_device();
        let used = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[1, 2], dev).unwrap(), "used");
        let idle = Parameter::new(Tensor::from_f32(&[3.0, 4.0], &[1, 2], dev).unwrap(), "idle");
        let mut opt = SGD::new(&[used.clone(), idle.clone()], 0.5, 0.9);

        set_grad(&used, &[1.0, 1.0]);
        opt.step().unwrap();
        assert_close(&values(&used), &[0.5, 1.5]);
        assert_close(&values(&idle), &[3.0, 4.0]);

        opt.zero_grad();
        set_grad(&used, &[1.0, 1.0]);
        set_grad(&idle, &[2.0, 2.0]);
        opt.step().unwrap();
        // used: v = 0.9*1 + 1 = 1.9. idle: first gradient, v = 2.
        assert_close(&values(&used), &[-0.45, 0.55]);
        assert_close(&values(&idle), &[2.0, 3.0]);
    }

    #[test]
    fn test_sgd_parameter_groups_different_lr() {
        let dev = crate::tensor::test_device();
        let p_fast = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[1, 2], dev).unwrap(), "fast");
        let p_slow = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[1, 2], dev).unwrap(), "slow");

        let mut opt = SGD::with_groups(0.0)
            .group(std::slice::from_ref(&p_fast), 1.0)
            .group(std::slice::from_ref(&p_slow), 0.001)
            .build();

        let x = Variable::new(Tensor::from_f32(&[1.0], &[1, 1], dev).unwrap(), false);
        let y_fast = x.matmul(&p_fast.variable).unwrap();
        let y_slow = x.matmul(&p_slow.variable).unwrap();
        let loss = y_fast.add(&y_slow).unwrap().sum().unwrap();
        loss.backward().unwrap();

        let fast_before = p_fast.variable.data().to_f32_vec().unwrap();
        let slow_before = p_slow.variable.data().to_f32_vec().unwrap();
        opt.step().unwrap();
        let fast_after = p_fast.variable.data().to_f32_vec().unwrap();
        let slow_after = p_slow.variable.data().to_f32_vec().unwrap();

        let fast_delta: f64 = fast_before
            .iter()
            .zip(&fast_after)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum();
        let slow_delta: f64 = slow_before
            .iter()
            .zip(&slow_after)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum();

        assert!(
            fast_delta > slow_delta * 100.0,
            "fast group (lr=1.0) should move much more than slow (lr=0.001): fast={}, slow={}",
            fast_delta,
            slow_delta
        );
    }

    #[test]
    fn test_set_lr_affects_actual_update_magnitude() {
        let dev = crate::tensor::test_device();

        // Two identical params, same gradient, different LR via set_lr
        let p_lo = Parameter::new(Tensor::from_f32(&[5.0], &[1], dev).unwrap(), "lo");
        let p_hi = Parameter::new(Tensor::from_f32(&[5.0], &[1], dev).unwrap(), "hi");

        let mut opt_lo = SGD::new(std::slice::from_ref(&p_lo), 0.001, 0.0);
        let mut opt_hi = SGD::new(std::slice::from_ref(&p_hi), 0.001, 0.0);
        opt_hi.set_lr(1.0);

        // Compute gradients for both
        let x = Variable::new(Tensor::from_f32(&[1.0], &[1], dev).unwrap(), false);
        let loss_lo = x.mul(&p_lo.variable).unwrap().sum().unwrap();
        loss_lo.backward().unwrap();
        let loss_hi = x.mul(&p_hi.variable).unwrap().sum().unwrap();
        loss_hi.backward().unwrap();

        opt_lo.step().unwrap();
        opt_hi.step().unwrap();

        let val_lo = p_lo.variable.data().to_f32_vec().unwrap()[0];
        let val_hi = p_hi.variable.data().to_f32_vec().unwrap()[0];

        let delta_lo = (5.0 - val_lo).abs();
        let delta_hi = (5.0 - val_hi).abs();

        assert!(
            delta_hi > delta_lo * 100.0,
            "set_lr(1.0) should produce much larger update than 0.001: hi={}, lo={}",
            delta_hi,
            delta_lo
        );
    }
}
