//! Adagrad optimizer.

use std::io::{Read, Write};

use crate::autograd::{Variable, no_grad};
use crate::tensor::Result;

use crate::nn::checkpoint::{
    read_f64_le, read_i64_le, read_tensor_state, read_u32_le, write_f64_le, write_i64_le,
    write_tensor_state, write_u32_le,
};
use crate::nn::parameter::Parameter;

use super::{Optimizer, Stateful};

/// Adagrad optimizer (Duchi et al., 2011).
///
/// Adapts learning rate per-parameter based on historical gradient magnitude.
/// Good for sparse gradients (NLP embeddings).
///
/// Update rule:
///   state_sum += grad^2
///   param -= lr * grad / (sqrt(state_sum) + eps)
pub struct Adagrad {
    params: Vec<Variable>,
    lr: f64,
    eps: f64,
    weight_decay: f64,
    lr_decay: f64,
    state_sum: Vec<Option<crate::tensor::Tensor>>,
    /// Per-param step counts, incremented only when the param has a grad
    /// (the lr_decay schedule is per-param, PyTorch parity).
    steps: Vec<i64>,
}

/// Builder for Adagrad optimizer.
pub struct AdagradBuilder {
    params: Vec<Parameter>,
    lr: f64,
    eps: f64,
    weight_decay: f64,
    lr_decay: f64,
}

impl AdagradBuilder {
    /// Set epsilon for numerical stability (default: 1e-10).
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
    /// Set L2 penalty / weight decay (default: 0.0).
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
    /// Set learning rate decay applied each step: `clr = lr / (1 + (step-1) * lr_decay)` (default: 0.0).
    pub fn lr_decay(mut self, lr_decay: f64) -> Self {
        self.lr_decay = lr_decay;
        self
    }

    /// Build the Adagrad optimizer.
    pub fn build(self) -> Adagrad {
        let n = self.params.len();
        Adagrad {
            params: self.params.iter().map(|p| p.variable.clone()).collect(),
            lr: self.lr,
            eps: self.eps,
            weight_decay: self.weight_decay,
            lr_decay: self.lr_decay,
            state_sum: vec![None; n],
            steps: vec![0; n],
        }
    }
}

impl Adagrad {
    /// Create a new Adagrad optimizer with default parameters:
    /// eps=1e-10, weight_decay=0, lr_decay=0.
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        Adagrad {
            params: params.iter().map(|p| p.variable.clone()).collect(),
            lr,
            eps: 1e-10,
            weight_decay: 0.0,
            lr_decay: 0.0,
            state_sum: vec![None; n],
            steps: vec![0; n],
        }
    }

    /// Create a builder for Adagrad with customizable options.
    pub fn builder(params: &[Parameter], lr: f64) -> AdagradBuilder {
        AdagradBuilder {
            params: params.to_vec(),
            lr,
            eps: 1e-10,
            weight_decay: 0.0,
            lr_decay: 0.0,
        }
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 {
        self.lr
    }
}

impl Optimizer for Adagrad {
    fn lr(&self) -> f64 {
        self.lr
    }
    /// One batched update for all parameters, PyTorch's multi-tensor
    /// form: a handful of kernels for the whole list, where a
    /// per-parameter loop costs several calls and temporaries for each
    /// tensor.
    fn step(&mut self) -> Result<()> {
        use crate::tensor::Tensor;

        no_grad(|| {
            let (idx, data, grads) = super::with_grads(&self.params, 0..self.params.len())?;
            if idx.is_empty() {
                return Ok(());
            }
            // Per-param step: the lr_decay schedule follows each param's
            // own update count, so a late-unfrozen param starts its
            // schedule fresh (PyTorch parity). Hence one rate per tensor.
            let minus_clr: Vec<f64> = idx
                .iter()
                .map(|&i| {
                    self.steps[i] += 1;
                    -self.lr / (1.0 + (self.steps[i] - 1) as f64 * self.lr_decay)
                })
                .collect();
            // g += wd * p, into new tensors: the stored gradient is the
            // caller's to read after the step.
            let grads = if self.weight_decay > 0.0 {
                Tensor::foreach_add_list(&grads, &data, self.weight_decay)?
            } else {
                grads
            };

            // state_sum += g^2
            let sum = super::state_list(&mut self.state_sum, &idx, &grads)?;
            Tensor::foreach_addcmul_(&sum, &grads, &grads, 1.0)?;

            // p -= clr * g / (sqrt(state_sum) + eps)
            let denom = Tensor::foreach_sqrt(&sum)?;
            Tensor::foreach_add_scalar_(&denom, self.eps)?;
            Tensor::foreach_addcdiv_scalars_(&data, &grads, &denom, &minus_clr)
        })
    }

    fn reset_state(&mut self) {
        // Accumulated squared-grad sums back to fresh, step counts to 0
        // (lr_decay schedule restarts). Length preserved for per-param indexing.
        for slot in &mut self.state_sum {
            *slot = None;
        }
        self.steps.fill(0);
    }

    fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad_set_to_none();
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn save_state_to(&self, path: &str) -> Result<()> {
        <Self as Stateful>::save_state_file(self, path)
    }
}

impl Stateful for Adagrad {
    fn state_kind(&self) -> super::StateKind {
        super::StateKind::Adagrad
    }

    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, self.params.len() as u32)?;
        write_f64_le(w, self.lr)?;
        write_f64_le(w, self.eps)?;
        write_f64_le(w, self.weight_decay)?;
        write_f64_le(w, self.lr_decay)?;
        for i in 0..self.params.len() {
            write_tensor_state(w, self.state_sum[i].as_ref())?;
            write_i64_le(w, self.steps[i])?;
        }
        // Empty group table: Adagrad has no group support yet, but the
        // slot keeps the payload shape uniform with grouped optimizers.
        super::write_groups(w, &[])?;
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let count = read_u32_le(r)? as usize;
        if count != self.params.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "Adagrad: param count mismatch: checkpoint={} optimizer={}",
                count,
                self.params.len()
            )));
        }
        self.lr = read_f64_le(r)?;
        self.eps = read_f64_le(r)?;
        self.weight_decay = read_f64_le(r)?;
        self.lr_decay = read_f64_le(r)?;
        for i in 0..self.params.len() {
            let dev = self.params[i].data().device();
            self.state_sum[i] = read_tensor_state(r, dev)?;
            self.steps[i] = read_i64_le(r)?;
        }
        let groups = super::read_groups(r, self.params.len(), "Adagrad")?;
        if !groups.is_empty() {
            return Err(crate::tensor::TensorError::new(
                "Adagrad: state file carries a group table, but this flodl's \
                 Adagrad has no parameter-group support",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::{assert_close, make_param, set_grad, state_tmp, values};
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn adagrad_with_lr_decay_matches_the_closed_form() {
        // sum += g^2; p -= clr * g / (sqrt(sum) + eps), with
        // clr = lr / (1 + (step - 1) * lr_decay) counted per parameter.
        let dev = crate::tensor::test_device();
        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[1, 2], dev).unwrap(), "p");
        let late = Parameter::new(Tensor::from_f32(&[3.0, 4.0], &[1, 2], dev).unwrap(), "late");
        let mut opt = Adagrad::builder(&[p.clone(), late.clone()], 0.5)
            .lr_decay(0.5)
            .build();

        set_grad(&p, &[1.0, 2.0]);
        opt.step().unwrap();
        // sum = [1, 4], clr = 0.5: every element moves by 0.5.
        assert_close(&values(&p), &[0.5, 1.5]);
        assert_close(&values(&late), &[3.0, 4.0]);

        opt.zero_grad();
        set_grad(&p, &[1.0, 2.0]);
        set_grad(&late, &[2.0, 2.0]);
        opt.step().unwrap();
        // p: second step, clr = 1/3, sum = [2, 8], g / sqrt(sum) = 0.707107.
        assert_close(&values(&p), &[0.264_298, 1.264_298]);
        // late: its own first step, clr = 0.5, in the same batched call.
        assert_close(&values(&late), &[2.5, 3.5]);
    }

    #[test]
    fn adagrad_weight_decay_leaves_the_gradient() {
        let dev = crate::tensor::test_device();
        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[1, 2], dev).unwrap(), "p");
        let mut opt = Adagrad::builder(std::slice::from_ref(&p), 0.5)
            .weight_decay(0.1)
            .build();
        set_grad(&p, &[1.0, 1.0]);
        opt.step().unwrap();
        // g = [1.1, 1.2], sum = g^2: a first step moves by lr whatever g is.
        assert_close(&values(&p), &[0.5, 1.5]);
        assert_close(
            &p.variable.grad().unwrap().to_f32_vec().unwrap(),
            &[1.0, 1.0],
        );
    }

    #[test]
    fn test_adagrad_state_file_roundtrip() {
        // Locks the Stateful impl added for Adagrad: per-param steps + lr
        // round-trip through the .optim header.
        let dev = crate::tensor::test_device();
        let p = make_param("w", &[2]);
        let mut opt = Adagrad::new(std::slice::from_ref(&p), 0.02);
        p.variable
            .set_grad(Tensor::from_f32(&[0.1, 0.2], &[2], dev).unwrap());
        opt.step().unwrap();

        let path = state_tmp("adagrad_roundtrip.optim");
        opt.save_state_to(&path).unwrap();

        let mut opt2 = Adagrad::new(std::slice::from_ref(&p), 0.5);
        opt2.load_state_file(&path).unwrap();
        assert_eq!(opt2.steps, opt.steps);
        assert!((opt2.lr - 0.02).abs() < 1e-12);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_adagrad_reset_state_clears_accum_and_steps() {
        // reset_state must wipe the accumulated squared-grad sums and the
        // per-param step counters (lr_decay schedule restarts).
        let dev = crate::tensor::test_device();
        let p = make_param("w", &[2]);
        let mut opt = Adagrad::new(std::slice::from_ref(&p), 0.01);
        for _ in 0..3 {
            p.variable
                .set_grad(Tensor::from_f32(&[0.1, -0.2], &[2], dev).unwrap());
            opt.step().unwrap();
        }
        assert!(
            opt.steps.iter().any(|&s| s > 0),
            "warm-up should advance steps"
        );
        opt.reset_state();
        assert!(opt.steps.iter().all(|&s| s == 0), "steps must reset to 0");
        assert!(
            opt.state_sum.iter().all(|s| s.is_none()),
            "state_sum must be cleared"
        );
    }

    #[test]
    fn test_adagrad_steps() {
        let p = make_param("w", &[1]);
        let before = p.variable.data().item().unwrap();
        let mut opt = Adagrad::new(std::slice::from_ref(&p), 0.5);
        let x = Variable::new(
            Tensor::from_f32(&[2.0], &[1], crate::tensor::test_device()).unwrap(),
            false,
        );
        let loss = x.mul(&p.variable).unwrap().sum().unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();
        let after = p.variable.data().item().unwrap();
        // Parameter should change
        assert!(
            (after - before).abs() > 1e-6,
            "Adagrad step should change parameter"
        );
    }

    #[test]
    fn test_adagrad_convergence_50_steps() {
        use crate::nn::{Linear, Module, loss::mse_loss};

        let dev = crate::tensor::test_device();
        let model = Linear::on_device(4, 1, dev).unwrap();
        let mut opt = Adagrad::new(&model.parameters(), 0.1);

        let x = Variable::new(
            Tensor::from_f32(
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
                &[4, 4],
                dev,
            )
            .unwrap(),
            false,
        );
        let target = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1], dev).unwrap(),
            false,
        );

        let first_loss;
        {
            let pred = model.forward(&x).unwrap();
            first_loss = mse_loss(&pred, &target).unwrap().item().unwrap();
        }

        for _ in 0..50 {
            opt.zero_grad();
            let pred = model.forward(&x).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
        }

        let pred = model.forward(&x).unwrap();
        let final_loss = mse_loss(&pred, &target).unwrap().item().unwrap();
        assert!(
            final_loss < first_loss * 0.5,
            "Adagrad should converge: first={}, final={}",
            first_loss,
            final_loss
        );
    }
}
