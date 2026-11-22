//! NAdam (Nesterov-accelerated Adam) optimizer.

use std::io::{Read, Write};

use crate::autograd::{Variable, no_grad};
use crate::tensor::Result;

use crate::nn::checkpoint::{
    read_f64_le, read_i64_le, read_tensor_state, read_u32_le, write_f64_le, write_i64_le,
    write_tensor_state, write_u32_le,
};
use crate::nn::parameter::Parameter;

use super::{Optimizer, Stateful};

/// NAdam optimizer (Dozat, 2016).
///
/// Incorporates Nesterov momentum into Adam. Equivalent to Adam with
/// a look-ahead gradient, providing faster convergence on some tasks.
///
/// Update rule:
///   m = beta1 * m + (1 - beta1) * grad
///   v = beta2 * v + (1 - beta2) * grad^2
///   m_hat = beta1 * m / (1 - beta1^(t+1)) + (1 - beta1) * grad / (1 - beta1^t)
///   param -= lr * m_hat / (sqrt(v / (1 - beta2^t)) + eps)
pub struct NAdam {
    params: Vec<Variable>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    m: Vec<Option<crate::tensor::Tensor>>,
    v: Vec<Option<crate::tensor::Tensor>>,
    /// Per-param step counts, incremented only when the param has a grad
    /// (the Nesterov bias-correction schedule is per-param).
    steps: Vec<i64>,
}

impl NAdam {
    /// Create a new NAdam optimizer with default betas (0.9, 0.999), eps (1e-8),
    /// and no weight decay.
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        NAdam {
            params: params.iter().map(|p| p.variable.clone()).collect(),
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: vec![None; n],
            v: vec![None; n],
            steps: vec![0; n],
        }
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 {
        self.lr
    }
}

impl Optimizer for NAdam {
    fn lr(&self) -> f64 {
        self.lr
    }
    /// One batched update for all parameters: a handful of multi-tensor
    /// kernels, where a per-parameter loop costs several calls and
    /// temporaries for each tensor. Bias correction follows each
    /// parameter's own step count, so parameters are updated in buckets
    /// of equal count (one bucket unless some joined late).
    fn step(&mut self) -> Result<()> {
        use crate::tensor::Tensor;

        let (b1, b2, eps, lr) = (self.beta1, self.beta2, self.eps, self.lr);
        no_grad(|| {
            let (idx, data, grads) = super::with_grads(&self.params, 0..self.params.len())?;
            if idx.is_empty() {
                return Ok(());
            }
            // g += wd * p, into new tensors: the stored gradient is the
            // caller's to read after the step.
            let grads = if self.weight_decay > 0.0 {
                Tensor::foreach_add_list(&grads, &data, self.weight_decay)?
            } else {
                grads
            };

            // m = b1 * m + (1 - b1) * g;  v = b2 * v + (1 - b2) * g^2
            let m = super::state_list(&mut self.m, &idx, &grads)?;
            let v = super::state_list(&mut self.v, &idx, &grads)?;
            Tensor::foreach_lerp_scalar_(&m, &grads, 1.0 - b1)?;
            Tensor::foreach_mul_scalar_(&v, b2)?;
            Tensor::foreach_addcmul_(&v, &grads, &grads, 1.0 - b2)?;

            for (t, at) in super::advance_steps(&mut self.steps, &idx) {
                let t = t as f64;
                let (b1t, b2t) = (b1.powf(t), b2.powf(t));
                let (p, g, m, v) = (
                    super::pick(&data, &at),
                    super::pick(&grads, &at),
                    super::pick(&m, &at),
                    super::pick(&v, &at),
                );
                // sqrt(v / (1 - b2^t)) + eps
                let denom = Tensor::foreach_sqrt(&v)?;
                Tensor::foreach_mul_scalar_(&denom, 1.0 / (1.0 - b2t).sqrt())?;
                Tensor::foreach_add_scalar_(&denom, eps)?;
                // The Nesterov-corrected first moment, as its two terms:
                // p -= lr * (b1 * m / (1 - b1^(t+1)) + (1 - b1) * g / (1 - b1^t)) / denom
                let b1t1 = b1.powf(t + 1.0);
                Tensor::foreach_addcdiv_(&p, &m, &denom, -lr * b1 / (1.0 - b1t1))?;
                Tensor::foreach_addcdiv_(&p, &g, &denom, -lr * (1.0 - b1) / (1.0 - b1t))?;
            }
            Ok(())
        })
    }

    fn reset_state(&mut self) {
        // Moment estimates back to fresh, step counts to 0 (Nesterov
        // momentum schedule restarts). Lengths preserved for per-param indexing.
        for slot in &mut self.m {
            *slot = None;
        }
        for slot in &mut self.v {
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

impl Stateful for NAdam {
    fn state_kind(&self) -> super::StateKind {
        super::StateKind::NAdam
    }

    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, self.params.len() as u32)?;
        write_f64_le(w, self.lr)?;
        write_f64_le(w, self.beta1)?;
        write_f64_le(w, self.beta2)?;
        write_f64_le(w, self.eps)?;
        write_f64_le(w, self.weight_decay)?;
        for i in 0..self.params.len() {
            write_tensor_state(w, self.m[i].as_ref())?;
            write_tensor_state(w, self.v[i].as_ref())?;
            write_i64_le(w, self.steps[i])?;
        }
        // Empty group table: NAdam has no group support yet, but the
        // slot keeps the payload shape uniform with grouped optimizers.
        super::write_groups(w, &[])?;
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let count = read_u32_le(r)? as usize;
        if count != self.params.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "NAdam: param count mismatch: checkpoint={} optimizer={}",
                count,
                self.params.len()
            )));
        }
        self.lr = read_f64_le(r)?;
        self.beta1 = read_f64_le(r)?;
        self.beta2 = read_f64_le(r)?;
        self.eps = read_f64_le(r)?;
        self.weight_decay = read_f64_le(r)?;
        for i in 0..self.params.len() {
            let dev = self.params[i].data().device();
            self.m[i] = read_tensor_state(r, dev)?;
            self.v[i] = read_tensor_state(r, dev)?;
            self.steps[i] = read_i64_le(r)?;
        }
        let groups = super::read_groups(r, self.params.len(), "NAdam")?;
        if !groups.is_empty() {
            return Err(crate::tensor::TensorError::new(
                "NAdam: state file carries a group table, but this flodl's \
                 NAdam has no parameter-group support",
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
    fn nadam_matches_a_reference_over_seven_steps() {
        // Values from a float64 transcription of the update rule. `late`
        // gets its first gradient at step 3, so from then on the two
        // parameters sit at different step counts in one batched call.
        let dev = crate::tensor::test_device();
        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[1, 2], dev).unwrap(), "p");
        let late = Parameter::new(Tensor::from_f32(&[3.0, 4.0], &[1, 2], dev).unwrap(), "late");
        let mut opt = NAdam::new(&[p.clone(), late.clone()], 0.1);

        for step in 1..=7 {
            opt.zero_grad();
            set_grad(
                &p,
                if step % 2 == 1 {
                    &[1.0, 2.0]
                } else {
                    &[0.5, -1.0]
                },
            );
            if step >= 3 {
                set_grad(&late, &[2.0, 1.0]);
            }
            opt.step().unwrap();
            match step {
                2 => {
                    assert_close(&values(&p), &[0.760_519, 1.869_118]);
                    assert_close(&values(&late), &[3.0, 4.0]);
                }
                5 => {
                    assert_close(&values(&p), &[0.456_265, 1.716_571]);
                    assert_close(&values(&late), &[2.629_078, 3.629_078]);
                }
                7 => {
                    assert_close(&values(&p), &[0.265_355, 1.649_059]);
                    assert_close(&values(&late), &[2.421_342, 3.421_342]);
                }
                _ => {}
            }
        }
        assert_eq!(opt.steps, vec![7, 5]);
    }

    #[test]
    fn nadam_weight_decay_leaves_the_gradient() {
        let dev = crate::tensor::test_device();
        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[1, 2], dev).unwrap(), "p");
        let mut opt = NAdam::new(std::slice::from_ref(&p), 0.1);
        opt.weight_decay = 0.1;
        set_grad(&p, &[1.0, 1.0]);
        opt.step().unwrap();
        assert_close(
            &p.variable.grad().unwrap().to_f32_vec().unwrap(),
            &[1.0, 1.0],
        );
        assert_ne!(values(&p), vec![1.0, 2.0]);
    }

    #[test]
    fn test_nadam_state_file_roundtrip() {
        // Locks the Stateful impl added for NAdam: per-param steps + lr
        // round-trip through the .optim header.
        let dev = crate::tensor::test_device();
        let p = make_param("w", &[2]);
        let mut opt = NAdam::new(std::slice::from_ref(&p), 0.02);
        p.variable
            .set_grad(Tensor::from_f32(&[0.1, 0.2], &[2], dev).unwrap());
        opt.step().unwrap();

        let path = state_tmp("nadam_roundtrip.optim");
        opt.save_state_to(&path).unwrap();

        let mut opt2 = NAdam::new(std::slice::from_ref(&p), 0.5);
        opt2.load_state_file(&path).unwrap();
        assert_eq!(opt2.steps, opt.steps);
        assert!((opt2.lr - 0.02).abs() < 1e-12);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_nadam_reset_state_clears_moments_and_steps() {
        let dev = crate::tensor::test_device();
        let p = make_param("w", &[2]);
        let mut opt = NAdam::new(std::slice::from_ref(&p), 0.01);
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
        assert!(opt.m.iter().all(|s| s.is_none()), "m must be cleared");
        assert!(opt.v.iter().all(|s| s.is_none()), "v must be cleared");
    }

    #[test]
    fn test_nadam_steps() {
        let p = make_param("w", &[1]);
        let before = p.variable.data().item().unwrap();
        let mut opt = NAdam::new(std::slice::from_ref(&p), 0.01);
        let x = Variable::new(
            Tensor::from_f32(&[2.0], &[1], crate::tensor::test_device()).unwrap(),
            false,
        );
        let loss = x.mul(&p.variable).unwrap().sum().unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();
        let after = p.variable.data().item().unwrap();
        assert!(
            (after - before).abs() > 1e-6,
            "NAdam step should change parameter"
        );
    }

    #[test]
    fn test_nadam_convergence_100_steps() {
        use crate::nn::{Linear, Module, loss::mse_loss};

        let dev = crate::tensor::test_device();
        let model = Linear::on_device(4, 1, dev).unwrap();
        let mut opt = NAdam::new(&model.parameters(), 0.01);

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

        for _ in 0..100 {
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
            "NAdam should converge: first={}, final={}",
            first_loss,
            final_loss
        );
    }
}
