//! RAdam (Rectified Adam) optimizer.

use std::io::{Read, Write};

use crate::autograd::{Variable, no_grad};
use crate::tensor::Result;

use crate::nn::checkpoint::{
    read_f64_le, read_i64_le, read_tensor_state, read_u32_le, write_f64_le, write_i64_le,
    write_tensor_state, write_u32_le,
};
use crate::nn::parameter::Parameter;

use super::{Optimizer, Stateful};

/// RAdam optimizer (Liu et al., 2020).
///
/// Rectified Adam: uses a variance-rectification term to automatically
/// switch between SGD-like updates (early training) and Adam updates.
/// No warmup scheduler needed.
pub struct RAdam {
    params: Vec<Variable>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    m: Vec<Option<crate::tensor::Tensor>>,
    v: Vec<Option<crate::tensor::Tensor>>,
    /// Per-param step counts, incremented only when the param has a grad
    /// (bias correction + rectification follow each param's own steps).
    steps: Vec<i64>,
}

impl RAdam {
    /// Create a new RAdam optimizer with default betas (0.9, 0.999), eps (1e-8),
    /// and no weight decay.
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        RAdam {
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

    /// Set L2 weight decay (default 0.0), added to the gradient.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 {
        self.lr
    }
}

impl Optimizer for RAdam {
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
        // Maximum length of the approximated SMA.
        let rho_inf = 2.0 / (1.0 - b2) - 1.0;
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
                let (p, m, v) = (
                    super::pick(&data, &at),
                    super::pick(&m, &at),
                    super::pick(&v, &at),
                );
                let rho_t = rho_inf - 2.0 * t * b2t / (1.0 - b2t);
                if rho_t > 5.0 {
                    // Variance is tractable: the rectified Adam update,
                    // p -= lr * rect * (m / (1 - b1^t)) * sqrt(1 - b2^t) / (sqrt(v) + eps)
                    let rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                        / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                        .sqrt();
                    // sqrt(v) + eps; the bias correction of v goes into
                    // the coefficient, which leaves eps unscaled as in
                    // PyTorch.
                    let denom = Tensor::foreach_sqrt(&v)?;
                    Tensor::foreach_add_scalar_(&denom, eps)?;
                    let coeff = -lr * rect * (1.0 - b2t).sqrt() / (1.0 - b1t);
                    Tensor::foreach_addcdiv_(&p, &m, &denom, coeff)?;
                } else {
                    // Variance is intractable: p -= lr * m / (1 - b1^t)
                    Tensor::foreach_add_list_(&p, &m, -lr / (1.0 - b1t))?;
                }
            }
            Ok(())
        })
    }

    fn reset_state(&mut self) {
        // Moment estimates back to fresh, step counts to 0 (rectification
        // schedule restarts). Lengths preserved for per-param indexing.
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

impl Stateful for RAdam {
    fn state_kind(&self) -> super::StateKind {
        super::StateKind::RAdam
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
        // Empty group table: RAdam has no group support yet, but the
        // slot keeps the payload shape uniform with grouped optimizers.
        super::write_groups(w, &[])?;
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let count = read_u32_le(r)? as usize;
        if count != self.params.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "RAdam: param count mismatch: checkpoint={} optimizer={}",
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
        let groups = super::read_groups(r, self.params.len(), "RAdam")?;
        if !groups.is_empty() {
            return Err(crate::tensor::TensorError::new(
                "RAdam: state file carries a group table, but this flodl's \
                 RAdam has no parameter-group support",
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
    fn radam_matches_a_reference_over_seven_steps() {
        // Values from a float64 transcription of the update rule. `late`
        // gets its first gradient at step 3, so from then on the two
        // parameters sit at different step counts in one batched call.
        // Rectification starts at a parameter's sixth step, so steps 6
        // and 7 run the rectified branch for `p` and the plain one for
        // `late` side by side.
        let dev = crate::tensor::test_device();
        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[1, 2], dev).unwrap(), "p");
        let late = Parameter::new(Tensor::from_f32(&[3.0, 4.0], &[1, 2], dev).unwrap(), "late");
        let mut opt = RAdam::new(&[p.clone(), late.clone()], 0.1);

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
                    assert_close(&values(&p), &[0.826_316, 1.757_895]);
                    assert_close(&values(&late), &[3.0, 4.0]);
                }
                5 => {
                    assert_close(&values(&p), &[0.589_126, 1.534_758]);
                    assert_close(&values(&late), &[2.4, 3.7]);
                }
                7 => {
                    assert_close(&values(&p), &[0.583_590, 1.532_632]);
                    assert_close(&values(&late), &[2.0, 3.5]);
                }
                _ => {}
            }
        }
        assert_eq!(opt.steps, vec![7, 5]);
    }

    #[test]
    fn radam_adds_eps_outside_the_bias_correction() {
        // PyTorch's rectified step divides by sqrt(v) + eps and corrects
        // the bias of v in the numerator. An eps large enough to matter
        // tells that apart from adding it to the corrected sqrt(v).
        let dev = crate::tensor::test_device();
        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[1, 2], dev).unwrap(), "p");
        let mut opt = RAdam::new(std::slice::from_ref(&p), 0.1);
        opt.eps = 0.1;
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
            opt.step().unwrap();
        }
        assert_close(&values(&p), &[0.586_938, 1.533_547]);
    }

    #[test]
    fn radam_weight_decay_leaves_the_gradient() {
        let dev = crate::tensor::test_device();
        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[1, 2], dev).unwrap(), "p");
        let mut opt = RAdam::new(std::slice::from_ref(&p), 0.1).weight_decay(0.1);
        set_grad(&p, &[1.0, 1.0]);
        opt.step().unwrap();
        assert_close(
            &p.variable.grad().unwrap().to_f32_vec().unwrap(),
            &[1.0, 1.0],
        );
        assert_ne!(values(&p), vec![1.0, 2.0]);
    }

    #[test]
    fn test_radam_state_file_roundtrip() {
        // Locks the Stateful impl added for RAdam: save after a step, load
        // into a freshly-constructed optimizer (different lr), and the
        // per-param steps + lr must round-trip through the .optim header.
        let dev = crate::tensor::test_device();
        let p = make_param("w", &[2]);
        let mut opt = RAdam::new(std::slice::from_ref(&p), 0.02);
        p.variable
            .set_grad(Tensor::from_f32(&[0.1, 0.2], &[2], dev).unwrap());
        opt.step().unwrap();

        let path = state_tmp("radam_roundtrip.optim");
        opt.save_state_to(&path).unwrap();

        let mut opt2 = RAdam::new(std::slice::from_ref(&p), 0.5);
        opt2.load_state_file(&path).unwrap();
        assert_eq!(opt2.steps, opt.steps);
        assert!((opt2.lr - 0.02).abs() < 1e-12);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_radam_reset_state_clears_moments_and_steps() {
        // reset_state (DiLoCo disposable-inner) must wipe the moment
        // estimates and per-param step counters.
        let dev = crate::tensor::test_device();
        let p = make_param("w", &[2]);
        let mut opt = RAdam::new(std::slice::from_ref(&p), 0.01);
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
    fn test_radam_steps() {
        let p = make_param("w", &[1]);
        let before = p.variable.data().item().unwrap();
        let mut opt = RAdam::new(std::slice::from_ref(&p), 0.01);
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
            "RAdam step should change parameter"
        );
    }

    #[test]
    fn test_radam_convergence_100_steps() {
        use crate::nn::{Linear, Module, loss::mse_loss};

        let dev = crate::tensor::test_device();
        let model = Linear::on_device(4, 1, dev).unwrap();
        // RAdam uses SGD-like updates for early steps (rho_t <= 5), so needs
        // more iterations and a slightly higher LR than vanilla Adam.
        let mut opt = RAdam::new(&model.parameters(), 0.05);

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
            "RAdam should converge: first={}, final={}",
            first_loss,
            final_loss
        );
    }
}
