pub mod parameter;
pub mod init;
pub mod linear;
pub mod activation;
pub mod loss;
pub mod optim;
pub mod clip;
pub mod scheduler;
pub mod dropout;
pub mod layernorm;
pub mod embedding;
pub mod grucell;
pub mod lstmcell;

pub use parameter::Parameter;
pub use linear::Linear;
pub use activation::{ReLU, Sigmoid, Tanh, GELU, SiLU};
pub use loss::{mse_loss, cross_entropy_loss, bce_with_logits_loss, l1_loss, smooth_l1_loss, kl_div_loss};
pub use optim::{Optimizer, SGD, Adam};
pub use clip::{clip_grad_norm, clip_grad_value};
pub use scheduler::{Scheduler, StepDecay, CosineScheduler, WarmupScheduler, PlateauScheduler};
pub use dropout::Dropout;
pub use layernorm::LayerNorm;
pub use embedding::Embedding;
pub use grucell::GRUCell;
pub use lstmcell::LSTMCell;

use std::collections::HashMap;

use crate::autograd::Variable;
use crate::tensor::Result;

/// The core module trait: forward pass + parameter access.
pub trait Module {
    fn forward(&self, input: &Variable) -> Result<Variable>;
    fn parameters(&self) -> Vec<Parameter>;
}

/// Module that can receive additional named inputs via graph Using().
pub trait NamedInputModule: Module {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable>;
}

/// Recursively collect parameters from a module and its sub-modules.
pub fn collect_parameters(modules: &[&dyn Module]) -> Vec<Parameter> {
    let mut params = Vec::new();
    for m in modules {
        params.extend(m.parameters());
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Variable;
    use crate::tensor::{Device, Tensor};

    fn from_f32(data: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_f32(data, shape, Device::CPU).unwrap()
    }

    #[test]
    fn test_linear_forward() {
        let model = Linear::new(3, 2).unwrap();

        // Set known weights for deterministic test
        // W = [[1,2,3],[4,5,6]] shape [2,3]
        model.weight.variable.set_data(
            from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        );
        model.bias.as_ref().unwrap().variable.set_data(
            from_f32(&[0.1, 0.2], &[2]),
        );

        // x = [[1, 1, 1]] shape [1, 3]
        let x = Variable::new(from_f32(&[1.0, 1.0, 1.0], &[1, 3]), false);
        let y = model.forward(&x).unwrap();

        // y = x @ W^T + b = [[1,1,1]] @ [[1,4],[2,5],[3,6]] + [0.1,0.2]
        // = [[6, 15]] + [[0.1, 0.2]] = [[6.1, 15.2]]
        let data = y.data().to_f32_vec().unwrap();
        assert!((data[0] - 6.1).abs() < 1e-5);
        assert!((data[1] - 15.2).abs() < 1e-5);
    }

    #[test]
    fn test_linear_backward() {
        let model = Linear::new(3, 2).unwrap();
        model.weight.variable.set_data(
            from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        );
        model.bias.as_ref().unwrap().variable.set_data(
            from_f32(&[0.0, 0.0], &[2]),
        );

        let x = Variable::new(from_f32(&[1.0, 1.0, 1.0], &[1, 3]), true);
        let y = model.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        // dL/dW = grad_out^T @ x
        // grad_out = [[1,1]], x = [[1,1,1]]
        // dL/dW = [[1],[1]] @ [[1,1,1]] = [[1,1,1],[1,1,1]]
        let gw = model.weight.variable.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(gw, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // dL/db = sum of grad_out along batch = [1, 1]
        let gb = model.bias.as_ref().unwrap().variable.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(gb, vec![1.0, 1.0]);

        // dL/dx = grad_out @ W = [[1,1]] @ [[1,2,3],[4,5,6]] = [[5,7,9]]
        let gx = x.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(gx, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mse_loss() {
        let pred = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), false);
        let target = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), false);
        let loss = mse_loss(&pred, &target).unwrap();
        assert!((loss.item().unwrap()).abs() < 1e-7);

        let pred2 = Variable::new(from_f32(&[2.0, 3.0, 4.0], &[3]), false);
        let loss2 = mse_loss(&pred2, &target).unwrap();
        // (1² + 1² + 1²) / 3 = 1.0
        assert!((loss2.item().unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sgd_step() {
        let model = Linear::new(2, 1).unwrap();
        model.weight.variable.set_data(from_f32(&[1.0, 1.0], &[1, 2]));
        model.bias.as_ref().unwrap().variable.set_data(from_f32(&[0.0], &[1]));

        let params = model.parameters();
        let mut optim = SGD::new(&params, 0.1, 0.0);

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let target = Variable::new(from_f32(&[5.0], &[1, 1]), false);

        // Forward + backward
        let pred = model.forward(&x).unwrap();
        let loss = mse_loss(&pred, &target).unwrap();
        let loss_before = loss.item().unwrap();
        loss.backward().unwrap();
        optim.step().unwrap();

        // Loss should decrease after one step
        optim.zero_grad();
        let pred2 = model.forward(&x).unwrap();
        let loss2 = mse_loss(&pred2, &target).unwrap();
        assert!(loss2.item().unwrap() < loss_before, "loss should decrease");
    }

    #[test]
    fn test_linear_regression_sgd() {
        // y = 2*x + 1
        let model = Linear::new(1, 1).unwrap();
        let params = model.parameters();
        let mut optim = SGD::new(&params, 0.01, 0.0);

        let x = Variable::new(
            from_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1]),
            false,
        );
        let target = Variable::new(
            from_f32(&[3.0, 5.0, 7.0, 9.0], &[4, 1]),
            false,
        );

        let mut last_loss = f64::MAX;
        for _ in 0..800 {
            optim.zero_grad();
            let pred = model.forward(&x).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();
            last_loss = loss.item().unwrap();
            loss.backward().unwrap();
            optim.step().unwrap();
        }

        assert!(
            last_loss < 0.01,
            "SGD should converge on linear regression, got loss={}",
            last_loss
        );
    }

    #[test]
    fn test_linear_regression_adam() {
        // y = 2*x + 1
        let model = Linear::new(1, 1).unwrap();
        let params = model.parameters();
        let mut optim = Adam::new(&params, 0.1);

        let x = Variable::new(
            from_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1]),
            false,
        );
        let target = Variable::new(
            from_f32(&[3.0, 5.0, 7.0, 9.0], &[4, 1]),
            false,
        );

        let mut last_loss = f64::MAX;
        for _ in 0..500 {
            optim.zero_grad();
            let pred = model.forward(&x).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();
            last_loss = loss.item().unwrap();
            loss.backward().unwrap();
            optim.step().unwrap();
        }

        assert!(
            last_loss < 0.02,
            "Adam should converge on linear regression, got loss={}",
            last_loss
        );
    }

    #[test]
    fn test_relu_module() {
        let relu = ReLU::new();
        let x = Variable::new(from_f32(&[1.0, -1.0, 2.0, -2.0], &[4]), false);
        let y = relu.forward(&x).unwrap();
        assert_eq!(y.data().to_f32_vec().unwrap(), vec![1.0, 0.0, 2.0, 0.0]);
        assert!(relu.parameters().is_empty());
    }

    #[test]
    fn test_collect_parameters() {
        let l1 = Linear::new(3, 4).unwrap();
        let l2 = Linear::new(4, 2).unwrap();
        let params = collect_parameters(&[&l1, &l2]);
        // l1: weight + bias = 2, l2: weight + bias = 2
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_sgd_momentum() {
        let model = Linear::new(1, 1).unwrap();
        let params = model.parameters();
        let mut optim = SGD::new(&params, 0.01, 0.9);

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1]), false);
        let target = Variable::new(from_f32(&[3.0, 5.0, 7.0, 9.0], &[4, 1]), false);

        let mut last_loss = f64::MAX;
        for _ in 0..200 {
            optim.zero_grad();
            let pred = model.forward(&x).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();
            last_loss = loss.item().unwrap();
            loss.backward().unwrap();
            optim.step().unwrap();
        }

        assert!(
            last_loss < 0.01,
            "SGD with momentum should converge, got loss={}",
            last_loss
        );
    }

    // --- Loss function tests ---

    #[test]
    fn test_cross_entropy_loss() {
        // 2 classes, batch of 2
        // Logits: [[2.0, 1.0], [1.0, 3.0]]
        // One-hot targets: [[1, 0], [0, 1]]
        let pred = Variable::new(
            from_f32(&[2.0, 1.0, 1.0, 3.0], &[2, 2]),
            true,
        );
        let target = Variable::new(
            from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]),
            false,
        );
        let loss = cross_entropy_loss(&pred, &target).unwrap();
        let val = loss.item().unwrap();

        // Manual: log_softmax([2,1]) = [2-log(e²+e), 1-log(e²+e)]
        // e² + e ≈ 10.107
        // log_softmax([2,1]) ≈ [2-2.313, 1-2.313] = [-0.313, -1.313]
        // For sample 0: target=[1,0] → -(-0.313) = 0.313
        // log_softmax([1,3]) = [1-log(e+e³), 3-log(e+e³)]
        // e + e³ ≈ 22.804
        // log_softmax([1,3]) ≈ [1-3.127, 3-3.127] = [-2.127, -0.127]
        // For sample 1: target=[0,1] → -(-0.127) = 0.127
        // mean = (0.313 + 0.127) / 2 = 0.220
        assert!(val > 0.0, "cross entropy should be positive");
        assert!((val - 0.22).abs() < 0.02, "expected ~0.22, got {}", val);

        // Test backward
        loss.backward().unwrap();
        assert!(pred.grad().is_some());
    }

    #[test]
    fn test_cross_entropy_converges() {
        // Train a Linear to classify 2 classes
        let model = Linear::new(2, 2).unwrap();
        let params = model.parameters();
        let mut optim = SGD::new(&params, 0.1, 0.0);

        // Data: class 0 = [1, 0], class 1 = [0, 1]
        let x = Variable::new(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]), false);
        let target = Variable::new(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]), false);

        let mut last_loss = f64::MAX;
        for _ in 0..200 {
            optim.zero_grad();
            let pred = model.forward(&x).unwrap();
            let loss = cross_entropy_loss(&pred, &target).unwrap();
            last_loss = loss.item().unwrap();
            loss.backward().unwrap();
            optim.step().unwrap();
        }

        assert!(last_loss < 0.1, "cross entropy should converge, got {}", last_loss);
    }

    #[test]
    fn test_bce_with_logits_loss() {
        // Logits = 0 → sigmoid(0) = 0.5, target = 1
        // BCE = -log(0.5) = 0.693
        let pred = Variable::new(from_f32(&[0.0], &[1]), true);
        let target = Variable::new(from_f32(&[1.0], &[1]), false);
        let loss = bce_with_logits_loss(&pred, &target).unwrap();
        let val = loss.item().unwrap();
        assert!(
            (val - 0.693).abs() < 0.01,
            "expected ~0.693, got {}",
            val
        );

        // Large positive logit, target=1 → loss ≈ 0
        let pred2 = Variable::new(from_f32(&[10.0], &[1]), false);
        let loss2 = bce_with_logits_loss(&pred2, &target).unwrap();
        assert!(loss2.item().unwrap() < 0.001);

        // Backward
        loss.backward().unwrap();
        assert!(pred.grad().is_some());
    }

    #[test]
    fn test_l1_loss() {
        let pred = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
        let target = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), false);
        let loss = l1_loss(&pred, &target).unwrap();
        assert!((loss.item().unwrap()).abs() < 1e-6);

        let pred2 = Variable::new(from_f32(&[2.0, 4.0, 6.0], &[3]), true);
        let loss2 = l1_loss(&pred2, &target).unwrap();
        // |2-1| + |4-2| + |6-3| = 1+2+3 = 6, mean = 2.0
        assert!((loss2.item().unwrap() - 2.0).abs() < 1e-5);

        loss2.backward().unwrap();
        assert!(pred2.grad().is_some());
    }

    #[test]
    fn test_smooth_l1_loss() {
        // Small diff (within beta): 0.5 * x² / beta
        let pred = Variable::new(from_f32(&[1.5], &[1]), true);
        let target = Variable::new(from_f32(&[1.0], &[1]), false);
        let loss = smooth_l1_loss(&pred, &target, 1.0).unwrap();
        // |0.5| < 1.0 → 0.5 * 0.25 / 1.0 = 0.125
        assert!((loss.item().unwrap() - 0.125).abs() < 1e-5, "got {}", loss.item().unwrap());

        // Large diff (outside beta): |x| - 0.5*beta
        let pred2 = Variable::new(from_f32(&[3.0], &[1]), true);
        let loss2 = smooth_l1_loss(&pred2, &target, 1.0).unwrap();
        // |2.0| >= 1.0 → 2.0 - 0.5 = 1.5
        assert!((loss2.item().unwrap() - 1.5).abs() < 1e-5);

        loss2.backward().unwrap();
        assert!(pred2.grad().is_some());
    }

    #[test]
    fn test_kl_div_loss() {
        // KL(P || Q) where Q = log_probs, P = probs
        // Uniform distribution: KL = 0
        let log_probs = Variable::new(
            from_f32(&[-0.693, -0.693, -0.693, -0.693], &[2, 2]),
            true,
        );
        let probs = Variable::new(
            from_f32(&[0.5, 0.5, 0.5, 0.5], &[2, 2]),
            false,
        );
        let loss = kl_div_loss(&log_probs, &probs).unwrap();
        // When Q ≈ P, KL ≈ 0
        assert!(loss.item().unwrap().abs() < 0.01, "KL should be ~0, got {}", loss.item().unwrap());

        loss.backward().unwrap();
        assert!(log_probs.grad().is_some());
    }

    // --- Gradient clipping tests ---

    #[test]
    fn test_clip_grad_norm() {
        let model = Linear::new(2, 1).unwrap();
        let params = model.parameters();

        let x = Variable::new(from_f32(&[10.0, 20.0], &[1, 2]), false);
        let target = Variable::new(from_f32(&[0.0], &[1, 1]), false);
        let pred = model.forward(&x).unwrap();
        let loss = mse_loss(&pred, &target).unwrap();
        loss.backward().unwrap();

        let norm_before = clip_grad_norm(&params, 1.0).unwrap();
        assert!(norm_before > 1.0, "large input should produce large gradients");

        // After clipping, total norm should be <= max_norm
        let mut total_sq = 0.0f64;
        for p in &params {
            if let Some(g) = p.variable.grad() {
                for &v in &g.to_f32_vec().unwrap() {
                    total_sq += (v as f64) * (v as f64);
                }
            }
        }
        let clipped_norm = total_sq.sqrt();
        assert!(
            clipped_norm <= 1.0 + 1e-5,
            "clipped norm should be <= 1.0, got {}",
            clipped_norm
        );
    }

    #[test]
    fn test_clip_grad_value() {
        let model = Linear::new(2, 1).unwrap();
        let params = model.parameters();

        let x = Variable::new(from_f32(&[10.0, 20.0], &[1, 2]), false);
        let target = Variable::new(from_f32(&[0.0], &[1, 1]), false);
        let pred = model.forward(&x).unwrap();
        let loss = mse_loss(&pred, &target).unwrap();
        loss.backward().unwrap();

        clip_grad_value(&params, 0.5).unwrap();

        for p in &params {
            if let Some(g) = p.variable.grad() {
                for &v in &g.to_f32_vec().unwrap() {
                    assert!(
                        v.abs() <= 0.5 + 1e-6,
                        "all grads should be clamped to [-0.5, 0.5], got {}",
                        v
                    );
                }
            }
        }
    }

    // --- Scheduler tests ---

    #[test]
    fn test_step_decay_scheduler() {
        let model = Linear::new(1, 1).unwrap();
        let params = model.parameters();
        let mut optim = SGD::new(&params, 0.1, 0.0);

        let mut sched = StepDecay::new(&mut optim, 0.1, 3, 0.5);
        assert!((sched.lr() - 0.1).abs() < 1e-10);

        sched.step(); // step 1
        sched.step(); // step 2
        assert!((sched.lr() - 0.1).abs() < 1e-10); // no decay yet

        sched.step(); // step 3 → decay
        assert!((sched.lr() - 0.05).abs() < 1e-10);

        sched.step(); // step 4
        sched.step(); // step 5
        sched.step(); // step 6 → decay again
        assert!((sched.lr() - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_scheduler() {
        let model = Linear::new(1, 1).unwrap();
        let params = model.parameters();
        let mut optim = SGD::new(&params, 0.1, 0.0);

        let mut sched = CosineScheduler::new(&mut optim, 0.1, 0.001, 100);

        // At step 0, lr = base_lr
        assert!((sched.lr() - 0.1).abs() < 1e-10);

        // At step 50, lr should be roughly midpoint
        for _ in 0..50 {
            sched.step();
        }
        let mid_lr = sched.lr();
        assert!(mid_lr > 0.001 && mid_lr < 0.1, "mid lr={}", mid_lr);

        // At step 100, lr should be min_lr
        for _ in 0..50 {
            sched.step();
        }
        assert!((sched.lr() - 0.001).abs() < 1e-5, "end lr={}", sched.lr());
    }

    #[test]
    fn test_plateau_scheduler() {
        let model = Linear::new(1, 1).unwrap();
        let params = model.parameters();
        let mut optim = SGD::new(&params, 0.1, 0.0);

        let mut sched = PlateauScheduler::new(&mut optim, 0.1, 3, 0.5, 0.001);

        // Improving metrics: no decay
        sched.observe(1.0);
        sched.observe(0.9);
        sched.observe(0.8);
        assert!((sched.lr() - 0.1).abs() < 1e-10);

        // Stagnating: patience=3
        sched.observe(0.81); // wait=1
        sched.observe(0.82); // wait=2
        sched.observe(0.83); // wait=3 → decay
        assert!((sched.lr() - 0.05).abs() < 1e-10);
    }

    // --- GELU / SiLU tests ---

    #[test]
    fn test_gelu() {
        let gelu = GELU::new();
        let x = Variable::new(from_f32(&[0.0, 1.0, -1.0], &[3]), true);
        let y = gelu.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        assert!(data[0].abs() < 0.01, "GELU(0)={}", data[0]);
        assert!((data[1] - 0.841).abs() < 0.01, "GELU(1)={}", data[1]);
        assert!((data[2] - (-0.159)).abs() < 0.01, "GELU(-1)={}", data[2]);

        // Backward
        let loss = y.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_silu() {
        let silu = SiLU::new();
        let x = Variable::new(from_f32(&[0.0, 2.0, -2.0], &[3]), true);
        let y = silu.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // SiLU(0) = 0, SiLU(2) = 2*sigmoid(2) ≈ 1.762, SiLU(-2) = -2*sigmoid(-2) ≈ -0.238
        assert!(data[0].abs() < 0.01);
        assert!((data[1] - 1.762).abs() < 0.02, "SiLU(2)={}", data[1]);

        let loss = y.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());
    }

    // --- Dropout test ---

    #[test]
    fn test_dropout() {
        let drop = Dropout::new(0.5);

        // Training mode: some values should be zeroed
        let x = Variable::new(from_f32(&[1.0; 100], &[10, 10]), false);
        let y = drop.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        let zeros = data.iter().filter(|&&v| v == 0.0).count();
        // With p=0.5, roughly half should be zero (very approximate)
        assert!(zeros > 10 && zeros < 90, "zeros={} of 100", zeros);
        // Non-zero values should be scaled by 1/(1-0.5) = 2
        for &v in &data {
            if v != 0.0 {
                assert!((v - 2.0).abs() < 1e-5, "scaled value should be 2.0, got {}", v);
            }
        }

        // Eval mode: identity
        drop.set_training(false);
        let y_eval = drop.forward(&x).unwrap();
        let eval_data = y_eval.data().to_f32_vec().unwrap();
        assert!(eval_data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    // --- LayerNorm test ---

    #[test]
    fn test_layernorm() {
        let ln = LayerNorm::new(4).unwrap();
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]), true);
        let y = ln.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        assert_eq!(y.shape(), vec![1, 4]);

        // After normalization, mean ≈ 0, std ≈ 1 (before gamma/beta)
        // With gamma=1, beta=0 (defaults), output should be normalized
        let mean: f32 = data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 0.1, "mean should be ~0, got {}", mean);

        let loss = y.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());
        assert_eq!(ln.parameters().len(), 2); // weight + bias
    }

    // --- Embedding test ---

    #[test]
    fn test_embedding() {
        let emb = Embedding::new(5, 3).unwrap();

        // Look up indices [0, 2, 4]
        let indices = Variable::new(from_f32(&[0.0, 2.0, 4.0], &[3]), false);
        let y = emb.forward(&indices).unwrap();
        assert_eq!(y.shape(), vec![3, 3]); // 3 tokens × 3 dims

        // Same index should give same embedding
        let idx_same = Variable::new(from_f32(&[1.0, 1.0], &[2]), false);
        let y2 = emb.forward(&idx_same).unwrap();
        let data = y2.data().to_f32_vec().unwrap();
        assert_eq!(&data[0..3], &data[3..6]);

        assert_eq!(emb.parameters().len(), 1); // weight only
    }

    #[test]
    fn test_embedding_backward() {
        let emb = Embedding::new(5, 3).unwrap();
        let indices = Variable::new(from_f32(&[0.0, 2.0], &[2]), false);
        let y = emb.forward(&indices).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let grad = emb.weight.variable.grad().unwrap();
        let grad_data = grad.to_f32_vec().unwrap();
        // Rows 0 and 2 should have gradient, others zero
        assert!(grad_data[0..3].iter().all(|&v| (v - 1.0).abs() < 1e-5)); // row 0
        assert!(grad_data[3..6].iter().all(|&v| v.abs() < 1e-5)); // row 1 (unused)
        assert!(grad_data[6..9].iter().all(|&v| (v - 1.0).abs() < 1e-5)); // row 2
    }

    // --- GRUCell test ---

    #[test]
    fn test_grucell() {
        let gru = GRUCell::new(4, 3).unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]), true);
        let h1 = gru.forward_step(&x, None).unwrap();
        assert_eq!(h1.shape(), vec![1, 3]);

        // Second step with previous hidden state
        let h2 = gru.forward_step(&x, Some(&h1)).unwrap();
        assert_eq!(h2.shape(), vec![1, 3]);

        // Backward
        let loss = h2.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());

        // Should have 6 Linear modules' params
        // 3 input (w+b=2 each) + 3 hidden (w only=1 each) = 9
        assert_eq!(gru.parameters().len(), 9);
    }

    // --- LSTMCell test ---

    #[test]
    fn test_lstmcell() {
        let lstm = LSTMCell::new(4, 3).unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]), true);
        let state1 = lstm.forward_step(&x, None).unwrap();
        assert_eq!(state1.shape(), vec![1, 6]); // packed: [batch, 2*hidden]

        // Unpack h
        let h1 = state1.narrow(1, 0, 3).unwrap();
        assert_eq!(h1.shape(), vec![1, 3]);

        // Second step
        let state2 = lstm.forward_step(&x, Some(&state1)).unwrap();
        assert_eq!(state2.shape(), vec![1, 6]);

        // Backward
        let loss = state2.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());

        // 4 input Linear (w+b=2 each) + 4 hidden Linear (w only=1 each) = 12
        assert_eq!(lstm.parameters().len(), 12);
    }
}
