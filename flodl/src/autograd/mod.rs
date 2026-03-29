//! Reverse-mode automatic differentiation backed by libtorch.
//!
//! Variables wrap tensors with gradient tracking. When `requires_grad` is
//! true, libtorch's native autograd engine records operations and computes
//! gradients. Calling `backward()` delegates to libtorch's C++ backward
//! engine — no Rust-side graph walking.
//!
//! ```ignore
//! let x = Variable::new(tensor_x, true);
//! let w = Variable::new(tensor_w, true);
//! let loss = x.matmul(&w)?.sum()?;
//! loss.backward()?;
//! println!("{:?}", w.grad()); // gradient of loss w.r.t. w
//! ```

mod variable;
mod ops;
mod context;

pub use variable::Variable;
pub use context::{no_grad, is_grad_enabled, NoGradGuard};
pub use ops::{linear, gru_cell, lstm_cell, layer_norm, conv2d, conv1d, conv_transpose2d, conv_transpose1d, group_norm, max_pool2d, avg_pool2d, adaptive_avg_pool2d, grid_sample, embedding_bag};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, Device};

    fn from_f32(data: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_f32(data, shape, crate::tensor::test_device()).unwrap()
    }

    #[test]
    fn test_simple_gradient() {
        // y = 2*x, dy/dx = 2
        let x = Variable::new(from_f32(&[3.0], &[1]), true);
        let two = Variable::new(from_f32(&[2.0], &[1]), false);
        let y = x.mul(&two).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap();
        assert!((grad.item().unwrap() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_add_gradient() {
        let a = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
        let b = Variable::new(from_f32(&[4.0, 5.0, 6.0], &[3]), true);
        let c = a.add(&b).unwrap().sum().unwrap();
        c.backward().unwrap();

        let ga = a.grad().unwrap().to_f32_vec().unwrap();
        let gb = b.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(ga, vec![1.0, 1.0, 1.0]);
        assert_eq!(gb, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mul_gradient() {
        let x = Variable::new(from_f32(&[2.0, 3.0], &[2]), true);
        let y = Variable::new(from_f32(&[4.0, 5.0], &[2]), true);
        let z = x.mul(&y).unwrap().sum().unwrap();
        z.backward().unwrap();

        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![4.0, 5.0]);
        assert_eq!(y.grad().unwrap().to_f32_vec().unwrap(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_chain_rule() {
        let x = Variable::new(from_f32(&[1.0, -2.0, 3.0], &[3]), true);
        let w = Variable::new(from_f32(&[2.0, 2.0, 2.0], &[3]), true);
        let b = Variable::new(from_f32(&[0.0, 5.0, 0.0], &[3]), false);

        let y = x.mul(&w).unwrap()
            .add(&b).unwrap()
            .relu().unwrap()
            .sum().unwrap();
        assert!((y.item().unwrap() - 9.0).abs() < 1e-5);

        y.backward().unwrap();

        let gx = x.grad().unwrap().to_f32_vec().unwrap();
        let gw = w.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(gx, vec![2.0, 2.0, 2.0]);
        assert_eq!(gw, vec![1.0, -2.0, 3.0]);
    }

    #[test]
    fn test_matmul_gradient() {
        let a = Variable::new(
            from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let b = Variable::new(
            from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]), true);
        let c = a.matmul(&b).unwrap().sum().unwrap();
        c.backward().unwrap();

        let ga = a.grad().unwrap().to_f32_vec().unwrap();
        let gb = b.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(ga, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(gb, vec![4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn test_sigmoid_gradient() {
        let x = Variable::new(from_f32(&[0.0], &[1]), true);
        let y = x.sigmoid().unwrap().sum().unwrap();
        y.backward().unwrap();
        assert!((x.grad().unwrap().item().unwrap() - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_relu_gradient() {
        let x = Variable::new(from_f32(&[1.0, -1.0, 2.0, -2.0], &[4]), true);
        let y = x.relu().unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_exp_log_gradient() {
        let x = Variable::new(from_f32(&[2.0], &[1]), true);
        let y = x.log().unwrap().exp().unwrap().sum().unwrap();
        y.backward().unwrap();
        assert!((x.grad().unwrap().item().unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_no_grad() {
        let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let y = no_grad(|| x.mul_scalar(3.0).unwrap());
        assert!(!y.requires_grad());
    }

    #[test]
    fn test_detach() {
        let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let y = x.mul_scalar(2.0).unwrap();
        let z = y.detach();
        // Detached variable should not require grad
        assert!(!z.requires_grad());
        // Operations on detached variable don't track gradients
        let w = z.mul_scalar(3.0).unwrap();
        assert!(!w.requires_grad());
        // x should have no gradient (no backward was called through x)
        assert!(x.grad().is_none());
    }

    #[test]
    fn test_gradient_accumulation() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
        let y = Variable::new(from_f32(&[1.0, 1.0, 1.0], &[3]), false);

        let z1 = x.add(&y).unwrap().sum().unwrap();
        z1.backward().unwrap();
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0, 1.0]);

        // Second backward accumulates
        let z2 = x.mul_scalar(2.0).unwrap().sum().unwrap();
        z2.backward().unwrap();
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_zero_grad() {
        let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let y = x.mul_scalar(3.0).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![3.0, 3.0]);

        x.zero_grad();
        // After zero_grad, grad should be zero (libtorch zeros it, doesn't remove it)
        let g = x.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(g, vec![0.0, 0.0]);
    }

    #[test]
    fn test_sub_gradient() {
        let a = Variable::new(from_f32(&[5.0, 6.0], &[2]), true);
        let b = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let c = a.sub(&b).unwrap().sum().unwrap();
        c.backward().unwrap();

        assert_eq!(a.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0]);
        assert_eq!(b.grad().unwrap().to_f32_vec().unwrap(), vec![-1.0, -1.0]);
    }

    #[test]
    fn test_tanh_gradient() {
        let x = Variable::new(from_f32(&[0.0], &[1]), true);
        let y = x.tanh().unwrap().sum().unwrap();
        y.backward().unwrap();
        assert!((x.grad().unwrap().item().unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_neg_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let y = x.neg().unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![-1.0, -1.0]);
    }

    #[test]
    fn test_numerical_gradient() {
        let eps = 1e-4;
        let x_data = vec![1.0_f32, 2.0, -3.0];

        let x = Variable::new(from_f32(&x_data, &[3]), true);
        let y = x.mul(&x).unwrap().sum().unwrap();
        y.backward().unwrap();
        let analytical = x.grad().unwrap().to_f32_vec().unwrap();

        for i in 0..3 {
            let mut x_plus = x_data.clone();
            let mut x_minus = x_data.clone();
            x_plus[i] += eps as f32;
            x_minus[i] -= eps as f32;

            let fp = from_f32(&x_plus, &[3]).mul(&from_f32(&x_plus, &[3]))
                .unwrap().sum().unwrap().item().unwrap();
            let fm = from_f32(&x_minus, &[3]).mul(&from_f32(&x_minus, &[3]))
                .unwrap().sum().unwrap().item().unwrap();

            let numerical = (fp - fm) / (2.0 * eps);
            assert!(
                (analytical[i] as f64 - numerical).abs() < 0.01,
                "grad mismatch at {}: analytical={}, numerical={}",
                i, analytical[i], numerical
            );
        }
    }

    #[test]
    fn test_diamond_graph() {
        let x = Variable::new(from_f32(&[3.0], &[1]), true);
        let y1 = x.mul_scalar(2.0).unwrap();
        let y2 = x.mul_scalar(3.0).unwrap();
        let z = y1.add(&y2).unwrap().sum().unwrap();
        z.backward().unwrap();
        assert!((x.grad().unwrap().item().unwrap() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_drop_frees_backward_graph() {
        let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        {
            let y = x.mul_scalar(2.0).unwrap().relu().unwrap().sum().unwrap();
            y.backward().unwrap();
        }
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![2.0, 2.0]);
    }

    #[test]
    fn test_sin_cos_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let y = x.sin().unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - 1.0_f32.cos()).abs() < 1e-5);
        assert!((grad[1] - 2.0_f32.cos()).abs() < 1e-5);

        x.zero_grad();
        let y2 = x.cos().unwrap().sum().unwrap();
        y2.backward().unwrap();
        let grad2 = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad2[0] - (-1.0_f32.sin())).abs() < 1e-5);
        assert!((grad2[1] - (-2.0_f32.sin())).abs() < 1e-5);
    }

    #[test]
    fn test_reciprocal_gradient() {
        let x = Variable::new(from_f32(&[2.0, 4.0], &[2]), true);
        let y = x.reciprocal().unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - (-0.25)).abs() < 1e-5);
        assert!((grad[1] - (-0.0625)).abs() < 1e-5);
    }

    #[test]
    fn test_var_std_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
        let v = x.var().unwrap();
        assert!((v.data().item().unwrap() - 1.0).abs() < 1e-5);
        v.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - (-1.0)).abs() < 1e-5);
        assert!((grad[1] - 0.0).abs() < 1e-5);
        assert!((grad[2] - 1.0).abs() < 1e-5);

        x.zero_grad();
        let s = x.std().unwrap();
        assert!((s.data().item().unwrap() - 1.0).abs() < 1e-5);
        s.backward().unwrap();
        let grad2 = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad2[0] - (-0.5)).abs() < 1e-4);
        assert!((grad2[1] - 0.0).abs() < 1e-4);
        assert!((grad2[2] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_var_dim_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let v = x.var_dim(1, false).unwrap().sum().unwrap();
        v.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - (-1.0)).abs() < 1e-5);
        assert!((grad[1] - 1.0).abs() < 1e-5);
        assert!((grad[2] - (-1.0)).abs() < 1e-5);
        assert!((grad[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gather_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let idx = Tensor::from_i64(&[0, 0, 1, 0], &[2, 2], crate::tensor::test_device()).unwrap();
        let y = x.gather(1, &idx).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - 2.0).abs() < 1e-5);
        assert!((grad[1] - 0.0).abs() < 1e-5);
        assert!((grad[2] - 1.0).abs() < 1e-5);
        assert!((grad[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_topk_gradient() {
        let x = Variable::new(from_f32(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]), true);
        let (values, _indices) = x.topk(2, 0, true, true).unwrap();
        let y = values.sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - 0.0).abs() < 1e-5);
        assert!((grad[1] - 0.0).abs() < 1e-5);
        assert!((grad[2] - 1.0).abs() < 1e-5);
        assert!((grad[3] - 0.0).abs() < 1e-5);
        assert!((grad[4] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_repeat_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let y = x.repeat(&[3]).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - 3.0).abs() < 1e-5);
        assert!((grad[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_pad_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
        let y = x.pad(&[1, 1], 0.0).unwrap();
        assert_eq!(y.data().shape(), vec![5]);
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(grad, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_chunk_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]), true);
        let chunks = x.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        let c0 = chunks[0].mul_scalar(2.0).unwrap();
        let c1 = chunks[1].mul_scalar(3.0).unwrap();
        let loss = c0.sum().unwrap().add(&c1.sum().unwrap()).unwrap().add(&chunks[2].sum().unwrap()).unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(grad, vec![2.0, 2.0, 3.0, 3.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sort_gradient() {
        let x = Variable::new(from_f32(&[3.0, 1.0, 2.0], &[3]), true);
        let (sorted, _indices) = x.sort(0, false).unwrap();
        let weights = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), false);
        let loss = sorted.mul(&weights).unwrap().sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - 3.0).abs() < 1e-5);
        assert!((grad[1] - 1.0).abs() < 1e-5);
        assert!((grad[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_div_gradient() {
        let a = Variable::new(from_f32(&[6.0], &[1]), true);
        let b = Variable::new(from_f32(&[3.0], &[1]), true);
        let y = a.div(&b).unwrap().sum().unwrap();
        y.backward().unwrap();

        assert!((a.grad().unwrap().item().unwrap() - 1.0 / 3.0).abs() < 1e-5);
        assert!((b.grad().unwrap().item().unwrap() - (-6.0 / 9.0)).abs() < 1e-5);
    }

    #[test]
    fn test_add_scalar_div_scalar_gradient() {
        let x = Variable::new(from_f32(&[2.0, 4.0], &[2]), true);
        let y = x.add_scalar(10.0).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0]);

        x.zero_grad();
        let y2 = x.div_scalar(4.0).unwrap().sum().unwrap();
        y2.backward().unwrap();
        let g = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((g[0] - 0.25).abs() < 1e-5);
        assert!((g[1] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_sqrt_gradient() {
        let x = Variable::new(from_f32(&[4.0, 9.0], &[2]), true);
        let y = x.sqrt().unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - 0.25).abs() < 1e-5);
        assert!((grad[1] - 1.0 / 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_abs_gradient() {
        let x = Variable::new(from_f32(&[-3.0, 5.0], &[2]), true);
        let y = x.abs().unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - (-1.0)).abs() < 1e-4);
        assert!((grad[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_pow_scalar_gradient() {
        let x = Variable::new(from_f32(&[2.0, 3.0], &[2]), true);
        let y = x.pow_scalar(3.0).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - 12.0).abs() < 1e-4);
        assert!((grad[1] - 27.0).abs() < 1e-4);
    }

    #[test]
    fn test_clamp_gradient() {
        let x = Variable::new(from_f32(&[-1.0, 0.5, 2.0], &[3]), true);
        let y = x.clamp(0.0, 1.0).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - 0.0).abs() < 1e-5);
        assert!((grad[1] - 1.0).abs() < 1e-5);
        assert!((grad[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]), true);
        let y = x.mean().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        for &g in &grad {
            assert!((g - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sum_dim_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let y = x.sum_dim(1, false).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mean_dim_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let y = x.mean_dim(1, false).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        for &g in &grad {
            assert!((g - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_max_dim_gradient() {
        // [[1, 4], [3, 2]] — max along dim=1 gives [4, 3]
        let x = Variable::new(from_f32(&[1.0, 4.0, 3.0, 2.0], &[2, 2]), true);
        let y = x.max_dim(1, false).unwrap().sum().unwrap();
        y.backward().unwrap();

        // Gradient flows only to argmax positions: (0,1) and (1,0)
        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(grad, vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_max_dim_keepdim_gradient() {
        let x = Variable::new(from_f32(&[1.0, 4.0, 3.0, 2.0], &[2, 2]), true);
        let y = x.max_dim(1, true).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(grad, vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_min_dim_gradient() {
        // [[1, 4], [3, 2]] — min along dim=1 gives [1, 2]
        let x = Variable::new(from_f32(&[1.0, 4.0, 3.0, 2.0], &[2, 2]), true);
        let y = x.min_dim(1, false).unwrap().sum().unwrap();
        y.backward().unwrap();

        // Gradient flows only to argmin positions: (0,0) and (1,1)
        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(grad, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_softmax_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
        let w = Variable::new(from_f32(&[1.0, 0.0, 0.0], &[3]), false);
        let y = x.softmax(0).unwrap().mul(&w).unwrap().sum().unwrap();
        y.backward().unwrap();

        let eps = 1e-4;
        let x_data = vec![1.0_f32, 2.0, 3.0];
        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        for i in 0..3 {
            let mut xp = x_data.clone();
            let mut xm = x_data.clone();
            xp[i] += eps as f32;
            xm[i] -= eps as f32;

            let fp = from_f32(&xp, &[3]).softmax(0).unwrap().to_f32_vec().unwrap()[0] as f64;
            let fm = from_f32(&xm, &[3]).softmax(0).unwrap().to_f32_vec().unwrap()[0] as f64;
            let numerical = (fp - fm) / (2.0 * eps);
            assert!(
                (grad[i] as f64 - numerical).abs() < 0.01,
                "softmax grad[{}]: analytical={}, numerical={}", i, grad[i], numerical
            );
        }
    }

    #[test]
    fn test_log_softmax_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
        let w = Variable::new(from_f32(&[1.0, 0.0, 0.0], &[3]), false);
        let y = x.log_softmax(0).unwrap().mul(&w).unwrap().sum().unwrap();
        y.backward().unwrap();

        let eps = 1e-4;
        let x_data = vec![1.0_f32, 2.0, 3.0];
        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        for i in 0..3 {
            let mut xp = x_data.clone();
            let mut xm = x_data.clone();
            xp[i] += eps as f32;
            xm[i] -= eps as f32;
            let fp = from_f32(&xp, &[3]).log_softmax(0).unwrap().to_f32_vec().unwrap()[0] as f64;
            let fm = from_f32(&xm, &[3]).log_softmax(0).unwrap().to_f32_vec().unwrap()[0] as f64;
            let numerical = (fp - fm) / (2.0 * eps);
            assert!(
                (grad[i] as f64 - numerical).abs() < 0.01,
                "log_softmax grad[{}]: analytical={}, numerical={}", i, grad[i], numerical
            );
        }
    }

    #[test]
    fn test_gelu_silu_gradient() {
        let x_data = vec![0.5_f32, -0.5, 1.0];
        let x = Variable::new(from_f32(&x_data, &[3]), true);
        let y = x.gelu().unwrap().sum().unwrap();
        y.backward().unwrap();
        let grad = x.grad().unwrap().to_f32_vec().unwrap();

        let eps = 1e-4;
        for i in 0..3 {
            let mut xp = x_data.clone();
            let mut xm = x_data.clone();
            xp[i] += eps as f32;
            xm[i] -= eps as f32;
            let fp: f64 = from_f32(&xp, &[3]).gelu().unwrap().sum().unwrap().item().unwrap();
            let fm: f64 = from_f32(&xm, &[3]).gelu().unwrap().sum().unwrap().item().unwrap();
            assert!((grad[i] as f64 - (fp - fm) / (2.0 * eps)).abs() < 0.01);
        }

        x.zero_grad();
        let y2 = x.silu().unwrap().sum().unwrap();
        y2.backward().unwrap();
        let grad2 = x.grad().unwrap().to_f32_vec().unwrap();
        for i in 0..3 {
            let mut xp = x_data.clone();
            let mut xm = x_data.clone();
            xp[i] += eps as f32;
            xm[i] -= eps as f32;
            let fp: f64 = from_f32(&xp, &[3]).silu().unwrap().sum().unwrap().item().unwrap();
            let fm: f64 = from_f32(&xm, &[3]).silu().unwrap().sum().unwrap().item().unwrap();
            assert!((grad2[i] as f64 - (fp - fm) / (2.0 * eps)).abs() < 0.01);
        }
    }

    #[test]
    fn test_reshape_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let y = x.reshape(&[4]).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().shape(), vec![2, 2]);
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![1.0; 4]);
    }

    #[test]
    fn test_transpose_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), true);
        let w = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]), false);
        let y = x.transpose(0, 1).unwrap().mul(&w).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().shape(), vec![2, 3]);
    }

    #[test]
    fn test_narrow_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]), true);
        let y = x.narrow(0, 1, 3).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_select_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), true);
        let y = x.select(0, 1).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_expand_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), true);
        let y = x.expand(&[4, 3]).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_permute_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), true);
        let w = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]), false);
        let y = x.permute(&[1, 0]).unwrap().mul(&w).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().shape(), vec![2, 3]);
    }

    #[test]
    fn test_squeeze_unsqueeze_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), true);
        let y = x.squeeze(0).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().shape(), vec![1, 3]);

        x.zero_grad();
        let y2 = x.unsqueeze(2).unwrap().sum().unwrap();
        y2.backward().unwrap();
        assert_eq!(x.grad().unwrap().shape(), vec![1, 3]);
    }

    #[test]
    fn test_cat_gradient() {
        let a = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let b = Variable::new(from_f32(&[3.0, 4.0, 5.0], &[3]), true);
        let y = a.cat(&b, 0).unwrap().sum().unwrap();
        y.backward().unwrap();

        assert_eq!(a.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0]);
        assert_eq!(b.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_index_select_gradient() {
        let x = Variable::new(from_f32(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5]), true);
        let idx = Tensor::from_f32(&[0.0, 2.0, 2.0], &[3], crate::tensor::test_device()).unwrap()
            .to_dtype(crate::tensor::DType::Int64).unwrap();
        let y = x.index_select(0, &idx).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - 1.0).abs() < 1e-5);
        assert!((grad[1] - 0.0).abs() < 1e-5);
        assert!((grad[2] - 2.0).abs() < 1e-5);
        assert!((grad[3] - 0.0).abs() < 1e-5);
        assert!((grad[4] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_std_dim_gradient() {
        let x = Variable::new(from_f32(&[1.0, 3.0, 5.0, 7.0], &[2, 2]), true);
        let y = x.std_dim(1, false).unwrap().sum().unwrap();
        y.backward().unwrap();

        let eps = 1e-4;
        let x_data = vec![1.0_f32, 3.0, 5.0, 7.0];
        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        for i in 0..4 {
            let mut xp = x_data.clone();
            let mut xm = x_data.clone();
            xp[i] += eps as f32;
            xm[i] -= eps as f32;
            let fp: f64 = from_f32(&xp, &[2, 2]).std_dim(1, false).unwrap()
                .sum().unwrap().item().unwrap();
            let fm: f64 = from_f32(&xm, &[2, 2]).std_dim(1, false).unwrap()
                .sum().unwrap().item().unwrap();
            let numerical = (fp - fm) / (2.0 * eps);
            assert!(
                (grad[i] as f64 - numerical).abs() < 0.01,
                "std_dim grad[{}]: analytical={}, numerical={}", i, grad[i], numerical
            );
        }
    }

    #[test]
    fn test_cat_many_gradient() {
        let a = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let b = Variable::new(from_f32(&[3.0, 4.0, 5.0], &[3]), true);
        let c = Variable::new(from_f32(&[6.0], &[1]), true);
        let catted = Variable::cat_many(&[&a, &b, &c], 0).unwrap();
        assert_eq!(catted.data().shape(), vec![6]);

        // Weight each element differently to verify gradient routing
        let w = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]), false);
        let loss = catted.mul(&w).unwrap().sum().unwrap();
        loss.backward().unwrap();

        assert_eq!(a.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(b.grad().unwrap().to_f32_vec().unwrap(), vec![3.0, 4.0, 5.0]);
        assert_eq!(c.grad().unwrap().to_f32_vec().unwrap(), vec![6.0]);
    }

    #[test]
    fn test_stack_gradient() {
        let a = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let b = Variable::new(from_f32(&[3.0, 4.0], &[2]), true);
        let c = Variable::new(from_f32(&[5.0, 6.0], &[2]), true);
        let stacked = Variable::stack(&[a.clone(), b.clone(), c.clone()], 0).unwrap();
        assert_eq!(stacked.data().shape(), vec![3, 2]);

        let w = Variable::new(from_f32(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0], &[3, 2]), false);
        let loss = stacked.mul(&w).unwrap().sum().unwrap();
        loss.backward().unwrap();

        assert_eq!(a.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 0.0]);
        assert_eq!(b.grad().unwrap().to_f32_vec().unwrap(), vec![0.0, 1.0]);
        assert_eq!(c.grad().unwrap().to_f32_vec().unwrap(), vec![2.0, 0.0]);
    }

    #[test]
    fn test_triu_gradient() {
        // 2x2 matrix, triu with diagonal=1 keeps only upper triangle (above main diag)
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let y = x.triu(1).unwrap();
        let vals = y.data().to_f32_vec().unwrap();
        assert_eq!(vals, vec![0.0, 2.0, 0.0, 0.0]); // only (0,1) survives

        let loss = y.sum().unwrap();
        loss.backward().unwrap();
        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        // gradient flows only through upper triangle
        assert_eq!(grad, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_triu_batch_gradient() {
        // [2, 3, 3] batch — triu applied per-matrix
        let x = Variable::new(from_f32(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
            16.0, 17.0, 18.0,
        ], &[2, 3, 3]), true);
        let y = x.triu(0).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        let expected = vec![
            1.0, 1.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
        ];
        assert_eq!(grad, expected);
    }

    #[test]
    fn test_shared_variable_multiple_adds() {
        let state0 = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let bias = Variable::new(from_f32(&[0.0, 0.0], &[2]), true);

        let state1 = state0.add(&bias).unwrap();
        let state2 = state1.add(&bias).unwrap();
        let state3 = state2.add(&bias).unwrap();
        let loss = state3.sum().unwrap();
        loss.backward().unwrap();

        let grad = bias.grad().unwrap().to_f32_vec().unwrap();
        assert!(
            (grad[0] - 3.0).abs() < 1e-5 && (grad[1] - 3.0).abs() < 1e-5,
            "bias grad should be [3,3], got {:?}",
            grad
        );
    }

    #[test]
    fn test_backward_frees_grad_fn_chain() {
        if crate::tensor::test_device() != Device::CPU { return; }
        // Verify that backward() + detach_() doesn't leak C++ autograd Nodes.
        // Simulates a training loop: multiple loss terms per step, many steps.
        use crate::nn::{Module, Linear};
        use crate::nn::optim::{Adam, Optimizer};

        fn get_rss_kb() -> usize {
            std::fs::read_to_string("/proc/self/statm")
                .ok()
                .and_then(|s| s.split_whitespace().nth(1)?.parse::<usize>().ok())
                .map(|pages| pages * 4)
                .unwrap_or(0)
        }

        let linear = Linear::on_device(256, 256, crate::tensor::test_device()).unwrap();
        let params = linear.parameters();
        let mut opt = Adam::new(&params, 0.001);
        let opts = crate::tensor::test_opts();

        // Warm up — let allocators settle
        for _ in 0..50 {
            let x = Variable::new(Tensor::randn(&[32, 256], opts).unwrap(), false);
            let y = linear.forward(&x).unwrap();
            let loss = y.sum().unwrap();
            opt.zero_grad();
            loss.backward().unwrap();
            opt.step().unwrap();
        }

        crate::tensor::malloc_trim();
        let rss_before = get_rss_kb();

        for _ in 0..2000 {
            let x = Variable::new(Tensor::randn(&[32, 256], opts).unwrap(), false);
            let y = linear.forward(&x).unwrap();
            // Multiple loss terms (like FBRL training)
            let l1 = y.sum().unwrap();
            let l2 = y.mean().unwrap();
            let total = l1.add(&l2).unwrap();
            opt.zero_grad();
            total.backward().unwrap();
            opt.step().unwrap();
        }

        crate::tensor::malloc_trim();
        let rss_after = get_rss_kb();
        let growth_mb = (rss_after as f64 - rss_before as f64) / 1024.0;
        assert!(
            growth_mb < 50.0,
            "RSS grew by {growth_mb:.1}MB over 2000 training steps — possible grad_fn leak"
        );
    }

    /// Phased leak isolation: measures handle count and RSS growth for
    /// each training phase independently. Run with --nocapture to see output.
    #[test]
    fn test_leak_isolation_phases() {
        if crate::tensor::test_device() != Device::CPU { return; }
        use crate::nn::{Module, Linear, clip_grad_norm, cross_entropy_loss};
        use crate::nn::optim::{Adam, Optimizer};
        use crate::nn::parameter::Parameter;
        use crate::tensor::{TensorOptions, live_tensor_count, rss_kb, malloc_trim};

        let opts = crate::tensor::test_opts();
        let iters = 1000;
        let batch = 32;
        let dim = 128;
        let classes: i64 = 26;

        // Build a model similar to FBRL: two linears + loss
        let linear1 = Linear::on_device(dim, dim, crate::tensor::test_device()).unwrap();
        let linear2 = Linear::on_device(dim, classes, crate::tensor::test_device()).unwrap();
        let params: Vec<Parameter> = linear1.parameters().into_iter()
            .chain(linear2.parameters()).collect();
        let mut opt = Adam::new(&params, 0.001);

        // Warm up
        for _ in 0..50 {
            let x = Variable::new(Tensor::randn(&[batch, dim], opts).unwrap(), false);
            let h = linear1.forward(&x).unwrap().relu().unwrap();
            let logits = linear2.forward(&h).unwrap();
            let target = Variable::new(
                Tensor::zeros(&[batch], TensorOptions { dtype: crate::tensor::DType::Int64, ..opts }).unwrap(), false);
            let loss = cross_entropy_loss(&logits, &target).unwrap();
            opt.zero_grad();
            loss.backward().unwrap();
            clip_grad_norm(&params, 1.0).unwrap();
            opt.step().unwrap();
        }

        // --- Phase 1: Forward only ---
        malloc_trim();
        let h0 = live_tensor_count();
        let r0 = rss_kb();
        for _ in 0..iters {
            let x = Variable::new(Tensor::randn(&[batch, dim], opts).unwrap(), false);
            let h = linear1.forward(&x).unwrap().relu().unwrap();
            let _logits = linear2.forward(&h).unwrap();
        }
        malloc_trim();
        let h1 = live_tensor_count();
        let r1 = rss_kb();
        eprintln!(
            "Phase 1 (forward only):  handles {:+}  RSS {:+.1}MB",
            h1 as i64 - h0 as i64,
            (r1 as f64 - r0 as f64) / 1024.0
        );

        // --- Phase 2: Forward + backward ---
        malloc_trim();
        let h0 = live_tensor_count();
        let r0 = rss_kb();
        for _ in 0..iters {
            let x = Variable::new(Tensor::randn(&[batch, dim], opts).unwrap(), false);
            let h = linear1.forward(&x).unwrap().relu().unwrap();
            let logits = linear2.forward(&h).unwrap();
            let target = Variable::new(
                Tensor::zeros(&[batch], TensorOptions { dtype: crate::tensor::DType::Int64, ..opts }).unwrap(), false);
            let loss = cross_entropy_loss(&logits, &target).unwrap();
            loss.backward().unwrap();
        }
        malloc_trim();
        let h1 = live_tensor_count();
        let r1 = rss_kb();
        eprintln!(
            "Phase 2 (fwd+bwd):      handles {:+}  RSS {:+.1}MB",
            h1 as i64 - h0 as i64,
            (r1 as f64 - r0 as f64) / 1024.0
        );

        // --- Phase 3: Full training step ---
        malloc_trim();
        let h0 = live_tensor_count();
        let r0 = rss_kb();
        for _ in 0..iters {
            let x = Variable::new(Tensor::randn(&[batch, dim], opts).unwrap(), false);
            let h = linear1.forward(&x).unwrap().relu().unwrap();
            let logits = linear2.forward(&h).unwrap();
            let target = Variable::new(
                Tensor::zeros(&[batch], TensorOptions { dtype: crate::tensor::DType::Int64, ..opts }).unwrap(), false);
            let l1 = cross_entropy_loss(&logits, &target).unwrap();
            let l2 = logits.mean().unwrap();
            let total = l1.add(&l2).unwrap();
            opt.zero_grad();
            total.backward().unwrap();
            clip_grad_norm(&params, 1.0).unwrap();
            opt.step().unwrap();
        }
        malloc_trim();
        let h1 = live_tensor_count();
        let r1 = rss_kb();
        eprintln!(
            "Phase 3 (full step):    handles {:+}  RSS {:+.1}MB",
            h1 as i64 - h0 as i64,
            (r1 as f64 - r0 as f64) / 1024.0
        );

        // The handle count should not grow between phases
        // (small fluctuations OK, but not proportional to iters)
        assert!(
            (h1 as i64 - h0 as i64).unsigned_abs() < 100,
            "handle count drifted by {} over {iters} steps — tensor handle leak!",
            h1 as i64 - h0 as i64
        );
    }

    /// Graph-with-loop leak test: simulates FBRL pattern (graph loops, tags,
    /// multiple loss terms, optimizer). Run with --nocapture to see diagnostics.
    #[test]
    fn test_graph_loop_leak() {
        if crate::tensor::test_device() != Device::CPU { return; }
        use crate::nn::{Module, Linear, cross_entropy_loss, clip_grad_norm};
        use crate::nn::optim::{Adam, Optimizer};
        use crate::nn::parameter::Parameter;
        use crate::graph::FlowBuilder;
        use crate::tensor::{TensorOptions, live_tensor_count, rss_kb, malloc_trim};

        let opts = crate::tensor::test_opts();
        let batch: i64 = 16;
        let dim = 64;
        let classes: i64 = 26;
        let loop_iters = 4;

        // Build a graph: Linear → loop(Linear+ReLU, 4 iters) → tag("out") → Linear
        let graph = FlowBuilder::from(Linear::on_device(dim, dim, crate::tensor::test_device()).unwrap())
            .loop_body(Linear::on_device(dim, dim, crate::tensor::test_device()).unwrap()).for_n(loop_iters)
            .tag("loop_out")
            .through(Linear::on_device(dim, classes, crate::tensor::test_device()).unwrap())
            .tag("logits")
            .build()
            .unwrap();

        let params: Vec<Parameter> = graph.named_parameters()
            .into_iter().map(|(_name, p)| p).collect();
        let mut opt = Adam::new(&params, 0.001);

        let iters = 500;

        // Warm up
        for _ in 0..30 {
            let x = Variable::new(Tensor::randn(&[batch, dim], opts).unwrap(), false);
            let y = graph.forward(&x).unwrap();
            let target = Variable::new(
                Tensor::zeros(&[batch], TensorOptions { dtype: crate::tensor::DType::Int64, ..opts }).unwrap(), false);
            let loss = cross_entropy_loss(&y, &target).unwrap();
            opt.zero_grad();
            loss.backward().unwrap();
            clip_grad_norm(&params, 1.0).unwrap();
            opt.step().unwrap();
            graph.detach_state();
        }

        // Sample handle count at 25% and 75% of the loop to avoid noise
        // from other tests running in parallel (shared global counter).
        let mut h_at_25 = 0u64;
        let mut h_at_75 = 0u64;
        malloc_trim();
        let rss_before = rss_kb();

        for i in 0..iters {
            let x = Variable::new(Tensor::randn(&[batch, dim], opts).unwrap(), false);
            let y = graph.forward(&x).unwrap();
            let target = Variable::new(
                Tensor::zeros(&[batch], TensorOptions { dtype: crate::tensor::DType::Int64, ..opts }).unwrap(), false);
            // Multiple loss terms (like FBRL)
            let l1 = cross_entropy_loss(&y, &target).unwrap();
            let l2 = y.mean().unwrap();
            let total = l1.add(&l2).unwrap();
            opt.zero_grad();
            total.backward().unwrap();
            clip_grad_norm(&params, 1.0).unwrap();
            opt.step().unwrap();
            graph.detach_state();
            // Also record metrics (like FBRL does)
            graph.record_scalar("loss", total.item().unwrap());
            if i == iters / 4 { h_at_25 = live_tensor_count(); }
            if i == iters * 3 / 4 { h_at_75 = live_tensor_count(); }
        }
        graph.flush(&[]);

        malloc_trim();
        let rss_after = rss_kb();
        let handle_drift = h_at_75 as i64 - h_at_25 as i64;
        let rss_growth_mb = (rss_after as f64 - rss_before as f64) / 1024.0;

        eprintln!(
            "Graph+loop ({iters} steps, {loop_iters} loop iters): handles {:+} (25%-75%)  RSS {:+.1}MB",
            handle_drift, rss_growth_mb
        );

        assert!(
            handle_drift.unsigned_abs() < 50,
            "handle count drifted by {handle_drift} between 25% and 75% — tensor handle leak!"
        );
        assert!(
            rss_growth_mb < 30.0,
            "RSS grew by {rss_growth_mb:.1}MB over {iters} graph+loop steps"
        );
    }

    #[test]
    fn test_autograd_node_count() {
        let opts = crate::tensor::test_opts();

        // Leaf variable: 0 nodes
        let x = Variable::new(Tensor::randn(&[2, 3], opts).unwrap(), true);
        assert_eq!(x.autograd_node_count(), 0);

        // Single op: 1 node (MulBackward)
        let y = x.mul_scalar(2.0).unwrap();
        assert!(y.autograd_node_count() >= 1);

        // Chain: mul -> sum -> 2 nodes
        let z = y.sum().unwrap();
        assert!(z.autograd_node_count() >= 2);

        // Fused linear has fewer nodes than manual matmul+add
        let w = Variable::new(Tensor::randn(&[4, 3], opts).unwrap(), true);
        let b = Variable::new(Tensor::zeros(&[4], opts).unwrap(), true);
        let inp = Variable::new(Tensor::randn(&[2, 3], opts).unwrap(), false);

        // Fused: single linear node
        let fused = linear(&inp, &w, Some(&b)).unwrap();
        let fused_nodes = fused.autograd_node_count();

        // Manual: matmul + add = 2 nodes
        let wt = w.transpose(0, 1).unwrap();
        let manual = inp.matmul(&wt).unwrap().add(&b).unwrap();
        let manual_nodes = manual.autograd_node_count();

        assert!(
            fused_nodes < manual_nodes,
            "fused linear ({fused_nodes}) should have fewer nodes than manual ({manual_nodes})"
        );
    }
}
