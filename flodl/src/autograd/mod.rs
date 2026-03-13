//! Reverse-mode automatic differentiation.
//!
//! Variables wrap tensors with gradient tracking. When `requires_grad` is
//! true, operations build a computation graph. Calling `backward()` walks
//! the graph in reverse, accumulating gradients at each leaf variable.
//!
//! ```ignore
//! let x = Variable::new(tensor_x, true);
//! let w = Variable::new(tensor_w, true);
//! let loss = x.matmul(&w)?.sum()?;
//! loss.backward()?;
//! println!("{:?}", w.grad()); // gradient of loss w.r.t. w
//! ```
//!
//! Saved tensors live inside backward closures. When a `GradFn` is dropped
//! (after backward processes its node), the closure drops, which drops the
//! saved tensors — deterministic VRAM release with zero infrastructure.

mod variable;
mod engine;
mod ops;
mod context;

pub use variable::Variable;
pub use context::{no_grad, is_grad_enabled, NoGradGuard};
pub use ops::{layer_norm, conv2d, conv_transpose2d, adaptive_avg_pool2d, grid_sample};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, Device};

    fn from_f32(data: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_f32(data, shape, Device::CPU).unwrap()
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

        // d(sum(a+b))/da = [1,1,1], d(sum(a+b))/db = [1,1,1]
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

        // d(sum(x*y))/dx = y = [4,5], d(sum(x*y))/dy = x = [2,3]
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![4.0, 5.0]);
        assert_eq!(y.grad().unwrap().to_f32_vec().unwrap(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_chain_rule() {
        // y = relu(x*w + b).sum()
        let x = Variable::new(from_f32(&[1.0, -2.0, 3.0], &[3]), true);
        let w = Variable::new(from_f32(&[2.0, 2.0, 2.0], &[3]), true);
        let b = Variable::new(from_f32(&[0.0, 5.0, 0.0], &[3]), false);

        // x*w = [2, -4, 6], +b = [2, 1, 6], relu = [2, 1, 6], sum = 9
        let y = x.mul(&w).unwrap()
            .add(&b).unwrap()
            .relu().unwrap()
            .sum().unwrap();
        assert!((y.item().unwrap() - 9.0).abs() < 1e-5);

        y.backward().unwrap();

        // relu mask = [1, 1, 1] (all positive after add)
        // d(sum)/d(relu_out) = [1,1,1]
        // d(relu)/d(add_out) = [1,1,1] (all > 0)
        // d(add)/d(mul_out) = [1,1,1]
        // d(mul)/dx = w = [2,2,2]
        // d(mul)/dw = x = [1,-2,3]
        let gx = x.grad().unwrap().to_f32_vec().unwrap();
        let gw = w.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(gx, vec![2.0, 2.0, 2.0]);
        assert_eq!(gw, vec![1.0, -2.0, 3.0]);
    }

    #[test]
    fn test_matmul_gradient() {
        // C = A @ B, loss = sum(C)
        // A = [[1,2],[3,4]], B = [[1,0],[0,1]] (identity)
        // C = A, loss = 1+2+3+4 = 10
        let a = Variable::new(
            from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let b = Variable::new(
            from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]), true);
        let c = a.matmul(&b).unwrap().sum().unwrap();
        c.backward().unwrap();

        // dL/dA = grad @ B^T = [[1,1],[1,1]] @ [[1,0],[0,1]] = [[1,1],[1,1]]
        // dL/dB = A^T @ grad = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        let ga = a.grad().unwrap().to_f32_vec().unwrap();
        let gb = b.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(ga, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(gb, vec![4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn test_sigmoid_gradient() {
        // sigmoid(0) = 0.5, sigmoid'(0) = 0.25
        let x = Variable::new(from_f32(&[0.0], &[1]), true);
        let y = x.sigmoid().unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().item().unwrap();
        assert!((grad - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_relu_gradient() {
        let x = Variable::new(from_f32(&[1.0, -1.0, 2.0, -2.0], &[4]), true);
        let y = x.relu().unwrap().sum().unwrap();
        y.backward().unwrap();

        // relu'(x) = 1 if x>0, 0 otherwise
        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(grad, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_exp_log_gradient() {
        // d(exp(x))/dx = exp(x), d(log(x))/dx = 1/x
        // exp(log(x)) = x, so d(exp(log(x)))/dx = 1
        let x = Variable::new(from_f32(&[2.0], &[1]), true);
        let y = x.log().unwrap().exp().unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().item().unwrap();
        assert!((grad - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_no_grad() {
        let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let y = no_grad(|| x.mul_scalar(3.0).unwrap());

        // y should not track gradients
        assert!(!y.requires_grad());
    }

    #[test]
    fn test_detach() {
        let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let y = x.mul_scalar(2.0).unwrap();
        let z = y.detach().mul_scalar(3.0).unwrap().sum().unwrap();
        z.backward().unwrap();

        // Detach breaks the graph — x should have no gradient
        assert!(x.grad().is_none());
    }

    #[test]
    fn test_gradient_accumulation() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
        let y = Variable::new(from_f32(&[1.0, 1.0, 1.0], &[3]), false);

        // First backward
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
        assert!(x.grad().is_none());
    }

    #[test]
    fn test_sub_gradient() {
        let a = Variable::new(from_f32(&[5.0, 6.0], &[2]), true);
        let b = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let c = a.sub(&b).unwrap().sum().unwrap();
        c.backward().unwrap();

        // d(sum(a-b))/da = [1,1], d(sum(a-b))/db = [-1,-1]
        assert_eq!(a.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0]);
        assert_eq!(b.grad().unwrap().to_f32_vec().unwrap(), vec![-1.0, -1.0]);
    }

    #[test]
    fn test_tanh_gradient() {
        // tanh(0) = 0, tanh'(0) = 1
        let x = Variable::new(from_f32(&[0.0], &[1]), true);
        let y = x.tanh_act().unwrap().sum().unwrap();
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
        // Numerical gradient check: f(x) = (x^2).sum()
        // f'(x) = 2x
        let eps = 1e-4;
        let x_data = vec![1.0_f32, 2.0, -3.0];

        // Analytical gradient
        let x = Variable::new(from_f32(&x_data, &[3]), true);
        let y = x.mul(&x).unwrap().sum().unwrap();
        y.backward().unwrap();
        let analytical = x.grad().unwrap().to_f32_vec().unwrap();

        // Numerical gradient
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
    fn test_shared_variable_multiple_adds() {
        // Pattern: same leaf used as input in multiple sequential add operations
        // (like a bias in a loop body across 3 iterations)
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
    fn test_diamond_graph() {
        // Diamond: x → y1, x → y2, z = y1 + y2
        // Both paths contribute gradient to x
        let x = Variable::new(from_f32(&[3.0], &[1]), true);
        let y1 = x.mul_scalar(2.0).unwrap(); // 6
        let y2 = x.mul_scalar(3.0).unwrap(); // 9
        let z = y1.add(&y2).unwrap().sum().unwrap(); // 15
        z.backward().unwrap();

        // dx = 2 + 3 = 5
        assert!((x.grad().unwrap().item().unwrap() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_drop_frees_backward_graph() {
        // Verify that dropping the output variable frees the backward graph
        // (closures and saved tensors).
        let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        {
            let y = x.mul_scalar(2.0).unwrap().relu().unwrap().sum().unwrap();
            y.backward().unwrap();
            // y drops here — gradFn, closures, and saved tensors all freed
        }
        // x should have gradient
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![2.0, 2.0]);
    }

    #[test]
    fn test_sin_cos_gradient() {
        // d(sin(x))/dx = cos(x), d(cos(x))/dx = -sin(x)
        let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let y = x.sin().unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        // cos(1) ≈ 0.5403, cos(2) ≈ -0.4161
        assert!((grad[0] - 1.0_f32.cos()).abs() < 1e-5);
        assert!((grad[1] - 2.0_f32.cos()).abs() < 1e-5);

        x.zero_grad();
        let y2 = x.cos().unwrap().sum().unwrap();
        y2.backward().unwrap();
        let grad2 = x.grad().unwrap().to_f32_vec().unwrap();
        // -sin(1) ≈ -0.8414, -sin(2) ≈ -0.9093
        assert!((grad2[0] - (-1.0_f32.sin())).abs() < 1e-5);
        assert!((grad2[1] - (-2.0_f32.sin())).abs() < 1e-5);
    }

    #[test]
    fn test_reciprocal_gradient() {
        // d(1/x)/dx = -1/x²
        let x = Variable::new(from_f32(&[2.0, 4.0], &[2]), true);
        let y = x.reciprocal().unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        // -1/4 = -0.25, -1/16 = -0.0625
        assert!((grad[0] - (-0.25)).abs() < 1e-5);
        assert!((grad[1] - (-0.0625)).abs() < 1e-5);
    }

    #[test]
    fn test_var_std_gradient() {
        // var([1,2,3]) = 1.0 (Bessel: sum((x-2)²)/2 = (1+0+1)/2 = 1)
        // d(var)/dx_i = 2*(x_i - mean) / (N-1)
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
        let v = x.var().unwrap();
        assert!((v.data().item().unwrap() - 1.0).abs() < 1e-5);
        v.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        // 2*(1-2)/2 = -1, 2*(2-2)/2 = 0, 2*(3-2)/2 = 1
        assert!((grad[0] - (-1.0)).abs() < 1e-5);
        assert!((grad[1] - 0.0).abs() < 1e-5);
        assert!((grad[2] - 1.0).abs() < 1e-5);

        // std = sqrt(var), backward through composition
        x.zero_grad();
        let s = x.std().unwrap();
        assert!((s.data().item().unwrap() - 1.0).abs() < 1e-5);
        s.backward().unwrap();
        let grad2 = x.grad().unwrap().to_f32_vec().unwrap();
        // d(std)/dx = d(sqrt(var))/dx = 1/(2*sqrt(var)) * d(var)/dx
        // = 0.5 * [-1, 0, 1] = [-0.5, 0, 0.5]
        assert!((grad2[0] - (-0.5)).abs() < 1e-4);
        assert!((grad2[1] - 0.0).abs() < 1e-4);
        assert!((grad2[2] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_var_dim_gradient() {
        // [[1, 2], [3, 4]] — var along dim=1
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let v = x.var_dim(1, false).unwrap().sum().unwrap();
        v.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        // Row 0: mean=1.5, d(var)/dx = 2*(x-1.5)/1 → [-1, 1]
        // Row 1: mean=3.5, d(var)/dx = 2*(x-3.5)/1 → [-1, 1]
        assert!((grad[0] - (-1.0)).abs() < 1e-5);
        assert!((grad[1] - 1.0).abs() < 1e-5);
        assert!((grad[2] - (-1.0)).abs() < 1e-5);
        assert!((grad[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gather_gradient() {
        // gather scatters gradient back via scatter_add
        // input = [[1, 2], [3, 4]], index = [[0, 0], [1, 0]], dim=1
        // output = [[1, 1], [4, 3]]
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let idx = Tensor::from_i64(&[0, 0, 1, 0], &[2, 2]).unwrap();
        let y = x.gather(1, &idx).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        // Row 0: idx=[0,0] → column 0 gets 2 grads, column 1 gets 0
        // Row 1: idx=[1,0] → column 0 gets 1, column 1 gets 1
        assert!((grad[0] - 2.0).abs() < 1e-5);
        assert!((grad[1] - 0.0).abs() < 1e-5);
        assert!((grad[2] - 1.0).abs() < 1e-5);
        assert!((grad[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_topk_gradient() {
        // topk selects top values, gradient goes back to original positions
        let x = Variable::new(from_f32(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]), true);
        let (values, _indices) = x.topk(2, 0, true, true).unwrap();
        let y = values.sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        // Top 2 are indices 4 (5.0) and 2 (4.0) — those get gradient 1.0
        assert!((grad[0] - 0.0).abs() < 1e-5); // 3.0 not in top 2
        assert!((grad[1] - 0.0).abs() < 1e-5); // 1.0
        assert!((grad[2] - 1.0).abs() < 1e-5); // 4.0 in top 2
        assert!((grad[3] - 0.0).abs() < 1e-5); // 1.0
        assert!((grad[4] - 1.0).abs() < 1e-5); // 5.0 in top 2
    }

    #[test]
    fn test_repeat_gradient() {
        // [1, 2].repeat([3]) = [1, 2, 1, 2, 1, 2]
        let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
        let y = x.repeat(&[3]).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        // Each element appears 3 times, so grad = [3, 3]
        assert!((grad[0] - 3.0).abs() < 1e-5);
        assert!((grad[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_pad_gradient() {
        // Pad [1, 2, 3] with 1 zero on each side → [0, 1, 2, 3, 0]
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
        let y = x.pad(&[1, 1], 0.0).unwrap();
        assert_eq!(y.data().shape(), vec![5]);
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        // Gradient passes through non-padded positions: all 1.0
        assert_eq!(grad, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_chunk_gradient() {
        // Chunk [1, 2, 3, 4, 5, 6] into 3 pieces, multiply each by different scalar
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]), true);
        let chunks = x.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        // chunks: [1,2], [3,4], [5,6] — multiply first by 2, second by 3
        let c0 = chunks[0].mul_scalar(2.0).unwrap();
        let c1 = chunks[1].mul_scalar(3.0).unwrap();
        let loss = c0.sum().unwrap().add(&c1.sum().unwrap()).unwrap().add(&chunks[2].sum().unwrap()).unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        // [2, 2, 3, 3, 1, 1]
        assert_eq!(grad, vec![2.0, 2.0, 3.0, 3.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sort_gradient() {
        let x = Variable::new(from_f32(&[3.0, 1.0, 2.0], &[3]), true);
        let (sorted, _indices) = x.sort(0, false).unwrap();
        // sorted = [1, 2, 3], multiply by [1, 2, 3]
        let weights = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), false);
        let loss = sorted.mul(&weights).unwrap().sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        // Original: [3, 1, 2] → sorted positions: 3→idx2, 1→idx0, 2→idx1
        // Weights at sorted positions: 3.0 gets weight 3, 1.0 gets weight 1, 2.0 gets weight 2
        assert!((grad[0] - 3.0).abs() < 1e-5); // x[0]=3.0 → largest → weight 3
        assert!((grad[1] - 1.0).abs() < 1e-5); // x[1]=1.0 → smallest → weight 1
        assert!((grad[2] - 2.0).abs() < 1e-5); // x[2]=2.0 → middle → weight 2
    }

    #[test]
    fn test_div_gradient() {
        // d(a/b)/da = 1/b, d(a/b)/db = -a/b²
        let a = Variable::new(from_f32(&[6.0], &[1]), true);
        let b = Variable::new(from_f32(&[3.0], &[1]), true);
        let y = a.div(&b).unwrap().sum().unwrap();
        y.backward().unwrap();

        // da = 1/3 ≈ 0.3333
        assert!((a.grad().unwrap().item().unwrap() - 1.0 / 3.0).abs() < 1e-5);
        // db = -6/9 ≈ -0.6667
        assert!((b.grad().unwrap().item().unwrap() - (-6.0 / 9.0)).abs() < 1e-5);
    }

    #[test]
    fn test_add_scalar_div_scalar_gradient() {
        let x = Variable::new(from_f32(&[2.0, 4.0], &[2]), true);
        // add_scalar: gradient passes through unchanged
        let y = x.add_scalar(10.0).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0]);

        x.zero_grad();
        // div_scalar: d(x/c)/dx = 1/c
        let y2 = x.div_scalar(4.0).unwrap().sum().unwrap();
        y2.backward().unwrap();
        let g = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((g[0] - 0.25).abs() < 1e-5);
        assert!((g[1] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_sqrt_gradient() {
        // d(sqrt(x))/dx = 0.5/sqrt(x)
        let x = Variable::new(from_f32(&[4.0, 9.0], &[2]), true);
        let y = x.sqrt().unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - 0.25).abs() < 1e-5); // 0.5/2
        assert!((grad[1] - 1.0 / 6.0).abs() < 1e-5); // 0.5/3
    }

    #[test]
    fn test_abs_gradient() {
        // d|x|/dx ≈ sign(x)
        let x = Variable::new(from_f32(&[-3.0, 5.0], &[2]), true);
        let y = x.abs().unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - (-1.0)).abs() < 1e-4);
        assert!((grad[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_pow_scalar_gradient() {
        // d(x^3)/dx = 3x²
        let x = Variable::new(from_f32(&[2.0, 3.0], &[2]), true);
        let y = x.pow_scalar(3.0).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - 12.0).abs() < 1e-4); // 3*4
        assert!((grad[1] - 27.0).abs() < 1e-4); // 3*9
    }

    #[test]
    fn test_clamp_gradient() {
        // gradient passes through where input was not clamped
        let x = Variable::new(from_f32(&[-1.0, 0.5, 2.0], &[3]), true);
        let y = x.clamp(0.0, 1.0).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((grad[0] - 0.0).abs() < 1e-5); // clamped → 0
        assert!((grad[1] - 1.0).abs() < 1e-5); // pass-through
        assert!((grad[2] - 0.0).abs() < 1e-5); // clamped → 0
    }

    #[test]
    fn test_mean_gradient() {
        // d(mean)/dx_i = 1/N
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
        // sum_dim: gradient expands back
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let y = x.sum_dim(1, false).unwrap().sum().unwrap();
        y.backward().unwrap();

        assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mean_dim_gradient() {
        // d(mean_dim)/dx = 1/dim_size, expanded
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let y = x.mean_dim(1, false).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        for &g in &grad {
            assert!((g - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
        // Weighted sum after softmax
        let w = Variable::new(from_f32(&[1.0, 0.0, 0.0], &[3]), false);
        let y = x.softmax(0).unwrap().mul(&w).unwrap().sum().unwrap();
        y.backward().unwrap();

        // Numerical check
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
        // Numerical gradient check for gelu
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

        // Same for silu
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
        // Transpose backward should un-transpose the gradient
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), true);
        let w = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]), false);
        let y = x.transpose(0, 1).unwrap().mul(&w).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap().shape(), vec![2, 3]);
    }

    #[test]
    fn test_narrow_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]), true);
        // Narrow: take elements [1..3] → [2, 3, 4]
        let y = x.narrow(0, 1, 3).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(grad, vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_select_gradient() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), true);
        // Select row 1 → [4, 5, 6]
        let y = x.select(0, 1).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(grad, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_expand_gradient() {
        // Expand [1, 3] → [4, 3], backward sums over expanded dims
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), true);
        let y = x.expand(&[4, 3]).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(grad, vec![4.0, 4.0, 4.0]);
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
        let idx = Tensor::from_f32(&[0.0, 2.0, 2.0], &[3], crate::tensor::Device::CPU).unwrap()
            .to_dtype(crate::tensor::DType::Int64).unwrap();
        let y = x.index_select(0, &idx).unwrap().sum().unwrap();
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_f32_vec().unwrap();
        // idx [0, 2, 2]: position 0 gets 1, position 2 gets 2, rest 0
        assert!((grad[0] - 1.0).abs() < 1e-5);
        assert!((grad[1] - 0.0).abs() < 1e-5);
        assert!((grad[2] - 2.0).abs() < 1e-5);
        assert!((grad[3] - 0.0).abs() < 1e-5);
        assert!((grad[4] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_std_dim_gradient() {
        // std_dim composes var_dim + sqrt
        let x = Variable::new(from_f32(&[1.0, 3.0, 5.0, 7.0], &[2, 2]), true);
        let y = x.std_dim(1, false).unwrap().sum().unwrap();
        y.backward().unwrap();

        // Numerical check
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
}
