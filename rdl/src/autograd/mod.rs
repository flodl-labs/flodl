mod variable;
mod engine;
mod ops;
mod context;

pub use variable::Variable;
pub use context::{no_grad, is_grad_enabled};
pub use ops::layer_norm;

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
}
