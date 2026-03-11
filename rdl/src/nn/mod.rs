pub mod parameter;
pub mod init;
pub mod linear;
pub mod activation;
pub mod loss;
pub mod optim;

pub use parameter::Parameter;
pub use linear::Linear;
pub use activation::{ReLU, Sigmoid, Tanh};
pub use loss::mse_loss;
pub use optim::{Optimizer, SGD, Adam};

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
}
