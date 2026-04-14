    use super::*;
    use crate::graph::{
        FlowBuilder, Reduce, MergeOp,
        SoftmaxRouter, SigmoidRouter, FixedSelector, ArgmaxSelector,
        ThresholdHalt, LearnedHalt,
    };
    use crate::autograd::Variable;
    use crate::nn::{Linear, NamedInputModule, ReLU, Sigmoid, mse_loss, Optimizer, SGD};
    use crate::tensor::Tensor;
    use std::collections::HashMap;

    fn from_f32(data: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_f32(data, shape, crate::tensor::test_device()).unwrap()
    }

    // --- Helper modules for testing ---

    /// Doubles the input: forward(x) = 2*x
    struct Doubler;
    impl Module for Doubler {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            input.add(input)
        }
    }

    /// Adds a learnable bias at each step (for gradient accumulation testing).
    struct BiasStep {
        bias: Parameter,
    }
    impl BiasStep {
        fn new(size: i64) -> Result<Self> {
            let data = Tensor::zeros(&[size], crate::tensor::test_opts())?;
            let var = Variable::new(data, true);
            Ok(BiasStep {
                bias: Parameter {
                    variable: var,
                    name: "loop_bias".to_string(),
                },
            })
        }
    }
    impl Module for BiasStep {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            input.add(&self.bias.variable)
        }
        fn parameters(&self) -> Vec<Parameter> {
            vec![self.bias.clone()]
        }
    }

    /// Module that adds a tagged ref to the stream (for Using tests).
    struct AddRefModule;
    impl Module for AddRefModule {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            Ok(input.clone())
        }
        fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }
    }
    impl NamedInputModule for AddRefModule {
        fn forward_named(
            &self,
            input: &Variable,
            refs: &HashMap<String, Variable>,
        ) -> Result<Variable> {
            if let Some(ctx) = refs.get("ctx") {
                input.add(ctx)
            } else {
                Ok(input.clone())
            }
        }
    }

    // --- Core graph tests (from before) ---

    #[test]
    fn test_single_module() {
        let l = Linear::on_device(3, 2, crate::tensor::test_device()).unwrap();
        let graph = FlowBuilder::from(l).build().unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn test_linear_chain() {
        let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .through(ReLU::new())
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn test_also_residual() {
        let l1 = Linear::on_device(3, 3, crate::tensor::test_device()).unwrap();
        l1.weight.variable.set_data(from_f32(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
        ));
        l1.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0, 0.0], &[3]));

        let l2 = Linear::on_device(3, 3, crate::tensor::test_device()).unwrap();
        l2.weight.variable.set_data(from_f32(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
        ));
        l2.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[1.0, 1.0, 1.0], &[3]));

        // l1(x) + l2(l1(x)) = x + (x + 1) = 2x + 1
        let graph = FlowBuilder::from(l1).also(l2).build().unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 5.0).abs() < 1e-5);
        assert!((data[2] - 7.0).abs() < 1e-5);
    }

    // --- Fork tests ---

    #[test]
    fn test_fork_basic() {
        // Fork runs a side module but main stream continues unchanged.
        // identity(x) → fork(linear) tagged "side" → through(ReLU)
        // Main stream: ReLU(identity(x)) = ReLU(x)
        // Side output: linear(x) accessible via tagged("side")
        let l = Linear::on_device(2, 3, crate::tensor::test_device()).unwrap();

        let graph = FlowBuilder::from(Identity)
            .fork(l)
            .tag("side")
            .through(ReLU::new())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, -2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();

        // Main stream went through ReLU(identity(x)) → shape [1, 2]
        assert_eq!(y.shape(), vec![1, 2]);
        let data = y.data().to_f32_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 0.0).abs() < 1e-5); // ReLU(-2) = 0

        // Side output is linear(x) → shape [1, 3]
        let side = graph.tagged("side").unwrap();
        assert_eq!(side.shape(), vec![1, 3]);
    }

    #[test]
    fn test_fork_multiple() {
        // Two forks from the same stream: letter_head and case_head pattern
        let head_a = Linear::on_device(4, 3, crate::tensor::test_device()).unwrap();
        let head_b = Linear::on_device(4, 2, crate::tensor::test_device()).unwrap();

        let graph = FlowBuilder::from(Linear::on_device(2, 4, crate::tensor::test_device()).unwrap())
            .tag("latent")
            .fork(head_a)
            .tag("head_a")
            .fork(head_b)
            .tag("head_b")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();

        // Main stream is still the linear(2→4) output
        assert_eq!(y.shape(), vec![1, 4]);

        // Both forks produced their outputs
        let a = graph.tagged("head_a").unwrap();
        assert_eq!(a.shape(), vec![1, 3]);
        let b = graph.tagged("head_b").unwrap();
        assert_eq!(b.shape(), vec![1, 2]);
    }

    #[test]
    fn test_fork_backward() {
        // Gradients flow through both forks and the main stream
        let graph = FlowBuilder::from(Linear::on_device(2, 4, crate::tensor::test_device()).unwrap())
            .fork(Linear::on_device(4, 3, crate::tensor::test_device()).unwrap())
            .tag("side")
            .through(Linear::on_device(4, 1, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();

        // Loss from main stream + side output
        let side = graph.tagged("side").unwrap();
        let loss = y.sum().unwrap().add(&side.sum().unwrap()).unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some(), "input should have gradient");
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    // --- Split/Merge tests ---

    #[test]
    fn test_split_merge_add() {
        let graph = FlowBuilder::from(Linear::on_device(3, 3, crate::tensor::test_device()).unwrap())
            .split(vec![Box::new(ReLU::new()), Box::new(Sigmoid::new())])
            .merge(MergeOp::Add)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, -1.0, 2.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 3]);
    }

    #[test]
    fn test_split_merge_mean() {
        let l = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
        l.weight
            .variable
            .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        l.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0], &[2]));

        let b1 = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
        b1.weight
            .variable
            .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        b1.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0], &[2]));
        let b2 = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
        b2.weight
            .variable
            .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        b2.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0], &[2]));

        let graph = FlowBuilder::from(l)
            .split(vec![Box::new(b1), Box::new(b2)])
            .merge(MergeOp::Mean)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[3.0, 7.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_parameters() {
        let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .through(ReLU::new())
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let params = graph.parameters();
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_graph_backward() {
        let l1 = Linear::on_device(3, 2, crate::tensor::test_device()).unwrap();
        let l2 = Linear::on_device(2, 1, crate::tensor::test_device()).unwrap();

        let graph = FlowBuilder::from(l1)
            .through(ReLU::new())
            .through(l2)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_graph_as_module() {
        let inner = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .through(ReLU::new())
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = outer.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
        assert_eq!(outer.parameters().len(), 4);
    }

    #[test]
    fn test_training_loop() {
        let graph = FlowBuilder::from(Linear::on_device(1, 1, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let params = graph.parameters();
        let mut optim = SGD::new(&params, 0.01, 0.0);

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1]), false);
        let target = Variable::new(from_f32(&[3.0, 5.0, 7.0, 9.0], &[4, 1]), false);

        let mut last_loss = f64::MAX;
        for _ in 0..800 {
            optim.zero_grad();
            let pred = graph.forward(&x).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();
            last_loss = loss.item().unwrap();
            loss.backward().unwrap();
            optim.step().unwrap();
        }

        assert!(last_loss < 0.01, "got loss={}", last_loss);
    }

    #[test]
    fn test_also_backward() {
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .also(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_split_merge_backward() {
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .split(vec![
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
            ])
            .merge(MergeOp::Add)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_build_error_open_streams() {
        let result = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .split(vec![Box::new(ReLU::new()), Box::new(Sigmoid::new())])
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_build_error_duplicate_tag() {
        let result = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .tag("features")
            .through(ReLU::new())
            .tag("features")
            .build();
        assert!(result.is_err());
    }

    // --- Using tests ---

    #[test]
    fn test_using_backward_ref() {
        // Tag a point, then use it downstream
        // Graph: linear(x) → tag("ctx") → through(AddRef).using("ctx")
        // AddRef adds ctx to stream: stream + ctx = 2 * linear(x)
        let l = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
        l.weight
            .variable
            .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        l.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0], &[2]));

        let graph = FlowBuilder::from(l)
            .tag("ctx")
            .through(AddRefModule)
            .using(&["ctx"])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[3.0, 5.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        // identity(x) = [3, 5], then AddRef adds ctx ([3, 5]) = [6, 10]
        assert!((data[0] - 6.0).abs() < 1e-5);
        assert!((data[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_using_backward_gradients() {
        let l = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
        let graph = FlowBuilder::from(l)
            .tag("ctx")
            .through(AddRefModule)
            .using(&["ctx"])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_using_error_plain_module() {
        // Using on a plain module (not NamedInputModule) should error
        let result = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .tag("ctx")
            .through(ReLU::new())
            .using(&["ctx"])
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_using_error_unknown_tag() {
        let result = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .through(AddRefModule)
            .using(&["nonexistent"])
            .build();
        assert!(result.is_err());
    }

    // --- Loop tests ---

    #[test]
    fn test_loop_for() {
        // Doubler × 3 iterations: [1, 2] → [8, 16]
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Doubler)
            .for_n(3)
            .build()
            .unwrap();

        // Set linear to identity
        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 8.0).abs() < 1e-5, "1*2^3=8, got {}", data[0]);
        assert!((data[1] - 16.0).abs() < 1e-5, "2*2^3=16, got {}", data[1]);
    }

    #[test]
    fn test_loop_for_backward() {
        // Loop with a learnable bias — gradient should accumulate across iterations
        let bias_step = BiasStep::new(2).unwrap();
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(bias_step)
            .for_n(3)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        // All parameters should have gradients
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }

        // The bias gradient should be 3 (accumulated from 3 iterations)
        // dL/db = 1 per iteration, 3 iterations → grad = [3, 3]
        // (because sum reduces to scalar, dL/d_each_element = 1, and bias contributes at each step)
        let all_params = graph.parameters();
        // Find the loop_bias parameter (from BiasStep, not Linear's "bias")
        let bias_param = all_params.iter().find(|p| p.name == "loop_bias").unwrap();
        let grad = bias_param.variable.grad().unwrap().to_f32_vec().unwrap();
        assert!(
            (grad[0] - 3.0).abs() < 1e-5,
            "bias grad should be 3, got {}",
            grad[0]
        );
    }

    #[test]
    fn test_loop_while() {
        // While max < 10: double. Input [1, 2] → double until max >= 10
        // Iter 0: check [1,2] max=2 < 10 → double → [2,4]
        // Iter 1: check [2,4] max=4 < 10 → double → [4,8]
        // Iter 2: check [4,8] max=8 < 10 → double → [8,16]
        // Iter 3: check [8,16] max=16 >= 10 → halt
        // Result: [8, 16]
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Doubler)
            .while_cond(ThresholdHalt::new(10.0), 20)
            .build()
            .unwrap();

        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 8.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 16.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_loop_while_immediate_halt() {
        // Threshold 0.5 — input [1, 2] max=2 > 0.5, halt immediately
        // While checks before body, so body never runs
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Doubler)
            .while_cond(ThresholdHalt::new(0.5), 20)
            .build()
            .unwrap();

        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        // Body never ran — output = input
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_loop_until() {
        // Until max > 10: double. Body runs at least once.
        // Input [1, 2]
        // Iter 0: double → [2, 4], check max=4 <= 10 → continue
        // Iter 1: double → [4, 8], check max=8 <= 10 → continue
        // Iter 2: double → [8, 16], check max=16 > 10 → halt
        // Result: [8, 16]
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Doubler)
            .until_cond(ThresholdHalt::new(10.0), 20)
            .build()
            .unwrap();

        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 8.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 16.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_loop_until_at_least_once() {
        // Until with threshold 0.5 — input [1, 2] would halt immediately in While,
        // but Until always runs body at least once
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Doubler)
            .until_cond(ThresholdHalt::new(0.5), 20)
            .build()
            .unwrap();

        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        // Body ran once: [2, 4]
        assert!((data[0] - 2.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 4.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_loop_parameters() {
        // Loop with learnable body — parameters should include body params
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .for_n(3)
            .build()
            .unwrap();

        let params = graph.parameters();
        // From module: weight + bias = 2, loop body Linear: weight + bias = 2
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_loop_while_parameters() {
        // While loop with body + condition — both contribute parameters
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .loop_body(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .while_cond(Linear::on_device(2, 1, crate::tensor::test_device()).unwrap(), 10)
            .build()
            .unwrap();

        let params = graph.parameters();
        // From module: 2, loop body: 2, condition: 2 = 6
        assert_eq!(params.len(), 6);
    }

    #[test]
    fn test_loop_in_chain() {
        // Linear → Loop(ReLU) × 3 → Linear
        let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .loop_body(ReLU::new())
            .for_n(3)
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn test_loop_using_backward_ref() {
        // Tag a tensor, then use it inside a loop body via .using()
        // Graph: identity → tag("ctx") → loop_body(AddRefModule).for_n(3).using("ctx")
        // Each iteration: state = state + ctx
        // So after 3 iterations: state = x + 3*x = 4*x
        let graph = FlowBuilder::from(Identity)
            .tag("ctx")
            .loop_body(AddRefModule)
            .for_n(3)
            .using(&["ctx"])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[2.0, 3.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        // x = [2, 3], after 3 iterations of (state + ctx): [8, 12]
        assert!((data[0] - 8.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 12.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_loop_using_backward_gradients() {
        // Ensure gradients flow through loop+using
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .tag("ctx")
            .loop_body(AddRefModule)
            .for_n(2)
            .using(&["ctx"])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some(), "input should have gradient");
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    // --- Forward reference tests ---

    /// Nil-safe add: skips nil inputs, adds rest. For forward ref state accumulation.
    struct NilSafeAdd;
    impl Module for NilSafeAdd {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            Ok(input.clone())
        }
        fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }
    }
    impl NamedInputModule for NilSafeAdd {
        fn forward_named(
            &self,
            input: &Variable,
            refs: &HashMap<String, Variable>,
        ) -> Result<Variable> {
            if let Some(memory) = refs.get("memory") {
                input.add(memory)
            } else {
                Ok(input.clone())
            }
        }
    }

    use crate::nn::Identity;

    #[test]
    fn test_flowbuilder_new() {
        // FlowBuilder::new() starts with implicit Identity
        let graph = FlowBuilder::new()
            .tag("input")
            .through(Linear::on_device(3, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn test_forward_ref() {
        // Forward reference: using() before tag(). State carries between forward() calls.
        // Graph: entry → NilSafeAdd.Using("memory") → Identity.Tag("memory")
        // Pass 1: add gets [stream, zeros] (memory is nil/zeroed) → Identity → state captured
        // Pass 2: add gets [stream, prev_output] → sum → Identity → state captured
        let graph = FlowBuilder::from(Identity)
            .through(NilSafeAdd)
            .using(&["memory"])
            .through(Identity)
            .tag("memory")
            .build()
            .unwrap();

        assert!(graph.has_state());

        // Pass 1: [1,2] + zeros → [1,2]
        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y1 = graph.forward(&x).unwrap();
        let d1 = y1.data().to_f32_vec().unwrap();
        assert!((d1[0] - 1.0).abs() < 1e-5, "pass1[0]: got {}", d1[0]);
        assert!((d1[1] - 2.0).abs() < 1e-5, "pass1[1]: got {}", d1[1]);

        // Pass 2: [1,2] + [1,2] → [2,4]
        let y2 = graph.forward(&x).unwrap();
        let d2 = y2.data().to_f32_vec().unwrap();
        assert!((d2[0] - 2.0).abs() < 1e-5, "pass2[0]: got {}", d2[0]);
        assert!((d2[1] - 4.0).abs() < 1e-5, "pass2[1]: got {}", d2[1]);

        // Pass 3: [1,2] + [2,4] → [3,6]
        let y3 = graph.forward(&x).unwrap();
        let d3 = y3.data().to_f32_vec().unwrap();
        assert!((d3[0] - 3.0).abs() < 1e-5, "pass3[0]: got {}", d3[0]);
        assert!((d3[1] - 6.0).abs() < 1e-5, "pass3[1]: got {}", d3[1]);
    }

    #[test]
    fn test_forward_ref_reset_state() {
        let graph = FlowBuilder::from(Identity)
            .through(NilSafeAdd)
            .using(&["memory"])
            .through(Identity)
            .tag("memory")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);

        // Build up state
        graph.forward(&x).unwrap();
        graph.forward(&x).unwrap();
        let y_before = graph.forward(&x).unwrap();
        let d_before = y_before.data().to_f32_vec().unwrap();
        assert!((d_before[0] - 3.0).abs() < 1e-5);

        // Reset and verify state is cleared
        graph.reset_state();
        let y_after = graph.forward(&x).unwrap();
        let d_after = y_after.data().to_f32_vec().unwrap();
        assert!((d_after[0] - 1.0).abs() < 1e-5, "after reset: got {}", d_after[0]);
    }

    #[test]
    fn test_forward_ref_detach_state() {
        let graph = FlowBuilder::from(Identity)
            .through(NilSafeAdd)
            .using(&["memory"])
            .through(Identity)
            .tag("memory")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);

        // Run forward, accumulate state
        let y1 = graph.forward(&x).unwrap();
        let _ = y1.sum().unwrap();

        // Detach state — values preserved but gradient chain broken
        graph.detach_state();

        // State should still have values (not reset)
        let y2 = graph.forward(&x).unwrap();
        let d2 = y2.data().to_f32_vec().unwrap();
        assert!((d2[0] - 2.0).abs() < 1e-5, "detach preserves values: got {}", d2[0]);
    }

    #[test]
    fn test_forward_ref_backward() {
        // Gradients should flow through forward-ref connections
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .through(NilSafeAdd)
            .using(&["memory"])
            .through(Identity)
            .tag("memory")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some(), "input should have gradient");
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_forward_ref_unresolved_error() {
        // Using a tag that is never defined should error at build
        let result = FlowBuilder::from(Identity)
            .through(NilSafeAdd)
            .using(&["nonexistent"])
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_ref_mixed_refs() {
        // Mix backward ref (tag before using) and forward ref (using before tag)
        // "ctx" is backward (AddRefModule expects "ctx"), "memory" is forward (NilSafeAdd expects "memory")
        let graph = FlowBuilder::from(Identity)
            .tag("ctx")
            .through(AddRefModule)
            .using(&["ctx"])
            .through(NilSafeAdd)
            .using(&["memory"])
            .through(Identity)
            .tag("memory")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);

        // Pass 1: entry=[1,2], AddRef adds ctx=[1,2] → [2,4], NilSafeAdd +zeros → [2,4]
        let y1 = graph.forward(&x).unwrap();
        let d1 = y1.data().to_f32_vec().unwrap();
        assert!((d1[0] - 2.0).abs() < 1e-5, "mixed pass1[0]: got {}", d1[0]);

        // Pass 2: entry=[1,2], AddRef adds ctx=[1,2] → [2,4], NilSafeAdd +[2,4] → [4,8]
        let y2 = graph.forward(&x).unwrap();
        let d2 = y2.data().to_f32_vec().unwrap();
        assert!((d2[0] - 4.0).abs() < 1e-5, "mixed pass2[0]: got {}", d2[0]);
    }

    // --- Switch tests ---

    /// Triples input.
    struct Tripler;
    impl Module for Tripler {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            input.add(&input.add(input)?)
        }
        fn parameters(&self) -> Vec<Parameter> { vec![] }
    }

    #[test]
    fn test_switch_selects_branch() {
        // Branch 0: double, Branch 1: triple. Router selects branch 1.
        let graph = FlowBuilder::from(Identity)
            .switch(FixedSelector::new(1), vec![Box::new(Doubler), Box::new(Tripler)])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        assert!((data[0] - 3.0).abs() < 1e-5, "triple [1]=3, got {}", data[0]);
        assert!((data[1] - 6.0).abs() < 1e-5, "triple [2]=6, got {}", data[1]);
    }

    #[test]
    fn test_switch_branch0() {
        let graph = FlowBuilder::from(Identity)
            .switch(FixedSelector::new(0), vec![Box::new(Doubler), Box::new(Tripler)])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-5, "double [1]=2, got {}", data[0]);
        assert!((data[1] - 4.0).abs() < 1e-5, "double [2]=4, got {}", data[1]);
    }

    #[test]
    fn test_switch_backward() {
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .switch(FixedSelector::new(0), vec![
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
            ])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        // Only entry + selected branch params should have gradients
        // (router has no params, unselected branch wasn't executed)
    }

    #[test]
    fn test_switch_parameters() {
        let graph = FlowBuilder::from(Identity)
            .switch(
                Linear::on_device(2, 1, crate::tensor::test_device()).unwrap(),
                vec![
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                ],
            )
            .build()
            .unwrap();

        let params = graph.parameters();
        // Router: 2, Branch0: 2, Branch1: 2 = 6
        assert_eq!(params.len(), 6);
    }

    // --- Gate tests ---

    /// Router that outputs equal weights for all experts.
    struct EqualRouter(usize);
    impl Module for EqualRouter {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            let batch = input.shape()[0];
            let w = 1.0 / self.0 as f32;
            let data = vec![w; batch as usize * self.0];
            Ok(Variable::new(
                Tensor::from_f32(&data, &[batch, self.0 as i64], crate::tensor::test_device())?,
                false,
            ))
        }
        fn parameters(&self) -> Vec<Parameter> { vec![] }
    }

    #[test]
    fn test_gate_equal_weights() {
        // Equal weights: output = mean of expert outputs
        let graph = FlowBuilder::from(Identity)
            .gate(EqualRouter(2), vec![Box::new(Doubler), Box::new(Tripler)])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[2.0, 4.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // double=[4,8], triple=[6,12], mean = [5, 10]
        assert!((data[0] - 5.0).abs() < 1e-5, "gate[0]=5, got {}", data[0]);
        assert!((data[1] - 10.0).abs() < 1e-5, "gate[1]=10, got {}", data[1]);
    }

    #[test]
    fn test_gate_backward() {
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .gate(
                Linear::on_device(2, 2, crate::tensor::test_device()).unwrap(),
                vec![
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                ],
            )
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_gate_parameters() {
        let graph = FlowBuilder::from(Identity)
            .gate(
                Linear::on_device(2, 2, crate::tensor::test_device()).unwrap(),
                vec![
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                ],
            )
            .build()
            .unwrap();

        let params = graph.parameters();
        // Router: 2, Expert0: 2, Expert1: 2 = 6
        assert_eq!(params.len(), 6);
    }

    // --- Map tests ---

    #[test]
    fn test_map_each() {
        // Map doubler over 3 elements along dim 0
        let graph = FlowBuilder::from(Identity)
            .map(Doubler)
            .each()
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert_eq!(y.shape(), vec![3, 2]);
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[5] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_map_batched() {
        // Batched: pass full tensor, skip element-wise
        let graph = FlowBuilder::from(Identity)
            .map(Doubler)
            .batched()
            .each()
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_map_backward() {
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .map(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .each()
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    // --- Observation tests ---

    /// Scalar output module: sum all elements to a single value.
    struct ScalarSum;
    impl Module for ScalarSum {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            input.sum()
        }
    }

    #[test]
    fn test_tagged_capture() {
        // Tag intermediate output and retrieve it after forward
        let graph = FlowBuilder::from(Identity)
            .tag("features")
            .through(Doubler)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let _ = graph.forward(&x).unwrap();

        // Tagged value should be the identity output (before doubling)
        let features = graph.tagged("features").unwrap();
        let data = features.data().to_f32_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);

        assert!(graph.tagged("nonexistent").is_none());
    }

    #[test]
    fn test_tagged_updates_each_forward() {
        let graph = FlowBuilder::from(Doubler)
            .tag("doubled")
            .build()
            .unwrap();

        let x1 = Variable::new(from_f32(&[1.0], &[1, 1]), false);
        let _ = graph.forward(&x1).unwrap();
        let v1 = graph.tagged("doubled").unwrap().item().unwrap();
        assert!((v1 - 2.0).abs() < 1e-5);

        let x2 = Variable::new(from_f32(&[5.0], &[1, 1]), false);
        let _ = graph.forward(&x2).unwrap();
        let v2 = graph.tagged("doubled").unwrap().item().unwrap();
        assert!((v2 - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_tag_names() {
        let graph = FlowBuilder::from(Identity)
            .tag("a")
            .through(Identity)
            .tag("b")
            .build()
            .unwrap();

        let mut names = graph.tag_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_collect_flush_trend() {
        // Simulate a training loop with collect → flush → trend
        let graph = FlowBuilder::from(ScalarSum)
            .tag("loss")
            .build()
            .unwrap();

        // Epoch 1: 3 batches with different inputs
        for val in &[1.0f32, 2.0, 3.0] {
            let x = Variable::new(from_f32(&[*val], &[1, 1]), false);
            let _ = graph.forward(&x).unwrap();
            graph.collect(&["loss"]).unwrap();
        }
        // batch buffer should have [1, 2, 3]
        let collected = graph.collected("loss");
        assert_eq!(collected.len(), 3);

        graph.flush(&["loss"]);
        assert_eq!(graph.flush_count(), 1);

        // Epoch 2: 3 batches
        for val in &[0.5f32, 0.3, 0.2] {
            let x = Variable::new(from_f32(&[*val], &[1, 1]), false);
            let _ = graph.forward(&x).unwrap();
            graph.collect(&["loss"]).unwrap();
        }
        graph.flush(&["loss"]);
        assert_eq!(graph.flush_count(), 2);

        // Trend should show decrease: epoch1 mean=2.0, epoch2 mean≈0.333
        let trend = graph.trend("loss");
        assert_eq!(trend.len(), 2);
        assert!((trend.values()[0] - 2.0).abs() < 1e-5);
        assert!((trend.values()[1] - (1.0 / 3.0)).abs() < 1e-5);
        assert!(trend.improving(0));
    }

    #[test]
    fn test_record_external_values() {
        let graph = FlowBuilder::from(Identity).build().unwrap();

        graph.record("external_loss", &[0.5, 0.4, 0.3]);
        graph.flush(&["external_loss"]);

        graph.record("external_loss", &[0.1, 0.05]);
        graph.flush(&["external_loss"]);

        let trend = graph.trend("external_loss");
        assert_eq!(trend.len(), 2);
        assert!((trend.values()[0] - 0.4).abs() < 1e-5); // mean(0.5, 0.4, 0.3)
        assert!((trend.values()[1] - 0.075).abs() < 1e-5); // mean(0.1, 0.05)
        assert!(trend.improving(0));
    }

    #[test]
    fn test_flush_all() {
        let graph = FlowBuilder::from(Identity).build().unwrap();

        graph.record("a", &[1.0, 2.0]);
        graph.record("b", &[3.0, 4.0]);
        graph.flush(&[]); // flush all

        assert_eq!(graph.trend("a").len(), 1);
        assert_eq!(graph.trend("b").len(), 1);
    }

    #[test]
    fn test_reset_trend() {
        let graph = FlowBuilder::from(Identity).build().unwrap();

        graph.record("loss", &[1.0]);
        graph.flush(&[]);
        assert_eq!(graph.trend("loss").len(), 1);

        graph.reset_trend(&["loss"]);
        assert_eq!(graph.trend("loss").len(), 0);
    }

    #[test]
    fn test_trends_group() {
        let graph = FlowBuilder::from(Identity).build().unwrap();

        // Two decreasing series
        for epoch in &[10.0, 8.0, 6.0, 4.0] {
            graph.record("a", &[*epoch]);
            graph.record("b", &[*epoch * 0.5]);
            graph.flush(&[]);
        }

        let tg = graph.trends(&["a", "b"]);
        assert_eq!(tg.len(), 2);
        assert!(tg.all_improving(0));
    }

    // --- TagGroup tests ---

    #[test]
    fn test_tag_group() {
        // Split into 3 branches with tag_group, then merge
        let graph = FlowBuilder::from(Identity)
            .split(vec![
                Box::new(Doubler),
                Box::new(Tripler),
                Box::new(Identity),
            ])
            .tag_group("branch")
            .merge(MergeOp::Add)
            .build()
            .unwrap();

        // Check group registration
        let members = graph.tag_group("branch").unwrap();
        assert_eq!(members, &["branch_0", "branch_1", "branch_2"]);

        // Non-existent group returns None
        assert!(graph.tag_group("nonexistent").is_none());

        // Tags work for observation
        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let _ = graph.forward(&x).unwrap();

        let b0 = graph.tagged("branch_0").unwrap();
        let b0_data = b0.data().to_f32_vec().unwrap();
        assert!((b0_data[0] - 2.0).abs() < 1e-5, "doubler: got {}", b0_data[0]);

        let b1 = graph.tagged("branch_1").unwrap();
        let b1_data = b1.data().to_f32_vec().unwrap();
        assert!((b1_data[0] - 3.0).abs() < 1e-5, "tripler: got {}", b1_data[0]);
    }

    #[test]
    fn test_tag_group_observation() {
        // Tag group with collect/flush and trends expansion
        let graph = FlowBuilder::from(Identity)
            .split(vec![Box::new(ScalarSum), Box::new(ScalarSum)])
            .tag_group("head")
            .merge(MergeOp::Add)
            .build()
            .unwrap();

        // Run a few epochs
        for epoch in &[1.0f32, 2.0, 3.0] {
            let x = Variable::new(from_f32(&[*epoch], &[1, 1]), false);
            let _ = graph.forward(&x).unwrap();
            graph.collect(&["head_0", "head_1"]).unwrap();
            graph.flush(&["head_0", "head_1"]);
        }

        // Trends with group expansion
        let tg = graph.trends(&["head"]);
        assert_eq!(tg.len(), 2); // head_0 and head_1
    }

    #[test]
    fn test_tag_group_errors() {
        // tag_group on single stream should error
        let result = FlowBuilder::from(Identity)
            .tag_group("bad")
            .build();
        assert!(result.is_err());

        // Duplicate group name
        let result = FlowBuilder::from(Identity)
            .split(vec![Box::new(Doubler), Box::new(Tripler)])
            .tag_group("x")
            .merge(MergeOp::Add)
            .split(vec![Box::new(Doubler), Box::new(Tripler)])
            .tag_group("x")
            .merge(MergeOp::Add)
            .build();
        assert!(result.is_err());
    }

    // --- Input tests ---

    /// Module that adds all refs to input (for multi-input testing).
    struct SumRefs;
    impl Module for SumRefs {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            Ok(input.clone())
        }
        fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }
    }
    impl NamedInputModule for SumRefs {
        fn forward_named(
            &self,
            input: &Variable,
            refs: &HashMap<String, Variable>,
        ) -> Result<Variable> {
            let mut result = input.clone();
            for v in refs.values() {
                result = result.add(v)?;
            }
            Ok(result)
        }
    }

    #[test]
    fn test_input_auxiliary() {
        // Graph with auxiliary inputs: From(identity) + Input("ctx")
        // Downstream: through(SumRefs).using("ctx")
        let graph = FlowBuilder::from(Identity)
            .input(&["ctx"])
            .through(SumRefs)
            .using(&["ctx"])
            .build()
            .unwrap();

        let main = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let ctx = Variable::new(from_f32(&[10.0, 20.0], &[1, 2]), false);

        let y = graph.forward_multi(&[main, ctx]).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // SumRefs adds ctx to main: [1+10, 2+20] = [11, 22]
        assert!((data[0] - 11.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 22.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_input_multiple() {
        // Graph with two auxiliary inputs
        let graph = FlowBuilder::from(Identity)
            .input(&["a", "b"])
            .through(SumRefs)
            .using(&["a", "b"])
            .build()
            .unwrap();

        let main = Variable::new(from_f32(&[1.0], &[1, 1]), false);
        let a = Variable::new(from_f32(&[10.0], &[1, 1]), false);
        let b = Variable::new(from_f32(&[100.0], &[1, 1]), false);

        let y = graph.forward_multi(&[main, a, b]).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // 1 + 10 + 100 = 111
        assert!((data[0] - 111.0).abs() < 1e-5, "got {}", data[0]);
    }

    #[test]
    fn test_input_error_count_mismatch() {
        let graph = FlowBuilder::from(Identity)
            .input(&["ctx"])
            .build()
            .unwrap();

        // forward() with single input should fail (expects 2: main + ctx)
        let x = Variable::new(from_f32(&[1.0], &[1, 1]), false);
        assert!(graph.forward(&x).is_err());
    }

    // --- Graph set_training test ---

    #[test]
    fn test_graph_set_training() {
        use crate::nn::Dropout;

        let graph = FlowBuilder::from(Linear::on_device(3, 3, crate::tensor::test_device()).unwrap())
            .through(Dropout::new(0.5))
            .build()
            .unwrap();

        // Training mode: dropout is active
        let x = Variable::new(from_f32(&[1.0; 12], &[4, 3]), false);
        let y1 = graph.forward(&x).unwrap();
        assert_eq!(y1.shape(), vec![4, 3]);

        // Set eval via graph
        graph.set_training(false);
        let y2 = graph.forward(&x).unwrap();
        let y3 = graph.forward(&x).unwrap();
        assert_eq!(y2.shape(), vec![4, 3]);

        // In eval: dropout is identity, so repeated forward gives same output
        let d2 = y2.data().to_f32_vec().unwrap();
        let d3 = y3.data().to_f32_vec().unwrap();
        let same = d2.iter().zip(d3.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(same, "eval mode should be deterministic (no dropout)");
    }

    // --- walk_modules test ---

    #[test]
    fn test_walk_modules() {
        use crate::nn::walk_modules;

        let l1 = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
        let mut count = 0;
        walk_modules(&l1, &mut |_| count += 1);
        assert_eq!(count, 1); // leaf module, no children
    }

    // --- Profiling tests ---

    #[test]
    fn test_profiling_basic() {
        let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .tag("encoder")
            .through(ReLU::new())
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .tag("decoder")
            .build()
            .unwrap();

        // No profiling by default
        assert!(!graph.profiling());
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        graph.forward(&x).unwrap();
        assert!(graph.profile().is_none());

        // Enable profiling
        graph.enable_profiling();
        assert!(graph.profiling());
        graph.forward(&x).unwrap();

        let p = graph.profile().unwrap();
        assert!(p.total.as_nanos() > 0, "total should be nonzero");
        assert!(!p.nodes.is_empty(), "should have node timings");
        assert!(!p.levels.is_empty(), "should have level timings");

        // Tagged node timing
        let enc_dur = p.timing("encoder");
        assert!(enc_dur.as_nanos() > 0, "encoder timing should be nonzero");
        let dec_dur = p.timing("decoder");
        assert!(dec_dur.as_nanos() > 0, "decoder timing should be nonzero");
        assert!(p.timing("nonexistent").is_zero());

        // Graph-level timing shortcut
        assert!(graph.timing("encoder").as_nanos() > 0);

        // Display
        let s = p.to_string();
        assert!(s.contains("Forward:"));
        assert!(s.contains("Level"));

        // Disable
        graph.disable_profiling();
        assert!(!graph.profiling());
        graph.forward(&x).unwrap();
        assert!(graph.profile().is_none());
    }

    #[test]
    fn test_profiling_timing_trend() {
        let graph = FlowBuilder::from(ScalarSum)
            .tag("loss")
            .build()
            .unwrap();

        graph.enable_profiling();

        // Simulate 2 epochs, 3 batches each
        for _ in 0..2 {
            for val in &[1.0f32, 2.0, 3.0] {
                let x = Variable::new(from_f32(&[*val], &[1, 1]), false);
                graph.forward(&x).unwrap();
                graph.collect_timings(&["loss"]);
            }
            graph.flush_timings(&[]);
        }

        let trend = graph.timing_trend("loss");
        assert_eq!(trend.len(), 2, "2 epochs flushed");
        assert!(trend.values()[0] > 0.0, "timing values should be positive");

        // Reset
        graph.reset_timing_trend(&["loss"]);
        assert_eq!(graph.timing_trend("loss").len(), 0);
    }

    // --- DOT tests ---

    #[test]
    fn test_dot_basic() {
        let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .tag("enc")
            .through(ReLU::new())
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let dot = graph.dot();
        assert!(dot.contains("digraph G"));
        assert!(dot.contains("level 0"));
        assert!(dot.contains("#enc"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_dot_with_profile() {
        let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
            .tag("enc")
            .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);

        // Without profiling: dot_with_profile falls back to structural
        let dot1 = graph.dot_with_profile();
        assert!(dot1.contains("digraph G"));

        // With profiling: includes timing annotations
        graph.enable_profiling();
        graph.forward(&x).unwrap();
        let dot2 = graph.dot_with_profile();
        assert!(dot2.contains("digraph G"));
        assert!(dot2.contains("Forward:"));
    }

    // --- Traced tests ---

    /// A loop body that implements trace() — captures per-iteration side data.
    struct TracingDoubler {
        last_output: RefCell<Option<Variable>>,
    }
    impl TracingDoubler {
        fn new() -> Self {
            TracingDoubler {
                last_output: RefCell::new(None),
            }
        }
    }
    impl Module for TracingDoubler {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            let out = input.add(input)?;
            *self.last_output.borrow_mut() = Some(out.clone());
            Ok(out)
        }
        fn trace(&self) -> Option<Variable> {
            self.last_output.borrow().clone()
        }
    }

    #[test]
    fn test_loop_traces() {
        // Loop(TracingDoubler) × 3: [1,2] → [2,4] → [4,8] → [8,16]
        // traces should capture [2,4], [4,8], [8,16]
        let graph = FlowBuilder::from(Identity)
            .loop_body(TracingDoubler::new())
            .for_n(3)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        assert!((data[0] - 8.0).abs() < 1e-5);

        // Get traces — should find them on the loop node
        let traces = graph.traces("any").unwrap();
        assert_eq!(traces.len(), 3, "3 iterations = 3 traces");

        let t0 = traces[0].data().to_f32_vec().unwrap();
        assert!((t0[0] - 2.0).abs() < 1e-5, "iter0: [2,4], got {}", t0[0]);

        let t1 = traces[1].data().to_f32_vec().unwrap();
        assert!((t1[0] - 4.0).abs() < 1e-5, "iter1: [4,8], got {}", t1[0]);

        let t2 = traces[2].data().to_f32_vec().unwrap();
        assert!((t2[0] - 8.0).abs() < 1e-5, "iter2: [8,16], got {}", t2[0]);
    }

    #[test]
    fn test_loop_traces_cleared_each_forward() {
        let graph = FlowBuilder::from(Identity)
            .loop_body(TracingDoubler::new())
            .for_n(2)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0], &[1, 1]), false);
        graph.forward(&x).unwrap();
        let traces1 = graph.traces("any").unwrap();
        assert_eq!(traces1.len(), 2);

        // Second forward should clear and re-populate
        graph.forward(&x).unwrap();
        let traces2 = graph.traces("any").unwrap();
        assert_eq!(traces2.len(), 2);
    }

    #[test]
    fn test_loop_no_traces_without_trace_impl() {
        // Doubler doesn't implement trace() (returns None by default)
        let graph = FlowBuilder::from(Identity)
            .loop_body(Doubler)
            .for_n(3)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0], &[1, 1]), false);
        graph.forward(&x).unwrap();

        // No traces since Doubler's trace() returns None
        assert!(graph.traces("any").is_none());
    }

    // --- Router tests ---

    #[test]
    fn test_softmax_router_gate() {
        // SoftmaxRouter with 2 experts: double + triple, weights from learned router
        let graph = FlowBuilder::from(Identity)
            .gate(
                SoftmaxRouter::on_device(2, 2, crate::tensor::test_device()).unwrap(),
                vec![Box::new(Doubler), Box::new(Tripler)],
            )
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        // Output should be a weighted combination — just verify it runs and has correct shape
        assert_eq!(y.shape(), vec![1, 2]);
        // Router has 2 params (weight + bias), experts have 0
        let params = graph.parameters();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_softmax_router_backward() {
        let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .gate(
                SoftmaxRouter::on_device(2, 2, crate::tensor::test_device()).unwrap(),
                vec![
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                    Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                ],
            )
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} missing gradient", p.name);
        }
    }

    #[test]
    fn test_sigmoid_router_gate() {
        let graph = FlowBuilder::from(Identity)
            .gate(
                SigmoidRouter::on_device(2, 2, crate::tensor::test_device()).unwrap(),
                vec![Box::new(Doubler), Box::new(Tripler)],
            )
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn test_fixed_selector_switch() {
        // FixedSelector(1) always picks branch 1 (Tripler)
        let graph = FlowBuilder::from(Identity)
            .switch(FixedSelector::new(1), vec![Box::new(Doubler), Box::new(Tripler)])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[2.0, 3.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        assert!((data[0] - 6.0).abs() < 1e-5, "triple 2=6, got {}", data[0]);
        assert!((data[1] - 9.0).abs() < 1e-5, "triple 3=9, got {}", data[1]);
    }

    #[test]
    fn test_argmax_selector_switch() {
        let graph = FlowBuilder::from(Identity)
            .switch(
                ArgmaxSelector::on_device(2, 2, crate::tensor::test_device()).unwrap(),
                vec![Box::new(Doubler), Box::new(Tripler)],
            )
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        // Should select one branch — just verify it runs and has correct shape
        assert_eq!(y.shape(), vec![1, 2]);
        // ArgmaxSelector has params from its Linear projection
        assert_eq!(graph.parameters().len(), 2);
    }

    // --- Halt tests ---

    #[test]
    fn test_threshold_halt_while() {
        // body = Doubler, halt when max > 10
        // input [1,2] → iter1 [2,4] → iter2 [4,8] → iter3 [8,16] halt (16 > 10)
        let graph = FlowBuilder::from(Identity)
            .loop_body(Doubler)
            .while_cond(ThresholdHalt::new(10.0), 20)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // Should stop at [8, 16] (max=16 > 10)
        assert!((data[0] - 8.0).abs() < 1e-5, "expected 8, got {}", data[0]);
        assert!((data[1] - 16.0).abs() < 1e-5, "expected 16, got {}", data[1]);
    }

    #[test]
    fn test_threshold_halt_until() {
        // Until: body runs first, then check
        // input [1,2] → iter1 body [2,4] check (max=4 < 10 continue)
        //             → iter2 body [4,8] check (max=8 < 10 continue)
        //             → iter3 body [8,16] check (max=16 > 10 halt)
        let graph = FlowBuilder::from(Identity)
            .loop_body(Doubler)
            .until_cond(ThresholdHalt::new(10.0), 20)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // Should stop at [8, 16] (max=16 > 10)
        assert!((data[0] - 8.0).abs() < 1e-5, "expected 8, got {}", data[0]);
        assert!((data[1] - 16.0).abs() < 1e-5, "expected 16, got {}", data[1]);
    }

    #[test]
    fn test_threshold_halt_immediate() {
        // Threshold already exceeded: while should not iterate
        let graph = FlowBuilder::from(Identity)
            .loop_body(Doubler)
            .while_cond(ThresholdHalt::new(0.5), 20)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // max=2.0 > 0.5 → halt immediately, input passes through
        assert!((data[0] - 1.0).abs() < 1e-5, "expected 1, got {}", data[0]);
        assert!((data[1] - 2.0).abs() < 1e-5, "expected 2, got {}", data[1]);
    }

    #[test]
    fn test_learned_halt_parameters() {
        let graph = FlowBuilder::from(Identity)
            .loop_body(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
            .until_cond(LearnedHalt::on_device(2, crate::tensor::test_device()).unwrap(), 5)
            .build()
            .unwrap();

        // Body Linear: 2 params, LearnedHalt Linear(2→1): 2 params = 4
        let params = graph.parameters();
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_named_parameters_unique() {
        let graph = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(ReLU::new())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let named = graph.named_parameters();
        // Two Linear layers: 2 params each (weight + bias) = 4
        assert_eq!(named.len(), 4);

        // All names should be unique
        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(names.len(), unique.len(), "duplicate names: {:?}", names);
    }

    #[test]
    fn test_named_parameters_tagged_prefix() {
        let graph = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .tag("encoder")
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let named = graph.named_parameters();
        // First Linear is tagged "encoder", second is untagged
        let encoder_params: Vec<&str> = named.iter()
            .filter(|(n, _)| n.starts_with("encoder/"))
            .map(|(n, _)| n.as_str())
            .collect();
        assert_eq!(encoder_params.len(), 2, "tagged node should have 2 params with 'encoder/' prefix");

        // Untagged node uses its node_id (like "linear_2")
        let untagged: Vec<&str> = named.iter()
            .filter(|(n, _)| !n.starts_with("encoder/"))
            .map(|(n, _)| n.as_str())
            .collect();
        assert_eq!(untagged.len(), 2, "untagged node should have 2 params");
        assert!(untagged[0].contains('/'), "should have prefix/name format: {}", untagged[0]);
    }

    // --- Structural hash tests ---

    #[test]
    fn test_structural_hash_deterministic() {
        let g1 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(ReLU::new())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let g2 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(ReLU::new())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        assert_eq!(g1.structural_hash(), g2.structural_hash());
    }

    #[test]
    fn test_structural_hash_differs() {
        let g1 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        // Different architecture: different hidden size
        let g2 = FlowBuilder::from(Linear::on_device(4, 16, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(16, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        assert_ne!(g1.structural_hash(), g2.structural_hash());
    }

    #[test]
    fn test_short_hash_length() {
        let g = FlowBuilder::from(Linear::on_device(2, 3, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        assert_eq!(g.structural_hash().len(), 64);
        assert_eq!(g.short_hash().len(), 8);
        assert!(g.structural_hash().starts_with(g.short_hash()));
    }

    #[test]
    fn test_label_default_none() {
        let g = FlowBuilder::from(Linear::on_device(2, 3, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();
        assert!(g.label().is_none());
    }

    #[test]
    fn test_label_set() {
        let g = FlowBuilder::from(Linear::on_device(2, 3, crate::tensor::test_device()).unwrap())
            .label("my-model")
            .build()
            .unwrap();
        assert_eq!(g.label(), Some("my-model"));
    }

    #[test]
    fn test_label_does_not_affect_hash() {
        let g1 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let g2 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .label("different-label")
            .build()
            .unwrap();

        assert_eq!(g1.structural_hash(), g2.structural_hash());
    }

    #[test]
    fn test_graph_save_load_checkpoint() {
        let g = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .tag("enc")
            .through(ReLU::new())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .tag("dec")
            .build()
            .unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_graph_ckpt.fdl");
        let path_str = path.to_str().unwrap();

        // Save
        g.save_checkpoint(path_str).unwrap();

        // Build identical architecture, load into it
        let g2 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .tag("enc")
            .through(ReLU::new())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .tag("dec")
            .build()
            .unwrap();

        let report = g2.load_checkpoint(path_str).unwrap();
        assert_eq!(report.loaded.len(), 4); // 2 Linear × (weight + bias)
        assert!(report.skipped.is_empty());
        assert!(report.missing.is_empty());

        // Verify weights match
        for ((n1, p1), (n2, p2)) in g.named_parameters().iter().zip(g2.named_parameters().iter()) {
            assert_eq!(n1, n2);
            assert_eq!(p1.variable.data().to_f32_vec().unwrap(),
                       p2.variable.data().to_f32_vec().unwrap());
        }

        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn test_graph_checkpoint_hash_mismatch() {
        let g1 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_graph_ckpt_mismatch.fdl");
        let path_str = path.to_str().unwrap();

        g1.save_checkpoint(path_str).unwrap();

        // Different architecture
        let g2 = FlowBuilder::from(Linear::on_device(4, 16, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(16, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let result = g2.load_checkpoint(path_str);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("architecture mismatch"));

        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn test_graph_checkpoint_gz() {
        let g = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_graph_ckpt.fdl.gz");
        let path_str = path.to_str().unwrap();

        g.save_checkpoint(path_str).unwrap();

        let g2 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
            .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
            .build()
            .unwrap();

        let report = g2.load_checkpoint(path_str).unwrap();
        assert_eq!(report.loaded.len(), 4);

        std::fs::remove_file(path_str).ok();
    }

    // --- collect_with reduction tests ---

    #[test]
    fn test_collect_with_sum_reduction() {
        // Non-scalar tagged output reduced via Sum
        let graph = FlowBuilder::from(Identity)
            .tag("features")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect_with(&["features"], Reduce::Sum).unwrap();

        let collected = graph.collected("features");
        assert_eq!(collected.len(), 1);
        assert!((collected[0] - 6.0).abs() < 1e-5, "sum([1,2,3]) = 6, got {}", collected[0]);
    }

    #[test]
    fn test_collect_with_mean_reduction() {
        let graph = FlowBuilder::from(Identity)
            .tag("out")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[2.0, 4.0, 6.0], &[1, 3]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect_with(&["out"], Reduce::Mean).unwrap();

        let collected = graph.collected("out");
        assert!((collected[0] - 4.0).abs() < 1e-5, "mean([2,4,6]) = 4, got {}", collected[0]);
    }

    #[test]
    fn test_collect_with_max_reduction() {
        let graph = FlowBuilder::from(Identity)
            .tag("out")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 5.0, 3.0], &[1, 3]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect_with(&["out"], Reduce::Max).unwrap();

        let collected = graph.collected("out");
        assert!((collected[0] - 5.0).abs() < 1e-5, "max([1,5,3]) = 5, got {}", collected[0]);
    }

    #[test]
    fn test_collect_with_min_reduction() {
        let graph = FlowBuilder::from(Identity)
            .tag("out")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[-2.0, 0.0, 3.0], &[1, 3]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect_with(&["out"], Reduce::Min).unwrap();

        let collected = graph.collected("out");
        assert!((collected[0] - (-2.0)).abs() < 1e-5, "min([-2,0,3]) = -2, got {}", collected[0]);
    }

    #[test]
    fn test_collect_with_norm_reduction() {
        let graph = FlowBuilder::from(Identity)
            .tag("out")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[3.0, 4.0], &[1, 2]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect_with(&["out"], Reduce::Norm).unwrap();

        let collected = graph.collected("out");
        // L2 norm of [3, 4] = 5
        assert!((collected[0] - 5.0).abs() < 1e-4, "norm([3,4]) = 5, got {}", collected[0]);
    }

    #[test]
    fn test_collect_rejects_non_scalar() {
        // Plain collect() should reject non-scalar outputs
        let graph = FlowBuilder::from(Identity)
            .tag("out")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let _ = graph.forward(&x).unwrap();
        assert!(graph.collect(&["out"]).is_err());
    }

    #[test]
    fn test_collect_with_scalar_passthrough() {
        // collect_with on already-scalar output should work without reduction
        let graph = FlowBuilder::from(ScalarSum)
            .tag("loss")
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[3.0, 7.0], &[1, 2]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect_with(&["loss"], Reduce::Max).unwrap();

        let collected = graph.collected("loss");
        // ScalarSum yields 10.0 (scalar), so it should pass through directly
        assert!((collected[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_collect_with_flush_trend_pipeline() {
        // Full pipeline: non-scalar → reduce → flush → trend
        let graph = FlowBuilder::from(Identity)
            .tag("h")
            .build()
            .unwrap();

        // Epoch 1: two batches with decreasing norms
        let x1 = Variable::new(from_f32(&[3.0, 4.0], &[1, 2]), false);
        let _ = graph.forward(&x1).unwrap();
        graph.collect_with(&["h"], Reduce::Norm).unwrap();

        let x2 = Variable::new(from_f32(&[1.0, 0.0], &[1, 2]), false);
        let _ = graph.forward(&x2).unwrap();
        graph.collect_with(&["h"], Reduce::Norm).unwrap();

        graph.flush(&["h"]);

        // Epoch 2
        let x3 = Variable::new(from_f32(&[0.5, 0.5], &[1, 2]), false);
        let _ = graph.forward(&x3).unwrap();
        graph.collect_with(&["h"], Reduce::Norm).unwrap();
        graph.flush(&["h"]);

        let trend = graph.trend("h");
        assert_eq!(trend.len(), 2);
        // Epoch 1 mean: (5.0 + 1.0) / 2 = 3.0
        assert!((trend.values()[0] - 3.0).abs() < 1e-4);
        assert!(trend.improving(0)); // norms should be decreasing
    }

    // --- Map.over and Map.slices tests ---

    #[test]
    fn test_map_over_tag() {
        // Tag a tensor, then map over it from a different stream position
        let graph = FlowBuilder::from(Identity)
            .tag("features")
            .through(Doubler)        // stream is now 2x
            .map(Doubler)
            .over("features")        // map over original (1x), not current stream (2x)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // .over("features") maps Doubler over the tagged value (original x)
        // Doubler: x + x = 2x, applied element-wise along dim 0
        assert_eq!(y.shape(), vec![2, 2]);
        assert!((data[0] - 2.0).abs() < 1e-5);  // 1.0 * 2
        assert!((data[1] - 4.0).abs() < 1e-5);  // 2.0 * 2
        assert!((data[2] - 6.0).abs() < 1e-5);  // 3.0 * 2
        assert!((data[3] - 8.0).abs() < 1e-5);  // 4.0 * 2
    }

    #[test]
    fn test_map_over_unknown_tag_error() {
        let result = FlowBuilder::from(Identity)
            .map(Doubler)
            .over("nonexistent")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_map_slices() {
        // Input [2, 4], slices(2): decompose → [4, 2], map Doubler, recompose → [2, 4]
        let graph = FlowBuilder::from(Identity)
            .map(Doubler)
            .slices(2)
            .build()
            .unwrap();

        let x = Variable::new(
            from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]),
            false,
        );
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        // Each element doubled
        assert_eq!(y.shape(), vec![2, 4]);
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[7] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_map_slices_batched() {
        // Same as above but with batched fast path
        let graph = FlowBuilder::from(Identity)
            .map(Doubler)
            .batched()
            .slices(2)
            .build()
            .unwrap();

        let x = Variable::new(
            from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]),
            false,
        );
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert_eq!(y.shape(), vec![2, 4]);
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[7] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_map_slices_gradient() {
        // Input [2, 4] → slices(2) decomposes to [4, 2] → Linear(2, 3) → [4, 3] → recompose [2, 6]
        let graph = FlowBuilder::from(Identity)
            .map(Linear::on_device(2, 3, crate::tensor::test_device()).unwrap())
            .slices(2)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]), true);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 6]); // 3 * 2 slices = 6
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_map_slices_not_divisible_error() {
        let graph = FlowBuilder::from(Identity)
            .map(Doubler)
            .slices(3)
            .build()
            .unwrap();

        // [2, 4] with slices(3) — 4 not divisible by 3
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]), false);
        assert!(graph.forward(&x).is_err());
    }

    // -----------------------------------------------------------------------
    // Graph::set_scheduler -- regression guard
    // -----------------------------------------------------------------------
    //
    // Original bug (2026-04-13): sync mode (Ddp::setup_with + graph.step())
    // had no scheduler plumbing, so the optimizer LR stayed constant for the
    // entire run regardless of what scheduler the user attached. These tests
    // assert that set_scheduler drives the optimizer LR through step(), that
    // training_step advances once per step(), and that lr_scale is applied
    // multiplicatively.

    /// Trivial scheduler used to assert the LR pipeline.
    struct LinearSched(f64);
    impl crate::nn::Scheduler for LinearSched {
        fn lr(&self, step: usize) -> f64 { step as f64 * self.0 }
    }

    /// Build a tiny Graph + optimizer + a fake gradient so step() can run end
    /// to end on CPU. Keeps the test cheap (no CUDA needed).
    fn graph_with_optim(initial_lr: f64) -> (crate::graph::Graph, Variable) {
        use crate::nn::SGD;
        let dev = crate::tensor::test_device();
        let graph = FlowBuilder::from(Linear::on_device(2, 1, dev).unwrap())
            .build()
            .unwrap();
        graph.set_optimizer(|p| SGD::new(p, initial_lr, 0.0));
        // Run one forward+backward so .grad() is populated and step() can do work.
        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        (graph, x)
    }

    fn current_optim_lr(graph: &crate::graph::Graph) -> f64 {
        graph.optimizer.borrow().as_ref().map(|o| o.lr()).unwrap()
    }

    #[test]
    fn test_graph_set_scheduler_drives_optimizer_lr() {
        let (graph, x) = graph_with_optim(0.0); // start at 0 so we detect writes
        graph.set_scheduler(std::sync::Arc::new(LinearSched(0.1)));
        assert_eq!(graph.training_step(), 0);

        // Three step()s: scheduler queried at training_step before increment.
        for expected_step in 0..3 {
            // Forward + backward to populate gradients (step() needs them).
            let y = graph.forward(&x).unwrap();
            y.sum().unwrap().backward().unwrap();
            graph.step().unwrap();
            // After step(), training_step has advanced and the LR set BEFORE
            // optimizer.step() reflects the *previous* training_step value.
            let expected_lr = expected_step as f64 * 0.1;
            assert!((current_optim_lr(&graph) - expected_lr).abs() < 1e-9,
                "after step {}: expected LR {expected_lr}, got {}",
                expected_step + 1, current_optim_lr(&graph));
            assert_eq!(graph.training_step(), expected_step + 1);
        }
    }

    #[test]
    fn test_graph_lr_scale_multiplies_scheduler_output() {
        let (graph, x) = graph_with_optim(0.0);
        graph.set_scheduler(std::sync::Arc::new(LinearSched(0.1)));
        graph.set_lr_scale(2.5);

        let y = graph.forward(&x).unwrap();
        y.sum().unwrap().backward().unwrap();
        graph.step().unwrap();
        // Step 0: scheduler returns 0.0 -> 0.0 * 2.5 = 0.0 (boring)
        assert!(current_optim_lr(&graph).abs() < 1e-9);

        let y = graph.forward(&x).unwrap();
        y.sum().unwrap().backward().unwrap();
        graph.step().unwrap();
        // Step 1: scheduler returns 0.1 -> 0.1 * 2.5 = 0.25
        assert!((current_optim_lr(&graph) - 0.25).abs() < 1e-9,
            "expected LR 0.25 (sched 0.1 * scale 2.5), got {}",
            current_optim_lr(&graph));
    }

    #[test]
    fn test_graph_no_scheduler_leaves_lr_alone() {
        // Without a scheduler, step() must NOT touch the optimizer's LR.
        let (graph, x) = graph_with_optim(0.123);
        // Don't attach any scheduler.
        let y = graph.forward(&x).unwrap();
        y.sum().unwrap().backward().unwrap();
        graph.step().unwrap();
        assert!((current_optim_lr(&graph) - 0.123).abs() < 1e-9,
            "no scheduler attached: LR must be untouched, got {}",
            current_optim_lr(&graph));
        // training_step still increments (it's a per-step counter, scheduler-independent).
        assert_eq!(graph.training_step(), 1);
    }
