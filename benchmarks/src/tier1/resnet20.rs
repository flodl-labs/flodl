//! ResNet-20 benchmark: CIFAR-shaped input, SGD with momentum and weight decay.
//!
//! The launch-heavy case: 65 small parameter tensors and a step of a few
//! milliseconds, so per-step framework overhead (the optimizer step, the
//! dispatch of many small kernels) is a visible share of the time. The
//! other tier-1 entries are dominated by a few large kernels and use Adam.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

struct Block {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    skip: Option<(Conv2d, BatchNorm2d)>,
}

impl Block {
    fn new(in_ch: i64, out_ch: i64, stride: i64, device: Device) -> Result<Self> {
        let conv = |i, o, k, s, p| {
            Conv2d::configure(i, o, k)
                .with_stride(s)
                .with_padding(p)
                .without_bias()
                .on_device(device)
                .done()
        };
        let skip = if stride != 1 || in_ch != out_ch {
            Some((conv(in_ch, out_ch, 1, stride, 0)?, BatchNorm2d::on_device(out_ch, device)?))
        } else {
            None
        };
        Ok(Block {
            conv1: conv(in_ch, out_ch, 3, stride, 1)?,
            bn1: BatchNorm2d::on_device(out_ch, device)?,
            conv2: conv(out_ch, out_ch, 3, 1, 1)?,
            bn2: BatchNorm2d::on_device(out_ch, device)?,
            skip,
        })
    }

    fn forward(&self, x: &Variable) -> Result<Variable> {
        let identity = match &self.skip {
            Some((conv, bn)) => bn.forward(&conv.forward(x)?)?,
            None => x.clone(),
        };
        let out = self.bn1.forward(&self.conv1.forward(x)?)?.relu()?;
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?;
        out.add(&identity)?.relu()
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some((conv, bn)) = &self.skip {
            params.extend(conv.parameters());
            params.extend(bn.parameters());
        }
        params
    }
}

/// ResNet-20 for 32x32 inputs (He et al. 2015): a stem and three stages of
/// three basic blocks at 16, 32 and 64 channels.
pub struct ResNet20 {
    stem: Conv2d,
    stem_bn: BatchNorm2d,
    blocks: Vec<Block>,
    fc: Linear,
}

impl ResNet20 {
    pub fn new(device: Device) -> Result<Self> {
        let stem = Conv2d::configure(3, 16, 3)
            .with_padding(1)
            .without_bias()
            .on_device(device)
            .done()?;
        let mut blocks = Vec::new();
        for (in_ch, out_ch, stride) in [(16, 16, 1), (16, 32, 2), (32, 64, 2)] {
            blocks.push(Block::new(in_ch, out_ch, stride, device)?);
            for _ in 1..3 {
                blocks.push(Block::new(out_ch, out_ch, 1, device)?);
            }
        }
        Ok(ResNet20 {
            stem,
            stem_bn: BatchNorm2d::on_device(16, device)?,
            blocks,
            fc: Linear::on_device(64, 10, device)?,
        })
    }

    pub fn forward(&self, input: &Variable) -> Result<Variable> {
        let mut x = self.stem_bn.forward(&self.stem.forward(input)?)?.relu()?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = adaptive_avg_pool2d(&x, [1, 1])?;
        let x = x.reshape(&[x.shape()[0], -1])?;
        self.fc.forward(&x)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.stem.parameters());
        params.extend(self.stem_bn.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.fc.parameters());
        params
    }
}

pub fn run(device: Device) -> Result<BenchResult> {
    let config = BenchConfig {
        name: "resnet20".into(),
        batch_size: 64,
        batches_per_epoch: 100,
        ..Default::default()
    };

    let model = ResNet20::new(device)?;
    let params = model.parameters();
    let mut optimizer = SGD::new(&params, 0.1, 0.9).weight_decay(1e-4);

    let image_opts = TensorOptions { dtype: DType::Float32, device };
    let label_opts = TensorOptions { dtype: DType::Int64, device };
    let batches: Vec<(Tensor, Tensor)> = (0..config.batches_per_epoch)
        .map(|_| {
            let x = Tensor::randn(&[config.batch_size as i64, 3, 32, 32], image_opts).unwrap();
            let y = Tensor::randint(0, 10, &[config.batch_size as i64], label_opts).unwrap();
            (x, y)
        })
        .collect();

    run_benchmark(&config, &params, |_epoch, _warmup| {
        let mut total_loss = 0.0;
        for (x, y) in &batches {
            let input = Variable::new(x.clone(), false);
            let target = Variable::new(y.clone(), false);
            let loss = cross_entropy_loss(&model.forward(&input)?, &target)?;

            optimizer.zero_grad();
            loss.backward()?;
            optimizer.step()?;

            total_loss += loss.item()?;
        }
        Ok(total_loss / batches.len() as f64)
    })
}
