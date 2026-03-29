use std::io::{Read, Write};

use crate::autograd::{Variable, no_grad};
use crate::tensor::Result;

use super::checkpoint::{
    write_tensor_state, read_tensor_state, write_f64_le, read_f64_le,
    write_u32_le, read_u32_le, write_i64_le, read_i64_le,
};
use super::parameter::Parameter;

/// Optimizer trait: step, zero gradients, and adjust learning rate.
pub trait Optimizer {
    /// Perform a single optimization step using accumulated gradients.
    fn step(&mut self) -> Result<()>;
    /// Reset all parameter gradients to zero.
    fn zero_grad(&self);
    /// Update the learning rate (all groups if grouped).
    fn set_lr(&mut self, lr: f64);
    /// Set learning rate for a specific parameter group (0-indexed).
    /// Falls back to `set_lr` for single-group optimizers.
    fn set_group_lr(&mut self, _group: usize, lr: f64) {
        self.set_lr(lr);
    }
}

/// Per-group learning rate metadata.
struct GroupMeta {
    lr: f64,
    range: std::ops::Range<usize>,
}

/// Save/load training state (learning rates, momentum buffers, step counters).
/// Implement for optimizers and other stateful training components.
pub trait Stateful {
    /// Serialize optimizer state (lr, momentum buffers, etc.) to a writer.
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()>;
    /// Restore optimizer state from a reader.
    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()>;

    /// Save state to a file. Uses gzip compression if path ends with `.gz`.
    fn save_state_file(&self, path: &str) -> Result<()> {
        let f = std::fs::File::create(path).map_err(|e| {
            crate::tensor::TensorError::new(&format!("io: {}", e))
        })?;
        if path.ends_with(".gz") {
            let mut w = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            self.save_state(&mut w)?;
            w.finish().map_err(|e| {
                crate::tensor::TensorError::new(&format!("io: {}", e))
            })?;
            Ok(())
        } else {
            let mut w = std::io::BufWriter::new(f);
            self.save_state(&mut w)
        }
    }

    /// Load state from a file. Detects gzip from `.gz` extension.
    fn load_state_file(&mut self, path: &str) -> Result<()> {
        let f = std::fs::File::open(path).map_err(|e| {
            crate::tensor::TensorError::new(&format!("io: {}", e))
        })?;
        if path.ends_with(".gz") {
            let mut r = flate2::read::GzDecoder::new(f);
            self.load_state(&mut r)
        } else {
            let mut r = std::io::BufReader::new(f);
            self.load_state(&mut r)
        }
    }
}

/// SGD with optional momentum.
///
/// With momentum: `v = momentum * v + grad; param -= lr * v`.
/// Without momentum: `param -= lr * grad`.
///
/// ```ignore
/// let mut optim = SGD::new(&model.parameters(), 0.01, 0.9);
/// for batch in &data {
///     optim.zero_grad();
///     let loss = mse_loss(&model.forward(&batch.x)?, &batch.y)?;
///     loss.backward()?;
///     optim.step()?;
/// }
/// ```
pub struct SGD {
    params: Vec<Variable>,
    lr: f64,
    momentum: f64,
    velocity: Vec<Option<crate::tensor::Tensor>>,
    groups: Vec<GroupMeta>,
}

impl SGD {
    /// Create a new SGD optimizer. Set `momentum` to 0.0 for vanilla SGD.
    pub fn new(params: &[Parameter], lr: f64, momentum: f64) -> Self {
        let variables: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        let velocity = vec![None; variables.len()];
        SGD {
            params: variables,
            lr,
            momentum,
            velocity,
            groups: vec![],
        }
    }

    /// Create a builder for SGD with per-group learning rates.
    pub fn with_groups(momentum: f64) -> SGDBuilder {
        SGDBuilder { momentum, groups: vec![] }
    }

    /// Current learning rate (base LR, or first group's LR).
    pub fn lr(&self) -> f64 {
        self.lr
    }

    fn lr_for_param(&self, i: usize) -> f64 {
        for g in &self.groups {
            if g.range.contains(&i) {
                return g.lr;
            }
        }
        self.lr
    }
}

/// Builder for SGD with per-group learning rates.
pub struct SGDBuilder {
    momentum: f64,
    groups: Vec<(Vec<Variable>, f64)>,
}

impl SGDBuilder {
    /// Add a parameter group with its own learning rate.
    pub fn group(mut self, params: &[Parameter], lr: f64) -> Self {
        let vars: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        self.groups.push((vars, lr));
        self
    }

    /// Build the SGD optimizer.
    pub fn build(self) -> SGD {
        let mut all_params = Vec::new();
        let mut groups = Vec::new();
        let base_lr = self.groups.first().map(|(_, lr)| *lr).unwrap_or(0.01);

        for (vars, lr) in self.groups {
            let start = all_params.len();
            all_params.extend(vars);
            let end = all_params.len();
            groups.push(GroupMeta { lr, range: start..end });
        }

        let velocity = vec![None; all_params.len()];
        SGD {
            params: all_params,
            lr: base_lr,
            momentum: self.momentum,
            velocity,
            groups,
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (i, param) in self.params.iter().enumerate() {
                if let Some(grad) = param.grad() {
                    let lr = self.lr_for_param(i);
                    let data = param.data().detach()?;
                    if self.momentum > 0.0 {
                        let v = match self.velocity[i].take() {
                            Some(v) => {
                                v.mul_scalar_(self.momentum)?;
                                v.add_(&grad)?;
                                v
                            }
                            // mul_scalar creates a new tensor with independent storage,
                            // unlike clone() which shares storage with grad
                            None => grad.mul_scalar(1.0)?,
                        };
                        // data -= lr * v
                        let scaled = v.mul_scalar(lr)?;
                        data.sub_(&scaled)?;
                        self.velocity[i] = Some(v);
                    } else {
                        // data -= lr * grad
                        let scaled = grad.mul_scalar(lr)?;
                        data.sub_(&scaled)?;
                    }
                }
            }
            Ok(())
        })
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad_set_to_none();
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
        for g in &mut self.groups {
            g.lr = lr;
        }
    }

    fn set_group_lr(&mut self, group: usize, lr: f64) {
        if let Some(g) = self.groups.get_mut(group) {
            g.lr = lr;
        }
    }
}

impl Stateful for SGD {
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, self.params.len() as u32)?;
        write_f64_le(w, self.lr)?;
        for v in &self.velocity {
            write_tensor_state(w, v.as_ref())?;
        }
        // Groups
        write_u32_le(w, self.groups.len() as u32)?;
        for g in &self.groups {
            write_f64_le(w, g.lr)?;
            write_i64_le(w, g.range.start as i64)?;
            write_i64_le(w, g.range.end as i64)?;
        }
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let count = read_u32_le(r)? as usize;
        if count != self.params.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "SGD: param count mismatch: checkpoint={} optimizer={}", count, self.params.len()
            )));
        }
        self.lr = read_f64_le(r)?;
        for (i, param) in self.params.iter().enumerate() {
            let dev = param.data().device();
            self.velocity[i] = read_tensor_state(r, dev)?;
        }
        // Groups
        let ng = read_u32_le(r)? as usize;
        self.groups.clear();
        for _ in 0..ng {
            let lr = read_f64_le(r)?;
            let start = read_i64_le(r)? as usize;
            let end = read_i64_le(r)? as usize;
            self.groups.push(GroupMeta { lr, range: start..end });
        }
        Ok(())
    }
}

/// Adam optimizer with bias correction (Kingma & Ba, 2014).
///
/// Maintains per-parameter first and second moment estimates with
/// bias correction. Default betas: (0.9, 0.999), eps: 1e-8.
///
/// ```ignore
/// let mut optim = Adam::new(&model.parameters(), 0.001);
/// ```
pub struct Adam {
    params: Vec<Variable>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    m: Vec<Option<crate::tensor::Tensor>>,
    v: Vec<Option<crate::tensor::Tensor>>,
    t: usize,
    groups: Vec<GroupMeta>,
}

impl Adam {
    /// Create a new Adam optimizer with default betas (0.9, 0.999) and eps (1e-8).
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        Adam {
            params: params.iter().map(|p| p.variable.clone()).collect(),
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: vec![None; n],
            v: vec![None; n],
            t: 0,
            groups: vec![],
        }
    }

    /// Create a builder for Adam with per-group learning rates.
    pub fn with_groups() -> AdamBuilder {
        AdamBuilder { groups: vec![] }
    }

    /// Current learning rate (base LR, or first group's LR).
    pub fn lr(&self) -> f64 {
        self.lr
    }
}

/// Builder for Adam with per-group learning rates.
pub struct AdamBuilder {
    groups: Vec<(Vec<Variable>, f64)>,
}

impl AdamBuilder {
    /// Add a parameter group with its own learning rate.
    pub fn group(mut self, params: &[Parameter], lr: f64) -> Self {
        let vars: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        self.groups.push((vars, lr));
        self
    }

    /// Build the Adam optimizer.
    pub fn build(self) -> Adam {
        let mut all_params = Vec::new();
        let mut groups = Vec::new();
        let base_lr = self.groups.first().map(|(_, lr)| *lr).unwrap_or(1e-3);

        for (vars, lr) in self.groups {
            let start = all_params.len();
            all_params.extend(vars);
            let end = all_params.len();
            groups.push(GroupMeta { lr, range: start..end });
        }

        let n = all_params.len();
        Adam {
            params: all_params,
            lr: base_lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: vec![None; n],
            v: vec![None; n],
            t: 0,
            groups,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> Result<()> {
        self.adam_update(0.0)
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad_set_to_none();
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
        for g in &mut self.groups {
            g.lr = lr;
        }
    }

    fn set_group_lr(&mut self, group: usize, lr: f64) {
        if let Some(g) = self.groups.get_mut(group) {
            g.lr = lr;
        }
    }
}

impl Adam {
    fn adam_update(&mut self, weight_decay: f64) -> Result<()> {
        self.t += 1;

        no_grad(|| {
            // Determine effective groups (single group if none configured)
            let effective_groups: Vec<(f64, std::ops::Range<usize>)> = if self.groups.is_empty() {
                vec![(self.lr, 0..self.params.len())]
            } else {
                self.groups.iter().map(|g| (g.lr, g.range.clone())).collect()
            };

            for (lr, range) in &effective_groups {
                let mut p_tensors = Vec::new();
                let mut g_tensors = Vec::new();
                let mut m_tensors = Vec::new();
                let mut v_tensors = Vec::new();

                for i in range.clone() {
                    if let Some(grad) = self.params[i].grad() {
                        // Lazy-init moment buffers as zeros on first step
                        if self.m[i].is_none() {
                            self.m[i] = Some(crate::tensor::Tensor::zeros_like(&grad)?);
                        }
                        if self.v[i].is_none() {
                            self.v[i] = Some(crate::tensor::Tensor::zeros_like(&grad)?);
                        }

                        p_tensors.push(self.params[i].data());
                        g_tensors.push(grad);
                        m_tensors.push(self.m[i].as_ref().unwrap().clone());
                        v_tensors.push(self.v[i].as_ref().unwrap().clone());
                    }
                }

                if !p_tensors.is_empty() {
                    // Single fused kernel for all params in this group
                    crate::tensor::Tensor::fused_adamw_(
                        &p_tensors, &g_tensors, &m_tensors, &v_tensors,
                        *lr, self.beta1, self.beta2, self.eps,
                        weight_decay, self.t as i64, None, None,
                    )?;
                }
            }
            Ok(())
        })
    }
}

impl Stateful for Adam {
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, self.params.len() as u32)?;
        write_f64_le(w, self.lr)?;
        write_i64_le(w, self.t as i64)?;
        for i in 0..self.params.len() {
            write_tensor_state(w, self.m[i].as_ref())?;
            write_tensor_state(w, self.v[i].as_ref())?;
        }
        // Groups
        write_u32_le(w, self.groups.len() as u32)?;
        for g in &self.groups {
            write_f64_le(w, g.lr)?;
            write_i64_le(w, g.range.start as i64)?;
            write_i64_le(w, g.range.end as i64)?;
        }
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let count = read_u32_le(r)? as usize;
        if count != self.params.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "Adam: param count mismatch: checkpoint={} optimizer={}", count, self.params.len()
            )));
        }
        self.lr = read_f64_le(r)?;
        self.t = read_i64_le(r)? as usize;
        for i in 0..self.params.len() {
            let dev = self.params[i].data().device();
            self.m[i] = read_tensor_state(r, dev)?;
            self.v[i] = read_tensor_state(r, dev)?;
        }
        // Groups
        let ng = read_u32_le(r)? as usize;
        self.groups.clear();
        for _ in 0..ng {
            let lr = read_f64_le(r)?;
            let start = read_i64_le(r)? as usize;
            let end = read_i64_le(r)? as usize;
            self.groups.push(GroupMeta { lr, range: start..end });
        }
        Ok(())
    }
}

/// AdamW optimizer — Adam with decoupled weight decay (Loshchilov & Hutter, 2017).
///
/// Unlike L2 regularization, weight decay is applied directly to parameters,
/// not to gradients. This distinction matters for adaptive optimizers and
/// generally improves generalization.
///
/// ```ignore
/// let mut optim = AdamW::new(&model.parameters(), 0.001, 0.01);
/// ```
pub struct AdamW {
    adam: Adam,
    weight_decay: f64,
}

impl AdamW {
    /// Create a new AdamW optimizer. `weight_decay` is applied directly to
    /// parameters (decoupled), not to gradients. Typical values: 0.01--0.1.
    pub fn new(params: &[Parameter], lr: f64, weight_decay: f64) -> Self {
        AdamW {
            adam: Adam::new(params, lr),
            weight_decay,
        }
    }

    /// Create a builder for AdamW with per-group learning rates.
    pub fn with_groups(weight_decay: f64) -> AdamWBuilder {
        AdamWBuilder { weight_decay, groups: vec![] }
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 {
        self.adam.lr
    }
}

/// Builder for AdamW with per-group learning rates.
pub struct AdamWBuilder {
    weight_decay: f64,
    groups: Vec<(Vec<Variable>, f64)>,
}

impl AdamWBuilder {
    /// Add a parameter group with its own learning rate.
    pub fn group(mut self, params: &[Parameter], lr: f64) -> Self {
        let vars: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        self.groups.push((vars, lr));
        self
    }

    /// Build the AdamW optimizer.
    pub fn build(self) -> AdamW {
        let mut all_params = Vec::new();
        let mut groups = Vec::new();
        let base_lr = self.groups.first().map(|(_, lr)| *lr).unwrap_or(1e-3);

        for (vars, lr) in self.groups {
            let start = all_params.len();
            all_params.extend(vars);
            let end = all_params.len();
            groups.push(GroupMeta { lr, range: start..end });
        }

        let n = all_params.len();
        AdamW {
            adam: Adam {
                params: all_params,
                lr: base_lr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                m: vec![None; n],
                v: vec![None; n],
                t: 0,
                groups,
            },
            weight_decay: self.weight_decay,
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) -> Result<()> {
        self.adam.adam_update(self.weight_decay)
    }

    fn zero_grad(&self) {
        self.adam.zero_grad()
    }

    fn set_lr(&mut self, lr: f64) {
        self.adam.set_lr(lr);
    }

    fn set_group_lr(&mut self, group: usize, lr: f64) {
        self.adam.set_group_lr(group, lr);
    }
}

impl Stateful for AdamW {
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_f64_le(w, self.weight_decay)?;
        self.adam.save_state(w)
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        self.weight_decay = read_f64_le(r)?;
        self.adam.load_state(r)
    }
}

/// RMSprop optimizer (Hinton, 2012).
///
/// Maintains a running average of squared gradients to normalize the update.
/// Optionally supports momentum and weight decay.
///
/// Update rule (without momentum):
///   v = alpha * v + (1 - alpha) * grad^2
///   param -= lr * grad / (sqrt(v) + eps)
///
/// With momentum:
///   v = alpha * v + (1 - alpha) * grad^2
///   buf = momentum * buf + grad / (sqrt(v) + eps)
///   param -= lr * buf
///
/// ```ignore
/// let mut optim = RMSprop::new(&model.parameters(), 0.01);
/// // Or with options:
/// let mut optim = RMSprop::builder(&model.parameters(), 0.01)
///     .alpha(0.99)
///     .momentum(0.9)
///     .weight_decay(1e-4)
///     .build();
/// ```
pub struct RMSprop {
    params: Vec<Variable>,
    lr: f64,
    alpha: f64,
    eps: f64,
    weight_decay: f64,
    momentum: f64,
    /// Running average of squared gradients
    v: Vec<Option<crate::tensor::Tensor>>,
    /// Momentum buffer (only used when momentum > 0)
    buf: Vec<Option<crate::tensor::Tensor>>,
    groups: Vec<GroupMeta>,
}

impl RMSprop {
    /// Create a new RMSprop optimizer with default parameters:
    /// alpha=0.99, eps=1e-8, weight_decay=0, momentum=0.
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        RMSprop {
            params: params.iter().map(|p| p.variable.clone()).collect(),
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            v: vec![None; n],
            buf: vec![None; n],
            groups: vec![],
        }
    }

    /// Create a builder for RMSprop with customizable options.
    pub fn builder(params: &[Parameter], lr: f64) -> RMSpropBuilder {
        RMSpropBuilder {
            params: params.to_vec(),
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
        }
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 {
        self.lr
    }

    fn lr_for_param(&self, i: usize) -> f64 {
        for g in &self.groups {
            if g.range.contains(&i) {
                return g.lr;
            }
        }
        self.lr
    }
}

/// Builder for RMSprop with customizable options.
pub struct RMSpropBuilder {
    params: Vec<Parameter>,
    lr: f64,
    alpha: f64,
    eps: f64,
    weight_decay: f64,
    momentum: f64,
}

impl RMSpropBuilder {
    /// Smoothing constant (default 0.99).
    pub fn alpha(mut self, alpha: f64) -> Self { self.alpha = alpha; self }
    /// Term added for numerical stability (default 1e-8).
    pub fn eps(mut self, eps: f64) -> Self { self.eps = eps; self }
    /// Weight decay (L2 penalty, default 0).
    pub fn weight_decay(mut self, wd: f64) -> Self { self.weight_decay = wd; self }
    /// Momentum factor (default 0).
    pub fn momentum(mut self, momentum: f64) -> Self { self.momentum = momentum; self }

    /// Build the RMSprop optimizer.
    pub fn build(self) -> RMSprop {
        let n = self.params.len();
        RMSprop {
            params: self.params.iter().map(|p| p.variable.clone()).collect(),
            lr: self.lr,
            alpha: self.alpha,
            eps: self.eps,
            weight_decay: self.weight_decay,
            momentum: self.momentum,
            v: vec![None; n],
            buf: vec![None; n],
            groups: vec![],
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (i, param) in self.params.iter().enumerate() {
                if let Some(mut grad) = param.grad() {
                    let lr = self.lr_for_param(i);
                    let data = param.data().detach()?;

                    // Weight decay: add wd * param to gradient
                    if self.weight_decay > 0.0 {
                        grad = grad.add(&data.mul_scalar(self.weight_decay)?)?;
                    }

                    // Update running average of squared gradients
                    // v = alpha * v + (1 - alpha) * grad^2
                    let grad_sq = grad.mul(&grad)?;
                    let v = match self.v[i].take() {
                        Some(v) => {
                            v.mul_scalar_(self.alpha)?;
                            let scaled = grad_sq.mul_scalar(1.0 - self.alpha)?;
                            v.add_(&scaled)?;
                            v
                        }
                        None => grad_sq.mul_scalar(1.0 - self.alpha)?,
                    };

                    // Compute update: grad / (sqrt(v) + eps)
                    let denom = v.sqrt()?.add_scalar(self.eps)?;
                    let update = grad.div(&denom)?;

                    if self.momentum > 0.0 {
                        // buf = momentum * buf + update
                        let b = match self.buf[i].take() {
                            Some(b) => {
                                b.mul_scalar_(self.momentum)?;
                                b.add_(&update)?;
                                b
                            }
                            None => update.mul_scalar(1.0)?,
                        };
                        let scaled = b.mul_scalar(lr)?;
                        data.sub_(&scaled)?;
                        self.buf[i] = Some(b);
                    } else {
                        let scaled = update.mul_scalar(lr)?;
                        data.sub_(&scaled)?;
                    }

                    self.v[i] = Some(v);
                }
            }
            Ok(())
        })
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad_set_to_none();
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
        for g in &mut self.groups {
            g.lr = lr;
        }
    }

    fn set_group_lr(&mut self, group: usize, lr: f64) {
        if let Some(g) = self.groups.get_mut(group) {
            g.lr = lr;
        }
    }
}

impl Stateful for RMSprop {
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, self.params.len() as u32)?;
        write_f64_le(w, self.lr)?;
        write_f64_le(w, self.alpha)?;
        write_f64_le(w, self.eps)?;
        write_f64_le(w, self.weight_decay)?;
        write_f64_le(w, self.momentum)?;
        for i in 0..self.params.len() {
            write_tensor_state(w, self.v[i].as_ref())?;
            write_tensor_state(w, self.buf[i].as_ref())?;
        }
        // Groups
        write_u32_le(w, self.groups.len() as u32)?;
        for g in &self.groups {
            write_f64_le(w, g.lr)?;
            write_i64_le(w, g.range.start as i64)?;
            write_i64_le(w, g.range.end as i64)?;
        }
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let count = read_u32_le(r)? as usize;
        if count != self.params.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "RMSprop: param count mismatch: checkpoint={} optimizer={}", count, self.params.len()
            )));
        }
        self.lr = read_f64_le(r)?;
        self.alpha = read_f64_le(r)?;
        self.eps = read_f64_le(r)?;
        self.weight_decay = read_f64_le(r)?;
        self.momentum = read_f64_le(r)?;
        for i in 0..self.params.len() {
            let dev = self.params[i].data().device();
            self.v[i] = read_tensor_state(r, dev)?;
            self.buf[i] = read_tensor_state(r, dev)?;
        }
        // Groups
        let ng = read_u32_le(r)? as usize;
        self.groups.clear();
        for _ in 0..ng {
            let lr = read_f64_le(r)?;
            let start = read_i64_le(r)? as usize;
            let end = read_i64_le(r)? as usize;
            self.groups.push(GroupMeta { lr, range: start..end });
        }
        Ok(())
    }
}

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
    step_count: u64,
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
    pub fn eps(mut self, eps: f64) -> Self { self.eps = eps; self }
    pub fn weight_decay(mut self, wd: f64) -> Self { self.weight_decay = wd; self }
    pub fn lr_decay(mut self, lr_decay: f64) -> Self { self.lr_decay = lr_decay; self }

    pub fn build(self) -> Adagrad {
        let n = self.params.len();
        Adagrad {
            params: self.params.iter().map(|p| p.variable.clone()).collect(),
            lr: self.lr, eps: self.eps,
            weight_decay: self.weight_decay, lr_decay: self.lr_decay,
            state_sum: vec![None; n],
            step_count: 0,
        }
    }
}

impl Adagrad {
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        Adagrad {
            params: params.iter().map(|p| p.variable.clone()).collect(),
            lr, eps: 1e-10, weight_decay: 0.0, lr_decay: 0.0,
            state_sum: vec![None; n],
            step_count: 0,
        }
    }

    pub fn builder(params: &[Parameter], lr: f64) -> AdagradBuilder {
        AdagradBuilder {
            params: params.to_vec(), lr, eps: 1e-10, weight_decay: 0.0, lr_decay: 0.0,
        }
    }

    pub fn lr(&self) -> f64 { self.lr }
}

impl Optimizer for Adagrad {
    fn step(&mut self) -> Result<()> {
        self.step_count += 1;
        let clr = self.lr / (1.0 + (self.step_count - 1) as f64 * self.lr_decay);
        no_grad(|| {
            for (i, param) in self.params.iter().enumerate() {
                if let Some(mut grad) = param.grad() {
                    let data = param.data().detach()?;
                    if self.weight_decay > 0.0 {
                        grad = grad.add(&data.mul_scalar(self.weight_decay)?)?;
                    }
                    let grad2 = grad.mul(&grad)?;
                    let ss = match self.state_sum[i].take() {
                        Some(ss) => ss.add(&grad2)?,
                        None => grad2,
                    };
                    let update = grad.div(&ss.sqrt()?.add_scalar(self.eps)?)?.mul_scalar(clr)?;
                    data.sub_(&update)?;
                    self.state_sum[i] = Some(ss);
                }
            }
            Ok(())
        })
    }

    fn zero_grad(&self) {
        for p in &self.params { p.zero_grad_set_to_none(); }
    }

    fn set_lr(&mut self, lr: f64) { self.lr = lr; }
}

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
    step_count: u64,
}

impl RAdam {
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        RAdam {
            params: params.iter().map(|p| p.variable.clone()).collect(),
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0,
            m: vec![None; n], v: vec![None; n], step_count: 0,
        }
    }

    pub fn lr(&self) -> f64 { self.lr }
}

impl Optimizer for RAdam {
    fn step(&mut self) -> Result<()> {
        self.step_count += 1;
        let t = self.step_count as f64;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let b1t = b1.powf(t);
        let b2t = b2.powf(t);
        // Maximum length of approximated SMA
        let rho_inf = 2.0 / (1.0 - b2) - 1.0;
        let rho_t = rho_inf - 2.0 * t * b2t / (1.0 - b2t);

        no_grad(|| {
            for (i, param) in self.params.iter().enumerate() {
                if let Some(mut grad) = param.grad() {
                    let data = param.data().detach()?;
                    if self.weight_decay > 0.0 {
                        grad = grad.add(&data.mul_scalar(self.weight_decay)?)?;
                    }

                    // Update biased first moment
                    let m_new = match self.m[i].take() {
                        Some(m) => m.mul_scalar(b1)?.add(&grad.mul_scalar(1.0 - b1)?)?,
                        None => grad.mul_scalar(1.0 - b1)?,
                    };
                    // Update biased second moment
                    let grad2 = grad.mul(&grad)?;
                    let v_new = match self.v[i].take() {
                        Some(v) => v.mul_scalar(b2)?.add(&grad2.mul_scalar(1.0 - b2)?)?,
                        None => grad2.mul_scalar(1.0 - b2)?,
                    };

                    let m_hat = m_new.mul_scalar(1.0 / (1.0 - b1t))?;

                    if rho_t > 5.0 {
                        // Variance is tractable: use Adam-like update
                        let v_hat = v_new.mul_scalar(1.0 / (1.0 - b2t))?;
                        let rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                                    ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)).sqrt();
                        let update = m_hat.div(&v_hat.sqrt()?.add_scalar(self.eps)?)?.mul_scalar(self.lr * rect)?;
                        data.sub_(&update)?;
                    } else {
                        // Variance is intractable: use SGD-like update
                        let update = m_hat.mul_scalar(self.lr)?;
                        data.sub_(&update)?;
                    }
                    self.m[i] = Some(m_new);
                    self.v[i] = Some(v_new);
                }
            }
            Ok(())
        })
    }

    fn zero_grad(&self) {
        for p in &self.params { p.zero_grad_set_to_none(); }
    }

    fn set_lr(&mut self, lr: f64) { self.lr = lr; }
}

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
    step_count: u64,
}

impl NAdam {
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        NAdam {
            params: params.iter().map(|p| p.variable.clone()).collect(),
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0,
            m: vec![None; n], v: vec![None; n], step_count: 0,
        }
    }

    pub fn lr(&self) -> f64 { self.lr }
}

impl Optimizer for NAdam {
    fn step(&mut self) -> Result<()> {
        self.step_count += 1;
        let t = self.step_count as f64;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let b1t = b1.powf(t);
        let b2t = b2.powf(t);
        let b1t1 = b1.powf(t + 1.0);

        no_grad(|| {
            for (i, param) in self.params.iter().enumerate() {
                if let Some(mut grad) = param.grad() {
                    let data = param.data().detach()?;
                    if self.weight_decay > 0.0 {
                        grad = grad.add(&data.mul_scalar(self.weight_decay)?)?;
                    }

                    let m_new = match self.m[i].take() {
                        Some(m) => m.mul_scalar(b1)?.add(&grad.mul_scalar(1.0 - b1)?)?,
                        None => grad.mul_scalar(1.0 - b1)?,
                    };
                    let grad2 = grad.mul(&grad)?;
                    let v_new = match self.v[i].take() {
                        Some(v) => v.mul_scalar(b2)?.add(&grad2.mul_scalar(1.0 - b2)?)?,
                        None => grad2.mul_scalar(1.0 - b2)?,
                    };

                    // Nesterov-corrected first moment
                    let m_hat = m_new.mul_scalar(b1 / (1.0 - b1t1))?
                        .add(&grad.mul_scalar((1.0 - b1) / (1.0 - b1t))?)?;
                    let v_hat = v_new.mul_scalar(1.0 / (1.0 - b2t))?;

                    let update = m_hat.div(&v_hat.sqrt()?.add_scalar(self.eps)?)?.mul_scalar(self.lr)?;
                    data.sub_(&update)?;

                    self.m[i] = Some(m_new);
                    self.v[i] = Some(v_new);
                }
            }
            Ok(())
        })
    }

    fn zero_grad(&self) {
        for p in &self.params { p.zero_grad_set_to_none(); }
    }

    fn set_lr(&mut self, lr: f64) { self.lr = lr; }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, TensorOptions};

    fn make_param(name: &str, shape: &[i64]) -> Parameter {
        let t = Tensor::randn(shape, TensorOptions {
            dtype: crate::tensor::DType::Float32,
            device: crate::tensor::test_device(),
        }).unwrap();
        Parameter::new(t, name)
    }

    #[test]
    fn test_adam_backward_compat() {
        // Adam::new still works with a single LR
        let p = make_param("w", &[3, 2]);
        let mut opt = Adam::new(std::slice::from_ref(&p), 0.01);

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.matmul(&p.variable).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let before = p.variable.data().to_f32_vec().unwrap();
        opt.step().unwrap();
        let after = p.variable.data().to_f32_vec().unwrap();
        assert_ne!(before, after, "params should change after step");
    }

    #[test]
    fn test_adam_two_groups_different_lr() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);

        // Group 0: high LR, Group 1: very low LR
        let mut opt = Adam::with_groups()
            .group(std::slice::from_ref(&p1), 0.1)
            .group(std::slice::from_ref(&p2), 1e-10)
            .build();

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        // Both params participate
        let y1 = x.matmul(&p1.variable).unwrap();
        let y2 = x.matmul(&p2.variable).unwrap();
        let loss = y1.add(&y2).unwrap().sum().unwrap();
        loss.backward().unwrap();

        let p1_before = p1.variable.data().to_f32_vec().unwrap();
        let p2_before = p2.variable.data().to_f32_vec().unwrap();
        opt.step().unwrap();
        let p1_after = p1.variable.data().to_f32_vec().unwrap();
        let p2_after = p2.variable.data().to_f32_vec().unwrap();

        // p1 should change substantially (high LR), p2 barely moves (tiny LR)
        let p1_delta: f64 = p1_before.iter().zip(&p1_after)
            .map(|(a, b)| (a - b).abs() as f64).sum();
        let p2_delta: f64 = p2_before.iter().zip(&p2_after)
            .map(|(a, b)| (a - b).abs() as f64).sum();

        assert!(p1_delta > p2_delta * 1e6,
            "high-LR group should move much more: p1_delta={}, p2_delta={}", p1_delta, p2_delta);
    }

    #[test]
    fn test_set_group_lr_changes_one_group() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);

        let mut opt = Adam::with_groups()
            .group(std::slice::from_ref(&p1), 0.01)
            .group(std::slice::from_ref(&p2), 0.01)
            .build();

        opt.set_group_lr(1, 0.99);
        // Group 0 unchanged, group 1 updated
        assert!((opt.groups[0].lr - 0.01).abs() < 1e-12);
        assert!((opt.groups[1].lr - 0.99).abs() < 1e-12);
    }

    #[test]
    fn test_set_lr_changes_all_groups() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);

        let mut opt = Adam::with_groups()
            .group(std::slice::from_ref(&p1), 0.01)
            .group(std::slice::from_ref(&p2), 0.05)
            .build();

        opt.set_lr(0.42);
        assert!((opt.lr - 0.42).abs() < 1e-12);
        assert!((opt.groups[0].lr - 0.42).abs() < 1e-12);
        assert!((opt.groups[1].lr - 0.42).abs() < 1e-12);
    }

    #[test]
    fn test_frozen_params_in_group_no_crash() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);
        p1.freeze().unwrap();

        let mut opt = Adam::with_groups()
            .group(&[p1, p2.clone()], 0.01)
            .build();

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.matmul(&p2.variable).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        // Should not crash even though p1 is frozen (no grad)
        opt.step().unwrap();
        opt.zero_grad();
    }

    #[test]
    fn test_adam_save_load_with_groups() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);

        let mut opt = Adam::with_groups()
            .group(std::slice::from_ref(&p1), 0.01)
            .group(std::slice::from_ref(&p2), 0.05)
            .build();

        // Do a step to populate moment buffers
        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y1 = x.matmul(&p1.variable).unwrap();
        let y2 = x.matmul(&p2.variable).unwrap();
        let loss = y1.add(&y2).unwrap().sum().unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();

        // Save
        let mut buf = Vec::new();
        opt.save_state(&mut buf).unwrap();

        // Load into fresh optimizer with same structure
        let mut opt2 = Adam::with_groups()
            .group(std::slice::from_ref(&p1), 0.99)
            .group(std::slice::from_ref(&p2), 0.99)
            .build();

        let mut cursor = std::io::Cursor::new(&buf);
        opt2.load_state(&mut cursor).unwrap();

        assert_eq!(opt2.t, opt.t);
        assert!((opt2.groups[0].lr - 0.01).abs() < 1e-12);
        assert!((opt2.groups[1].lr - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_fused_adam_numerical_correctness() {
        // Known param/grad/m/v, verify against hand-computed expected values
        let param = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], crate::tensor::test_device()).unwrap();
        let grad = Tensor::from_f32(&[0.1, 0.2, 0.3, 0.4], &[4], crate::tensor::test_device()).unwrap();
        let m = Tensor::zeros(&[4], crate::tensor::test_opts()).unwrap();
        let v = Tensor::zeros(&[4], crate::tensor::test_opts()).unwrap();

        let lr = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let step: i64 = 1;

        param.adam_step(&grad, &m, &v, lr, beta1, beta2, eps, 0.0, step).unwrap();

        // After step 1 with zero initial moments:
        // m = 0.1 * grad, v = 0.001 * grad^2
        // bc1 = 0.1, bc2 = 0.001
        // step_size = lr / bc1 = 0.01
        // denom = sqrt(v / bc2) + eps = |grad| + eps
        // update = step_size * m / denom ≈ step_size * 0.1*grad / |grad| ≈ 0.001 * sign(grad)
        // With positive grad: param -= 0.001

        let p_data = param.to_f32_vec().unwrap();
        let m_data = m.to_f32_vec().unwrap();
        let v_data = v.to_f32_vec().unwrap();

        // m = (1-beta1)*grad = 0.1 * [0.1, 0.2, 0.3, 0.4]
        for (i, &g) in [0.1f32, 0.2, 0.3, 0.4].iter().enumerate() {
            assert!((m_data[i] - 0.1 * g).abs() < 1e-6,
                "m[{}]: got {}, expected {}", i, m_data[i], 0.1 * g);
        }

        // v = (1-beta2)*grad^2 = 0.001 * [0.01, 0.04, 0.09, 0.16]
        for (i, &g) in [0.1f32, 0.2, 0.3, 0.4].iter().enumerate() {
            assert!((v_data[i] - 0.001 * g * g).abs() < 1e-9,
                "v[{}]: got {}, expected {}", i, v_data[i], 0.001 * g * g);
        }

        // Each param element should have decreased by approximately lr
        let orig = [1.0f32, 2.0, 3.0, 4.0];
        for (i, &o) in orig.iter().enumerate() {
            assert!((p_data[i] - (o - lr as f32)).abs() < 1e-5,
                "p[{}]: got {}, expected ~{}", i, p_data[i], o - lr as f32);
        }
    }

    #[test]
    fn test_fused_adamw_weight_decay() {
        let param = Tensor::from_f32(&[1.0, 2.0], &[2], crate::tensor::test_device()).unwrap();
        let grad = Tensor::from_f32(&[0.1, 0.1], &[2], crate::tensor::test_device()).unwrap();
        let m = Tensor::zeros(&[2], crate::tensor::test_opts()).unwrap();
        let v = Tensor::zeros(&[2], crate::tensor::test_opts()).unwrap();

        let lr = 0.001;
        let wd = 0.01;

        param.adam_step(&grad, &m, &v, lr, 0.9, 0.999, 1e-8, wd, 1).unwrap();

        let p_data = param.to_f32_vec().unwrap();
        // Weight decay: p *= (1 - lr * wd) = (1 - 0.00001)
        // Then Adam update subtracts ~lr from each element
        // param[0] should be slightly less than 1.0 - 0.001
        // param[1] should be slightly less than 2.0 - 0.001, but also
        // decayed more because 2.0 * lr * wd > 1.0 * lr * wd
        assert!(p_data[0] < 1.0, "p[0] should decrease: got {}", p_data[0]);
        assert!(p_data[1] < 2.0, "p[1] should decrease: got {}", p_data[1]);
        // Weight decay asymmetry: param[1] decays more (larger value)
        let decay_0 = 1.0 - p_data[0] as f64;
        let decay_1 = 2.0 - p_data[1] as f64;
        assert!(decay_1 > decay_0, "larger param should decay more: d0={}, d1={}", decay_0, decay_1);
    }

    #[test]
    fn test_fused_adam_multi_step_convergence() {
        // Run multiple steps, verify m/v accumulate correctly
        let param = Tensor::from_f32(&[5.0], &[1], crate::tensor::test_device()).unwrap();
        let grad = Tensor::from_f32(&[1.0], &[1], crate::tensor::test_device()).unwrap();
        let m = Tensor::zeros(&[1], crate::tensor::test_opts()).unwrap();
        let v = Tensor::zeros(&[1], crate::tensor::test_opts()).unwrap();

        for step in 1..=10 {
            param.adam_step(&grad, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, step).unwrap();
        }

        // After 10 steps with constant gradient=1:
        // m should converge toward 1.0, v should converge toward 1.0
        let m_data = m.to_f32_vec().unwrap();
        let p_data = param.to_f32_vec().unwrap();

        // m = 1 - 0.9^10 ≈ 0.6513
        assert!((m_data[0] - 0.6513).abs() < 0.01,
            "m after 10 steps: got {}", m_data[0]);
        // v should be non-zero (accumulating)
        assert!(v.to_f32_vec().unwrap()[0] > 0.0, "v should accumulate");
        // param should have decreased
        assert!(p_data[0] < 5.0, "param should decrease: got {}", p_data[0]);
    }

    #[test]
    fn test_rmsprop_basic() {
        let p = make_param("w", &[3, 2]);
        let mut opt = RMSprop::new(std::slice::from_ref(&p), 0.01);

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.matmul(&p.variable).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let before = p.variable.data().to_f32_vec().unwrap();
        opt.step().unwrap();
        let after = p.variable.data().to_f32_vec().unwrap();
        assert_ne!(before, after, "params should change after step");
    }

    #[test]
    fn test_rmsprop_with_momentum() {
        let p = make_param("w", &[3, 2]);
        let mut opt = RMSprop::builder(std::slice::from_ref(&p), 0.01)
            .momentum(0.9)
            .build();

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );

        // Two steps to exercise momentum buffer
        for _ in 0..2 {
            opt.zero_grad();
            let y = x.matmul(&p.variable).unwrap();
            let loss = y.sum().unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
        }

        // Should not crash and param should have changed
        let data = p.variable.data().to_f32_vec().unwrap();
        assert!(data.iter().any(|&v| v.abs() > 0.0), "params should be non-zero");
    }

    #[test]
    fn test_rmsprop_with_weight_decay() {
        // Create two params with identical initial values via from_f32
        let init = [0.5f32, -0.3, 0.1, 0.8, -0.2, 0.4];
        let dev = crate::tensor::test_device();

        let p1 = Parameter::new(
            Tensor::from_f32(&init, &[3, 2], dev).unwrap(), "w1");
        let p2 = Parameter::new(
            Tensor::from_f32(&init, &[3, 2], dev).unwrap(), "w2");

        let mut opt_wd = RMSprop::builder(std::slice::from_ref(&p1), 0.01)
            .weight_decay(0.1)
            .build();
        let mut opt_plain = RMSprop::new(std::slice::from_ref(&p2), 0.01);

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], dev).unwrap(),
            false,
        );

        // Run multiple steps so accumulated v differs enough to produce
        // visibly different updates (single step RMSprop is dominated by sign)
        for _ in 0..10 {
            opt_wd.zero_grad();
            let y1 = x.matmul(&p1.variable).unwrap();
            y1.sum().unwrap().backward().unwrap();
            opt_wd.step().unwrap();

            opt_plain.zero_grad();
            let y2 = x.matmul(&p2.variable).unwrap();
            y2.sum().unwrap().backward().unwrap();
            opt_plain.step().unwrap();
        }

        let d1 = p1.variable.data().to_f32_vec().unwrap();
        let d2 = p2.variable.data().to_f32_vec().unwrap();
        // Weight decay should cause different parameter trajectories
        assert_ne!(d1, d2, "weight decay should produce different results after 10 steps");
    }

    #[test]
    fn test_rmsprop_convergence() {
        // Verify step works and changes param
        let p = Parameter::new(
            Tensor::from_f32(&[5.0], &[1], crate::tensor::test_device()).unwrap(),
            "x",
        );
        let mut opt = RMSprop::new(std::slice::from_ref(&p), 0.1);

        let x = Variable::new(
            Tensor::from_f32(&[1.0], &[1], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.mul(&p.variable).unwrap();
        let loss = y.mul(&y).unwrap().sum().unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();

        let val = p.variable.data().to_f32_vec().unwrap()[0];
        assert!(val < 5.0, "param should decrease from 5.0, got {}", val);
    }

    #[test]
    fn test_rmsprop_save_load() {
        let p = make_param("w", &[3, 2]);
        let mut opt = RMSprop::builder(std::slice::from_ref(&p), 0.01)
            .momentum(0.9)
            .alpha(0.95)
            .weight_decay(0.01)
            .build();

        // Do a step to populate buffers
        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.matmul(&p.variable).unwrap();
        y.sum().unwrap().backward().unwrap();
        opt.step().unwrap();

        // Save
        let mut buf = Vec::new();
        opt.save_state(&mut buf).unwrap();

        // Load into fresh optimizer
        let mut opt2 = RMSprop::builder(std::slice::from_ref(&p), 0.99)
            .build();
        let mut cursor = std::io::Cursor::new(&buf);
        opt2.load_state(&mut cursor).unwrap();

        assert!((opt2.lr - 0.01).abs() < 1e-12);
        assert!((opt2.alpha - 0.95).abs() < 1e-12);
        assert!((opt2.momentum - 0.9).abs() < 1e-12);
        assert!((opt2.weight_decay - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_rmsprop_builder_defaults() {
        let p = make_param("w", &[2]);
        let opt = RMSprop::new(std::slice::from_ref(&p), 0.01);
        assert!((opt.alpha - 0.99).abs() < 1e-12);
        assert!((opt.eps - 1e-8).abs() < 1e-15);
        assert!((opt.weight_decay).abs() < 1e-12);
        assert!((opt.momentum).abs() < 1e-12);
    }

    #[test]
    fn test_rmsprop_frozen_params() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);
        p1.freeze().unwrap();

        let mut opt = RMSprop::new(&[p1, p2.clone()], 0.01);

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.matmul(&p2.variable).unwrap();
        y.sum().unwrap().backward().unwrap();

        // Should not crash with frozen param
        opt.step().unwrap();
        opt.zero_grad();
    }

    #[test]
    fn test_adagrad_steps() {
        let p = make_param("w", &[1]);
        let before = p.variable.data().item().unwrap();
        let mut opt = Adagrad::new(std::slice::from_ref(&p), 0.5);
        let x = Variable::new(
            Tensor::from_f32(&[2.0], &[1], crate::tensor::test_device()).unwrap(), false,
        );
        let loss = x.mul(&p.variable).unwrap().sum().unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();
        let after = p.variable.data().item().unwrap();
        // Parameter should change
        assert!((after - before).abs() > 1e-6, "Adagrad step should change parameter");
    }

    #[test]
    fn test_radam_steps() {
        let p = make_param("w", &[1]);
        let before = p.variable.data().item().unwrap();
        let mut opt = RAdam::new(std::slice::from_ref(&p), 0.01);
        let x = Variable::new(
            Tensor::from_f32(&[2.0], &[1], crate::tensor::test_device()).unwrap(), false,
        );
        let loss = x.mul(&p.variable).unwrap().sum().unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();
        let after = p.variable.data().item().unwrap();
        assert!((after - before).abs() > 1e-6, "RAdam step should change parameter");
    }

    #[test]
    fn test_nadam_steps() {
        let p = make_param("w", &[1]);
        let before = p.variable.data().item().unwrap();
        let mut opt = NAdam::new(std::slice::from_ref(&p), 0.01);
        let x = Variable::new(
            Tensor::from_f32(&[2.0], &[1], crate::tensor::test_device()).unwrap(), false,
        );
        let loss = x.mul(&p.variable).unwrap().sum().unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();
        let after = p.variable.data().item().unwrap();
        assert!((after - before).abs() > 1e-6, "NAdam step should change parameter");
    }
}
