/// Learning rate scheduler interface.
/// Schedulers are pure strategy objects — they compute the LR for a given step
/// without holding a reference to the optimizer. The user bridges them:
///     `optimizer.set_lr(sched.lr(step))`
pub trait Scheduler {
    /// Learning rate at the given step (0-based).
    fn lr(&self, step: usize) -> f64;
}

/// Staircase decay: multiply lr by gamma every `step_size` steps.
pub struct StepDecay {
    base_lr: f64,
    step_size: usize,
    gamma: f64,
}

impl StepDecay {
    /// Create a step decay scheduler: lr *= gamma every `step_size` steps.
    pub fn new(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        StepDecay {
            base_lr,
            step_size,
            gamma,
        }
    }

    pub fn lr(&self, step: usize) -> f64 {
        let decays = step / self.step_size;
        self.base_lr * self.gamma.powi(decays as i32)
    }
}

impl Scheduler for StepDecay {
    fn lr(&self, step: usize) -> f64 {
        StepDecay::lr(self, step)
    }
}

/// Cosine annealing from base_lr to min_lr over total_steps.
pub struct CosineScheduler {
    base_lr: f64,
    min_lr: f64,
    total_steps: usize,
}

impl CosineScheduler {
    /// Create a cosine annealing scheduler from `base_lr` to `min_lr` over `total_steps`.
    pub fn new(base_lr: f64, min_lr: f64, total_steps: usize) -> Self {
        CosineScheduler {
            base_lr,
            min_lr,
            total_steps,
        }
    }

    pub fn lr(&self, step: usize) -> f64 {
        let t = (step.min(self.total_steps) as f64) / (self.total_steps as f64);
        self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (t * std::f64::consts::PI).cos())
    }
}

impl Scheduler for CosineScheduler {
    fn lr(&self, step: usize) -> f64 {
        CosineScheduler::lr(self, step)
    }
}

/// Linear warmup followed by another scheduler.
/// Ramps lr from 0 to target_lr over warmup_steps, then delegates.
pub struct WarmupScheduler<S: Scheduler> {
    inner: S,
    target_lr: f64,
    warmup_steps: usize,
}

impl<S: Scheduler> WarmupScheduler<S> {
    /// Create a warmup scheduler that ramps from 0 to `target_lr` then delegates to `inner`.
    pub fn new(inner: S, target_lr: f64, warmup_steps: usize) -> Self {
        WarmupScheduler {
            inner,
            target_lr,
            warmup_steps,
        }
    }

    pub fn lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            self.target_lr * (step as f64 + 1.0) / (self.warmup_steps as f64)
        } else {
            self.inner.lr(step - self.warmup_steps)
        }
    }
}

impl<S: Scheduler> Scheduler for WarmupScheduler<S> {
    fn lr(&self, step: usize) -> f64 {
        WarmupScheduler::lr(self, step)
    }
}

/// Reduce learning rate when a metric plateaus.
/// Call `observe(metric)` each epoch (lower is better).
///
/// Unlike step-based schedulers (`StepDecay`, `CosineScheduler`, etc.),
/// PlateauScheduler is reactive — it does not implement the [`Scheduler`]
/// trait because its LR depends on observed metrics, not step count.
/// This matches PyTorch's `ReduceLROnPlateau` which also has a different
/// interface from other schedulers.
///
/// ```ignore
/// let mut sched = PlateauScheduler::new(0.01, 5, 0.5, 1e-6);
/// for epoch in 0..100 {
///     let loss = train_epoch(&model, &data)?;
///     let lr = sched.observe(loss);
///     optimizer.set_lr(lr);
/// }
/// ```
pub struct PlateauScheduler {
    patience: usize,
    factor: f64,
    min_lr: f64,
    current_lr: f64,
    best: f64,
    wait: usize,
}

impl PlateauScheduler {
    /// Create a plateau scheduler that reduces lr by `factor` after `patience` epochs
    /// without improvement, down to `min_lr`.
    pub fn new(
        base_lr: f64,
        patience: usize,
        factor: f64,
        min_lr: f64,
    ) -> Self {
        PlateauScheduler {
            patience,
            factor,
            min_lr,
            current_lr: base_lr,
            best: f64::INFINITY,
            wait: 0,
        }
    }

    /// Feed an observed metric (lower is better). Reduces lr after
    /// `patience` epochs without improvement. Returns the current lr.
    pub fn observe(&mut self, metric: f64) -> f64 {
        if metric < self.best {
            self.best = metric;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                self.wait = 0;
            }
        }
        self.current_lr
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 {
        self.current_lr
    }
}
