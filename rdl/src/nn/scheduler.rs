use super::optim::Optimizer;

/// Learning rate scheduler interface.
pub trait Scheduler {
    /// Advance one step and update the optimizer's learning rate.
    fn step(&mut self);
    /// Current learning rate.
    fn lr(&self) -> f64;
}

/// Staircase decay: multiply lr by gamma every `step_size` steps.
pub struct StepDecay<'a> {
    optimizer: &'a mut dyn Optimizer,
    step_size: usize,
    gamma: f64,
    current_step: usize,
    current_lr: f64,
}

impl<'a> StepDecay<'a> {
    pub fn new(optimizer: &'a mut dyn Optimizer, base_lr: f64, step_size: usize, gamma: f64) -> Self {
        StepDecay {
            optimizer,
            step_size,
            gamma,
            current_step: 0,
            current_lr: base_lr,
        }
    }
}

impl Scheduler for StepDecay<'_> {
    fn step(&mut self) {
        self.current_step += 1;
        if self.current_step % self.step_size == 0 {
            self.current_lr *= self.gamma;
            self.optimizer.set_lr(self.current_lr);
        }
    }

    fn lr(&self) -> f64 {
        self.current_lr
    }
}

/// Cosine annealing from base_lr to min_lr over total_steps.
pub struct CosineScheduler<'a> {
    optimizer: &'a mut dyn Optimizer,
    base_lr: f64,
    min_lr: f64,
    total_steps: usize,
    current_step: usize,
}

impl<'a> CosineScheduler<'a> {
    pub fn new(
        optimizer: &'a mut dyn Optimizer,
        base_lr: f64,
        min_lr: f64,
        total_steps: usize,
    ) -> Self {
        CosineScheduler {
            optimizer,
            base_lr,
            min_lr,
            total_steps,
            current_step: 0,
        }
    }
}

impl Scheduler for CosineScheduler<'_> {
    fn step(&mut self) {
        self.current_step += 1;
        let t = (self.current_step.min(self.total_steps) as f64) / (self.total_steps as f64);
        let lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (t * std::f64::consts::PI).cos());
        self.optimizer.set_lr(lr);
    }

    fn lr(&self) -> f64 {
        let t = (self.current_step.min(self.total_steps) as f64) / (self.total_steps as f64);
        self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (t * std::f64::consts::PI).cos())
    }
}

/// Linear warmup followed by another scheduler.
/// Ramps lr from 0 to target_lr over warmup_steps, then delegates.
pub struct WarmupScheduler<'a, S: Scheduler> {
    optimizer: &'a mut dyn Optimizer,
    inner: S,
    target_lr: f64,
    warmup_steps: usize,
    current_step: usize,
}

impl<'a, S: Scheduler> WarmupScheduler<'a, S> {
    pub fn new(
        optimizer: &'a mut dyn Optimizer,
        inner: S,
        target_lr: f64,
        warmup_steps: usize,
    ) -> Self {
        WarmupScheduler {
            optimizer,
            inner,
            target_lr,
            warmup_steps,
            current_step: 0,
        }
    }
}

impl<S: Scheduler> Scheduler for WarmupScheduler<'_, S> {
    fn step(&mut self) {
        self.current_step += 1;
        if self.current_step <= self.warmup_steps {
            let lr = self.target_lr * (self.current_step as f64) / (self.warmup_steps as f64);
            self.optimizer.set_lr(lr);
        } else {
            self.inner.step();
        }
    }

    fn lr(&self) -> f64 {
        if self.current_step <= self.warmup_steps {
            self.target_lr * (self.current_step as f64) / (self.warmup_steps as f64)
        } else {
            self.inner.lr()
        }
    }
}

/// Reduce learning rate when a metric plateaus.
/// Call `observe(metric)` each epoch (lower is better).
pub struct PlateauScheduler<'a> {
    optimizer: &'a mut dyn Optimizer,
    patience: usize,
    factor: f64,
    min_lr: f64,
    current_lr: f64,
    best: f64,
    wait: usize,
}

impl<'a> PlateauScheduler<'a> {
    pub fn new(
        optimizer: &'a mut dyn Optimizer,
        base_lr: f64,
        patience: usize,
        factor: f64,
        min_lr: f64,
    ) -> Self {
        PlateauScheduler {
            optimizer,
            patience,
            factor,
            min_lr,
            current_lr: base_lr,
            best: f64::INFINITY,
            wait: 0,
        }
    }

    /// Feed an observed metric (lower is better). Reduces lr after
    /// `patience` epochs without improvement.
    pub fn observe(&mut self, metric: f64) {
        if metric < self.best {
            self.best = metric;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                self.optimizer.set_lr(self.current_lr);
                self.wait = 0;
            }
        }
    }

    pub fn lr(&self) -> f64 {
        self.current_lr
    }
}
