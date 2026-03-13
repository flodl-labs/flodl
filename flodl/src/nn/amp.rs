use std::io::{Read, Write};

use crate::autograd::Variable;
use crate::tensor::{DType, Result, Tensor};

use super::checkpoint::{write_f64_le, read_f64_le, write_i64_le, read_i64_le};
use super::optim::Stateful;
use super::parameter::Parameter;

/// Cast all parameters to a different dtype.
///
/// No-op for parameters already at the target dtype.
pub fn cast_parameters(params: &[Parameter], dtype: DType) {
    for p in params {
        if p.variable.data().dtype() != dtype
            && let Ok(t) = p.variable.data().to_dtype(dtype)
        {
            p.variable.set_data(t);
        }
    }
}

/// GradScaler for mixed precision training.
///
/// Scales loss before backward to prevent gradient underflow in float16,
/// then unscales gradients before optimizer step. Dynamically adjusts
/// scale factor based on whether inf/nan gradients are detected.
///
/// ```ignore
/// let mut scaler = GradScaler::new();
/// let scaled_loss = scaler.scale(&loss)?;
/// scaled_loss.backward()?;
/// scaler.step(&params, &mut || optim.step())?;
/// scaler.update();
/// ```
pub struct GradScaler {
    scale: f64,
    growth: f64,
    backoff: f64,
    interval: i64,
    steps_since_growth: i64,
    found_inf: bool,
}

impl Default for GradScaler {
    fn default() -> Self {
        GradScaler {
            scale: 65536.0,
            growth: 2.0,
            backoff: 0.5,
            interval: 2000,
            steps_since_growth: 0,
            found_inf: false,
        }
    }
}

impl GradScaler {
    /// Create a new GradScaler with default settings.
    ///
    /// Initial scale: 2^16 = 65536, growth: 2.0, backoff: 0.5, interval: 2000.
    pub fn new() -> Self {
        Self::default()
    }

    /// Scale the loss before backward. Returns loss * scale.
    pub fn scale(&self, loss: &Variable) -> Result<Variable> {
        loss.mul_scalar(self.scale)
    }

    /// Current scale factor.
    pub fn scale_factor(&self) -> f64 {
        self.scale
    }

    /// Unscale gradients, check for inf/nan, and step the optimizer.
    ///
    /// Returns true if the step was taken (all gradients finite).
    /// Returns false if inf/nan detected (optimizer step skipped).
    pub fn step(&mut self, params: &[Parameter], step_fn: &mut dyn FnMut() -> Result<()>) -> Result<bool> {
        let inv_scale = 1.0 / self.scale;

        // Unscale and check all gradients
        let mut unscaled_grads: Vec<Option<Tensor>> = Vec::with_capacity(params.len());
        for p in params {
            if let Some(grad) = p.variable.grad() {
                let unscaled = grad.mul_scalar(inv_scale)?;
                if !unscaled.all_finite()? {
                    self.found_inf = true;
                    return Ok(false);
                }
                unscaled_grads.push(Some(unscaled));
            } else {
                unscaled_grads.push(None);
            }
        }

        // Replace gradients with unscaled versions
        for (p, ug) in params.iter().zip(unscaled_grads) {
            if let Some(g) = ug {
                p.variable.set_grad(g);
            }
        }

        // Step the optimizer
        step_fn()?;
        Ok(true)
    }

    /// Update the scale factor after each step.
    ///
    /// Call this after every `step()` call, regardless of whether it succeeded.
    pub fn update(&mut self) {
        if self.found_inf {
            self.scale *= self.backoff;
            self.steps_since_growth = 0;
        } else {
            self.steps_since_growth += 1;
            if self.steps_since_growth >= self.interval {
                self.scale *= self.growth;
                self.steps_since_growth = 0;
            }
        }
        self.found_inf = false;
    }
}

impl Stateful for GradScaler {
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_f64_le(w, self.scale)?;
        write_i64_le(w, self.steps_since_growth)?;
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        self.scale = read_f64_le(r)?;
        self.steps_since_growth = read_i64_le(r)?;
        Ok(())
    }
}
