use crate::tensor::{Result, Tensor};

use super::Parameter;

/// Clip gradients by total L2 norm. Scales all gradients so that the
/// total norm does not exceed `max_norm`. Returns the original total norm.
pub fn clip_grad_norm(params: &[Parameter], max_norm: f64) -> Result<f64> {
    // Compute total L2 norm across all parameter gradients
    let mut total_norm_sq = 0.0f64;
    for p in params {
        if let Some(grad) = p.variable.grad() {
            let grad_data = grad.to_f32_vec()?;
            for &v in &grad_data {
                total_norm_sq += (v as f64) * (v as f64);
            }
        }
    }
    let total_norm = total_norm_sq.sqrt();

    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for p in params {
            if let Some(grad) = p.variable.grad() {
                let scaled = grad.mul_scalar(scale)?;
                p.variable.set_grad(scaled);
            }
        }
    }

    Ok(total_norm)
}

/// Clip gradients element-wise to `[-max_val, max_val]`.
/// Returns the maximum absolute gradient value before clipping.
pub fn clip_grad_value(params: &[Parameter], max_val: f64) -> Result<f64> {
    let mut max_abs = 0.0f64;

    for p in params {
        if let Some(grad) = p.variable.grad() {
            let grad_data = grad.to_f32_vec()?;
            let local_max = grad_data.iter().map(|v| v.abs() as f64).fold(0.0f64, f64::max);
            if local_max > max_abs {
                max_abs = local_max;
            }

            if local_max > max_val {
                // Clamp each element
                let clamped: Vec<f32> = grad_data
                    .iter()
                    .map(|&v| v.max(-max_val as f32).min(max_val as f32))
                    .collect();
                let shape = grad.shape();
                let new_grad = Tensor::from_f32(&clamped, &shape, grad.device())?;
                p.variable.set_grad(new_grad);
            }
        }
    }

    Ok(max_abs)
}
