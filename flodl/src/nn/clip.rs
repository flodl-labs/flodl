use crate::tensor::Result;

use super::Parameter;

/// Clip gradients by total L2 norm. Scales all gradients so that the
/// total norm does not exceed `max_norm`. Returns the original total norm.
///
/// All computation stays on-device — no GPU→CPU data copies except
/// one scalar at the end to compare against `max_norm`.
pub fn clip_grad_norm(params: &[Parameter], max_norm: f64) -> Result<f64> {
    // Accumulate sum of squared norms on-device (one scalar per param)
    let mut norm_sq_sum: Option<crate::tensor::Tensor> = None;
    for p in params {
        if let Some(grad) = p.variable.grad() {
            let n2 = grad.norm()?.pow_scalar(2.0)?;
            norm_sq_sum = Some(match norm_sq_sum {
                Some(acc) => acc.add(&n2)?,
                None => n2,
            });
        }
    }

    let total_norm = match norm_sq_sum {
        Some(s) => s.sqrt()?.item()?,  // one scalar read from device
        None => return Ok(0.0),
    };

    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for p in params {
            if let Some(grad) = p.variable.grad() {
                p.variable.set_grad(grad.mul_scalar(scale)?);
            }
        }
    }

    Ok(total_norm)
}

/// Clip gradients element-wise to `[-max_val, max_val]`.
/// Returns the maximum absolute gradient value before clipping.
///
/// Clamping stays on-device. Only one scalar per parameter is read
/// to track the global maximum.
pub fn clip_grad_value(params: &[Parameter], max_val: f64) -> Result<f64> {
    let mut global_max = 0.0f64;

    for p in params {
        if let Some(grad) = p.variable.grad() {
            // Read one scalar per param to track max (tiny transfer)
            let local_max = grad.abs()?.max()?.item()?;
            if local_max > global_max {
                global_max = local_max;
            }
            // Clamp on-device — no data copy
            p.variable.set_grad(grad.clamp(-max_val, max_val)?);
        }
    }

    Ok(global_max)
}
