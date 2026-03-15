use crate::tensor::{Result, Tensor};

use super::Parameter;

/// Clip gradients by total L2 norm. Scales all gradients so that the
/// total norm does not exceed `max_norm`. Returns the original total norm.
///
/// Uses a single fused C++ call — computes global norm across all params
/// and scales in-place. No per-param FFI roundtrips.
pub fn clip_grad_norm(params: &[Parameter], max_norm: f64) -> Result<f64> {
    let handles: Vec<_> = params.iter().map(|p| p.variable.data()).collect();
    Tensor::clip_grad_norm_fused(&handles, max_norm)
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
