use crate::autograd::Variable;
use crate::tensor::{Result, TensorError};

/// Mean Squared Error loss: mean((pred - target)²)
///
/// Uses a single fused libtorch kernel (1 autograd node).
pub fn mse_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let result = pred.data().mse_loss(&target.data(), 1)?; // 1 = Mean
    Ok(Variable::wrap(result))
}

/// Cross-entropy loss with numerically stable log-softmax.
///
/// `pred` shape: `[batch, classes]` (raw logits).
///
/// `target` accepts two formats (auto-detected by libtorch):
/// - **Class indices** `[batch]` (Int64) — like PyTorch's `F.cross_entropy`.
///   Each value is the correct class index (0..classes-1).
/// - **One-hot / soft labels** `[batch, classes]` (Float) — probability vectors.
///
/// Uses a single fused libtorch kernel (1 autograd node).
///
/// ```ignore
/// // With class indices (no one-hot allocation needed):
/// let labels = Variable::new(Tensor::from_i64(&[0, 2, 1], &[3], Device::CPU)?, false);
/// let loss = cross_entropy_loss(&logits, &labels)?;
///
/// // With one-hot targets (same as before):
/// let onehot = Variable::new(Tensor::from_f32(&[1.,0.,0., 0.,0.,1., 0.,1.,0.], &[3, 3], Device::CPU)?, false);
/// let loss = cross_entropy_loss(&logits, &onehot)?;
/// ```
pub fn cross_entropy_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let pred_shape = pred.shape();
    if pred_shape.len() != 2 {
        return Err(TensorError::new("cross_entropy_loss: pred must be 2D [batch, classes]"));
    }
    let result = pred.data().cross_entropy_loss(
        &target.data(), 1, -100, 0.0, // Mean, ignore_index=-100, no smoothing
    )?;
    Ok(Variable::wrap(result))
}

/// Binary cross-entropy loss from probabilities (NOT logits).
///
/// `pred`: probabilities in \[0, 1\] (e.g. after sigmoid). Any shape.
/// `target`: binary labels (same shape, values 0 or 1).
///
/// For raw logits, prefer `bce_with_logits_loss` which is numerically stable.
///
/// Uses a single fused libtorch kernel (1 autograd node).
pub fn bce_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let result = pred.data().bce_loss(&target.data(), 1)?; // 1 = Mean
    Ok(Variable::wrap(result))
}

/// Binary cross-entropy loss from raw logits (numerically stable).
///
/// `pred`: raw logits (any shape).
/// `target`: binary labels (same shape, values 0 or 1).
///
/// Uses a single fused libtorch kernel (1 autograd node).
pub fn bce_with_logits_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let result = pred.data().bce_with_logits_loss(&target.data(), 1)?; // 1 = Mean
    Ok(Variable::wrap(result))
}

/// L1 (Mean Absolute Error) loss: mean(|pred - target|)
///
/// Uses a single fused libtorch kernel (1 autograd node).
pub fn l1_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let result = pred.data().l1_loss(&target.data(), 1)?; // 1 = Mean
    Ok(Variable::wrap(result))
}

/// Smooth L1 (Huber) loss with configurable beta.
///
/// For |x| < beta: 0.5 * x² / beta
/// For |x| >= beta: |x| - 0.5 * beta
///
/// Uses a single fused libtorch kernel (1 autograd node).
pub fn smooth_l1_loss(pred: &Variable, target: &Variable, beta: f64) -> Result<Variable> {
    if beta <= 0.0 {
        return Err(TensorError::new("smooth_l1_loss: beta must be positive"));
    }
    let result = pred.data().smooth_l1_loss(&target.data(), 1, beta)?; // 1 = Mean
    Ok(Variable::wrap(result))
}

/// KL Divergence loss (batchmean reduction).
///
/// `input`: log-probabilities (output of log_softmax).
/// `target`: probabilities (true distribution).
///
/// Computes: `sum(target * (log(target) - input)) / batch`
///
/// Uses a single fused libtorch kernel (1 autograd node).
/// Follows PyTorch convention: target * (log(target) - input).
pub fn kl_div_loss(input: &Variable, target: &Variable) -> Result<Variable> {
    let shape = input.shape();
    if shape.is_empty() {
        return Err(TensorError::new("kl_div_loss: input must be at least 1D"));
    }
    let batch = shape[0] as f64;
    // Use sum reduction then divide by batch (batchmean).
    // PyTorch's "batchmean" is a Python wrapper, not in the C++ kernel.
    let sum = input.data().kl_div_loss(&target.data(), 2, false)?; // 2 = Sum
    let result = sum.mul_scalar(1.0 / batch)?;
    Ok(Variable::wrap(result))
}
