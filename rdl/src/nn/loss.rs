use crate::autograd::Variable;
use crate::tensor::{Result, Tensor, TensorError};

/// Mean Squared Error loss: mean((pred - target)²)
pub fn mse_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let diff = pred.sub(target)?;
    let sq = diff.mul(&diff)?;
    let total = sq.sum()?;
    let n = pred.numel() as f64;
    total.mul_scalar(1.0 / n)
}

/// Cross-entropy loss with numerically stable log-softmax.
///
/// `pred` shape: `[batch, classes]` (raw logits).
/// `target` shape: `[batch, classes]` (one-hot or soft labels).
///
/// Computes: `-mean(sum(target * log_softmax(pred), dim=1))`
pub fn cross_entropy_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let shape = pred.shape();
    if shape.len() != 2 {
        return Err(TensorError::new("cross_entropy_loss: pred must be 2D [batch, classes]"));
    }
    let batch = shape[0];

    // Native log-softmax: numerically stable (handles max subtraction internally)
    let log_softmax = pred.log_softmax(1)?;

    // -mean(sum(target * log_softmax, dim=1))
    let weighted = target.mul(&log_softmax)?;
    let per_sample = weighted.sum_dim(1, false)?; // [batch]
    let total = per_sample.sum()?;
    total.mul_scalar(-1.0 / batch as f64)
}

/// Binary cross-entropy loss from raw logits (numerically stable).
///
/// `pred`: raw logits (any shape).
/// `target`: binary labels (same shape, values 0 or 1).
///
/// Computes: `mean(relu(x) - x*t + log(1 + exp(-|x|)))`
pub fn bce_with_logits_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let n = pred.numel() as f64;

    // relu(x) - x*t + log(1 + exp(-|x|))
    let relu_x = pred.relu()?;
    let x_times_t = pred.mul(target)?;
    let abs_x = pred.abs()?;
    let neg_abs_x = abs_x.neg()?;
    let exp_neg_abs = neg_abs_x.exp()?;
    let one_plus_exp = exp_neg_abs.add_scalar(1.0)?;
    let log_term = one_plus_exp.log()?;

    let loss = relu_x.sub(&x_times_t)?.add(&log_term)?;
    let total = loss.sum()?;
    total.mul_scalar(1.0 / n)
}

/// L1 (Mean Absolute Error) loss: mean(|pred - target|)
pub fn l1_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let diff = pred.sub(target)?;
    let abs_diff = diff.abs()?;
    let total = abs_diff.sum()?;
    let n = pred.numel() as f64;
    total.mul_scalar(1.0 / n)
}

/// Smooth L1 (Huber) loss with configurable beta.
///
/// For |x| < beta: 0.5 * x² / beta
/// For |x| >= beta: |x| - 0.5 * beta
///
/// Equivalent to Huber loss with delta = beta.
pub fn smooth_l1_loss(pred: &Variable, target: &Variable, beta: f64) -> Result<Variable> {
    if beta <= 0.0 {
        return Err(TensorError::new("smooth_l1_loss: beta must be positive"));
    }
    let n = pred.numel() as f64;
    let diff = pred.sub(target)?;
    let abs_diff = diff.abs()?;

    // Use the identity: smooth_l1(x) = |x| - 0.5*beta + 0.5*min(|x|, beta)²/beta
    // But that requires min. Instead, build from the quadratic and linear parts:
    //
    // smooth_l1(x) = where(|x| < beta, 0.5*x²/beta, |x| - 0.5*beta)
    //
    // We can compute both branches and blend with a differentiable approximation,
    // but since this is a loss function (not an intermediate op), we'll compute
    // both and select at the tensor level using gt_scalar as a mask.
    let quadratic = diff.mul(&diff)?.mul_scalar(0.5 / beta)?;
    let linear = abs_diff.add_scalar(-0.5 * beta)?;

    // mask: 1 where |diff| >= beta, 0 where < beta
    let mask_tensor = abs_diff.data().gt_scalar(beta)?;
    let mask = Variable::new(mask_tensor, false);

    // loss = mask * linear + (1 - mask) * quadratic
    let one = Variable::new(Tensor::ones_like(&mask.data())?, false);
    let inv_mask = one.sub(&mask)?;
    let loss = mask.mul(&linear)?.add(&inv_mask.mul(&quadratic)?)?;

    let total = loss.sum()?;
    total.mul_scalar(1.0 / n)
}

/// KL Divergence loss (batchmean reduction).
///
/// `input`: log-probabilities (output of log_softmax).
/// `target`: probabilities (true distribution).
///
/// Computes: `sum(target * (log(target) - input)) / batch`
///
/// Follows PyTorch convention: target * (log(target) - input).
pub fn kl_div_loss(input: &Variable, target: &Variable) -> Result<Variable> {
    let shape = input.shape();
    if shape.is_empty() {
        return Err(TensorError::new("kl_div_loss: input must be at least 1D"));
    }
    let batch = shape[0] as f64;

    // target * (log(target) - input)
    // For numerical stability, skip terms where target ≈ 0
    // We use target * log(target + eps) to avoid log(0)
    let target_log = target.add_scalar(1e-12)?.log()?;
    let diff = target_log.sub(input)?;
    let kl = target.mul(&diff)?;
    let total = kl.sum()?;
    total.mul_scalar(1.0 / batch)
}
