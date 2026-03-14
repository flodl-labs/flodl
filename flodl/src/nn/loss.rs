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
///
/// `target` accepts two formats (auto-detected):
/// - **Class indices** `[batch]` (Int64) — like PyTorch's `F.cross_entropy`.
///   Each value is the correct class index (0..classes-1).
/// - **One-hot / soft labels** `[batch, classes]` (Float) — probability vectors.
///
/// ```ignore
/// // With class indices (no one-hot allocation needed):
/// let labels = Variable::new(Tensor::from_i64(&[0, 2, 1], &[3])?, false);
/// let loss = cross_entropy_loss(&logits, &labels)?;
///
/// // With one-hot targets (same as before):
/// let onehot = Variable::new(Tensor::from_f32(&[1.,0.,0., 0.,0.,1., 0.,1.,0.], &[3, 3])?, false);
/// let loss = cross_entropy_loss(&logits, &onehot)?;
/// ```
pub fn cross_entropy_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let pred_shape = pred.shape();
    if pred_shape.len() != 2 {
        return Err(TensorError::new("cross_entropy_loss: pred must be 2D [batch, classes]"));
    }
    let batch = pred_shape[0];

    // Native log-softmax: numerically stable (handles max subtraction internally)
    let log_softmax = pred.log_softmax(1)?;

    let target_shape = target.shape();

    if target_shape.len() == 1 && target_shape[0] == batch {
        // Index mode: target is [batch] class indices.
        // Gather the log-prob at each target index: log_softmax[i, target[i]]
        let indices = target.data().to_i64_vec()?;
        let classes = pred_shape[1];
        for &idx in &indices {
            if idx < 0 || idx >= classes {
                return Err(TensorError::new(&format!(
                    "cross_entropy_loss: target index {} out of range [0, {})", idx, classes
                )));
            }
        }
        let idx_tensor = target.data().reshape(&[batch, 1])?;
        let selected = log_softmax.gather(1, &idx_tensor)?; // [batch, 1]
        let per_sample = selected.reshape(&[batch])?;
        let total = per_sample.sum()?;
        total.mul_scalar(-1.0 / batch as f64)
    } else if target_shape.len() == 2 && target_shape[0] == batch && target_shape[1] == pred_shape[1] {
        // One-hot / soft-label mode: target is [batch, classes].
        let weighted = target.mul(&log_softmax)?;
        let per_sample = weighted.sum_dim(1, false)?; // [batch]
        let total = per_sample.sum()?;
        total.mul_scalar(-1.0 / batch as f64)
    } else {
        Err(TensorError::new(&format!(
            "cross_entropy_loss: target shape {:?} doesn't match pred shape {:?} — \
             expected [batch] indices or [batch, classes] one-hot",
            target_shape, pred_shape
        )))
    }
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
