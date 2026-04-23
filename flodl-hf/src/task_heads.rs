//! Shared task-head types and helpers.
//!
//! [`Answer`] and [`TokenPrediction`] are the public output types for
//! question-answering and token-classification heads across every
//! model family (BERT, RoBERTa, DistilBERT, …). Centralising them here
//! means callers can match on one type regardless of which model
//! produced the prediction, and the three families don't each define
//! their own copy.
//!
//! The module-internal helpers (`default_labels`, `check_num_labels`,
//! `logits_to_sorted_labels`, `extract_best_span`) carry the behaviour
//! that's identical across families — label fallback naming, positive-
//! `num_labels` guard, softmax-and-sort for sequence classification,
//! and the QA span-search algorithm.

use flodl::{cross_entropy_loss, Result, TensorError, Variable};

/// One extracted answer span from a
/// `*ForQuestionAnswering::answer` / `::answer_batch` call.
#[derive(Debug, Clone)]
pub struct Answer {
    /// Decoded answer text (special tokens stripped).
    pub text: String,
    /// Start token index in the input sequence.
    pub start: usize,
    /// End token index in the input sequence (inclusive).
    pub end: usize,
    /// `start_logit + end_logit` at the chosen span. Unnormalised — use
    /// for comparing spans within one `answer` call, not across calls.
    pub score: f32,
}

/// One labelled token inside a `*ForTokenClassification` prediction.
#[cfg(feature = "tokenizer")]
#[derive(Debug, Clone)]
pub struct TokenPrediction {
    /// The subword string the tokenizer produced for this position.
    pub token: String,
    /// The highest-scoring label for this token.
    pub label: String,
    /// Softmax probability of `label`.
    pub score: f32,
    /// `true` for real tokens, `false` for padding (`attention_mask ==
    /// 0`). Lets callers drop padding without re-tokenising.
    pub attends: bool,
}

/// Default label names when a checkpoint ships without `id2label`.
/// Mirrors HF Python's `LABEL_0`, `LABEL_1`, ...
pub(crate) fn default_labels(n: i64) -> Vec<String> {
    (0..n).map(|i| format!("LABEL_{i}")).collect()
}

/// Clamp `num_labels` to a positive value or surface a loud error.
/// Used by every task-head `on_device` constructor so the error
/// message is consistent across families.
pub(crate) fn check_num_labels(n: i64) -> Result<i64> {
    if n <= 0 {
        return Err(TensorError::new(&format!(
            "num_labels must be > 0, got {n}",
        )));
    }
    Ok(n)
}

// ── Loss functions ───────────────────────────────────────────────────────

/// Compute sequence-classification loss.
///
/// `logits`: `[batch, num_labels]` - the raw output of a
/// `*ForSequenceClassification` head's classifier.
///
/// `labels`: `[batch]` Int64 class indices (hard labels) or
/// `[batch, num_labels]` Float soft labels. Auto-detected by libtorch
/// through [`cross_entropy_loss`].
///
/// Mirrors HF Python's `*ForSequenceClassification(..., labels=labels)`
/// loss computation (the `single_label_classification` branch). A
/// dedicated regression branch (`problem_type == "regression"`, MSE) is
/// not yet exposed - pass raw logits to [`flodl::mse_loss`] directly for
/// that case.
pub fn sequence_classification_loss(
    logits: &Variable,
    labels: &Variable,
) -> Result<Variable> {
    let shape = logits.shape();
    if shape.len() != 2 {
        return Err(TensorError::new(&format!(
            "sequence_classification_loss: logits must be [batch, num_labels], got {shape:?}",
        )));
    }
    cross_entropy_loss(logits, labels)
}

/// Compute token-classification loss.
///
/// `logits`: `[batch, seq_len, num_labels]` - the raw output of a
/// `*ForTokenClassification` head.
///
/// `labels`: `[batch, seq_len]` Int64 class indices. Use `-100` at
/// positions the loss should ignore: special tokens (`[CLS]`, `[SEP]`,
/// padding) and any non-first subword of a word under the standard
/// BIO-to-subword alignment rule. Matches HF Python's
/// `CrossEntropyLoss(ignore_index=-100)` default, which flodl's
/// [`cross_entropy_loss`] wires through natively.
pub fn token_classification_loss(
    logits: &Variable,
    labels: &Variable,
) -> Result<Variable> {
    let shape = logits.shape();
    if shape.len() != 3 {
        return Err(TensorError::new(&format!(
            "token_classification_loss: logits must be [batch, seq_len, num_labels], got {shape:?}",
        )));
    }
    let num_labels = shape[2];
    let flat_logits = logits.reshape(&[-1, num_labels])?;
    let flat_labels = labels.reshape(&[-1])?;
    cross_entropy_loss(&flat_logits, &flat_labels)
}

/// Compute extractive question-answering loss.
///
/// `logits`: `[batch, seq_len, 2]` - the raw output of a
/// `*ForQuestionAnswering` head. Start logits are on slice `0` of the
/// last axis, end logits on slice `1`.
///
/// `start_positions`, `end_positions`: `[batch]` Int64 token indices of
/// the gold span's inclusive start and end. Both must lie in
/// `[0, seq_len)`; out-of-bounds positions are a caller error rather
/// than a silently-ignored training signal. (HF Python clamps then uses
/// `ignore_index == seq_len` to drop no-answer examples; flodl's
/// [`cross_entropy_loss`] fixes `ignore_index=-100`, so filter
/// no-answer examples upstream or assign them position `0` per your
/// dataset convention.)
///
/// Returns `(start_loss + end_loss) / 2`, matching HF Python's
/// `*ForQuestionAnswering(..., start_positions=..., end_positions=...)`.
pub fn question_answering_loss(
    logits: &Variable,
    start_positions: &Variable,
    end_positions: &Variable,
) -> Result<Variable> {
    let shape = logits.shape();
    if shape.len() != 3 || shape[2] != 2 {
        return Err(TensorError::new(&format!(
            "question_answering_loss: logits must be [batch, seq_len, 2], got {shape:?}",
        )));
    }
    let start_logits = logits.narrow(-1, 0, 1)?.squeeze(-1)?;
    let end_logits   = logits.narrow(-1, 1, 1)?.squeeze(-1)?;
    let start_loss = cross_entropy_loss(&start_logits, start_positions)?;
    let end_loss   = cross_entropy_loss(&end_logits, end_positions)?;
    start_loss.add(&end_loss)?.mul_scalar(0.5)
}

/// Apply softmax to a `[batch, num_labels]` logits tensor and return a
/// sorted `(label, score)` list per batch entry, descending by score.
pub(crate) fn logits_to_sorted_labels(
    logits: &Variable,
    id2label: &[String],
) -> Result<Vec<Vec<(String, f32)>>> {
    let probs = logits.softmax(-1)?;
    let shape = probs.shape();
    assert_eq!(shape.len(), 2, "expected [batch, num_labels], got {shape:?}");
    let batch = shape[0] as usize;
    let n = shape[1] as usize;
    assert_eq!(
        n,
        id2label.len(),
        "classifier output width {n} != id2label count {}",
        id2label.len(),
    );
    let flat = probs.data().to_f32_vec()?;
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut row: Vec<(String, f32)> = (0..n)
            .map(|k| (id2label[k].clone(), flat[b * n + k]))
            .collect();
        row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.push(row);
    }
    Ok(out)
}

/// Extract best QA spans from a `[B, S, 2]` logits tensor.
///
/// Algorithm matches HF Python's "argmax over valid positions, then
/// multiply": softmax start and end independently along the sequence
/// axis, then pick the `(i, j)` with `i <= j` maximising
/// `start_prob[i] + end_prob[j]`. Valid positions are restricted to
/// tokens with `sequence_ids == 1` (the context region in a
/// question/context pair encoding); `-1` (specials / padding)
/// already excludes padding, so no separate attention-mask check is
/// needed.
///
/// Decoding of the chosen span goes through the attached tokenizer
/// with `skip_special_tokens=true`, matching HF's default behaviour.
///
/// Errors:
/// - if a batch entry has no tokens with `sequence_id == 1`
///   (tokenizer didn't produce a pair encoding), since there's no
///   meaningful span to extract.
/// - on any shape mismatch: `logits.shape()` must be `[B, S, 2]`.
#[cfg(feature = "tokenizer")]
pub(crate) fn extract_best_span(
    logits: &Variable,
    enc: &crate::tokenizer::EncodedBatch,
    tokenizer: &crate::tokenizer::HfTokenizer,
) -> Result<Vec<Answer>> {
    let shape = logits.shape();
    assert_eq!(shape.len(), 3, "expected [B, S, 2], got {shape:?}");
    let batch = shape[0] as usize;
    let seq = shape[1] as usize;
    assert_eq!(shape[2], 2, "QA head must be 2-wide, got {}", shape[2]);

    let starts = logits.narrow(-1, 0, 1)?.softmax(1)?;
    let ends = logits.narrow(-1, 1, 1)?.softmax(1)?;
    let starts_flat = starts.data().to_f32_vec()?;
    let ends_flat = ends.data().to_f32_vec()?;
    let sequence_ids: Vec<i64> = enc.sequence_ids.data().to_i64_vec()?;
    let input_ids: Vec<i64> = enc.input_ids.data().to_i64_vec()?;

    let mut answers = Vec::with_capacity(batch);
    for b in 0..batch {
        let offset = b * seq;
        let valid: Vec<usize> = (0..seq)
            .filter(|&s| sequence_ids[offset + s] == 1)
            .collect();
        if valid.is_empty() {
            return Err(TensorError::new(
                "QA extract: no context tokens (sequence_id == 1) found; \
                 tokenizer did not produce a pair encoding",
            ));
        }
        let mut best = (valid[0], valid[0], f32::NEG_INFINITY);
        for &i in &valid {
            let sp = starts_flat[offset + i];
            for &j in valid.iter().filter(|&&j| j >= i) {
                let ep = ends_flat[offset + j];
                let score = sp + ep;
                if score > best.2 {
                    best = (i, j, score);
                }
            }
        }
        let (start, end, score) = best;
        let span_ids: Vec<u32> = input_ids[offset + start..=offset + end]
            .iter()
            .map(|&x| x as u32)
            .collect();
        let text = tokenizer
            .inner()
            .decode(&span_ids, /*skip_special_tokens=*/ true)
            .map_err(|e| TensorError::new(&format!("qa decode: {e}")))?;
        answers.push(Answer { text, start, end, score });
    }
    Ok(answers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flodl::{DType, Device, Tensor, TensorOptions};
    fn cpu() -> Device { Device::CPU }

    #[test]
    fn default_labels_generates_label_k_fallback() {
        assert_eq!(default_labels(3), vec!["LABEL_0", "LABEL_1", "LABEL_2"]);
        assert!(default_labels(0).is_empty());
    }

    #[test]
    fn check_num_labels_rejects_nonpositive() {
        assert_eq!(check_num_labels(3).unwrap(), 3);
        assert!(check_num_labels(0).is_err());
        assert!(check_num_labels(-1).is_err());
    }

    fn logits_2d(data: &[f32], rows: i64, cols: i64) -> Variable {
        Variable::new(
            Tensor::from_f32(data, &[rows, cols], cpu()).unwrap(),
            true,
        )
    }

    fn labels_1d(data: &[i64], n: i64) -> Variable {
        Variable::new(
            Tensor::from_i64(data, &[n], cpu()).unwrap(),
            false,
        )
    }

    #[test]
    fn sequence_classification_loss_rejects_wrong_rank() {
        let logits = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], cpu()).unwrap(),
            true,
        );
        let labels = labels_1d(&[0, 1, 0, 1], 4);
        let err = sequence_classification_loss(&logits, &labels).unwrap_err();
        assert!(err.to_string().contains("must be [batch, num_labels]"));
    }

    #[test]
    fn sequence_classification_loss_backward_flows() {
        // 2 batches, 3 classes. Correct class gets high logit so loss is small.
        let logits = logits_2d(&[5.0, 0.1, 0.1, 0.1, 5.0, 0.1], 2, 3);
        let labels = labels_1d(&[0, 1], 2);
        let loss = sequence_classification_loss(&logits, &labels).unwrap();
        loss.backward().unwrap();
        assert!(logits.grad().is_some(), "logits must receive grad");
        let loss_val = loss.data().to_f32_vec().unwrap()[0];
        assert!(loss_val < 0.1, "expected small loss, got {loss_val}");
    }

    #[test]
    fn token_classification_loss_flattens_and_ignores_minus_100() {
        // batch=2, seq=3, num_labels=2. Labels: [[0, -100, 1], [1, 0, -100]].
        // Position with -100 should not contribute to the loss.
        let logits_data = [
            5.0, 0.0,   0.0, 0.0,   0.0, 5.0,   // batch 0
            0.0, 5.0,   5.0, 0.0,   0.0, 0.0,   // batch 1
        ];
        let logits = Variable::new(
            Tensor::from_f32(&logits_data, &[2, 3, 2], cpu()).unwrap(),
            true,
        );
        let labels = Variable::new(
            Tensor::from_i64(&[0, -100, 1, 1, 0, -100], &[2, 3], cpu()).unwrap(),
            false,
        );
        let loss = token_classification_loss(&logits, &labels).unwrap();
        loss.backward().unwrap();
        assert!(logits.grad().is_some(), "logits must receive grad");
        // All 4 non-ignored positions are confidently correct, so loss is tiny.
        let loss_val = loss.data().to_f32_vec().unwrap()[0];
        assert!(loss_val < 0.1, "expected small loss (all correct), got {loss_val}");
    }

    #[test]
    fn token_classification_loss_rejects_wrong_rank() {
        let logits = logits_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let labels = labels_1d(&[0, 1], 2);
        let err = token_classification_loss(&logits, &labels).unwrap_err();
        assert!(err.to_string().contains("[batch, seq_len, num_labels]"));
    }

    #[test]
    fn question_answering_loss_averages_two_heads() {
        // batch=2, seq=4. QA logits stack (start, end) on last dim.
        // Gold spans: batch 0 → (start=1, end=2), batch 1 → (start=0, end=3).
        // Logits peaked at the gold positions for both heads: loss ~0.
        let opts = TensorOptions { dtype: DType::Float32, device: cpu() };
        let logits_flat = Tensor::zeros(&[2, 4, 2], opts).unwrap();
        // Write peaks via an inplace fill through addition: build from raw data instead.
        let raw: Vec<f32> = {
            let mut v = vec![0.0_f32; 2 * 4 * 2];
            // helper to compute linear index from (b, s, k)
            let ix = |b: usize, s: usize, k: usize| (b * 4 + s) * 2 + k;
            // batch 0 start=1, end=2
            v[ix(0, 1, 0)] = 10.0;
            v[ix(0, 2, 1)] = 10.0;
            // batch 1 start=0, end=3
            v[ix(1, 0, 0)] = 10.0;
            v[ix(1, 3, 1)] = 10.0;
            v
        };
        drop(logits_flat);
        let logits = Variable::new(
            Tensor::from_f32(&raw, &[2, 4, 2], cpu()).unwrap(),
            true,
        );
        let starts = labels_1d(&[1, 0], 2);
        let ends   = labels_1d(&[2, 3], 2);
        let loss = question_answering_loss(&logits, &starts, &ends).unwrap();
        loss.backward().unwrap();
        assert!(logits.grad().is_some(), "logits must receive grad");
        let loss_val = loss.data().to_f32_vec().unwrap()[0];
        assert!(loss_val < 0.01, "expected tiny loss at peaked logits, got {loss_val}");
    }

    #[test]
    fn question_answering_loss_rejects_wrong_last_dim() {
        let logits = Variable::new(
            Tensor::from_f32(&[0.0_f32; 12], &[2, 3, 2], cpu()).unwrap(),
            true,
        );
        let starts = labels_1d(&[0, 1], 2);
        let ends   = labels_1d(&[2, 2], 2);
        assert!(question_answering_loss(&logits, &starts, &ends).is_ok());

        let bad = Variable::new(
            Tensor::from_f32(&[0.0_f32; 18], &[2, 3, 3], cpu()).unwrap(),
            true,
        );
        let err = question_answering_loss(&bad, &starts, &ends).unwrap_err();
        assert!(err.to_string().contains("[batch, seq_len, 2]"));
    }
}
