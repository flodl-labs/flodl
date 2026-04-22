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

#[cfg(feature = "tokenizer")]
use flodl::Variable;
use flodl::{Result, TensorError};

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
}
