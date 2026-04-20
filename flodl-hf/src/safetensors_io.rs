//! Safetensors format I/O and load-time validation.
//!
//! This module's primary public surface today is [`LoadValidation`] — a
//! structured diff between what a flodl model expects and what a HuggingFace
//! safetensors file actually contains. It surfaces three failure modes:
//!
//! 1. **Missing** — keys the model expects but the checkpoint lacks
//!    (typo in a `FlowBuilder::tag(...)` string, wrong model variant, etc.).
//! 2. **Unused** — keys the checkpoint contains but the model never asks for
//!    (architecture mismatch, stale pretrained file, etc.).
//! 3. **Shape mismatch** — a key matches by name but the tensor dimensions
//!    disagree (vocab size change, hidden dim change, head count change).
//!
//! The validator is the safety net behind flodl-hf's "string-named tag"
//! convention: typos and drift are caught at load time with a loud,
//! actionable error listing every key that disagrees.
//!
//! Actual tensor-data copying is a follow-up task and intentionally lives
//! outside this first pass.
//!
//! # Example (validator only)
//!
//! ```
//! use std::collections::HashMap;
//! use flodl_hf::safetensors_io::{ExpectedParam, validate_keys};
//!
//! let expected = vec![
//!     ExpectedParam { key: "bert.embeddings.word_embeddings.weight".into(), shape: vec![30522, 768] },
//!     ExpectedParam { key: "bert.pooler.dense.bias".into(),              shape: vec![768] },
//! ];
//! let mut actual: HashMap<String, Vec<i64>> = HashMap::new();
//! actual.insert("bert.embeddings.word_embeddings.weight".into(), vec![30522, 768]);
//! actual.insert("bert.pooler.dense.bias".into(),                  vec![768]);
//!
//! let v = validate_keys(&expected, &actual);
//! assert!(v.is_ok());
//! ```

use std::collections::{HashMap, HashSet};

use flodl::{Graph, Result, TensorError};

use crate::path::hf_key_from_flodl_key;

/// A parameter the model expects to find in a checkpoint.
///
/// `key`: the HF-dotted key as it appears in a safetensors file
/// (e.g. `bert.encoder.layer.0.attention.self.query.weight`).
/// `shape`: the tensor shape flodl will try to assign to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedParam {
    pub key: String,
    pub shape: Vec<i64>,
}

/// A single shape disagreement between model and checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeMismatch {
    pub key: String,
    pub expected: Vec<i64>,
    pub found: Vec<i64>,
}

/// The result of validating a checkpoint's key set against a model's
/// expected parameters.
///
/// Build this with [`validate_keys`]. Check [`is_ok`](LoadValidation::is_ok)
/// or convert to a loud `Result` via
/// [`into_result`](LoadValidation::into_result).
#[derive(Debug, Default, Clone)]
pub struct LoadValidation {
    pub missing: Vec<String>,
    pub unused: Vec<String>,
    pub shape_mismatches: Vec<ShapeMismatch>,
}

impl LoadValidation {
    /// True when there are no missing keys, no unused keys, and no shape
    /// mismatches.
    pub fn is_ok(&self) -> bool {
        self.missing.is_empty() && self.unused.is_empty() && self.shape_mismatches.is_empty()
    }

    /// Convert to a flodl `Result` — returns `Ok(())` when the validation is
    /// clean, otherwise a [`TensorError`] whose message lists every
    /// disagreement (truncated to the first 20 entries per bucket).
    pub fn into_result(self) -> Result<()> {
        if self.is_ok() {
            return Ok(());
        }
        let mut msg = String::from("safetensors checkpoint does not match model:\n");
        if !self.missing.is_empty() {
            msg.push_str(&format!(
                "  {} missing key(s) (model expects, checkpoint lacks):\n",
                self.missing.len(),
            ));
            for k in self.missing.iter().take(20) {
                msg.push_str(&format!("    - {k}\n"));
            }
            if self.missing.len() > 20 {
                msg.push_str(&format!("    ... and {} more\n", self.missing.len() - 20));
            }
        }
        if !self.unused.is_empty() {
            msg.push_str(&format!(
                "  {} unused key(s) (checkpoint has, model lacks):\n",
                self.unused.len(),
            ));
            for k in self.unused.iter().take(20) {
                msg.push_str(&format!("    - {k}\n"));
            }
            if self.unused.len() > 20 {
                msg.push_str(&format!("    ... and {} more\n", self.unused.len() - 20));
            }
        }
        if !self.shape_mismatches.is_empty() {
            msg.push_str(&format!(
                "  {} shape mismatch(es):\n",
                self.shape_mismatches.len(),
            ));
            for m in self.shape_mismatches.iter().take(20) {
                msg.push_str(&format!(
                    "    - {}: expected {:?}, found {:?}\n",
                    m.key, m.expected, m.found,
                ));
            }
            if self.shape_mismatches.len() > 20 {
                msg.push_str(&format!(
                    "    ... and {} more\n",
                    self.shape_mismatches.len() - 20,
                ));
            }
        }
        Err(TensorError::new(&msg))
    }
}

/// Validate model expectations against the `(key → shape)` map extracted
/// from a safetensors file.
///
/// Output is sorted (missing / unused / mismatches all ascending by key) so
/// error messages are stable across runs, which matters for diffing error
/// logs and writing tests.
pub fn validate_keys(
    expected: &[ExpectedParam],
    actual: &HashMap<String, Vec<i64>>,
) -> LoadValidation {
    let expected_keys: HashSet<&str> = expected.iter().map(|p| p.key.as_str()).collect();
    let mut v = LoadValidation::default();

    for p in expected {
        match actual.get(&p.key) {
            None => v.missing.push(p.key.clone()),
            Some(found) if found != &p.shape => {
                v.shape_mismatches.push(ShapeMismatch {
                    key: p.key.clone(),
                    expected: p.shape.clone(),
                    found: found.clone(),
                });
            }
            Some(_) => {}
        }
    }
    for k in actual.keys() {
        if !expected_keys.contains(k.as_str()) {
            v.unused.push(k.clone());
        }
    }
    v.missing.sort();
    v.unused.sort();
    v.shape_mismatches.sort_by(|a, b| a.key.cmp(&b.key));
    v
}

/// Collect expected parameters + buffers from a `Graph`, with keys already
/// converted to HF-dotted form via [`hf_key_from_flodl_key`].
///
/// Use this to drive [`validate_keys`] in the common case where the model
/// is a flodl `Graph` built via `FlowBuilder`.
pub fn expected_from_graph(graph: &Graph) -> Vec<ExpectedParam> {
    let mut out = Vec::new();
    for (k, p) in graph.named_parameters() {
        out.push(ExpectedParam {
            key: hf_key_from_flodl_key(&k),
            shape: p.variable.shape(),
        });
    }
    for (k, b) in graph.named_buffers() {
        out.push(ExpectedParam {
            key: hf_key_from_flodl_key(&k),
            shape: b.shape(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn actual_map(entries: &[(&str, &[i64])]) -> HashMap<String, Vec<i64>> {
        entries.iter().map(|(k, s)| ((*k).to_string(), s.to_vec())).collect()
    }

    #[test]
    fn all_keys_match_returns_ok() {
        let expected = vec![
            ExpectedParam { key: "bert.embeddings.word_embeddings.weight".into(), shape: vec![30522, 768] },
            ExpectedParam { key: "bert.pooler.dense.bias".into(),                  shape: vec![768] },
        ];
        let actual = actual_map(&[
            ("bert.embeddings.word_embeddings.weight", &[30522, 768]),
            ("bert.pooler.dense.bias",                  &[768]),
        ]);
        let v = validate_keys(&expected, &actual);
        assert!(v.is_ok());
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn missing_key_is_reported() {
        let expected = vec![
            ExpectedParam { key: "bert.pooler.dense.weight".into(), shape: vec![768, 768] },
        ];
        let actual = actual_map(&[]);
        let v = validate_keys(&expected, &actual);
        assert_eq!(v.missing, vec!["bert.pooler.dense.weight"]);
        assert!(v.unused.is_empty());
        assert!(v.shape_mismatches.is_empty());
    }

    #[test]
    fn unused_checkpoint_key_is_reported() {
        let expected: Vec<ExpectedParam> = Vec::new();
        let actual = actual_map(&[("bert.something.extra", &[4])]);
        let v = validate_keys(&expected, &actual);
        assert_eq!(v.unused, vec!["bert.something.extra"]);
        assert!(v.missing.is_empty());
        assert!(v.shape_mismatches.is_empty());
    }

    #[test]
    fn shape_mismatch_is_reported() {
        // Vocab size drift: checkpoint has 30522 tokens, model expects 50257.
        let expected = vec![
            ExpectedParam {
                key: "bert.embeddings.word_embeddings.weight".into(),
                shape: vec![50257, 768],
            },
        ];
        let actual = actual_map(&[
            ("bert.embeddings.word_embeddings.weight", &[30522, 768]),
        ]);
        let v = validate_keys(&expected, &actual);
        assert!(v.missing.is_empty());
        assert!(v.unused.is_empty());
        assert_eq!(v.shape_mismatches.len(), 1);
        assert_eq!(v.shape_mismatches[0].key, "bert.embeddings.word_embeddings.weight");
        assert_eq!(v.shape_mismatches[0].expected, vec![50257, 768]);
        assert_eq!(v.shape_mismatches[0].found,    vec![30522, 768]);
    }

    #[test]
    fn typo_queri_vs_query_reports_both_missing_and_unused() {
        // The motivating bug: author typed "queri" in a tag, checkpoint has "query".
        // Validator must surface both sides so the typo is unambiguous.
        let expected = vec![
            ExpectedParam { key: "bert.encoder.layer.0.attention.self.queri.weight".into(), shape: vec![768, 768] },
        ];
        let actual = actual_map(&[
            ("bert.encoder.layer.0.attention.self.query.weight", &[768, 768]),
        ]);
        let v = validate_keys(&expected, &actual);
        assert_eq!(v.missing, vec!["bert.encoder.layer.0.attention.self.queri.weight"]);
        assert_eq!(v.unused,  vec!["bert.encoder.layer.0.attention.self.query.weight"]);
    }

    #[test]
    fn mixed_failures_accumulate() {
        let expected = vec![
            ExpectedParam { key: "ok.weight".into(),        shape: vec![4] },
            ExpectedParam { key: "missing.weight".into(),   shape: vec![8] },
            ExpectedParam { key: "wrong_shape.weight".into(), shape: vec![16] },
        ];
        let actual = actual_map(&[
            ("ok.weight",            &[4]),
            ("wrong_shape.weight",   &[32]),
            ("extra.weight",         &[1]),
        ]);
        let v = validate_keys(&expected, &actual);
        assert_eq!(v.missing,          vec!["missing.weight"]);
        assert_eq!(v.unused,           vec!["extra.weight"]);
        assert_eq!(v.shape_mismatches.len(), 1);
        assert_eq!(v.shape_mismatches[0].key, "wrong_shape.weight");
    }

    #[test]
    fn into_result_error_message_lists_every_bucket() {
        let expected = vec![
            ExpectedParam { key: "m.w".into(),  shape: vec![2] },
            ExpectedParam { key: "sm.w".into(), shape: vec![3] },
        ];
        let actual = actual_map(&[
            ("sm.w",    &[4]),
            ("extra.w", &[1]),
        ]);
        let v = validate_keys(&expected, &actual);
        let err = v.into_result().unwrap_err().to_string();
        assert!(err.contains("1 missing key"),     "missing bucket not in msg: {err}");
        assert!(err.contains("1 unused key"),      "unused bucket not in msg: {err}");
        assert!(err.contains("1 shape mismatch"),  "shape bucket not in msg: {err}");
        assert!(err.contains("m.w"));
        assert!(err.contains("extra.w"));
        assert!(err.contains("sm.w"));
        assert!(err.contains("[3]"));
        assert!(err.contains("[4]"));
    }

    #[test]
    fn output_is_sorted_for_stable_messages() {
        let expected = vec![
            ExpectedParam { key: "z.w".into(), shape: vec![1] },
            ExpectedParam { key: "a.w".into(), shape: vec![1] },
        ];
        let actual = actual_map(&[
            ("m.w", &[1]),
            ("c.w", &[1]),
        ]);
        let v = validate_keys(&expected, &actual);
        assert_eq!(v.missing, vec!["a.w", "z.w"]);
        assert_eq!(v.unused,  vec!["c.w", "m.w"]);
    }

    #[test]
    fn empty_everywhere_is_ok() {
        let v = validate_keys(&[], &HashMap::new());
        assert!(v.is_ok());
        assert!(v.missing.is_empty());
        assert!(v.unused.is_empty());
        assert!(v.shape_mismatches.is_empty());
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn into_result_truncates_long_missing_list() {
        // 25 missing keys — "... and N more" tail should appear for any
        // bucket longer than the 20-entry cap.
        let expected: Vec<ExpectedParam> = (0..25)
            .map(|i| ExpectedParam { key: format!("key.{i:02}"), shape: vec![1] })
            .collect();
        let v = validate_keys(&expected, &HashMap::new());
        assert_eq!(v.missing.len(), 25);
        let err = v.into_result().unwrap_err().to_string();
        assert!(err.contains("25 missing key"), "header must show full count: {err}");
        assert!(err.contains("... and 5 more"),
            "truncation tail must show remaining count: {err}");
        // First 20 keys listed; 21st onwards must not appear verbatim.
        assert!(err.contains("key.00"));
        assert!(err.contains("key.19"));
        assert!(!err.contains("key.20"));
    }

    #[test]
    fn expected_from_graph_converts_slash_to_dot() {
        use flodl::{FlowBuilder, Linear, Module};
        let fb = FlowBuilder::new()
            .through(Linear::new(4, 2).unwrap()).tag("bert.pooler.dense");
        let graph = fb.build().unwrap();
        let expected = expected_from_graph(&graph);
        // Graph::named_parameters gives "bert.pooler.dense/weight" and
        // "bert.pooler.dense/bias". expected_from_graph must swap the last
        // slash for a dot.
        let keys: Vec<&str> = expected.iter().map(|e| e.key.as_str()).collect();
        assert!(keys.contains(&"bert.pooler.dense.weight"),
                "expected HF-dotted key missing, got {keys:?}");
        assert!(keys.contains(&"bert.pooler.dense.bias"),
                "expected HF-dotted key missing, got {keys:?}");
        // Sanity check: the parameter count matches Graph's own view.
        assert_eq!(expected.len(), graph.parameters().len());
    }
}
