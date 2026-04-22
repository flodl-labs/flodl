//! Shared `config.json` parsing helpers used by every `*Config::from_json_str`.
//!
//! HuggingFace model configs agree on a small vocabulary for the fields
//! that affect model shape and task-head metadata (`vocab_size`,
//! `num_labels`, `id2label`, …). Each model's `from_json_str` was
//! re-implementing the same per-field readers and the same
//! `id2label → Vec<String>` normalization; this module centralizes
//! both.

use serde_json::Value;

use flodl::{Result, TensorError};

/// Read a required integer field. Errors if missing, null, or not an
/// integer.
pub(crate) fn required_i64(v: &Value, key: &str) -> Result<i64> {
    v.get(key).and_then(|x| x.as_i64()).ok_or_else(|| {
        TensorError::new(&format!(
            "config.json missing required integer field: {key}",
        ))
    })
}

/// Read a required string field. Errors if missing, null, or not a
/// string. Used by [`crate::models::auto::AutoConfig`] to dispatch on
/// `model_type`.
pub(crate) fn required_string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key).and_then(|x| x.as_str()).ok_or_else(|| {
        TensorError::new(&format!(
            "config.json missing required string field: {key}",
        ))
    })
}

/// Read an integer field with a fallback default. Accepts both
/// explicit integers and absent / null values.
pub(crate) fn optional_i64(v: &Value, key: &str, default: i64) -> i64 {
    v.get(key).and_then(|x| x.as_i64()).unwrap_or(default)
}

/// Read an optional integer field. `None` when the key is absent or
/// explicitly null; used for `pad_token_id` where BERT-family configs
/// may legitimately omit it (e.g. LLaMA-derived `pad == eos` setups
/// prefer `None`).
pub(crate) fn optional_i64_or_none(v: &Value, key: &str) -> Option<i64> {
    v.get(key).and_then(|x| x.as_i64())
}

/// Read a float field with a fallback default.
pub(crate) fn optional_f64(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
}

/// Read a boolean field with a fallback default.
pub(crate) fn optional_bool(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
}

/// Parse HuggingFace's `id2label` field into an ordered `Vec<String>`.
///
/// HF stores the mapping as `{"0": "LABEL_0", "1": "LABEL_1", ...}`
/// (string keys, string values). This helper sorts by integer key so
/// the returned vector is indexable by class id — `vec[k]` matches HF
/// Python's `config.id2label[k]`.
///
/// Errors on:
/// - a `k` that can't be parsed as an integer,
/// - a value that isn't a string,
/// - non-contiguous ids (gap or duplicate), which would silently
///   misalign names with logits rows.
///
/// Returns `Ok(None)` when the key is absent or not an object.
pub(crate) fn parse_id2label(v: &Value) -> Result<Option<Vec<String>>> {
    let obj = match v.get("id2label").and_then(|x| x.as_object()) {
        Some(obj) => obj,
        None => return Ok(None),
    };
    let mut pairs: Vec<(i64, String)> = Vec::with_capacity(obj.len());
    for (k, val) in obj {
        let id: i64 = k.parse().map_err(|_| {
            TensorError::new(&format!(
                "config.json: id2label key {k:?} is not an integer",
            ))
        })?;
        let label = val.as_str().ok_or_else(|| {
            TensorError::new(&format!(
                "config.json: id2label[{k}] is not a string",
            ))
        })?;
        pairs.push((id, label.to_string()));
    }
    pairs.sort_by_key(|(id, _)| *id);
    for (idx, (id, _)) in pairs.iter().enumerate() {
        if *id != idx as i64 {
            return Err(TensorError::new(&format!(
                "config.json: id2label must have contiguous ids 0..N, \
                 but index {idx} has id {id}",
            )));
        }
    }
    Ok(Some(pairs.into_iter().map(|(_, s)| s).collect()))
}

/// Derive `num_labels` from the config.
///
/// Priority: explicit `num_labels` field wins; otherwise fall back to
/// the length of the already-parsed `id2label` list. Returns `None`
/// when neither is present (base model configs that aren't fine-tuned
/// as task heads).
pub(crate) fn parse_num_labels(v: &Value, id2label: Option<&[String]>) -> Option<i64> {
    v.get("num_labels")
        .and_then(|x| x.as_i64())
        .or_else(|| id2label.map(|v| v.len() as i64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_i64_reads_or_errors() {
        let v: Value = serde_json::from_str(r#"{"a": 42}"#).unwrap();
        assert_eq!(required_i64(&v, "a").unwrap(), 42);
        assert!(required_i64(&v, "missing").is_err());
    }

    #[test]
    fn required_string_reads_or_errors() {
        let v: Value = serde_json::from_str(r#"{"model_type": "bert", "num": 7}"#).unwrap();
        assert_eq!(required_string(&v, "model_type").unwrap(), "bert");
        assert!(required_string(&v, "missing").is_err());
        assert!(
            required_string(&v, "num").is_err(),
            "non-string values must error"
        );
    }

    #[test]
    fn optional_i64_falls_back() {
        let v: Value = serde_json::from_str(r#"{"a": 42}"#).unwrap();
        assert_eq!(optional_i64(&v, "a", 7), 42);
        assert_eq!(optional_i64(&v, "missing", 7), 7);
    }

    #[test]
    fn optional_i64_or_none_treats_absence_as_none() {
        let v: Value = serde_json::from_str(r#"{"a": 0, "b": null}"#).unwrap();
        assert_eq!(optional_i64_or_none(&v, "a"), Some(0));
        assert_eq!(optional_i64_or_none(&v, "b"), None);
        assert_eq!(optional_i64_or_none(&v, "missing"), None);
    }

    #[test]
    fn optional_f64_and_bool_defaults() {
        let v: Value = serde_json::from_str(r#"{"x": 0.5, "b": true}"#).unwrap();
        assert!((optional_f64(&v, "x", 1.0) - 0.5).abs() < 1e-12);
        assert!((optional_f64(&v, "missing", 1.0) - 1.0).abs() < 1e-12);
        assert!(optional_bool(&v, "b", false));
        assert!(!optional_bool(&v, "missing", false));
    }

    #[test]
    fn parse_id2label_orders_and_rejects_gaps() {
        let v: Value = serde_json::from_str(
            r#"{"id2label": {"2": "c", "0": "a", "1": "b"}}"#,
        ).unwrap();
        let out = parse_id2label(&v).unwrap().unwrap();
        assert_eq!(out, vec!["a", "b", "c"]);

        let gap: Value = serde_json::from_str(
            r#"{"id2label": {"0": "a", "2": "c"}}"#,
        ).unwrap();
        let err = parse_id2label(&gap).unwrap_err();
        assert!(format!("{err}").contains("contiguous"), "got: {err}");
    }

    #[test]
    fn parse_id2label_absent_is_none() {
        let v: Value = serde_json::from_str(r#"{}"#).unwrap();
        assert!(parse_id2label(&v).unwrap().is_none());
    }

    #[test]
    fn parse_num_labels_explicit_wins() {
        let v: Value = serde_json::from_str(r#"{"num_labels": 5}"#).unwrap();
        let labels = vec!["a".to_string(), "b".to_string()];
        assert_eq!(parse_num_labels(&v, Some(&labels)), Some(5));
    }

    #[test]
    fn parse_num_labels_falls_back_to_id2label_len() {
        let v: Value = serde_json::from_str(r#"{}"#).unwrap();
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(parse_num_labels(&v, Some(&labels)), Some(3));
    }

    #[test]
    fn parse_num_labels_none_when_both_missing() {
        let v: Value = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(parse_num_labels(&v, None), None);
    }
}
