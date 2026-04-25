//! HuggingFace-compatible export: write a trained flodl model as a
//! directory HF Python can load with `AutoModel.from_pretrained`.
//!
//! # What this produces
//!
//! A directory containing:
//! - `model.safetensors` — the graph's named parameters and buffers,
//!   serialised via [`crate::safetensors_io::save_safetensors_file_from_graph`].
//!   HF-dotted keys are applied (`gamma`/`beta` stay as modern
//!   `weight`/`bias`; no legacy rename is emitted).
//! - `config.json` — model architecture metadata. The caller provides
//!   this as a pre-built string, typically from the family's
//!   `to_json_str()` method (e.g.
//!   [`BertConfig::to_json_str`](crate::models::bert::BertConfig::to_json_str)).
//!
//! Tokenizer files are **not** emitted — `tokenizer.json`,
//! `vocab.txt`, `sentencepiece.bpe.model`, and the various `*_config.json`
//! files are orthogonal to model state and stay the caller's
//! responsibility (usually: copy them from the base checkpoint
//! directory).
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use flodl::Device;
//! use flodl_hf::export::export_hf_dir;
//! use flodl_hf::models::bert::{BertConfig, BertModel};
//!
//! # fn run() -> flodl::Result<()> {
//! let config = BertConfig::bert_base_uncased();
//! let graph  = BertModel::on_device(&config, Device::CPU)?;
//! export_hf_dir(&graph, &config.to_json_str(), Path::new("/tmp/my-bert"))?;
//! # Ok(())
//! # }
//! ```
//!
//! After this call, `/tmp/my-bert` is ready for
//! `AutoModel.from_pretrained("/tmp/my-bert")` in HF Python.

use std::path::Path;

use flodl::{Device, Graph, Result, TensorError};

use crate::models::albert::AlbertModel;
use crate::models::auto::AutoConfig;
use crate::models::bert::BertModel;
use crate::models::deberta_v2::DebertaV2Model;
use crate::models::distilbert::DistilBertModel;
use crate::models::roberta::RobertaModel;
use crate::models::xlm_roberta::XlmRobertaModel;
use crate::safetensors_io::save_safetensors_file_from_graph;

/// Write a HuggingFace-compatible export directory.
///
/// Creates `out_dir` (and any missing parents) if it doesn't exist,
/// then writes `model.safetensors` + `config.json` inside it. Both
/// files are overwritten on re-export — no staging or atomic rename
/// is attempted; the caller is expected to export into a fresh dir.
///
/// The `config_json` string is written verbatim. Build it with the
/// family's `to_json_str()` so the emitted `model_type` and field
/// names match what HF `AutoConfig` expects.
///
/// Returns an error if directory creation or file writes fail; I/O
/// error messages include the offending path.
pub fn export_hf_dir(graph: &Graph, config_json: &str, out_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(out_dir).map_err(|e| {
        TensorError::new(&format!(
            "export_hf_dir: create_dir_all {}: {e}",
            out_dir.display(),
        ))
    })?;

    let model_path = out_dir.join("model.safetensors");
    save_safetensors_file_from_graph(graph, &model_path)?;

    let config_path = out_dir.join("config.json");
    std::fs::write(&config_path, config_json).map_err(|e| {
        TensorError::new(&format!(
            "export_hf_dir: write {}: {e}",
            config_path.display(),
        ))
    })?;

    Ok(())
}

/// Build a graph matching `config` for HF round-trip export, on `device`.
///
/// `has_pooler` selects between with-pooler and without-pooler variants
/// for pooler-bearing families (BERT, RoBERTa, XLM-R, ALBERT). Pass
/// `true` when the source checkpoint ships pooler weights — matches
/// HF `AutoModel.from_pretrained` parity. Pooler-less families
/// (DistilBERT, DeBERTa-v2) ignore the flag.
///
/// Use this when you have a parsed `AutoConfig` (e.g. read from a
/// flodl checkpoint sidecar) and need to instantiate the matching
/// graph topology before loading weights via
/// [`flodl::Graph::load_checkpoint`]. For Hub-fetch flows reach for
/// [`crate::models::auto::AutoModel::from_pretrained_for_export`]
/// directly — it builds, fetches, and loads in one call.
pub fn build_for_export(
    config: &AutoConfig,
    has_pooler: bool,
    device: Device,
) -> Result<Graph> {
    match config {
        AutoConfig::Bert(c) => {
            if has_pooler {
                BertModel::on_device(c, device)
            } else {
                BertModel::on_device_without_pooler(c, device)
            }
        }
        AutoConfig::Roberta(c) => {
            if has_pooler {
                RobertaModel::on_device(c, device)
            } else {
                RobertaModel::on_device_without_pooler(c, device)
            }
        }
        AutoConfig::DistilBert(c) => DistilBertModel::on_device(c, device),
        AutoConfig::XlmRoberta(c) => {
            if has_pooler {
                XlmRobertaModel::on_device(c, device)
            } else {
                XlmRobertaModel::on_device_without_pooler(c, device)
            }
        }
        AutoConfig::Albert(c) => {
            if has_pooler {
                AlbertModel::on_device(c, device)
            } else {
                AlbertModel::on_device_without_pooler(c, device)
            }
        }
        AutoConfig::DebertaV2(c) => DebertaV2Model::on_device(c, device),
    }
}

/// Convenience: detect pooler presence from a list of checkpoint keys
/// (e.g. the output of [`flodl::checkpoint_keys`]). Returns `true` when
/// any key starts with `pooler` — matches the convention used by
/// flodl-hf's family backbones.
pub fn keys_have_pooler(keys: &[String]) -> bool {
    keys.iter().any(|k| k.starts_with("pooler/") || k.starts_with("pooler."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bert::{BertConfig, BertModel};
    use flodl::Device;

    /// Unique tempdir root per test — `std::env::temp_dir()` + pid +
    /// tag matches the pattern used elsewhere in this crate (see
    /// `tests/roundtrip_common/mod.rs`) and avoids pulling in a
    /// `tempfile` dev-dep for two unit tests.
    fn unique_tempdir(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("flodl_hf_export_{tag}_{}", std::process::id()))
    }

    /// Smoke test: a BERT preset exports into a fresh dir, producing
    /// exactly `model.safetensors` + `config.json` with non-empty
    /// contents and a config that parses back to the same preset.
    #[test]
    fn export_hf_dir_writes_both_files() {
        let out = unique_tempdir("writes_both").join("bert-export");
        let _ = std::fs::remove_dir_all(&out);

        let config = BertConfig::bert_base_uncased();
        let graph = BertModel::on_device(&config, Device::CPU).unwrap();

        export_hf_dir(&graph, &config.to_json_str(), &out).unwrap();

        let model_path = out.join("model.safetensors");
        let config_path = out.join("config.json");

        assert!(model_path.exists(), "model.safetensors not written");
        assert!(config_path.exists(), "config.json not written");
        assert!(
            std::fs::metadata(&model_path).unwrap().len() > 0,
            "model.safetensors is empty",
        );

        // Config round-trips back through the parser.
        let written = std::fs::read_to_string(&config_path).unwrap();
        let recovered = BertConfig::from_json_str(&written).unwrap();
        assert_eq!(recovered.to_json_str(), config.to_json_str());

        let _ = std::fs::remove_dir_all(&out);
    }

    /// Creates missing parent directories — callers shouldn't need to
    /// pre-make the full path.
    #[test]
    fn export_hf_dir_creates_missing_parents() {
        let root = unique_tempdir("missing_parents");
        let _ = std::fs::remove_dir_all(&root);
        let nested = root.join("a").join("b").join("c");

        let config = BertConfig::bert_base_uncased();
        let graph = BertModel::on_device(&config, Device::CPU).unwrap();

        export_hf_dir(&graph, &config.to_json_str(), &nested).unwrap();

        assert!(nested.join("model.safetensors").exists());
        assert!(nested.join("config.json").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn keys_have_pooler_detects_slash_separator() {
        let keys = vec![
            "encoder/layer/0/attention/query/weight".to_string(),
            "pooler/dense/weight".to_string(),
            "pooler/dense/bias".to_string(),
        ];
        assert!(keys_have_pooler(&keys));
    }

    #[test]
    fn keys_have_pooler_detects_dot_separator() {
        let keys = vec![
            "encoder.layer.0.attention.query.weight".to_string(),
            "pooler.dense.weight".to_string(),
        ];
        assert!(keys_have_pooler(&keys));
    }

    #[test]
    fn keys_have_pooler_returns_false_for_encoder_only() {
        let keys = vec![
            "encoder/layer/0/attention/query/weight".to_string(),
            "encoder/layer/0/attention/query/bias".to_string(),
        ];
        assert!(!keys_have_pooler(&keys));
    }

    #[test]
    fn keys_have_pooler_does_not_match_substrings() {
        // A key containing "pooler" mid-string must not false-positive.
        let keys = vec!["encoder/some_pooler_thing/weight".to_string()];
        assert!(!keys_have_pooler(&keys));
    }

    #[test]
    fn build_for_export_dispatches_on_family() {
        // Each family returns Ok(_); pooler-toggle only affects bert/
        // roberta/xlm-roberta/albert.
        for has_pooler in [false, true] {
            let bert = AutoConfig::Bert(crate::models::bert::BertConfig::bert_base_uncased());
            assert!(build_for_export(&bert, has_pooler, Device::CPU).is_ok());
        }
        // DistilBERT is pooler-less; both bools should produce the same shape.
        let distil = AutoConfig::DistilBert(
            crate::models::distilbert::DistilBertConfig::distilbert_base_uncased(),
        );
        let g_a = build_for_export(&distil, false, Device::CPU).unwrap();
        let g_b = build_for_export(&distil, true, Device::CPU).unwrap();
        assert_eq!(g_a.structural_hash(), g_b.structural_hash());
    }
}
