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

use flodl::{Graph, Result, TensorError};

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
}
