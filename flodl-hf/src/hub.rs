//! HuggingFace Hub download and local cache integration.
//!
//! Thin façade over the [`hf_hub`] crate: pulls `config.json` and
//! `model.safetensors` from a Hub repo, parses the config, builds the
//! matching flodl model, and loads the weights. Downloads go through
//! `hf_hub`'s cache, so a second call on the same repo reuses the local
//! copy.
//!
//! Entry points live as inherent methods on the model types
//! ([`BertModel::from_pretrained`]) so user code reads the same as HF
//! Python's `BertModel.from_pretrained(...)`.

use hf_hub::api::sync::Api;

use flodl::{Device, Graph, Result, TensorError};

use crate::models::bert::{BertConfig, BertModel};
use crate::safetensors_io::{
    bert_legacy_key_rename, load_safetensors_into_graph_with_rename_allow_unused,
};

impl BertModel {
    /// Download a pretrained BERT checkpoint from the HuggingFace Hub and
    /// return a fully-initialised [`Graph`] on CPU.
    ///
    /// Convenience wrapper over [`BertModel::from_pretrained_on_device`]
    /// with `Device::CPU`.
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    /// Download a pretrained BERT checkpoint from the HuggingFace Hub and
    /// return a fully-initialised [`Graph`] on `device`.
    ///
    /// Pulls `config.json` and `model.safetensors` from `repo_id` via
    /// `hf_hub` (using its on-disk cache), parses the config, builds a
    /// matching graph, and loads the safetensors weights.
    ///
    /// `repo_id` is the HF-style identifier, e.g. `"bert-base-uncased"`
    /// or `"google-bert/bert-base-multilingual-cased"`.
    ///
    /// Errors on: hub API init failure, network / HTTP failure,
    /// config parse failure, shape/key mismatch against the built graph,
    /// and any I/O error reading the cached safetensors file. Nothing
    /// partial is returned — the graph is either fully loaded or the
    /// call errors out.
    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let api = Api::new()
            .map_err(|e| TensorError::new(&format!("hf-hub init: {e}")))?;
        let repo = api.model(repo_id.to_string());

        let config_path = repo.get("config.json").map_err(|e| {
            TensorError::new(&format!("hf-hub fetch {repo_id}/config.json: {e}"))
        })?;
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            TensorError::new(&format!("read {}: {e}", config_path.display()))
        })?;
        let config = BertConfig::from_json_str(&config_str)?;

        let graph = BertModel::on_device(&config, device)?;

        let weights_path = repo.get("model.safetensors").map_err(|e| {
            TensorError::new(&format!(
                "hf-hub fetch {repo_id}/model.safetensors: {e}",
            ))
        })?;
        let weights_bytes = std::fs::read(&weights_path).map_err(|e| {
            TensorError::new(&format!("read {}: {e}", weights_path.display()))
        })?;
        // HF base models (e.g. `bert-base-uncased`) live inside checkpoints
        // that also carry pretraining / task heads. Accept those as
        // "unused" here and log them so the user can tell what was
        // discarded vs silently dropped.
        let unused = load_safetensors_into_graph_with_rename_allow_unused(
            &graph, &weights_bytes, bert_legacy_key_rename,
        )?;
        if !unused.is_empty() {
            eprintln!(
                "from_pretrained({repo_id}): ignored {} checkpoint key(s) not used by BertModel \
                 (task heads etc.):",
                unused.len(),
            );
            for k in unused.iter().take(20) {
                eprintln!("  - {k}");
            }
            if unused.len() > 20 {
                eprintln!("  ... and {} more", unused.len() - 20);
            }
        }

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live-network integration test: pulls `bert-base-uncased` from the
    /// HuggingFace Hub, builds the graph, loads the weights, and runs a
    /// forward pass. `#[ignore]` by default — run manually with
    /// `fdl test -- --ignored bert_from_pretrained_live` when the host
    /// has network access (and is happy to cache ~440MB of weights under
    /// `~/.cache/huggingface/`).
    #[test]
    #[ignore = "network + ~440MB cache write"]
    fn bert_from_pretrained_live() {
        use flodl::nn::Module;
        use flodl::{DType, Tensor, TensorOptions, Variable};
        use crate::models::bert::build_extended_attention_mask;

        let graph = BertModel::from_pretrained("bert-base-uncased").unwrap();
        graph.eval();

        // Tiny forward pass to prove the loaded graph works end-to-end.
        let dev = Device::CPU;
        let batch = 1;
        let seq = 4;
        let input_ids = Variable::new(
            Tensor::from_i64(&[101, 7592, 2088, 102], &[batch, seq], dev).unwrap(),
            false,
        );
        let position_ids = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let token_type_ids = Variable::new(
            Tensor::from_i64(&[0, 0, 0, 0], &[batch, seq], dev).unwrap(),
            false,
        );
        let mask_flat = Tensor::ones(&[batch, seq], TensorOptions {
            dtype: DType::Float32, device: dev,
        }).unwrap();
        let attention_mask = Variable::new(
            build_extended_attention_mask(&mask_flat).unwrap(),
            false,
        );

        let out = graph
            .forward_multi(&[input_ids, position_ids, token_type_ids, attention_mask])
            .unwrap();
        assert_eq!(out.shape(), vec![batch, 768]);
    }
}
