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

use std::path::PathBuf;

use hf_hub::api::sync::{Api, ApiBuilder};

use flodl::{Device, Graph, Result, TensorError};

use crate::models::bert::{
    BertConfig, BertForQuestionAnswering, BertForSequenceClassification,
    BertForTokenClassification, BertModel,
};
use crate::models::roberta::{
    RobertaConfig, RobertaForQuestionAnswering, RobertaForSequenceClassification,
    RobertaForTokenClassification, RobertaModel,
};
use crate::safetensors_io::{
    bert_legacy_key_rename, load_safetensors_into_graph_with_rename_allow_unused,
};
#[cfg(feature = "tokenizer")]
use crate::tokenizer::HfTokenizer;

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
        let (config, weights) = fetch_bert_config_and_weights(repo_id)?;
        let graph = BertModel::on_device(&config, device)?;
        // HF base checkpoints (e.g. `bert-base-uncased`) ship as
        // `BertForPreTraining`, which carries MLM + NSP heads that a
        // bare `BertModel` has no slot for. `load_weights_with_logging`
        // tolerates those and names them on stderr.
        load_weights_with_logging(repo_id, &graph, &weights)?;
        Ok(graph)
    }
}

/// Environment variable `fetch_safetensors` honours when looking for a
/// locally-converted `model.safetensors` before hitting the Hub.
/// Matches the `HF_HOME` the Docker services are configured with
/// (`/workspace/.hf-cache` via docker-compose.yml).
const HF_HOME_ENV: &str = "HF_HOME";

/// Default cache root when `HF_HOME` is not set. Mirrors HF Python's
/// `~/.cache/huggingface/` convention.
fn default_hf_home() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache").join("huggingface")
    } else {
        PathBuf::from("/tmp/huggingface")
    }
}

/// Locally-converted safetensors path for `repo_id`, if one exists.
///
/// flodl-hf writes converted weights to
/// `<HF_HOME>/flodl-converted/<repo_id>/model.safetensors` via
/// `fdl flodl-hf convert <repo_id>`. Checking this path first lets
/// `from_pretrained` transparently use the converted copy without
/// touching the network.
fn flodl_converted_path(repo_id: &str) -> PathBuf {
    let hf_home = std::env::var_os(HF_HOME_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(default_hf_home);
    hf_home
        .join("flodl-converted")
        .join(repo_id)
        .join("model.safetensors")
}

/// Resolve `model.safetensors` for `repo_id`, preferring a locally-
/// converted copy over the Hub.
///
/// Order:
/// 1. `<HF_HOME>/flodl-converted/<repo_id>/model.safetensors` — produced
///    by `fdl flodl-hf convert <repo_id>` for `.bin`-only repos.
/// 2. `api.model(repo_id).get("model.safetensors")` — the normal Hub
///    fetch, goes through hf-hub's own on-disk cache.
///
/// On failure at step 2, the returned error explicitly points the user
/// at `fdl flodl-hf convert <repo_id>` for the common `.bin`-only case.
fn fetch_safetensors(api: &Api, repo_id: &str) -> Result<PathBuf> {
    let converted = flodl_converted_path(repo_id);
    if converted.exists() {
        eprintln!(
            "from_pretrained({repo_id}): using flodl-converted safetensors at {}",
            converted.display(),
        );
        return Ok(converted);
    }
    api.model(repo_id.to_string())
        .get("model.safetensors")
        .map_err(|e| {
            TensorError::new(&format!(
                "hf-hub fetch {repo_id}/model.safetensors: {e}\n\
                 If this repo ships only `pytorch_model.bin`, convert it first:\n  \
                 fdl flodl-hf convert {repo_id}",
            ))
        })
}

/// Pull `config.json` + `model.safetensors` from a Hub repo and return
/// `(config_string, weights_bytes)`. Config parsing is left to the
/// caller so the same fetch path serves every model family
/// (`BertConfig::from_json_str`, `RobertaConfig::from_json_str`, …).
fn fetch_config_str_and_weights(repo_id: &str) -> Result<(String, Vec<u8>)> {
    // `ApiBuilder::from_env()` reads `HF_HOME` for the cache location.
    // `Api::new()` hardcodes `~/.cache/huggingface/hub/` and silently
    // ignores `HF_HOME`, so every run would redownload into the dev
    // container's ephemeral `$HOME`.
    let api = ApiBuilder::from_env()
        .build()
        .map_err(|e| TensorError::new(&format!("hf-hub init: {e}")))?;
    let repo = api.model(repo_id.to_string());

    let config_path = repo.get("config.json").map_err(|e| {
        TensorError::new(&format!("hf-hub fetch {repo_id}/config.json: {e}"))
    })?;
    let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
        TensorError::new(&format!("read {}: {e}", config_path.display()))
    })?;

    let weights_path = fetch_safetensors(&api, repo_id)?;
    let weights = std::fs::read(&weights_path).map_err(|e| {
        TensorError::new(&format!("read {}: {e}", weights_path.display()))
    })?;
    Ok((config_str, weights))
}

/// Convenience wrapper: [`fetch_config_str_and_weights`] + parse as
/// [`BertConfig`]. Keeps the BERT `from_pretrained` call sites tidy.
fn fetch_bert_config_and_weights(repo_id: &str) -> Result<(BertConfig, Vec<u8>)> {
    let (config_str, weights) = fetch_config_str_and_weights(repo_id)?;
    let config = BertConfig::from_json_str(&config_str)?;
    Ok((config, weights))
}

/// Convenience wrapper: [`fetch_config_str_and_weights`] + parse as
/// [`RobertaConfig`].
fn fetch_roberta_config_and_weights(repo_id: &str) -> Result<(RobertaConfig, Vec<u8>)> {
    let (config_str, weights) = fetch_config_str_and_weights(repo_id)?;
    let config = RobertaConfig::from_json_str(&config_str)?;
    Ok((config, weights))
}

/// Best-effort tokenizer download for task-head `from_pretrained` paths.
///
/// `HfTokenizer::from_pretrained` requires the repo to ship a fast-
/// tokenizer `tokenizer.json`. Legacy checkpoints (pre-~2022, many
/// older fine-tunes, hand-uploaded models) only carry the slow-tokenizer
/// triple `tokenizer_config.json` + `vocab.txt` + `special_tokens_map.json`;
/// HF Python rebuilds a fast tokenizer from those on the fly, but the
/// Rust `tokenizers` crate does not. Failing `from_pretrained` over a
/// missing `tokenizer.json` breaks the HF-API parity that AutoModel (no
/// required tokenizer) ships. We log and continue — `predict()` /
/// `answer()` will then error with a clear "attach a tokenizer" message
/// at call time.
#[cfg(feature = "tokenizer")]
fn try_load_tokenizer(repo_id: &str) -> Option<HfTokenizer> {
    match HfTokenizer::from_pretrained(repo_id) {
        Ok(tok) => Some(tok),
        Err(e) => {
            // Common case: legacy repos ship only the slow-tokenizer
            // triple (vocab.txt + tokenizer_config.json + special_tokens_map.json),
            // no `tokenizer.json`. Surface one actionable line without
            // echoing the URL or the verbose HfTokenizer error.
            let terse = if e.to_string().contains("404") {
                "no tokenizer.json on Hub".to_string()
            } else {
                e.to_string()
            };
            eprintln!(
                "from_pretrained({repo_id}): tokenizer not attached ({terse}) \
                 — predict()/answer() need .with_tokenizer()",
            );
            None
        }
    }
}

/// Load safetensors into a graph, logging any discarded checkpoint keys
/// to stderr. Shared by every `from_pretrained` path.
fn load_weights_with_logging(
    repo_id: &str,
    graph: &Graph,
    bytes: &[u8],
) -> Result<()> {
    let unused = load_safetensors_into_graph_with_rename_allow_unused(
        graph, bytes, bert_legacy_key_rename,
    )?;
    if !unused.is_empty() {
        eprintln!(
            "from_pretrained({repo_id}): ignored {} checkpoint key(s) not used by the model:",
            unused.len(),
        );
        for k in unused.iter().take(20) {
            eprintln!("  - {k}");
        }
        if unused.len() > 20 {
            eprintln!("  ... and {} more", unused.len() - 20);
        }
    }
    Ok(())
}

impl BertForSequenceClassification {
    /// Download a fine-tuned `BertForSequenceClassification` checkpoint
    /// from the Hub and return a ready-to-use predictor on CPU.
    ///
    /// The config must carry `num_labels` (or `id2label`) so the head's
    /// output width is known. Popular checkpoints: `nateraw/bert-base-uncased-emotion`
    /// (6 emotions, `.bin`-only — needs `fdl flodl-hf convert` first),
    /// `nlptown/bert-base-multilingual-uncased-sentiment` (5-star rating,
    /// safetensors on main), `unitary/toxic-bert` (6-label toxicity).
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    /// Device-aware variant of [`from_pretrained`](Self::from_pretrained).
    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_bert_config_and_weights(repo_id)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl BertForTokenClassification {
    /// Download a fine-tuned `BertForTokenClassification` checkpoint
    /// (NER, POS tagging, …) from the Hub. Popular checkpoints:
    /// `dslim/bert-base-NER`,
    /// `dbmdz/bert-large-cased-finetuned-conll03-english`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_bert_config_and_weights(repo_id)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl BertForQuestionAnswering {
    /// Download a fine-tuned `BertForQuestionAnswering` checkpoint
    /// (SQuAD, etc.) from the Hub. Popular checkpoints:
    /// `csarron/bert-base-uncased-squad-v1`,
    /// `bert-large-uncased-whole-word-masking-finetuned-squad`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_bert_config_and_weights(repo_id)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

// ── RoBERTa from_pretrained ──────────────────────────────────────────────
//
// Each RoBERTa head mirrors its BERT counterpart above. The only
// family-specific bits are the config type (`RobertaConfig`) and the
// graph the weights load into. The tokenizer, safetensors loader, and
// HF legacy key rewrite are model-agnostic — `bert_legacy_key_rename`
// only rewrites `LayerNorm.gamma`/`LayerNorm.beta` suffixes, which is a
// no-op on modern RoBERTa checkpoints and harmless if it isn't.

impl RobertaModel {
    /// Download a pretrained RoBERTa checkpoint from the HuggingFace
    /// Hub and return a fully-initialised [`Graph`] on CPU.
    ///
    /// Returns a pooler-free backbone (graph emits
    /// `last_hidden_state` of shape `[B, S, hidden]`). RoBERTa
    /// pretraining drops BERT's NSP objective, so most checkpoints
    /// (including `roberta-base`) don't carry pooler weights; HF
    /// Python silently random-initialises them on load, which
    /// produces non-reproducible `pooler_output`. flodl-hf takes the
    /// opposite default: no pooler, strict weight load. Reach for
    /// [`RobertaModel::on_device`] if a specific checkpoint is known
    /// to ship its own pooler.
    ///
    /// `repo_id` is the HF-style identifier, e.g. `"roberta-base"` or
    /// `"FacebookAI/roberta-large"`. HF base checkpoints ship as
    /// `RobertaForMaskedLM` with an `lm_head` that a bare
    /// `RobertaModel` has no slot for; `load_weights_with_logging`
    /// tolerates those and names them on stderr.
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    /// Device-aware variant of [`from_pretrained`](Self::from_pretrained).
    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_roberta_config_and_weights(repo_id)?;
        let graph = RobertaModel::on_device_without_pooler(&config, device)?;
        load_weights_with_logging(repo_id, &graph, &weights)?;
        Ok(graph)
    }
}

impl RobertaForSequenceClassification {
    /// Download a fine-tuned `RobertaForSequenceClassification`
    /// checkpoint from the Hub and return a ready-to-use predictor on
    /// CPU.
    ///
    /// Popular checkpoints:
    /// `cardiffnlp/twitter-roberta-base-sentiment-latest` (3-label
    /// sentiment), `roberta-large-mnli` (3-label NLI),
    /// `SamLowe/roberta-base-go_emotions` (28 emotions).
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_roberta_config_and_weights(repo_id)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl RobertaForTokenClassification {
    /// Download a fine-tuned `RobertaForTokenClassification`
    /// checkpoint (NER, POS tagging, …) from the Hub. Popular
    /// checkpoints: `Jean-Baptiste/roberta-large-ner-english`,
    /// `obi/deid_roberta_i2b2`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_roberta_config_and_weights(repo_id)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl RobertaForQuestionAnswering {
    /// Download a fine-tuned `RobertaForQuestionAnswering` checkpoint
    /// (SQuAD, etc.) from the Hub. Popular checkpoints:
    /// `deepset/roberta-base-squad2`, `csarron/roberta-base-squad-v1`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_roberta_config_and_weights(repo_id)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

#[cfg(feature = "tokenizer")]
impl HfTokenizer {
    /// Download `tokenizer.json` from a HuggingFace Hub repo and wrap it.
    ///
    /// Uses the same `hf_hub` cache as [`BertModel::from_pretrained`], so
    /// a model already pulled from a given repo won't re-download the
    /// tokenizer either (and vice versa).
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        let api = ApiBuilder::from_env()
            .build()
            .map_err(|e| TensorError::new(&format!("hf-hub init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let path = repo.get("tokenizer.json").map_err(|e| {
            TensorError::new(&format!("hf-hub fetch {repo_id}/tokenizer.json: {e}"))
        })?;
        Self::from_file(&path)
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
