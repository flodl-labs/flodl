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

use crate::models::auto::{
    AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
};
use crate::models::albert::{
    AlbertConfig, AlbertForMaskedLM, AlbertForQuestionAnswering,
    AlbertForSequenceClassification, AlbertForTokenClassification, AlbertModel,
};
use crate::models::bert::{
    BertConfig, BertForMaskedLM, BertForQuestionAnswering, BertForSequenceClassification,
    BertForTokenClassification, BertModel,
};
use crate::models::deberta_v2::{
    DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForQuestionAnswering,
    DebertaV2ForSequenceClassification, DebertaV2ForTokenClassification, DebertaV2Model,
};
use crate::models::distilbert::{
    DistilBertConfig, DistilBertForMaskedLM, DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification, DistilBertForTokenClassification, DistilBertModel,
};
use crate::models::roberta::{
    RobertaConfig, RobertaForMaskedLM, RobertaForQuestionAnswering,
    RobertaForSequenceClassification, RobertaForTokenClassification, RobertaModel,
};
use crate::models::xlm_roberta::{
    XlmRobertaConfig, XlmRobertaForMaskedLM, XlmRobertaForQuestionAnswering,
    XlmRobertaForSequenceClassification, XlmRobertaForTokenClassification, XlmRobertaModel,
};
use crate::safetensors_io::{
    bert_legacy_key_rename, load_safetensors_into_graph_with_rename_allow_unused,
};
use safetensors::SafeTensors;
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
    /// Picks `on_device` (with pooler) when the checkpoint ships pooler
    /// weights and `on_device_without_pooler` when it doesn't, so an
    /// encoder-only BERT checkpoint loads strict without missing-key
    /// errors and a pooler-bearing one keeps its `pooler_output` slot.
    /// Reach for [`BertModel::on_device`] /
    /// [`BertModel::on_device_without_pooler`] directly when the call
    /// site needs a guaranteed shape regardless of what the Hub repo
    /// happens to ship.
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
        let graph = if weights_have_pooler(&weights)? {
            BertModel::on_device(&config, device)?
        } else {
            BertModel::on_device_without_pooler(&config, device)?
        };
        // HF base checkpoints (e.g. `bert-base-uncased`) ship as
        // `BertForPreTraining`, which carries MLM + NSP heads that a
        // bare `BertModel` has no slot for. `load_weights_with_logging`
        // tolerates those and names them on stderr.
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config.with_architectures("BertModel").to_json_str());
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

/// Fetch a Hub repo's `config.json` as a string, going through
/// hf-hub's on-disk cache. Extracted from
/// [`fetch_config_str_and_weights`] so config-only callers (e.g.
/// [`AutoConfig::from_pretrained`]) don't pay the safetensors read.
fn fetch_config_str(repo_id: &str) -> Result<String> {
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
    std::fs::read_to_string(&config_path).map_err(|e| {
        TensorError::new(&format!("read {}: {e}", config_path.display()))
    })
}

/// Pull `config.json` + `model.safetensors` from a Hub repo and return
/// `(config_string, weights_bytes)`. Config parsing is left to the
/// caller so the same fetch path serves every model family
/// (`BertConfig::from_json_str`, `RobertaConfig::from_json_str`, …).
fn fetch_config_str_and_weights(repo_id: &str) -> Result<(String, Vec<u8>)> {
    let config_str = fetch_config_str(repo_id)?;

    let api = ApiBuilder::from_env()
        .build()
        .map_err(|e| TensorError::new(&format!("hf-hub init: {e}")))?;
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

/// Convenience wrapper: [`fetch_config_str_and_weights`] + parse as
/// [`DistilBertConfig`].
fn fetch_distilbert_config_and_weights(repo_id: &str) -> Result<(DistilBertConfig, Vec<u8>)> {
    let (config_str, weights) = fetch_config_str_and_weights(repo_id)?;
    let config = DistilBertConfig::from_json_str(&config_str)?;
    Ok((config, weights))
}

/// Convenience wrapper: [`fetch_config_str_and_weights`] + parse as
/// [`XlmRobertaConfig`].
fn fetch_xlm_roberta_config_and_weights(repo_id: &str) -> Result<(XlmRobertaConfig, Vec<u8>)> {
    let (config_str, weights) = fetch_config_str_and_weights(repo_id)?;
    let config = XlmRobertaConfig::from_json_str(&config_str)?;
    Ok((config, weights))
}

/// Convenience wrapper: [`fetch_config_str_and_weights`] + parse as
/// [`AlbertConfig`].
fn fetch_albert_config_and_weights(repo_id: &str) -> Result<(AlbertConfig, Vec<u8>)> {
    let (config_str, weights) = fetch_config_str_and_weights(repo_id)?;
    let config = AlbertConfig::from_json_str(&config_str)?;
    Ok((config, weights))
}

/// Convenience wrapper: [`fetch_config_str_and_weights`] + parse as
/// [`DebertaV2Config`].
fn fetch_deberta_v2_config_and_weights(repo_id: &str) -> Result<(DebertaV2Config, Vec<u8>)> {
    let (config_str, weights) = fetch_config_str_and_weights(repo_id)?;
    let config = DebertaV2Config::from_json_str(&config_str)?;
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
/// Inspect a safetensors blob and report whether it carries the pooler
/// `Linear` weights for any of the pooler-bearing families (BERT,
/// RoBERTa, XLM-R, ALBERT).
///
/// Matches both shapes: BERT-style `pooler.dense.{weight,bias}` (a
/// `BertPooler { dense: Linear }` wrapper) and ALBERT-style flat
/// `pooler.{weight,bias}` (HF's `AlbertModel.pooler` is a bare
/// `nn.Linear`). Pooler-less families (DistilBERT, DeBERTa-v2) never
/// match either shape, so the helper is a safe no-op for them.
///
/// Used by every pooler-bearing family's `from_pretrained_on_device`
/// (and `AutoModel::from_pretrained_for_export_on_device`) to pick
/// `on_device` vs `on_device_without_pooler` based on what the
/// checkpoint actually ships, rather than baking a per-family default
/// that's always wrong for some Hub repos (e.g. `roberta-base` has no
/// pooler; `bert-base-uncased` does).
fn weights_have_pooler(weights: &[u8]) -> Result<bool> {
    let st = SafeTensors::deserialize(weights)
        .map_err(|e| TensorError::new(&format!("safetensors parse error: {e}")))?;
    Ok(st.names().iter().any(|n| {
        n.ends_with("pooler.dense.weight")
            || n.ends_with("pooler.dense.bias")
            || n.ends_with("pooler.weight")
            || n.ends_with("pooler.bias")
    }))
}

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
        head.graph().set_source_config(
            config.with_architectures("BertForSequenceClassification").to_json_str(),
        );
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
        head.graph().set_source_config(
            config.with_architectures("BertForTokenClassification").to_json_str(),
        );
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
        head.graph().set_source_config(
            config.with_architectures("BertForQuestionAnswering").to_json_str(),
        );
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl BertForMaskedLM {
    /// Download a BERT MLM checkpoint (`bert-base-uncased`,
    /// `bert-base-cased`, any `*-mlm` fine-tune) from the Hub.
    ///
    /// The decoder weight is tied to the word-embedding table, so
    /// checkpoints that redundantly save `cls.predictions.decoder.weight`
    /// alongside `bert.embeddings.word_embeddings.weight` (HF's
    /// historical save format) load cleanly — the decoder key is
    /// ignored by `load_safetensors_into_graph_with_rename_allow_unused`,
    /// and the embedding key populates the single tied Parameter.
    /// Checkpoints that skip the redundant decoder key (modern HF saves)
    /// also load cleanly.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_bert_config_and_weights(repo_id)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config.with_architectures("BertForMaskedLM").to_json_str(),
        );
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
    /// Picks `on_device` (with pooler) when the checkpoint ships pooler
    /// weights and `on_device_without_pooler` when it doesn't, so the
    /// graph shape matches the Hub repo regardless of whether the
    /// checkpoint kept BERT's pooler. Most RoBERTa checkpoints
    /// (including `roberta-base`) drop the pooler with the NSP
    /// objective, but some downstream-tuned variants keep it. HF
    /// Python silently random-initialises a missing pooler on load,
    /// producing non-reproducible `pooler_output`; flodl-hf instead
    /// builds a pooler-free backbone in that case so the load stays
    /// strict and reproducible. Reach for [`RobertaModel::on_device`]
    /// directly when a graph slot for the pooler is needed regardless
    /// of what the checkpoint ships.
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
        let graph = if weights_have_pooler(&weights)? {
            RobertaModel::on_device(&config, device)?
        } else {
            RobertaModel::on_device_without_pooler(&config, device)?
        };
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config.with_architectures("RobertaModel").to_json_str());
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
        head.graph().set_source_config(config.with_architectures("RobertaForSequenceClassification").to_json_str());
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
        head.graph().set_source_config(config.with_architectures("RobertaForTokenClassification").to_json_str());
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
        head.graph().set_source_config(config.with_architectures("RobertaForQuestionAnswering").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl RobertaForMaskedLM {
    /// Download a RoBERTa MLM checkpoint (`roberta-base`,
    /// `roberta-large`, any `*-mlm` domain-adaptation fine-tune) from
    /// the Hub.
    ///
    /// The decoder weight is tied to
    /// `roberta.embeddings.word_embeddings.weight`; checkpoints that
    /// redundantly save `lm_head.decoder.weight` alongside it, or ship
    /// an extra `lm_head.bias` tied-to-decoder-bias, load cleanly —
    /// `load_safetensors_into_graph_with_rename_allow_unused` silently
    /// ignores keys absent from the graph's deduplicated
    /// `named_parameters()`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_roberta_config_and_weights(repo_id)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("RobertaForMaskedLM").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

// ── DistilBERT from_pretrained ───────────────────────────────────────────
//
// Each DistilBERT head mirrors its BERT / RoBERTa counterparts above.
// Family-specific bits: the config type (`DistilBertConfig`) and the
// fact that the backbone graph never ships a pooler — `DistilBertModel`
// is pooler-free by construction, matching every public DistilBERT
// checkpoint.

impl DistilBertModel {
    /// Download a pretrained DistilBERT checkpoint from the HuggingFace
    /// Hub and return a fully-initialised pooler-free [`Graph`] on CPU.
    ///
    /// `repo_id` examples: `"distilbert/distilbert-base-uncased"`,
    /// `"distilbert/distilbert-base-cased"`. HF base checkpoints ship
    /// as `DistilBertForMaskedLM` with an extra MLM head
    /// (`vocab_transform.*`, `vocab_layer_norm.*`, `vocab_projector.*`)
    /// that a bare `DistilBertModel` has no slot for;
    /// `load_weights_with_logging` tolerates those and names them on
    /// stderr.
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    /// Device-aware variant of [`from_pretrained`](Self::from_pretrained).
    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_distilbert_config_and_weights(repo_id)?;
        let graph = DistilBertModel::on_device(&config, device)?;
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config.with_architectures("DistilBertModel").to_json_str());
        Ok(graph)
    }
}

impl DistilBertForSequenceClassification {
    /// Download a fine-tuned `DistilBertForSequenceClassification`
    /// checkpoint from the Hub and return a ready-to-use predictor on
    /// CPU.
    ///
    /// Popular checkpoints:
    /// `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
    /// (3-label multilingual sentiment),
    /// `distilbert-base-uncased-finetuned-sst-2-english` (2-label
    /// sentiment — older, lacks `tokenizer.json`).
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_distilbert_config_and_weights(repo_id)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("DistilBertForSequenceClassification").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl DistilBertForTokenClassification {
    /// Download a fine-tuned `DistilBertForTokenClassification`
    /// checkpoint (NER, POS tagging, …) from the Hub. Popular
    /// checkpoint: `dslim/distilbert-NER` (PER/ORG/LOC/MISC, 9 labels
    /// including `O`).
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_distilbert_config_and_weights(repo_id)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("DistilBertForTokenClassification").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl DistilBertForQuestionAnswering {
    /// Download a fine-tuned `DistilBertForQuestionAnswering`
    /// checkpoint (SQuAD, etc.) from the Hub. Canonical:
    /// `distilbert/distilbert-base-cased-distilled-squad`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_distilbert_config_and_weights(repo_id)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("DistilBertForQuestionAnswering").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl DistilBertForMaskedLM {
    /// Download a DistilBERT MLM checkpoint (`distilbert-base-uncased`,
    /// `distilbert-base-cased`, any `*-mlm` domain-adaptation fine-tune)
    /// from the Hub.
    ///
    /// The `vocab_projector` weight is tied to
    /// `distilbert.embeddings.word_embeddings.weight`; checkpoints that
    /// redundantly save `vocab_projector.weight` load cleanly — the
    /// loader silently ignores keys absent from the graph's deduplicated
    /// `named_parameters()`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_distilbert_config_and_weights(repo_id)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("DistilBertForMaskedLM").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

// ── XLM-RoBERTa from_pretrained ──────────────────────────────────────────
//
// Each head mirrors the RoBERTa counterpart — XLM-R's state_dict uses
// the `roberta.*` prefix verbatim, so the shared safetensors loader
// path loads cleanly with no key rewriting beyond the standard legacy
// LayerNorm.gamma/beta rename. The family-specific bits are just the
// config type (`XlmRobertaConfig`) and the head constructors routing
// through the XLM-RoBERTa public surface.

impl XlmRobertaModel {
    /// Download a pretrained XLM-RoBERTa checkpoint from the HuggingFace
    /// Hub and return a fully-initialised [`Graph`] on CPU.
    ///
    /// Picks `on_device` (with pooler) when the checkpoint ships pooler
    /// weights and `on_device_without_pooler` when it doesn't, matching
    /// whatever the Hub repo carries. `FacebookAI/xlm-roberta-base`
    /// keeps the pooler; many encoder-only variants (e.g.
    /// `xlm-roberta-base`) don't. HF base checkpoints ship as
    /// `XLMRobertaForMaskedLM` and the `lm_head.*` keys are tolerated
    /// (logged, ignored) by `load_weights_with_logging`.
    ///
    /// `repo_id` examples: `"xlm-roberta-base"`, `"xlm-roberta-large"`,
    /// `"FacebookAI/xlm-roberta-base"`.
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    /// Device-aware variant of [`from_pretrained`](Self::from_pretrained).
    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_xlm_roberta_config_and_weights(repo_id)?;
        let graph = if weights_have_pooler(&weights)? {
            XlmRobertaModel::on_device(&config, device)?
        } else {
            XlmRobertaModel::on_device_without_pooler(&config, device)?
        };
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config.with_architectures("XLMRobertaModel").to_json_str());
        Ok(graph)
    }
}

impl XlmRobertaForSequenceClassification {
    /// Download a fine-tuned `XLMRobertaForSequenceClassification`
    /// checkpoint from the Hub. Popular checkpoints:
    /// `cardiffnlp/twitter-xlm-roberta-base-sentiment` (3-label
    /// multilingual sentiment), `joeddav/xlm-roberta-large-xnli`
    /// (zero-shot NLI), `papluca/xlm-roberta-base-language-detection`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_xlm_roberta_config_and_weights(repo_id)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("XLMRobertaForSequenceClassification").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl XlmRobertaForTokenClassification {
    /// Download a fine-tuned `XLMRobertaForTokenClassification`
    /// checkpoint (NER, POS tagging, …) from the Hub. Popular
    /// checkpoints: `Davlan/xlm-roberta-base-ner-hrl` (multilingual
    /// NER), `Davlan/xlm-roberta-base-finetuned-conll03-english`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_xlm_roberta_config_and_weights(repo_id)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("XLMRobertaForTokenClassification").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl XlmRobertaForQuestionAnswering {
    /// Download a fine-tuned `XLMRobertaForQuestionAnswering`
    /// checkpoint (multilingual SQuAD, …) from the Hub. Popular
    /// checkpoints: `deepset/xlm-roberta-base-squad2`,
    /// `deepset/xlm-roberta-large-squad2`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_xlm_roberta_config_and_weights(repo_id)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("XLMRobertaForQuestionAnswering").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl XlmRobertaForMaskedLM {
    /// Download an XLM-RoBERTa MLM checkpoint (`xlm-roberta-base`,
    /// `xlm-roberta-large`, any `*-mlm` domain-adaptation fine-tune)
    /// from the Hub.
    ///
    /// The decoder weight is tied to
    /// `roberta.embeddings.word_embeddings.weight`; checkpoints that
    /// redundantly save `lm_head.decoder.weight`, or ship an extra
    /// `lm_head.bias` tied-to-decoder-bias, load cleanly — the loader
    /// silently ignores keys absent from the graph's deduplicated
    /// `named_parameters()`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_xlm_roberta_config_and_weights(repo_id)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("XLMRobertaForMaskedLM").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

// ── ALBERT from_pretrained ───────────────────────────────────────────────
//
// Each head mirrors the BERT counterpart on the load side — the
// family-specific bits (factorised embeddings, cross-layer sharing)
// are all absorbed by the backbone builder. HF ALBERT base checkpoints
// ship as `AlbertForMaskedLM` with `predictions.*` keys that a bare
// `AlbertModel` has no slot for; `load_weights_with_logging`
// tolerates and names those on stderr.

impl AlbertModel {
    /// Download a pretrained ALBERT checkpoint from the HuggingFace
    /// Hub and return a fully-initialised [`Graph`] on CPU.
    ///
    /// Picks `on_device` (with pooler) when the checkpoint ships pooler
    /// weights and `on_device_without_pooler` when it doesn't.
    /// `albert-base-v2` ships its pooler as a flat
    /// `albert.pooler.{weight,bias}` (HF's `AlbertModel.pooler` is a
    /// bare `nn.Linear`, not a `BertPooler`-style `.dense` wrapper),
    /// and the dynamic detection picks both shapes.
    ///
    /// `repo_id` examples: `"albert-base-v2"`, `"albert-large-v2"`,
    /// `"albert/albert-base-v2"`.
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_albert_config_and_weights(repo_id)?;
        let graph = if weights_have_pooler(&weights)? {
            AlbertModel::on_device(&config, device)?
        } else {
            AlbertModel::on_device_without_pooler(&config, device)?
        };
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config.with_architectures("AlbertModel").to_json_str());
        Ok(graph)
    }
}

impl AlbertForSequenceClassification {
    /// Download a fine-tuned `AlbertForSequenceClassification`
    /// checkpoint from the Hub. Popular checkpoints:
    /// `textattack/albert-base-v2-SST-2` (binary sentiment),
    /// `textattack/albert-base-v2-MRPC`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_albert_config_and_weights(repo_id)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("AlbertForSequenceClassification").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl AlbertForTokenClassification {
    /// Download a fine-tuned `AlbertForTokenClassification` checkpoint
    /// (NER, POS tagging, …) from the Hub.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_albert_config_and_weights(repo_id)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("AlbertForTokenClassification").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl AlbertForQuestionAnswering {
    /// Download a fine-tuned `AlbertForQuestionAnswering` checkpoint
    /// (SQuAD, …) from the Hub. Popular checkpoint:
    /// `twmkn9/albert-base-v2-squad2`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_albert_config_and_weights(repo_id)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("AlbertForQuestionAnswering").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl AlbertForMaskedLM {
    /// Download an ALBERT MLM checkpoint (`albert-base-v2`,
    /// `albert-large-v2`, any `*-mlm` domain-adaptation fine-tune)
    /// from the Hub.
    ///
    /// The decoder weight is tied to
    /// `albert.embeddings.word_embeddings.weight`; HF's historical
    /// save format also emits `predictions.bias` (tied to
    /// `decoder.bias`) as a top-level key. Both redundant keys are
    /// silently ignored by
    /// `load_safetensors_into_graph_with_rename_allow_unused`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_albert_config_and_weights(repo_id)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("AlbertForMaskedLM").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

// ── DeBERTa-v2 from_pretrained ───────────────────────────────────────────
//
// DeBERTa-v3 checkpoints ship under the `deberta-v2` architecture name
// in HuggingFace transformers (the v3 distinction is a config knob, not
// a separate model class). HF base checkpoints save as
// `DebertaV2ForMaskedLM`, so the `lm_predictions.*` keys a bare
// `DebertaV2Model` has no slot for are tolerated by
// `load_weights_with_logging` and named on stderr.

impl DebertaV2Model {
    /// Download a pretrained DeBERTa-v2/v3 checkpoint from the
    /// HuggingFace Hub and return a fully-initialised [`Graph`] on
    /// CPU.
    ///
    /// `repo_id` examples: `"microsoft/deberta-v3-base"`,
    /// `"microsoft/deberta-v3-large"`, `"microsoft/deberta-v3-small"`.
    /// v1 DeBERTa checkpoints are rejected at config-parse time (see
    /// [`DebertaV2Config::from_json_str`]).
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_deberta_v2_config_and_weights(repo_id)?;
        let graph = DebertaV2Model::on_device(&config, device)?;
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config.with_architectures("DebertaV2Model").to_json_str());
        Ok(graph)
    }
}

impl DebertaV2ForSequenceClassification {
    /// Download a fine-tuned `DebertaV2ForSequenceClassification`
    /// checkpoint from the Hub. Popular checkpoints:
    /// `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (NLI),
    /// `cross-encoder/nli-deberta-v3-base`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_deberta_v2_config_and_weights(repo_id)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("DebertaV2ForSequenceClassification").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl DebertaV2ForTokenClassification {
    /// Download a fine-tuned `DebertaV2ForTokenClassification`
    /// checkpoint (NER, POS tagging, …) from the Hub.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_deberta_v2_config_and_weights(repo_id)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("DebertaV2ForTokenClassification").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl DebertaV2ForQuestionAnswering {
    /// Download a fine-tuned `DebertaV2ForQuestionAnswering` checkpoint
    /// (SQuAD, …) from the Hub.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_deberta_v2_config_and_weights(repo_id)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("DebertaV2ForQuestionAnswering").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl DebertaV2ForMaskedLM {
    /// Download a DeBERTa-v2/v3 MLM checkpoint from the Hub. Base
    /// checkpoints like `microsoft/deberta-v3-base` ship as the MLM
    /// variant, so this is the natural starting point for domain
    /// adaptation and continued pretraining.
    ///
    /// The decoder weight is tied to
    /// `deberta.embeddings.word_embeddings.weight`; the separate
    /// `lm_predictions.lm_head.bias` is a fresh `[vocab_size]`
    /// parameter.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_deberta_v2_config_and_weights(repo_id)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(config.with_architectures("DebertaV2ForMaskedLM").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

// ── AutoModel from_pretrained ────────────────────────────────────────────
//
// Model-type dispatch: pull `config.json`, read `model_type`, then route
// to the matching family. The backbone path ([`AutoModel`]) returns a
// [`Graph`] whose shape is always `[batch, seq_len, hidden]` — BERT is
// routed through [`BertModel::on_device_without_pooler`] to keep that
// invariant uniform across families. The task-head paths return the
// corresponding [`AutoModelFor*`] enum, dispatching the tokenizer
// attach + weight load through each concrete family's loader.

/// Fetch a `config.json` string and safetensors blob, then parse the
/// config through [`AutoConfig::from_json_str`]. Shared by every
/// `AutoModelFor*::from_pretrained` path so model-type dispatch
/// happens exactly once per call.
fn fetch_auto_config_and_weights(repo_id: &str) -> Result<(AutoConfig, Vec<u8>)> {
    let (config_str, weights) = fetch_config_str_and_weights(repo_id)?;
    let config = AutoConfig::from_json_str(&config_str)?;
    Ok((config, weights))
}

impl AutoConfig {
    /// Fetch `config.json` for `repo_id` from the HuggingFace Hub and
    /// parse it via [`AutoConfig::from_json_str`], dispatching on
    /// `model_type` to the matching family.
    ///
    /// Useful when a caller needs the parsed config independently of
    /// the weights — e.g. [`crate::export::export_hf_dir`] needs it to
    /// emit the output dir's `config.json`. hf-hub's on-disk cache
    /// means repeated calls (including alongside
    /// [`AutoModel::from_pretrained`]) don't re-download.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        let config_str = fetch_config_str(repo_id)?;
        AutoConfig::from_json_str(&config_str)
    }
}

impl AutoModel {
    /// Download a pretrained encoder from the HuggingFace Hub, auto-
    /// detecting the family (BERT / RoBERTa / DistilBERT) from the
    /// repo's `config.json`.
    ///
    /// The returned [`Graph`] emits `last_hidden_state` of shape
    /// `[batch, seq_len, hidden]` across all three families — BERT is
    /// routed through [`BertModel::on_device_without_pooler`] so its
    /// output stays consistent with RoBERTa and DistilBERT. If the
    /// BERT pooler output is specifically required, call
    /// [`BertModel::from_pretrained`] directly.
    ///
    /// `repo_id` examples:
    /// - `"bert-base-uncased"` (BERT)
    /// - `"roberta-base"` (RoBERTa)
    /// - `"distilbert/distilbert-base-uncased"` (DistilBERT)
    ///
    /// The returned graph's `forward_multi` input count differs by
    /// family (BERT: 4, RoBERTa: 3, DistilBERT: 2); see the
    /// [`auto`](crate::models::auto) module documentation.
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    /// Device-aware variant of [`from_pretrained`](Self::from_pretrained).
    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_auto_config_and_weights(repo_id)?;
        // Capture the source config before consuming it in the match,
        // and normalise `architectures` per family to the base class
        // name actually being built (e.g. `bert-base-uncased` ships
        // `architectures: ["BertForPreTraining"]` but this loader
        // builds a bare `BertModel` with the head dropped). Without
        // normalisation, a subsequent `save_checkpoint` sidecar would
        // carry an arch name that `classify_architecture` rejects on
        // `--checkpoint` re-export.
        let (graph, config_json) = match config {
            AutoConfig::Bert(c) => (
                BertModel::on_device_without_pooler(&c, device)?,
                c.with_architectures("BertModel").to_json_str(),
            ),
            AutoConfig::Roberta(c) => (
                RobertaModel::on_device_without_pooler(&c, device)?,
                c.with_architectures("RobertaModel").to_json_str(),
            ),
            AutoConfig::DistilBert(c) => (
                DistilBertModel::on_device(&c, device)?,
                c.with_architectures("DistilBertModel").to_json_str(),
            ),
            AutoConfig::XlmRoberta(c) => (
                XlmRobertaModel::on_device_without_pooler(&c, device)?,
                c.with_architectures("XLMRobertaModel").to_json_str(),
            ),
            AutoConfig::Albert(c) => (
                AlbertModel::on_device_without_pooler(&c, device)?,
                c.with_architectures("AlbertModel").to_json_str(),
            ),
            AutoConfig::DebertaV2(c) => (
                DebertaV2Model::on_device(&c, device)?,
                c.with_architectures("DebertaV2Model").to_json_str(),
            ),
        };
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config_json);
        Ok(graph)
    }

    /// Download a pretrained model preserving the **full HF base layout**,
    /// including the pooler where the family defines one.
    ///
    /// Use this for round-trip-to-HF export workflows: HF's
    /// `AutoModel.from_pretrained(<exported_dir>)` expects every parameter
    /// the original Hub checkpoint shipped, and silently re-initializes
    /// any it can't find. The default
    /// [`from_pretrained`](Self::from_pretrained) drops family-specific
    /// pooler nodes for cross-family `last_hidden_state` consistency, so
    /// exports built from it are missing pooler weights and fail HF-side
    /// fidelity checks.
    ///
    /// Pooler-bearing families (BERT, RoBERTa, XLM-R, ALBERT) auto-pick
    /// `on_device` vs `on_device_without_pooler` by inspecting the
    /// checkpoint: with-pooler when pooler weights ship, without-pooler
    /// when the Hub repo is encoder-only (e.g. `roberta-base`,
    /// `FacebookAI/xlm-roberta-base`). Pooler-less families (DistilBERT,
    /// DeBERTa-v2) behave identically to
    /// [`from_pretrained`](Self::from_pretrained).
    ///
    /// **Auto-detect head from `architectures[0]`** (no `--head` flag).
    /// The upstream config's `architectures[0]` suffix decides which
    /// head this loader builds:
    ///
    /// | suffix                          | builder                                |
    /// |---------------------------------|-----------------------------------------|
    /// | `*ForSequenceClassification`    | `AutoModelForSequenceClassification`    |
    /// | `*ForTokenClassification`       | `AutoModelForTokenClassification`       |
    /// | `*ForQuestionAnswering`         | `AutoModelForQuestionAnswering`         |
    /// | `*ForMaskedLM`                  | `AutoModelForMaskedLM`                  |
    /// | `*Model` / no `For`             | base backbone (this method's old default) |
    /// | unrecognised `For{Other}`       | base backbone, with an info message     |
    ///
    /// Multi-head pretrained class names (`BertForPreTraining` etc.)
    /// fall back to the base backbone — mirrors HF Python's
    /// `AutoModel.from_pretrained` behaviour, where loading a
    /// pretraining checkpoint as `AutoModel` silently drops the heads.
    pub fn from_pretrained_for_export(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_for_export_on_device(repo_id, Device::CPU)
    }

    /// Device-aware variant of
    /// [`from_pretrained_for_export`](Self::from_pretrained_for_export).
    pub fn from_pretrained_for_export_on_device(
        repo_id: &str,
        device: Device,
    ) -> Result<Graph> {
        // Auto-detect the head from architectures[0]. Fetch only
        // config.json first (cheap) so we can decide the dispatch
        // without paying the full safetensors download for paths that
        // delegate to a head loader (which fetches its own).
        let config_str = fetch_config_str(repo_id)?;
        let probe = AutoConfig::from_json_str(&config_str)?;
        let arch_first = probe
            .architectures()
            .and_then(|a| a.first().cloned());
        let head_kind = match arch_first.as_deref() {
            Some(arch) => classify_for_hub_export(arch),
            None => HubExportHead::Base,
        };

        match head_kind {
            HubExportHead::Base => Self::base_from_pretrained_for_export(repo_id, device),
            HubExportHead::SeqCls => {
                let head = crate::models::auto::AutoModelForSequenceClassification
                    ::from_pretrained_on_device(repo_id, device)?;
                Ok(head.into_graph())
            }
            HubExportHead::TokCls => {
                let head = crate::models::auto::AutoModelForTokenClassification
                    ::from_pretrained_on_device(repo_id, device)?;
                Ok(head.into_graph())
            }
            HubExportHead::Qa => {
                let head = crate::models::auto::AutoModelForQuestionAnswering
                    ::from_pretrained_on_device(repo_id, device)?;
                Ok(head.into_graph())
            }
            HubExportHead::Mlm => {
                let head = crate::models::auto::AutoModelForMaskedLM
                    ::from_pretrained_on_device(repo_id, device)?;
                Ok(head.into_graph())
            }
        }
    }

    /// Base-backbone path of [`from_pretrained_for_export_on_device`].
    /// Stays separate from the auto-dispatcher so the
    /// `<Family>Model::on_device` vs `<Family>Model::on_device_without_pooler`
    /// pooler-presence logic doesn't bleed into the head paths
    /// (heads carry their own pooler / classifier wiring already).
    fn base_from_pretrained_for_export(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_auto_config_and_weights(repo_id)?;

        // Pick with-pooler vs without-pooler dynamically based on what
        // the checkpoint actually ships. Some Hub repos for pooler-
        // bearing families are encoder-only (e.g. `roberta-base`,
        // `FacebookAI/xlm-roberta-base`). Building with a pooler whose
        // weights aren't in the checkpoint trips the missing-keys
        // validation in `load_safetensors_into_graph_with_rename_allow_unused`.
        let has_pooler = weights_have_pooler(&weights)?;

        // Normalise `architectures` to the base class name on each
        // family arm. The Hub's source config typically tags a head
        // class (e.g. `bert-base-uncased` ships
        // `architectures: ["BertForPreTraining"]`) while this loader,
        // mirroring HF's `AutoModel.from_pretrained`, builds the base
        // backbone and silently drops head keys. The sidecar emitted
        // by a subsequent `save_checkpoint` must reflect what was
        // actually built, otherwise `build_for_export` re-dispatches
        // to the head class on `--checkpoint` re-export and produces
        // a graph whose structural hash no longer matches the saved
        // file (see `flodl-hf/tests/checkpoint_export_soak.rs`).
        let (graph, config_json) = match config {
            AutoConfig::Bert(c) => {
                let g = if has_pooler {
                    BertModel::on_device(&c, device)?
                } else {
                    BertModel::on_device_without_pooler(&c, device)?
                };
                (g, c.with_architectures("BertModel").to_json_str())
            }
            AutoConfig::Roberta(c) => {
                let g = if has_pooler {
                    RobertaModel::on_device(&c, device)?
                } else {
                    RobertaModel::on_device_without_pooler(&c, device)?
                };
                (g, c.with_architectures("RobertaModel").to_json_str())
            }
            AutoConfig::DistilBert(c) => {
                let g = DistilBertModel::on_device(&c, device)?;
                (g, c.with_architectures("DistilBertModel").to_json_str())
            }
            AutoConfig::XlmRoberta(c) => {
                let g = if has_pooler {
                    XlmRobertaModel::on_device(&c, device)?
                } else {
                    XlmRobertaModel::on_device_without_pooler(&c, device)?
                };
                (g, c.with_architectures("XLMRobertaModel").to_json_str())
            }
            AutoConfig::Albert(c) => {
                let g = if has_pooler {
                    AlbertModel::on_device(&c, device)?
                } else {
                    AlbertModel::on_device_without_pooler(&c, device)?
                };
                (g, c.with_architectures("AlbertModel").to_json_str())
            }
            AutoConfig::DebertaV2(c) => {
                let g = DebertaV2Model::on_device(&c, device)?;
                (g, c.with_architectures("DebertaV2Model").to_json_str())
            }
        };
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config_json);
        Ok(graph)
    }
}

/// Internal head-dispatch tag for the auto-detecting
/// `from_pretrained_for_export` path. Mirrors `export::HeadKind` but
/// kept separate because the Hub-mode policy is more permissive:
/// unrecognised `For{Other}` suffixes fall back to base instead of
/// erroring (a `bert-base-uncased` checkpoint advertising
/// `BertForPreTraining` is a real Hub case that should still produce
/// a base backbone, mirroring HF Python's `AutoModel.from_pretrained`).
enum HubExportHead {
    Base,
    SeqCls,
    TokCls,
    Qa,
    Mlm,
}

/// Permissive variant of `export::classify_architecture` for Hub-mode
/// auto-dispatch. Unsupported `For{Other}` (multi-head pretraining
/// classes etc.) falls back to base instead of erroring; the
/// downstream `<Family>Model::on_device` path tolerates head-class
/// keys via `allow_unused` on weight load. Recognised heads dispatch
/// to the matching `AutoModelFor*` loader.
fn classify_for_hub_export(arch: &str) -> HubExportHead {
    if arch.ends_with("ForSequenceClassification") {
        HubExportHead::SeqCls
    } else if arch.ends_with("ForTokenClassification") {
        HubExportHead::TokCls
    } else if arch.ends_with("ForQuestionAnswering") {
        HubExportHead::Qa
    } else if arch.ends_with("ForMaskedLM") {
        HubExportHead::Mlm
    } else {
        // `*Model` / no-`For` / unsupported `For{Other}` → base.
        // Multi-head class names like `BertForPreTraining` land here,
        // matching HF Python's `AutoModel.from_pretrained` policy of
        // building the base backbone and silently dropping the heads.
        HubExportHead::Base
    }
}

impl AutoModelForSequenceClassification {
    /// Download a fine-tuned sequence-classification checkpoint from
    /// the Hub, auto-detecting the family from `config.json`. The
    /// config must carry `num_labels` (or `id2label`); otherwise the
    /// concrete family's `num_labels_from_config` surfaces a loud
    /// error.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_auto_config_and_weights(repo_id)?;
        let (head, config_json) = match config {
            AutoConfig::Bert(c) => {
                let num_labels = BertForSequenceClassification::num_labels_from_config(&c)?;
                let h = BertForSequenceClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("BertForSequenceClassification").to_json_str();
                (Self::Bert(h), cj)
            }
            AutoConfig::Roberta(c) => {
                let num_labels = RobertaForSequenceClassification::num_labels_from_config(&c)?;
                let h = RobertaForSequenceClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("RobertaForSequenceClassification").to_json_str();
                (Self::Roberta(h), cj)
            }
            AutoConfig::DistilBert(c) => {
                let num_labels = DistilBertForSequenceClassification::num_labels_from_config(&c)?;
                let h = DistilBertForSequenceClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("DistilBertForSequenceClassification").to_json_str();
                (Self::DistilBert(h), cj)
            }
            AutoConfig::XlmRoberta(c) => {
                let num_labels = XlmRobertaForSequenceClassification::num_labels_from_config(&c)?;
                let h = XlmRobertaForSequenceClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("XLMRobertaForSequenceClassification").to_json_str();
                (Self::XlmRoberta(h), cj)
            }
            AutoConfig::Albert(c) => {
                let num_labels = AlbertForSequenceClassification::num_labels_from_config(&c)?;
                let h = AlbertForSequenceClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("AlbertForSequenceClassification").to_json_str();
                (Self::Albert(h), cj)
            }
            AutoConfig::DebertaV2(c) => {
                let num_labels = DebertaV2ForSequenceClassification::num_labels_from_config(&c)?;
                let h = DebertaV2ForSequenceClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("DebertaV2ForSequenceClassification").to_json_str();
                (Self::DebertaV2(h), cj)
            }
        };
        head.graph().set_source_config(config_json);
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl AutoModelForTokenClassification {
    /// Download a fine-tuned token-classification checkpoint (NER,
    /// POS, …) from the Hub, auto-detecting the family.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_auto_config_and_weights(repo_id)?;
        let (head, config_json) = match config {
            AutoConfig::Bert(c) => {
                let num_labels = BertForTokenClassification::num_labels_from_config(&c)?;
                let h = BertForTokenClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("BertForTokenClassification").to_json_str();
                (Self::Bert(h), cj)
            }
            AutoConfig::Roberta(c) => {
                let num_labels = RobertaForTokenClassification::num_labels_from_config(&c)?;
                let h = RobertaForTokenClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("RobertaForTokenClassification").to_json_str();
                (Self::Roberta(h), cj)
            }
            AutoConfig::DistilBert(c) => {
                let num_labels = DistilBertForTokenClassification::num_labels_from_config(&c)?;
                let h = DistilBertForTokenClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("DistilBertForTokenClassification").to_json_str();
                (Self::DistilBert(h), cj)
            }
            AutoConfig::XlmRoberta(c) => {
                let num_labels = XlmRobertaForTokenClassification::num_labels_from_config(&c)?;
                let h = XlmRobertaForTokenClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("XLMRobertaForTokenClassification").to_json_str();
                (Self::XlmRoberta(h), cj)
            }
            AutoConfig::Albert(c) => {
                let num_labels = AlbertForTokenClassification::num_labels_from_config(&c)?;
                let h = AlbertForTokenClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("AlbertForTokenClassification").to_json_str();
                (Self::Albert(h), cj)
            }
            AutoConfig::DebertaV2(c) => {
                let num_labels = DebertaV2ForTokenClassification::num_labels_from_config(&c)?;
                let h = DebertaV2ForTokenClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("DebertaV2ForTokenClassification").to_json_str();
                (Self::DebertaV2(h), cj)
            }
        };
        head.graph().set_source_config(config_json);
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl AutoModelForQuestionAnswering {
    /// Download a fine-tuned extractive-QA checkpoint (SQuAD, …) from
    /// the Hub, auto-detecting the family. QA heads have a fixed
    /// 2-wide output (start, end logits) independent of `num_labels`,
    /// so the config carries no head-size metadata requirement.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_auto_config_and_weights(repo_id)?;
        let (head, config_json) = match config {
            AutoConfig::Bert(c) => {
                let h = BertForQuestionAnswering::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("BertForQuestionAnswering").to_json_str();
                (Self::Bert(h), cj)
            }
            AutoConfig::Roberta(c) => {
                let h = RobertaForQuestionAnswering::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("RobertaForQuestionAnswering").to_json_str();
                (Self::Roberta(h), cj)
            }
            AutoConfig::DistilBert(c) => {
                let h = DistilBertForQuestionAnswering::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("DistilBertForQuestionAnswering").to_json_str();
                (Self::DistilBert(h), cj)
            }
            AutoConfig::XlmRoberta(c) => {
                let h = XlmRobertaForQuestionAnswering::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("XLMRobertaForQuestionAnswering").to_json_str();
                (Self::XlmRoberta(h), cj)
            }
            AutoConfig::Albert(c) => {
                let h = AlbertForQuestionAnswering::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("AlbertForQuestionAnswering").to_json_str();
                (Self::Albert(h), cj)
            }
            AutoConfig::DebertaV2(c) => {
                let h = DebertaV2ForQuestionAnswering::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("DebertaV2ForQuestionAnswering").to_json_str();
                (Self::DebertaV2(h), cj)
            }
        };
        head.graph().set_source_config(config_json);
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl AutoModelForMaskedLM {
    /// Download a masked-language-modelling checkpoint from the Hub,
    /// auto-detecting the family. MLM heads have no `num_labels`
    /// requirement — the decoder is tied to the word-embedding table
    /// and outputs `vocab_size` logits per position.
    ///
    /// Typical use is continued pretraining / domain adaptation on
    /// base checkpoints (`bert-base-uncased`, `roberta-base`,
    /// `distilbert-base-uncased`); each family's `from_pretrained`
    /// tolerates the redundant decoder-weight key some HF save
    /// formats ship, silently ignoring it during load.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_auto_config_and_weights(repo_id)?;
        let (head, config_json) = match config {
            AutoConfig::Bert(c) => {
                let h = BertForMaskedLM::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("BertForMaskedLM").to_json_str();
                (Self::Bert(h), cj)
            }
            AutoConfig::Roberta(c) => {
                let h = RobertaForMaskedLM::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("RobertaForMaskedLM").to_json_str();
                (Self::Roberta(h), cj)
            }
            AutoConfig::DistilBert(c) => {
                let h = DistilBertForMaskedLM::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("DistilBertForMaskedLM").to_json_str();
                (Self::DistilBert(h), cj)
            }
            AutoConfig::XlmRoberta(c) => {
                let h = XlmRobertaForMaskedLM::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("XLMRobertaForMaskedLM").to_json_str();
                (Self::XlmRoberta(h), cj)
            }
            AutoConfig::Albert(c) => {
                let h = AlbertForMaskedLM::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("AlbertForMaskedLM").to_json_str();
                (Self::Albert(h), cj)
            }
            AutoConfig::DebertaV2(c) => {
                let h = DebertaV2ForMaskedLM::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("DebertaV2ForMaskedLM").to_json_str();
                (Self::DebertaV2(h), cj)
            }
        };
        head.graph().set_source_config(config_json);
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

    /// Tiny synthetic BERT config — small enough that allocating a head
    /// graph in unit tests is millisecond-scale. Mirrors the
    /// `tiny_bert_config` private helper in `models/bert.rs` tests.
    fn tiny_bert_config() -> crate::models::bert::BertConfig {
        crate::models::bert::BertConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            intermediate_size: 32,
            max_position_embeddings: 8,
            type_vocab_size: 2,
            pad_token_id: Some(0),
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            hidden_act: flodl::nn::GeluApprox::Exact,
            num_labels: None,
            id2label: None,
            architectures: None,
        }
    }

    /// Sidecar path naming mirror of flodl's `sidecar_config_path`:
    /// strip `.gz` if present, then replace the trailing extension with
    /// `config.json`. Used by `save_checkpoint` when `source_config`
    /// is set on the graph.
    fn sidecar_for_checkpoint(checkpoint: &str) -> std::path::PathBuf {
        let mut p = std::path::PathBuf::from(checkpoint);
        if p.extension().and_then(|e| e.to_str()) == Some("gz") {
            p.set_extension("");
        }
        p.set_extension("config.json");
        p
    }

    /// Regression test for the head-side architecture normalisation:
    /// every `<Family>For{Head}::from_pretrained_on_device` MUST stamp
    /// `source_config` with `architectures: ["<Family>For{Head}"]`,
    /// regardless of what the upstream Hub config advertised. Otherwise
    /// a subsequent `save_checkpoint → --checkpoint re-export` cycle
    /// trips `classify_architecture` on the multi-head class name some
    /// Hub repos ship (`bert-base-uncased` → `BertForPreTraining`,
    /// which `classify_architecture` rejects loudly).
    ///
    /// The test simulates the from_pretrained_on_device contract
    /// without hitting the Hub: synthesizes a config whose upstream
    /// `architectures` is `BertForPreTraining`, then applies the same
    /// `with_architectures("BertForMaskedLM")` call the loader does
    /// before stamping `source_config`. save_checkpoint emits the
    /// sidecar; build_for_export reads it and dispatches to MLM.
    #[test]
    fn head_save_checkpoint_emits_normalised_architectures_sidecar() {
        use crate::export::build_for_export;
        use crate::models::auto::AutoConfig;
        use crate::models::bert::BertForMaskedLM;
        use flodl::Device;

        // Upstream-style config carrying the multi-head class name a
        // user pulling `bert-base-uncased` from the Hub would see.
        let upstream = tiny_bert_config().with_architectures("BertForPreTraining");

        // Build the MLM head and stamp source_config exactly the way
        // BertForMaskedLM::from_pretrained_on_device does post-fix.
        let head = BertForMaskedLM::on_device(&upstream, Device::CPU).unwrap();
        head.graph().set_source_config(
            upstream.with_architectures("BertForMaskedLM").to_json_str(),
        );

        // save_checkpoint emits the sidecar.
        let pid = std::process::id();
        let ckpt = std::env::temp_dir().join(format!("flodl_hf_mlm_norm_{pid}.fdl"));
        let ckpt_str = ckpt.to_string_lossy().to_string();
        head.graph().save_checkpoint(&ckpt_str).unwrap();
        let sidecar = sidecar_for_checkpoint(&ckpt_str);
        let sidecar_str = std::fs::read_to_string(&sidecar).unwrap();

        // Sidecar carries the head class, not the upstream multi-head class.
        let parsed = AutoConfig::from_json_str(&sidecar_str).unwrap();
        let arch = parsed.architectures().unwrap();
        assert_eq!(
            arch,
            ["BertForMaskedLM"],
            "save_checkpoint sidecar must reflect the head class actually built; \
             without the with_architectures call upstream's BertForPreTraining \
             would leak through and fail classify_architecture on re-export",
        );

        // build_for_export dispatches without an "unsupported architecture" error
        // and produces a graph with the same structural hash as the original.
        let rebuilt = build_for_export(&parsed, false, Device::CPU).unwrap();
        assert_eq!(
            rebuilt.structural_hash(),
            head.graph().structural_hash(),
            "build_for_export from sidecar must rebuild the same MLM topology",
        );

        // Cleanup.
        let _ = std::fs::remove_file(&ckpt);
        let _ = std::fs::remove_file(&sidecar);
    }
}
