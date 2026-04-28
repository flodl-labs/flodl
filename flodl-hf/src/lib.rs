//! # flodl-hf
//!
//! HuggingFace integration for [flodl](https://flodl.dev): safetensors I/O,
//! hub downloads, tokenizers, and pre-built transformer architectures.
//!
//! This crate is a sibling to `flodl` and depends on it for tensor, module,
//! and named-parameter primitives. Transformer building blocks come from
//! `flodl::nn`.
//!
//! ## Scope
//!
//! - [`safetensors_io`] — load/save named tensor dicts from safetensors files.
//! - [`hub`] — download models from the HuggingFace Hub with local caching.
//! - [`tokenizer`] — wrappers over the HuggingFace `tokenizers` crate.
//! - [`models`] — pre-built architectures (BERT first, LLaMA next).
//! - [`path`] — dotted-path builder for HF-compatible module naming.
//! - [`task_heads`] — shared [`task_heads::Answer`] / [`task_heads::TokenPrediction`]
//!   output types + internal helpers reused by every `*For*` task head.
//! - [`export`] — write a HF-compatible directory (`model.safetensors` +
//!   `config.json`) from a trained flodl graph.

pub(crate) mod config_json;
pub mod export;
#[cfg(feature = "hub")]
pub mod hub;
pub mod models;
pub mod path;
pub mod safetensors_io;
pub mod task_heads;
#[cfg(feature = "tokenizer")]
pub mod tokenizer;
