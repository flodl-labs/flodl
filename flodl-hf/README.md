# flodl-hf

HuggingFace integration for [flodl](https://flodl.dev): safetensors I/O,
hub downloads, tokenizers, and pre-built transformer architectures.

## Status

BERT family in progress. End-to-end parity with HuggingFace Python on
`bert-base-uncased` (`9.8e-7` max-abs-diff on pooled output). Task heads
(`BertForSequenceClassification`, `BertForTokenClassification`,
`BertForQuestionAnswering`) ship with runnable examples and `_live`
parity tests against real fine-tuned checkpoints on the Hub.

Next: RoBERTa, DistilBERT, ELECTRA, then `AutoModel` routing. Follow-up
arc: LLaMA family (needs RoPE + GQA + SwiGLU).

## Install

Three feature profiles cover the main use cases. Pick the one that matches
what you're doing.

### Full HuggingFace experience (default)

```toml
flodl-hf = "0.5.1"
```

Pulls: `safetensors` + `hf-hub` + `tokenizers`. Everything needed to
`load("bert-base-uncased")` out of the box.

### Vision-only (hub, no tokenizer)

Useful for ViT, CLIP vision towers, or any image model where tokenization
is not needed. Drops the `tokenizers` crate and its regex + unicode surface.

```toml
flodl-hf = { version = "0.5.1", default-features = false, features = ["hub"] }
```

### Offline / minimal (safetensors-only)

For air-gapped environments, embedded training, or pipelines that load
checkpoints from local disk. Drops both hub downloads and tokenizers. No
network, no async runtime, no TLS stack, no regex.

```toml
flodl-hf = { version = "0.5.1", default-features = false }
```

### Feature matrix

| Feature     | Adds dependency       | Enables                           |
|-------------|-----------------------|-----------------------------------|
| `hub`       | `hf-hub` (sync, rustls) | Download models from the Hub    |
| `tokenizer` | `tokenizers`          | Text tokenization for LLMs, BERT  |
| `cuda`      | `flodl/cuda`          | GPU-accelerated tensor ops        |

`safetensors` is always included. Without it the crate has no purpose.

## Design

This crate is a sibling to `flodl` and depends on it for `Tensor`, `Module`,
and the named-parameter machinery. Transformer blocks are built on top of
flodl's `nn` module (LayerNorm, MultiheadAttention, Embedding, etc.).

## Quick examples

```rust
// Emotion classification (6 emotions)
use flodl_hf::models::bert::BertForSequenceClassification;

let clf = BertForSequenceClassification::from_pretrained(
    "nateraw/bert-base-uncased-emotion",
)?;
let top = clf.predict(&["I love this framework"])?;
println!("{} ({:.3})", top[0][0].0, top[0][0].1); // "joy (0.983)"
```

Note: some older Hub uploads ship only `pytorch_model.bin`. Run
`fdl flodl-hf convert <repo_id>` once to produce a safetensors copy
in the local cache, then `from_pretrained` picks it up automatically.

```rust
// Named-entity recognition
use flodl_hf::models::bert::BertForTokenClassification;

let ner = BertForTokenClassification::from_pretrained("dslim/bert-base-NER")?;
for t in &ner.predict(&["Fabrice lives in Paris"])?[0] {
    if t.attends && t.label != "O" {
        println!("{} → {} ({:.3})", t.token, t.label, t.score);
    }
}
```

```rust
// Extractive QA
use flodl_hf::models::bert::BertForQuestionAnswering;

let qa = BertForQuestionAnswering::from_pretrained(
    "csarron/bert-base-uncased-squad-v1",
)?;
let a = qa.answer(
    "Where does Fabrice live?",
    "Fabrice lives in Paris and writes Rust deep learning code.",
)?;
println!("{:?}", a.text); // "paris"
```

Runnable: `fdl flodl-hf example bert-embed` / `bert-classify` / `bert-ner` / `bert-qa`.

## Roadmap

- [x] Safetensors read/write for named tensor dicts
- [x] `hf-hub` download + local cache wrappers
- [x] `tokenizers` crate integration
- [x] BERT (base-uncased parity with `transformers` library)
- [x] BERT task heads: sequence / token classification + question answering
- [ ] RoBERTa, DistilBERT, ELECTRA — the BERT-family completion arc
- [ ] `AutoModel` / `AutoConfig` dispatch
- [ ] LLaMA (RoPE, GQA, SwiGLU, then the architecture)
- [ ] LoRA adapters
- [ ] ViT

## License

MIT. See repository root.
