# flodl-hf

HuggingFace integration for [flodl](https://flodl.dev): safetensors I/O,
hub downloads, tokenizers, and pre-built transformer architectures with
PyTorch-verified numerical parity.

## Status

BERT family complete: BERT, RoBERTa, DistilBERT with full task heads
(embed, sequence classification, token classification, extractive QA).
`AutoModel` routes automatically from `config.json`'s `model_type`. Every
head has `_live` parity tests against the HuggingFace Python reference;
observed `max_abs_diff` is under 1e-5 across the board.

## Getting started

Inside an existing flodl project:

```bash
fdl add flodl-hf
cd flodl-hf
fdl classify
```

This drops a `flodl-hf/` side crate pinned to the same flodl version
you already have, with a one-file `AutoModel` example, `fdl.yml` with
runnable commands, and a README documenting feature flavors and the
`.bin` → safetensors convert workflow.

Scaffolding a new project from scratch:

```bash
fdl init my-model --with-hf        # or answer "y" at the prompt
cd my-model/flodl-hf && fdl classify
```

## Manual install

If you prefer to wire `flodl-hf` straight into your main crate's
`Cargo.toml`, three feature profiles cover the common cases:

### Full HuggingFace experience (default)

```toml
flodl-hf = "0.5.2"
```

Pulls: `safetensors` + `hf-hub` + `tokenizers`. Everything needed to
load `bert-base-uncased` out of the box.

### Vision-only (hub, no tokenizer)

Useful for ViT, CLIP vision towers, or any image model where tokenization
is not needed. Drops the `tokenizers` crate and its regex + unicode
surface.

```toml
flodl-hf = { version = "0.5.2", default-features = false, features = ["hub"] }
```

### Offline / minimal (safetensors-only)

For air-gapped environments, embedded training, or pipelines that load
checkpoints from local disk. Drops both hub downloads and tokenizers. No
network, no async runtime, no TLS stack, no regex.

```toml
flodl-hf = { version = "0.5.2", default-features = false }
```

### Feature matrix

| Feature     | Adds dependency         | Enables                          |
|-------------|-------------------------|----------------------------------|
| `hub`       | `hf-hub` (sync, rustls) | Download models from the Hub     |
| `tokenizer` | `tokenizers`            | Text tokenization for LLMs, BERT |
| `cuda`      | `flodl/cuda`            | GPU-accelerated tensor ops       |

`safetensors` is always included. Without it the crate has no purpose.

## Design

This crate is a sibling to `flodl` and depends on it for `Tensor`,
`Module`, and the named-parameter machinery. Transformer blocks are
built on top of flodl's `nn` module (`LayerNorm`, `MultiheadAttention`,
`Embedding`, …). Hub loading validates safetensors keys against the
graph's expected parameter set before touching any weight, so the graph
is either fully initialised or the call errors.

## Quick examples

```rust
// Family-agnostic sequence classification
use flodl_hf::models::auto::AutoModelForSequenceClassification;

let clf = AutoModelForSequenceClassification::from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
)?;
let top = clf.predict(&["I love this framework"])?;
println!("{} ({:.3})", top[0][0].0, top[0][0].1);
```

```rust
// BERT-specific sequence classification
use flodl_hf::models::bert::BertForSequenceClassification;

let clf = BertForSequenceClassification::from_pretrained(
    "nateraw/bert-base-uncased-emotion",
)?;
let top = clf.predict(&["I love this framework"])?;
println!("{} ({:.3})", top[0][0].0, top[0][0].1);
```

Some older Hub uploads ship only `pytorch_model.bin`. Run
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
    "Where does fab2s live?",
    "fab2s lives in the latent and writes Rust deep learning code.",
)?;
println!("{:?}", a.text);
```

Runnable:
`fdl flodl-hf example bert-embed` / `bert-classify` / `bert-ner` / `bert-qa`,
plus `roberta-*` / `distilbert-*` / `auto-classify` for the same four
task shapes across every family.

## Roadmap

- [x] Safetensors read/write for named tensor dicts
- [x] `hf-hub` download + local cache wrappers
- [x] `tokenizers` crate integration
- [x] BERT (base-uncased parity with `transformers` library)
- [x] BERT task heads: sequence / token classification + question answering
- [x] RoBERTa family (base + three task heads, PyTorch parity)
- [x] DistilBERT family (base + three task heads, PyTorch parity)
- [x] `AutoModel` / `AutoConfig` dispatch across the three families
- [x] `fdl add flodl-hf` scaffold for on-site discovery
- [ ] ModernBERT (RoPE, GeGLU, alternating local/global attention)
- [ ] LLaMA (RoPE, GQA, SwiGLU, then the architecture)
- [ ] LoRA adapters
- [ ] ViT

## License

floDl is open-sourced software licensed under the [MIT license](https://github.com/fab2s/floDl/blob/main/LICENSE).
