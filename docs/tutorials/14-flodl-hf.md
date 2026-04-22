# Tutorial 14: HuggingFace Integration

Load BERT, RoBERTa, and DistilBERT checkpoints from the HuggingFace Hub
in a single line, run sequence classification, NER, or extractive QA,
and get PyTorch-verified numerical parity out of the box. All of this
lives in the [`flodl-hf`](https://crates.io/crates/flodl-hf) sibling
crate; this tutorial covers how to use it inside a floDl project.

> **Prerequisites**: [Tensors](01-tensors.md),
> [Modules](03-modules.md), and [Graph Builder](05-graph-builder.md).
> Familiarity with HuggingFace Transformers helps but is not required.

> **Time**: ~20 minutes.

## Quick start

Inside an existing flodl project (one scaffolded with `fdl init`):

```bash
fdl add flodl-hf       # scaffolds ./flodl-hf/ side crate
cd flodl-hf
fdl classify           # runs a live AutoModel classifier
```

Scaffolding a fresh project with HuggingFace included from day one:

```bash
fdl init my-model --with-hf
cd my-model/flodl-hf
fdl classify
```

`fdl add flodl-hf` drops a standalone cargo crate under `./flodl-hf/`
with its own `Cargo.toml`, a one-file `AutoModel` classifier
(`src/main.rs`), an `fdl.yml` with runnable commands, and a `README`
covering the feature flavors and `.bin` conversion workflow. The
scaffold version-locks `flodl-hf` to the same `flodl` version the host
project uses, so the two crates stay in sync.

Nothing in the host project's `Cargo.toml` or `fdl.yml` is touched. The
scaffold is a side playground for discovery; wiring flodl-hf into the
main crate stays the caller's decision.

## Install

If you prefer to wire `flodl-hf` directly into your main crate, three
feature profiles cover the common cases.

### Full HuggingFace experience (default)

```toml
flodl-hf = "0.5.2"
```

Pulls `safetensors` + `hf-hub` + `tokenizers`. Everything needed to
load `bert-base-uncased` out of the box, including text tokenization
and Hub downloads.

### Vision-only (hub, no tokenizer)

For ViT, CLIP vision towers, or any image model that does not need
tokenization. Drops the `tokenizers` crate and its regex + unicode
surface.

```toml
flodl-hf = { version = "0.5.2", default-features = false, features = ["hub"] }
```

### Offline / minimal (safetensors-only)

For air-gapped environments, embedded training, or pipelines that load
checkpoints from local disk. Drops Hub downloads and tokenizers. No
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

`safetensors` is always included. The HTTP backend is `ureq` +
`rustls-tls`; no tokio, no openssl.

## `AutoModel`: family-agnostic loading

`AutoModel` inspects `config.json`'s `model_type` field and dispatches
to the right architecture (BERT, RoBERTa, or DistilBERT) without the
caller knowing which family the checkpoint belongs to. This mirrors
HuggingFace Python's `AutoModel` / `AutoModelFor*` entry points.

```rust
use flodl_hf::models::auto::AutoModelForSequenceClassification;

let clf = AutoModelForSequenceClassification::from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
)?;

let results = clf.predict(&["I love this framework"])?;
for (label, score) in &results[0] {
    println!("{} ({:.3})", label, score);
}
```

The same three-line caller works for `bert-base-uncased`,
`roberta-base`, `distilbert-base-uncased`, or any fine-tune on top of
those. Swap the repo id and the family wiring happens under the hood.

Four `AutoModel` entry points cover the common task shapes:

| Entry point                             | Task                        | Output shape                        |
|-----------------------------------------|-----------------------------|-------------------------------------|
| `AutoModel`                             | Backbone (hidden states)    | `[batch, seq_len, hidden]`          |
| `AutoModelForSequenceClassification`    | Whole-text labels           | `Vec<Vec<(String, f32)>>`           |
| `AutoModelForTokenClassification`       | Per-token labels (NER)      | `Vec<Vec<TokenPrediction>>`         |
| `AutoModelForQuestionAnswering`         | Extractive answer span      | `Answer { text, start, end, score }`|

## Per-family entry points

When the family is known upfront, use the concrete type. Same API,
no dispatch layer, and a slightly richer surface (BERT keeps its
pooler; RoBERTa exposes `on_device` for checkpoints that ship one).

### Sequence classification

```rust
use flodl_hf::models::bert::BertForSequenceClassification;

let clf = BertForSequenceClassification::from_pretrained(
    "nateraw/bert-base-uncased-emotion",
)?;
let top = clf.predict(&["I love this framework"])?;
println!("{} ({:.3})", top[0][0].0, top[0][0].1);
```

`predict(&[&str])` returns a per-input `Vec<(String, f32)>` sorted by
probability, with label names parsed from the checkpoint's `id2label`
(or `LABEL_k` as a fallback). The BERT head is
`pooler_output -> Dropout -> Linear(hidden, num_labels)`; RoBERTa uses
its native two-layer head (`Dropout -> dense -> tanh -> Dropout -> out_proj`)
on the `<s>` hidden state; DistilBERT has its own two-layer head
(`select(CLS) -> pre_classifier -> ReLU -> Dropout -> classifier`).
Same call site, different internals.

### Token classification (NER)

```rust
use flodl_hf::models::bert::BertForTokenClassification;

let ner = BertForTokenClassification::from_pretrained("dslim/bert-base-NER")?;
for t in &ner.predict(&["fab2s lives in Latent"])?[0] {
    if t.attends && t.label != "O" {
        println!("{} -> {} ({:.3})", t.token, t.label, t.score);
    }
}
```

`TokenPrediction { token, label, score, attends }` covers each
sub-token. The `attends` flag mirrors the attention mask, so padding
drops cleanly from the result. Works with BERT
(`dslim/bert-base-NER`), RoBERTa
(`Jean-Baptiste/roberta-large-ner-english`), and DistilBERT
(`dslim/distilbert-NER`).

### Extractive question answering

```rust
use flodl_hf::models::bert::BertForQuestionAnswering;

let qa = BertForQuestionAnswering::from_pretrained(
    "csarron/bert-base-uncased-squad-v1",
)?;
let a = qa.answer(
    "Where does fab2s live?",
    "fab2s lives in Latent and writes Rust deep learning code.",
)?;
println!("{:?} (score {:.3})", a.text, a.score);
```

The span search is restricted to context tokens via the tokenizer's
`sequence_ids`, so the question region cannot answer itself. Works
with SQuAD-family fine-tunes across all three families:
`csarron/bert-base-uncased-squad-v1`,
`deepset/roberta-base-squad2`,
`distilbert/distilbert-base-cased-distilled-squad`.

`answer_batch(&[(question, context)])` runs a batch of pairs in one
forward.

### Embeddings

```rust
use flodl_hf::models::bert::BertModel;

let model = BertModel::from_pretrained("bert-base-uncased")?;
// model is a flodl::Graph; run it with forward_multi
```

BERT returns pooled output (CLS passed through tanh) by default;
RoBERTa and DistilBERT return `last_hidden_state` because their
reference checkpoints either lack a pooler or ship one randomly
initialised. `BertModel::on_device_without_pooler` matches HuggingFace
Python's `add_pooling_layer=False` if you want the hidden state
directly.

The runnable `*_embed` examples (see below) wire the tokenizer to the
model end-to-end and print per-sentence vectors.

## Tokenizer

`HfTokenizer` is a thin wrapper over the `tokenizers` crate. One
wrapper serves BERT, RoBERTa, DistilBERT, and future families: the
loaded `tokenizer.json` carries the model-specific pre-tokenizer and
post-processor.

```rust
use flodl_hf::tokenizer::HfTokenizer;

let tok = HfTokenizer::from_pretrained("bert-base-uncased")?;
let batch = tok.encode(&["hello world", "another input"])?;
// batch.input_ids / attention_mask / token_type_ids / position_ids
// are i64 [B, S] Variables; sequence_ids carries the paired-segment
// tag (0 first / 1 second / -1 special).
```

`encode_pairs(&[(q, c)])` produces paired encodings with
`token_type_ids == 1` on the second segment, required for QA and
useful for NLI or sentence-pair classification.

Padding defaults to `BatchLongest` with `pad_id = [PAD]` when
`tokenizer.json` has no padding config of its own. There is no default
truncation: oversized texts error loudly at the model rather than
silently truncate. If you need truncation, configure it on the
`Tokenizer` directly before encoding.

Task-head wrappers (`*ForSequenceClassification`, etc.) pull the
tokenizer from the same repo id at `from_pretrained` time, so
`predict(&[&str])` takes raw text without manual tokenization. Direct
`BertModel` / `RobertaModel` / `DistilBertModel` users wire the
tokenizer themselves.

## Loading from local disk

Every `from_pretrained` variant has a sibling that skips the Hub and
reads a local safetensors file. Useful for air-gapped deploys and for
users on the `default-features = false` profile.

```rust
use flodl_hf::models::bert::{BertConfig, BertModel};
use flodl_hf::safetensors_io::load_safetensors_file_into_graph;

let config = BertConfig::from_json_str(&std::fs::read_to_string("config.json")?)?;
let mut graph = BertModel::build(&config)?;
load_safetensors_file_into_graph(&mut graph, "model.safetensors")?;
```

The loader runs a strict key-set validation before touching any
parameter:

- Missing keys (expected by the graph but absent from the checkpoint)
- Unused keys (present in the checkpoint but not consumed)
- Shape mismatches

A disagreement in any bucket errors with up to 20 entries per bucket
and a `"... and N more"` truncation tail. The graph is either fully
loaded or fully untouched; no silent drift.

Rename-aware variants handle legacy checkpoint conventions (for
example BERT's pre-2020 `LayerNorm.gamma` / `LayerNorm.beta` to
`weight` / `bias`). Allow-unused variants log and skip extra keys
instead of erroring, used under the hood by `*For*::from_pretrained`
when a base-model checkpoint carries task-specific heads flodl-hf does
not consume.

## Parity with PyTorch

Every architecture and task head has an `_live` integration test that
asserts `max_abs_diff <= 1e-5` on a pinned checkpoint against the
HuggingFace Python reference. Observed values on the reference host:

| Checkpoint                                            | Test                               | Observed `max_abs_diff` |
|-------------------------------------------------------|------------------------------------|-------------------------|
| `bert-base-uncased`                                   | Backbone `pooler_output`           | 9.835e-7                |
| `nateraw/bert-base-uncased-emotion`                   | SeqCls logits                      | Under 1e-5              |
| `dslim/bert-base-NER`                                 | TokenCls logits                    | Under 1e-5              |
| `csarron/bert-base-uncased-squad-v1`                  | QA start/end logits                | Under 1e-5              |
| `roberta-base`                                        | Backbone hidden state              | Under 1e-5              |
| `distilbert-base-uncased`                             | Backbone hidden state              | 1.431e-6                |
| `lxyuan/distilbert-*-sentiments-student`              | SeqCls logits                      | 2.384e-7                |
| `dslim/distilbert-NER`                                | TokenCls logits                    | 3.815e-6                |
| `distilbert/distilbert-base-cased-distilled-squad`    | QA start/end logits                | 2.623e-6                |

Run the parity gates locally:

```bash
fdl test-live
```

This executes `cargo test live -- --nocapture --ignored`, picking up
any test with a `_live` suffix behind `#[ignore]`. The tests need
network access (for Hub downloads) and cache weights under
`./.hf-cache/` via the `HF_HOME` env var.

The parity fixtures themselves are regenerated through
`fdl flodl-hf parity-bert` and siblings (one command per checkpoint,
twelve total). These run a Python Docker service
(`hf-parity`, `python:3.12-slim` + torch 2.8.0 CPU) to produce the
reference outputs; flodl-hf then consumes the resulting safetensors
files at test time. Contributors rerun these when bumping checkpoint
shas; end users do not need to.

## Checkpoints with only `pytorch_model.bin`

Some older Hub uploads ship only the unsafe PyTorch pickle format. For
those, run the one-off converter:

```bash
fdl flodl-hf convert <repo_id>
```

This writes a `model.safetensors` into the local Hub cache, after
which `from_pretrained` picks it up automatically.

## Supported families and roadmap

Landed in 0.5.2:

- BERT family (base-uncased tested), with all three task heads
- RoBERTa family, with all three task heads
- DistilBERT family, with all three task heads
- `AutoModel` / `AutoConfig` dispatch across the three families
- `fdl add flodl-hf` + `fdl init --with-hf`

On the roadmap:

- ModernBERT (RoPE, GeGLU, alternating local/global attention)
- LLaMA (RoPE, GQA, SwiGLU)
- LoRA adapters
- ViT

## Runnable examples

Thirteen examples ship with flodl-hf, one per family × task plus the
`AutoModel` demo:

```bash
fdl flodl-hf example auto-classify                   # any family
fdl flodl-hf example bert-embed                      # also: bert-classify / -ner / -qa
fdl flodl-hf example roberta-embed                   # also: roberta-classify / -ner / -qa
fdl flodl-hf example distilbert-embed                # also: distilbert-classify / -ner / -qa
```

Each example downloads a real fine-tune, runs a small pinned batch,
and prints top labels, entities, or extracted spans.

## Further reading

- [`flodl-hf` crate README](https://github.com/fab2s/floDl/blob/main/flodl-hf/README.md)
- [flodl-hf examples](https://github.com/fab2s/floDl/tree/main/flodl-hf/examples)
- [The floDl CLI](../cli.md) (see `fdl add`, `fdl flodl-hf`, `fdl test-live`)

---

Previous: [Data Loading](13-data-loading.md) |
Next: [DDP Reference](../ddp.md)
