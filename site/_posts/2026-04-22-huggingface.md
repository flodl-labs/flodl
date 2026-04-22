---
title: "HuggingFace, in Rust"
subtitle: "v0.5.2: AutoModel for BERT, RoBERTa, DistilBERT with three task heads each, parity-tested against the PyTorch reference, fdl add flodl-hf to scaffold a playground in one command"
date: 2026-04-22
description: "flodl 0.5.2 ships the first cut of HuggingFace integration as the sibling crate flodl-hf. BERT, RoBERTa, and DistilBERT load via from_pretrained, with three task heads per family (sequence classification, NER, extractive QA). AutoModel routes from config.json's model_type. Every architecture and head has _live parity tests against the HuggingFace Python reference at max_abs_diff under 1e-5. fdl add flodl-hf scaffolds a working playground in one command."
---

BERT in flodl is now `from_pretrained("bert-base-uncased")?`. So is
RoBERTa. So is DistilBERT. So is any sentiment, NER, or SQuAD
fine-tune sitting on top of those three families. The new sibling
crate [`flodl-hf`](https://crates.io/crates/flodl-hf) ships in 0.5.2
with PyTorch-verified numerical parity (`max_abs_diff` under 1e-5 on
nine pinned checkpoints), three task heads per family, and one CLI
command to drop a working playground inside any flodl project.

Three deliverables, one theme: **the HuggingFace ecosystem, accessible
from a Rust binary, with the same single-call ergonomics PyTorch users
already know**.

## One line, any family

`AutoModel` inspects `config.json`'s `model_type` field and dispatches
to the right architecture without the caller knowing which family the
checkpoint belongs to. Same shape as HuggingFace Python's `AutoModel`,
same one-line API:

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

The same three lines work for `bert-base-uncased`, `roberta-base`,
`distilbert-base-uncased`, or any fine-tune on top of those. Swap the
repo id; the family wiring happens under the hood. Four `AutoModel`
entry points cover the common task shapes:

| Entry point                             | Task                        | Output                                |
|-----------------------------------------|-----------------------------|---------------------------------------|
| `AutoModel`                             | Backbone (hidden states)    | `[batch, seq_len, hidden]`            |
| `AutoModelForSequenceClassification`    | Whole-text labels           | `Vec<Vec<(String, f32)>>` per input   |
| `AutoModelForTokenClassification`       | Per-token labels (NER)      | `Vec<Vec<TokenPrediction>>` per input |
| `AutoModelForQuestionAnswering`         | Extractive answer span      | `Answer { text, start, end, score }`  |

When the family is known upfront, the per-family types
(`BertForSequenceClassification`, `RobertaForQuestionAnswering`,
`DistilBertForTokenClassification`, etc.) take exactly the same shape.
No dispatch overhead; same call site. Pick your level of generality
without changing your code structure.

## Three task heads, three families, parity-tested

`flodl-hf` 0.5.2 covers nine head-and-family combinations end to end.
Each one has a `_live` integration test that asserts
`max_abs_diff <= 1e-5` against the HuggingFace Python reference on a
pinned fine-tune. Observed values on the reference host:

| Checkpoint                                            | Head                | Observed `max_abs_diff` |
|-------------------------------------------------------|---------------------|-------------------------|
| `bert-base-uncased`                                   | Backbone (pooler)   | 9.835e-7                |
| `nateraw/bert-base-uncased-emotion`                   | SeqCls              | under 1e-5              |
| `dslim/bert-base-NER`                                 | TokenCls            | under 1e-5              |
| `csarron/bert-base-uncased-squad-v1`                  | QA                  | under 1e-5              |
| `roberta-base`                                        | Backbone            | under 1e-5              |
| `distilbert-base-uncased`                             | Backbone            | 1.431e-6                |
| `lxyuan/distilbert-*-sentiments-student`              | SeqCls              | 2.384e-7                |
| `dslim/distilbert-NER`                                | TokenCls            | 3.815e-6                |
| `distilbert/distilbert-base-cased-distilled-squad`    | QA                  | 2.623e-6                |

Every entry in that table is one runnable example, three lines of
code, one fine-tune downloaded from the Hub. Sequence classification:

```rust
use flodl_hf::models::bert::BertForSequenceClassification;

let clf = BertForSequenceClassification::from_pretrained(
    "nateraw/bert-base-uncased-emotion",
)?;
let top = clf.predict(&["I love this framework"])?;
println!("{} ({:.3})", top[0][0].0, top[0][0].1);
```

NER, with the `attends` flag on each `TokenPrediction` mirroring the
attention mask so padding drops out cleanly:

```rust
use flodl_hf::models::bert::BertForTokenClassification;

let ner = BertForTokenClassification::from_pretrained("dslim/bert-base-NER")?;
for t in &ner.predict(&["fab2s lives in Latent"])?[0] {
    if t.attends && t.label != "O" {
        println!("{} -> {} ({:.3})", t.token, t.label, t.score);
    }
}
```

Extractive QA, with span search restricted to context tokens via the
tokenizer's `sequence_ids` (the question region cannot answer itself):

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

The parity tests run in a dedicated Docker service (`hf-parity`,
`python:3.12-slim` plus torch 2.8.0 CPU and `transformers ~4.46`),
forcing `SDPBackend.MATH` for determinism. The reference outputs are
captured as safetensors fixtures pinned with `source_sha` and
`torch_version` metadata, then consumed by Rust integration tests at
`fdl test-live` time. Anyone can rerun the gates locally; the green
build is reproducible.

The tokenizer wrapper is model-agnostic. `HfTokenizer::from_pretrained`
loads `tokenizer.json` from any repo; the loaded JSON carries the
model-specific pre-tokenizer and post-processor. One wrapper serves
BERT, RoBERTa, DistilBERT, and (when they land) ModernBERT, LLaMA,
ViT.

## One command to scaffold a playground

API parity is half the story. The other half is **discoverability**:
how does a Rust developer with a fresh flodl project find out that
`flodl-hf` exists, learn the install flavors, and run an
`AutoModel.from_pretrained` call without reading three pages of docs
and editing `Cargo.toml` by hand.

```bash
fdl add flodl-hf       # scaffolds ./flodl-hf/ side crate
cd flodl-hf
fdl classify           # runs an AutoModel classifier on a real fine-tune
```

That drops a standalone cargo crate under `./flodl-hf/`: pinned to
the same `flodl` version the host project uses, with a one-file
`AutoModel` example, an `fdl.yml` declaring runnable commands
(`classify`, `bert`, `roberta-sentiment`, `distilbert-sentiment`, plus
`build` / `check` / `shell`), and a README documenting the three
feature flavors. Mode-aware: if the host project uses Docker, the
scaffold's `fdl.yml` keeps `docker: dev` on every command; if the host
is native, the `docker:` lines are stripped.

Nothing in the host project's root `Cargo.toml` or `fdl.yml` is
touched. The scaffold is a side crate for hands-on discovery; wiring
flodl-hf into the user's main code stays their decision. The generated
README walks them through it when they're ready.

For new projects:

```bash
fdl init my-model --with-hf            # Docker scaffold + flodl-hf side crate
fdl init my-model --native --with-hf   # Native scaffold + flodl-hf side crate
```

`fdl init` also asks "Include flodl-hf?" interactively when no mode
flag is passed, so a first-time user gets the option without having to
know the flag exists.

## Feature flavors

Three install profiles, picked at the dependency level:

```toml
flodl-hf = "0.5.2"                                                             # Full: safetensors + hf-hub + tokenizers
flodl-hf = { version = "0.5.2", default-features = false, features = ["hub"] } # Vision-only: hub, no tokenizer
flodl-hf = { version = "0.5.2", default-features = false }                     # Offline: safetensors only
```

The vision-only profile drops the `tokenizers` crate and its regex +
unicode surface. Useful for ViT, CLIP vision towers, or any image
model that does not need text. The offline profile drops Hub downloads
too: no network, no async runtime, no TLS stack, no regex. Useful for
air-gapped deploys, embedded training, or pipelines that load
checkpoints from local disk.

The HTTP backend on the `hub` feature is `ureq` plus `rustls-tls`. No
tokio, no openssl. The dev Docker image has no `libssl-dev` because
rustls is now the convention for any HTTP dep in the flodl stack.

## What flodl gained underneath

Three flodl-internal pieces had to land for BERT to work cleanly:

- **`scaled_dot_product_attention` FFI**: the fused softmax(QK^T/sqrt(d))V
  + optional masking + optional dropout kernel that PyTorch ships as
  `torch.nn.functional.scaled_dot_product_attention`. Used internally
  by `BertSelfAttention`; available to any flodl model that wants
  fused attention.
- **Native `torch.embedding` FFI** with `padding_idx`. The previous
  `index_select + reshape` path could not mask the padding row's
  gradient. The new `Embedding::with_padding_idx` constructor wraps
  `at::embedding` directly; the PAD row's gradient is masked to zero
  during backward by the native kernel.
- **`LayerNorm` with custom epsilon**: HuggingFace BERT uses
  `eps = 1e-12` (not PyTorch's `1e-5` default).
  `LayerNorm::with_eps(eps)` and `LayerNorm::on_device_with_eps(eps,
  device)` plug straight into the underlying kernel.

Everyone benefits, not just `flodl-hf`. `Tensor::scaled_dot_product_attention`
is part of the public flodl API now; any flodl model can use it.

## Three families today, more soon

Today: BERT, RoBERTa, DistilBERT, with embed / SeqCls / TokenCls / QA
heads each. The shared encoder lives in `flodl-hf/src/models/transformer_layer.rs`
as a single `TransformerLayer` module with a `LayerNaming` const swap;
`bert.rs`, `roberta.rs`, and `distilbert.rs` each carry only their
family-specific embeddings, pooler, and task heads. The collapse cuts
~3000 lines of duplicated encoder code; numbers are unchanged at
`max_abs_diff <= 1e-5` on all eight pinned checkpoints because the
parity tests gate the refactor.

What's next on the [roadmap](https://github.com/fab2s/floDl/blob/main/ROADMAP.md):

- **ModernBERT** (RoPE, GeGLU, alternating local/global attention)
- **LLaMA** (RoPE, GQA, SwiGLU)
- **LoRA adapters**
- **ViT**

Then the fine-tuning loop with ElChe on heterogeneous consumer GPUs,
which is the original arc this whole thing pulls from. flodl was built
to make distributed deep learning practical on the hardware people
actually own; serving published Hub checkpoints is the gateway to
fine-tuning them.

## Where to go next

- **Read the tutorial**: [HuggingFace Integration](/guide/flodl-hf)
  walks through the install flavors, AutoModel dispatch, per-family
  entry points, the tokenizer, loading from local disk, and the parity
  test infrastructure.
- **Skim the crate README**:
  [flodl-hf on GitHub](https://github.com/fab2s/floDl/tree/main/flodl-hf)
  is the short version, with the install matrix and a code block per
  task shape.
- **Try it**: `cargo new my-model && cd my-model && fdl init . --with-hf`
  drops a working AutoModel scaffold in under thirty seconds (assuming
  libtorch is already provisioned).
- **Read the changelog**: the
  [0.5.2 entry](https://github.com/fab2s/floDl/blob/main/CHANGELOG.md)
  is the per-feature record, including the flodl-internal pieces (SDPA
  FFI, `torch.embedding` FFI with `padding_idx`, `LayerNorm` epsilon).

And if you're hitting a checkpoint that ships only as
`pytorch_model.bin`, run `fdl flodl-hf convert <repo_id>` once. It
writes a `model.safetensors` into the local Hub cache, and
`from_pretrained` picks it up automatically.

The HuggingFace ecosystem is now one `from_pretrained` call away in
Rust. Everything else in flodl, the autograd engine, the named-graph
builder, the heterogeneous DDP trainer, the CUDA Graph capture path,
sits underneath unchanged. The point of `flodl-hf` is that you can
load any of those nine fine-tunes without thinking about it, and then
do whatever else you want with the resulting `Graph`.

That's a wrap on phase one. Phase two is fine-tuning.
