# flodl-hf

HuggingFace integration for [flodl](https://flodl.dev): safetensors I/O,
hub downloads, tokenizers, and pre-built transformer architectures.

## Status

Scaffold. Implementation in progress, starting with BERT as the integration
shakedown. Follow-up: LLaMA (needs RoPE + GQA + SwiGLU).

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

## Roadmap

- [ ] Safetensors read/write for named tensor dicts
- [ ] `hf-hub` download + local cache wrappers
- [ ] `tokenizers` crate integration
- [ ] BERT (base-uncased parity with `transformers` library)
- [ ] LLaMA (RoPE, GQA, SwiGLU, then the architecture)
- [ ] LoRA adapters
- [ ] ViT

## License

MIT. See repository root.
