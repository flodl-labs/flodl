#!/usr/bin/env python3
"""Generate the albert-base-v2 reference fixture for flodl parity tests.

Runs HuggingFace `AlbertModel` on a pinned input and writes both the
tokenised inputs and the reference outputs to
`flodl-hf/tests/fixtures/albert_base_v2_parity.safetensors`.

ALBERT's encoder takes the same 4-input shape as BERT
(`input_ids`, `position_ids`, `token_type_ids`, `attention_mask`).
We pass `position_ids` explicitly (`arange(seq_len)`) so the fixture
is reproducible and the flodl graph — which takes `position_ids` as a
named input — receives the same values HF would compute internally.

`add_pooling_layer=False` matches flodl-hf's
`AlbertModel::from_pretrained` default: the bare backbone is the
target of the parity comparison (`last_hidden_state` lives in
`hidden_size` after the factorised `embedding_hidden_mapping_in`
projection).

Run via `fdl flodl-hf parity-albert`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import model_info
from safetensors.torch import save_file
from transformers import AlbertModel, AutoTokenizer

from _hf_cache_utils import ensure_refs_main

MODEL_ID = "albert-base-v2"
REVISION: str | None = None

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "albert_base_v2_parity.safetensors"
)

PROMPT = "fab2s writes Rust"


def main() -> None:
    torch.set_grad_enabled(False)

    sha = REVISION or model_info(MODEL_ID).sha
    print(f"using {MODEL_ID} @ {sha}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=sha)
    # `add_pooling_layer=False` matches flodl-hf's pooler-free backbone:
    # `albert-base-v2` ships pooler weights but the parity target is the
    # encoder output, and skipping the pooler keeps the comparison
    # focused on the layers that change between checkpoints.
    model = AlbertModel.from_pretrained(
        MODEL_ID, revision=sha, add_pooling_layer=False,
    ).eval()
    ensure_refs_main(MODEL_ID, sha)

    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to(torch.int64)
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)
    token_type_ids = enc.get(
        "token_type_ids", torch.zeros_like(input_ids),
    ).to(torch.int64)
    attention_mask = enc["attention_mask"].to(torch.int64)

    # Force the math SDPA backend for reproducibility (matches the
    # backend flodl's libtorch build picks for these shapes).
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

    tensors = {
        "inputs.input_ids":          input_ids,
        "inputs.position_ids":       position_ids,
        "inputs.token_type_ids":     token_type_ids,
        "inputs.attention_mask":     attention_mask,
        "outputs.last_hidden_state": out.last_hidden_state.contiguous(),
    }
    metadata = {
        "source_model":  MODEL_ID,
        "source_sha":    sha,
        "torch_version": torch.__version__,
        "sdpa_backend":  "math",
        "prompt":        PROMPT,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(FIXTURE_PATH), metadata=metadata)
    print(f"wrote {FIXTURE_PATH}")
    print(f"  last_hidden_state {tuple(out.last_hidden_state.shape)} "
          f"range [{out.last_hidden_state.min():.4f}, "
          f"{out.last_hidden_state.max():.4f}]")


if __name__ == "__main__":
    main()
