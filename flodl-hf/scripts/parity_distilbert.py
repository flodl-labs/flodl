#!/usr/bin/env python3
"""Generate the distilbert-base-uncased reference fixture for flodl
parity tests.

Runs HuggingFace `DistilBertModel` on a pinned input and writes the
tokenised inputs plus the reference outputs to
`flodl-hf/tests/fixtures/distilbert_base_parity.safetensors`.

DistilBERT is single-segment — the tokenizer still emits
`token_type_ids` (it inherits from the BERT tokenizer) but the model
ignores them, so they're not in the fixture. Position ids are
sequential `0..S`, computed internally both by HF Python and by the
flodl graph, so they aren't stored either. The fixture therefore
carries only `input_ids` + `attention_mask`.

Run via `fdl flodl-hf parity-distilbert`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import model_info
from safetensors.torch import save_file
from transformers import AutoTokenizer, DistilBertModel

from _hf_cache_utils import ensure_refs_main

MODEL_ID = "distilbert/distilbert-base-uncased"
REVISION: str | None = None

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "distilbert_base_parity.safetensors"
)

PROMPT = "Fabrice writes Rust"


def main() -> None:
    torch.set_grad_enabled(False)

    sha = REVISION or model_info(MODEL_ID).sha
    print(f"using {MODEL_ID} @ {sha}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=sha)
    model = DistilBertModel.from_pretrained(MODEL_ID, revision=sha).eval()
    ensure_refs_main(MODEL_ID, sha)

    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to(torch.int64)
    attention_mask = enc["attention_mask"].to(torch.int64)

    # Force the math SDPA backend for reproducibility (matches the
    # backend flodl's libtorch build uses for these shapes).
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    tensors = {
        "inputs.input_ids":         input_ids,
        "inputs.attention_mask":    attention_mask,
        "outputs.last_hidden_state": out.last_hidden_state.contiguous(),
    }
    metadata = {
        "source_model": MODEL_ID,
        "source_sha": sha,
        "torch_version": torch.__version__,
        "sdpa_backend": "math",
        "prompt": PROMPT,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(FIXTURE_PATH), metadata=metadata)
    print(f"wrote {FIXTURE_PATH}")
    print(f"  last_hidden_state {tuple(out.last_hidden_state.shape)} "
          f"range [{out.last_hidden_state.min():.4f}, "
          f"{out.last_hidden_state.max():.4f}]")


if __name__ == "__main__":
    main()
