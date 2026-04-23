#!/usr/bin/env python3
"""Generate the DeBERTa-v3-base reference fixture for flodl parity tests.

Runs HuggingFace `DebertaV2Model` on a pinned input and writes both the
tokenised inputs and the reference outputs to
`flodl-hf/tests/fixtures/deberta_v2_parity.safetensors`.

DeBERTa-v3 ships under the `deberta-v2` architecture in `transformers`
(v3 is a config variant, not a separate class — see
`DebertaV2Config.model_type == "deberta-v2"`). Disentangled attention
uses a separate `rel_embeddings` tensor; `position_biased_input=False`
means the base model does NOT consume absolute positions, and
`type_vocab_size=0` means no `token_type_ids` — so the fixture saves
only `input_ids` + `attention_mask`.

Run via `fdl flodl-hf parity-deberta-v2`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import model_info
from safetensors.torch import save_file
from transformers import AutoTokenizer, DebertaV2Model

from _hf_cache_utils import ensure_refs_main

MODEL_ID = "microsoft/deberta-v3-base"
REVISION: str | None = None

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "deberta_v2_parity.safetensors"
)

PROMPT = "fab2s writes Rust"


def main() -> None:
    torch.set_grad_enabled(False)

    sha = REVISION or model_info(MODEL_ID).sha
    print(f"using {MODEL_ID} @ {sha}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=sha)
    model = DebertaV2Model.from_pretrained(MODEL_ID, revision=sha).eval()
    ensure_refs_main(MODEL_ID, sha)

    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to(torch.int64)
    attention_mask = enc["attention_mask"].to(torch.int64)

    # Force the math SDPA backend: matches the backend flodl's libtorch build
    # picks for these shapes, keeps parity deterministic. DeBERTa-v2's
    # disentangled attention is its own code path (not F.scaled_dot_product)
    # so the backend pin is belt-and-braces, but harmless.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    tensors = {
        "inputs.input_ids":          input_ids,
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
