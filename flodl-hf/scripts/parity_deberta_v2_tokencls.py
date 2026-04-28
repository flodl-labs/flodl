#!/usr/bin/env python3
"""Generate the parity fixture for `DebertaV2ForTokenClassification`.

Runs HuggingFace `DebertaV2ForTokenClassification` on a pinned sentence
and writes inputs + per-token reference logits to
`flodl-hf/tests/fixtures/deberta_v2_tokencls_parity.safetensors`.

Fixture model: `blaze999/Medical-NER` — DeBERTa-v3-base fine-tune on a
medical NER corpus (~40 entity classes). 2-input convention:
`input_ids` + `attention_mask` (no `token_type_ids`).

Run via `fdl flodl-hf parity deberta-v2-tokencls`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import model_info
from safetensors.torch import save_file
from transformers import AutoTokenizer, DebertaV2ForTokenClassification

from _hf_cache_utils import ensure_refs_main

MODEL_ID = "blaze999/Medical-NER"
REVISION: str | None = None

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "deberta_v2_tokencls_parity.safetensors"
)

PROMPT = "fab2s writes Rust code in Latent"


def main() -> None:
    torch.set_grad_enabled(False)

    sha = REVISION or model_info(MODEL_ID).sha
    print(f"using {MODEL_ID} @ {sha}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=sha)
    model = DebertaV2ForTokenClassification.from_pretrained(
        MODEL_ID, revision=sha,
    ).eval()
    ensure_refs_main(MODEL_ID, sha)

    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to(torch.int64)
    attention_mask = enc["attention_mask"].to(torch.int64)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    tensors = {
        "inputs.input_ids":      input_ids,
        "inputs.attention_mask": attention_mask,
        "outputs.logits":        out.logits.contiguous(),
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
    print(f"  logits {tuple(out.logits.shape)} "
          f"range [{out.logits.min():.4f}, {out.logits.max():.4f}]")


if __name__ == "__main__":
    main()
