#!/usr/bin/env python3
"""Generate the parity fixture for `DebertaV2ForSequenceClassification`.

Runs HuggingFace `DebertaV2ForSequenceClassification` on a pinned text
input and writes both the tokenised inputs and the reference logits to
`flodl-hf/tests/fixtures/deberta_v2_seqcls_parity.safetensors`.

Fixture model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
(3-label NLI: entailment / neutral / contradiction). DeBERTa-v3 ships
under the `deberta-v2` architecture in `transformers`. Inputs follow
the 2-input convention: `input_ids` + `attention_mask` (no
`token_type_ids`, `type_vocab_size=0`).

Run via `fdl flodl-hf parity deberta-v2-seqcls`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import model_info
from safetensors.torch import save_file
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification

from _hf_cache_utils import ensure_refs_main

MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
REVISION: str | None = None

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "deberta_v2_seqcls_parity.safetensors"
)

PROMPT = "I really love this new Rust framework"


def main() -> None:
    torch.set_grad_enabled(False)

    sha = REVISION or model_info(MODEL_ID).sha
    print(f"using {MODEL_ID} @ {sha}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=sha)
    model = DebertaV2ForSequenceClassification.from_pretrained(
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
