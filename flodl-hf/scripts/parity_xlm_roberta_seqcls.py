#!/usr/bin/env python3
"""Generate the parity fixture for `XLMRobertaForSequenceClassification`.

Runs HuggingFace `XLMRobertaForSequenceClassification` on a pinned text
input and writes both the tokenised inputs and the reference logits to
`flodl-hf/tests/fixtures/xlm_roberta_seqcls_parity.safetensors`.

Fixture model: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
(3-label sentiment: negative / neutral / positive).

Run via `fdl flodl-hf parity xlm-roberta-seqcls`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import model_info
from safetensors.torch import save_file
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification

from _hf_cache_utils import ensure_refs_main

MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
REVISION: str | None = None

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "xlm_roberta_seqcls_parity.safetensors"
)

PROMPT = "I really love this new Rust framework"


def sequence_ids_tensor(enc) -> torch.Tensor:
    batch = enc["input_ids"].shape[0]
    rows = []
    for b in range(batch):
        rows.append([
            -1 if v is None else int(v)
            for v in enc.sequence_ids(b)
        ])
    return torch.tensor(rows, dtype=torch.int64)


def main() -> None:
    torch.set_grad_enabled(False)

    sha = REVISION or model_info(MODEL_ID).sha
    print(f"using {MODEL_ID} @ {sha}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=sha)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_ID, revision=sha,
    ).eval()
    ensure_refs_main(MODEL_ID, sha)

    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to(torch.int64)
    attention_mask = enc["attention_mask"].to(torch.int64)
    token_type_ids = enc.get(
        "token_type_ids", torch.zeros_like(input_ids),
    ).to(torch.int64)
    sequence_ids = sequence_ids_tensor(enc)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    tensors = {
        "inputs.input_ids":      input_ids,
        "inputs.token_type_ids": token_type_ids,
        "inputs.attention_mask": attention_mask,
        "inputs.sequence_ids":   sequence_ids,
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
