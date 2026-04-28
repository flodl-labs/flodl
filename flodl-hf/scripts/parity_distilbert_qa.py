#!/usr/bin/env python3
"""Generate the parity fixture for `DistilBertForQuestionAnswering`.

Fixture model: `distilbert/distilbert-base-cased-distilled-squad`
(extractive QA, canonical SQuAD fine-tune).

Run via `fdl flodl-hf parity distilbert-qa`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import model_info
from safetensors.torch import save_file
from transformers import AutoTokenizer, DistilBertForQuestionAnswering

from _hf_cache_utils import ensure_refs_main

MODEL_ID = "distilbert/distilbert-base-cased-distilled-squad"
REVISION: str | None = None

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "distilbert_qa_parity.safetensors"
)

QUESTION = "Where does fab2s live?"
CONTEXT = "fab2s lives in Latent and writes Rust deep learning code."


def sequence_ids_tensor(enc) -> torch.Tensor:
    """Pack `enc.sequence_ids(batch_index)` into an int64 tensor with
    `-1` for `None` (specials / padding). Mirrors the RoBERTa helper."""
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
    model = DistilBertForQuestionAnswering.from_pretrained(
        MODEL_ID, revision=sha,
    ).eval()
    ensure_refs_main(MODEL_ID, sha)

    enc = tok(QUESTION, CONTEXT, return_tensors="pt")
    input_ids = enc["input_ids"].to(torch.int64)
    attention_mask = enc["attention_mask"].to(torch.int64)
    sequence_ids = sequence_ids_tensor(enc)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    # HF QA outputs `start_logits` and `end_logits` as `[B, S]` tensors
    # each; the flodl graph emits a fused `[B, S, 2]` from
    # `Linear(dim, 2)`. Stack the two axes along `dim=-1` to match.
    logits = torch.stack(
        [out.start_logits, out.end_logits], dim=-1,
    ).contiguous()

    tensors = {
        "inputs.input_ids":      input_ids,
        "inputs.attention_mask": attention_mask,
        "inputs.sequence_ids":   sequence_ids,
        "outputs.logits":        logits,
    }
    metadata = {
        "source_model": MODEL_ID,
        "source_sha": sha,
        "torch_version": torch.__version__,
        "sdpa_backend": "math",
        "prompt": f"{QUESTION} || {CONTEXT}",
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(FIXTURE_PATH), metadata=metadata)
    print(f"wrote {FIXTURE_PATH}")
    print(f"  logits {tuple(logits.shape)} "
          f"range [{logits.min():.4f}, {logits.max():.4f}]")


if __name__ == "__main__":
    main()
