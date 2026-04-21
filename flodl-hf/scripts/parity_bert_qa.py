#!/usr/bin/env python3
"""Generate the parity fixture for `BertForQuestionAnswering`.

Runs HuggingFace `BertForQuestionAnswering` on a pinned (question, context)
pair and writes inputs + start/end logits to
`flodl-hf/tests/fixtures/bert_qa_parity.safetensors`.

Fixture model: `csarron/bert-base-uncased-squad-v1` (SQuAD v1, 2-wide head).

Run via `fdl flodl-hf parity-bert-qa`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import model_info
from safetensors.torch import save_file
from transformers import AutoTokenizer, BertForQuestionAnswering

from _hf_cache_utils import ensure_refs_main

MODEL_ID = "csarron/bert-base-uncased-squad-v1"
REVISION: str | None = None

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "bert_qa_parity.safetensors"
)

QUESTION = "Where does Fabrice live?"
CONTEXT = "Fabrice lives in Paris and writes Rust deep learning code."


def main() -> None:
    torch.set_grad_enabled(False)

    sha = REVISION or model_info(MODEL_ID).sha
    print(f"using {MODEL_ID} @ {sha}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=sha)
    model = BertForQuestionAnswering.from_pretrained(MODEL_ID, revision=sha).eval()
    ensure_refs_main(MODEL_ID, sha)

    enc = tok(QUESTION, CONTEXT, return_tensors="pt")
    input_ids = enc["input_ids"].to(torch.int64)
    attention_mask = enc["attention_mask"].to(torch.int64)
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids)).to(torch.int64)
    seq = input_ids.shape[1]
    position_ids = torch.arange(seq, dtype=torch.int64).unsqueeze(0)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

    # Stack start/end as a `[1, seq, 2]` tensor so the flodl test can
    # compare against a single logits block matching its own head output.
    logits = torch.stack([out.start_logits, out.end_logits], dim=-1).contiguous()

    tensors = {
        "inputs.input_ids":       input_ids,
        "inputs.position_ids":    position_ids,
        "inputs.token_type_ids":  token_type_ids,
        "inputs.attention_mask":  attention_mask,
        "outputs.logits":         logits,
    }
    metadata = {
        "source_model": MODEL_ID,
        "source_sha": sha,
        "torch_version": torch.__version__,
        "sdpa_backend": "math",
        "question": QUESTION,
        "context": CONTEXT,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(FIXTURE_PATH), metadata=metadata)
    print(f"wrote {FIXTURE_PATH}")
    print(f"  logits {tuple(logits.shape)} "
          f"range [{logits.min():.4f}, {logits.max():.4f}]")


if __name__ == "__main__":
    main()
