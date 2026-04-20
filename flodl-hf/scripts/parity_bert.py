#!/usr/bin/env python3
"""Generate the BERT-base-uncased reference fixture for flodl parity tests.

Runs HuggingFace `BertModel` on a pinned input and writes both the inputs
and the reference outputs to
`flodl-hf/tests/fixtures/bert_base_uncased_parity.safetensors`.

Run via `fdl parity-bert` — the container has torch + transformers +
safetensors pre-installed. See `flodl-hf/scripts/Dockerfile.parity`.

The resolved model SHA is written into the safetensors metadata so the
fixture's provenance is traceable even if the Hub tag later moves.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import model_info
from safetensors.torch import save_file
from transformers import BertModel

MODEL_ID = "bert-base-uncased"
# `None` resolves to the current main-branch SHA at download time and is
# recorded in the safetensors metadata. Pin to a specific SHA here if a
# future refresh needs to be locked down.
REVISION: str | None = None

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "bert_base_uncased_parity.safetensors"
)


def main() -> None:
    torch.set_grad_enabled(False)

    sha = REVISION or model_info(MODEL_ID).sha
    print(f"using {MODEL_ID} @ {sha}")

    model = BertModel.from_pretrained(MODEL_ID, revision=sha).eval()

    input_ids = torch.tensor([[101, 7592, 2088, 102]], dtype=torch.int64)
    position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    token_type_ids = torch.zeros_like(input_ids)
    attention_mask = torch.ones_like(input_ids)

    # Force the math SDPA backend: it is fully deterministic on CPU fp32 and
    # is the backend flodl's libtorch build picks for these shapes. Removes
    # one variable from the parity budget.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

    tensors = {
        "inputs.input_ids": input_ids,
        "inputs.position_ids": position_ids,
        "inputs.token_type_ids": token_type_ids,
        "inputs.attention_mask": attention_mask,
        "outputs.last_hidden_state": out.last_hidden_state.contiguous(),
        "outputs.pooler_output": out.pooler_output.contiguous(),
    }
    metadata = {
        "source_model": MODEL_ID,
        "source_sha": sha,
        "torch_version": torch.__version__,
        "sdpa_backend": "math",
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(FIXTURE_PATH), metadata=metadata)
    print(f"wrote {FIXTURE_PATH}")
    print(f"  last_hidden_state {tuple(out.last_hidden_state.shape)} "
          f"range [{out.last_hidden_state.min():.4f}, "
          f"{out.last_hidden_state.max():.4f}]")
    print(f"  pooler_output     {tuple(out.pooler_output.shape)} "
          f"range [{out.pooler_output.min():.4f}, "
          f"{out.pooler_output.max():.4f}]")


if __name__ == "__main__":
    main()
