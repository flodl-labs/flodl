#!/usr/bin/env python3
"""Generate the roberta-base reference fixture for flodl parity tests.

Runs HuggingFace `RobertaModel` on a pinned input and writes both the
tokenised inputs and the reference outputs to
`flodl-hf/tests/fixtures/roberta_base_parity.safetensors`.

RoBERTa computes `position_ids` internally from `input_ids` using its
padding-offset convention, so the fixture does NOT save them — the
flodl graph follows the same path.

Run via `fdl flodl-hf parity-roberta`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import model_info
from safetensors.torch import save_file
from transformers import AutoTokenizer, RobertaModel

from _hf_cache_utils import ensure_refs_main

MODEL_ID = "roberta-base"
REVISION: str | None = None

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "roberta_base_parity.safetensors"
)

PROMPT = "Fabrice writes Rust"


def sequence_ids_tensor(enc) -> torch.Tensor:
    """Pack `enc.sequence_ids(batch_index)` into an int64 tensor, with
    `-1` for `None` (specials / padding)."""
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
    # `add_pooling_layer=False` matches flodl-hf's
    # `RobertaModel::from_pretrained` default: `roberta-base` doesn't
    # ship pooler weights, so including the pooler here would leave it
    # random-initialised and produce non-reproducible output.
    model = RobertaModel.from_pretrained(
        MODEL_ID, revision=sha, add_pooling_layer=False,
    ).eval()
    ensure_refs_main(MODEL_ID, sha)

    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to(torch.int64)
    attention_mask = enc["attention_mask"].to(torch.int64)
    token_type_ids = enc.get(
        "token_type_ids", torch.zeros_like(input_ids),
    ).to(torch.int64)
    sequence_ids = sequence_ids_tensor(enc)

    # Force the math SDPA backend for reproducibility (matches the
    # backend flodl's libtorch build uses for these shapes).
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids intentionally omitted — RobertaModel computes
            # them internally, and the flodl graph does the same.
        )

    tensors = {
        "inputs.input_ids":         input_ids,
        "inputs.token_type_ids":    token_type_ids,
        "inputs.attention_mask":    attention_mask,
        "inputs.sequence_ids":      sequence_ids,
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
