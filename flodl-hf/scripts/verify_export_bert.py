#!/usr/bin/env python3
"""Verify HF `BertModel.from_pretrained` can load a flodl export and that
its forward pass is bit-exact to the Hub-loaded source on a fixed input.

Reads `tests/.exports/bert/` (stage it first with
`fdl flodl-hf export bert-base-uncased flodl-hf/tests/.exports/bert`),
loads it via `BertModel.from_pretrained(<dir>)`, runs forward on the
same input the parity fixture uses, and compares against
`BertModel.from_pretrained("bert-base-uncased")` on the same input.

If both come from the same weights through identical PyTorch code,
results are bit-identical. Any drift means the exported safetensors or
config.json failed to round-trip a key, dtype, or shape.

Run via `fdl flodl-hf verify-export-bert` (the hf-parity container has
torch + transformers pre-installed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import BertModel

from _hf_cache_utils import ensure_refs_main

SOURCE_REPO = "bert-base-uncased"
EXPORT_DIR = Path(__file__).resolve().parents[1] / "tests" / ".exports" / "bert"


def main() -> int:
    if not EXPORT_DIR.exists():
        print(
            f"error: export dir not found at {EXPORT_DIR}\n"
            f"hint: stage it first with\n"
            f"  fdl flodl-hf export {SOURCE_REPO} flodl-hf/tests/.exports/bert",
            file=sys.stderr,
        )
        return 1

    torch.set_grad_enabled(False)

    print(f"loading export from {EXPORT_DIR}")
    exported = BertModel.from_pretrained(str(EXPORT_DIR)).eval()

    print(f"loading reference from Hub: {SOURCE_REPO}")
    reference = BertModel.from_pretrained(SOURCE_REPO).eval()
    # Mirror the parity scripts' refs/main pin for the Rust-side cache.
    from huggingface_hub import model_info
    ensure_refs_main(SOURCE_REPO, model_info(SOURCE_REPO).sha)

    input_ids = torch.tensor([[101, 7592, 2088, 102]], dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)

    # Same SDPA backend as parity_bert.py — math is deterministic CPU fp32
    # and matches the backend libtorch picks for these shapes.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out_exp = exported(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        out_ref = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

    h_exp, h_ref = out_exp.last_hidden_state, out_ref.last_hidden_state
    if not torch.equal(h_exp, h_ref):
        d = (h_exp - h_ref).abs()
        print(
            f"FAIL: last_hidden_state differs from Hub reference.\n"
            f"  shape {tuple(h_exp.shape)}\n"
            f"  max abs diff:  {d.max().item():.6e}\n"
            f"  mean abs diff: {d.mean().item():.6e}",
            file=sys.stderr,
        )
        return 1

    p_exp, p_ref = out_exp.pooler_output, out_ref.pooler_output
    if not torch.equal(p_exp, p_ref):
        d = (p_exp - p_ref).abs()
        print(
            f"FAIL: pooler_output differs from Hub reference.\n"
            f"  shape {tuple(p_exp.shape)}\n"
            f"  max abs diff: {d.max().item():.6e}",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: AutoModel.from_pretrained({EXPORT_DIR.name}) bit-exact matches "
        f"{SOURCE_REPO} on last_hidden_state {tuple(h_exp.shape)} "
        f"+ pooler_output {tuple(p_exp.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
