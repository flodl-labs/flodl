"""Shared logic for `verify_export_<family>.py` scripts.

Each family verifier is a 5-line shim that calls
[`run_verify_export`] with its own torch model class, repo id, and the
expected output attributes. The Python and torch heavy-lifting is
identical across families — only the model class and tokenizer
specifics differ.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
from huggingface_hub import model_info
from transformers import AutoTokenizer

from _hf_cache_utils import ensure_refs_main


def run_verify_export(
    model_cls,
    repo_id: str,
    family_label: str,
    *,
    outputs_to_check: Sequence[str] = ("last_hidden_state",),
    prompt: str = "fab2s writes Rust",
    extra_kwargs: dict | None = None,
) -> int:
    """Verify HF `model_cls.from_pretrained(<flodl_export>)` matches
    `model_cls.from_pretrained(<repo_id>)` bit-exact on the same input.

    `outputs_to_check` are attribute names on the model output (e.g.
    `last_hidden_state`, `pooler_output`). Each is asserted via
    `torch.equal`. Returns 0 on full match, 1 on first failure.

    `extra_kwargs` are passed through to both `from_pretrained` calls
    (e.g. `add_pooling_layer=False` for repos that don't ship a pooler).
    Forwarded to BOTH sides so the comparison stays apples-to-apples.
    """
    extra_kwargs = extra_kwargs or {}
    export_dir = Path(__file__).resolve().parents[1] / "tests" / ".exports" / family_label

    if not export_dir.exists():
        print(
            f"error: export dir not found at {export_dir}\n"
            f"hint: stage it first with\n"
            f"  fdl flodl-hf export {repo_id} flodl-hf/tests/.exports/{family_label}",
            file=sys.stderr,
        )
        return 1

    torch.set_grad_enabled(False)

    print(f"loading export from {export_dir}")
    exported = model_cls.from_pretrained(str(export_dir), **extra_kwargs).eval()

    print(f"loading reference from Hub: {repo_id}")
    reference = model_cls.from_pretrained(repo_id, **extra_kwargs).eval()
    ensure_refs_main(repo_id, model_info(repo_id).sha)

    tok = AutoTokenizer.from_pretrained(repo_id)
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(torch.int64)
    attention_mask = enc["attention_mask"].to(torch.int64)
    fwd_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "token_type_ids" in enc:
        fwd_kwargs["token_type_ids"] = enc["token_type_ids"].to(torch.int64)

    # Force math SDPA backend — deterministic on CPU fp32 and matches the
    # backend libtorch picks for these shapes (same as parity_*.py).
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out_exp = exported(**fwd_kwargs)
        out_ref = reference(**fwd_kwargs)

    for name in outputs_to_check:
        a = getattr(out_exp, name)
        b = getattr(out_ref, name)
        if not torch.equal(a, b):
            d = (a - b).abs()
            print(
                f"FAIL: {name} differs from Hub reference.\n"
                f"  shape {tuple(a.shape)}\n"
                f"  max abs diff:  {d.max().item():.6e}\n"
                f"  mean abs diff: {d.mean().item():.6e}",
                file=sys.stderr,
            )
            return 1

    summary = ", ".join(
        f"{n} {tuple(getattr(out_exp, n).shape)}" for n in outputs_to_check
    )
    print(
        f"OK: {model_cls.__name__}.from_pretrained({export_dir.name}) "
        f"bit-exact matches {repo_id} on {summary}"
    )
    return 0
