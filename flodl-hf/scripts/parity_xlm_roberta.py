#!/usr/bin/env python3
"""Generate the xlm-roberta-base reference fixture for flodl parity tests.

Runs HuggingFace `XLMRobertaModel` on a pinned multilingual input and
writes both the tokenised inputs and the reference outputs to
`flodl-hf/tests/fixtures/xlm_roberta_base_parity.safetensors`.

XLM-RoBERTa shares RoBERTa's embedding shape and computes
`position_ids` internally from `input_ids` using the padding-offset
convention, so the fixture does NOT save them — the flodl graph
follows the same path.

The prompt is row 0 of FLORES-200 devtest concatenated across 7
scripts (Latin/Chinese/Arabic/Cyrillic/Devanagari/Japanese/Korean),
chosen to exercise the multilingual SentencePiece vocabulary in a
single forward pass. Pinning to FLORES-200 devtest row 0 keeps the
fixture reproducible across runs.

Run via `fdl flodl-hf parity xlm-roberta`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import model_info
from safetensors.torch import save_file
from transformers import AutoTokenizer, XLMRobertaModel

from _hf_cache_utils import ensure_refs_main

MODEL_ID = "xlm-roberta-base"
REVISION: str | None = None

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "xlm_roberta_base_parity.safetensors"
)

# FLORES-200 language codes covering 7 distinct scripts.
FLORES_LANGS = [
    "eng_Latn",
    "zho_Hans",
    "arb_Arab",
    "rus_Cyrl",
    "hin_Deva",
    "jpn_Jpan",
    "kor_Hang",
]
FLORES_SPLIT = "devtest"
FLORES_ROW = 0


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


def build_prompt() -> str:
    pieces = []
    for lang in FLORES_LANGS:
        # `facebook/flores` ships a loader script that unzips the
        # release tarball — `trust_remote_code=True` runs it. This is
        # contained to the parity Docker image, which is rebuilt from
        # `Dockerfile.parity` and never executes user input.
        ds = load_dataset(
            "facebook/flores", lang, split=FLORES_SPLIT,
            trust_remote_code=True,
        )
        pieces.append(ds[FLORES_ROW]["sentence"])
    return " ".join(pieces)


def main() -> None:
    torch.set_grad_enabled(False)

    sha = REVISION or model_info(MODEL_ID).sha
    print(f"using {MODEL_ID} @ {sha}")

    prompt = build_prompt()
    print(f"  prompt: {len(prompt)} chars across {len(FLORES_LANGS)} scripts")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=sha)
    # Pooler enabled (HF default): matches the with-pooler graph
    # `XlmRobertaModel::from_pretrained` returns dynamically when the
    # checkpoint ships pooler weights. `xlm-roberta-base` does ship
    # them as `roberta.pooler.dense.{weight,bias}`.
    model = XLMRobertaModel.from_pretrained(MODEL_ID, revision=sha).eval()
    ensure_refs_main(MODEL_ID, sha)

    enc = tok(prompt, return_tensors="pt")
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
            # position_ids intentionally omitted — XLMRobertaModel
            # computes them internally, and the flodl graph does the
            # same.
        )

    tensors = {
        "inputs.input_ids":         input_ids,
        "inputs.token_type_ids":    token_type_ids,
        "inputs.attention_mask":    attention_mask,
        "inputs.sequence_ids":      sequence_ids,
        "outputs.last_hidden_state": out.last_hidden_state.contiguous(),
        "outputs.pooler_output":    out.pooler_output.contiguous(),
    }
    metadata = {
        "source_model":   MODEL_ID,
        "source_sha":     sha,
        "torch_version":  torch.__version__,
        "sdpa_backend":   "math",
        "flores_split":   FLORES_SPLIT,
        "flores_row":     str(FLORES_ROW),
        "flores_langs":   ",".join(FLORES_LANGS),
        "prompt":         prompt,
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
