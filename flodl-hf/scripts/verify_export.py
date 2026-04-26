#!/usr/bin/env python3
"""Generic flodl-export verifier.

Loads `<dir>/model.safetensors` + `<dir>/config.json` through the
matching HF `AutoModelFor*` class, then optionally compares against a
Hub source. Two modes:

- **Default** — load the export, plus load the same checkpoint a
  second time directly from the Hub source repo, then assert both
  loadability AND bit-exact forward parity for a fixed prompt.
- **`--no-hub-source`** — load the export, run loadability only.
  Required when the export has no Hub equivalent (fine-tuned model
  exported from a flodl checkpoint, custom training data, etc.).

Loadability is always checked on the staged export — zero
`missing_keys` / `unexpected_keys` from `from_pretrained`. That alone
catches most parameter-naming drift (head naming divergence, dropped
pooler keys, etc.). Forward parity adds bit-exact value validation but
needs a comparable upstream.

Bytes-identity (`model.safetensors` byte-equality vs Hub) is already
covered by the Rust `_live` head-roundtrip tests in
`flodl-hf/tests/{family}_head_export_roundtrip.rs`; this script picks
up where the Rust side stops.

Auto-detection
--------------
The dispatch key is `(config.model_type, config.architectures[0])`:

| `architectures[0]` suffix       | `AutoModelFor*`                       |
|---------------------------------|---------------------------------------|
| `*Model` (no `For`)             | `AutoModel`                           |
| `*ForSequenceClassification`    | `AutoModelForSequenceClassification`  |
| `*ForTokenClassification`       | `AutoModelForTokenClassification`     |
| `*ForQuestionAnswering`         | `AutoModelForQuestionAnswering`       |
| `*ForMaskedLM`                  | `AutoModelForMaskedLM`                |

Suffix match (not exact match) — family prefix is irrelevant. Mirrors
`flodl_hf::export::classify_architecture` on the Rust side.

Hub source recovery
-------------------
Order: `--hub-source` flag > `flodl_source_repo` (stamped by
`export --hub`) > `_name_or_path` (rarely present in canonical configs)
> loud error pointing at `--hub-source` and `--no-hub-source` as
options.

Usage
-----
    python verify_export.py <dir>                          # full
    python verify_export.py <dir> --hub-source <repo>      # explicit
    python verify_export.py <dir> --no-hub-source          # loadability only

Run via `fdl flodl-hf verify-export <dir>` (which routes this through
the `hf-parity` container).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import torch
from huggingface_hub import model_info
from safetensors import safe_open
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from _hf_cache_utils import ensure_refs_main


# Architecture suffix → (AutoModelFor* class, output attrs to compare).
# Order matters: `Model` is the catch-all and must come last (it
# matches both `*Model` and any `For{Other}` not explicitly handled).
DISPATCH: list[tuple[str, type, tuple[str, ...]]] = [
    ("ForSequenceClassification", AutoModelForSequenceClassification, ("logits",)),
    ("ForTokenClassification", AutoModelForTokenClassification, ("logits",)),
    ("ForQuestionAnswering", AutoModelForQuestionAnswering, ("start_logits", "end_logits")),
    ("ForMaskedLM", AutoModelForMaskedLM, ("logits",)),
    # `AutoModel` (base) — the suffix entry stays generic so any
    # `*Model` class name lands here. Pooler handling is decided per
    # checkpoint via safetensors-key inspection in `pick_extra_kwargs`.
    ("Model", AutoModel, ("last_hidden_state",)),
]


def dispatch_for(architecture: str) -> tuple[type, tuple[str, ...]]:
    """Map an HF architecture class name (e.g.
    `BertForSequenceClassification`) to `(AutoModelFor*, output_attrs)`.

    Suffix match — same family-agnostic logic as
    `flodl_hf::export::classify_architecture`. Unknown `For{Other}`
    suffixes raise loudly with the supported set.
    """
    if "For" in architecture:
        for suffix, cls, outputs in DISPATCH:
            if suffix == "Model":
                continue  # catch-all, handled below
            if architecture.endswith(suffix):
                return cls, outputs
        raise SystemExit(
            f"verify-export: unsupported architecture {architecture!r}.\n"
            f"  flodl-hf currently dispatches "
            f"{{Model, ForSequenceClassification, ForTokenClassification, "
            f"ForQuestionAnswering, ForMaskedLM}}.\n"
            f"  Other heads (NextSentencePrediction, MultipleChoice, "
            f"Pretraining, ...) are not yet wired."
        )
    # Bare `*Model` or any non-`For*` name — treat as base backbone.
    return AutoModel, ("last_hidden_state",)


def resolve_path(arg: str) -> Path:
    """Anchor a relative path against `FDL_PROJECT_ROOT` when set.

    `fdl` injects this env var inside docker-compose-managed services
    so argv paths resolve from the host shell's invocation root
    regardless of the container's `working_dir` (the hf-parity service
    runs in `/workspace/flodl-hf`, so a naive relative
    `flodl-hf/tests/.exports/bert` would otherwise become
    `/workspace/flodl-hf/flodl-hf/tests/.exports/bert`). Mirrors the
    `resolve_path` helper in `examples/export_hf.rs`. Absolute paths
    and host-side runs are unaffected.
    """
    p = Path(arg)
    if p.is_absolute():
        return p
    root = os.environ.get("FDL_PROJECT_ROOT")
    if root:
        return Path(root) / p
    return p


def safetensors_keys(path: Path) -> set[str]:
    """Return the parameter / buffer key set of a safetensors file
    without allocating the tensor data.
    """
    with safe_open(str(path), framework="pt") as f:
        return set(f.keys())


def pick_extra_kwargs(
    cls: type,
    keys: set[str],
) -> tuple[dict, tuple[str, ...]]:
    """Decide `from_pretrained` kwargs and the output attrs to compare.

    Only `AutoModel` (base backbone) needs runtime adjustment:
    pooler-bearing checkpoints get `pooler_output` added to the
    comparison; pooler-less ones pass `add_pooling_layer=False` on
    BOTH sides so neither instantiates a random pooler that would
    break bit-exact comparison.

    Task heads (`AutoModelFor*`) carry their own pooler / classifier
    wiring inside the head; the comparison stays on the head's primary
    output(s) and `extra_kwargs` is empty.
    """
    if cls is not AutoModel:
        return {}, dispatch_for_cls_outputs(cls)
    # Mirror `flodl_hf::export::keys_have_pooler`: suffix-match the four
    # family pooler key shapes so we catch BERT-family checkpoints that
    # carry the `<family>.` prefix (e.g. `bert.pooler.dense.weight` for
    # `bert-base-uncased`, where the source class is `BertForPreTraining`)
    # as well as ALBERT's flat `pooler.{weight,bias}` shape.
    pooler_suffixes = (
        "pooler.dense.weight",
        "pooler.dense.bias",
        "pooler.weight",
        "pooler.bias",
    )
    has_pooler = any(k.endswith(s) for k in keys for s in pooler_suffixes)
    if has_pooler:
        return {}, ("last_hidden_state", "pooler_output")
    # `add_pooling_layer` exists on BERT-family AutoModel classes
    # (BertModel, RobertaModel, DistilBertModel, XLMRobertaModel,
    # AlbertModel) but not on DeBERTa-v2's `DebertaV2Model`, which is
    # pooler-less by design. Probe the constructor signature so we
    # only pass the kwarg where it's accepted.
    import inspect
    try:
        params = inspect.signature(cls.__init__).parameters
    except (ValueError, TypeError):
        params = {}
    if "add_pooling_layer" in params:
        return {"add_pooling_layer": False}, ("last_hidden_state",)
    return {}, ("last_hidden_state",)


def dispatch_for_cls_outputs(cls: type) -> tuple[str, ...]:
    """Look up the comparison-output attrs for a non-`AutoModel`
    dispatch class. Used by `pick_extra_kwargs` to short-circuit the
    pooler logic for task heads.
    """
    for _, c, outputs in DISPATCH:
        if c is cls:
            return outputs
    raise RuntimeError(f"unreachable: dispatch class {cls!r} has no outputs entry")


def resolve_hub_source(
    cli_override: str | None,
    config: dict,
) -> str:
    """Recover the Hub source repo for the export directory.

    Order: `--hub-source` flag wins; else `flodl_source_repo` (stamped
    by `fdl flodl-hf export --hub`); else `_name_or_path` (rare in
    canonical configs since `to_json_str()` doesn't emit it); else
    abort with a hint pointing at `--hub-source` and `--no-hub-source`.
    """
    if cli_override:
        return cli_override
    for key in ("flodl_source_repo", "_name_or_path"):
        val = config.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    raise SystemExit(
        "verify-export: no Hub source repo recorded in config.json.\n"
        "  Pass --hub-source <repo> for explicit recovery, e.g.\n"
        "    fdl flodl-hf verify-export <dir> --hub-source bert-base-uncased\n"
        "  Or pass --no-hub-source to skip forward parity (loadability\n"
        "  only — the right choice for fine-tuned exports with no Hub\n"
        "  equivalent).\n"
        "  (Exports produced by `fdl flodl-hf export --hub` since the\n"
        "  flodl_source_repo stamp landed do not need this flag.)"
    )


def make_inputs(repo_id: str, prompt: str) -> dict:
    """Tokenize `prompt` with the Hub source's tokenizer and return the
    forward-kwargs both sides will receive verbatim.

    Includes `token_type_ids` only when the tokenizer emits them
    (BERT family does, RoBERTa / XLM-R / DistilBERT typically don't).
    """
    tok = AutoTokenizer.from_pretrained(repo_id)
    enc = tok(prompt, return_tensors="pt")
    fwd = {
        "input_ids": enc["input_ids"].to(torch.int64),
        "attention_mask": enc["attention_mask"].to(torch.int64),
    }
    if "token_type_ids" in enc:
        fwd["token_type_ids"] = enc["token_type_ids"].to(torch.int64)
    return fwd


def report_load(model_kind: str, repo_or_dir: str, info) -> None:
    """Report a `from_pretrained` load — fail loudly on any unexpected
    or missing keys, since either points at parameter-name drift the
    Rust roundtrip tests would have missed.

    Exception: when loading the bare `AutoModel`, missing `pooler.*`
    keys are tolerated. Some pretraining checkpoints (notably
    `roberta-base`) ship encoder-only safetensors and HF Python
    initialises the pooler from scratch with a "newly initialized"
    warning rather than a load failure. flodl's base export mirrors
    this faithfully (no random pooler weights baked in), so the
    "missing" set lines up with what HF would report against the
    upstream repo itself; treating it as a fail would punish an
    accurate round-trip. Unexpected keys remain a hard fail.

    `info` is the `_LoadingInfo` namedtuple-like dict returned by
    `from_pretrained(..., output_loading_info=True)`.
    """
    missing = info.get("missing_keys") or []
    unexpected = info.get("unexpected_keys") or []
    benign_missing: list[str] = []
    if model_kind == "AutoModel":
        kept = []
        for k in missing:
            if k.startswith("pooler."):
                benign_missing.append(k)
            else:
                kept.append(k)
        missing = kept
    if missing or unexpected:
        raise SystemExit(
            f"FAIL loadability ({model_kind} <- {repo_or_dir}):\n"
            f"  {len(missing)} missing key(s): {missing[:10]}\n"
            f"  {len(unexpected)} unexpected key(s): {unexpected[:10]}"
        )
    if benign_missing:
        print(
            f"PASS loadability ({model_kind} <- {repo_or_dir}) "
            f"[tolerating {len(benign_missing)} pooler key(s) "
            f"not present in source: {benign_missing}]"
        )
    else:
        print(f"PASS loadability ({model_kind} <- {repo_or_dir})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a flodl export round-trips through HF "
            "AutoModelFor*.from_pretrained bit-exact vs the Hub source. "
            "Auto-detects family + head from <dir>/config.json."
        ),
    )
    parser.add_argument(
        "dir",
        help="Staged export directory from `fdl flodl-hf export --hub <repo> --out <dir>`",
    )
    parser.add_argument(
        "--hub-source",
        metavar="REPO",
        help=(
            "Override the Hub source repo (else read from "
            "config.json's flodl_source_repo / _name_or_path)."
        ),
    )
    parser.add_argument(
        "--no-hub-source",
        action="store_true",
        help=(
            "Skip the Hub-side load and forward parity check. "
            "Loadability is still verified on the staged export. "
            "Use for fine-tuned exports that have no Hub equivalent "
            "(e.g. `fdl flodl-hf export --checkpoint trained.fdl`)."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="fab2s writes Rust",
        help="Forward-parity prompt (same string fed to both sides).",
    )
    args = parser.parse_args()
    if args.no_hub_source and args.hub_source:
        parser.error("--no-hub-source and --hub-source are mutually exclusive")

    export_dir = resolve_path(args.dir).resolve()
    config_path = export_dir / "config.json"
    model_path = export_dir / "model.safetensors"
    for p in (config_path, model_path):
        if not p.exists():
            print(f"error: {p} not found", file=sys.stderr)
            return 1

    config = json.loads(config_path.read_text())
    architectures: Sequence[str] = config.get("architectures") or []
    if not architectures:
        print(
            f"error: config.json missing `architectures` field. "
            f"flodl exports always emit it; was this dir produced by "
            f"`fdl flodl-hf export`?",
            file=sys.stderr,
        )
        return 1
    architecture = architectures[0]
    model_type = config.get("model_type", "<unset>")

    cls, _ = dispatch_for(architecture)
    keys = safetensors_keys(model_path)
    extra_kwargs, outputs_to_check = pick_extra_kwargs(cls, keys)
    repo_id = None if args.no_hub_source else resolve_hub_source(args.hub_source, config)

    hub_label = repo_id if repo_id is not None else "<skipped: --no-hub-source>"
    outputs_label = (
        outputs_to_check if repo_id is not None else "<skipped: --no-hub-source>"
    )
    print(
        f"verify-export {export_dir}\n"
        f"  model_type={model_type} architecture={architecture} "
        f"-> {cls.__name__}\n"
        f"  hub source: {hub_label}\n"
        f"  outputs to compare: {outputs_label}"
    )

    torch.set_grad_enabled(False)

    # Force eager (non-meta) initialisation. flodl's MLM exports omit
    # the tied `cls.predictions.decoder.weight` (it aliases the word
    # embedding); HF's default `from_pretrained` path on torch 2.x
    # ships some Linear weights via meta-device lazy init, and the
    # tying step doesn't always materialise them, tripping a "Tensor
    # on device meta" error in forward. `low_cpu_mem_usage=False`
    # disables the meta-init path; tying then happens against real
    # CPU tensors.
    print(f"loading export from {export_dir} ...")
    exported, info_exp = cls.from_pretrained(
        str(export_dir),
        output_loading_info=True,
        low_cpu_mem_usage=False,
        **extra_kwargs,
    )
    exported = exported.eval()
    if hasattr(exported, "tie_weights"):
        exported.tie_weights()
    report_load(cls.__name__, str(export_dir), info_exp)

    if repo_id is None:
        # Loadability-only mode. The export round-trips through HF
        # `from_pretrained` cleanly; we cannot run forward parity
        # without a comparable upstream.
        print("SKIP forward parity (--no-hub-source)")
        print(f"verify-export OK (loadability only) {export_dir}")
        return 0

    print(f"loading reference from Hub: {repo_id} ...")
    reference = cls.from_pretrained(
        repo_id,
        low_cpu_mem_usage=False,
        **extra_kwargs,
    ).eval()
    if hasattr(reference, "tie_weights"):
        reference.tie_weights()
    # Match flodl's hf-hub cache shape so subsequent Rust loads
    # don't re-download (parity scripts do the same dance).
    ensure_refs_main(repo_id, model_info(repo_id).sha)
    # Loadability is asserted only on the staged export — the Hub
    # source typically ships a head-class checkpoint (e.g.
    # `bert-base-uncased` is `BertForPreTraining`), so loading it as
    # `AutoModel` naturally drops MLM / NSP head keys. That's
    # pre-existing baseline noise on the upstream side, not flodl
    # drift.

    fwd_kwargs = make_inputs(repo_id, args.prompt)

    # Force math SDPA backend — deterministic on CPU fp32 and matches
    # the backend libtorch picks for these shapes (mirrors parity_*.py
    # and the previous per-family verify_export_*.py scripts).
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out_exp = exported(**fwd_kwargs)
        out_ref = reference(**fwd_kwargs)

    for name in outputs_to_check:
        a = getattr(out_exp, name, None)
        b = getattr(out_ref, name, None)
        if a is None or b is None:
            print(
                f"FAIL forward: output {name!r} missing on "
                f"{'export' if a is None else 'reference'} side",
                file=sys.stderr,
            )
            return 1
        if not torch.equal(a, b):
            d = (a - b).abs()
            print(
                f"FAIL forward: {name} differs from Hub reference.\n"
                f"  shape {tuple(a.shape)}\n"
                f"  max abs diff:  {d.max().item():.6e}\n"
                f"  mean abs diff: {d.mean().item():.6e}",
                file=sys.stderr,
            )
            return 1
        print(
            f"PASS forward {name} {tuple(a.shape)} bit-exact "
            f"(max abs diff = 0)"
        )

    print(f"verify-export OK {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
