#!/usr/bin/env python3
"""Convert a HuggingFace repo's `pytorch_model.bin` to `model.safetensors`.

Older Hub uploads (pre-~2022, most `textattack/*`, `csarron/*`,
`nateraw/*`, etc.) ship only `pytorch_model.bin`. Rust's `hf-hub` crate
can't load that — the Rust ecosystem rightly has no pickle parser
(pickle is an arbitrary-code-execution format; safetensors was invented
to replace it).

This script does the conversion explicitly, in the `hf-parity` Docker
container where `torch` + `safetensors` + `huggingface_hub` are
pre-installed. Output lands in a flodl-managed cache location that
`flodl-hf/src/hub.rs::fetch_safetensors_with_convert_fallback` checks
before the Hub.

Usage (from `fdl flodl-hf convert <repo_id>`):

    python convert_bin_to_safetensors.py nateraw/bert-base-uncased-emotion

Writes:

    /workspace/.hf-cache/flodl-converted/<repo_id>/model.safetensors

Idempotent — re-runs that find an existing converted file skip the
download and conversion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

# Shared with the `dev` container via docker-compose.yml (both services set
# HF_HOME=/workspace/.hf-cache). `flodl-converted` is a flodl-specific
# subdirectory that hf-hub itself never touches, so it won't clash with
# hf-hub's own cache layout at `.hf-cache/hub/`.
CONVERT_ROOT = Path("/workspace/.hf-cache/flodl-converted")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: convert_bin_to_safetensors.py <repo_id>", file=sys.stderr)
        print("  e.g. convert_bin_to_safetensors.py nateraw/bert-base-uncased-emotion",
              file=sys.stderr)
        sys.exit(2)

    repo_id = sys.argv[1]
    out_dir = CONVERT_ROOT / repo_id
    out_file = out_dir / "model.safetensors"

    if out_file.exists():
        print(f"already converted: {out_file}")
        print(f"  size: {out_file.stat().st_size / (1024**2):.1f} MiB")
        return

    print(f"downloading pytorch_model.bin for {repo_id}")
    bin_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    print(f"loading {bin_path}")

    # `weights_only=True` restricts pickle to the safe subset needed for
    # tensor state dicts (no class instantiation, no REDUCE). Available
    # since torch 2.1; the hf-parity image uses torch 2.8 so this is safe
    # to require. A sharded `pytorch_model.bin.index.json` would need a
    # more elaborate loader — not supported here yet.
    state = torch.load(bin_path, map_location="cpu", weights_only=True)

    if not isinstance(state, dict):
        print(
            f"ERROR: expected a tensor state dict, got {type(state).__name__}. "
            "Sharded checkpoints (pytorch_model.bin.index.json) are not yet supported.",
            file=sys.stderr,
        )
        sys.exit(1)

    # safetensors requires contiguous tensors; some state dicts carry
    # views (transposed weights in older models). Force contiguity to
    # make save_file succeed regardless of how the original was saved.
    state = {k: v.contiguous() for k, v in state.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_file))
    size_mib = out_file.stat().st_size / (1024**2)
    print(f"wrote {out_file}")
    print(f"  {len(state)} tensors, {size_mib:.1f} MiB")


if __name__ == "__main__":
    main()
