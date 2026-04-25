#!/usr/bin/env python3
"""Verify HF `AlbertModel.from_pretrained(<flodl-export>)` matches the
Hub source bit-exact. See [`_export_verify.run_verify_export`].
Run via `fdl flodl-hf verify-export-albert` after staging.
"""

from __future__ import annotations

from transformers import AlbertModel

from _export_verify import run_verify_export

if __name__ == "__main__":
    raise SystemExit(
        run_verify_export(
            AlbertModel,
            repo_id="albert/albert-base-v2",
            family_label="albert",
            outputs_to_check=("last_hidden_state", "pooler_output"),
        )
    )
