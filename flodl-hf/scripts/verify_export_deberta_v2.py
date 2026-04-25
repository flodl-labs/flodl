#!/usr/bin/env python3
"""Verify HF `DebertaV2Model.from_pretrained(<flodl-export>)` matches
the Hub source bit-exact. See [`_export_verify.run_verify_export`].
Run via `fdl flodl-hf verify-export-deberta-v2` after staging.
"""

from __future__ import annotations

from transformers import DebertaV2Model

from _export_verify import run_verify_export

if __name__ == "__main__":
    raise SystemExit(
        run_verify_export(
            DebertaV2Model,
            repo_id="microsoft/deberta-v3-base",
            family_label="deberta-v2",
            outputs_to_check=("last_hidden_state",),
        )
    )
