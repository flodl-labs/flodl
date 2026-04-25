#!/usr/bin/env python3
"""Verify HF `DistilBertModel.from_pretrained(<flodl-export>)` matches
the Hub source bit-exact. DistilBERT has no pooler concept. See
[`_export_verify.run_verify_export`]. Run via
`fdl flodl-hf verify-export-distilbert` after staging.
"""

from __future__ import annotations

from transformers import DistilBertModel

from _export_verify import run_verify_export

if __name__ == "__main__":
    raise SystemExit(
        run_verify_export(
            DistilBertModel,
            repo_id="distilbert/distilbert-base-uncased",
            family_label="distilbert",
            outputs_to_check=("last_hidden_state",),
        )
    )
