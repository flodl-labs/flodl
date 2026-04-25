#!/usr/bin/env python3
"""Verify HF `XLMRobertaModel.from_pretrained(<flodl-export>)` matches
the Hub source bit-exact. See [`_export_verify.run_verify_export`].
Run via `fdl flodl-hf verify-export-xlm-roberta` after staging.
"""

from __future__ import annotations

from transformers import XLMRobertaModel

from _export_verify import run_verify_export

if __name__ == "__main__":
    raise SystemExit(
        run_verify_export(
            XLMRobertaModel,
            repo_id="FacebookAI/xlm-roberta-base",
            family_label="xlm-roberta",
            # Hub `FacebookAI/xlm-roberta-base` ships pooler weights
            # (unlike `roberta-base`); the export preserves them, so
            # both encoder and pooler outputs round-trip bit-exact.
            outputs_to_check=("last_hidden_state", "pooler_output"),
        )
    )
