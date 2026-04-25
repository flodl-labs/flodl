#!/usr/bin/env python3
"""Verify HF `BertModel.from_pretrained(<flodl-export>)` matches the Hub
source bit-exact on a fixed input. See [`_export_verify.run_verify_export`]
for the shared pipeline. Run via `fdl flodl-hf verify-export-bert` after
staging the export with `fdl flodl-hf export bert-base-uncased
flodl-hf/tests/.exports/bert`.
"""

from __future__ import annotations

from transformers import BertModel

from _export_verify import run_verify_export

if __name__ == "__main__":
    raise SystemExit(
        run_verify_export(
            BertModel,
            repo_id="bert-base-uncased",
            family_label="bert",
            # bert-base-uncased ships pooler weights — verify both
            # encoder and pooler outputs round-trip bit-exact.
            outputs_to_check=("last_hidden_state", "pooler_output"),
        )
    )
