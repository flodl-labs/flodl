#!/usr/bin/env python3
"""Verify HF `RobertaModel.from_pretrained(<flodl-export>)` matches the
Hub source bit-exact. See [`_export_verify.run_verify_export`].
Run via `fdl flodl-hf verify-export-roberta` after staging.
"""

from __future__ import annotations

from transformers import RobertaModel

from _export_verify import run_verify_export

if __name__ == "__main__":
    raise SystemExit(
        run_verify_export(
            RobertaModel,
            repo_id="roberta-base",
            family_label="roberta",
            # roberta-base on Hub ships only the encoder — no pooler
            # weights. Pass `add_pooling_layer=False` on both sides so
            # neither side instantiates a random pooler that would
            # produce non-reproducible `pooler_output`.
            extra_kwargs={"add_pooling_layer": False},
            outputs_to_check=("last_hidden_state",),
        )
    )
