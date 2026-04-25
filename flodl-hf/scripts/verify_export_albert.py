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
            # ALBERT pooler naming gap: HF stores the pooler as a flat
            # Linear (`albert.pooler.{weight,bias}`), but flodl-hf's
            # AlbertPooler wraps the Linear in a `dense` field, producing
            # `albert.pooler.dense.{weight,bias}` — a key set HF doesn't
            # recognize. Until that's restructured, only the encoder
            # round-trips through AutoModel reload. Tracked as follow-up.
            extra_kwargs={"add_pooling_layer": False},
            outputs_to_check=("last_hidden_state",),
        )
    )
