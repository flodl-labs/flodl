"""Shared HuggingFace cache helpers for flodl-hf parity scripts.

Keeps the parity scripts self-contained (no extra Cargo/pip dep for
shared utilities) while de-duplicating the one hf-hub cache quirk we
need to work around.
"""

from __future__ import annotations

import os
from pathlib import Path


def ensure_refs_main(repo_id: str, sha: str) -> None:
    """Ensure `$HF_HOME/hub/models--<org>--<repo>/refs/main` holds `sha`.

    Python's `huggingface_hub` only writes `refs/main` when called with
    `revision="main"` (the default) or no revision. When parity scripts
    pin to an explicit SHA for fixture reproducibility, Python skips
    `refs/main` — but Rust's `hf-hub::Cache::get()` reads `refs/main`
    first and returns `None` on miss, triggering a re-download even
    though the actual blobs and snapshots are fully cached on disk.

    Writing `refs/main` ourselves closes that gap without giving up
    SHA pinning on the Python side.
    """
    hf_home = os.environ.get("HF_HOME")
    hub_root = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
    repo_folder = "models--" + repo_id.replace("/", "--")
    refs_main = hub_root / repo_folder / "refs" / "main"
    refs_main.parent.mkdir(parents=True, exist_ok=True)
    refs_main.write_text(sha.strip())
