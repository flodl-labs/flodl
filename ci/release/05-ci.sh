#!/bin/sh
# Delegate to `fdl ci`: cargo build + cargo test + cargo clippy +
# rustdoc strict. Matches the CI CPU job, so a clean run here means
# the release branch will land green on GitHub.

set -eu
cd "$(git rev-parse --show-toplevel)"

if ! command -v fdl >/dev/null 2>&1; then
    echo "FAIL: fdl not on PATH"
    exit 1
fi

fdl ci
echo "PASS: fdl ci clean"
