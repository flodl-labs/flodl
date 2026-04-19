#!/bin/sh
# Verify CHANGELOG.md has a dated entry for the current workspace version.
#
# Matches `## [X.Y.Z] - YYYY-MM-DD`. Catches the two classic mistakes:
#   - Bumped Cargo.toml but forgot to move `[Unreleased]` to `[X.Y.Z]`.
#   - Added the header but forgot the date.

set -eu
cd "$(git rev-parse --show-toplevel)"

VERSION=$(awk -F '"' '/^version *=/ { print $2; exit }' Cargo.toml)

if ! grep -qE "^## \[$VERSION\] - [0-9]{4}-[0-9]{2}-[0-9]{2}\b" CHANGELOG.md; then
    echo "FAIL: CHANGELOG.md has no '## [$VERSION] - YYYY-MM-DD' header"
    echo "  Cargo.toml version: $VERSION"
    echo "  CHANGELOG headers found (top 3):"
    grep -E '^## \[' CHANGELOG.md | head -3 | sed 's/^/    /'
    exit 1
fi

echo "PASS: CHANGELOG has a dated entry for $VERSION"
