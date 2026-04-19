#!/bin/sh
# Verify the working tree is ready to be tagged.
#
# Passes when: no uncommitted changes, and the tag named after the
# workspace version does not already exist. Warns (doesn't fail) on
# untracked files or a non-main branch -- those are common during
# release prep (e.g. a release-0.x.y branch) but worth surfacing.

set -eu
cd "$(git rev-parse --show-toplevel)"

FAIL=0

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "FAIL: uncommitted changes in working tree"
    git status --short | sed 's/^/  /'
    FAIL=1
fi

UNTRACKED=$(git ls-files --others --exclude-standard)
if [ -n "$UNTRACKED" ]; then
    echo "WARN: untracked files (review before tagging):"
    echo "$UNTRACKED" | sed 's/^/  /'
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo "WARN: not on main (current branch: $BRANCH)"
fi

VERSION=$(awk -F '"' '/^version *=/ { print $2; exit }' Cargo.toml)
if [ -z "$VERSION" ]; then
    echo "FAIL: could not parse workspace version from Cargo.toml"
    FAIL=1
elif git rev-parse "$VERSION" >/dev/null 2>&1; then
    echo "FAIL: tag $VERSION already exists (bump Cargo.toml)"
    FAIL=1
else
    echo "INFO: target tag $VERSION is free"
fi

[ "$FAIL" = 0 ] && echo "PASS: git state clean for $VERSION"
exit "$FAIL"
