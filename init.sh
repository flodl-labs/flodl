#!/bin/sh
# floDl bootstrap -- fetches the `fdl` binary and hands off to `fdl init`.
#
# This script exists for backwards compatibility with the original
#
#   curl -sL https://raw.githubusercontent.com/flodl-labs/flodl/main/init.sh | sh -s my-project
#
# one-liner. New users should install fdl once and use it directly
# from then on:
#
#   curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl
#   ./fdl install                       # promote to ~/.local/bin/fdl
#   fdl init <name>                     # interactive: mounted / baked / native
#
# Every argument after <project-name> is forwarded to `fdl init`, so
# `--docker` and `--native` work the same way as running fdl directly.

set -e

REPO="flodl-labs/flodl"

if [ "$#" -eq 0 ]; then
    cat >&2 <<'EOF'
Usage: sh init.sh <project-name> [fdl init flags]
       curl -sL .../init.sh | sh -s <project-name> [flags]

Examples:
       sh init.sh my-project                    # interactive mode pick (mounted default)
       sh init.sh my-project --docker           # libtorch baked into image
       sh init.sh my-project --native           # no Docker, cargo on host
EOF
    exit 1
fi

# --- Fetch a single-use fdl binary ---
#
# $FDL_BIN (optional): skip download and use a pre-existing binary.
# Mainly for local development and CI -- the `test-init` Makefile
# target uses this to exercise the current checkout rather than the
# last-released binary on GitHub.
if [ -n "${FDL_BIN:-}" ] && [ -x "$FDL_BIN" ]; then
    exec "$FDL_BIN" init "$@"
fi

# The ./fdl bootstrap shipped with every scaffold will re-download a
# persistent copy on first use, so this temp binary is disposable.
CLI=$(mktemp "${TMPDIR:-/tmp}/fdl-bootstrap-XXXXXX")
trap 'rm -f "$CLI"' EXIT

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
ARTIFACT="flodl-cli-${OS}-${ARCH}"
case "$OS" in
    *mingw*|*msys*|*cygwin*) ARTIFACT="flodl-cli-windows-x86_64.exe" ;;
esac

LATEST_TAG=""
if command -v curl >/dev/null 2>&1; then
    LATEST_TAG=$(curl -sI "https://github.com/$REPO/releases/latest" 2>/dev/null \
        | grep -i '^location:' | sed 's|.*/||' | tr -d '\r\n')
fi

GOT_CLI=false
if [ -n "$LATEST_TAG" ]; then
    URL="https://github.com/$REPO/releases/download/${LATEST_TAG}/${ARTIFACT}"
    if curl -sfL "$URL" -o "$CLI" 2>/dev/null \
       && chmod +x "$CLI" \
       && "$CLI" --version >/dev/null 2>&1; then
        GOT_CLI=true
    fi
fi

if ! $GOT_CLI && command -v cargo >/dev/null 2>&1; then
    echo "No pre-compiled binary for $OS/$ARCH; building from source..."
    TMPBUILD=$(mktemp -d "${TMPDIR:-/tmp}/flodl-build-XXXXXX")
    trap 'rm -f "$CLI"; rm -rf "$TMPBUILD"' EXIT
    git clone --depth 1 "https://github.com/$REPO.git" "$TMPBUILD/src" >/dev/null 2>&1
    (cd "$TMPBUILD/src" && cargo build --release -p flodl-cli >/dev/null 2>&1)
    cp "$TMPBUILD/src/target/release/fdl" "$CLI"
    chmod +x "$CLI"
    GOT_CLI=true
fi

if ! $GOT_CLI; then
    cat >&2 <<EOF
error: could not obtain fdl binary.

  No pre-compiled binary for $OS/$ARCH, and cargo is not available
  to build from source.

  Alternative: clone the repo and run fdl directly:
    git clone https://github.com/$REPO.git
    cd flodl && ./fdl init $1
EOF
    exit 1
fi

# --- Hand off ---
# fdl init itself offers to install globally at the end, so users land
# on the canonical `fdl <cmd>` workflow without extra round-trips.
exec "$CLI" init "$@"
