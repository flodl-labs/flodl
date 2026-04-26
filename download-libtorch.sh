#!/bin/sh
# floDl libtorch bootstrap -- fetches `fdl` and hands off to
# `fdl libtorch download`.
#
# This script exists for backwards compatibility with the original
#
#   curl -sL .../download-libtorch.sh | sh -s -- --cuda 12.8
#
# one-liner. New users should install fdl once and use it directly:
#
#   curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl
#   ./fdl install
#   fdl libtorch download --cuda 12.8    # or --cpu, or auto-detect
#
# Every argument is forwarded to `fdl libtorch download` unchanged,
# except the legacy `--project` flag which is dropped -- fdl detects
# project vs. standalone context automatically (in-project installs
# land in ./libtorch/; standalone installs land in $FLODL_HOME/libtorch/).

set -e

REPO="flodl-labs/flodl"

# --- Drop the legacy --project flag ---
# fdl auto-detects the install location, so --project is a no-op.
# Anything else passes through verbatim to `fdl libtorch download`.
FORWARDED=""
for arg; do
    case "$arg" in
        --project)
            echo "note: --project is auto-detected by fdl; dropping it" >&2
            ;;
        *)
            # sh-safe positional re-build: quote each arg for eval set --
            esc=$(printf '%s' "$arg" | sed "s/'/'\\\\''/g")
            FORWARDED="$FORWARDED '$esc'"
            ;;
    esac
done
eval "set -- $FORWARDED"

# --- Local-binary override (used by CI / dev workflows) ---
if [ -n "${FDL_BIN:-}" ] && [ -x "$FDL_BIN" ]; then
    exec "$FDL_BIN" libtorch download "$@"
fi

# --- Fetch a single-use fdl binary ---
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

  No pre-compiled binary for $OS/$ARCH, and cargo is not available.

  Alternative: clone the repo and use fdl directly:
    git clone https://github.com/$REPO.git
    cd flodl && ./fdl libtorch download $*
EOF
    exit 1
fi

# --- Hand off ---
exec "$CLI" libtorch download "$@"
