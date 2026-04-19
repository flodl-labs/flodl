#!/bin/sh
# `cargo publish --dry-run` for every workspace crate that is published
# to crates.io, in dependency order.
#
# Runs inside the `dev` docker service so libtorch is available to the
# build step: `flodl-sys` and `flodl` both link against libtorch, and a
# host without `LIBTORCH` exported would fail at `build.rs` or link
# time. Running all four crates in docker keeps the environment
# uniform and close to what crates.io / docs.rs sees.
#
# Catches: missing package metadata, stale `path = "../foo"` deps
# without `version = "..."` companions, oversized crates, uncommitted
# file rejection, link-time breakage.
#
# This does NOT actually publish -- dry-run stops right before upload.

set -u
cd "$(git rev-parse --show-toplevel)"

# Mirror flodl-cli/src/run.rs::libtorch_env -- docker-compose.yml uses
# LIBTORCH_CPU_PATH (always) and LIBTORCH_HOST_PATH / CUDA_VERSION /
# CUDA_TAG (when an active CUDA variant exists) to pick mount points
# and image tags. Exporting them here gives docker-compose the same
# resolved state that `fdl build` would see.
ACTIVE=$(tr -d '[:space:]' < libtorch/.active 2>/dev/null || true)
export LIBTORCH_CPU_PATH="./libtorch/precompiled/cpu"
if [ -n "$ACTIVE" ]; then
    export LIBTORCH_HOST_PATH="./libtorch/$ACTIVE"
    ARCH_CUDA=$(grep '^cuda=' "./libtorch/$ACTIVE/.arch" 2>/dev/null | cut -d= -f2 || true)
    if [ -n "$ARCH_CUDA" ] && [ "$ARCH_CUDA" != "none" ]; then
        case "$ARCH_CUDA" in
            *.*.*) CUDA_VERSION="$ARCH_CUDA" ;;
            *)     CUDA_VERSION="$ARCH_CUDA.0" ;;
        esac
        CUDA_TAG=$(echo "$CUDA_VERSION" | cut -d. -f1,2)
        export CUDA_VERSION CUDA_TAG
    fi
fi

# Dependency order (leaves first). ddp-bench and benchmarks are
# workspace-internal only, never published.
CRATES="flodl-sys flodl-cli-macros flodl flodl-cli"

FAIL=0
for crate in $CRATES; do
    if [ ! -d "$crate" ]; then
        echo "WARN: $crate directory missing, skipping"
        continue
    fi
    echo ""
    echo "=== $crate (in docker dev) ==="
    if ! docker compose run --rm -T dev cargo publish --dry-run -p "$crate"; then
        echo "FAIL: $crate failed cargo publish --dry-run"
        FAIL=1
    fi
done

echo ""
[ "$FAIL" = 0 ] && echo "PASS: all published crates pass cargo publish --dry-run (docker dev)"
exit "$FAIL"
