#!/bin/sh
# Prints the Rust version every lint gate runs on.
#
# The gates are `-D warnings` clippy runs, and clippy grows lints with
# each release, so a gate on floating `stable` goes red with no commit
# behind it, and two gates on different releases disagree about the same
# tree. The version is pinned instead, and moving it is a commit.
#
# It lives in the Dockerfiles, as `ARG RUST_VERSION=<x.y.z>`, because the
# published CI images are tagged by their Dockerfile's content hash (see
# .github/workflows/images.yml): a bump has to change those bytes for a
# new image to be built and published. The workflows read it from here
# so that the gates outside a container follow the same value.
#
# Fails when a Dockerfile carries no pin or when they disagree, which is
# what makes this the consistency check as well as the reader.
#
# To bump: change the ARG in every file listed below, rebuild the local
# images (`docker compose build dev`, and the GPU one in use), run
# `fdl clippy`, fix what the new release flags, and open the PR.
set -eu

cd "$(dirname "$0")/.."

PIN=""
for f in Dockerfile Dockerfile.cuda Dockerfile.rocm Dockerfile.rhel; do
    v=$(sed -n 's/^ARG RUST_VERSION=//p' "$f" | tr -d '\r')
    case "$v" in
        [0-9]*.[0-9]*.[0-9]*) ;;
        *)
            echo "rust-pin: $f has no 'ARG RUST_VERSION=<x.y.z>' line (found '$v')" >&2
            exit 1
            ;;
    esac
    if [ -z "$PIN" ]; then
        PIN=$v
    elif [ "$v" != "$PIN" ]; then
        echo "rust-pin: $f pins $v but Dockerfile pins $PIN; they must agree" >&2
        exit 1
    fi
done

printf '%s\n' "$PIN"
