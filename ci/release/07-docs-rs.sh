#!/bin/sh
# docs.rs build simulation under the nightly toolchain, run inside the
# `docs-rs` docker service (same flags docs.rs itself uses).
#
# Catches `cfg_attr(docsrs, ...)`-related warnings, missing intra-doc
# links, and nightly-only rustdoc regressions that `fdl ci` won't
# surface on stable.

set -eu
cd "$(git rev-parse --show-toplevel)"

make docs-rs
echo "PASS: docs.rs simulation clean"
