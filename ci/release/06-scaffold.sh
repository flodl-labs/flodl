#!/bin/sh
# End-to-end check of the `init.sh` -> `fdl init` scaffolding path.
# Builds a fresh flodl-cli, scaffolds a --docker project, asserts that
# every expected file is present, and validates the generated
# docker-compose.yml.

set -eu
cd "$(git rev-parse --show-toplevel)"

make test-init
echo "PASS: make test-init clean"
