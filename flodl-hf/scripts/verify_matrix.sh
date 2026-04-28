#!/usr/bin/env bash
# Bootstrap shim for `fdl flodl-hf verify-matrix`.
#
# fdl invokes entry: directly via Command::new — no shell — so we
# can't probe for python interpreters in the fdl.yml entry itself.
# This shim runs on the host (no docker:), tries python3 then python,
# and prints an actionable install hint when neither is on PATH.
#
# Forwards argv verbatim to the Python runner, which handles all the
# real work (cell iteration, fdl dispatch, grid printing).

set -e

for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
        exec "$py" "$(dirname "$0")/verify_matrix.py" "$@"
    fi
done

# Bash builtin printf — keeps the error message self-contained even
# when coreutils (cat, dirname, etc.) aren't on PATH.
printf '%s\n' \
    'error: `fdl flodl-hf verify-matrix` requires `python3` (or `python`) on host PATH.' \
    '' \
    'Neither was found. Install Python 3.x via your distro package manager:' \
    '  debian / ubuntu / wsl2:  sudo apt install python3' \
    '  fedora / rhel:           sudo dnf install python3' \
    '  arch:                    sudo pacman -S python' \
    '  macos:                   brew install python3' \
    '' \
    'The host runner only uses the Python stdlib (subprocess, json, argparse) — no' \
    'extra packages needed. Heavy deps (torch, transformers, safetensors) live in' \
    'the hf-parity container, not on the host.' >&2
exit 127
