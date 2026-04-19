#!/bin/sh
# Release-readiness orchestrator.
#
# Runs every ci/release/NN-*.sh in numeric order, captures each
# script's pass/fail, and prints a summary table at the end. Exits
# non-zero iff any script failed.
#
# Individual scripts can also be invoked directly (they each chdir
# to the repo root), so `sh ci/release/03-lint-docs.sh` is fine for
# iterating on a single check.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PASS_LIST=""
FAIL_LIST=""
for script in $(ls -1 [0-9][0-9]-*.sh 2>/dev/null | sort); do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $script"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if sh "./$script"; then
        PASS_LIST="$PASS_LIST $script"
    else
        FAIL_LIST="$FAIL_LIST $script"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for s in $PASS_LIST; do printf '  \033[32mPASS\033[0m  %s\n' "$s"; done
for s in $FAIL_LIST; do printf '  \033[31mFAIL\033[0m  %s\n' "$s"; done

if [ -n "$FAIL_LIST" ]; then
    echo ""
    echo "Release NOT ready. See failures above."
    exit 1
fi

echo ""
echo "All checks passed. Ready to tag."
