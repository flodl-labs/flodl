#!/bin/sh
# Verify everything that names the workspace version agrees with it.
#
#   A. CHANGELOG.md has a dated `## [X.Y.Z] - YYYY-MM-DD` entry. Catches
#      the two classic mistakes: bumped Cargo.toml but forgot to move
#      `[Unreleased]`, or added the header but forgot the date.
#   B. The internal `=X.Y.Z` pins in `[workspace.dependencies]` match the
#      version every member inherits from `[workspace.package]`.
#   C. Doc version pins match. Docs quote the exact line `fdl add` writes
#      (`flodl-hf = "=X.Y.Z"`), and flodl-hf depends on `flodl "=X.Y.Z"`,
#      so the two genuinely move together. Nothing else notices when they
#      don't: the number is prose, it compiles nowhere, and it sat three
#      releases behind before this check existed.

set -u
cd "$(git rev-parse --show-toplevel)"

VERSION=$(awk -F '"' '/^version *=/ { print $2; exit }' Cargo.toml)
FAIL=0

# --- A. CHANGELOG has a dated entry ---
if ! grep -qE "^## \[$VERSION\] - [0-9]{4}-[0-9]{2}-[0-9]{2}\b" CHANGELOG.md; then
    echo "FAIL: CHANGELOG.md has no '## [$VERSION] - YYYY-MM-DD' header"
    echo "  Cargo.toml version: $VERSION"
    echo "  CHANGELOG headers found (top 3):"
    grep -E '^## \[' CHANGELOG.md | head -3 | sed 's/^/    /'
    FAIL=1
fi

# --- B. Internal workspace pins ---
# The pins are exact on purpose: a consumer resolving `flodl` X.Y against
# `flodl-sys` X.(Y-1) would pair a safe wrapper with a shim it was not
# built for. Cargo catches a MISSED bump at resolve; it cannot catch one
# bumped to the wrong value.
BAD_PIN=$(grep -nE '^(flodl|flodl-cli|flodl-cli-macros|flodl-hw|flodl-sys) *= *\{ *version *= *"=' Cargo.toml |
    grep -vE "\"=$VERSION\"" || true)

if [ -n "$BAD_PIN" ]; then
    echo "FAIL: [workspace.dependencies] pins do not match version $VERSION"
    echo "$BAD_PIN" | sed 's/^/  /'
    FAIL=1
fi

# --- C. Doc version pins ---
# Scoped to tracked markdown. CHANGELOG.md is excluded: it records
# historical state, so an old release's entry legitimately quotes an old
# pin.
# Two spellings, and the second one is why this check nearly shipped
# useless: docs pin BOTH `flodl-hf = "=X.Y.Z"` and the feature-selecting
# table form `flodl-hf = { version = "=X.Y.Z", default-features = false }`.
# The original pattern saw only the first, so at the 0.8.0 bump four
# table-form pins still said 0.5.3 and the gate went green.
STALE_DOC=$(git grep -nE '(flodl|flodl-cli|flodl-cli-macros|flodl-hw|flodl-hf|flodl-sys) *= *(\{ *version *= *)?"=[0-9]+\.[0-9]+\.[0-9]+"' \
    -- '*.md' ':!CHANGELOG.md' ':!site/_site' ':!site/.jekyll-cache' ':!site/_posts' 2>/dev/null |
    grep -vE "\"=$VERSION\"" || true)

if [ -n "$STALE_DOC" ]; then
    echo "FAIL: doc version pins do not match version $VERSION"
    echo "$STALE_DOC" | sed 's/^/  /'
    FAIL=1
fi

# --- D. Stated MSRV in docs ---
# The `msrv` CI job proves each declared `rust-version` is the floor that
# actually compiles. Nothing proved the docs said the same numbers, and
# they didn't: 0.7.0 shipped `rust-version = "1.85"`, the floor was
# corrected twice on the way to 1.91, and README kept telling users 1.85
# the whole time. Any tracked markdown stating a minimum has to track the
# manifests.
#
# Two floors since 0.9.0: the workspace's for the core crates, and
# flodl-hf's own higher one. A version on a floor statement is checked
# against flodl-hf's when `flodl-hf` names it ("flodl-hf 1.95",
# "1.95+ for `flodl-hf`"), against the core floor otherwise.
#
# A floor statement is any line naming the floor in one of the ways the
# docs actually do: `[Rust](https://rustup.rs/) X.Y+`, `Rust X.Y+`, "Rust
# floor", "minimum Rust", "MSRV". The first two alone let "the Rust floor
# is now 1.91" in README's release callout go stale unseen. Only Rust 1.x
# versions with a two-digit minor are read, so crate versions on the same
# line (hf-xet 1.7, sysinfo 0.39) are not mistaken for floors.
#
# CHANGELOG.md is history and excluded. So is every UPGRADE.md section but
# the first: each section states the floor of ITS release, which is
# correct forever.
MSRV=$(awk -F '"' '/^rust-version *=/ { print $2; exit }' Cargo.toml)
HF_MSRV=$(awk -F '"' '/^rust-version *=/ { print $2; exit }' flodl-hf/Cargo.toml)
HF_MSRV=${HF_MSRV:-$MSRV}
FLOOR_RE='rustup\.rs/\) [0-9]|Rust [0-9]+\.[0-9]+\+|Rust floor|[Mm]inimum Rust|MSRV'

{
    git grep -nE "$FLOOR_RE" \
        -- '*.md' ':!CHANGELOG.md' ':!UPGRADE.md' ':!site/_site' ':!site/.jekyll-cache' ':!site/_posts' \
        2>/dev/null || true
    # ENVIRON, not -v: awk processes escapes in -v values, which mangles
    # the backslashes in the pattern (and differs between awks).
    FLOOR_RE="$FLOOR_RE" awk '
        BEGIN { re = ENVIRON["FLOOR_RE"] }
        /^## Upgrading to / { n++ }
        n == 1 && $0 ~ re { print "UPGRADE.md:" NR ":" $0 }
    ' UPGRADE.md
} > "${TMPDIR:-/tmp}/fdl-msrv-claims.$$"

STALE_MSRV=$(awk -v core="$MSRV" -v hf="$HF_MSRV" '
    {
        # "file:line:" prefix, then the text the claim is read from.
        i = index($0, ":"); j = index(substr($0, i + 1), ":")
        text = substr($0, i + j + 1)
        pos = 1
        while (match(substr(text, pos), /1\.[0-9][0-9][0-9]?/)) {
            at = pos + RSTART - 1
            tok = substr(text, at, RLENGTH)
            pos = at + RLENGTH
            prev = (at > 1) ? substr(text, at - 1, 1) : ""
            if (prev ~ /[0-9.]/ || substr(text, pos, 1) ~ /[0-9]/) continue
            from = (at > 14) ? at - 14 : 1
            before = substr(text, from, at - from)
            after = substr(text, pos, 22)
            want = core
            if (before ~ /flodl-hf/ || after ~ /^\+? for `?flodl-hf/) want = hf
            if (tok != want) print $0 "  [states " tok ", manifest says " want "]"
        }
    }
' "${TMPDIR:-/tmp}/fdl-msrv-claims.$$")
rm -f "${TMPDIR:-/tmp}/fdl-msrv-claims.$$"

if [ -n "$STALE_MSRV" ]; then
    echo "FAIL: docs state a minimum Rust version the manifests do not declare"
    echo "  core crates: $MSRV (Cargo.toml), flodl-hf: $HF_MSRV (flodl-hf/Cargo.toml)"
    echo "$STALE_MSRV" | sed 's/^/  /'
    FAIL=1
fi

[ "$FAIL" = 0 ] && echo "PASS: CHANGELOG, workspace pins, doc pins say $VERSION; docs state MSRV $MSRV (flodl-hf $HF_MSRV)"
exit "$FAIL"
