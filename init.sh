#!/bin/sh
# floDl project scaffolding -- generates a ready-to-build Rust DL project.
#
# Downloads the flodl-cli binary and uses it to scaffold the project.
# Templates come from a single source (the CLI), so they're always up to date.
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/fab2s/floDl/main/init.sh | sh -s my-project
#   cd my-project && make build
#
# Or run locally:
#   sh init.sh my-project

set -e

REPO="fab2s/floDl"
PROJECT_NAME="${1:?Usage: sh init.sh <project-name>}"

# --- Validate project name ---

case "$PROJECT_NAME" in
    *[!a-zA-Z0-9_-]*)
        echo "error: project name must contain only letters, digits, hyphens, underscores" >&2
        exit 1
        ;;
esac

if [ -e "$PROJECT_NAME" ]; then
    echo "error: '$PROJECT_NAME' already exists" >&2
    exit 1
fi

# --- Dependency checks ---

if ! command -v docker >/dev/null 2>&1; then
    echo "error: Docker is required but not installed." >&2
    echo "" >&2
    echo "Install Docker:" >&2
    echo "  Linux:   https://docs.docker.com/engine/install/" >&2
    echo "  macOS:   https://docs.docker.com/desktop/install/mac-install/" >&2
    echo "  Windows: https://docs.docker.com/desktop/install/windows-install/" >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "error: Docker is installed but not running (or you lack permissions)." >&2
    echo "" >&2
    echo "  Start Docker:     sudo systemctl start docker" >&2
    echo "  Or add yourself:  sudo usermod -aG docker \$USER  (then log out/in)" >&2
    echo "  Docker Desktop:   make sure the app is running" >&2
    exit 1
fi

if ! command -v make >/dev/null 2>&1; then
    echo "error: make is required but not installed." >&2
    echo "" >&2
    echo "Install make:" >&2
    echo "  Ubuntu/Debian:  sudo apt install make" >&2
    echo "  Fedora/RHEL:    sudo dnf install make" >&2
    echo "  macOS:          xcode-select --install" >&2
    echo "  Windows (WSL):  sudo apt install make" >&2
    exit 1
fi

# --- Get the CLI binary ---

CLI=$(mktemp "${TMPDIR:-/tmp}/flodl-cli-XXXXXX")
trap 'rm -f "$CLI"' EXIT

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "$ARCH" in
    arm64) ARCH="arm64" ;;
    *)     ;;
esac
ARTIFACT="flodl-cli-${OS}-${ARCH}"

# Get latest release tag
LATEST_TAG=""
if command -v curl >/dev/null 2>&1; then
    LATEST_TAG=$(curl -sI "https://github.com/$REPO/releases/latest" 2>/dev/null \
        | grep -i '^location:' | sed 's|.*/||' | tr -d '\r\n')
fi

GOT_CLI=false
if [ -n "$LATEST_TAG" ]; then
    URL="https://github.com/$REPO/releases/download/${LATEST_TAG}/${ARTIFACT}"
    if curl -sfL "$URL" -o "$CLI" 2>/dev/null; then
        chmod +x "$CLI"
        if "$CLI" version >/dev/null 2>&1; then
            GOT_CLI=true
        fi
    fi
fi

# Fall back to cargo if available
if ! $GOT_CLI && command -v cargo >/dev/null 2>&1; then
    echo "Downloading pre-compiled CLI failed; building from source..."
    TMPDIR_BUILD=$(mktemp -d "${TMPDIR:-/tmp}/flodl-build-XXXXXX")
    trap 'rm -f "$CLI"; rm -rf "$TMPDIR_BUILD"' EXIT
    git clone --depth 1 "https://github.com/$REPO.git" "$TMPDIR_BUILD/flodl" 2>/dev/null
    (cd "$TMPDIR_BUILD/flodl" && cargo build --release -p flodl-cli 2>/dev/null)
    cp "$TMPDIR_BUILD/flodl/target/release/fdl" "$CLI"
    chmod +x "$CLI"
    if "$CLI" version >/dev/null 2>&1; then
        GOT_CLI=true
    fi
fi

if ! $GOT_CLI; then
    echo "error: Could not obtain flodl-cli binary." >&2
    echo "" >&2
    echo "  No pre-compiled binary found for $OS/$ARCH," >&2
    echo "  and Rust is not available to build from source." >&2
    echo "" >&2
    echo "  Alternative: clone the repo and use fdl directly:" >&2
    echo "    git clone https://github.com/$REPO.git" >&2
    echo "    cd floDl && ./fdl init $PROJECT_NAME --docker" >&2
    exit 1
fi

# --- Scaffold the project ---

"$CLI" init "$PROJECT_NAME" --docker
