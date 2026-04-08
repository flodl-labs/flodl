# The floDl CLI

`fdl` is floDl's command-line tool. It handles hardware detection, libtorch
management, project scaffolding, and guided setup. It is a pure Rust binary
with zero native dependencies (no libtorch needed to run), so it compiles in
under a second and works on any machine with Rust or Docker.

It works both inside a floDl project and standalone. When standalone, libtorch
is managed under `~/.flodl/` (override with `$FLODL_HOME`).

## Install

```bash
# Option 1: cargo install (requires Rust)
cargo install flodl-cli

# Option 2: download pre-compiled binary (no Rust needed)
curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl
```

The `fdl` bootstrap script downloads a pre-compiled binary from GitHub Releases
on first use. It falls back to `cargo build` if no binary is available for your
platform.

For developers working on flodl itself:

```bash
cargo build --release -p flodl-cli
./target/release/fdl help
```

## Commands

### `fdl setup`

Interactive wizard that walks you through everything:

1. **Detects your system** -- CPU, RAM, Docker, Rust, GPUs
2. **Downloads libtorch** -- auto-picks the right variant for your GPU(s)
3. **Configures your build** -- Docker or native, builds images if needed

```bash
fdl setup                  # interactive (asks questions)
fdl setup --non-interactive  # auto-detect everything, no prompts
fdl setup --force          # re-download even if libtorch exists
fdl setup -y               # alias for --non-interactive
```

The wizard handles tricky scenarios automatically:

- **No GPU?** Downloads CPU libtorch.
- **Volta+ GPUs (sm_70+)?** Downloads cu128.
- **Pre-Volta GPUs (sm_50-sm_61)?** Downloads cu126.
- **Mixed GPUs (old + new)?** Offers to build from source or pick the best
  pre-built variant.

### `fdl libtorch`

Manage libtorch installations. floDl stores variants under `libtorch/` in your
project, with metadata in `.arch` files and an `.active` pointer to the
current variant.

#### `fdl libtorch download`

Download a pre-built libtorch from PyTorch's official mirrors.

```bash
fdl libtorch download              # auto-detect GPU, pick best variant
fdl libtorch download --cpu        # force CPU-only (~200MB)
fdl libtorch download --cuda 12.8  # CUDA 12.8 / cu128 (~2GB)
fdl libtorch download --cuda 12.6  # CUDA 12.6 / cu126 (~2GB)
fdl libtorch download --dry-run    # show what would happen
fdl libtorch download --path ~/lib # install to custom directory
```

**Variant coverage:**

| Variant | Architectures | GPUs |
|---------|--------------|------|
| CPU | -- | Any (no GPU acceleration) |
| cu126 | sm_50 to sm_90 | Maxwell through Ada Lovelace |
| cu128 | sm_70 to sm_120 | Volta through Blackwell |

If your GPUs span both ranges (e.g. GTX 1060 + RTX 5060 Ti), no single
pre-built variant covers both. Use `fdl libtorch build` instead.

#### `fdl libtorch build`

Compile libtorch from PyTorch source for your exact GPU combination. Takes 2-6
hours depending on CPU cores. Two build methods are available:

- **Docker** (default when available) -- isolated, reproducible, resumes via
  layer caching. Requires Docker.
- **Native** -- faster, builds directly on your host. Requires CUDA toolkit
  (nvcc), cmake, python3, git, and gcc.

When both are available, the CLI asks which you prefer. Use `--docker` or
`--native` to skip the prompt.

```bash
fdl libtorch build                         # auto-detect GPUs and backend
fdl libtorch build --native                # force native build
fdl libtorch build --docker                # force Docker build
fdl libtorch build --archs "6.1;12.0"      # explicit architectures
fdl libtorch build --jobs 8                # parallel compilation jobs
fdl libtorch build --dry-run               # show plan without building
```

The output goes to `libtorch/builds/<arch>/` (e.g. `libtorch/builds/sm61-sm120/`).

**Native build requirements:**

| Tool | Purpose | Install |
|------|---------|---------|
| nvcc | CUDA compiler | [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) |
| cmake | Build system | `apt install cmake` / `brew install cmake` |
| python3 | PyTorch build scripts | Usually pre-installed |
| git | Clone PyTorch source | `apt install git` |
| gcc/g++ | C++ compilation | `apt install gcc g++` |

Python packages (pyyaml, jinja2, etc.) are installed automatically via pip.
The PyTorch source is cached at `libtorch/.build-cache/pytorch/`, so
re-running after a failure skips the clone.

#### `fdl libtorch list`

Show all installed libtorch variants and which one is active.

```bash
fdl libtorch list          # human-readable
fdl libtorch list --json   # machine-readable
```

Example output:

```
  builds/sm61-sm120 (active)
  precompiled/cpu
  precompiled/cu128
```

#### `fdl libtorch activate`

Switch the active libtorch variant. Docker and Make targets automatically use
whatever `.active` points to.

```bash
fdl libtorch activate precompiled/cu128
fdl libtorch activate builds/sm61-sm120
```

#### `fdl libtorch remove`

Delete an installed variant. If it was active, clears the `.active` pointer.

```bash
fdl libtorch remove precompiled/cu126
```

#### `fdl libtorch info`

Show details of the active variant.

```bash
fdl libtorch info
```

```
Active:   builds/sm61-sm120
Version:  2.10.0
CUDA:     12.8
Archs:    6.1 12.0
Source:   compiled
```

### `fdl init`

Scaffold a new floDl project with everything you need to start training.

```bash
fdl init my-model              # mounted libtorch (recommended)
fdl init my-model --docker     # standalone Docker (libtorch baked in)
```

Both modes generate:

- `Cargo.toml` with floDl dependency and optimized profiles
- `src/main.rs` with a complete training template
- `Makefile` with build/test/run/shell targets
- `Dockerfile` and `docker-compose.yml`
- `.gitignore`

The mounted mode also includes `download-libtorch.sh` for self-contained setup
in the scaffolded project.

### `fdl diagnose`

Hardware and compatibility report. Useful for debugging setup issues or
verifying your GPU + libtorch combination works.

```bash
fdl diagnose               # human-readable report
fdl diagnose --json        # machine-readable output
```

Example output:

```
floDl Diagnostics
=================

System
  CPU:         Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz (16 threads, 24GB RAM)
  OS:          Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2)
  Docker:      29.3.1

CUDA
  Driver:      576.88
  Devices:     2
  [0] NVIDIA GeForce RTX 5060 Ti -- sm_120, 15GB VRAM
  [1] NVIDIA GeForce GTX 1060 6GB -- sm_61, 6GB VRAM

libtorch
  Active:      builds/sm61-sm120
  Version:     2.10.0
  CUDA:        12.8
  Archs:       6.1 12.0
  Source:      compiled
  Variants:    builds/sm61-sm120, precompiled/cpu

Compatibility
  GPU 0 (RTX 5060 Ti, sm_120):  OK
  GPU 1 (GTX 1060 6GB, sm_61):  OK

  All GPUs compatible with active libtorch.
```

The JSON output is useful for CI pipelines and automated tooling:

```bash
fdl diagnose --json | python3 -m json.tool
```

## Architecture

The CLI is built as a pure Rust binary with **zero external crate dependencies**.
GPU detection uses `nvidia-smi`, downloads use `curl`/`wget`, and zip
extraction uses `unzip` (or PowerShell on Windows). This means:

- **~750KB binary** -- trivially distributable
- **Compiles in under 1 second** -- no C++ compilation, no libtorch linking
- **Cross-platform** -- Linux x86_64/aarch64, macOS arm64, Windows x86_64
- **No runtime dependencies** -- works on any machine, GPU features degrade
  gracefully when nvidia-smi is absent

### Distribution

Pre-compiled binaries are published to GitHub Releases on every tagged release.
The `fdl` shell script is a thin bootstrap that downloads the right binary:

1. Detects your OS and architecture
2. Fetches the latest release from `github.com/fab2s/floDl/releases`
3. Downloads the matching binary (~750KB)
4. Falls back to `cargo build` if no binary is available

This means `fdl` can be `curl`-ed from the repo and run immediately -- no Rust,
no Docker, no compilation.

The `init.sh` one-liner uses the same mechanism: it downloads the CLI binary
to a temp directory, runs `flodl-cli init --docker`, and cleans up. All
scaffolding templates come from a single source (the CLI binary), so they're
always up to date.

## libtorch Directory Layout

The CLI manages libtorch installations under `libtorch/` in your project root:

```
libtorch/
  .active                          # points to current variant (e.g. "builds/sm61-sm120")
  precompiled/
    cpu/                           # pre-built CPU variant
      lib/ include/ share/
      .arch                        # metadata: cuda=none, torch=2.10.0, ...
    cu126/                         # pre-built CUDA 12.6
      ...
    cu128/                         # pre-built CUDA 12.8
      ...
  builds/
    sm61-sm120/                    # source-built for specific GPUs
      lib/ include/ share/
      .arch                        # metadata: cuda=12.8, archs=6.1 12.0, source=compiled
```

The `.arch` file format:

```
cuda=12.8
torch=2.10.0
archs=6.1 12.0
source=compiled
```

Docker Compose and Make targets read `.active` to mount the right libtorch
variant automatically. You never need to set `LIBTORCH_PATH` manually when
using Docker.

## Makefile Integration

The CLI integrates with Make:

```bash
make setup          # runs fdl setup --non-interactive
make build-libtorch # runs fdl libtorch build
make cli            # builds the CLI binary (cargo build --release -p flodl-cli)
```
