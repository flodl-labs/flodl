# The floDl CLI

`fdl` is floDl's command-line tool. It handles hardware detection, libtorch
management, project scaffolding, guided setup, and doubles as a project
task runner driven by a declarative `fdl.yml` manifest. It is a pure Rust
binary with **zero native dependencies** (no libtorch needed to run),
compiles in under a second, and works on any machine with Rust or Docker.

`fdl` is useful in three contexts, and this reference is structured around
them:

1. **[Standalone](#1-standalone-no-project-required)** -- just the binary,
   no project around. Hardware probing, libtorch install, scaffolding,
   skill bundles, `fdl install`.
2. **[Inside a floDl project](#2-inside-a-flodl-project-the-fdlyml-manifest)** --
   any directory (or ancestor) that contains an `fdl.yml`. Manifest-driven
   task dispatch, environment overlays, schema introspection, preset
   sub-commands, value-aware completions.
3. **[In the flodl source checkout](#3-in-the-flodl-source-checkout)** --
   the cloned repo's `fdl.yml` ships the concrete command set used to
   develop flodl itself (`fdl test`, `fdl cuda-test`, `fdl ddp-bench …`,
   `fdl self-build`, etc.).

Standalone, libtorch is managed under `~/.flodl/` (override with
`$FLODL_HOME`). In a project, it is managed under `./libtorch/` in the
project root.

---

## Install

```bash
# Option 1: cargo install (requires Rust)
cargo install flodl-cli

# Option 2: download a pre-compiled binary (no Rust needed)
curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl
```

The `fdl` bootstrap script downloads the right pre-compiled binary from
GitHub Releases on first use. It falls back to `cargo build` if no binary
is available for your platform.

For developers working on flodl itself:

```bash
cargo build --release -p flodl-cli
./target/release/fdl --help
```

### Make it global

```bash
fdl install                  # copies to ~/.local/bin/fdl
fdl install --dev            # symlink instead (developers: tracks local builds)
fdl install --check          # compare installed vs latest GitHub release
```

`fdl install` downloads the latest release from GitHub if a newer version
is available. It detects your shell and prints PATH instructions if
`~/.local/bin` is not yet on your PATH. Use `--dev` to symlink instead of
copy, so `cargo build --release -p flodl-cli` instantly updates the
global `fdl`.

---

## Global flags

Every `fdl` invocation accepts the following flags before the command
name (or in some positions, after):

| Flag             | Effect                                                         |
|------------------|----------------------------------------------------------------|
| `-h`, `--help`   | Show help for the current command scope.                       |
| `-V`, `--version`| Print the CLI version.                                         |
| `--env <name>`   | Apply `fdl.<name>.yml` overlay on top of `fdl.yml`.            |
| `-v`             | Verbose output.                                                |
| `-vv`            | Debug output.                                                  |
| `-vvv`           | Trace output (maximum detail).                                 |
| `-q`, `--quiet`  | Suppress non-error output.                                     |
| `--ansi`         | Force ANSI color (bypass TTY / `NO_COLOR` detection).          |
| `--no-ansi`      | Disable ANSI color output.                                     |

Verbosity flags propagate into the framework's logging system
(`flodl::log`) and into Docker child commands via `FLODL_VERBOSITY`.
Equivalent without the CLI: `FLODL_VERBOSITY=verbose cargo run`. The
variable accepts integers `0`–`4` or names
`quiet`/`normal`/`verbose`/`debug`/`trace`. Level `normal` (1) is the
default when no verbosity flag is passed.

```bash
fdl -v ddp-bench quick    # verbose: DDP sync, cadence changes, prefetch detail
fdl -vv cuda-test         # debug: per-batch timing, internal loops
fdl -vvv shell            # trace: extreme granularity
fdl --quiet test          # errors only
fdl --no-ansi config show # plain output for pipes and CI
```

Some flag names are **reserved** by the CLI and cannot be shadowed by
derived argument structs (see [Declaring flags in Rust](#declaring-flags-in-rust)):
`--help`, `--version`, `--quiet`, `--env`, and the shorts `-h`, `-V`,
`-q`, `-v`, `-e`.

---

## 1. Standalone: no project required

These commands work from any directory. They don't need an `fdl.yml`, a
Cargo project, or a flodl checkout.

### `fdl setup`

Interactive wizard that walks you through everything:

1. **Detects your system** -- CPU, RAM, Docker, Rust, GPUs.
2. **Downloads libtorch** -- auto-picks the right variant for your GPU(s).
3. **Configures your build** -- Docker or native, builds images if needed.

```bash
fdl setup                      # interactive (asks questions)
fdl setup --non-interactive    # auto-detect everything, no prompts
fdl setup -y                   # alias for --non-interactive
fdl setup --force              # re-download even if libtorch exists
```

The wizard handles tricky scenarios automatically:

- **No GPU?** Downloads CPU libtorch.
- **Volta+ GPUs (sm_70+)?** Downloads cu128.
- **Pre-Volta GPUs (sm_50–sm_61)?** Downloads cu126.
- **Mixed GPUs (old + new)?** Offers to build from source or pick the best
  pre-built variant.

### `fdl libtorch`

Manage libtorch installations. Variants live under `libtorch/` in your
project (or `$FLODL_HOME/libtorch/` when standalone), each with a
metadata `.arch` file. An `.active` pointer selects the current one.

#### `fdl libtorch download`

Download a pre-built libtorch from PyTorch's official mirrors.

```bash
fdl libtorch download              # auto-detect GPU, pick best variant
fdl libtorch download --cpu        # force CPU-only (~200MB)
fdl libtorch download --cuda 12.8  # CUDA 12.8 / cu128 (~2GB)
fdl libtorch download --cuda 12.6  # CUDA 12.6 / cu126 (~2GB)
fdl libtorch download --path ~/lib # install to a custom directory
fdl libtorch download --no-activate # install but do not switch `.active`
fdl libtorch download --dry-run    # show what would happen
```

`--cuda` only accepts `12.6` or `12.8` (the published pre-built
versions). Auto-completion offers both.

**Variant coverage:**

| Variant | Architectures  | GPUs                              |
|---------|----------------|-----------------------------------|
| CPU     | --              | Any (no GPU acceleration)          |
| cu126   | sm_50 to sm_90 | Maxwell through Ada Lovelace      |
| cu128   | sm_70 to sm_120 | Volta through Blackwell          |

If your GPUs span both ranges (e.g. GTX 1060 + RTX 5060 Ti), no single
pre-built variant covers both. Use `fdl libtorch build` instead.

#### `fdl libtorch build`

Compile libtorch from PyTorch source for your exact GPU combination.
Takes 2–6 hours depending on CPU cores. Two build methods are available:

- **Docker** (default when available) -- isolated, reproducible, resumes
  via layer caching. Requires Docker.
- **Native** -- faster, builds directly on your host. Requires CUDA
  toolkit (nvcc), cmake, python3, git, and gcc.

When both are available, the CLI asks which you prefer. Use `--docker`
or `--native` to skip the prompt.

```bash
fdl libtorch build                         # auto-detect GPUs and backend
fdl libtorch build --native                # force native build
fdl libtorch build --docker                # force Docker build
fdl libtorch build --archs "6.1;12.0"      # explicit architectures
fdl libtorch build --jobs 8                # parallel compilation jobs (default: 6)
fdl libtorch build --dry-run               # show plan without building
```

Output lands in `libtorch/builds/<arch-signature>/` (e.g.
`libtorch/builds/sm61-sm120/`).

**Native build requirements:**

| Tool     | Purpose               | Install                                              |
|----------|-----------------------|------------------------------------------------------|
| nvcc     | CUDA compiler         | [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) |
| cmake    | Build system          | `apt install cmake` / `brew install cmake`           |
| python3  | PyTorch build scripts | Usually pre-installed                                 |
| git      | Clone PyTorch source  | `apt install git`                                     |
| gcc/g++  | C++ compilation       | `apt install gcc g++`                                 |

Python packages (pyyaml, jinja2, etc.) install automatically via pip.
The PyTorch source is cached at `libtorch/.build-cache/pytorch/`, so
re-running after a failure skips the clone.

#### `fdl libtorch list / info / activate / remove`

```bash
fdl libtorch list            # human-readable
fdl libtorch list --json     # machine-readable
fdl libtorch info            # show active variant details
fdl libtorch activate <name> # switch the active variant
fdl libtorch remove <name>   # delete a variant (clears .active if it was active)
```

`activate` and `remove` take a variant name as shown by
`fdl libtorch list` (e.g. `precompiled/cu128`, `builds/sm61-sm120`).
Passing no name prints the list and exits.

Example `info` output:

```
Active:   builds/sm61-sm120
Version:  2.10.0
CUDA:     12.8
Archs:    6.1 12.0
Source:   compiled
```

#### Using `fdl` as a standalone libtorch manager (tch-rs / PyTorch C++)

The libtorch-management and diagnostics commands are independent of
flodl and fill a gap PyTorch itself never filled: a proper installer.
`fdl` works as a drop-in libtorch manager for:

- **tch-rs projects** -- download the right libtorch, point `LIBTORCH`
  at it, build. No more hand-fetching URLs from the PyTorch
  get-started page.
- **PyTorch C++ development** -- juggle CPU, CUDA 12.6, CUDA 12.8, and
  source-built variants on the same host without symlink choreography.
- **Mixed-GPU systems** -- when no single pre-built variant covers
  your architectures (e.g. GTX 1060 sm_61 + RTX 5060 Ti sm_120),
  `fdl libtorch build` compiles PyTorch from source with the exact
  archs you need. Docker-isolated by default, native toolchain
  supported.
- **CI pipelines** -- `fdl diagnose --json` emits a machine-readable
  hardware and compatibility report to gate jobs on GPU presence or
  libtorch version.

Standalone (no project directory), everything installs under
`$FLODL_HOME` (default `~/.flodl/`). Pick any location you prefer and
export it before the first command:

```bash
export FLODL_HOME=~/.libtorch-variants
```

**Example A: PyTorch C++ (LibTorch via CMake) on an RTX 50-series GPU.**

This is the canonical C++ API workflow from
[pytorch.org/cppdocs](https://pytorch.org/cppdocs/installing.html),
with `fdl` replacing the manual URL-and-unzip dance:

```bash
# 1. Inspect hardware and download the matching libtorch.
fdl diagnose                          # confirm GPU arch (sm_120 in this case)
fdl libtorch download --cuda 12.8     # ~2GB, unpacks to $FLODL_HOME/libtorch/precompiled/cu128

# 2. Point CMake at it.
export LIBTORCH=$FLODL_HOME/libtorch/precompiled/cu128
```

Minimal `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(my_model)

find_package(Torch REQUIRED)
add_executable(my_model main.cpp)
target_link_libraries(my_model "${TORCH_LIBRARIES}")
set_property(TARGET my_model PROPERTY CXX_STANDARD 17)
```

Build and run:

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH ..
cmake --build . --parallel

# Runtime: expose libtorch's shared libs.
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
./my_model
```

To switch CUDA versions (e.g. back to 12.6 for legacy code), install
the other variant with `fdl libtorch download --cuda 12.6`, flip it
with `fdl libtorch activate precompiled/cu126`, re-export `LIBTORCH`,
and re-run CMake. No reinstall, no URL hunting.

**Example B: Rust via tch-rs on the same hardware.**

```bash
# Same download + LIBTORCH export as Example A.
export LIBTORCH=$FLODL_HOME/libtorch/precompiled/cu128
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
cargo add tch
cargo build
```

**Juggling variants across projects.** Install as many as you need
side by side, then flip the active pointer; `LIBTORCH` follows
`.active` when you source it from the `fdl libtorch info` output:

```bash
fdl libtorch download --cpu           # ~200MB, for laptops / CI
fdl libtorch download --cuda 12.6     # legacy CUDA projects
fdl libtorch download --cuda 12.8     # latest

fdl libtorch activate precompiled/cu126   # work on legacy code
fdl libtorch activate precompiled/cu128   # work on RTX 50-series code
fdl libtorch info                         # confirm what's active
```

**Mixed GPUs (no pre-built variant covers you).** If `fdl diagnose`
reports architectures that span both pre-built ranges, build from
source and `fdl` will pick up the compiled variant automatically:

```bash
fdl libtorch build --archs "6.1;12.0"     # Pascal + Blackwell
fdl libtorch list
#   builds/sm61-sm120 (active)
#   precompiled/cu128
export LIBTORCH=$FLODL_HOME/libtorch/builds/sm61-sm120
```

**CI gating example.** Use `diagnose --json` to skip GPU jobs when no
compatible device is present:

```bash
if fdl diagnose --json | jq -e '.cuda.devices | length > 0' > /dev/null; then
    cargo test --features cuda
else
    echo "no GPU detected, skipping CUDA tests"
fi
```

None of the above touches flodl itself -- `fdl` is just the libtorch
installer / activator / diagnostics tool in this mode.

### `fdl init`

Scaffold a new floDl project with everything you need to start training.

```bash
fdl init my-model            # mounted libtorch (recommended)
fdl init my-model --docker   # standalone Docker (libtorch baked in)
```

Both modes generate:

- `Cargo.toml` -- flodl dependency and optimized profiles.
- `src/main.rs` -- complete training template.
- `Makefile` -- `build` / `test` / `run` / `shell` targets wrapping Docker.
- `Dockerfile(s)` -- mode-specific (mounted vs. baked-in libtorch).
- `docker-compose.yml`.
- `.gitignore`.

The mounted mode also ships a self-contained `fdl` bootstrap script so
new contributors don't need a global install.

> The scaffold ships a **Makefile** by default, not an `fdl.yml`.
> The Makefile wraps Docker for the common tasks, which is enough for
> most projects. To adopt the manifest workflow (preset sub-commands,
> env overlays, schema introspection, shell completions), add an
> `fdl.yml` at the project root -- see
> [Inside a floDl project](#2-inside-a-flodl-project-the-fdlyml-manifest).

### `fdl diagnose`

Hardware and compatibility report. Useful for debugging setup issues or
verifying your GPU + libtorch combination works.

```bash
fdl diagnose             # human-readable report
fdl diagnose --json      # machine-readable for CI and tooling
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
fdl diagnose --json | jq '.cuda.devices[] | .sm'
```

### `fdl api-ref`

Generate a structured API reference from the flodl source. Extracts all
public types, constructors, methods, builder patterns, trait
implementations, and doc examples.

```bash
fdl api-ref                # human-readable (170+ types, 1700+ lines)
fdl api-ref --json         # structured JSON for tooling
fdl api-ref --path ~/src   # explicit source path
```

Source discovery (in order):

1. Walk up from cwd for a flodl checkout.
2. Cargo registry (`~/.cargo/registry/src/`).
3. Cached GitHub download (`~/.flodl/api-ref-cache/<version>/`).
4. Download the latest release source from GitHub (cached for next time).

This means `fdl api-ref` works anywhere, even without a local checkout.
First run on a fresh machine downloads ~2MB of source; subsequent runs
use the cache.

Example output (abbreviated):

```
flodl API Reference v0.5.0
========================================

## Modules (nn)

### Linear
  Fully connected layer: `y = x @ W^T + b`.
  file: nn/linear.rs
  constructors:
    pub fn new(in_features: i64, out_features: i64) -> Result<Self>
    pub fn on_device(in_features: i64, out_features: i64, device: Device) -> Result<Self>

### Conv2d  (implements: Module)
  file: nn/conv2d.rs
  constructors:
    pub fn configure(in_ch: i64, out_ch: i64, kernel: impl Into<KernelSize>) -> Conv2dBuilder
  builder:
    .with_stride()  .with_padding()  .with_dilation()  .done()
```

The JSON output is designed for AI-assisted porting tools. An agent can
read the full API surface, match PyTorch patterns to flodl equivalents,
and generate a working port.

### `fdl skill`

Manage AI coding assistant skills. Detects your tool, installs the right
skill files.

```bash
fdl skill list                       # show available skills and detected tools
fdl skill install                    # auto-detect tool, install all skills
fdl skill install --tool claude      # force Claude Code
fdl skill install --tool cursor      # force Cursor
fdl skill install --skill port       # install a single skill only
```

**Supported tools:**

| Tool        | Detection                       | Install target                  |
|-------------|---------------------------------|---------------------------------|
| Claude Code | `.claude/` directory            | `.claude/skills/<skill>/SKILL.md` |
| Cursor      | `.cursor/` or `.cursorrules`    | `.cursorrules` (appended)       |

**Available skills** (as of v0.5.0):

| Skill  | Description                                                                                     |
|--------|-------------------------------------------------------------------------------------------------|
| `port` | Port PyTorch scripts to flodl. Reads source, maps patterns, generates Rust project, validates with `cargo check`. |

After installing, use `/port my_model.py` in Claude Code, or ask
"Port this PyTorch code to flodl" in Cursor. Skill files are embedded
in the `fdl` binary, so this works anywhere, even without a flodl
checkout. Inside the repo, it uses the latest `ai/skills/` files from
the source tree.

### `fdl completions` / `fdl autocomplete`

Generate shell completion scripts. Completions are project-aware: they
reflect the current `fdl.yml`'s `commands:` (all three kinds) plus every
sub-command's own nested entries, and are **value-aware** for flags
declared with `choices:`.

```bash
fdl completions bash > ~/.local/share/bash-completion/completions/fdl
fdl completions zsh  > "${fpath[1]}/_fdl"
fdl completions fish > ~/.config/fish/completions/fdl.fish

fdl autocomplete    # auto-detect and install into the right shell
```

Example of value-aware completion:

```bash
fdl libtorch download --cuda <TAB>   # offers: 12.6  12.8
fdl ddp-bench quick --model <TAB>    # offers values from fdl.yml `choices:`
```

Re-running `fdl completions` picks up new entries as `fdl.yml` evolves.

---

## 2. Inside a floDl project: the `fdl.yml` manifest

Any directory (or ancestor) that contains `fdl.yml`, `fdl.yaml`, or
`fdl.json` is a floDl project in the manifest sense. In that context,
`fdl` doubles as a project task runner: `fdl <name>` dispatches into
the manifest, and a small set of meta-commands (`fdl config`,
`fdl schema`, plus the manifest sub-commands themselves) become
available.

If only `fdl.yml.example` (or `.dist`) exists, `fdl` offers to copy it
to the real (gitignored) `fdl.yml` so users can customise locally.

### The general principle

`fdl.yml` gives you four composable building blocks:

1. **Link any script as a sub-command** via `run:`.
2. **Declare arguments and options in Rust** on binaries via
   `#[derive(FdlArgs)]`. `fdl` probes them with `--fdl-schema` and
   inherits typed help + completion for free.
3. **Layer environments** with `fdl.<env>.yml` overlays (dev / ci /
   prod variations of the same command tree).
4. **Fall back to shell environment variables** per-option with
   `#[option(env = "…")]` -- argv wins, then env var, then default.

End-to-end example. A cargo-backed sub-command with Rust-declared flags,
an env overlay, and an env-var fallback for a secret:

```rust
// src/bin/train.rs in your project
use flodl_cli::FdlArgs;

/// Train the model.
#[derive(FdlArgs, Debug)]
pub struct TrainArgs {
    /// Device to train on.
    #[option(choices = &["cpu", "cuda"], default = "cuda")]
    pub device: String,

    /// Number of epochs.
    #[option(short = 'e', default = "10")]
    pub epochs: u32,

    /// Weights & Biases API key (argv > env > absent).
    #[option(env = "WANDB_API_KEY")]
    pub wandb_api_key: Option<String>,

    /// Dataset path.
    #[arg]
    pub dataset: std::path::PathBuf,
}
```

```yaml
# fdl.yml -- base manifest
description: My training project

commands:
  # Path-kind: loads ./train/fdl.yml, which declares an `entry:` pointing
  # at the cargo binary. Extra argv after `fdl train ...` flows through
  # to the entry, validated against the FdlArgs schema.
  train:
```

```yaml
# train/fdl.yml -- sub-command configuration
description: Train the model
docker: dev
entry: cargo run --release --bin train --
```

```yaml
# fdl.ci.yml -- CI overlay, deep-merged over fdl.yml
commands:
  train:
    # Overlay the child's entry for CI: CPU, one epoch.
    entry: cargo run --release --bin train -- --device cpu --epochs 1
```

Usage:

```bash
# Base config: GPU training, 10 epochs. Extra args flow to the binary.
fdl train ./data/train.bin --epochs 50

# CI overlay: CPU, one epoch. Secret picked up from the environment.
WANDB_API_KEY=xxx fdl --env ci train ./data/train.bin

# Introspect the fully-resolved config (base + overlay).
fdl config show ci

# Help flows through #[derive(FdlArgs)] -> --fdl-schema -> render_help,
# so values, choices, and env fallbacks are all visible here.
fdl train --help
```

Path-kind sub-commands with an `entry:` forward every extra argv token to
the underlying binary, where the derived parser validates it. `run:`-kind
commands (shown in the next section) are closed scripts and do not
forward extra args; use shell-level `$VAR` inside the script if you need
dynamic values.

`#[derive(FdlArgs)]` is re-exported as `flodl_cli::FdlArgs`. See the
[`flodl-cli-macros`
README](https://crates.io/crates/flodl-cli-macros) for the full
attribute surface (`short`, `default`, `choices`, `env`, `completer`,
`variadic`).

### Command kinds: `run` / `path` / preset

`fdl.yml` declares a unified `commands:` map. Each entry is exactly one
of three kinds, chosen by which fields are set:

- **Run** -- `run:` is set. Executes the inline shell script, optionally
  wrapped in `docker compose run --rm <service>` when `docker:` is set.
- **Path** -- `path:` is set (or, by convention, the entry is empty/null
  and a sibling directory named `<command>/` with its own `fdl.yml`
  exists). Loads the nested manifest and recurses.
- **Preset** -- neither `run:` nor `path:` is set. Inline `ddp:` /
  `training:` / `output:` / `options:` fields merge over the enclosing
  config and invoke its `entry:`. Only legal inside a sub-command
  (path-kind entry's own `fdl.yml`).

```yaml
description: flodl - Rust deep learning framework

commands:
  test:
    description: Run all CPU tests
    run: cargo test -- --nocapture
    docker: dev
  cuda-test:
    description: Run CUDA tests (parallel)
    run: cargo test --features cuda -- --nocapture
    docker: cuda
  shell:
    run: bash
    docker: dev
  ddp-bench:          # convention default: loads ./ddp-bench/fdl.yml
```

```bash
fdl test              # runs "test" in the "dev" docker service
fdl cuda-test         # runs in the "cuda" service
fdl shell             # opens an interactive shell
fdl ddp-bench --list  # dispatches into the ddp-bench sub-command
```

When a `run:` command declares `docker: <service>`, `fdl` wraps it in
`docker compose run --rm <service> bash -c "…"`. Without `docker:`, it
runs on the host. `docker:` is only valid on `run:` commands --
declaring it on a `path:` or preset entry is rejected at load time.

### Declaring flags in Rust

Binaries can declare their argv surface with `#[derive(FdlArgs)]`. `fdl`
probes each cargo-backed entry with a hidden `--fdl-schema` flag, caches
the resulting JSON under `<cmd-dir>/.fdl/schema-cache/<cmd>.json`, and
uses it to drive:

- `fdl <cmd> --help` -- typed, color-annotated help rendered from the
  doc-comments and attributes.
- Shell completion -- choices, short/long forms, value types.
- Validation -- unknown flags error with a clear message.

One struct is the single source of truth. The doc-comments become help
text. The attribute metadata becomes schema. The struct fields become
typed values in your `main()`.

```rust
use flodl_cli::{FdlArgs, parse_or_schema};

/// Run the training benchmark suite.
#[derive(FdlArgs, Debug)]
pub struct BenchArgs {
    /// Model to train (or `all` for the full suite).
    #[option(short = 'm', choices = &["all", "linear", "mlp", "lenet",
                                      "resnet", "char-rnn", "gpt-nano"],
             default = "all")]
    pub model: String,

    /// DDP mode to exercise.
    #[option(choices = &["solo-0", "nccl-cadence", "nccl-async",
                         "cpu-cadence", "cpu-async"],
             default = "nccl-cadence")]
    pub mode: String,

    /// Epochs to run (overrides the preset default).
    #[option(short = 'e', default = "10")]
    pub epochs: u32,

    /// Write a Markdown convergence report to this path.
    #[option]
    pub report: Option<String>,

    /// Weights & Biases API key (read from env if flag absent).
    #[option(env = "WANDB_API_KEY")]
    pub wandb_key: Option<String>,

    /// Extra dataset paths to include.
    #[arg(variadic)]
    pub datasets: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: BenchArgs = parse_or_schema();
    // args.model, args.mode, args.epochs, ... are typed values.
    Ok(())
}
```

With this struct in place:

- `cargo run --bin bench -- --help` renders an ANSI-coloured help page
  with the doc-comments as descriptions.
- `cargo run --bin bench -- --fdl-schema` emits JSON describing every
  flag. `fdl` calls this on first use and caches the result.
- `fdl bench --model <TAB>` in a completion-enabled shell offers
  `all linear mlp lenet resnet char-rnn gpt-nano`.
- `fdl bench --wandb-key <value>` works, and so does leaving the flag
  off with `WANDB_API_KEY=...` in the environment.
- Unknown flags and invalid choices fail with a clear error before your
  binary starts.

#### Attribute reference

Each field must carry exactly one of `#[option(...)]` (named flag,
kebab-cased from the field name) or `#[arg(...)]` (positional). The
field's Rust type determines cardinality.

| Shape       | Meaning                                               |
|-------------|-------------------------------------------------------|
| `bool`      | Flag is present or absent; no value. Absent = `false`. |
| `T`         | Scalar, required. `#[option]` must supply `default`.  |
| `Option<T>` | Scalar, optional. Absent = `None`.                    |
| `Vec<T>`    | `#[option]`: repeatable. `#[arg]`: variadic (last).    |

`#[option]` keys:

| Key         | Example          | Notes                                                     |
|-------------|------------------|-----------------------------------------------------------|
| `short`     | `'c'`            | Single-char short flag.                                   |
| `default`   | `"string"`       | Parsed via `FromStr` at run time; required on bare `T`.   |
| `choices`   | `&["a", "b"]`    | Accepted values; enforced by the parser.                  |
| `env`       | `"VAR_NAME"`     | Env fallback when the flag is absent; skipped on `bool`.  |
| `completer` | `"name"`         | Named completer for shell completion scripts.             |

`#[arg]` keys:

| Key         | Example          | Notes                                                     |
|-------------|------------------|-----------------------------------------------------------|
| `default`   | `"string"`       | Makes the positional optional.                            |
| `choices`   | `&["a", "b"]`    | Accepted values.                                          |
| `variadic`  | bare or `= true` | Requires `Vec<T>`; must be the last positional.           |
| `completer` | `"name"`         | Named completer for shell completion scripts.             |

Validation runs at derive time: required positionals cannot follow
optional ones, variadic must be last, reserved flags cannot be
shadowed, and duplicate long/short flags error out. Errors point at
the offending field, not at a run-time parser message.

See the [`fdl schema`](#fdl-schema-and---fdl-schema) section for how to
refresh the cache after rebuilding, and the
[`flodl-cli-macros`](https://crates.io/crates/flodl-cli-macros) README
and [`flodl-cli` docs.rs page](https://docs.rs/flodl-cli) for the full
attribute surface and internals.

### Environment overlays

The `--env <name>` flag tells `fdl` to deep-merge `fdl.<name>.yml` on
top of the base `fdl.yml` before resolving any command. Three equivalent
forms are supported, in this precedence order:

1. `fdl --env <name> <cmd>` -- explicit flag.
2. `FDL_ENV=<name> fdl <cmd>` -- environment variable.
3. `fdl <name> <cmd>` -- first-arg convention. Only fires when `<name>`
   matches a known overlay file AND does not collide with an existing
   command name (ambiguity errors loudly).

```bash
fdl --env ci test                    # flag form
FDL_ENV=ci fdl test                  # env var form
fdl ci test                          # first-arg form (if fdl.ci.yml exists)
```

Explicit selectors (flag / env var) fail loudly if the overlay file is
missing. The first-arg form silently falls through to normal dispatch
when no matching file exists, so existing commands are never shadowed.

Typical overlay files:

- `fdl.dev.yml` -- fast iteration (shorter epochs, smaller batches).
- `fdl.ci.yml` -- CPU-only, minimal epochs, strict validation.
- `fdl.prod.yml` -- full runs, checkpoint to cloud storage.

Use `fdl config show <env>` to preview the resolved merged config.

### Preset sub-commands

A sub-command directory (e.g. `ddp-bench/`) has its own `fdl.yaml` with
an `entry:`, optional `docker:`, structured `ddp` / `training` /
`output` sections, and a `commands:` map whose entries are **presets** --
inline overrides of this config's `entry`:

```yaml
description: DDP validation and benchmark suite
docker: cuda
entry: cargo run --release --features cuda --

training:
  epochs: 5
  seed: 42

ddp:
  policy: cadence
  backend: nccl
  divergence_threshold: 0.05
  lr_scale_ratio: 1.0

commands:
  quick:
    description: Fast smoke test
    training: { epochs: 1 }
    options: { model: linear, mode: solo-0, batches: 100 }
  validate:
    options: { model: all, mode: all, validate: true }
```

Then:

```bash
fdl ddp-bench quick                   # runs the "quick" preset
fdl ddp-bench validate --report out   # preset + extra flags
fdl ddp-bench --help                  # description + presets + defaults
fdl ddp-bench validate --help         # resolved options
```

A sub-command's `commands:` may mix kinds freely: a preset sits
alongside a nested `path:` (another directory) or a standalone `run:`
helper. `fdl <cmd> --help` splits them into an **Arguments** section
(the single preset slot, with values indented underneath -- override the
placeholder via `arg-name:`) and a **Commands** section (real
sub-commands with their own behaviour).

The `ddp:` section maps 1:1 to flodl's `DdpConfig` / `DdpRunConfig`
(`mode`, `policy`, `backend`, `anchor`, `max_anchor`, `overhead_target`,
`divergence_threshold`, `max_batch_diff`, `speed_hint`,
`partition_ratios`, `progressive`, `max_grad_norm`, `lr_scale_ratio`,
`snapshot_timeout`, `checkpoint_every`, `timeline`). See
[docs/design/run-config.md][run-config] for the full schema and merge
semantics.

[run-config]: design/run-config.md

### `fdl config`

Inspect the resolved project configuration, with or without an overlay
applied.

```bash
fdl config show              # base fdl.yml
fdl config show ci           # base deep-merged with fdl.ci.yml
fdl --env ci config show     # same result, via the flag form
fdl ci config show           # same result, via first-arg
```

The output is the fully-merged YAML with per-layer annotations, so you
can see which file contributed which field. Useful for debugging
overlay behaviour before running a long job.

### `fdl schema` and `--fdl-schema`

Every binary whose argv struct uses `#[derive(FdlArgs)]` responds to a
hidden `--fdl-schema` flag by emitting a JSON description of its
arguments and options. `fdl` uses this to power help, completion, and
validation for project commands, caching the result per-command.

```bash
fdl schema list              # every cached schema with fresh/stale/orphan status
fdl schema list --json       # machine-readable
fdl schema clear             # delete all cached schemas
fdl schema clear ddp-bench   # delete one
fdl schema refresh           # re-probe every entry and rewrite the cache
fdl schema refresh ddp-bench # refresh one
```

Cached schemas live at `<cmd-dir>/.fdl/schema-cache/<cmd>.json`.

**Cargo entries must be built before `refresh`** -- `fdl` runs the
entry's `--fdl-schema` as a subprocess, which requires the binary to
exist. Rebuild the target first, then refresh:

```bash
cargo build --release --features cuda
fdl schema refresh ddp-bench
fdl ddp-bench --help         # now picks up the new schema
```

An individual command can also refresh its own cache on the next
invocation by passing `--refresh-schema`:

```bash
fdl ddp-bench --refresh-schema
```

This is handy during development: rebuild, run with the refresh flag,
and the cache updates automatically without calling `fdl schema
refresh` explicitly.

---

## 3. In the flodl source checkout

The flodl repo's own `fdl.yml` ships the concrete command set used to
develop floDl itself. These are examples of the manifest system from the
previous section, not built-in commands.

### Development loop

```bash
fdl check              # type-check without building
fdl build              # debug build
fdl clippy             # lint (tests + workspace + ddp-bench)
fdl test               # all CPU tests
fdl test-release       # tests in release mode
fdl doc                # rustdoc, strict (-D warnings)
```

### CUDA / GPU testing

```bash
fdl cuda-build            # build with CUDA feature
fdl cuda-clippy           # lint with CUDA feature
fdl cuda-test             # parallel CUDA tests (excludes NCCL / Graph)
fdl cuda-test-nccl        # NCCL/DDP tests only (isolated processes)
fdl cuda-test-graph       # CUDA Graph tests (exclusive GPU, single-threaded)
fdl cuda-test-serial      # remaining serial tests
fdl cuda-test-all         # full suite: parallel + NCCL isolated + serial
```

### Benchmarks

```bash
fdl bench                 # flodl vs PyTorch (CUDA)
fdl bench-cpu             # CPU-only benchmarks
```

### DDP validation suite

`ddp-bench/` is a `path:`-kind sub-command with its own `fdl.yml` and
preset commands. Example presets (from `ddp-bench/fdl.yml`):

```bash
fdl ddp-bench quick                   # fast smoke test (1 epoch, linear model)
fdl ddp-bench validate                # full DDP validation matrix
fdl ddp-bench validate --report out   # validation + write report to out/
fdl ddp-bench --help                  # list all presets + options
```

### Interactive shells

```bash
fdl shell         # dev container (CPU)
fdl cuda-shell    # CUDA container
```

### Re-building the CLI

After editing `flodl-cli/`:

```bash
fdl self-build    # rebuild fdl and replace the installed binary
```

This uses the currently-running `fdl` to rebuild itself, and swaps the
new binary into place atomically.

---

## libtorch directory layout

The CLI manages libtorch installations under `libtorch/` in your project
root:

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

Docker Compose and Make targets read `.active` to mount the right
libtorch variant automatically. You never need to set `LIBTORCH_PATH`
manually when using Docker.

---

## Architecture notes

The CLI is built as a pure Rust binary with **zero external crate
dependencies** beyond serde. GPU detection uses `nvidia-smi`, downloads
use `curl`/`wget`, and zip extraction uses `unzip` (or PowerShell on
Windows). This means:

- **~750KB binary** -- trivially distributable.
- **Compiles in under 1 second** -- no C++ compilation, no libtorch
  linking.
- **Cross-platform** -- Linux x86_64/aarch64, macOS arm64, Windows
  x86_64.
- **No runtime dependencies** -- works on any machine; GPU features
  degrade gracefully when `nvidia-smi` is absent.

Pre-compiled binaries are published to GitHub Releases on every tagged
release. The `fdl` shell script is a thin bootstrap that downloads the
right binary, falling back to `cargo build` if no binary is available
for your platform.
