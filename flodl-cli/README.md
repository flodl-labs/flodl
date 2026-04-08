# flodl-cli

Standalone libtorch manager and GPU diagnostic tool for Rust deep learning.

Part of the [floDl](https://flodl.dev) framework, but works independently.
No libtorch or Python required to install.

## Install

**From crates.io** (requires Rust):

```sh
cargo install flodl-cli
```

**Pre-compiled binaries** (no Rust needed):

```sh
curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl
./fdl setup
```

Binaries are published for Linux x86_64/aarch64, macOS arm64, and Windows x86_64
on every [GitHub Release](https://github.com/fab2s/floDl/releases).

## What it does

`fdl` manages libtorch installations (download, build from source, switch
between variants) and detects GPU hardware. It works both inside a floDl
project and standalone.

**Inside a project**: libtorch is stored in `./libtorch/` (mounted into Docker
at build time).

**Standalone**: libtorch is stored in `~/.flodl/libtorch/` (override with
`$FLODL_HOME`). Useful for tch-rs projects, PyTorch C++ development, or
managing multiple libtorch versions system-wide.

## Commands

```
fdl setup                      # guided setup wizard
fdl libtorch download          # auto-detect GPU, download matching libtorch
fdl libtorch download --cpu    # force CPU variant
fdl libtorch download --cuda 12.8
fdl libtorch build             # compile from source (custom GPU archs)
fdl libtorch list              # show installed variants
fdl libtorch activate <name>   # switch active variant
fdl libtorch info              # show active variant details
fdl libtorch remove <name>     # remove a variant
fdl init my-project            # scaffold a new floDl project
fdl diagnose                   # system + GPU + compatibility report
fdl diagnose --json            # machine-readable output
```

## GPU auto-detection

`fdl libtorch download` probes your GPUs via `nvidia-smi` and picks the right
libtorch variant:

- Volta+ (sm_70+): downloads cu128
- Pre-Volta (sm_50-sm_61): downloads cu126
- Mixed architectures: downloads cu126, suggests source build
- No GPU: downloads CPU variant

## Source builds

For mixed GPU setups (e.g., GTX 1060 + RTX 5060 Ti), no single pre-built
libtorch covers both architectures. `fdl libtorch build` compiles from
PyTorch source with your exact GPU architectures:

```sh
fdl libtorch build                    # auto-detect GPUs
fdl libtorch build --archs "6.1;12.0" # explicit architectures
fdl libtorch build --docker           # isolated Docker build
fdl libtorch build --native           # use host toolchain
```

Docker builds resume via layer cache if interrupted.

## Project scaffolding

```sh
fdl init my-model            # mounted libtorch (recommended)
fdl init my-model --docker   # libtorch baked into Docker image
cd my-model && make build
```

Generated projects include Dockerfiles, docker-compose.yml, Makefile, and
an annotated training template.

## Links

- [floDl framework](https://flodl.dev)
- [Tutorials](https://flodl.dev/guide/tensors)
- [DDP Reference](https://flodl.dev/guide/ddp-reference)
- [PyTorch Migration Guide](https://flodl.dev/guide/migration)
- [GitHub](https://github.com/fab2s/floDl)

## License

MIT
