# fdl.yaml spec

`fdl.yaml` is the project manifest for flodl projects. It defines scripts,
sub-commands, and training configurations. `fdl` discovers it automatically
at the project root, like `package.json` or `composer.json`.

## Quick start

```
fdl --help                             # show scripts, commands, built-ins
fdl build                              # run a script
fdl ddp-bench --model mlp              # pass args to a sub-command
fdl ddp-bench anchor-3                 # run a named job
fdl ddp-bench anchor-3 --model mlp     # job + extra args
fdl ddp-bench --list                   # show available jobs
```

## Format

YAML primary. JSON and TOML accepted, auto-detected by file extension.
Same Rust struct, three serde deserializers.

## Project manifest

### Root fdl.yaml

```yaml
# fdl.yaml (project root)
description: flodl - Rust deep learning framework

scripts:
  build: make build
  test: make test
  clippy: make clippy
  shell: make shell
  cuda-test: make cuda-test
  cuda-test-all:
    description: Full CUDA suite including NCCL isolated tests
    run: make cuda-test-all

commands:
  - ddp-bench/
  - benchmarks/
```

### Scripts

Named shell commands. Short form (string) or long form (with description):

```yaml
scripts:
  build: make build                      # short form
  cuda-test-all:                         # long form
    description: Full CUDA suite
    run: make cuda-test-all
```

### Commands

Sub-directories with their own `fdl.yaml`. Listed by path, descriptions
come from each child's own `fdl.yaml` (zero duplication). `fdl --help`
assembles the full picture automatically.

```yaml
commands:
  - ddp-bench/
  - benchmarks/
```

## Sub-command manifest

Each sub-command has its own `fdl.yaml` with an `entry` point,
optional structured config sections, and named jobs.

```yaml
# ddp-bench/fdl.yaml
description: DDP validation and benchmark suite
entry: cargo run --release --features cuda --

ddp:
  mode: nccl-async
  timeline: true

training:
  epochs: 5
  batch_size: 256
  seed: 42

output:
  dir: runs/

jobs:
  anchor-3:
    description: ElChe with tight sync
    ddp: { anchor: 3 }

  anchor-5:
    ddp: { anchor: 5 }

  anchor-10:
    ddp: { anchor: 10 }

  quick:
    description: Fast smoke test
    training: { epochs: 1, batches_per_epoch: 100 }
    options: { model: linear }

  full-sweep:
    description: All models, all modes
    options: { model: all, mode: all }

  solo-baseline:
    description: Solo GPU reference
    ddp: { mode: solo-0 }
    training: { lr: 0.001 }
```

### Entry

The default binary that bare arguments pass through to.
`fdl ddp-bench --model mlp` becomes:
`cargo run --release --features cuda -- --model mlp`

### Jobs

Named argument/option presets. A job merges its config with the
root-level defaults and passes the result to the entry point.

Jobs are resolved implicitly: `fdl ddp-bench anchor-3` checks
if `anchor-3` matches a job name. If yes, apply it. If no,
pass through as a bare argument to entry.

```
fdl ddp-bench anchor-3                 # job match -> merge config
fdl ddp-bench anchor-3 --model mlp     # job + extra CLI args
fdl ddp-bench --model mlp --epochs 10  # no job, pure pass-through
fdl ddp-bench --list                   # show available jobs
```

## CLI resolution

```
fdl <name> [args...]
  1. built-in?        (setup, init, diagnose)       -> execute
  2. root script?     (scripts section)              -> execute
  3. sub-command?      (commands section)             -> load child fdl.yaml
     3a. first arg matches a job?                    -> merge config, exec entry
     3b. otherwise                                   -> pass args to entry
  4. not found                                       -> error + suggestions
```

## Config sections

Structured sections that `fdl` understands natively. They appear at
the root level of a sub-command's `fdl.yaml` (defaults for all jobs)
and can be overridden per job.

### ddp

Maps 1:1 to `DdpConfig` and `DdpRunConfig`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| mode | string | - | Shorthand: `solo-{n}`, `sync`, `{nccl\|cpu}-{sync\|cadence\|async}` |
| policy | string | - | Decomposed: `sync`, `cadence`, `async` (mutually exclusive with mode) |
| backend | string | - | Decomposed: `nccl`, `cpu` (mutually exclusive with mode) |
| anchor | auto/int | auto | Initial slow-GPU batches between syncs |
| max_anchor | int | 200 | Gradient staleness ceiling |
| overhead_target | float | 0.10 | AllReduce overhead fraction target |
| divergence_threshold | float | 0.05 | Param norm drift trigger (Async only) |
| max_batch_diff | int/null | null | Max batch lead fast over slow. null=unlimited, 0=lockstep |
| speed_hint | object | null | `{slow_rank: int, ratio: float}`, optional warm-start |
| partition_ratios | list/null | null | Explicit e.g. [0.7, 0.3], disables auto-balance |
| progressive | auto/bool | auto | Chunk dispatch mode |
| max_grad_norm | float/null | null | Per-rank L2 gradient clip before AllReduce |
| auto_scale_lr | bool | true | Scale LR by world_size |
| snapshot_timeout | int | 5 | Seconds, CPU backend only |
| checkpoint_every | int/null | null | Epochs between DDP checkpoints |
| timeline | bool | false | Record timeline events |

### training

Pure scalars that map directly to CLI args or RunConfig fields.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| epochs | int | - | Number of epochs |
| batch_size | int | - | Batch size per GPU |
| batches_per_epoch | int/null | null | Override (null = derive from dataset) |
| lr | float | - | Learning rate |
| seed | int | 42 | RNG seed |

### output

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| dir | string | runs/ | Base output directory. Job name appended. |
| timeline | bool | false | Save timeline JSON/CSV/HTML |
| monitor | int/null | null | Dashboard port (null = disabled) |

### options

Pass-through key-value pairs rendered as `--key value` on the command line.
Used in jobs for arguments that don't have a structured section yet.

```yaml
jobs:
  full-sweep:
    options: { model: all, mode: all }
```

### model (future)

Declarative alternative to entry/target. Follows conventions that define
how to build, train, and deploy a model. Not yet implemented.

```yaml
model:
  name: letter-subscan
  entry: models/subscan        # convention: look for model.yaml, build.rs, etc.
```

## Inheritance rules

1. Root-level sections in a sub-command's fdl.yaml are defaults for all jobs
2. Job-level sections deep-merge into root defaults
3. Job-level scalar wins over root-level scalar
4. Job-level map extends root-level map (matching keys overwritten)
5. CLI args override everything (appended after config-derived args)
6. `null` means "use default", not "clear the field"

## Execution model

```
fdl ddp-bench anchor-3 --model mlp
  -> load ddp-bench/fdl.yaml
  -> resolve job "anchor-3"
  -> deep-merge: root ddp + root training + job overrides
  -> translate structured sections to CLI args
  -> append explicit CLI args (--model mlp)
  -> exec: cargo run --release --features cuda -- [merged args] --model mlp
```

## Help assembly

`fdl --help` reads the root fdl.yaml and each child's description:

```
fdl - flodl project toolkit

Built-in:
  setup          Auto-detect hardware, download libtorch, build Docker
  init           Scaffold a new flodl project
  diagnose       Check GPU, libtorch, Docker health

Scripts:
  build          make build
  test           make test
  clippy         make clippy
  shell          make shell
  cuda-test      make cuda-test
  cuda-test-all  Full CUDA suite including NCCL isolated tests

Commands:
  ddp-bench      DDP validation and benchmark suite
  bench          flodl vs PyTorch comparison
```

`fdl ddp-bench --list`:

```
ddp-bench - DDP validation and benchmark suite

Jobs:
  anchor-3       ElChe with tight sync
  anchor-5       -
  anchor-10      -
  quick          Fast smoke test
  full-sweep     All models, all modes
  solo-baseline  Solo GPU reference

Entry: cargo run --release --features cuda --
  Use --help for entry point options
```

## Design principles

- **Convention over configuration.** fdl.yaml at root = project manifest.
  Sub-directories with fdl.yaml = sub-commands. No registration needed.
- **Progressive complexity.** Simple script -> named job -> full structured config.
  Each level is useful on its own.
- **No premature abstraction.** Structured sections (ddp, training) exist because
  fdl understands them. Everything else is pass-through options.
- **Self-documenting.** `--help` and `--list` assemble from descriptions in the
  YAML. New contributors see everything immediately.
- **Replaces Makefile.** Scripts + commands + Docker awareness = complete project CLI.
