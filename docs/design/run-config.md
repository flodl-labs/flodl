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
| divergence_threshold | float | 0.05 | Param norm drift trigger for the trend guard (Cadence and Async; Sync is a no-op by construction) |
| max_batch_diff | int/null | null | Max batch lead fast over slow. null=unlimited, 0=lockstep |
| speed_hint | object | null | `{slow_rank: int, ratio: float}`, optional warm-start |
| partition_ratios | list/null | null | Explicit e.g. [0.7, 0.3], disables auto-balance |
| progressive | auto/bool | auto | Chunk dispatch mode |
| max_grad_norm | float/null | null | Per-rank L2 gradient clip before AllReduce |
| lr_scale_ratio | float | 1.0 | LR scaling ratio: factor = 1 + (world_size-1)*ratio |
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

---

# Future Work: fdl.yaml as a Mature Config Layer

Three related design sketches that together extend fdl.yaml from "project
manifest" to "full project frontend." Each is independently useful and
independently implementable; together they make the scale-up story
(laptop -> workstation -> cluster -> cloud) a matter of editing YAML
rather than rewriting code.

**Status**: approved design direction, not yet implemented. Capture
before the details drift.

## 1. Multi-environment fdl.yaml inheritance

### Goal

One project, many environments. The same `fdl.yaml` layered with
per-environment overlays: `fdl.local.yml`, `fdl.ci.yml`, `fdl.cloud.yml`,
`fdl.prod.yml`. Research code that works on a laptop scales to a cluster
or a cross-datacenter cloud deployment by editing one YAML line, not by
rewriting the training script.

### Discovery and targeting

Two ways to select an environment:

**Convention (first-arg detection)** — zero ceremony:

```bash
fdl test                      # no env: uses fdl.yml, runs "test" script
fdl ci test                   # "ci" matches fdl.ci.yml -> env overlay, runs "test"
fdl local ddp-bench validate  # env=local, cmd=ddp-bench, job=validate
```

**Explicit flag** — disambiguation + automation:

```bash
fdl --env ci test
FDL_ENV=ci fdl test
```

### First-arg resolution rules

```
fdl <arg> [...]

1. Does a script/command named <arg> exist?   -> Y1
2. Does fdl.<arg>.yml exist in walk-up tree?  -> Y2

Y1 only:    run the script/command
Y2 only:    use fdl.<arg>.yml as overlay, expect next positional = command
Y1 + Y2:    LOUD CONFLICT — error out, show both candidates
Neither:    unknown command
```

The loud-conflict rule is the key. Silent precedence here would be a
footgun (renaming a script could accidentally shadow an env). Failing
on ambiguity the moment it's introduced is cheap insurance.

### `inherit-from:` for non-obvious topologies

Convention handles the 90% case. For cross-project inheritance or multi-
level chains, an explicit key:

```yaml
# fdl.prod.yml
inherit-from: fdl.cloud.yml   # which itself inherits from fdl.yml
ddp:
  hosts: { regions: [eu-west, us-east, ap-south] }
```

### Merge rules

- **Maps: deep-merge.** Recurse into nested maps; overlay keys win.
- **Scalars: replace.** Overlay value takes over.
- **Lists: replace entirely.** Order is always contentious; append/prepend
  modes cause more debugging pain than they save. Users who want to
  extend a list copy the full new list — unambiguous when reading.
- **`null` deletes.** `ddp: null` in an overlay removes the entire block,
  useful for "reset to defaults in this env." Avoids needing a separate
  `!unset` syntax.

Mental model: *any key you name in the overlay wins; any key you don't
mention stays.*

### Critical UX: `fdl config show`

Layered config is opaque without introspection. `fdl config show [<env>]`
prints the resolved config with source annotations:

```yaml
$ fdl config show prod
description: flodl
ddp:
  policy: diloco                # from fdl.prod.yml
  max_anchor: 500               # from fdl.cloud.yml
  overhead_target: 0.10         # from fdl.yml (default)
  hosts:                        # from fdl.prod.yml
    regions: [eu-west, us-east, ap-south]
training:
  epochs: 10                    # from $FDL_TRAINING__EPOCHS
  batch_size: 64                # from fdl.yml
```

Without this, every mysterious value becomes a multi-file debugging hunt.
With it, "why is it using that value" is one command.

### `fdl help` lists available envs

```
$ fdl help
...
Available environments:
  local     (fdl.local.yml)       [gitignored]
  ci        (fdl.ci.yml)
  cloud     (fdl.cloud.yml)
  prod      (fdl.prod.yml)

Use:  fdl <env> <command>   or   fdl --env <env> <command>
```

### MVP increment

~300 LOC in `flodl-cli/src/config.rs`:

1. First-arg env detection + conflict check (no explicit `--env` flag yet).
2. Deep-merge maps, replace lists, `null` deletes — post-parse on
   existing `ProjectConfig` / `CommandConfig` structs.
3. `fdl config show [<env>]` with source annotations.
4. `fdl help` lists available envs.

Everything else (`--env` flag, `inherit-from:`, env-var single-key
overrides like `FDL_DDP__POLICY`, `!append` escape hatches) is incremental
on top.

---

## 2. Option schemas and the `--fdl-schema` contract

### The duplication trap

Without a shared source of truth, option definitions end up in three
places: the binary's argv parser, the binary's `--help` output, and the
fdl.yaml schema. That's maintenance friction and drift waiting to happen.
ddp-bench's current `main.rs` has ~65 lines of argv parsing and ~35 lines
of help text encoding the same 17 flags. Mirroring that in YAML would
triple the cost.

The fix is one source of truth, with the other layers derived.

### The contract

A binary that wants fdl-side help exposes a single flag:

```
<binary> --fdl-schema
```

which prints its options as JSON to stdout and exits:

```json
{
  "description": "DDP validation and benchmark suite for flodl",
  "options": {
    "model": {
      "type": "string",
      "description": "Run specific model",
      "default": "all",
      "choices": ["logistic", "mlp", "lenet", "resnet", "resnet-graph",
                  "char-rnn", "gpt-nano", "conv-ae", "all"]
    },
    "mode": {
      "type": "string",
      "description": "Run specific DDP mode",
      "default": "solo-0",
      "choices": ["solo-0", "solo-1", "nccl-sync", "nccl-cadence",
                  "nccl-async", "cpu-sync", "cpu-cadence", "cpu-async", "all"]
    },
    "epochs":      { "type": "int",   "description": "Override epoch count" },
    "lr-scale":    { "type": "float", "description": "Multiply default LR" },
    "validate":    { "type": "bool",  "description": "Check against baseline" },
    "baseline":    { "type": "path",  "default": "baselines/baseline.json" },
    "tolerance":   { "type": "float", "default": 0.15 },
    "seed":        { "type": "int",   "default": 42 },
    "report":      { "type": "path",  "description": "Analyze runs and write report" }
  }
}
```

### Supported types

| Type       | Shell form                              | Completion hint             |
|------------|-----------------------------------------|------------------------------|
| `string`   | `--key value`                           | `choices:` list if declared  |
| `int`      | `--key 42`                              | none                         |
| `float`    | `--key 0.5`                             | none                         |
| `bool`     | `--key` (flag, no value)                | none                         |
| `path`     | `--key ./foo`                           | file path completion         |
| `list[T]`  | `--key a,b,c` or repeated `--key a ...` | T's completion per item      |

Kept deliberately small. Richer types belong in the binary's own
validation, not in the schema.

### fdl-side behavior

1. At project-load time, invoke `<entry> --fdl-schema` if the binary
   supports it. Cache output under `.fdl/schema-cache/<cmd>.json`.
2. Cache invalidates on binary mtime change.
3. `fdl <cmd> --help` renders from the schema (command description,
   options table, jobs list).
4. `fdl <cmd> <job> --help` renders the resolved options (schema +
   job defaults + env overlay + env vars + argv).
5. `fdl completions` integrates: `choices:` drive completion values,
   `type: path` drives file path completion.

### Fallback chain

If `<binary> --fdl-schema` fails or isn't supported:

1. **`--help`-parsing fallback**: invoke `<binary> --help`, parse the
   text (works for any `clap`-based binary; best-effort for hand-rolled).
2. **Pass-through**: if parsing fails, show "Options (from underlying
   command):" with raw `--help` text verbatim. No validation.

Adopting the schema is opt-in and purely additive. Existing commands
keep working.

### Validation (opt-in)

```yaml
# sub-command fdl.yaml
options:
  strict: true   # reject unknown flags at the fdl level before invoking
```

Default behavior stays pass-through. `strict: true` validates the argv
against the schema. This preserves "don't break working usage" — adding
schema support to an existing command surfaces help without risking
runtime rejections.

### Rendered help — the payoff

```
$ fdl ddp-bench --help
DDP validation and benchmark suite for flodl

USAGE:
    fdl ddp-bench [<job>] [OPTIONS]

JOBS:
    quick          Fast smoke test (linear, 1 epoch)
    full-sweep     All models, all DDP modes
    validate       Check convergence against structured baselines
    nccl-cadence   NCCL cadence for all models

OPTIONS:
    --model <MODEL>         Run specific model  [default: all]
                            [possible: logistic, mlp, lenet, resnet,
                             resnet-graph, char-rnn, gpt-nano, conv-ae, all]
    --mode <MODE>           Run specific DDP mode  [default: solo-0]
                            [possible: solo-0, solo-1, nccl-sync,
                             nccl-cadence, nccl-async, cpu-sync,
                             cpu-cadence, cpu-async, all]
    --epochs <N>            Override epoch count
    --lr-scale <F>          Multiply default LR
    --validate              Check against baseline
    --baseline <PATH>       [default: baselines/baseline.json]
    --tolerance <F>         [default: 0.15]
    --seed <N>              [default: 42]
    --report <PATH>         Analyze runs and write report

Run 'fdl ddp-bench <job> --help' for job-specific defaults.
```

Tab-completion:

```
$ fdl ddp-bench --model <TAB>
logistic  mlp  lenet  resnet  resnet-graph  char-rnn  gpt-nano  conv-ae  all

$ fdl ddp-bench --mode <TAB>
solo-0  solo-1  nccl-sync  nccl-cadence  nccl-async  cpu-sync  cpu-cadence  cpu-async  all

$ fdl ddp-bench --baseline <TAB>
# (file path completion because type: path)
```

---

## 3. The `flodl-args` crate: one struct, everything derived

### Goal

A binary declares its options **once**, as a Rust struct. The derive
macro generates:

- argv parser
- `--help` handler
- `--fdl-schema` handler
- field-level validation

No YAML mirror. No manual parsing loop. No second `print_help` function.
The struct definition *is* the source of truth.

### API shape

```rust
use flodl_args::{Args, parse_or_schema};

/// DDP validation and benchmark suite for flodl.
#[derive(Args, Debug)]
struct Cli {
    /// Run specific model (or "all")
    #[arg(default = "all",
          choices = &["logistic", "mlp", "lenet", "resnet", "resnet-graph",
                      "char-rnn", "gpt-nano", "conv-ae", "all"])]
    model: String,

    /// Run specific DDP mode (or "all")
    #[arg(default = "solo-0", choices = &DdpMode::ALL_NAMES)]
    mode: String,

    /// Override epoch count
    epochs: Option<usize>,

    /// Override batches per epoch
    batches: Option<usize>,

    /// Override batch size
    batch_size: Option<usize>,

    /// Multiply default LR (auto-scales when epochs > default)
    lr_scale: Option<f64>,

    /// Output directory
    #[arg(default = "runs")]
    output: String,

    /// Dataset cache directory
    #[arg(default = "data")]
    data_dir: std::path::PathBuf,

    /// Live dashboard port
    monitor: Option<u16>,

    /// Check results against baselines
    validate: bool,

    /// Save results as baseline
    save_baseline: bool,

    /// Baseline file
    #[arg(default = "baselines/baseline.json")]
    baseline: std::path::PathBuf,

    /// Validation tolerance (0.0-1.0)
    #[arg(default = 0.15)]
    tolerance: f64,

    /// RNG seed
    #[arg(default = 42)]
    seed: u64,

    /// Analyze runs and print report (optional FILE to save)
    report: Option<Option<String>>,

    /// Show available models and modes
    list: bool,
}

fn main() {
    // Handles --fdl-schema and --help automatically;
    // returns parsed struct or exits cleanly.
    let cli: Cli = parse_or_schema();
    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> flodl::tensor::Result<()> {
    // business logic using `cli.model`, `cli.mode`, etc.
    // No arg parsing. No help text. Nothing to forget.
}
```

### Conventions

- Doc comments (`///`) become option descriptions.
- `Option<T>` means "not required."
- `bool` means "switch" (no value).
- `Option<Option<T>>` means "optional with optional value"
  (e.g. `--report` alone vs `--report foo.md`).
- Field names snake_case in Rust, rendered as kebab-case in argv
  (`batch_size` -> `--batch-size`).
- `#[arg(...)]` attributes:
  - `default = <literal>` — explicit default
  - `choices = &[...]` — allowed values (drives completion + validation)
  - `name = "..."` — override the CLI name (rare)
  - `short = 'x'` — add a short form (e.g. `-v`)
  - `env = "..."` — read from environment variable when not in argv

### What ddp-bench shrinks by

- 65 lines of `while i < args.len() { match ... }` — gone
- 35 lines of `print_help` — gone
- Two places that had to stay in sync — consolidated into one struct

`main.rs` becomes ~20 lines (struct definition + `fn main` + `fn run`).

### Dependency policy

`flodl-cli` already follows a zero-external-crate principle (GPU detection
via `nvidia-smi`, HTTP via `curl`, unzip via `unzip`). `flodl-args`
should fit the same constraint where feasible.

Two deployment options:

- **Zero-dep**: hand-rolled `proc-macro` + argv parsing, ~500-800 LOC.
  More work but preserves the no-external-crate story.
- **Clap-backed (opt-in feature)**: thin wrapper over `clap` + `clap_derive`
  that adds the `--fdl-schema` handler. Much smaller to implement but
  pulls in `clap`, `syn`, `quote`.

Start with the clap-backed variant to de-risk the ergonomics, then
evaluate whether the zero-dep rewrite is worth the effort once the API
is settled.

### Clap-based binaries (third-party adoption)

Binaries that already use `clap` get fdl support via a tiny adapter:

```rust
use flodl_args_clap::fdl_schema_from;

fn main() {
    let cmd = Cli::command();   // clap's Command
    if std::env::args().any(|a| a == "--fdl-schema") {
        println!("{}", fdl_schema_from(&cmd));
        return;
    }
    let cli = Cli::parse();
    // ...
}
```

This walks `clap`'s `Command` tree and emits our JSON schema. No fdl
dependency in the binary beyond this one adapter call.

---

## Rollout order

1. **Document the schema format.** This section, written down. Done.
2. **fdl-side rendering.** `flodl-cli/src/run.rs` learns to render the
   JSON schema as `--help` output; `completions.rs` picks up `choices`
   and `type: path`. Schema still has to be written by hand in the YAML
   as an interim step. ~200 LOC.
3. **Schema-from-binary contract.** `flodl-cli` tries `<entry>
   --fdl-schema` before falling back to the YAML schema or parsed
   `--help`. Cache under `.fdl/schema-cache/`. ~100 LOC.
4. **`flodl-args` crate.** New workspace member. Derive macro +
   `parse_or_schema()`. The biggest chunk (~500-800 LOC depending on
   dep policy).
5. **Migrate ddp-bench.** First real user. Proves both the crate and
   the schema format. Shrinks `main.rs` by ~100 lines.
6. **Migrate `flodl-cli` itself.** Dogfood the crate. Validates the
   story all the way down.
7. **Multi-environment inheritance.** Orthogonal to the schema work,
   can ship in parallel or after. ~300 LOC. `fdl config show` is the
   critical UX piece.

Steps 1-3 are pure win with no new crate to maintain. Steps 4-6 are
the ergonomic endgame. Step 7 is the scale-up story.

---

## Why this matters

The scale-up story — laptop to datacenter in one YAML file — only works
if the config layer can carry the weight. Four things have to be true:

- **Options are self-documenting.** Otherwise every new scale gets a new
  "what do these flags mean" learning cost. The `struct Cli` approach
  makes that impossible to skip: add a field, add a doc comment, done.
- **Environments compose without duplication.** Otherwise `fdl.cloud.yml`
  and `fdl.prod.yml` become divergent codebases. Inheritance + `config
  show` keep them honest.
- **The frontend stays flat.** Otherwise new features leak out as new
  command-line surface, and the YAML-as-frontend promise dies. The
  schema contract means any capability added to the underlying binary
  surfaces in `fdl -h` automatically.
- **The right path is the easiest path.** Adopting `struct Cli` is
  strictly cheaper than hand-rolling argv parsing — fewer lines, no
  `print_help` to maintain, no YAML mirror to keep in sync. And the
  payoff is immediate: early validation (unknown values and bad types
  caught at fdl's boundary before Docker spin-up), "did you mean"
  suggestions from `choices:`, typed tab-completion, and a declaration
  that AI agents can consume in one read. No "silent this-value-doesn't-
  exist" failures surviving into the binary. Design principle: any
  contract that relies on "please keep these two places in sync"
  decays; this one has one place.

Together these make `fdl.yaml` a *stable* project interface — one that
research code, infra code, and AI agents can all target without
coordinating on which flag moved where.
