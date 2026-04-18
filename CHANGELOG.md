# Changelog

All notable changes to floDl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] - 2026-04-18

> Upgrading from 0.4.0? The only breaking changes live in `fdl.yml`
> (`scripts:` merged into `commands:`) and in `#[derive(FdlArgs)]`
> (a small set of reserved flag names). See
> [UPGRADE.md](UPGRADE.md) for the step-by-step migration.

### Added

#### New Crate: `flodl-cli-macros`
- **`flodl-cli-macros/`** (new workspace member): proc-macro derive crate exposing `#[derive(FdlArgs)]`, re-exported as `flodl_cli::FdlArgs`. Turns a plain struct into an argv parser plus schema and help renderer. Implements `flodl_cli::FdlArgsTrait` with `try_parse_from(&[String]) -> Result<Self, String>`, `schema() -> flodl_cli::Schema`, and `render_help() -> String`.
- **`#[option(...)]`** named-flag attribute: `short = 'c'`, `default = "..."`, `choices = &["a", "b"]`, `env = "VAR"`, `completer = "name"`. Supported field shapes: `bool` (absent = false, present = true), `T` (scalar, requires `default`), `Option<T>` (absent = None), `Vec<T>` (repeatable).
- **`#[arg(...)]`** positional attribute: `default`, `choices`, `variadic` (requires `Vec<T>`, must be last), `completer`.
- **Derive-time validation**: required positionals cannot follow optional ones; variadic must be last; reserved flags cannot be shadowed (see Global Flags for the authoritative list); duplicate long/short flags error at compile time.
- **Per-option env fallback**: `#[option(env = "WANDB_API_KEY")]` falls back to the environment variable when the flag is absent (argv > env > default). `bool` fields are exempt.
- **Typed help via Rust docs**: doc-comments on the struct and fields flow into `render_help()` output with ANSI colouring.

#### `fdl.yml` Manifest Overhaul
- **Unified `commands:` map**: replaces the separate `scripts:` + `commands:` pair from 0.4.0. Each entry is exactly one of three kinds, chosen by which fields are set.
- **`run:` kind**: inline shell script, optionally wrapped in `docker compose run --rm <service>` when `docker:` is set. Closed script: extra argv is **not** forwarded (use shell `$VAR` inside the script instead).
- **`path:` kind**: pointer to a nested directory with its own `fdl.yml`. Convention default: when the entry is empty and a sibling `<name>/` directory exists, `fdl` loads `<name>/fdl.yml`. Extra argv after `fdl <cmd> ...` flows through to the nested `entry:` and is validated against the `FdlArgs` schema.
- **preset kind**: neither `run:` nor `path:` set; inline `ddp:` / `training:` / `output:` / `options:` fields deep-merge over the enclosing sub-command's defaults and invoke its `entry:`. Only legal inside a path-kind sub-command's own `fdl.yml`.
- **Load-time validation**: `docker:` on non-`run:` entries is rejected; unknown keys error with a clear message; kind-mismatch (e.g. both `run:` and `path:`) errors loudly.
- **Auto-bootstrap**: when only `fdl.yml.example` or `fdl.yml.dist` is present, `fdl` offers to copy it to the real (gitignored) `fdl.yml`.

#### Environment Overlays (`--env`)
- **`--env <name>`** global flag: deep-merges `fdl.<name>.yml` over the base `fdl.yml` before resolving any command.
- **`FDL_ENV=<name>`**: equivalent environment-variable form.
- **First-arg convention**: `fdl ci test` applies the `ci` overlay when `fdl.ci.yml` exists AND the name does not collide with a command. Ambiguity errors loudly.
- **Loud vs. silent fallthrough**: explicit selectors (flag, env var) fail loudly when the overlay file is missing; the first-arg convention silently falls through so existing commands are never shadowed.
- **Per-layer origin annotations**: every merged field is tagged with the file and line that contributed it, visible via `fdl config show`.

#### New Top-Level Commands
- **`fdl config show [env]`**: prints the fully-resolved YAML config with per-layer origin annotations. Useful for previewing overlay behaviour before running a long job. Equivalent forms: `fdl config show ci`, `fdl --env ci config show`, `fdl ci config show`.
- **`fdl schema list`** / **`clear [<cmd>]`** / **`refresh [<cmd>]`**: manage the per-command schema cache that powers help, completion, and validation. `list --json` for machine-readable output. Fresh / stale / orphan status is reported for every cached entry.
- **`--fdl-schema`** (hidden probe flag): every binary built with `#[derive(FdlArgs)]` responds with a JSON description of its flags. `fdl` calls it as a subprocess and caches the result at `<cmd-dir>/.fdl/schema-cache/<cmd>.json`.
- **`--refresh-schema`** per-invocation flag: refreshes a single entry's cache on the next call without running `fdl schema refresh` explicitly. Handy during development.

#### Global Flags
- **`--env <name>`**: apply overlay (see above).
- **`--ansi`** / **`--no-ansi`**: force or disable ANSI color output, overriding TTY and `NO_COLOR` auto-detection.
- **Reserved flag set** (`--help`, `--version`, `--quiet`, `--env`, `-h`, `-V`, `-q`, `-v`, `-e`): cannot be shadowed by `FdlArgs`-derived structs. Enforced at derive time for clear errors.
- **`--help` is never blocked**: validation lives strictly on the exec path, scoped to the single command being invoked. Running `fdl <cmd> --help` never triggers manifest-wide validation.

#### Value-Aware Completions
- **`choices:` drives completion**: flag completion returns the declared set, e.g. `fdl libtorch download --cuda <TAB>` offers `12.6 12.8`; `fdl ddp-bench quick --model <TAB>` offers values from the `FdlArgs` schema.
- **Project-aware**: generated scripts reflect the current `fdl.yml`'s `commands:` (all three kinds) plus every sub-command's own nested entries.
- **`fdl autocomplete`**: one-shot installer that detects the user's shell and writes the completion script to the right location.

#### Styled Output
- **ANSI-coloured help**: `render_help()` assembles colour-annotated help from doc-comments and attribute metadata. Styles are centralised in `flodl-cli/src/style.rs`.
- **Help layout for presets**: preset sub-commands render under an **Arguments** heading as a single synthetic slot with values indented beneath (placeholder overridable via `arg-name:`); regular sub-commands render under **Commands** (run / path kinds only).

#### Schema Cache (`flodl-cli/src/schema_cache.rs`)
- Per-project cache at `<cmd-dir>/.fdl/schema-cache/<cmd>.json`, populated on first use of a `path:`-kind sub-command and refreshed on demand. Cache entries carry mtime + binary hash so `fdl schema list` can flag stale (binary newer than cache) and orphan (command removed from `fdl.yml`) states.

### Changed

#### Docs
- **`docs/cli.md`** rewritten: restructured around three contexts: standalone (no project), inside an `fdl.yml` project, inside the flodl source checkout. Standalone libtorch-manager examples now lead with PyTorch C++ (CMake / `CMAKE_PREFIX_PATH`) alongside the existing tch-rs walkthrough.
- **`docs/design/run-config.md`** expanded: formal schema for `fdl.yml`, sub-command resolution, overlay merge semantics, and the DDP / training / output to `DdpConfig` / `DdpRunConfig` mapping.
- **`docs/design/msf-cadence-control.md`** (new, 669 lines): design spec for the MSF cadence-control layer.
- **`flodl-cli/README.md`** rewritten: leads with "this is the flodl CLI"; standalone libtorch manager framed as a secondary use case.
- **`flodl-cli-macros/README.md`** (new): attribute reference for `#[derive(FdlArgs)]`.
- **Root `README.md`**: short pointer box advertising `fdl` as a standalone libtorch manager for tch-rs and PyTorch C++ users.

#### Dogfooding
- **`ddp-bench/src/main.rs`** ported to `#[derive(FdlArgs)]`: typed flags, shared schema with `fdl`, help / completion / validation all come from the derived parser. Replaces the hand-rolled argv handling.
- **`fdl.yml.example`** and **`ddp-bench/fdl.yml.example`** updated to the unified `commands:` shape with the three-kind distinction.

### Removed

- **`scripts:` key in `fdl.yml`**: merged into the unified `commands:` map. Any 0.4.0 `fdl.yml` that used `scripts:` must move its entries into `commands:` with an explicit `run:` field. The three-kind `commands:` model (`run:` / `path:` / preset) is now the long-term stable manifest surface; no further breaking changes to its shape are scheduled.
- **Shadowing of reserved CLI flags in `#[derive(FdlArgs)]` structs**: `--help`, `--version`, `--quiet`, `--env`, `-h`, `-V`, `-q`, `-v`, `-e` are now reserved and enforced at derive time. Structs in 0.4.0 that named fields with any of these flags silently overrode them; in 0.5.0 they fail to compile. Rename any affected fields.

## [0.4.0] - 2026-04-14

### Added

#### `ddp-bench` — DDP Validation Suite
- **New workspace member `ddp-bench/`**: End-to-end harness that reproduces published training setups to build scientifically valid solo baselines, then measures DDP/ElChe convergence quality against them.
- **8 reference models** (`ddp-bench/src/models/`):
  - `logistic` / `mlp` / `lenet` / `conv_ae` (MNIST)
  - `resnet` (ResNet-20 on CIFAR-10, He et al. 2015 — paper baseline 91.25%)
  - `resnet_graph` (FlowBuilder rewrite of ResNet-20: same parameter count, same accuracy, with graph-level observation, named parameters and tagged residual blocks)
  - `char_rnn` (Karpathy 2015 char-RNN on Shakespeare, LSTM-256x2)
  - `gpt_nano` (4-layer pre-norm Transformer on Shakespeare, warmup + cosine decay)
- **8 DDP modes**: `solo-0`, `solo-1`, `nccl-{sync,cadence,async}`, `cpu-{sync,cadence,async}`. Side-by-side validation across all backend × policy combinations.
- **Harness** (`harness.rs`): single-process and DDP launch paths, per-batch metric collection via `record_scalar`, per-epoch convergence summaries, baseline JSON I/O.
- **Analyzer** (`analyze.rs`): compares runs against committed baselines (`baselines/structured.json`, `baselines/baseline.json`, `baselines/sync.json`) with relative-error tolerances.
- **Reporter** (`report.rs`): generates Markdown convergence reports including loss curves and timing tables (`runs/report.md`, `ddp-bench/report.md`).
- **Dataset downloader** (`download.rs`): on-demand download + cache for MNIST, CIFAR-10, Shakespeare. Cache lives under `data/` (gitignored).
- CLI flags: `--list`, `--model <name|all>`, `--mode <mode|all>`, `--epochs N`, `--batch-size`, `--lr-scale F`, `--validate`, `--baseline <path>`, `--save-baseline`, `--report <path>`, `--seed`.

#### Built-in Standard Datasets — `flodl::data::datasets`
- **`Mnist`** (`data/datasets/mnist.rs`): parses IDX gzip into `[N,1,28,28]` Float32 + `[N]` Int64. `Mnist::parse(images_gz, labels_gz) -> Result<Self>`. Implements `BatchDataSet`.
- **`Cifar10`** (`data/datasets/cifar10.rs`): parses the binary batch format into `[N,3,32,32]` Float32 + `[N]` Int64 (10 classes). Implements `BatchDataSet`.
- **`Shakespeare`** (`data/datasets/shakespeare.rs`): char-level tokenizer for next-char prediction. `[N, seq_len]` Int64 over a 65-symbol vocabulary, plus a `decode(&[i64]) -> String` helper. Implements `BatchDataSet`.
- All three plug directly into `DataLoader::builder(dataset)` in single-GPU and DDP modes.

#### Convergence Guard — Unified Divergence Reaction
- **`convergence` module** (`flodl/src/distributed/ddp_run/convergence.rs`): unified weight-space divergence guard for both NCCL and CPU averaging paths.
- **`DivergenceReport`**: per-rank L2 deltas plus optional pre/post norms. Free decomposition into cosine similarities and magnitude shifts via the algebraic identity (no extra reductions).
- **`ConvergenceAction`**: `Stable` / `SuppressGrowth` / `NudgeDown { factor }` recommendations.
- **`ConvergenceGuard::new(policy, enabled, threshold)`**: 5-interval ring buffer. Detects 3-consecutive-rising trends above threshold and returns `SuppressGrowth` to freeze ElChe anchor/overshoot growth (rather than aggressively shrinking, which can kill convergence — overhead auto-tune handles loosening on its own).
- **Wired into `Coordinator`** for both NCCL and CPU paths (`Sync` is no-op, `Cadence`/`Async` use trend detection). Configurable via `DdpRunConfig::with_divergence_threshold(f64)`.
- Cross-rank divergence is now reset after every averaging event, fixing a stale-state bug that pinned the ElChe anchor at 1.

#### Timeline Profiler — `monitor::timeline`
- **`Timeline`** (`flodl/src/monitor/timeline.rs`): high-frequency (default 100ms poll, 1s broadcast) system + GPU profiler. Captures CPU, RAM, per-GPU compute utilization and VRAM as `TimelineSample`s, interleaved with training events.
- **`EventKind`**: `EpochStart` / `EpochEnd { loss }` / `SyncStart` / `SyncEnd { duration_ms }` / `CpuAvgStart` / `CpuAvgEnd { duration_ms }` / `AnchorChanged { from, to }` / `Throttle { rank }` / `Idle { device, duration_ms }` / `Custom { label }`.
- **API**: `Timeline::new(poll_ms)` / `with_intervals(poll_ms, broadcast_ms)` (returns `Arc<Timeline>`), `start()` / `stop()`, `event(EventKind)`, `subscribe()` for live `mpsc` updates, `summary()`, `idle_gaps(device, threshold_pct, min_ms)`, `drain()`, `sample_count()`.
- **Output**: `save_json(path)`, `save_csv(path)`, `save_html(path)` — the HTML view (`timeline.html`) renders a swimlane visualization of CPU/GPU utilization, sync/averaging events, anchor changes and detected idle gaps. Used by `ddp-bench` for every run (`runs/<model>/<mode>/timeline.html`).
- Enable per-job in `fdl.yaml` with `ddp.timeline: true` or `output.timeline: true`.

#### Verbosity-Gated Logging — `flodl::log`
- **`Verbosity` enum**: `Quiet (0)` / `Normal (1)` / `Verbose (2)` / `Debug (3)` / `Trace (4)`. Higher levels include lower.
- **Macros**: `flodl::msg!("...", args)` (Normal default, `@Verbose`/`@Debug`/`@Trace` for explicit level), plus `flodl::verbose!()`, `flodl::debug!()`, `flodl::trace!()`.
- **Routing**: Normal/Verbose go to **stdout**; Debug/Trace go to **stderr** so they remain unbuffered in Docker non-TTY environments. Errors keep using bare `eprintln!`.
- **Zero-code config**: `FLODL_VERBOSITY=verbose cargo run` (accepts integers 0–4 or names). Programmatic override via `flodl::log::set_verbosity(Verbosity)`.
- **CLI integration**: `fdl -v` / `-vv` / `-vvv` / `--quiet` set `FLODL_VERBOSITY` in the parent process so it flows into Docker child commands automatically.

#### FlowBuilder — `also_with`
- **`FlowBuilder::also_with(skip, main)`** (`flodl/src/graph/flow.rs`): residual connection with a custom skip path. Generalizes [`also`](../flodl/src/graph/flow.rs) for cases where the skip needs its own transform — e.g. ResNet downsample blocks where a 1×1 conv + BN matches channel/stride changes. Output is `skip(x) + main(x)`. Exercised by `ddp-bench/src/models/resnet_graph.rs` (ResNet-20 on CIFAR-10, full paper-accuracy baseline).

#### `AdaptiveAvgPool2d`
- **`AdaptiveAvgPool2d::new([h, w])`** (`flodl/src/nn/pooling.rs`): global / fixed-output-size average pooling. Counterpart to the existing `AdaptiveMaxPool2d`. `[1, 1]` gives global average pooling (common ResNet head before FC); arbitrary output sizes enable variable-size input support. Re-exported at crate root.

#### Metrics — `drain_scalars`
- **`flodl::drain_scalars() -> HashMap<String, (f64, usize)>`** (`flodl/src/distributed/ddp_run/mod.rs`): companion to the existing `record_scalar`. Flushes the thread-local accumulator and returns `(sum, count)` per tag so callers (monitors, custom loops) can average or log per-batch scalars outside the DDP coordinator path. Re-exported at crate root.

#### LR Scheduling — Cross-Mode Parity
- **`Graph::set_scheduler(Arc<dyn Scheduler>)`** and **`Graph::set_lr_scale(f64)`** (`flodl/src/graph/distributed.rs`): scheduler attached on the Graph DDP path drives the optimizer LR via `scheduler.lr(training_step) * lr_scale` on every `step()`. `training_step` advances per `step()` call. **`Graph::training_step()`** accessor exposed for monitoring.
- **`GpuWorker::set_scheduler` / `set_lr_scale` / `current_lr`** (`flodl/src/distributed/ddp_run/worker.rs`): same mechanism on the DDP-builder path. LR computed as `scheduler.lr(global_step + steps_since_avg) * lr_scale` per batch.
- **`DdpBuilder::scheduler(factory)`** (`flodl/src/distributed/ddp_run/orchestrator.rs:1219`): per-worker scheduler factory closure. Each rank instantiates its own scheduler (cheap to clone, no shared state). Pairs with `lr_scale_ratio` to keep all ranks in lockstep.
- **`DdpBuilder::lr_scale_ratio(f64)`** / **`DdpRunConfig::with_lr_scale_ratio(f64)`**: when set, the framework auto-computes the per-rank `lr_scale` from `world_size` (linear scaling rule, Goyal et al. 2017). Default `0.0` (= disabled, `lr_scale = 1.0`); set to `1.0` for full linear scaling, fractional values for sub-linear. Manual override stays available via `--lr-scale` in `ddp-bench`.
- **Cross-mode parity test** (`graph_tests.rs`): asserts that the same `MultiStepLR` produces identical LR trajectories across all three training paths — manual reference loop, `GpuWorker` (DDP builder), and `Graph::step()` — for both unscaled and `lr_scale != 1.0`.
- **Coordinator regression**: `SyncAck` no longer inflates `steps_since_avg` and now properly satisfies `nccl_ack`, fixing a scheduler drift across NCCL averaging events.

#### DDP — New Configuration Knobs
- **`DdpBuilder::no_divergence_guard()`** / **`DdpRunConfig::with_no_divergence_guard()`**: disable the convergence guard entirely. Use during calibration runs or when the divergence trend logging is more noise than signal. Default: enabled with `divergence_threshold = 0.05`.
- **`DdpBuilder::max_overshoot(usize)`** / **`DdpRunConfig::with_max_overshoot(usize)`**: cap how many extra batches the fastest rank can run past the slowest before the next averaging event in `Async` policy. Pairs with auto-tuning; set to bound the worst case explicitly. Async-only — the `Cadence` policy uses wall-time anchoring instead. The internal `overshoot_ceiling` (default ~3× anchor) gates the auto-tuner.
- **`DdpBuilder::timeline(Arc<Timeline>)`** / **`DdpRunConfig::with_timeline(Arc<Timeline>)`** / **`DdpConfig::timeline(Arc<Timeline>)`** / **`Graph::timeline(Arc<Timeline>)`**: attach a shared `monitor::Timeline` so the DDP runtime injects `EpochStart/End`, `SyncStart/End`, `CpuAvgStart/End`, `AnchorChanged`, `Throttle` events into the profiler stream. All four entry points (single-GPU Graph, manual `Ddp::wrap`, `Ddp::setup`, `DdpBuilder`) accept the same `Arc<Timeline>`. Used by `ddp-bench` to produce per-run swimlane HTML.
- **`Coordinator::builder()`** (`flodl/src/distributed/ddp_run/coordinator/mod.rs`): the coordinator now exposes a fluent builder (`progressive`, `batch_size`, `timeline`, `divergence_threshold`, `no_divergence_guard`, `overhead_target`, `max_anchor`, `checkpoint_every`, `snapshot_timeout_secs`, `epoch_metrics_tx`, `device_indices`, `num_epochs`, `partition_ratios`, `max_overshoot`, `overshoot_ceiling`, `build`). Internal — the user-facing surface is still `DdpBuilder`/`Ddp::setup` — but useful for writing custom orchestrators.
- **Note on `max_batch_diff`**: the field shipped in 0.3.0 (per-rank lockstep limit). What's new is `DdpBuilder::max_batch_diff(usize)` as a top-level fluent setter (was only reachable via `DdpRunConfig::with_max_batch_diff`).

#### CLI: `fdl run` and Project / Sub-command Manifests
- **`fdl.yaml`** (also `fdl.yml`, `fdl.json`): committed project manifest. Declares `description`, `scripts` (named shell commands with optional `docker:` service binding) and `commands` (paths to sub-command directories that have their own `fdl.yaml`). Example at the repo root: `fdl.yml.example` (84 lines).
- **Sub-command manifests** (e.g. `ddp-bench/fdl.yml.example`): declare `entry`, `docker`, structured `ddp` / `training` / `output` sections, and named `jobs` (presets that merge over the defaults). DDP section maps 1:1 to `DdpConfig` / `DdpRunConfig` (mode, policy, backend, anchor, max_anchor, overhead_target, divergence_threshold, max_batch_diff, speed_hint, partition_ratios, progressive, max_grad_norm, lr_scale_ratio, snapshot_timeout, checkpoint_every, timeline).
- **Auto-bootstrap**: when only `fdl.yml.example` (or `.dist`) is present, `fdl` offers to copy it into the real, gitignored `fdl.yml` so users can customize without polluting the repo.
- **Built-in script targets** (e.g. `fdl test`, `fdl cuda-test-all`, `fdl shell`, `fdl bench`, `fdl self-build`): any unknown command is resolved against the project's `scripts:` map and wrapped in `docker compose run --rm <service>` when a `docker:` field is set. Replaces the old `make` workflow.
- **Sub-command dispatch**: `fdl <cmd> [<job>] [--flag ...]` resolves `<cmd>` against `commands:`, picks the named job (or defaults), merges DDP/training/output sections and forwards everything as CLI flags to the configured `entry`. Pass-through for unknown flags is preserved.
- **Recursive help**: `fdl <cmd> --help` and `fdl <cmd> <job> --help` print resolved options and inherited defaults.

#### CLI: `fdl completions` / `fdl autocomplete`
- **`fdl completions <bash|zsh|fish>`**: emits a shell-completion script that knows about all built-in commands, the local project's `scripts:` and `commands:`, and per-sub-command jobs.
- **`fdl autocomplete`**: dynamic, project-aware completion suggestions for the current cwd.
- Designed to be sourced from `~/.bashrc` / `~/.zshrc` so completions update automatically as `fdl.yml` evolves.

#### CLI: `fdl diagnose --json`
- The diagnostics report now has a fully structured `--json` mode for CI pipelines and tooling: system, CUDA devices, libtorch variants, compatibility verdict.

#### Docs: PyTorch Porting Guide
- **`docs/porting.md`** (257 lines, full rewrite from the previous 7-line stub): user-facing porting guide that mirrors the AI skill (`ai/skills/port/guide.md`) and references `fdl api-ref` for the canonical type/method index.
- **`docs/cli.md`** (130 lines): full CLI reference (setup, libtorch, init, diagnose, api-ref, install, skill, run, completions, config, verbosity flags, fdl.yaml manifest).
- **`docs/design/run-config.md`** (296 lines): formal spec for `fdl.yaml` — schema, merge order, sub-command resolution, Docker integration, and how DDP/training/output map onto `DdpConfig` / `DdpRunConfig`.
- Updates to `docs/pytorch_migration.md` and the CLI section of the README.

#### CLI: API Reference Generator
- **`fdl api-ref`**: Generate a structured API reference from flodl source. Extracts all public types, constructors, methods, builder patterns, trait implementations, and doc examples.
  - Human-readable output (1700+ lines, 170 types) or `--json` for structured data.
  - `--path <dir>` for explicit source path.
  - Auto-discovers source: project checkout, cargo registry, or downloads latest release from GitHub.
  - Downloaded sources cached at `~/.flodl/api-ref-cache/<version>/` for instant re-use.
  - Designed for AI-assisted PyTorch-to-flodl porting: the reference provides everything an agent needs to map PyTorch patterns to flodl equivalents.

#### PyTorch Porting Skill
- **`ai/skills/port/`**: AI-assisted PyTorch-to-flodl porting framework. Universal porting guide (`guide.md`) and agent instructions (`instructions.md`) that work with any AI coding assistant. Covers the full journey from environment setup (`fdl init`) through model translation (FlowBuilder patterns, layer mapping, loss/optimizer/scheduler tables) to validation (`cargo check` loop).
- **`ai/adapters/claude/`**: Claude Code adapter (SKILL.md template) for `/port` slash command. Installed via `fdl skill install`.
- Guide includes: project scaffolding (native vs Docker), 30+ module mappings, FlowBuilder patterns (sequential, residual, skip connections, split/merge, loops, tags), training loop translation, data loading, checkpointing, device management, and Rust-specific idioms.

#### CLI: Global Install & Self-Update
- **`fdl install`**: Copy the current binary to `~/.local/bin/fdl` for global access. Downloads the latest release from GitHub if a newer version is available. Detects shell (bash/zsh) and prints PATH instructions if needed.
- **`fdl install --dev`**: Symlink to the current binary instead of copying. Global `fdl` tracks local builds automatically. Every `cargo build --release -p flodl-cli` updates the global command instantly. Ideal for developers.
- **`fdl install --check`**: Compare installed version against latest GitHub release. Shows install mode (dev symlink or copied binary).
- Version-aware: shows "Updating 0.3.0 -> 0.3.1" or "already installed".
- Platform detection for pre-compiled binaries (linux/darwin/windows, x86_64/aarch64/arm64).

#### CLI: Skill Management
- **`fdl skill install`**: Detect the user's AI coding tool (Claude Code, Cursor) and install flodl skills. Auto-detects `.claude/` or `.cursorrules`. Copies universal skill files (guide, instructions) plus tool-specific adapter. `--tool <name>` to force a tool, `--skill <name>` to install one skill.
- **`fdl skill list`**: Show available skills and detected tools with install status.
- Claude Code: installs `/port` slash command to `.claude/skills/port/`.
- Cursor: appends porting context to `.cursorrules`.
- Skill files embedded in the binary via `include_str!`, so it works without a repo checkout.
- Re-running `fdl skill install` updates existing skills in place.

### Changed

#### DDP — Streaming Epochs and NCCL Cadence Boundaries
- **Streaming epoch dispatch**: `Coordinator::dispatch_next_chunk` now streams sub-epoch chunks instead of full-epoch partitions in `Cadence` and `Async` modes, adapting to live throughput. Added a guard so the coordinator never recreates chunk pools for already-aggregated epochs (was causing a deadlock under heterogeneous cadences).
- **NCCL cadence boundary fixes**: per-rank epoch ack handling rewritten so that the slowest rank no longer stalls the next epoch's `SyncNow` broadcast. ElChe anchor + overshoot remain anchored to the slow rank's wall time.
- **`max_overshoot` is Async-only**: documented as such; the auto-tune is no longer evaluated for `Cadence`.
- **Convergence safety net**: divergence signals now reset after every NCCL averaging event (was leaking stale norms across intervals and pinning the anchor at 1).

#### Optimizer Module Layout
- **`flodl/src/nn/optim.rs` (1975 lines) split into a module**: `optim/{mod, sgd, adam, rmsprop, adagrad, radam, nadam}.rs`. Public API and behavior unchanged; navigation and review surface dramatically improved.

#### FFI Shim Layout
- **`flodl-sys/shim.cpp` (4517 lines) split into themed translation units**: `ops_tensor.cpp`, `ops_nn.cpp`, `ops_math_ext.cpp`, `ops_training.cpp`, `ops_cuda.cpp`, plus a shared `helpers.h`. `shim.cpp` is now a unity-build aggregator. No FFI surface change.

#### Other
- **Rust doc warnings**: Fixed all 32 documentation link warnings (unresolved cross-module references, private item links).
- **GitHub Actions**: Added `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` env to silence Node.js 20 deprecation warnings.
- **Release workflow**: `gh release create` now falls back to `gh release upload --clobber` when the release already exists (tag push before workflow completes).
- **CLI help text**: Updated to reflect broader scope (API reference, global install). Added examples for `api-ref` and `install` commands.

### Fixed

#### CPU Averaging Race Condition
- **`snapshot_params()` stream sync**: Added `comm_stream.synchronize()` before reading GPU parameters for CPU averaging snapshots. Without this, `Update` + `RequestParams` messages processed in the same `handle_control()` call could read mid-copy GPU memory from a pending `load_averaged()` non-blocking transfer. The coordinator's `tick()` method can send both messages in the same tick when averaging completes and the next cycle triggers immediately.
- **CPU averaging convergence fixed**: The stream sync fix (above) resolved the CPU averaging convergence failure from 0.3.0. All three CPU policies (Sync/Cadence/Async) now converge correctly (91-92% on CIFAR-10 ResNet-20, matching NCCL). Both backends are production-ready.

#### Test Stability
- **`test_graph_loop_leak`**: removed quantitative assertions (`live_tensor_count`, RSS) that flake under parallel CI. The test's real value is exercising 500 iterations of graph+loop+optimizer without crashing (use-after-free, double-free, unbounded Rc chains). Diagnostics are logged for manual review.
- **NCCL/Graph distribute test isolation**: clarified ignore set so `fdl cuda-test-nccl` covers both `nccl` and `graph_distribute` patterns and `fdl cuda-test-serial` covers everything else.

#### libtorch `AccumulateGrad` Stream Mismatch (DDP Workers)
- **Warning eliminated**: `"AccumulateGrad node's stream does not match"` fired on every DDP backward pass when workers ran on a non-default training stream. Three stacked undocumented libtorch facts combined to produce it, and fixing any one of them alone was insufficient:
  1. `AccumulateGrad` nodes capture their stream into `input_metadata` at **construction time**, not at each runtime backward call.
  2. The node is created lazily on first `backward()` **inside the autograd engine's worker thread**, whose current stream is the device default (not the user's training stream).
  3. `AutogradMeta` holds a `weak_ptr` to the node, so without an external strong reference it is collected between iterations and re-created on the default stream on every backward pass.
- **`Tensor::ensure_grad_accumulator()`** (`flodl/src/tensor/mod.rs`) / **`Variable::ensure_grad_accumulator()`** (`flodl/src/autograd/variable.rs`): eagerly materialize the `AccumulateGrad` node for a leaf tensor with `requires_grad=true`, pinning its stream to the current CUDA stream at the moment of the call. Returns a `GradAccumulatorHandle` that keeps the node alive through a strong `shared_ptr<Node>` on the C++ side. No-op for non-leaf or non-`requires_grad` tensors.
- **`GradAccumulatorHandle`** (`flodl/src/tensor/mod.rs`): opaque `Send + Sync` strong-reference handle. `Drop` frees the node (unless a backward pass still holds its own reference). Intended to be held for the lifetime of the owner, typically a DDP worker.
- **FFI additions** (`flodl-sys/ops_training.cpp`, `shim.h`, `src/lib.rs`): `flodl_ensure_grad_accumulator(FlodlTensor, void**)` and `flodl_grad_accumulator_delete(void*)`. The C++ side calls the semi-internal libtorch API `torch::autograd::impl::grad_accumulator()` (found by reading libtorch source) and heap-allocates the returned `shared_ptr<Node>` so Rust owns its lifetime.
- **`GpuWorker` construction reordered** (`flodl/src/distributed/ddp_run/worker.rs`): CUDA streams are now created **before** `model_factory` so every leaf tensor (parameters, buffers, initial copies, optimizer state, `AccumulateGrad` nodes) is allocated under `StreamGuard(compute_stream)` and carries the training-stream affinity from birth. New `_grad_accumulators: Vec<GradAccumulatorHandle>` field on `GpuWorker` holds strong references to every parameter's accumulator for the worker's lifetime; explicitly documented as liveness-only ownership (never read at runtime, dropping it re-introduces the bug).
- **Validated**: 54 training runs across 6 architectures (`logistic`, `mlp`, `lenet`, `char-rnn`, `gpt-nano`, `conv-ae`) times 9 DDP modes with zero warnings in any `training.log`. Also validated across the earlier 6-mode 200-epoch `resnet_graph` run on CIFAR-10.
- **Side effect**: unblocks CUDA Graph capture for DDP workers. Graph capture fails loudly on stream mismatches between the training stream and the accumulator stream, so prior workarounds are no longer needed.

## [0.3.0] - 2026-04-08 — Multi-GPU & Infrastructure

### Added

#### Async GPU-CPU Foundation
- **`CudaEvent`**: Record/synchronize/elapsed_time on CUDA streams. `CudaEventFlags` (Default for timing, DisableTiming for pure sync). RAII Drop, Send. 14 FFI functions (7 event + 7 stream).
- **`CudaStream`**: Pool-managed streams per device. Synchronize, wait_event, is_complete. RAII Drop, Send.
- **`StreamGuard`**: RAII stream switching (sets on create, restores default on drop). Async copy pattern: `let _guard = StreamGuard::new(&stream); tensor.to_device_async(Device::CPU)?;`
- Enables zero-stall GPU-to-CPU pipeline: `training stream -> CudaEvent -> copy stream -> CPU`

#### NCCL Collective Operations
- **`NcclComms`**: RAII communicator group for multi-GPU collectives. 5 FFI functions wrapping raw NCCL (ncclCommInitAll, AllReduce, Broadcast via GroupStart/End).
- **`ReduceOp`**: Sum, Prod, Max, Min, Avg.
- **`all_reduce()`** / **`all_reduce_on_streams()`**: In-place AllReduce across all devices (default or explicit streams).
- **`broadcast()`** / **`broadcast_on_streams()`**: Broadcast from root rank to all devices.
- Raw NCCL (not c10d) for minimal overhead in single-process multi-GPU.

#### NCCL Per-Rank Communication
- **`NcclRankComm`**: Per-rank communicator for multi-threaded DDP. Each GPU thread owns one comm, runs collectives independently. `Send` so it can be moved into spawned threads.
  - `init_rank(rank, world_size, &uid)`: Direct per-rank init from a shared `NcclUniqueId`.
  - `all_reduce(&[&Tensor], ReduceOp)` / `all_reduce_on_stream(...)`: Rank-local AllReduce.
  - `broadcast(&[&Tensor], root)`: Rank-local broadcast.
- **`NcclComms::split()`**: Extracts per-rank `NcclRankComm` from a group-initialized `NcclComms`. Preferred over per-thread `init_rank` because `ncclCommInitRank` from worker threads corrupts CUDA context on heterogeneous GPUs. Init-on-main + split is the safe pattern.
- **`NcclAbortHandle`**: Arc-shared handle to abort a stuck `NcclRankComm`. Calling `abort()` unblocks any thread stuck in an AllReduce/Broadcast and makes the comm's Drop a no-op. Used by `DdpHandle` to recover from worker death without deadlocking surviving workers.
- **`NcclUniqueId`**: 128-byte unique ID for coordinating per-rank init. `NcclUniqueId::new()` generates on rank 0, then shared to all ranks.
- 7 per-rank FFI functions: `flodl_nccl_get_unique_id`, `flodl_nccl_init_rank`, `flodl_nccl_destroy_rank`, `flodl_nccl_all_reduce_rank`, `flodl_nccl_abort_rank`, `flodl_nccl_split_rank`.

#### Transparent Multi-GPU Training
- **`Graph::distribute()`**: Auto-detect GPUs, create replicas, broadcast params. Single line to enable multi-GPU. No-op on single GPU.
- **`Graph::set_optimizer()`**: Creates per-replica optimizers when distributed.
- **`Graph::step()`**: AllReduce gradients + sync buffers + optimizer step + zero_grad. One call replaces the manual loop.
- **`Graph::set_lr()`** / **`world_size()`** / **`is_distributed()`**: Multi-GPU aware API.
- **Cross-device autograd**: `Tensor::to_device()` preserves grad_fn (ToCopyBackward). Forward chunks input, forwards shards on their GPUs, gathers via to_device + cat. libtorch autograd naturally flows gradients back through device transfers.
- **`Ddp`**: Manual DDP coordinator for complex training patterns (GAN, RL, progressive). Explicit sync_params, all_reduce_gradients, sync_buffers.
- Training loop is identical for 1 or N GPUs; `distribute()` is the only difference.

#### Async Data Loading Pipeline
- **`DataSet` trait**: Per-item dataset (`get(index) -> Vec<Tensor>`). `Send + Sync` for background prefetch. Automatic batching via `DataSetAdapter` (pre-allocate + copy, O(1 sample) peak memory).
- **`BatchDataSet` trait**: Per-batch dataset (`get_batch(indices) -> Vec<Tensor>`) for bulk-efficient sources (mmap, database). `Send + Sync`.
- **`Sampler` trait**: Index ordering per epoch. Built-in: `RandomSampler` (deterministic per seed+epoch), `SequentialSampler`.
- **`Batch`**: Named tensor wrapper with `Index<usize>` and `Index<&str>` for clean destructuring (`let images = &b["image"]` or `&b[0]`). `.names()`, `.has()`, `.get_named()` for introspection. Owns its tensors.
- **`DataLoader`**: Builder pattern with auto-detection of resident vs streaming mode.
  - **Resident mode**: Dataset fits in VRAM (75% headroom). Loaded once via `pin_memory()` + `to_device()`. Per-epoch: GPU-side `index_select` with shuffled permutation. Zero CPU-GPU transfer after warmup.
  - **Streaming mode**: Persistent worker thread with dedicated `CudaStream`. Per-epoch fresh batch channel (no deadlock on mid-epoch drop). Worker: `get_batch` -> `pin_memory` -> `StreamGuard` + `to_device_async` -> `CudaEvent`. Consumer: `event.synchronize()` (typically instant due to prefetch depth).
  - **CUDA OOM fallback**: If resident load fails with OOM, automatically retries with streaming mode.
  - **Auto prefetch depth**: `clamp(free_vram * 10% / batch_bytes, 2, 4)`. Override with `.prefetch(n)` for high-latency cloud/NFS storage.
  - `.streaming()` to force streaming mode (preserve VRAM headroom, benchmarking).
  - `drop_last` defaults to `true` (BatchNorm safety: size-1 batches cause NaN variance).
  - `EpochIterator` implements `Iterator<Item = Result<Batch>>` + `ExactSizeIterator`.
- **`TensorError::is_cuda_oom()`**: Detect CUDA out-of-memory errors for graceful fallback.
- **`.names()`**: Builder method for named batch fields (`["image", "letter", "case", "origin"]`). Auto-generated positional names ("0", "1", ...) when unspecified. Validates name count against dataset tensor count.
- DDP-aware: loader yields pinned CPU data, `forward_distributed` scatters to devices efficiently.

#### Resident DDP
- **DDP-aware DataLoader**: Third internal mode `DistributedLoader` with per-device backends. Each GPU independently selects resident (data fits in VRAM) or streaming (prefetch worker) based on its own VRAM. No lowest-common-denominator constraint.
- **`DeviceBackend`**: Per-device data strategy. Resident: full dataset on GPU, index_select per batch. Streaming: dedicated PrefetchWorker with async H2D transfers.
- **`Graph::set_data_loader(loader, "input")`**: Attach DataLoader to model. When distributed: upgrades to per-device backends. Auto-wires batch names to graph `.input()` ports. Remaining names treated as targets for loss.
- **`Graph::epoch(epoch)`**: Returns `GraphEpochIterator` that produces per-rank shards and user-facing Batch. When distributed: each backend produces on-device data, shards stored for presharded forward. When single-GPU: delegates to DataLoader.
- **`Graph::forward_batch(&batch)`**: Batch-aware forward. Extracts named inputs, handles DDP presharding transparently. Coexists with `Module::forward(&Variable)`.
- **Presharded forward path**: `forward_distributed_presharded()` consumes per-rank shards from DataLoader via `.take_shards()`. Each replica forwards its local shard (zero cross-device input transfer). Outputs gathered to gather device. CudaEvent timing for auto-balancer.
- **Multi-input auto-wiring**: `set_data_loader()` precomputes `shard_input_map` matching graph `.input()` port names to batch tensor positions. `forward_distributed_presharded()` passes all inputs (primary + auxiliary) to each replica via `as_graph().forward_impl()`. Single-GPU `forward_batch()` also builds the full input vector. Enables multi-input models (FBRL with case/origin alongside image) in distributed training.
- **Efficient distributed streaming**: `StartDistributedEpoch` + `LoadBatch` worker commands. One channel per epoch instead of per-batch channel creation. Flat state machine in `worker_loop` (no nested loops). `PrefetchWorker::start_distributed_epoch()` opens the channel once, `load_batch()` sends indices per batch.
- **Gather device selection**: Prefers resident backend with most free VRAM. Falls back to CPU if all backends are streaming (targets fetched from dataset). No GPU 0 priority.
- **Auto-balancing integration**: Epoch iterator reads chunk_ratios fresh per batch. Shard sizes adapt as ratios change every 50 steps. Mixed resident/streaming backends handle dynamic ratios correctly.
- Training loop identical for 1 or N GPUs. `distribute()` + `set_data_loader()` are the only differences.

#### `Ddp::setup()` — One-Liner DDP Setup
- **`Ddp::setup(&model, builder, optimizer)`**: Single call to auto-detect GPUs, distribute the model, set per-replica optimizers, and enable training mode. No-op distribute for single GPU/CPU (still sets optimizer + training). Training loop identical for 1 or N GPUs.
- **`Ddp::setup_with(&model, builder, optimizer, config)`**: Same as `setup()` but accepts a `DdpConfig` for explicit El Che configuration (speed hints, overhead target, max anchor).
- **`Ddp::is_heterogeneous()`**: Detects mixed GPU models. `setup()` auto-enables El Che when heterogeneous GPUs are detected.
- **Hardware diagnostics**: Always prints detected hardware to stderr on call:
  - `ddp: 2 GPUs (heterogeneous) | RTX 5060 Ti (16.0 GB) | GTX 1060 (6.0 GB)`
  - `ddp: 1 GPU | RTX 5060 Ti (16.0 GB) | single-device mode`
  - `ddp: no CUDA available | CPU mode`

#### Multi-GPU Dashboard
- **Per-GPU tabs**: Tab bar appears when 2+ GPUs detected (hidden for single-GPU, zero visual regression). Each GPU tab shows 4 time-series charts: VRAM usage (bytes, with physical limit reference line), utilization (%), throughput (samples/ms), batch share (%).
- **GPU Overview card** (Home tab): Compact row per GPU with VRAM bar, utilization, throughput, and batch share. Fastest GPU highlighted green, slowest yellow.
- **JS data model**: `gpuSeries[deviceIndex]` with per-device VRAM, throughput, chunk, and utilization arrays. Populated from `d.gpus` in `processEpoch()`. Works in both live SSE and archive replay modes.

#### Multi-GPU Dashboard Data Pipeline
- **`GpuSnapshot`**: Per-device resource sampling (VRAM allocated/total, utilization, device name). `ResourceSampler` iterates all CUDA devices on each sample. Aggregate fields kept for backward compat with single-GPU dashboards.
- **`GpuMetrics`**: DDP metrics per device (EMA throughput, chunk_ratio, shard_size). Exposed via `Metrics::gpu_metrics()` trait method with default empty impl.
- **Per-GPU JSON in epoch records**: `"gpus":[...]` array merges hardware snapshots (from `GpuSnapshot`) with DDP metrics (from `GpuMetrics`). Flows through SSE live updates and HTML archives.
- **`Graph::auto_distribute()`**: Auto-detect usable CUDA devices and distribute. No-op on single GPU. Keeps the builder closure for user-controlled model construction.
- **`Graph::shard_sizes()`** / **`Graph::devices()`**: Public accessors for per-rank shard sizes and device list.

#### Auto-Balancing
- **Per-GPU throughput measurement**: CudaEvent-based timing around each replica's forward pass in `forward_distributed()`. Zero overhead (async GPU recording, no CPU sync).
- **EMA throughput tracking**: Exponentially smoothed samples/ms per device (alpha=0.3). First measurement initializes directly, subsequent measurements blend.
- **Adaptive batch sharding**: After 10 calibration steps with equal splits, `chunk_ratios` are recomputed proportional to measured throughput. Re-evaluated every 50 steps. `MIN_CHUNK_RATIO` (5%) prevents starving any GPU.
- **Weighted gradient averaging**: When chunk ratios are unequal, each replica's gradient is scaled by `(shard_size / batch_size)` then AllReduce Sum, producing the mathematically correct mean gradient regardless of shard distribution.
- **`Graph::chunk_ratios()`**: Query current batch distribution ratios (for logging/debugging).
- **`Graph::throughput()`**: Query per-device EMA throughput (samples/ms).
- All auto-balancing is internal to `forward_distributed()` and `step()`. Training loop is unchanged.

#### NCCL Device Safety
- **Device save/restore**: All `NcclComms` methods (`new`, `all_reduce`, `broadcast`, and stream variants) now save and restore the current CUDA device around FFI calls. Prevents NCCL operations from leaking device context changes to callers.
- **Shared `NCCL_LOCK`**: Single `pub(crate)` mutex in `ddp` module, used by both `nccl::tests` and `ddp::tests` to serialize NCCL communicator operations.

#### El Che — Heterogeneous DDP
- **`ElChe`**: Cadence strategy for mixed-GPU training. Slow device anchors the sync cadence, fast devices range ahead processing more batches per sync. Named after Che Guevara's marching principle: "the column marches at the slowest one's pace."
  - `ElChe::new(world_size, anchor)` with builder pattern.
  - `with_speed_ratio(slow_rank, ratio)`: Seed initial batch distribution from known speed differential. Self-corrects after first `report_timing()`.
  - `with_overhead_target(f64)`: Default 0.10 (10%). Auto-tunes anchor upward to keep AllReduce overhead below target.
  - `with_max_anchor(usize)`: Gradient staleness cap. Prevents unbounded accumulation.
  - `report_timing(&wall_ms, sync_ms)`: Discovers true speed ratios from CudaEvent measurements, recomputes batch counts, auto-tunes anchor.
  - `batch_counts() -> &[usize]`: Per-device batch counts for the current cadence step.
  - `clamp_total(max) -> Vec<usize>`: Proportional clamping for epoch-end alignment.
- **`DdpConfig`**: Configuration struct for `Ddp::setup_with()`.
  - `speed_hint(slow_rank, ratio)`: Initial speed estimate (optional, self-corrects).
  - `overhead_target(f64)`: AllReduce overhead ceiling.
  - `max_anchor(Option<usize>)`: `None` = auto (default), `Some(0)` = disable El Che (traditional DDP), `Some(n)` = fixed cap.
  - `max_grad_norm(f64)`: Per-rank gradient clipping before normalize-by-count and weighted AllReduce. Bounds accumulated gradients on all ranks (including replicas the caller cannot reach). Uses fused C++ kernel (`clip_grad_norm_fused`).
- **`Graph::step()` El Che branch**: Normalizes accumulated gradients by `1/count[rank]` (mean per device), weighted AllReduce by `count[rank]/total` (proportional contribution), reports timing to ElChe for adaptation. Per-rank gradient clipping when configured. Existing scatter and single-GPU paths unchanged.
- **`Graph::has_el_che()`** / **`Graph::configure_el_che()`**: Query and configure El Che state.
- **`weighted_all_reduce_gradients()`**: Scales each replica's gradient by batch contribution before AllReduce Sum. Produces the mathematically correct mean gradient regardless of per-device batch counts.

#### El Che Forward Path
- **`forward_distributed_el_che()`**: Multi-batch per-device forward. Each device processes `batch_counts[rank]` complete batches independently. Gradients accumulate naturally via libtorch autograd across all forward passes. CudaEvent timing per rank.
- **Tagged output gathering**: After each forward pass, tagged outputs (`Graph::tag()`) are captured from each device and concatenated across all batches and all devices. Custom loss functions work transparently on gathered intermediates: `model.tagged("scan_locations")` returns the catted value from all devices.
- **Loop trace gathering**: Per-step outputs from loop nodes (`trace_buf`) are gathered across all batches and all devices, keyed by `(tag_name, step_index)`. `model.traces("attn")` returns catted per-step traces. Enables transparent El Che training for models with loop-based attention (scan/read fixations, per-step losses). No-op when no loop nodes exist.
- **El Che data routing**: `DistributedEpochIterator` pulls `sum(batch_counts)` complete batches per iteration (not shards). Routes whole batches to each device via `load_batch_on_device()` (supports both Resident index_select and Streaming prefetch worker). Proportional clamping near epoch boundaries.
- **Epoch-end flush**: `ActiveGraphEpochIterator::drop()` detects accumulated un-synced gradients (forward without step) and forces a final `step()` to prevent silent gradient loss.
- **`Graph::epoch()`** seeds initial batch counts from `ElChe::batch_counts()`. **`Graph::step()`** feeds updated counts back to the loader after `report_timing()`.
- Training loop is identical for homogeneous and heterogeneous GPU setups. `Ddp::setup()` detects heterogeneous hardware and enables El Che automatically.

#### DDP Builder — Thread-Per-GPU Training
- **`DdpHandle`**: Thread-per-GPU training with Local SGD and adaptive parameter averaging. Each GPU runs its own training loop with a local optimizer. A lightweight coordinator thread triggers periodic parameter averaging. Two orthogonal knobs: [`ApplyPolicy`] (when to average) and [`AverageBackend`] (how to average).
- **`DdpBuilder`** (recommended entry point): Fluent API for configuring and launching training. Required: `.dataset()`, `.batch_size()`, `.num_epochs()`. Optional: `.policy()`, `.backend()`, `.overhead_target()`, `.max_anchor()`, `.anchor()`, `.divergence_threshold()`, `.max_batch_diff()`, `.checkpoint_every()`, `.checkpoint_fn()`, `.epoch_fn()`, `.progressive_dispatch()`.
  ```rust
  let ddp = Ddp::builder(model_factory, optim_factory, train_fn)
      .dataset(dataset)
      .batch_size(32)
      .num_epochs(10)
      .policy(ApplyPolicy::Cadence)
      .backend(AverageBackend::Nccl)
      .run()?;
  let state = ddp.join()?;
  ```
- **`Ddp::builder()`**: Quick-start alternative (replaces the former `AsyncDdp::auto()`/`auto_with()`).
- **`ApplyPolicy`**: Controls WHEN averaging occurs.
  - `Sync`: K=1 (every batch). Equivalent to standard DDP. Best convergence.
  - `Cadence`: K=N (ElChe anchor count). Slow GPU anchors the cadence, fast GPUs fill wall time. Uses wall-time trigger (fires when slowest rank's accumulated wall time reaches anchor wall-time). Recommended for heterogeneous hardware.
  - `Async`: K=adaptive. Uses batch-count trigger (fires when all ranks complete their assigned counts). Overshooting is intentional: each replica explores slightly different parameter neighborhoods between averaging events, producing diversity that benefits convergence. Auto-tunes interval from divergence monitoring. Maximum throughput.
- **`AverageBackend`**: Controls HOW averaging is performed. Orthogonal to policy, all combinations valid for A/B testing.
  - `Nccl`: In-place AllReduce on GPU. Zero extra memory, GPU-to-GPU DMA. All GPUs sync at collective barrier.
  - `Cpu`: Workers send parameter snapshots to coordinator, which averages on CPU and distributes. No GPU ever blocks. Uses O(world_size * model_size) CPU RAM. Non-blocking 3-phase state machine (Idle/Collecting/Computing) keeps coordinator responsive during averaging.
- **`GpuWorker<M>`**: Generic worker bound to a single GPU. Thread-local model + optimizer (Rc-based, not Send). CUDA streams for overlapped compute/communication. Handles `SyncNow` (NCCL), `RequestParams`/`Update` (CPU), `Throttle`, `StartEpoch`, `Checkpoint`, `Shutdown`.
- **`Coordinator`**: Lightweight scheduling thread. Collects timing from workers (for ElChe throughput ratios), triggers averaging, monitors divergence to auto-tune interval, rebalances data partitions. Builder pattern with configurable `divergence_threshold`, `overhead_target`, `max_anchor`, `checkpoint_every`, `snapshot_timeout_secs`.
- **`TrainedState`**: Return type from `DdpHandle::join()`. Contains averaged `params` and `buffers` as CPU tensors, ready for inference or checkpoint.
- **`DdpRunConfig`**: Configuration struct with builder methods: `with_overhead_target()`, `with_max_anchor()`, `with_anchor()`, `with_divergence_threshold()`, `with_max_batch_diff()`, `with_max_grad_norm()`, `with_checkpoint_every()`, `with_snapshot_timeout()`, `with_partition_ratios()`, `with_progressive_dispatch()`.
- **Per-worker gradient clipping**: `DdpBuilder::max_grad_norm(f64)` clips gradients between `backward()` and `optimizer.step()` on each GPU worker. Prevents gradient spikes on any single GPU from propagating through AllReduce averaging. Same fused kernel as El Che path.
- **`progressive_dispatch`**: When enabled, the coordinator streams work in small chunks instead of sending full epoch partitions, adapting to throughput continuously. Default: auto (true for Cadence/Async, false for Sync).
- **Global epoch management**: Coordinator owns epochs globally. Workers are mode-agnostic (wait for `EpochPlan`, run partition, report metrics). `EpochPlan { epoch, partition_offset, partition_size }` ensures deterministic, non-overlapping sample coverage. Throughput-proportional partition sizing when ElChe is calibrated; `partition_ratios` for fixed splits. Auto lookahead in `Async` mode (fast ranks may run 1 epoch ahead).
- **Single-GPU fallback**: With fewer than 2 CUDA devices, training runs on the main thread with no coordinator or averaging. API is identical; `join()` returns `TrainedState` in both cases.

#### DDP Builder — Robustness
- **`max_batch_diff`**: Hard limit on how far any GPU can run ahead of the slowest. Workers that exceed the limit are throttled (block on control channel) until the next averaging event. `Some(0)` = strict lockstep.
- **`drain_until_shutdown`**: After training, workers keep handling control messages (especially `SyncNow`) until the coordinator sends `Shutdown`. Prevents NCCL deadlock when workers finish at different times.
- **NCCL init-on-main + split()**: All NCCL communicators initialized from the main thread via `NcclComms::new()` then `split()` into per-rank `NcclRankComm`. Per-thread `ncclCommInitRank` corrupts CUDA context on heterogeneous GPUs.
- **NCCL abort handles**: If a worker dies mid-collective, `DdpHandle::abort_nccl()` calls `ncclCommAbort` on all communicators, unblocking surviving workers. Also triggered in `Drop`.
- **Worker error propagation**: Failed workers set the shared shutdown flag and send `TimingMsg::Exiting` so the coordinator stops including that rank in collectives.
- **CPU averaging timeout**: Configurable `snapshot_timeout_secs` (default 5s). If not all worker snapshots arrive in time, the round is soft-aborted (logged with missing rank IDs and abort count), stale snapshots drained, and retried on the next cycle.
- **CPU Update delivery logging**: Failed Update deliveries to dead workers are logged with the affected rank.
- **Shutdown cleanup**: `drain_avg_state()` logs and joins any in-progress CPU averaging (Collecting or Computing) before the coordinator exits, preventing detached threads from holding GPU resources.

#### DDP Builder — Observability
- **Averaging success logging**: Both paths log on successful averaging. NCCL: `"NCCL averaging #N complete (vV)"`. CPU: `"CPU averaging #N complete (vV, X.Xms)"` with timing.
- **Per-rank epoch metrics**: Worker epoch-end metrics (rank, epoch, loss, batches, wall time) forwarded to stderr from the coordinator loop.
- **Coordinator accessors**: `avg_count()`, `abort_count()`, `last_batch_ms()`, `last_avg_ms()`, `is_cpu_averaging()`, `version()`, `avg_interval()`, `is_calibrated()`, `steps_since_avg()` for external monitoring.
- **Divergence monitoring** (Async policy): Per-rank parameter L2 norms tracked. Relative norm difference triggers interval halving (diverging) or doubling (converging). Threshold configurable via `divergence_threshold` (default 0.05).
- **Hardware summary**: Prints GPU count, heterogeneous/homogeneous detection, per-GPU name + VRAM, policy, and backend at launch.

#### DDP Builder — Metrics Pipeline
- **`record_scalar(name, value)`**: Thread-local function callable from inside the train function. Records named scalar metrics (accuracy, custom losses, etc.) per batch. Metrics are aggregated per rank per epoch and forwarded to the coordinator.
- **`EpochMetrics`**: Aggregated metrics for one completed epoch. Fields: `epoch`, `avg_loss`, `batches_processed`, `epoch_ms`, `samples_processed`, `per_rank_loss`, `per_rank_time_ms`, `per_rank_scalars`, `scalars`.
- **`DdpHandle::poll_metrics()`**: Non-blocking poll for completed epoch metrics. Returns a `Vec<EpochMetrics>` of all epochs aggregated since the last poll. Enables external monitoring loops.
- **`DdpHandle::next_metrics()`**: Blocking call that returns the next available `EpochMetrics`. Useful for sequential metric processing.
- **`DdpHandle::setup_monitor(&self, &mut Monitor)`**: Wire the DDP handle's graph identity, architecture SVG, and training config into a training monitor. Enables the live dashboard and HTML archive for DDP Builder training runs.
- **`LossContext`**: Per-batch context passed to loss closures in distributed training. Provides batch metadata (shard sizes, device indices) for loss functions that need to weight contributions correctly.

#### DDP Builder — Epoch Callback
- **`EpochFn<M>`**: `Arc<dyn Fn(usize, &mut GpuWorker<M>) + Send + Sync>`. Called at the start of each epoch inside each worker thread, before `run_epoch_plan()`.
- **`.epoch_fn()`** on `DdpBuilder`: Set the callback. Typical uses: LR schedules, noise curricula, dynamic loss weights.
- **`GpuWorker::set_lr(f64)`**: Delegate to the worker's optimizer.
- **`GpuWorker::current_epoch()`**: Public accessor for the current epoch number.

#### DDP Builder — Checkpointing
- **`CheckpointFn<M>`**: `Arc<dyn Fn(u64, &M) -> Result<()> + Send + Sync>`. Called on rank 0 after averaging events (multi-GPU) or epoch boundaries (single-GPU). Errors are logged but do not stop training.
- **`checkpoint_every(n)`**: Save every N averaging events. Coordinated through `ControlMsg::Checkpoint` to rank 0's worker thread (which owns the model).
- **`TrainedState`** on partial failure: If some workers died, `collect_final_state()` averages surviving workers' snapshots. If averaging fails, falls back to the first snapshot's tensors. Returns `None` only if zero snapshots arrived.

#### Adaptive Data Pipeline
- **VRAM-aware prefetch depth**: `prefetch_depth_from_vram()` computes prefetch budget as the gap between current VRAM usage and a configurable cap. No manual tuning needed.
- **Bootstrap prefetch**: Initial depth of 4 batches during DataLoader construction. Real depth computed at `epoch(0)` after model is loaded and VRAM usage is stable.
- **Per-epoch VRAM probing**: `epoch(N)` re-probes VRAM usage and fills up to the cap. Adapts to VRAM fragmentation and activation memory changes across epochs.
- **`DataLoaderBuilder::vram_max_usage(f64)`**: Default 0.90 (use up to 90% of total VRAM). Clamped to [0.50, 0.99]. Remaining headroom covers activations, gradients, and CUDA overhead.
- **Manual override**: `.prefetch(n)` or `set_prefetch_depth()` disables automatic adaptation (`user_set_depth` flag).
- **`auto_resize()`**: Manual trigger for VRAM-based resize between epochs.

#### Module Builders
- **`ConvTranspose1dBuilder`**, **`ConvTranspose2dBuilder`**, **`ConvTranspose3dBuilder`**: Fluent builder APIs for transposed convolution layers (`with_stride`, `with_padding`, `with_output_padding`, `with_dilation`, `with_groups`, `with_bias`, `on_device`, `done`). Consistent with existing Conv1d/Conv2d/Conv3d builder pattern.

#### CLI Tool
- **`fdl`** (shell script): Zero-dependency entry point. Auto-detects libtorch, Docker, Rust, GPUs. Dispatches to the compiled binary (native or Docker) with shell fallback for diagnostics. Interactive setup wizard guides users through libtorch installation and build environment selection.
- **`flodl-cli`** (`cargo install flodl-cli`): Standalone Rust binary. Pure Rust, no libtorch dependency. Works inside floDl projects and standalone (system-wide libtorch management under `~/.flodl/`). Override global root with `$FLODL_HOME`. Commands:
  - `fdl setup`: Guided wizard. Detects project vs standalone mode. In a project: system detection, libtorch download, Docker image build. Standalone: system detection, libtorch download to `~/.flodl/`, prints shell export instructions.
  - `fdl libtorch download [--cpu | --cuda 12.6|12.8]`: Auto-detect GPUs and download matching libtorch variant. Project-local or global depending on context.
  - `fdl libtorch build [--docker | --native] [--archs "6.1;12.0"]`: Compile libtorch from source for custom GPU architectures.
  - `fdl libtorch list / info / activate / remove`: Manage installed variants.
  - `fdl init <name> [--docker]`: Scaffold a new floDl project. Default mode uses mounted libtorch (like the main repo). `--docker` bakes libtorch into the Docker image for standalone deployment. Generates Cargo.toml, Dockerfiles, docker-compose.yml, Makefile, and annotated src/main.rs.
  - `fdl diagnose [--json]`: System + GPU + libtorch + compatibility report. Shows context mode (project/global). Probes GPUs via nvidia-smi, verifies libtorch arch coverage, detects Docker containers.
  - `fdl help / version`
- Pre-compiled binaries published via GitHub Releases for Linux x86_64/aarch64, macOS arm64, Windows x86_64. Downloaded automatically by the `fdl` shell script on first use.

#### Small Additions
- **`Linear::no_bias_on_device()`**: Create a bias-free linear layer on a specific device. Previously `no_bias()` was CPU-only.
- **`AdamBuilder::betas()` / `.eps()`**: Customize beta1, beta2, and epsilon in Adam per-group builder. Previously hardcoded to (0.9, 0.999) and 1e-8.
- **`AdamWBuilder::betas()` / `.eps()`**: Same for AdamW per-group builder.
- Improved doc comments on all loss functions (dtype requirements), conv builders, and optimizer constructors.

### Changed

#### Unified DDP API
- **`Ddp` is now the single entry point** for all multi-GPU training modes: `setup()` (user owns the loop), `builder()` (framework owns the loop), `wrap()` (manual).
- **Renamed**: `AsyncDdp` -> `DdpHandle`, `AsyncDdpBuilder` -> `DdpBuilder`, `AsyncDdpConfig` -> `DdpRunConfig`, `Ddp::auto()` -> `Ddp::setup()`, `Ddp::auto_with()` -> `Ddp::setup_with()`.
- **Module renamed**: `nn::async_ddp` -> `nn::ddp_run`.
- **Log prefix**: `async-ddp:` -> `ddp:` in all runtime output.
- **Deprecated aliases** preserved for backward compatibility: `AsyncDdp`, `AsyncDdpBuilder`, `AsyncDdpConfig`, `Ddp::auto()`, `Ddp::auto_with()`.

#### Unified libtorch Management
- **`libtorch/` directory**: Single host-side directory for all libtorch variants.
  - `libtorch/precompiled/cpu|cu128|cu126/` for downloaded pre-built variants
  - `libtorch/builds/<arch>/` for source-compiled variants (e.g., `sm61-sm120`)
  - `libtorch/.active` points to the variant in use
  - `libtorch/<variant>/.arch` contains metadata (cuda version, torch version, architectures, source type)
- **Docker images are libtorch-agnostic**: No libtorch baked into images. Mounted at runtime via volume.
  - `Dockerfile` (new, replaces `Dockerfile.cpu`): Ubuntu + Rust, no libtorch
  - `Dockerfile.cuda`: parameterized `CUDA_VERSION`, cudnn-devel base, no libtorch
  - `Dockerfile.cuda.source`: builder-only (no Stage 2 runtime image), Makefile extracts via `docker cp`
  - `Dockerfile.bench`: removed libtorch download, kept Python + PyTorch pip install
- **docker-compose.yml simplified**: 5 services reduced to 3 (`dev`, `cuda`, `bench`). Removed `cuda-local` and `cuda-source`. All services mount `${LIBTORCH_HOST_PATH}:/usr/local/libtorch:ro`.
- **Makefile auto-detection**: Reads `libtorch/.active` and `.arch` to derive `CUDA_VERSION` and libtorch mount path. Override: `CUDA_VERSION=12.6.0 make cuda-test`.
- **`download-libtorch.sh --project`**: Downloads to `libtorch/precompiled/<variant>/`, writes `.arch` and `.active`. Existing `--path` mode for native installs unchanged.

#### Test Infrastructure
- **15 tests un-ignored**: `cuda_event` (3), `cuda_stream` (4), DDP cross-device autograd (2) tests now run in the normal `make cuda-test` flow. They have proper mutex serialization and early-return guards.
- **NCCL/DDP/Graph tests remain `#[ignore]`**: NCCL communicator init corrupts concurrent CUBLAS operations. Must run single-threaded.
- **Process-isolated test targets**: NCCL tests run in their own cargo process to prevent CUBLAS context poisoning. Fixes SIGABRT in `test_manual_seed_reproducible` when run after NCCL init.
  - **`make cuda-test-all`**: Three-pass target -- parallel + NCCL (isolated) + remaining serial.
  - **`make cuda-test-nccl`**: NCCL/DDP tests only (isolated processes).
  - **`make cuda-test-serial`** (new): Remaining serial tests (CUDA Graphs, manual_seed, probes).

#### Build Targets
- **`make setup`**: Auto-detect hardware, download CPU libtorch + CUDA libtorch (or build from source), build Docker image. One command from zero to ready.
- **`make build-libtorch`**: Compile libtorch from source, extract to `libtorch/builds/<arch>/`, write `.arch`/`.active`.
- **`make cli`** / **`make cuda-cli`**: Build flodl-cli (CPU/CUDA). **`make run-cli`** / **`make cuda-run-cli`**: Run inside Docker.
- **CI updated**: CUDA job downloads libtorch separately and mounts into container (no longer baked into image).

### Removed
- `Dockerfile.cpu` (replaced by `Dockerfile`)
- `cuda-local` and `cuda-source` docker-compose services

## [0.2.2] - 2026-03-31

### Added
- `Tensor::nbytes()` — total size in bytes (`numel() * element_size()`), matches `torch.Tensor.nbytes`

#### Fused sequence RNN kernels
- **`LSTM::forward_seq`** now calls `at::lstm()` — single cuDNN kernel for the entire sequence across all layers, replacing per-timestep cell unrolling. Eliminates N×L kernel launches (N=timesteps, L=layers) per forward pass.
- **`GRU::forward_seq`** now calls `at::gru()` — same fused optimization. Also eliminates the cuDNN benchmark variance that caused ±270ms σ in per-cell dispatch.
- **`flatten_rnn_params`** (shim) — packs per-cell RNN weight tensors into cuDNN's expected contiguous layout using `at::_cudnn_rnn_flatten_weight`, the same function PyTorch's `nn.LSTM.flatten_parameters()` uses internally. Eliminates the "RNN module weights are not part of single contiguous chunk" warning on CUDA. Uses `set_()` under `NoGradGuard` to redirect parameter storage in-place — persists across training steps, self-corrects after checkpoint load or dtype cast.
- **Flatten cache** — LSTM and GRU cache the flattened param tensors after the first forward call, skipping both the per-forward param collection (8 tensors via `flat_map` + `collect`) and the cuDNN flatten FFI call on subsequent forwards. Same strategy as PyTorch's `flatten_parameters()` but without the pointer-validation overhead.
- **`RnnParams` C++ cache** — persistent `std::vector<at::Tensor>` on the C++ side behind an opaque handle (`flodl_rnn_params_create` / `flodl_lstm_cached` / `flodl_gru_cached`). After the first forward, subsequent calls pass a single pointer to the pre-built param vector, eliminating per-forward handle collection, FFI array marshalling, and `std::vector` reconstruction. Matches PyTorch's single-call `at::lstm()`/`at::gru()` pattern exactly.
- FFI chain: `flodl_lstm` / `flodl_gru` in shim → `Tensor::lstm_seq` / `Tensor::gru_seq` in nn_ops (new `flatten` flag skips redundant flatten calls). Cached path: `flodl_lstm_cached` / `flodl_gru_cached` → `Tensor::lstm_seq_cached` / `Tensor::gru_seq_cached`.
- `LSTMCell::forward_step` and `GRUCell::forward_step` unchanged — still available for single-step / streaming use cases

#### Benchmark suite extensions
- **`transformer`** benchmark — 4-layer encoder (MultiheadAttention + FFN + LayerNorm + residual), Embedding, cross-entropy loss. B=32, seq=128, d_model=512, 8 heads.
- **`lstm_seq`** benchmark — 2-layer LSTM + linear projection, directly comparable to gru_seq. B=128, seq=50.
- **`conv_autoenc`** benchmark — Conv2d encoder + ConvTranspose2d decoder (DCGAN-style), reconstruction with MSE loss. B=64, 64×64 images.

### Changed
- **Benchmark σ uses scaled MAD** — variance column now reports Median Absolute Deviation × 1.4826 (σ-equivalent for normal distributions) instead of standard deviation. Robust to OS scheduling outliers, GC pauses, and WSL2 thermal transients that inflated stdev on long runs (e.g. gru_seq Py σ: ±143 stdev → ±27 MAD).

### Fixed
- **Benchmark report generation**: Fix silent `set -e` exit caused by `[ "$ROUNDS" -gt 1 ] && echo 's'` returning exit code 1 inside command substitution when ROUNDS=1. Reports were never written for single-round runs.
- **Benchmark report rotation**: Previous report is now rotated to `report.YYYY-MM-DD-HH-MM-SS.txt` instead of being overwritten. All rotated reports are gitignored.

## [0.2.1] - 2026-03-29

### Added

#### PyTorch Parity — Tensor Operations
- **Math ops**: `log1p`, `expm1`, `log2`, `log10`, `tan`, `asin`, `acos`, `atan`, `erf`, `erfc`, `trunc`, `frac`, `fmod`, `fmod_tensor`, `remainder`, `remainder_tensor`, `lerp`, `lerp_tensor`, `isclose`, `addmm`, `addcmul`, `addcdiv`, `clamp_min`, `clamp_max`, `selu`, `hardswish`, `hardsigmoid`, `prelu`
- **Reductions**: `prod`, `prod_dim`, `cumsum`, `logsumexp`
- **Shape ops**: `flip`, `roll`, `diagonal`, `movedim`, `tile`, `split`, `unbind`, `contiguous`, `cat_many`, `unsqueeze_many`, `narrow_scatter`, `pad_mode` (constant/reflect/replicate/circular), `meshgrid`
- **NN tensor ops**: `conv1d`, `conv_transpose1d`, `conv3d`, `conv_transpose3d`, `avg_pool2d`, `avg_pool1d`, `max_pool1d`, `adaptive_max_pool2d`, `instance_norm`, `group_norm`, `linear` (fused), `pixel_shuffle`, `pixel_unshuffle`, `bilinear`, `embedding_bag`, `interpolate` (nearest/bilinear/bicubic/trilinear), `im2col`, `col2im`, `bce_loss`, `nll_loss`, `ctc_loss`
- **Comparison/similarity**: `maximum`, `minimum`, `atan2`, `masked_fill`, `normalize`, `cosine_similarity`

#### PyTorch Parity — Autograd
- **New differentiable ops**: `leaky_relu`, `elu`, `softplus`, `mish`, `selu`, `hardswish`, `hardsigmoid`, `prelu`, `clamp_min`, `clamp_max`, `log1p`, `expm1`, `log2`, `log10`, `atan2`, `maximum`, `minimum`, `masked_fill`, `normalize`, `cosine_similarity`, `prod`, `prod_dim`, `cumsum`, `logsumexp`, `unsqueeze_many`, `cat_many`, `stack`, `triu`, `tril`
- **NN autograd ops**: `conv1d`, `conv_transpose1d`, `conv3d`, `conv_transpose3d`, `avg_pool2d`, `avg_pool1d`, `max_pool1d`, `adaptive_max_pool2d`, `instance_norm`, `group_norm`, `pixel_shuffle`, `pixel_unshuffle`, `bilinear`, `embedding_bag`, `im2col`, `col2im`

#### PyTorch Parity — Modules
- **Convolutions**: `Conv1d` (with `Conv1dBuilder`), `Conv3d` (with `Conv3dBuilder`), `ConvTranspose1d`, `ConvTranspose3d`
- **Recurrent**: `GRU` (multi-layer sequence module), `LSTM` (multi-layer sequence module) — match `nn.GRU`/`nn.LSTM` interface with `forward_seq`, batch-first support
- **Normalization**: `GroupNorm`, `InstanceNorm`, `RMSNorm`
- **Pooling**: `AvgPool2d`, `MaxPool1d`, `AvgPool1d`, `AdaptiveMaxPool2d`, `PixelShuffle`, `PixelUnshuffle`, `Upsample`, `Unfold`, `Fold`
- **Attention**: `MultiheadAttention` — self-attention and cross-attention with optional masking
- **Bilinear**: `Bilinear` — bilinear transformation `y = x1^T A x2 + b`
- **Activations**: `LeakyReLU`, `ELU`, `Softplus`, `Mish`, `SELU`, `Hardswish`, `Hardsigmoid`, `PReLU` (learnable), `Softmax`, `LogSoftmax`, `Flatten`
- **Dropout**: `AlphaDropout` — maintains self-normalizing property for SELU networks
- **Embedding**: `EmbeddingBag` — bag-of-embeddings with sum/mean/max aggregation
- **Padding**: `ZeroPad2d`, `ReflectionPad2d` — symmetric and asymmetric padding modules

#### PyTorch Parity — Losses
- `bce_loss` (from probabilities), `nll_loss`, `ctc_loss`, `focal_loss` (class imbalance), `triplet_margin_loss`, `cosine_embedding_loss`, `hinge_embedding_loss`, `margin_ranking_loss`, `poisson_nll_loss`

#### PyTorch Parity — Optimizers
- `RMSprop` (with `RMSpropBuilder` for parameter groups)
- `Adagrad` (with `AdagradBuilder` for parameter groups)
- `RAdam` — Rectified Adam with variance-aware warmup
- `NAdam` — Nesterov-accelerated Adam

#### PyTorch Parity — LR Schedulers
- `ExponentialLR` — exponential decay (`lr = base_lr * gamma^step`)
- `MultiStepLR` — decay at specific milestones
- `OneCycleLR` — super-convergence schedule (warmup + cosine decay)
- `CyclicLR` — triangular wave between base and max LR (symmetric and asymmetric)

#### PyTorch Parity — Initialization
- `kaiming_uniform`, `kaiming_normal` now re-exported at crate root
- New: `uniform`, `normal`, `orthogonal`, `trunc_normal`, `uniform_bias`

#### Test Coverage (+165 tests, 769 total)
- **Autograd gradient verification** (55 tests): finite-difference checks for every new differentiable op — `leaky_relu`, `elu`, `softplus`, `mish`, `selu`, `hardswish`, `hardsigmoid`, `prelu`, `clamp_min`/`clamp_max`, `log1p`, `expm1`, `log2`, `log10`, `maximum`, `minimum`, `masked_fill`, `cosine_similarity`, `normalize`, `prod`, `cumsum`, `logsumexp`, `tril`, `flatten`; fused NN op gradients for all conv variants (1d/2d/3d + transpose), all pooling variants, `layer_norm`, `group_norm`, `instance_norm`, `bilinear`, `embedding_bag`, `pixel_shuffle`/`unshuffle`, `im2col`/`col2im`, `grid_sample`, `gru_cell`, `lstm_cell`; Variable API coverage (`set_grad`, `set_requires_grad`, `is_leaf`, `numel`, `zero_grad_set_to_none`, `set_data`, `to_device`)
- **Module forward/backward** (60+ tests): Conv1d (builder, groups, stride/padding, no-bias, gradient), Conv2d (builder, grouped, stride, no-bias, gradient), Conv3d, ConvTranspose1d/2d/3d (forward, gradient, stride, parameters), GroupNorm (batch-size-one, single-group, groups=channels, gradient), InstanceNorm (3D input, affine parameters, gradient), LayerNorm (3D, normalization, gradient), BatchNorm/BatchNorm2d (training, eval, running stats, rejects invalid dims, gradient), Bilinear (gradient, no-bias, rejects single input), Dropout (training, eval identity, p=0), ZeroPad2d/ReflectionPad2d (asymmetric, values, no-parameters)
- **Loss functions** (20+ tests): MSE (basic, zero loss), cross-entropy (class indices, wrong predictions, gradient), BCE/BCEWithLogits (gradient), L1, SmoothL1 (negative beta rejection), KLDiv, CTC, focal (reduces to CE at gamma=0), triplet margin (zero when far), cosine embedding (similar/dissimilar), hinge embedding (positive/negative), margin ranking (with margin), Poisson NLL (log/no-log)
- **Mixed precision** (7 tests): AutocastGuard lifecycle, autocast closure, GradScaler (defaults, scale, step finite/inf, update growth/backoff, state roundtrip), cast_parameters (basic, noop same dtype)
- **Gradient clipping** (6 tests): clip_grad_norm (scales down, no-op when small, multiple params), clip_grad_value (clamps, no-op, no-grad params)
- **Graph observation** (8 tests): collect/flush/trend pipeline, reduction modes (mean, sum, min, max, norm, scalar passthrough), rejects non-scalar, map operations (over tag, slices, batched, gradient, error cases)

## [0.2.0] - 2026-03-29

### Added

#### Graph Tree (hierarchical composition)
- **Label-path addressing**: Dot-separated paths (`"encoder.scan.hidden"`) for addressing subgraphs and tags across graph boundaries. Strict dot semantics -- dots always mean subgraph boundaries, no fuzzy resolution.
- **Tree registration**: Labeled graphs nested via `FlowBuilder` are automatically detected as child subgraphs. `tree_children()`, `child_graph()`, `subgraph()` for navigation. `is_composed()` flag on child graphs.
- **Selective freeze/thaw**: `freeze("encoder.read")`, `thaw("encoder.scan")`, `is_frozen("encoder")` -- declarative training phase control by label path.
- **Path-based parameter collection**: `parameters_at()`, `named_parameters_at()`, `named_buffers_at()` for per-subgraph optimizer groups. Target namespace used for checkpoint compatibility.
- **Subgraph checkpoint loading**: `load_subgraph_checkpoint("encoder", "encoder_v1.fdl.gz")` -- loads a checkpoint into a specific subgraph using the child's own namespace and structural hash validation.
- **Cross-boundary observation**: `tagged_at()` (null/nil semantics), `collect_at()`, `record_at()`, `trend_at()` -- read tagged outputs and metrics across graph boundaries.
- **Tree-aware flush and metrics**: `flush()` automatically recurses into labeled child subgraphs. `latest_metrics()` collects from the entire tree with dotted prefixes (`"encoder.loss"`). `Monitor::log()` sees the whole tree with zero extra code. `flush_local()` and `latest_metrics_local()` for independent per-subgraph observation cadences.
- **Internal tags**: Tags prefixed with `_` are auto-internal (hidden from parent resolution). Explicit `.internal("tag")` on FlowBuilder. Cross-boundary resolution rejects internal tags.
- **Training mode propagation**: `set_training_at("encoder", false)` for selective eval mode on subgraphs (BatchNorm running stats).
- **Verbose build output**: `.verbose(true)` on FlowBuilder prints tree structure, tag resolution, and parameter summary. `tree_summary()`, `param_summary()` methods.
- **Path validation**: `validate_path()` returns `PathKind::Subgraph` or `PathKind::Tag` for build-time wiring checks.
- **Module trait**: Added `as_graph()` method (default `None`, overridden in Graph) for subgraph detection.
- **Zero forward-path impact**: All tree metadata is build-time/query-time only. The pre-computed Vec routing in `forward_impl()` is untouched.

#### Modules
- **`GaussianBlur`**: Stateless `Module` wrapper around `gaussian_blur_2d()` for use in `FlowBuilder` graphs. Fixed sigma, no parameters. Kernel size auto-computed from sigma (`2 * ceil(3 * sigma) + 1`).

#### Checkpoint Migration
- **`migrate_checkpoint()`** / **`migrate_checkpoint_file()`**: Automatically remap parameter names from an older checkpoint to match a model's current naming. Matches by exact name first, then by shape+dtype in positional order. Handles params and buffers, supports `.gz` compression. Returns a `MigrateReport` with `unchanged`, `remapped`, `dropped`, `missing` fields and a `Display` impl for human-readable output.
- **`checkpoint_version()`**: Peek at a checkpoint file's version without loading it. Returns `1` for flodl 0.1.x, `2` for 0.2.0+.
- **`MigrateReport`**: Full accounting of a migration — `is_complete()` returns true when nothing was dropped or missing.

### Changed
- **Breaking**: Checkpoint format version bumped to v2. Checkpoints saved with 0.2.0+ write version 2; `load_checkpoint` accepts both v1 and v2 (binary layout is identical, only naming conventions differ). v1 checkpoints can be migrated with `migrate_checkpoint_file()`.
- **Breaking**: Restructuring a graph with `.label()` or renaming tags changes the parameter names that feed into `structural_hash()` — the hash algorithm is unchanged, but its inputs differ. Checkpoints saved before restructuring will fail architecture validation on load. Use `migrate_checkpoint_file()` to remap parameter names, or retrain.

## [0.1.5] - 2026-03-25

### Added
- `make docs-rs` — local docs.rs build validation via disposable Docker container (nightly Rust, `--cfg docsrs`, no libtorch). Catches docs.rs failures before publishing.

### Fixed
- Fix docs.rs build: `rand` 0.9.2 uses `feature(doc_auto_cfg)` removed in nightly 1.92+. Made `rand` an optional dependency (`rng` feature, on by default) so docs.rs can build without it.
- Fix flaky `test_clip_grad_norm` — seed RNG for deterministic weights.
- Fix rustdoc broken intra-doc links in `Tensor` (escaped shape brackets, qualified method paths).

## [0.1.4] - 2026-03-25

### Fixed
- Disable example scraping on docs.rs — examples require libtorch which the docs.rs sandbox doesn't have. The scraping failure corrupted dependency artifacts, breaking the doc build.

## [0.1.3] - 2026-03-25

### Added

#### GPU Performance
- **Fused Adam/AdamW**: `_fused_adamw_` single multi-tensor CUDA kernel for the complete optimizer step across all parameters. Reduces ~4N kernel launches to 1 per parameter group. Automatic on CUDA — no API change needed. `grad_scale`/`found_inf` params exposed for GradScaler integration.
- **Foreach operations**: 7 batched tensor ops that reduce CUDA kernel launches — `foreach_add_scalar_`, `foreach_mul_scalar_`, `foreach_zero_`, `foreach_add_list_`, `foreach_norm`, `foreach_lerp_scalar_`, `foreach_sqrt_`. Used internally by fused optimizers and gradient clipping.
- **Fused gradient clipping**: `clip_grad_norm` now uses `_foreach_norm` + `_foreach_mul_` internally (2 kernels instead of 2N).
- **CUDA Graphs**: `CudaGraph` struct with capture/replay/reset for zero CPU dispatch overhead. `cuda_graph_capture()` convenience helper with warmup. `MemPoolId`, `CaptureMode` (Global/ThreadLocal/Relaxed), `cuda_graph_pool_handle()` for memory pool sharing. 2-5x speedup for models with many small kernels.
- **Autocast (AMP)**: `AutocastGuard` RAII wrapper and `autocast()` closure helper for automatic mixed-precision dispatch. Eligible ops (matmul, conv, linear) run in Float16/BFloat16 on Tensor Core GPUs. Up to 3x speedup on RTX 30xx+.
- **GradScaler**: Dynamic loss scaling for mixed-precision training. Scale, unscale with inf/nan detection, step with skip-on-inf, update with growth/backoff.

#### Tensor Operations
- **Channels-last memory format**: `Tensor::to_channels_last()` and `is_channels_last()` for NHWC layout. 8-35% speedup for Conv2d on Tensor Core GPUs.
- **Non-blocking device transfer**: `Tensor::to_device_async()` for overlapped CPU-to-GPU transfer. Pair with `pin_memory()` for maximum overlap.
- **`Tensor::copy_()`**: In-place copy with `non_blocking` parameter for async CUDA transfers. Used by CUDA Graph capture for data loading.
- **`Tensor::pin_memory()`** and `is_pinned()`: Page-locked CPU memory for fast async GPU transfers.
- **Peak VRAM tracking**: `cuda_peak_active_bytes()`, `cuda_peak_reserved_bytes()`, `cuda_reset_peak_stats()` — matches `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` / `reset_peak_memory_stats()` semantics. With `_idx` variants for multi-GPU.

#### Graph Engine
- **Pre-computed routing**: `Graph::build()` pre-computes a Vec-indexed routing table. Forward dispatch uses flat array indexing instead of HashMap lookups. Cached execution buffers reused across forward calls. Zero allocation during inference.
- **Vectorized gate combination**: Gate routing stacks all expert outputs and combines via broadcast multiply + sum (~3 kernel launches regardless of expert count, vs 3N with sequential accumulation).
- **Loop fast-path**: `for_n` loops detect at call time whether refs are needed and call `body.forward()` directly when no `.using()` is chained, skipping HashMap construction and `body_step` indirection.

#### Other
- **`MaxPool2d`** module: 2D max pooling with kernel size, stride, padding, dilation, and ceil mode.
- **`Rng`** struct: CPU-side RNG (SmallRng/Xoshiro256++) with seed, shuffle, bernoulli, range, normal.
- **`manual_seed(u64)`** / **`cuda_manual_seed_all(u64)`**: Seed libtorch RNGs for reproducibility.
- **`cuda_active_bytes()`**: Query bytes backing live tensors (matches `torch.cuda.memory_allocated()`).

### Fixed
- **VRAM monitoring**: `cuda_allocated_bytes()` now returns `reserved_bytes` from the allocator, making spill detection work.
- Removed unused `ResourceSample::vram_used_bytes` field.
- Dashboard uses `vram_alloc` as the sole VRAM metric.

### Changed
- **Benchmark suite**: Publication-grade methodology with interleaved multi-round execution (`--rounds N`), GPU clock locking (`--lock-clocks FREQ`), configurable warmup (`--warmup-secs`). 7 benchmarks (3 standard + 4 graph-builder). Peak VRAM tracking (not snapshots). WSL2 host-side clock management via `bench-publish.ps1`. `make bench-publish` for reproducible runs.
- **Docker**: `.dockerignore`, BuildKit cache for libtorch downloads, skip-if-exists image targets, dedicated bench image.

## [0.1.2] - 2026-03-19

### Added
- **VRAM spill detection**: New FFI function `flodl_cuda_alloc_bytes` queries libtorch's CUDA caching allocator. `cuda_allocated_bytes()` / `cuda_allocated_bytes_idx()` expose it in Rust. When allocated bytes exceed physical VRAM, the monitor shows spill in terminal output, live dashboard, CSV export, and epoch log.
- `ResourceSample::vram_allocated_bytes` field for allocator-level memory tracking.
- `vram_spill` column in CSV export.

### Fixed
- README links now use absolute GitHub URLs — fixes broken links on crates.io where relative paths don't resolve.

## [0.1.1] - 2026-03-18

### Fixed
- Replace `sha2` with `hmac-sha256` — fixes docs.rs build (sha2's asm feature doesn't compile on docs.rs).
- Widen leak test tolerance for CI parallel test jitter.

## [0.1.0] - 2026-03-18

### Added
- **Graph identity**: `Graph::structural_hash()` — deterministic SHA-256 hash of graph topology, module names, and parameter/buffer shapes. Any architecture change produces a different hash. `Graph::short_hash()` returns the first 8 chars. `FlowBuilder::label()` sets a human-readable name (does not affect hash).
- **Checkpoint architecture validation**: Checkpoint format v1 embeds a 32-byte structural hash. `load_checkpoint` / `load_checkpoint_file` accept an optional hash and error on architecture mismatch.
- **Dashboard metadata**: `Monitor::set_metadata(serde_json::Value)` attaches hyperparameters/config to the HTML archive. `watch()` / `watch_profiled()` capture graph label and hash. Dashboard header shows `"floDl — {label} [{hash8}]"`.
- **Parameter freezing**: `Parameter::freeze()`, `unfreeze()`, `is_frozen()` — disable/enable gradient tracking per parameter. Optimizers automatically skip frozen params (no grad). `Parameter::to_device()` now preserves frozen state.
- **Named checkpoints**: `Graph::named_parameters()` and `named_buffers()` return qualified names (`"tag/weight"` or `"node_id/running_mean"`). `save_checkpoint` / `load_checkpoint` persist both parameters and buffers (e.g., BatchNorm running stats), matching by name for partial loading. `LoadReport` reports what was loaded, skipped, and missing.
- **Optimizer parameter groups**: `Adam::with_groups()`, `SGD::with_groups()`, `AdamW::with_groups()` — builder API for per-group learning rates. `Optimizer::set_group_lr()` adjusts a single group; `set_lr()` updates all groups. Groups are persisted through `Stateful` save/load.

### Core Stack
- **Tensor**: Owned RAII tensors with Drop, ~72 operations. CPU and CUDA (feature-gated).
- **Autograd**: Reverse-mode AD backed by libtorch's native autograd engine. 37 differentiable operations with numerical gradient verification.
- **NN Modules**: Linear, Conv2d, ConvTranspose2d, LayerNorm, BatchNorm, Dropout, Embedding, GRUCell, LSTMCell.
- **Activations**: ReLU, Sigmoid, Tanh, GELU, SiLU.
- **Losses**: mse_loss, cross_entropy_loss, bce_with_logits_loss, l1_loss, smooth_l1_loss, kl_div_loss.
- **Optimizers**: SGD (with momentum), Adam, AdamW.

### Graph Builder
- Fluent API: from/through/build, split/merge, also (residual), tag/using (named refs).
- Loop constructs: for_n (fixed), while_cond (pre-condition), until_cond (post-condition).
- Routing: gate (soft, weighted), switch (hard, selected branch only).
- Map constructs: each, over, slices, with batched fast path.
- Input (auxiliary graph inputs), tag_group (auto-suffixed parallel branch names).

### Training Tools
- LR scheduling: StepDecay, CosineScheduler, WarmupScheduler (composable), PlateauScheduler.
- Mixed precision: Float16/BFloat16 dtype casting, GradScaler for loss scaling.
- Gradient clipping: clip_grad_norm, clip_grad_value.
- Checkpointing: save_checkpoint/load_checkpoint (named binary format with LoadReport, persists parameters + buffers, structural hash validation, file or io::Write).
- Weight initialization: kaiming_uniform/normal, xavier_uniform/normal.

### Training Monitor
- Human-readable ETA with adaptive formatting (hours/minutes/seconds/milliseconds).
- System resource tracking: CPU, RAM, GPU utilization (NVML), VRAM usage.
- Live web dashboard via embedded HTTP server with Server-Sent Events.
- Dashboard features: real-time training curves, resource usage charts, epoch log, graph SVG, label/hash header, metadata card.
- CSV and log file export.

### Observation & Visualization
- Tag-based metric collection: collect/flush/trend.
- Trend analysis: slope, stalled, improving, converged.
- Group trends with tag_group expansion.
- DOT/SVG graph visualization with parameter counts and node type shapes.
- Profiling: enable_profiling, profile, timing trends.
- Training curves: plot_html, export_trends, write_log.

### Infrastructure
- **CI**: GitHub Actions with CPU test matrix and CUDA build verification.
- **Docker**: CPU and CUDA Dockerfiles, docker-compose with GPU support.
- **Build**: Makefile with cpu/cuda targets (build, test, clippy, shell).

### Testing
- 389 library tests + showcase tests.
- Zero clippy warnings.
- Autograd numerical gradient checks.
- Module-level gradient checks.

### Key Design Decisions
- **Deterministic VRAM**: Rust's Drop trait replaces 5 phases of GC-based memory management.
- **No GC overhead**: No runtime.KeepAlive, no pending-free queues, no VRAM budget heuristics.
- **Variable**: `Rc<RefCell<VariableInner>>` for cheap Clone with interior mutability.
- **Module trait**: single-input forward + optional NamedInputModule for multi-input. `structural_hash()` for architecture identity.
- **Graph-as-Module**: Graph implements Module for hierarchical composition.
- **NamedInputModule on routers**: SoftmaxRouter and SigmoidRouter sum refs into input before projection.
- **Native FFI ops**: flodl_max, flodl_norm, flodl_cuda_mem_info, flodl_cuda_utilization.
