//! Command resolution and execution.
//!
//! Merges structured config sections into CLI arguments, resolves named
//! command presets, and spawns the target process (directly or through
//! Docker when a `docker:` service is declared).

use std::collections::BTreeMap;
use std::path::Path;
use std::process::{ExitCode, Stdio};

use crate::builtins;
use crate::cli_error;
use crate::config::{self, ArgSpec, CommandConfig, OptionSpec, ResolvedConfig, Schema};
use crate::libtorch;
use crate::style;

// ── Config to CLI args ──────────────────────────────────────────────────

/// Translate a resolved config into CLI arguments for the entry point.
pub fn config_to_args(resolved: &ResolvedConfig) -> Vec<String> {
    let mut args = Vec::new();

    // DDP section
    let d = &resolved.ddp;
    push_opt(&mut args, "--mode", &d.mode);
    push_opt(&mut args, "--policy", &d.policy);
    push_opt(&mut args, "--backend", &d.backend);
    push_value(&mut args, "--anchor", &d.anchor);
    push_num(&mut args, "--max-anchor", &d.max_anchor);
    push_float(&mut args, "--overhead-target", &d.overhead_target);
    push_float(&mut args, "--divergence-threshold", &d.divergence_threshold);
    push_value(&mut args, "--max-batch-diff", &d.max_batch_diff);
    push_float(&mut args, "--max-grad-norm", &d.max_grad_norm);
    push_num(&mut args, "--snapshot-timeout", &d.snapshot_timeout);
    push_num(&mut args, "--checkpoint-every", &d.checkpoint_every);
    push_value(&mut args, "--progressive", &d.progressive);
    if let Some(hint) = &d.speed_hint {
        args.push("--speed-hint".into());
        args.push(format!("{}:{}", hint.slow_rank, hint.ratio));
    }
    if let Some(ratios) = &d.partition_ratios {
        let s: Vec<String> = ratios.iter().map(|r| format!("{r}")).collect();
        args.push("--partition-ratios".into());
        args.push(s.join(","));
    }
    if let Some(ratio) = d.lr_scale_ratio {
        args.push("--lr-scale-ratio".into());
        args.push(format!("{ratio}"));
    }
    if d.timeline == Some(true) {
        args.push("--timeline".into());
    }

    // Training section
    let t = &resolved.training;
    push_num(&mut args, "--epochs", &t.epochs);
    push_num(&mut args, "--batch-size", &t.batch_size);
    push_num(&mut args, "--batches", &t.batches_per_epoch);
    push_float(&mut args, "--lr", &t.lr);
    push_num(&mut args, "--seed", &t.seed);

    // Output section
    let o = &resolved.output;
    push_opt(&mut args, "--output", &o.dir);
    push_num(&mut args, "--monitor", &o.monitor);

    // Pass-through options
    for (key, val) in &resolved.options {
        let flag = format!("--{}", key.replace('_', "-"));
        match val {
            serde_json::Value::Bool(true) => args.push(flag),
            serde_json::Value::Bool(false) => {}
            serde_json::Value::Null => {}
            other => {
                args.push(flag);
                args.push(value_to_string(other));
            }
        }
    }

    args
}

fn push_opt(args: &mut Vec<String>, flag: &str, val: &Option<String>) {
    if let Some(v) = val {
        args.push(flag.into());
        args.push(v.clone());
    }
}

fn push_num<T: std::fmt::Display>(args: &mut Vec<String>, flag: &str, val: &Option<T>) {
    if let Some(v) = val {
        args.push(flag.into());
        args.push(v.to_string());
    }
}

fn push_float(args: &mut Vec<String>, flag: &str, val: &Option<f64>) {
    if let Some(v) = val {
        args.push(flag.into());
        args.push(format!("{v}"));
    }
}

fn push_value(args: &mut Vec<String>, flag: &str, val: &Option<serde_json::Value>) {
    if let Some(v) = val {
        match v {
            serde_json::Value::Null => {}
            other => {
                args.push(flag.into());
                args.push(value_to_string(other));
            }
        }
    }
}

fn value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

// ── Docker detection ────────────────────────────────────────────────────

/// Check if we're already running inside a Docker container.
fn inside_docker() -> bool {
    Path::new("/.dockerenv").exists()
}

/// Resolve libtorch env vars from the project root, matching the Makefile logic:
///   LIBTORCH_HOST_PATH = ./libtorch/<active_variant>
///   LIBTORCH_CPU_PATH  = ./libtorch/precompiled/cpu
///   CUDA_VERSION, CUDA_TAG from .arch metadata
fn libtorch_env(project_root: &Path) -> Vec<(String, String)> {
    let mut env = Vec::new();

    // CPU path is always the same.
    env.push((
        "LIBTORCH_CPU_PATH".into(),
        "./libtorch/precompiled/cpu".into(),
    ));

    // Active variant for CUDA.
    if let Some(info) = libtorch::detect::read_active(project_root) {
        let host_path = format!("./libtorch/{}", info.path);
        env.push(("LIBTORCH_HOST_PATH".into(), host_path));

        // CUDA version from .arch metadata.
        if let Some(cuda) = &info.cuda_version {
            if cuda != "none" {
                let cuda_version = if cuda.matches('.').count() < 2 {
                    format!("{cuda}.0")
                } else {
                    cuda.clone()
                };
                let cuda_tag = cuda_version
                    .splitn(3, '.')
                    .take(2)
                    .collect::<Vec<_>>()
                    .join(".");
                env.push(("CUDA_VERSION".into(), cuda_version));
                env.push(("CUDA_TAG".into(), cuda_tag));
            }
        }
    }

    env
}

/// Spawn a shell command with libtorch env vars set.
///
/// `FLODL_VERBOSITY` is forwarded to Docker containers via the
/// `environment:` section in docker-compose.yml (bare variable name
/// passes the host value through when set, ignored otherwise).
fn spawn_docker_shell(command: &str, project_root: &Path) -> ExitCode {
    let env_vars = libtorch_env(project_root);

    let mut cmd = std::process::Command::new("sh");
    cmd.args(["-c", command])
        .current_dir(project_root)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .stdin(Stdio::inherit());

    for (key, val) in &env_vars {
        cmd.env(key, val);
    }

    match cmd.status() {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => {
            cli_error!("{e}");
            ExitCode::FAILURE
        }
    }
}

// ── Run-kind execution ──────────────────────────────────────────────────

/// Run an inline `run:` script, optionally wrapped in Docker.
pub fn exec_script(command: &str, docker_service: Option<&str>, cwd: &Path) -> ExitCode {
    match docker_service {
        Some(service) if !inside_docker() => {
            let docker_cmd =
                format!("docker compose run --rm {service} bash -c \"{command}\"");
            spawn_docker_shell(&docker_cmd, cwd)
        }
        _ => {
            let (shell, flag) = if cfg!(target_os = "windows") {
                ("cmd", "/C")
            } else {
                ("sh", "-c")
            };

            match std::process::Command::new(shell)
                .args([flag, command])
                .current_dir(cwd)
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .stdin(Stdio::inherit())
                .status()
            {
                Ok(s) if s.success() => ExitCode::SUCCESS,
                Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
                Err(e) => {
                    cli_error!("{e}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}

// ── Command execution ───────────────────────────────────────────────────

/// Execute a sub-command, optionally with a named preset (inline command).
///
/// `project_root` is needed to resolve Docker compose context and
/// compute the relative workdir for containerized execution.
pub fn exec_command(
    cmd_config: &CommandConfig,
    preset_name: Option<&str>,
    extra_args: &[String],
    cmd_dir: &Path,
    project_root: &Path,
) -> ExitCode {
    let entry = match &cmd_config.entry {
        Some(e) => e.as_str(),
        None => {
            eprintln!(
                "error: no entry point defined in {}/fdl.yaml",
                cmd_dir.display()
            );
            return ExitCode::FAILURE;
        }
    };

    // Tail validation pre-flight. Runs whenever a schema is present:
    // - `choices:` on declared options → always enforced.
    // - Unknown flags → rejected only when `schema.strict` is set
    //   (lenient mode tolerates pass-through flags the binary may
    //   consume directly).
    // fdl-generated args (from the structured ddp/training/output
    // blocks) are intentionally skipped — those are the binary's
    // surface, not the user's.
    if let Some(schema) = &cmd_config.schema {
        if let Err(e) = config::validate_tail(extra_args, schema) {
            cli_error!("{e}");
            return ExitCode::FAILURE;
        }
    }

    // Resolve config: preset overrides merged with root defaults.
    let resolved = match preset_name {
        Some(name) => match cmd_config.commands.get(name) {
            Some(preset) => {
                // Validate *this* preset only (choices + strict unknowns).
                // Whole-map validation is deferred so a broken sibling
                // preset doesn't block a correct one from running.
                if let Some(schema) = &cmd_config.schema {
                    if let Err(e) = config::validate_preset_for_exec(name, preset, schema) {
                        cli_error!("{e}");
                        return ExitCode::FAILURE;
                    }
                }
                config::merge_preset(cmd_config, preset)
            }
            None => {
                cli_error!("unknown command '{name}'");
                eprintln!();
                print_command_help(cmd_config, "");
                return ExitCode::FAILURE;
            }
        },
        None => config::defaults_only(cmd_config),
    };

    // Build argument list from config.
    let mut args = config_to_args(&resolved);

    // Append extra CLI args (these override config-derived args).
    args.extend(extra_args.iter().cloned());

    // Docker wrapping or direct execution.
    let use_docker = cmd_config.docker.is_some() && !inside_docker();

    if use_docker {
        let service = cmd_config.docker.as_deref().unwrap();
        let workdir = cmd_dir
            .strip_prefix(project_root)
            .unwrap_or(cmd_dir)
            .to_string_lossy();

        // Build the inner command: cd <workdir> && <entry> <args>
        let args_str = shell_join(&args);
        let inner = if workdir.is_empty() || workdir == "." {
            format!("{entry} {args_str}")
        } else {
            format!("cd {workdir} && {entry} {args_str}")
        };

        if preset_name.is_some() {
            eprintln!("fdl: [{service}] {inner}");
        }

        // Run via docker compose from the project root (with libtorch env).
        let docker_cmd = format!("docker compose run --rm {service} bash -c \"{inner}\"");
        spawn_docker_shell(&docker_cmd, project_root)
    } else {
        // Direct execution (inside container or no docker configured).
        let parts: Vec<&str> = entry.split_whitespace().collect();
        if parts.is_empty() {
            cli_error!("empty entry point");
            return ExitCode::FAILURE;
        }
        let program = parts[0];
        let entry_args = &parts[1..];

        if preset_name.is_some() {
            let preview: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            eprintln!("fdl: {entry} {}", preview.join(" "));
        }

        match std::process::Command::new(program)
            .args(entry_args)
            .args(&args)
            .current_dir(cmd_dir)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .stdin(Stdio::inherit())
            .status()
        {
            Ok(s) if s.success() => ExitCode::SUCCESS,
            Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
            Err(e) => {
                cli_error!("failed to execute '{program}': {e}");
                ExitCode::FAILURE
            }
        }
    }
}

/// Join args into a shell-safe string.
fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|a| {
            if a.contains(' ') || a.contains('"') || a.is_empty() {
                format!("'{}'", a.replace('\'', "'\\''"))
            } else {
                a.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// ── Help output ─────────────────────────────────────────────────────────

/// Print help for a `run:`-kind command. Shows the inline script that
/// will execute and the Docker service (if any). `run:` commands do not
/// forward argv, so there are no flags or positionals to document.
pub fn print_run_help(name: &str, description: Option<&str>, run: &str, docker: Option<&str>) {
    if let Some(desc) = description {
        eprintln!("{} {desc}", style::bold(name));
    } else {
        eprintln!("{}", style::bold(name));
    }
    eprintln!();
    eprintln!("{}:", style::yellow("Usage"));
    eprintln!("    fdl {name}");
    eprintln!();
    eprintln!("{}:", style::yellow("Runs"));
    if let Some(svc) = docker {
        eprintln!("    {} {svc} -c {run:?}", style::dim("docker compose run --rm"));
    } else {
        eprintln!("    {run}");
    }
    eprintln!();
    eprintln!(
        "{} run:-kind commands do not forward argv; the script runs as declared.",
        style::dim("Note:"),
    );
}

/// Print help for a sub-command (its arguments, nested commands, and
/// entry). Orchestrates the per-section helpers below.
pub fn print_command_help(cmd_config: &CommandConfig, name: &str) {
    let (presets, sub_cmds) = split_commands_by_kind(&cmd_config.commands);
    let preset_slot = cmd_config.arg_name.as_deref().unwrap_or("preset");

    print_title(cmd_config, name);
    print_usage_line(cmd_config, name, &presets, &sub_cmds, preset_slot);
    print_arguments_section(cmd_config, &presets, preset_slot);
    print_sub_commands_section(&sub_cmds);
    print_options_section(cmd_config);
    print_entry_section(cmd_config);
    print_defaults_section(cmd_config);
}

fn print_title(cmd_config: &CommandConfig, name: &str) {
    if let Some(desc) = &cmd_config.description {
        eprintln!("{} {desc}", style::bold(name));
    } else {
        eprintln!("{}", style::bold(name));
    }
}

fn print_usage_line(
    cmd_config: &CommandConfig,
    name: &str,
    presets: &CommandGroup,
    sub_cmds: &CommandGroup,
    preset_slot: &str,
) {
    // The first-positional slot reflects what is actually accepted here:
    // preset name, sub-command name, or either.
    let usage_tail = build_usage_tail(
        cmd_config.schema.as_ref(),
        !presets.is_empty(),
        !sub_cmds.is_empty(),
        preset_slot,
    );
    eprintln!();
    eprintln!("{}:", style::yellow("Usage"));
    eprintln!("    fdl {name}{usage_tail}");
}

fn print_arguments_section(
    cmd_config: &CommandConfig,
    presets: &CommandGroup,
    preset_slot: &str,
) {
    // Schema-declared positionals (typed slots on the entry binary) and
    // the preset slot (dispatched by fdl before the binary sees argv)
    // both land in the first-positional position, so they share one
    // section. Schema args render first; the preset slot with its
    // value list follows.
    let has_schema_args = cmd_config
        .schema
        .as_ref()
        .is_some_and(|s| !s.args.is_empty());
    if !has_schema_args && presets.is_empty() {
        return;
    }
    eprintln!();
    eprintln!("{}:", style::yellow("Arguments"));
    if let Some(schema) = &cmd_config.schema {
        for a in &schema.args {
            eprintln!("    {}", format_arg(a));
        }
    }
    if !presets.is_empty() {
        let slot_label = format!("[<{preset_slot}>]");
        eprintln!(
            "    {}  Named preset, one of:",
            style::green(&format!("{:<20}", slot_label))
        );
        for (pname, spec) in presets {
            let desc = spec.description.as_deref().unwrap_or("-");
            eprintln!(
                "      {}  {}",
                style::green(&format!("{:<18}", pname)),
                desc
            );
        }
    }
}

fn print_sub_commands_section(sub_cmds: &CommandGroup) {
    // Run/Path kinds only — true sub-commands with their own behavior
    // (an inline script or a nested fdl.yml).
    if sub_cmds.is_empty() {
        return;
    }
    eprintln!();
    eprintln!("{}:", style::yellow("Commands"));
    for (sub_name, sub_spec) in sub_cmds {
        let desc = sub_spec.description.as_deref().unwrap_or("-");
        eprintln!(
            "    {}  {}",
            style::green(&format!("{:<20}", sub_name)),
            desc
        );
    }
}

fn print_options_section(cmd_config: &CommandConfig) {
    // Schema-driven options. Renders only when a schema block is present
    // in fdl.yaml; the "Defaults" section covers ddp/training/output.
    let Some(schema) = &cmd_config.schema else {
        return;
    };
    if schema.options.is_empty() {
        return;
    }
    eprintln!();
    eprintln!("{}:", style::yellow("Options"));
    for (long, spec) in &schema.options {
        for line in format_option(long, spec) {
            eprintln!("    {line}");
        }
    }
}

fn print_entry_section(cmd_config: &CommandConfig) {
    let Some(entry) = &cmd_config.entry else {
        return;
    };
    eprintln!();
    eprintln!("{}:", style::yellow("Entry"));
    eprintln!("    {entry}");
    if let Some(service) = &cmd_config.docker {
        eprintln!(
            "     {}",
            style::dim(&format!("[docker: {service}]"))
        );
    }
    eprintln!();
    eprintln!(
        "    Any extra {} are forwarded to the entry point.",
        style::dim("[options]")
    );
}

fn print_defaults_section(cmd_config: &CommandConfig) {
    if cmd_config.ddp.is_none() && cmd_config.training.is_none() {
        return;
    }
    eprintln!();
    eprintln!("{}:", style::yellow("Defaults"));
    if let Some(d) = &cmd_config.ddp {
        if let Some(mode) = &d.mode {
            eprintln!("    {}  {mode}", style::dim("ddp.mode"));
        }
        if let Some(anchor) = &d.anchor {
            eprintln!("    {}  {}", style::dim("ddp.anchor"), value_to_string(anchor));
        }
    }
    if let Some(t) = &cmd_config.training {
        if let Some(e) = t.epochs {
            eprintln!("    {}  {e}", style::dim("training.epochs"));
        }
        if let Some(bs) = t.batch_size {
            eprintln!("    {}  {bs}", style::dim("training.batch_size"));
        }
        if let Some(lr) = t.lr {
            eprintln!("    {}  {lr}", style::dim("training.lr"));
        }
        if let Some(seed) = t.seed {
            eprintln!("    {}  {seed}", style::dim("training.seed"));
        }
    }
}

/// Print help for a named preset command nested inside a sub-command.
pub fn print_preset_help(cmd_config: &CommandConfig, cmd_name: &str, preset_name: &str) {
    let preset = match cmd_config.commands.get(preset_name) {
        Some(s) => s,
        None => {
            eprintln!("unknown command: {preset_name}");
            return;
        }
    };

    // Title.
    let desc = preset.description.as_deref().unwrap_or("(no description)");
    eprintln!(
        "{} {} {}",
        style::bold(cmd_name),
        style::green(preset_name),
        desc
    );

    eprintln!();
    eprintln!("{}:", style::yellow("Usage"));
    eprintln!(
        "    fdl {cmd_name} {preset_name} {}",
        style::dim("[extra options]")
    );

    // Show the merged config that this preset produces.
    let resolved = config::merge_preset(cmd_config, preset);

    eprintln!();
    eprintln!("{}:", style::yellow("Effective config"));

    // DDP fields.
    let d = &resolved.ddp;
    print_config_field("ddp.mode", &d.mode);
    print_config_value("ddp.anchor", &d.anchor);
    print_config_field("ddp.max_anchor", &d.max_anchor);
    print_config_field("ddp.overhead_target", &d.overhead_target);
    print_config_field("ddp.divergence_threshold", &d.divergence_threshold);
    print_config_value("ddp.max_batch_diff", &d.max_batch_diff);
    print_config_field("ddp.max_grad_norm", &d.max_grad_norm);
    if d.timeline == Some(true) {
        eprintln!("    {}  true", style::dim("ddp.timeline"));
    }

    // Training fields.
    let t = &resolved.training;
    print_config_field("training.epochs", &t.epochs);
    print_config_field("training.batch_size", &t.batch_size);
    print_config_field("training.batches_per_epoch", &t.batches_per_epoch);
    print_config_field("training.lr", &t.lr);
    print_config_field("training.seed", &t.seed);

    // Output fields.
    let o = &resolved.output;
    print_config_field("output.dir", &o.dir);
    print_config_field("output.monitor", &o.monitor);

    // Pass-through options.
    if !resolved.options.is_empty() {
        eprintln!();
        eprintln!("{}:", style::yellow("Options"));
        for (key, val) in &resolved.options {
            eprintln!(
                "    {}  {}",
                style::green(&format!("--{}", key.replace('_', "-"))),
                value_to_string(val)
            );
        }
    }

    // Show the effective command.
    if let Some(entry) = &cmd_config.entry {
        let args = config_to_args(&resolved);
        let args_str = args.join(" ");
        let docker_info = cmd_config
            .docker
            .as_ref()
            .map(|s| format!("[{s}] ", ))
            .unwrap_or_default();

        eprintln!();
        eprintln!("{}:", style::yellow("Effective command"));
        eprintln!(
            "    {}{}{}",
            style::dim(&docker_info),
            entry,
            if args_str.is_empty() {
                String::new()
            } else {
                format!(" {args_str}")
            }
        );
    }

    eprintln!();
    eprintln!(
        "Extra {} after the command name are appended to the entry.",
        style::dim("[options]")
    );
}

fn print_config_field<T: std::fmt::Display>(label: &str, val: &Option<T>) {
    if let Some(v) = val {
        eprintln!("    {}  {v}", style::dim(label));
    }
}

fn print_config_value(label: &str, val: &Option<serde_json::Value>) {
    if let Some(v) = val {
        if !v.is_null() {
            eprintln!("    {}  {}", style::dim(label), value_to_string(v));
        }
    }
}

/// Print the project help with scripts and commands.
pub fn print_project_help(
    project: &config::ProjectConfig,
    project_root: &Path,
    active_env: Option<&str>,
) {
    let visible_builtins = builtins::visible_top_level();
    if let Some(desc) = &project.description {
        eprintln!("{} {}", style::bold("fdl"), desc);
    } else {
        eprintln!("{} {}", style::bold("fdl"), env!("CARGO_PKG_VERSION"));
    }

    eprintln!();
    eprintln!("{}:", style::yellow("Usage"));
    eprintln!(
        "    fdl {} {}",
        style::dim("<command>"),
        style::dim("[options]")
    );

    eprintln!();
    eprintln!("{}:", style::yellow("Options"));
    eprintln!(
        "    {}  Show this help",
        style::green(&format!("{:<18}", "-h, --help"))
    );
    eprintln!(
        "    {}  Show version",
        style::green(&format!("{:<18}", "-V, --version"))
    );
    eprintln!(
        "    {}  Use fdl.<name>.yml overlay (also: FDL_ENV=<name>)",
        style::green(&format!("{:<18}", "--env <name>"))
    );
    eprintln!(
        "    {}  Verbose output",
        style::green(&format!("{:<18}", "-v"))
    );
    eprintln!(
        "    {}  Debug output",
        style::green(&format!("{:<18}", "-vv"))
    );
    eprintln!(
        "    {}  Trace output (maximum detail)",
        style::green(&format!("{:<18}", "-vvv"))
    );
    eprintln!(
        "    {}  Suppress non-error output",
        style::green(&format!("{:<18}", "-q, --quiet"))
    );
    eprintln!(
        "    {}  Force ANSI color (bypass TTY / NO_COLOR detection)",
        style::green(&format!("{:<18}", "--ansi"))
    );
    eprintln!(
        "    {}  Disable ANSI color output",
        style::green(&format!("{:<18}", "--no-ansi"))
    );

    // Built-in commands.
    eprintln!();
    eprintln!("{}:", style::yellow("Built-in"));
    for (name, desc) in &visible_builtins {
        eprintln!("    {}  {desc}", style::green(&format!("{:<18}", name)));
    }

    // Commands: unified section. Each entry in `project.commands` is one
    // of: an inline `run:` script, a `path:` (or convention-default)
    // pointer to a child fdl.yml, or — at nested levels only — an inline
    // preset. Descriptions come from the `CommandSpec`; for `path:`
    // commands missing their own description, fall back to loading the
    // child fdl.yml's description.
    if !project.commands.is_empty() {
        eprintln!();
        eprintln!("{}:", style::yellow("Commands"));
        for (name, spec) in &project.commands {
            let desc: String = match spec.description.clone() {
                Some(d) => d,
                None => {
                    // For path-kind entries, fall back to the child config's
                    // own description so `commands: { ddp-bench: }` still
                    // shows a useful blurb.
                    let is_path_kind = spec.run.is_none();
                    if is_path_kind {
                        let child_dir = spec.resolve_path(name, project_root);
                        config::load_command_with_env(&child_dir, active_env)
                            .ok()
                            .and_then(|c| c.description)
                            .unwrap_or_else(|| "(sub-command)".into())
                    } else {
                        spec.run
                            .as_deref()
                            .unwrap_or("(command)")
                            .to_string()
                    }
                }
            };
            eprintln!("    {}  {desc}", style::green(&format!("{:<18}", name)));
        }
    }

    // Available environments (sibling fdl.<env>.yml files at project root).
    if let Some(base_config) = config::find_config(project_root) {
        let envs = crate::overlay::list_envs(&base_config);
        if !envs.is_empty() {
            eprintln!();
            eprintln!("{}:", style::yellow("Environments"));
            for e in &envs {
                let active_marker = if Some(e.as_str()) == active_env {
                    style::green(" (active)")
                } else {
                    String::new()
                };
                eprintln!(
                    "    {}  Overlay from fdl.{}.yml{active_marker}",
                    style::green(&format!("{:<18}", e)),
                    e
                );
            }
            eprintln!();
            eprintln!(
                "Use {} to run a command with an environment overlay.",
                style::dim("fdl <env> <command>")
            );
        }
    }

    eprintln!();
    eprintln!(
        "Use {} for more information on a command.",
        style::dim("fdl <command> -h")
    );
}

// ── Schema-driven help helpers ──────────────────────────────────────────

/// Build the part of `fdl <cmd>...` after the command name: positionals
/// rendered as `<name>` (required) or `[<name>]` (optional), plus a slot
/// for the first-positional picker — `[<preset>]` when only presets exist,
/// `[<command>]` when only sub-commands exist, `[<preset>|<command>]` when
/// both — and `[options]`. The preset placeholder is customisable per
/// sub-command via `arg-name:`.
fn build_usage_tail(
    schema: Option<&Schema>,
    has_presets: bool,
    has_sub_commands: bool,
    preset_slot: &str,
) -> String {
    let mut parts = String::new();
    let slot = match (has_presets, has_sub_commands) {
        (true, false) => Some(format!("[<{preset_slot}>]")),
        (false, true) => Some("[<command>]".to_string()),
        (true, true) => Some(format!("[<{preset_slot}>|<command>]")),
        (false, false) => None,
    };
    if let Some(s) = slot {
        parts.push(' ');
        parts.push_str(&style::dim(&s));
    }
    if let Some(s) = schema {
        for a in &s.args {
            parts.push(' ');
            parts.push_str(&format_arg_usage(a));
        }
    }
    parts.push(' ');
    parts.push_str(&style::dim("[options]"));
    parts
}

type CommandGroup = Vec<(String, crate::config::CommandSpec)>;

/// Partition a `commands:` map into (presets, sub-commands) by resolved
/// `CommandKind`. Entries whose `kind()` errors (both run and path set)
/// are treated as sub-commands so they still render somewhere — the
/// error surfaces when the user tries to dispatch them.
fn split_commands_by_kind(
    commands: &BTreeMap<String, crate::config::CommandSpec>,
) -> (CommandGroup, CommandGroup) {
    use crate::config::CommandKind;
    let mut presets = Vec::new();
    let mut sub_cmds = Vec::new();
    for (k, v) in commands {
        match v.kind() {
            Ok(CommandKind::Preset) => presets.push((k.clone(), v.clone())),
            _ => sub_cmds.push((k.clone(), v.clone())),
        }
    }
    (presets, sub_cmds)
}

fn format_arg_usage(a: &ArgSpec) -> String {
    let suffix = if a.variadic { "..." } else { "" };
    let core = format!("<{}>{suffix}", a.name);
    if a.required && a.default.is_none() {
        style::green(&core)
    } else {
        style::dim(&format!("[{core}]"))
    }
}

fn format_arg(a: &ArgSpec) -> String {
    let mut left = format_arg_usage(a);
    // Target ~22-char visual width for the label column.
    let visible = visible_width(&left);
    if visible < 22 {
        for _ in 0..(22 - visible) {
            left.push(' ');
        }
    } else {
        left.push(' ');
    }
    let mut line = left;
    line.push_str(a.description.as_deref().unwrap_or("-"));
    append_default_and_choices(&mut line, &a.default, &a.choices, &a.ty);
    line
}

/// Format an option row. Returns one or more lines; `choices` list wraps
/// onto a second indented line when present, to keep the main row readable.
fn format_option(long: &str, spec: &OptionSpec) -> Vec<String> {
    let flag = match &spec.short {
        Some(s) => format!("-{s}, --{long}"),
        None => format!("    --{long}"),
    };
    let placeholder = option_placeholder(&spec.ty);
    let left = if placeholder.is_empty() {
        style::green(&flag)
    } else {
        style::green(&format!("{flag} {placeholder}"))
    };
    let visible = visible_width_for(&flag, placeholder);

    // Pad to 30 columns for alignment.
    let pad = if visible < 30 { 30 - visible } else { 1 };
    let mut line = format!("{left}{}", " ".repeat(pad));
    line.push_str(spec.description.as_deref().unwrap_or("-"));
    append_default_and_choices(&mut line, &spec.default, &spec.choices, &spec.ty);

    let mut out = vec![line];
    if let Some(env) = &spec.env {
        out.push(format!("{}  {}", " ".repeat(32), style::dim(&format!("[env: {env}]"))));
    }
    out
}

fn option_placeholder(ty: &str) -> &'static str {
    match ty {
        "bool" => "",
        "int" => "<N>",
        "float" => "<F>",
        "path" => "<PATH>",
        "list[path]" => "<PATH>...",
        t if t.starts_with("list[") => "<VALUE>...",
        _ => "<VALUE>",
    }
}

fn append_default_and_choices(
    line: &mut String,
    default: &Option<serde_json::Value>,
    choices: &Option<Vec<serde_json::Value>>,
    ty: &str,
) {
    if let Some(d) = default {
        // Skip noisy defaults: bool false, empty list, null.
        let is_empty_list = matches!(d, serde_json::Value::Array(a) if a.is_empty());
        let is_false = matches!(d, serde_json::Value::Bool(false));
        if !d.is_null() && !is_false && !is_empty_list {
            line.push_str(&format!(" {}", style::dim(&format!("[default: {}]", format_value(d)))));
        }
    }
    if let Some(choices) = choices {
        if !choices.is_empty() {
            let list = choices
                .iter()
                .map(format_value)
                .collect::<Vec<_>>()
                .join(", ");
            line.push_str(&format!(" {}", style::dim(&format!("[possible: {list}]"))));
        }
    }
    // Annotate list types so users know about repeat/comma semantics.
    if ty.starts_with("list[") {
        line.push_str(&format!(" {}", style::dim("(repeat or comma-separate)")));
    }
}

fn format_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Rough visible width helper: styled strings wrap their visible content
/// in ANSI escapes, so we use the unstyled inputs we started from.
fn visible_width(s: &str) -> usize {
    // The inputs we pass here come from pre-styling helpers that already
    // know the raw length. Strip ANSI to be safe.
    strip_ansi(s).chars().count()
}

fn visible_width_for(flag: &str, placeholder: &str) -> usize {
    if placeholder.is_empty() {
        flag.chars().count()
    } else {
        flag.chars().count() + 1 + placeholder.chars().count()
    }
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}
