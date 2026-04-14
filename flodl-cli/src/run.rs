//! Job resolution and command execution.
//!
//! Merges structured config sections into CLI arguments,
//! resolves jobs, and spawns the target process.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::{ExitCode, Stdio};

use crate::config::{self, CommandConfig, ResolvedConfig};
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
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

// ── Script execution ────────────────────────────────────────────────────

/// Run a script, optionally wrapped in Docker.
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
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}

// ── Command execution ───────────────────────────────────────────────────

/// Execute a sub-command, optionally with a named job.
///
/// `project_root` is needed to resolve Docker compose context and
/// compute the relative workdir for containerized execution.
pub fn exec_command(
    cmd_config: &CommandConfig,
    job_name: Option<&str>,
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

    // Resolve config: job overrides merged with root defaults.
    let resolved = match job_name {
        Some(name) => match cmd_config.jobs.get(name) {
            Some(job) => config::merge_job(cmd_config, job),
            None => {
                eprintln!("error: unknown job '{name}'");
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

        if job_name.is_some() {
            eprintln!("fdl: [{service}] {inner}");
        }

        // Run via docker compose from the project root (with libtorch env).
        let docker_cmd = format!("docker compose run --rm {service} bash -c \"{inner}\"");
        spawn_docker_shell(&docker_cmd, project_root)
    } else {
        // Direct execution (inside container or no docker configured).
        let parts: Vec<&str> = entry.split_whitespace().collect();
        if parts.is_empty() {
            eprintln!("error: empty entry point");
            return ExitCode::FAILURE;
        }
        let program = parts[0];
        let entry_args = &parts[1..];

        if job_name.is_some() {
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
                eprintln!("error: failed to execute '{program}': {e}");
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

/// Print help for a sub-command (its jobs and entry).
pub fn print_command_help(cmd_config: &CommandConfig, name: &str) {
    if let Some(desc) = &cmd_config.description {
        eprintln!("{} {desc}", style::bold(name));
    } else {
        eprintln!("{}", style::bold(name));
    }

    eprintln!();
    eprintln!("{}:", style::yellow("Usage"));
    eprintln!("    fdl {name} {} {}", style::dim("[job]"), style::dim("[options]"));

    if !cmd_config.jobs.is_empty() {
        eprintln!();
        eprintln!("{}:", style::yellow("Jobs"));
        for (job_name, job) in &cmd_config.jobs {
            let desc = job.description.as_deref().unwrap_or("-");
            eprintln!("    {}  {}", style::green(&format!("{:<20}", job_name)), desc);
        }
    }

    if let Some(entry) = &cmd_config.entry {
        eprintln!();
        eprintln!("{}:", style::yellow("Entry"));
        eprintln!("    {entry}");
        let docker_info = cmd_config
            .docker
            .as_ref()
            .map(|s| format!(" {}", style::dim(&format!("[docker: {s}]"))))
            .unwrap_or_default();
        if !docker_info.is_empty() {
            eprintln!("    {docker_info}");
        }
        eprintln!();
        eprintln!(
            "    Any extra {} are forwarded to the entry point.",
            style::dim("[options]")
        );
    }

    // Show default config summary.
    let has_ddp = cmd_config.ddp.is_some();
    let has_training = cmd_config.training.is_some();
    if has_ddp || has_training {
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
}

/// Print help for a specific job within a sub-command.
pub fn print_job_help(cmd_config: &CommandConfig, cmd_name: &str, job_name: &str) {
    let job = match cmd_config.jobs.get(job_name) {
        Some(j) => j,
        None => {
            eprintln!("unknown job: {job_name}");
            return;
        }
    };

    // Title.
    let desc = job
        .description
        .as_deref()
        .unwrap_or("(no description)");
    eprintln!(
        "{} {} {}",
        style::bold(cmd_name),
        style::green(job_name),
        desc
    );

    eprintln!();
    eprintln!("{}:", style::yellow("Usage"));
    eprintln!(
        "    fdl {cmd_name} {job_name} {}",
        style::dim("[extra options]")
    );

    // Show the merged config that this job produces.
    let resolved = config::merge_job(cmd_config, job);

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
        "Extra {} after the job name are appended to the command.",
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
    builtins: &[(&str, &str)],
) {
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

    // Built-in commands.
    eprintln!();
    eprintln!("{}:", style::yellow("Built-in"));
    for (name, desc) in builtins {
        eprintln!("    {}  {desc}", style::green(&format!("{:<18}", name)));
    }

    // Scripts.
    if !project.scripts.is_empty() {
        eprintln!();
        eprintln!("{}:", style::yellow("Scripts"));
        for (name, script) in &project.scripts {
            eprintln!(
                "    {}  {}",
                style::green(&format!("{:<18}", name)),
                script.description()
            );
        }
    }

    // Commands (load each child's description).
    if !project.commands.is_empty() {
        eprintln!();
        eprintln!("{}:", style::yellow("Commands"));
        for cmd_path in &project.commands {
            let short = config::command_name(cmd_path);
            let cmd_dir = project_root.join(cmd_path);
            let desc = config::load_command(&cmd_dir)
                .ok()
                .and_then(|c| c.description)
                .unwrap_or_else(|| "(sub-command)".into());
            eprintln!("    {}  {desc}", style::green(&format!("{:<18}", short)));
        }
    }

    eprintln!();
    eprintln!(
        "Use {} for more information on a command.",
        style::dim("fdl <command> -h")
    );
}

/// Format a flat options map for display.
pub fn _format_options(opts: &BTreeMap<String, serde_json::Value>) -> String {
    opts.iter()
        .map(|(k, v)| format!("--{} {}", k.replace('_', "-"), value_to_string(v)))
        .collect::<Vec<_>>()
        .join(" ")
}
