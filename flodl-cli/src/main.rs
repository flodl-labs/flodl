//! flodl-cli: command-line tool for the floDl deep learning framework.
//!
//! Provides hardware diagnostics, libtorch management, and project scaffolding.
//! Pure Rust binary with no libtorch dependency (GPU detection via nvidia-smi).
//!
//! Works both inside a floDl project and standalone. When standalone, libtorch
//! is managed under `~/.flodl/` (override with `$FLODL_HOME`).

use flodl_cli::{
    api_ref, completions, config, context, diagnose, dispatch, init, libtorch,
    overlay, parse_or_schema_from, run, schema_cache, setup, skill, util, FdlArgs,
};

use dispatch::{classify_path_step, PathOutcome};

use std::env;
use std::process::ExitCode;

use context::Context;

// ---------------------------------------------------------------------------
// FdlArgs structs (one per leaf sub-command)
//
// These dogfood the derive macro across flodl-cli itself. Each is parsed
// with `parse_or_schema_from(&argv)` from a sliced argv tail; the derive
// handles argv, `--help`, and `--fdl-schema` uniformly.
// ---------------------------------------------------------------------------

/// Interactive guided setup wizard.
#[derive(FdlArgs, Debug)]
struct SetupArgs {
    /// Skip all prompts and use auto-detected defaults.
    #[option(short = 'y')]
    non_interactive: bool,
    /// Re-download or rebuild even if libtorch exists.
    #[option]
    force: bool,
}

/// System and GPU diagnostics.
#[derive(FdlArgs, Debug)]
struct DiagnoseArgs {
    /// Emit machine-readable JSON.
    #[option]
    json: bool,
}

/// Generate flodl API reference.
#[derive(FdlArgs, Debug)]
struct ApiRefArgs {
    /// Emit machine-readable JSON.
    #[option]
    json: bool,
    /// Explicit flodl source path (defaults to detected project root).
    #[option]
    path: Option<String>,
}

/// Scaffold a new floDl project.
#[derive(FdlArgs, Debug)]
struct InitArgs {
    /// New project directory name.
    #[arg]
    name: Option<String>,
    /// Generate a Docker-based scaffold (libtorch baked into the image).
    #[option]
    docker: bool,
}

/// Install or update fdl globally (~/.local/bin/fdl).
#[derive(FdlArgs, Debug)]
struct InstallArgs {
    /// Check for updates without installing.
    #[option]
    check: bool,
    /// Symlink to the current binary (tracks local builds).
    #[option]
    dev: bool,
}

/// List installed libtorch variants.
#[derive(FdlArgs, Debug)]
struct LibtorchListArgs {
    /// Emit machine-readable JSON.
    #[option]
    json: bool,
}

/// Activate a libtorch variant.
#[derive(FdlArgs, Debug)]
struct LibtorchActivateArgs {
    /// Variant to activate (as shown by `fdl libtorch list`).
    #[arg]
    variant: Option<String>,
}

/// Remove a libtorch variant.
#[derive(FdlArgs, Debug)]
struct LibtorchRemoveArgs {
    /// Variant to remove (as shown by `fdl libtorch list`).
    #[arg]
    variant: Option<String>,
}

/// Download a pre-built libtorch variant.
#[derive(FdlArgs, Debug)]
struct LibtorchDownloadArgs {
    /// Force the CPU variant.
    #[option]
    cpu: bool,
    /// Pick a specific CUDA version (instead of auto-detect).
    #[option(choices = &["12.6", "12.8"])]
    cuda: Option<String>,
    /// Install libtorch to this directory (default: project libtorch/).
    #[option]
    path: Option<String>,
    /// Do not activate after download.
    #[option]
    no_activate: bool,
    /// Show what would happen without downloading.
    #[option]
    dry_run: bool,
}

/// Build libtorch from source.
#[derive(FdlArgs, Debug)]
struct LibtorchBuildArgs {
    /// Override CUDA architectures (semicolon-separated, e.g. "6.1;12.0").
    #[option]
    archs: Option<String>,
    /// Parallel compilation jobs.
    #[option(default = "6")]
    jobs: usize,
    /// Force Docker build (isolated, reproducible).
    #[option]
    docker: bool,
    /// Force native build (faster, requires host toolchain).
    #[option]
    native: bool,
    /// Show what would happen without building.
    #[option]
    dry_run: bool,
}

/// Install AI coding assistant skills.
#[derive(FdlArgs, Debug)]
struct SkillInstallArgs {
    /// Target tool (defaults to auto-detect).
    #[option]
    tool: Option<String>,
    /// Specific skill name (defaults to all detected skills).
    #[option]
    skill: Option<String>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    let raw_args: Vec<String> = env::args().collect();

    // Extract global verbosity flags before command dispatch.
    // Sets FLODL_VERBOSITY so child processes (including Docker) inherit it.
    let (args, verbosity) = extract_verbosity(&raw_args);
    if let Some(v) = verbosity {
        // SAFETY: called before any threads are spawned.
        unsafe {
            env::set_var("FLODL_VERBOSITY", v.to_string());
        }
    }

    // First-arg environment detection. If the first positional matches a
    // sibling `fdl.<arg>.yml` overlay AND is not also the name of a
    // built-in, script, or sub-command, consume it as an env selector.
    // Ambiguous cases (both a command and an env file match) are a loud
    // error rather than silent precedence.
    let (active_env, args) = match resolve_env(&args) {
        Ok(pair) => pair,
        Err(msg) => {
            eprintln!("error: {msg}");
            return ExitCode::FAILURE;
        }
    };

    // Bare `fdl` with no args behaves like `fdl --help`.
    let cmd = args.get(1).map(String::as_str).unwrap_or("--help");

    match cmd {
        "setup" => {
            let cli: SetupArgs = parse_sub("fdl setup", &args[1..]);
            let opts = setup::SetupOpts {
                non_interactive: cli.non_interactive,
                force: cli.force,
            };
            match setup::run(opts) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        "libtorch" => dispatch_libtorch(&args),
        "diagnose" => {
            let cli: DiagnoseArgs = parse_sub("fdl diagnose", &args[1..]);
            diagnose::run(cli.json);
            ExitCode::SUCCESS
        }
        "api-ref" => {
            let cli: ApiRefArgs = parse_sub("fdl api-ref", &args[1..]);
            match api_ref::run(cli.json, cli.path.as_deref()) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        "init" => {
            let cli: InitArgs = parse_sub("fdl init", &args[1..]);
            match init::run(cli.name.as_deref(), cli.docker) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        "install" => {
            let cli: InstallArgs = parse_sub("fdl install", &args[1..]);
            cmd_install(cli.check, cli.dev)
        }
        "skill" => dispatch_skill(&args),
        "completions" => {
            let shell = args.get(2).map(String::as_str).unwrap_or("bash");
            let cwd = env::current_dir().unwrap_or_default();
            let project = load_project_config(&cwd, active_env.as_deref());
            completions::generate(shell, project.as_ref().map(|(p, r)| (p, r.as_path())));
            ExitCode::SUCCESS
        }
        "autocomplete" => {
            let cwd = env::current_dir().unwrap_or_default();
            let project = load_project_config(&cwd, active_env.as_deref());
            completions::autocomplete(project.as_ref().map(|(p, r)| (p, r.as_path())));
            ExitCode::SUCCESS
        }
        "config" => cmd_config_show(&args[1..], active_env.as_deref()),
        "--help" | "-h" => {
            let cwd = env::current_dir().unwrap_or_default();
            if let Some((project, root)) = load_project_config(&cwd, active_env.as_deref()) {
                run::print_project_help(&project, &root, BUILTINS, active_env.as_deref());
            } else {
                print_usage();
            }
            ExitCode::SUCCESS
        }
        "version" | "--version" | "-V" => {
            println!("flodl-cli {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        other => dispatch_config(other, &args, active_env.as_deref()),
    }
}

/// First-arg environment resolution. Returns `(Some(env), args_without_env)`
/// when the first positional matches a sibling `fdl.<arg>.yml` overlay and
/// no built-in, script, or sub-command by that name exists. Returns
/// `(None, args)` when no env applies. Errors on ambiguity.
fn resolve_env(args: &[String]) -> Result<(Option<String>, Vec<String>), String> {
    let candidate = match args.get(1) {
        Some(a) if !a.starts_with('-') => a,
        _ => return Ok((None, args.to_vec())),
    };

    let cwd = env::current_dir().unwrap_or_default();
    let base_config = match config::find_config(&cwd) {
        Some(p) => p,
        None => return Ok((None, args.to_vec())),
    };
    let env_file = overlay::find_env_file(&base_config, candidate);
    if env_file.is_none() {
        return Ok((None, args.to_vec()));
    }

    // An overlay exists — check for collision with a command of the same name.
    let is_command = is_builtin_name(candidate) || is_project_command(&base_config, candidate);
    if is_command {
        return Err(format!(
            "ambiguous `{candidate}`: matches both a command and an env overlay \
             (fdl.{candidate}.yml).\nResolve by renaming one."
        ));
    }

    // Unambiguously an env; consume it and return the rest.
    let mut rest = Vec::with_capacity(args.len() - 1);
    rest.push(args[0].clone());
    rest.extend(args.iter().skip(2).cloned());
    Ok((Some(candidate.clone()), rest))
}

fn is_builtin_name(name: &str) -> bool {
    BUILTINS.iter().any(|(n, _)| *n == name) || HIDDEN_BUILTINS.contains(&name)
}

fn is_project_command(base_config: &std::path::Path, name: &str) -> bool {
    // Must NOT merge env overlays here — that would be circular when the
    // env name also matches a command key. Inspect the raw base only.
    let Ok(project) = config::load_project_with_env(base_config, None) else {
        return false;
    };
    project.commands.contains_key(name)
}

/// Thin wrapper over `parse_or_schema_from` that sets the program name
/// shown in `--help` output so `fdl setup --help` looks like
/// "fdl setup" rather than the crate name.
fn parse_sub<T: flodl_cli::FdlArgsTrait>(program: &str, tail: &[String]) -> T {
    let mut argv = Vec::with_capacity(tail.len() + 1);
    argv.push(program.to_string());
    // tail[0] is the sub-command name (e.g. "setup"); skip it so the derive
    // only sees flags and positionals that belong to the sub-command.
    argv.extend(tail.iter().skip(1).cloned());
    parse_or_schema_from::<T>(&argv)
}

// ---------------------------------------------------------------------------
// libtorch dispatch
// ---------------------------------------------------------------------------

fn dispatch_libtorch(args: &[String]) -> ExitCode {
    let sub = args.get(2).map(String::as_str).unwrap_or("--help");
    match sub {
        "list" => {
            let cli: LibtorchListArgs = parse_sub("fdl libtorch list", &args[2..]);
            cmd_libtorch_list(cli.json)
        }
        "info" => cmd_libtorch_info(),
        "activate" => {
            let cli: LibtorchActivateArgs = parse_sub("fdl libtorch activate", &args[2..]);
            cmd_libtorch_activate(cli.variant.as_deref())
        }
        "download" => {
            let cli: LibtorchDownloadArgs = parse_sub("fdl libtorch download", &args[2..]);
            cmd_libtorch_download(cli)
        }
        "build" => {
            let cli: LibtorchBuildArgs = parse_sub("fdl libtorch build", &args[2..]);
            cmd_libtorch_build(cli)
        }
        "remove" => {
            let cli: LibtorchRemoveArgs = parse_sub("fdl libtorch remove", &args[2..]);
            cmd_libtorch_remove(cli.variant.as_deref())
        }
        "--help" | "-h" => {
            print_libtorch_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown libtorch command: {other}");
            eprintln!();
            print_libtorch_usage();
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// skill dispatch
// ---------------------------------------------------------------------------

fn dispatch_skill(args: &[String]) -> ExitCode {
    let sub = args.get(2).map(String::as_str).unwrap_or("--help");
    match sub {
        "install" => {
            let cli: SkillInstallArgs = parse_sub("fdl skill install", &args[2..]);
            match skill::install(cli.tool.as_deref(), cli.skill.as_deref()) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        "list" => {
            skill::list();
            ExitCode::SUCCESS
        }
        "--help" | "-h" => {
            skill::print_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown skill command: {other}");
            skill::print_usage();
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// libtorch subcommands
// ---------------------------------------------------------------------------

fn cmd_libtorch_list(json: bool) -> ExitCode {
    let ctx = Context::resolve();
    let root = &ctx.root;
    let variants = libtorch::detect::list_variants(root);
    let active = libtorch::detect::read_active(root);
    let active_path = active.as_ref().map(|i| i.path.as_str());

    if json {
        print!("[");
        for (i, v) in variants.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            let is_active = active_path == Some(v.as_str());
            print!(
                "{{\"variant\":\"{}\",\"active\":{}}}",
                util::system::escape_json(v),
                is_active
            );
        }
        println!("]");
    } else if variants.is_empty() {
        println!("No libtorch variants installed.");
        println!("Run: fdl libtorch download");
    } else {
        for v in &variants {
            let marker = if active_path == Some(v.as_str()) {
                " (active)"
            } else {
                ""
            };
            println!("  {v}{marker}");
        }
    }

    ExitCode::SUCCESS
}

fn cmd_libtorch_info() -> ExitCode {
    let ctx = Context::resolve();
    let root = &ctx.root;
    match libtorch::detect::read_active(root) {
        Some(info) => {
            println!("Active:   {}", info.path);
            if let Some(v) = &info.torch_version {
                println!("Version:  {v}");
            }
            if let Some(c) = &info.cuda_version {
                println!("CUDA:     {c}");
            }
            if let Some(a) = &info.archs {
                println!("Archs:    {a}");
            }
            if let Some(s) = &info.source {
                println!("Source:   {s}");
            }
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("No active libtorch variant.");
            eprintln!("Run: fdl libtorch download");
            ExitCode::FAILURE
        }
    }
}

fn cmd_libtorch_activate(variant: Option<&str>) -> ExitCode {
    let ctx = Context::resolve();
    let root = &ctx.root;
    let variant = match variant {
        Some(v) => v,
        None => {
            eprintln!("usage: fdl libtorch activate <variant>");
            eprintln!();
            eprintln!("Available variants:");
            for v in libtorch::detect::list_variants(root) {
                eprintln!("  {v}");
            }
            return ExitCode::FAILURE;
        }
    };

    if !libtorch::detect::is_valid_variant(root, variant) {
        eprintln!("error: '{variant}' is not a valid libtorch variant");
        eprintln!("  Expected: libtorch/{variant}/lib/ to exist");
        eprintln!();
        eprintln!("Available variants:");
        for v in libtorch::detect::list_variants(root) {
            eprintln!("  {v}");
        }
        return ExitCode::FAILURE;
    }

    match libtorch::detect::set_active(root, variant) {
        Ok(()) => {
            println!("Active variant set to: {variant}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_libtorch_download(cli: LibtorchDownloadArgs) -> ExitCode {
    use libtorch::download::{DownloadOpts, Variant};
    use std::path::PathBuf;

    // --cpu and --cuda are mutually exclusive.
    if cli.cpu && cli.cuda.is_some() {
        eprintln!("error: --cpu and --cuda are mutually exclusive");
        return ExitCode::FAILURE;
    }

    let variant = if cli.cpu {
        Variant::Cpu
    } else {
        match cli.cuda.as_deref() {
            Some("12.6") => Variant::Cuda126,
            Some("12.8") => Variant::Cuda128,
            Some(_) => unreachable!("validated by #[option(choices = ...)]"),
            None => Variant::Auto,
        }
    };

    let opts = DownloadOpts {
        variant,
        custom_path: cli.path.map(PathBuf::from),
        activate: !cli.no_activate,
        dry_run: cli.dry_run,
    };

    match libtorch::download::run(opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_libtorch_build(cli: LibtorchBuildArgs) -> ExitCode {
    use libtorch::build::{BuildBackend, BuildOpts};

    if cli.jobs == 0 {
        eprintln!("error: --jobs must be a positive number");
        return ExitCode::FAILURE;
    }

    // --docker and --native are mutually exclusive; absent -> Auto.
    if cli.docker && cli.native {
        eprintln!("error: --docker and --native are mutually exclusive");
        return ExitCode::FAILURE;
    }

    let backend = if cli.docker {
        BuildBackend::Docker
    } else if cli.native {
        BuildBackend::Native
    } else {
        BuildBackend::Auto
    };

    let opts = BuildOpts {
        archs: cli.archs,
        max_jobs: cli.jobs,
        dry_run: cli.dry_run,
        backend,
    };

    match libtorch::build::run(opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_libtorch_remove(variant: Option<&str>) -> ExitCode {
    let ctx = Context::resolve();
    let root = &ctx.root;
    let variant = match variant {
        Some(v) => v,
        None => {
            eprintln!("usage: fdl libtorch remove <variant>");
            eprintln!();
            eprintln!("Installed variants:");
            for v in libtorch::detect::list_variants(root) {
                eprintln!("  {v}");
            }
            return ExitCode::FAILURE;
        }
    };

    match libtorch::manage::remove_variant(root, variant) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

fn cmd_install(check_only: bool, dev: bool) -> ExitCode {
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    let current_version = env!("CARGO_PKG_VERSION");

    let self_path = match env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: cannot determine own binary path: {e}");
            return ExitCode::FAILURE;
        }
    };

    let home = match env::var_os("HOME").or_else(|| env::var_os("USERPROFILE")) {
        Some(h) => PathBuf::from(h),
        None => {
            eprintln!("error: cannot determine home directory");
            return ExitCode::FAILURE;
        }
    };

    let bin_dir = home.join(".local/bin");
    let dest = bin_dir.join("fdl");

    // --check: compare versions
    if check_only {
        let latest = fetch_latest_github_tag();
        println!("Installed: {current_version}");
        // Check if current install is a symlink (dev mode)
        if dest.is_symlink() {
            if let Ok(target) = std::fs::read_link(&dest) {
                println!("Mode:      dev (symlink -> {})", target.display());
            }
        }
        match &latest {
            Some(tag) => {
                println!("Latest:    {tag}");
                if tag == current_version {
                    println!("Up to date.");
                } else {
                    println!("Update available. Run: fdl install");
                }
            }
            None => println!("Latest:    (could not check GitHub)"),
        }
        return ExitCode::SUCCESS;
    }

    // Create ~/.local/bin/ if needed
    if let Err(e) = std::fs::create_dir_all(&bin_dir) {
        eprintln!("error: cannot create {}: {}", bin_dir.display(), e);
        return ExitCode::FAILURE;
    }

    // --dev: symlink to the stable local-build location (~/.cargo/bin/fdl),
    // which is where `cargo install --path flodl-cli` (== `fdl self-build`)
    // drops the freshly-compiled binary. Every subsequent rebuild updates
    // that same path, so the symlink is kept pointing at today's build for
    // free. Falling back to `env::current_exe()` would footgun when the
    // user happens to be running the binary *from* `~/.local/bin/fdl`
    // itself — `canonicalize()` returns that same path and we'd create a
    // symlink to itself (Too many levels of symbolic links).
    if dev {
        #[cfg(unix)]
        {
            let cargo_bin = home.join(".cargo/bin/fdl");
            let self_canonical = self_path.canonicalize().unwrap_or(self_path.clone());

            // Prefer the cargo-install target; fall back to the running
            // binary only when cargo-install isn't present.
            let target = if cargo_bin.is_file() {
                cargo_bin.canonicalize().unwrap_or(cargo_bin)
            } else {
                self_canonical.clone()
            };

            // Self-loop guard: refuse to symlink dest to itself.
            //
            // Compare the fully-resolved `target` against the resolved
            // `dest` when it exists, and fall back to the raw `dest`
            // path otherwise. Raw-only would miss the case where
            // `~/.cargo/bin/fdl` already symlinks (transitively) back
            // to `~/.local/bin/fdl`; canonical-only would fail when
            // `dest` does not yet exist.
            let dest_resolved = dest.canonicalize().unwrap_or_else(|_| dest.clone());
            if target == dest_resolved || target == dest {
                eprintln!(
                    "error: --dev cannot symlink `{}` to itself.",
                    dest.display()
                );
                eprintln!();
                eprintln!(
                    "The currently-running `fdl` is installed at the dest path \
                     and no stable cargo build exists at `{}`.",
                    home.join(".cargo/bin/fdl").display()
                );
                eprintln!();
                eprintln!("Build one first:");
                eprintln!("    cargo install --path flodl-cli");
                eprintln!("    # or (from inside a flodl checkout):");
                eprintln!("    fdl self-build");
                eprintln!();
                eprintln!("Then rerun `fdl install --dev`.");
                return ExitCode::FAILURE;
            }

            // Remove existing (file or symlink) before linking.
            if dest.exists() || dest.is_symlink() {
                let _ = std::fs::remove_file(&dest);
            }

            match std::os::unix::fs::symlink(&target, &dest) {
                Ok(()) => {
                    println!("Linked fdl -> {}", target.display());
                    println!("Global fdl now tracks your local build.");
                    println!("Rebuild with: cargo install --path flodl-cli (or `fdl self-build`).");
                }
                Err(e) => {
                    eprintln!("error: symlink failed: {e}");
                    return ExitCode::FAILURE;
                }
            }

            return print_path_hint(&bin_dir);
        }

        #[cfg(not(unix))]
        {
            eprintln!("--dev mode requires Unix (symlinks). Use fdl install without --dev.");
            return ExitCode::FAILURE;
        }
    }

    // Normal install: download from GitHub if newer, else copy self
    let latest = fetch_latest_github_tag();

    // Check installed version
    let installed_version = if dest.exists() && !dest.is_symlink() {
        let self_canonical = self_path.canonicalize().unwrap_or(self_path.clone());
        let dest_canonical = dest.canonicalize().unwrap_or(dest.clone());
        if self_canonical == dest_canonical {
            Some(current_version.to_string())
        } else {
            Command::new(&dest)
                .arg("version")
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .output()
                .ok()
                .and_then(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .strip_prefix("flodl-cli ")
                        .map(|v| v.to_string())
                })
        }
    } else {
        None
    };

    // Was it a dev symlink? Switching to copy mode.
    let was_dev = dest.is_symlink();
    if was_dev {
        let _ = std::fs::remove_file(&dest);
    }

    // Decide source: download if newer, else copy self
    let source_path;
    let source_version;
    let mut downloaded_path: Option<PathBuf> = None;

    if let Some(ref tag) = latest {
        if tag != current_version && is_newer(tag, current_version) {
            match download_release_binary(tag, &home) {
                Ok(path) => {
                    source_version = tag.clone();
                    source_path = path.clone();
                    downloaded_path = Some(path);
                }
                Err(e) => {
                    eprintln!("warning: could not download {tag}: {e}");
                    eprintln!("Installing current binary ({current_version}) instead.");
                    source_path = self_path.clone();
                    source_version = current_version.to_string();
                }
            }
        } else {
            source_path = self_path.clone();
            source_version = current_version.to_string();
        }
    } else {
        source_path = self_path.clone();
        source_version = current_version.to_string();
    }

    // Check if update is needed
    if !was_dev {
        if let Some(ref iv) = installed_version {
            if iv == &source_version {
                println!("fdl {} is already installed at {}", iv, dest.display());
                if let Some(ref dl) = downloaded_path {
                    let _ = std::fs::remove_file(dl);
                }
                return ExitCode::SUCCESS;
            }
            println!("Updating fdl {iv} -> {source_version}");
        } else {
            println!("Installing fdl {source_version}");
        }
    } else {
        println!("Switching from dev symlink to installed copy ({source_version})");
    }

    // Copy
    if let Err(e) = std::fs::copy(&source_path, &dest) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }

    if let Some(ref dl) = downloaded_path {
        let _ = std::fs::remove_file(dl);
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755));
    }

    println!("Installed fdl {} to {}", source_version, dest.display());
    print_path_hint(&bin_dir)
}

fn print_path_hint(bin_dir: &std::path::Path) -> ExitCode {
    let path_var = env::var("PATH").unwrap_or_default();
    let bin_dir_str = bin_dir.to_string_lossy();
    let in_path = path_var.split(':').any(|p| p == bin_dir_str.as_ref());

    if !in_path {
        println!();
        println!("~/.local/bin is not in your PATH. Add it:");
        println!();
        let shell = env::var("SHELL").unwrap_or_default();
        if shell.contains("zsh") {
            println!("  echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.zshrc && source ~/.zshrc");
        } else {
            println!(
                "  echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc && source ~/.bashrc"
            );
        }
    }

    ExitCode::SUCCESS
}

/// Fetch the latest release tag from GitHub.
fn fetch_latest_github_tag() -> Option<String> {
    use std::process::{Command, Stdio};
    let output = Command::new("curl")
        .args(["-sI", "https://github.com/fab2s/floDl/releases/latest"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.to_lowercase().starts_with("location:") {
            let tag = line.rsplit('/').next()?.trim();
            if !tag.is_empty() {
                return Some(tag.to_string());
            }
        }
    }
    None
}

/// Simple version comparison: "0.3.1" > "0.3.0".
fn is_newer(a: &str, b: &str) -> bool {
    let parse = |v: &str| -> Vec<u32> { v.split('.').filter_map(|s| s.parse().ok()).collect() };
    let va = parse(a);
    let vb = parse(b);
    va > vb
}

/// Download the fdl binary for this platform from a GitHub release.
fn download_release_binary(tag: &str, home: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let os = if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        return Err("unsupported OS".into());
    };

    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        if cfg!(target_os = "macos") {
            "arm64"
        } else {
            "aarch64"
        }
    } else {
        return Err("unsupported architecture".into());
    };

    let ext = if cfg!(target_os = "windows") { ".exe" } else { "" };
    let artifact = format!("flodl-cli-{os}-{arch}{ext}");
    let url = format!("https://github.com/fab2s/floDl/releases/download/{tag}/{artifact}");

    let tmp = home.join(".flodl").join("tmp");
    std::fs::create_dir_all(&tmp).map_err(|e| format!("cannot create temp dir: {e}"))?;
    let dest = tmp.join(format!("fdl-{tag}{ext}"));

    println!("Downloading fdl {tag} from GitHub...");
    util::http::download_file(&url, &dest)?;

    Ok(dest)
}

// ---------------------------------------------------------------------------
// fdl.yaml dispatch
// ---------------------------------------------------------------------------

/// Built-in commands shown in `fdl --help`. Paired with their one-line
/// descriptions. The `is_builtin_name` check uses this list + [`HIDDEN_BUILTINS`]
/// as a single source of truth for "is this reserved?".
const BUILTINS: &[(&str, &str)] = &[
    ("setup", "Interactive guided setup"),
    ("libtorch", "Manage libtorch installations"),
    ("init", "Scaffold a new floDl project"),
    ("diagnose", "System and GPU diagnostics"),
    ("install", "Install or update fdl globally"),
    ("skill", "Manage AI coding assistant skills"),
    ("api-ref", "Generate flodl API reference"),
    ("config", "Inspect resolved project configuration"),
];

/// Reserved top-level names that don't appear in the help banner
/// (internal or already covered elsewhere) but must still be treated as
/// builtins for first-arg env-collision detection.
const HIDDEN_BUILTINS: &[&str] = &["completions", "autocomplete", "version"];

fn load_project_config(
    cwd: &std::path::Path,
    env: Option<&str>,
) -> Option<(config::ProjectConfig, std::path::PathBuf)> {
    let config_path = config::find_config(cwd)?;
    let root = config_path.parent()?.to_path_buf();
    let project = config::load_project_with_env(&config_path, env).ok()?;
    Some((project, root))
}


/// Dispatch an unknown top-level token through the unified `commands:`
/// graph declared in fdl.yml. Handles arbitrary nesting: each step either
/// recurses into a child fdl.yml (Path), executes a self-contained shell
/// command (Run), or invokes the enclosing entry with merged preset
/// fields (Preset).
fn dispatch_config(cmd: &str, args: &[String], env: Option<&str>) -> ExitCode {
    let cwd = env::current_dir().unwrap_or_default();
    let (project, project_root) = match load_project_config(&cwd, env) {
        Some(pair) => pair,
        None => {
            eprintln!("unknown command: {cmd}");
            eprintln!();
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    // Walk the command graph. Presets require an enclosing CommandConfig
    // (they reuse its `entry:`), so we track the most recently loaded
    // one as we descend.
    let mut commands: std::collections::BTreeMap<String, config::CommandSpec> =
        project.commands.clone();
    let mut enclosing_cfg: Option<config::CommandConfig> = None;
    let mut current_dir = project_root.clone();
    let mut name = cmd.to_string();
    let mut tail_idx = 2usize; // args[2..] is the tail after the first command

    loop {
        let spec = match commands.get(&name) {
            Some(s) => s.clone(),
            None => {
                eprintln!("unknown command: {name}");
                eprintln!();
                run::print_project_help(&project, &project_root, BUILTINS, env);
                return ExitCode::FAILURE;
            }
        };

        let tail = &args[tail_idx..];

        let kind = match spec.kind() {
            Ok(k) => k,
            Err(e) => {
                eprintln!("error in command `{name}`: {e}");
                return ExitCode::FAILURE;
            }
        };

        match kind {
            config::CommandKind::Run => {
                let run_cmd = spec.run.as_deref().expect("Run kind guarantees run is set");
                return run::exec_script(run_cmd, spec.docker.as_deref(), &current_dir);
            }
            config::CommandKind::Path => {
                match classify_path_step(&spec, &name, &current_dir, tail, env) {
                    PathOutcome::LoadFailed(msg) => {
                        eprintln!("error: {msg}");
                        return ExitCode::FAILURE;
                    }
                    PathOutcome::Descend {
                        child,
                        new_dir,
                        new_name,
                    } => {
                        commands = child.commands.clone();
                        enclosing_cfg = Some(*child);
                        current_dir = new_dir;
                        name = new_name;
                        tail_idx += 1;
                    }
                    PathOutcome::ShowHelp { child } => {
                        run::print_command_help(&child, &name);
                        return ExitCode::SUCCESS;
                    }
                    PathOutcome::RefreshSchema { child, child_dir } => {
                        return cmd_refresh_schema(&child, &child_dir, &name);
                    }
                    PathOutcome::Exec { child, child_dir } => {
                        return run::exec_command(
                            &child,
                            None,
                            tail,
                            &child_dir,
                            &project_root,
                        );
                    }
                }
            }
            config::CommandKind::Preset => {
                let Some(encl) = enclosing_cfg.as_ref() else {
                    eprintln!(
                        "error: preset command `{name}` has no enclosing \
                         fdl.yml (top-level commands must be `run:` or `path:`)"
                    );
                    return ExitCode::FAILURE;
                };

                if tail.iter().any(|a| a == "--help" || a == "-h") {
                    // Label the preset with the enclosing command path.
                    let parent_label = current_dir
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("");
                    run::print_preset_help(encl, parent_label, &name);
                    return ExitCode::SUCCESS;
                }

                return run::exec_command(encl, Some(&name), tail, &current_dir, &project_root);
            }
        }
    }
}

/// `fdl config show [<env>]` — print the resolved merged config.
///
/// `tail` is `args[1..]`: tail[0] is always "config", tail[1] is the
/// sub-command ("show"), tail[2..] carry options (an optional explicit
/// `<env>` that overrides the first-arg env detection).
fn cmd_config_show(tail: &[String], active_env: Option<&str>) -> ExitCode {
    let sub = tail.get(1).map(String::as_str).unwrap_or("--help");
    match sub {
        "show" => {}
        "--help" | "-h" => {
            print_config_usage();
            return ExitCode::SUCCESS;
        }
        other => {
            eprintln!("unknown config sub-command: {other}");
            eprintln!();
            print_config_usage();
            return ExitCode::FAILURE;
        }
    }

    // Optional explicit env override: `fdl config show prod`.
    let explicit_env = tail.get(2).map(String::as_str);
    let target_env = explicit_env.or(active_env);

    let cwd = env::current_dir().unwrap_or_default();
    let base = match config::find_config(&cwd) {
        Some(p) => p,
        None => {
            eprintln!("error: no fdl.yml found in {} or parent directories", cwd.display());
            return ExitCode::FAILURE;
        }
    };

    let merged = match config::load_merged_value(&base, target_env) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Source annotations header — which files were layered, in order.
    let sources = config::config_layer_sources(&base, target_env);
    for (i, p) in sources.iter().enumerate() {
        let tag = if i == 0 { "base" } else { "overlay" };
        println!("# {tag}: {}", p.display());
    }
    if target_env.is_some() && sources.len() == 1 {
        println!("# (env overlay requested but not found next to base)");
    }
    println!("#");

    match serde_yaml::to_string(&merged) {
        Ok(s) => {
            print!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: cannot serialize merged config: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_config_usage() {
    println!("fdl config -- inspect resolved project configuration");
    println!();
    println!("USAGE:");
    println!("    fdl config show [<env>]");
    println!();
    println!("Without an env argument, prints the base fdl.yml. With an env argument");
    println!("(e.g. `fdl config show ci`), prints the base deep-merged with");
    println!("fdl.<env>.yml. When invoked through the first-arg form");
    println!("(`fdl ci config show`), the env is already active and no extra");
    println!("argument is needed.");
}

/// `fdl <cmd> --refresh-schema`: run `<entry> --fdl-schema`, validate, cache.
///
/// Required for cargo-based entries, which are never auto-probed (compile
/// latency would ruin `--help`). Users build once, then run this explicitly.
fn cmd_refresh_schema(
    cmd_config: &config::CommandConfig,
    cmd_dir: &std::path::Path,
    cmd_name: &str,
) -> ExitCode {
    let entry = match &cmd_config.entry {
        Some(e) => e.as_str(),
        None => {
            eprintln!(
                "error: no entry point defined in {}/fdl.yml",
                cmd_dir.display()
            );
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Probing `{entry} --fdl-schema`...");
    let schema = match schema_cache::probe(entry, cmd_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            if schema_cache::is_cargo_entry(entry) {
                eprintln!();
                eprintln!("Hint: cargo-based entries must be built first.");
                eprintln!("Build with the right features, then rerun this command.");
            }
            return ExitCode::FAILURE;
        }
    };

    let cache = schema_cache::cache_path(cmd_dir, cmd_name);
    if let Err(e) = schema_cache::write_cache(&cache, &schema) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    eprintln!("Cached schema for `{cmd_name}` at {}", cache.display());
    eprintln!(
        "  {} options, {} positional args",
        schema.options.len(),
        schema.args.len()
    );
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

fn print_usage() {
    println!("flodl-cli {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("The floDl companion tool: setup, libtorch, diagnostics, API reference.");
    println!("Works anywhere. Uses project root when available, ~/.flodl/ otherwise.");
    println!();
    println!("USAGE:");
    println!("    fdl [options] <command> [command-options]");
    println!();
    println!("GLOBAL OPTIONS:");
    println!("    -v                 Verbose output (DDP sync, data loading detail)");
    println!("    -vv                Debug output (per-batch timing, loop internals)");
    println!("    -vvv               Trace output (maximum detail)");
    println!("    -q, --quiet        Suppress all non-error output");
    println!();
    println!("COMMANDS:");
    println!("    setup              Interactive guided setup");
    println!("    libtorch           Manage libtorch installations");
    println!("    init <name>        Scaffold a new floDl project");
    println!("        --docker       Generate Docker-based scaffold (libtorch baked in)");
    println!("    diagnose           System and GPU diagnostics");
    println!("        --json         Output as JSON");
    println!("    install             Install or update fdl globally (~/.local/bin)");
    println!("        --check        Check for updates without installing");
    println!("        --dev          Symlink to current binary (tracks local builds)");
    println!("    skill              Manage AI coding assistant skills");
    println!("        install        Install skills for detected tool (Claude, Cursor, ...)");
    println!("        list           Show available skills");
    println!("    api-ref            Generate flodl API reference");
    println!("        --json         Output as JSON");
    println!("        --path <dir>   Explicit flodl source path");
    println!("    version            Show version");
    println!();
    println!("Run `fdl --help` or `fdl <command> --help` for details.");
    println!();
    println!("INSTALL:");
    println!("    cargo install flodl-cli    # from crates.io");
    println!("    fdl install                # make current binary global (~/.local/bin/fdl)");
    println!();
    println!("EXAMPLES:");
    println!("    fdl setup                  # first-time setup");
    println!("    fdl libtorch download      # download pre-built libtorch");
    println!("    fdl libtorch list          # show installed variants");
    println!("    fdl init my-model          # scaffold with mounted libtorch");
    println!("    fdl diagnose               # hardware + compatibility report");
    println!("    fdl diagnose --json        # machine-readable output");
    println!("    fdl api-ref                # generate API reference");
    println!("    fdl api-ref --json         # structured JSON for tooling");
}

// ---------------------------------------------------------------------------
// Global verbosity flags
// ---------------------------------------------------------------------------

/// Extract verbosity flags from args, returning filtered args and the
/// `FLODL_VERBOSITY` value: Quiet=0, Normal=1, Verbose=2, Debug=3, Trace=4.
///
/// Supports `-v` (Verbose), `-vv` (Debug), `-vvv` (Trace), `--quiet`/`-q` (Quiet).
/// Flags can appear anywhere in the arg list and are stripped before dispatch.
fn extract_verbosity(args: &[String]) -> (Vec<String>, Option<u8>) {
    let mut level: Option<u8> = None;
    let mut filtered = Vec::with_capacity(args.len());

    for arg in args {
        match arg.as_str() {
            "-vvv" => level = Some(4), // Trace
            "-vv" => level = Some(3),  // Debug
            "-v" => level = Some(2),   // Verbose
            "--quiet" | "-q" => level = Some(0), // Quiet
            _ => filtered.push(arg.clone()),
        }
    }

    (filtered, level)
}

fn print_libtorch_usage() {
    println!("fdl libtorch -- manage libtorch installations");
    println!();
    println!("USAGE:");
    println!("    fdl libtorch <command> [options]");
    println!();
    println!("COMMANDS:");
    println!("    download           Download pre-built libtorch");
    println!("        --cpu          Force CPU variant");
    println!("        --cuda <ver>   Specific CUDA version (12.6, 12.8)");
    println!("    build              Build libtorch from source");
    println!("        --docker       Force Docker build (isolated, reproducible)");
    println!("        --native       Force native build (faster, requires host toolchain)");
    println!("        --archs <list> Override CUDA architectures");
    println!("        --jobs <n>     Parallel compilation jobs (default: 6)");
    println!("    list               Show installed variants");
    println!("        --json         JSON output");
    println!("    activate <name>    Set active variant");
    println!("    remove <name>      Remove a variant");
    println!("    info               Show active variant details");
}
