//! flodl-cli: command-line tool for the floDl deep learning framework.
//!
//! Provides hardware diagnostics, libtorch management, and project scaffolding.
//! Pure Rust binary with no libtorch dependency (GPU detection via nvidia-smi).
//!
//! Works both inside a floDl project and standalone. When standalone, libtorch
//! is managed under `~/.flodl/` (override with `$FLODL_HOME`).

mod api_ref;
mod completions;
mod config;
pub mod context;
mod diagnose;
mod init;
mod libtorch;
mod run;
mod setup;
mod skill;
mod style;
mod util;

use std::env;
use std::process::ExitCode;

use context::Context;

fn main() -> ExitCode {
    let raw_args: Vec<String> = env::args().collect();

    // Extract global verbosity flags before command dispatch.
    // Sets FLODL_VERBOSITY so child processes (including Docker) inherit it.
    let (args, verbosity) = extract_verbosity(&raw_args);
    if let Some(v) = verbosity {
        // SAFETY: called before any threads are spawned.
        unsafe { env::set_var("FLODL_VERBOSITY", v.to_string()); }
    }

    let cmd = args.get(1).map(String::as_str).unwrap_or("help");

    match cmd {
        "setup" => {
            let mut opts = setup::SetupOpts::default();
            for arg in &args[2..] {
                match arg.as_str() {
                    "--non-interactive" | "-y" => opts.non_interactive = true,
                    "--force" => opts.force = true,
                    other => {
                        eprintln!("unknown option for setup: {}", other);
                        return ExitCode::FAILURE;
                    }
                }
            }
            match setup::run(opts) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {}", e);
                    ExitCode::FAILURE
                }
            }
        }
        "libtorch" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "list" => {
                    let json = args.iter().any(|a| a == "--json");
                    cmd_libtorch_list(json)
                }
                "info" => cmd_libtorch_info(),
                "activate" => {
                    let variant = args.get(3).map(String::as_str);
                    cmd_libtorch_activate(variant)
                }
                "download" => cmd_libtorch_download(&args[2..]),
                "build" => cmd_libtorch_build(&args[2..]),
                "remove" => {
                    let variant = args.get(3).map(String::as_str);
                    cmd_libtorch_remove(variant)
                }
                "help" | "--help" | "-h" => {
                    print_libtorch_usage();
                    ExitCode::SUCCESS
                }
                other => {
                    eprintln!("unknown libtorch command: {}", other);
                    eprintln!();
                    print_libtorch_usage();
                    ExitCode::FAILURE
                }
            }
        }
        "diagnose" => {
            let json = args.iter().any(|a| a == "--json");
            diagnose::run(json);
            ExitCode::SUCCESS
        }
        "api-ref" => {
            let json = args.iter().any(|a| a == "--json");
            let path = args.iter().position(|a| a == "--path")
                .and_then(|i| args.get(i + 1))
                .map(String::as_str);
            match api_ref::run(json, path) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {}", e);
                    ExitCode::FAILURE
                }
            }
        }
        "init" => {
            let name = args.get(2).map(String::as_str);
            let docker = args.iter().any(|a| a == "--docker");
            match init::run(name, docker) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {}", e);
                    ExitCode::FAILURE
                }
            }
        }
        "install" => {
            let check = args.iter().any(|a| a == "--check");
            let dev = args.iter().any(|a| a == "--dev");
            cmd_install(check, dev)
        }
        "skill" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "install" => {
                    let tool = args.iter().position(|a| a == "--tool")
                        .and_then(|i| args.get(i + 1))
                        .map(String::as_str);
                    let skill_name = args.iter().position(|a| a == "--skill")
                        .and_then(|i| args.get(i + 1))
                        .map(String::as_str);
                    match skill::install(tool, skill_name) {
                        Ok(()) => ExitCode::SUCCESS,
                        Err(e) => {
                            eprintln!("error: {}", e);
                            ExitCode::FAILURE
                        }
                    }
                }
                "list" => {
                    skill::list();
                    ExitCode::SUCCESS
                }
                "help" | "--help" | "-h" => {
                    skill::print_usage();
                    ExitCode::SUCCESS
                }
                other => {
                    eprintln!("unknown skill command: {}", other);
                    skill::print_usage();
                    ExitCode::FAILURE
                }
            }
        }
        "completions" => {
            let shell = args.get(2).map(String::as_str).unwrap_or("bash");
            let cwd = env::current_dir().unwrap_or_default();
            let project = load_project_config(&cwd);
            completions::generate(shell, project.as_ref().map(|(p, r)| (p, r.as_path())));
            ExitCode::SUCCESS
        }
        "autocomplete" => {
            let cwd = env::current_dir().unwrap_or_default();
            let project = load_project_config(&cwd);
            completions::autocomplete(project.as_ref().map(|(p, r)| (p, r.as_path())));
            ExitCode::SUCCESS
        }
        "help" | "--help" | "-h" => {
            let cwd = env::current_dir().unwrap_or_default();
            if let Some((project, root)) = load_project_config(&cwd) {
                run::print_project_help(&project, &root, BUILTINS);
            } else {
                print_usage();
            }
            ExitCode::SUCCESS
        }
        "version" | "--version" | "-V" => {
            println!("flodl-cli {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        other => dispatch_config(other, &args)
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
    } else {
        if variants.is_empty() {
            println!("No libtorch variants installed.");
            println!("Run: fdl libtorch download");
        } else {
            for v in &variants {
                let marker = if active_path == Some(v.as_str()) {
                    " (active)"
                } else {
                    ""
                };
                println!("  {}{}", v, marker);
            }
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
                println!("Version:  {}", v);
            }
            if let Some(c) = &info.cuda_version {
                println!("CUDA:     {}", c);
            }
            if let Some(a) = &info.archs {
                println!("Archs:    {}", a);
            }
            if let Some(s) = &info.source {
                println!("Source:   {}", s);
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
                eprintln!("  {}", v);
            }
            return ExitCode::FAILURE;
        }
    };

    if !libtorch::detect::is_valid_variant(root, variant) {
        eprintln!("error: '{}' is not a valid libtorch variant", variant);
        eprintln!("  Expected: libtorch/{}/lib/ to exist", variant);
        eprintln!();
        eprintln!("Available variants:");
        for v in libtorch::detect::list_variants(root) {
            eprintln!("  {}", v);
        }
        return ExitCode::FAILURE;
    }

    match libtorch::detect::set_active(root, variant) {
        Ok(()) => {
            println!("Active variant set to: {}", variant);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn cmd_libtorch_download(args: &[String]) -> ExitCode {
    use libtorch::download::{DownloadOpts, Variant};
    use std::path::PathBuf;

    let mut opts = DownloadOpts::default();

    let mut i = 1; // skip "download" itself
    while i < args.len() {
        match args[i].as_str() {
            "--cpu" => {
                opts.variant = Variant::Cpu;
                i += 1;
            }
            "--cuda" => {
                if i + 1 >= args.len() {
                    eprintln!("error: --cuda requires a version (12.6 or 12.8)");
                    return ExitCode::FAILURE;
                }
                i += 1;
                match args[i].as_str() {
                    "12.6" => opts.variant = Variant::Cuda126,
                    "12.8" => opts.variant = Variant::Cuda128,
                    other => {
                        eprintln!("error: unsupported CUDA version '{}'", other);
                        eprintln!("  Available: 12.6, 12.8 (or --cpu)");
                        return ExitCode::FAILURE;
                    }
                }
                i += 1;
            }
            "--path" => {
                if i + 1 >= args.len() {
                    eprintln!("error: --path requires a directory");
                    return ExitCode::FAILURE;
                }
                i += 1;
                opts.custom_path = Some(PathBuf::from(&args[i]));
                i += 1;
            }
            "--no-activate" => {
                opts.activate = false;
                i += 1;
            }
            "--dry-run" => {
                opts.dry_run = true;
                i += 1;
            }
            other => {
                eprintln!("unknown option: {}", other);
                eprintln!();
                eprintln!("Usage: fdl libtorch download [--cpu | --cuda 12.6|12.8] [--path DIR] [--dry-run]");
                return ExitCode::FAILURE;
            }
        }
    }

    match libtorch::download::run(opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn cmd_libtorch_build(args: &[String]) -> ExitCode {
    use libtorch::build::{BuildOpts, BuildBackend};

    let mut opts = BuildOpts::default();

    let mut i = 1; // skip "build" itself
    while i < args.len() {
        match args[i].as_str() {
            "--archs" => {
                if i + 1 >= args.len() {
                    eprintln!("error: --archs requires a value (e.g. \"6.1;12.0\")");
                    return ExitCode::FAILURE;
                }
                i += 1;
                opts.archs = Some(args[i].clone());
                i += 1;
            }
            "--jobs" => {
                if i + 1 >= args.len() {
                    eprintln!("error: --jobs requires a number");
                    return ExitCode::FAILURE;
                }
                i += 1;
                match args[i].parse::<usize>() {
                    Ok(n) if n > 0 => opts.max_jobs = n,
                    _ => {
                        eprintln!("error: --jobs must be a positive number");
                        return ExitCode::FAILURE;
                    }
                }
                i += 1;
            }
            "--docker" => {
                opts.backend = BuildBackend::Docker;
                i += 1;
            }
            "--native" => {
                opts.backend = BuildBackend::Native;
                i += 1;
            }
            "--dry-run" => {
                opts.dry_run = true;
                i += 1;
            }
            other => {
                eprintln!("unknown option: {}", other);
                eprintln!();
                eprintln!("Usage: fdl libtorch build [--docker | --native] [--archs \"6.1;12.0\"] [--jobs N] [--dry-run]");
                return ExitCode::FAILURE;
            }
        }
    }

    match libtorch::build::run(opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", e);
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
                eprintln!("  {}", v);
            }
            return ExitCode::FAILURE;
        }
    };

    match libtorch::manage::remove_variant(root, variant) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", e);
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
            eprintln!("error: cannot determine own binary path: {}", e);
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
        println!("Installed: {}", current_version);
        // Check if current install is a symlink (dev mode)
        if dest.is_symlink() {
            if let Ok(target) = std::fs::read_link(&dest) {
                println!("Mode:      dev (symlink -> {})", target.display());
            }
        }
        match &latest {
            Some(tag) => {
                println!("Latest:    {}", tag);
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

    // --dev: symlink to current binary (always tracks latest build)
    if dev {
        let self_canonical = self_path.canonicalize().unwrap_or(self_path.clone());

        // Remove existing (file or symlink)
        if dest.exists() || dest.is_symlink() {
            let _ = std::fs::remove_file(&dest);
        }

        #[cfg(unix)]
        {
            match std::os::unix::fs::symlink(&self_canonical, &dest) {
                Ok(()) => {
                    println!("Linked fdl -> {}", self_canonical.display());
                    println!("Global fdl now tracks your local build.");
                    println!("Rebuild with: cargo build --release -p flodl-cli");
                }
                Err(e) => {
                    eprintln!("error: symlink failed: {}", e);
                    return ExitCode::FAILURE;
                }
            }
        }

        #[cfg(not(unix))]
        {
            eprintln!("--dev mode requires Unix (symlinks). Use fdl install without --dev.");
            return ExitCode::FAILURE;
        }

        return print_path_hint(&bin_dir);
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
                    eprintln!("warning: could not download {}: {}", tag, e);
                    eprintln!("Installing current binary ({}) instead.", current_version);
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
            println!("Updating fdl {} -> {}", iv, source_version);
        } else {
            println!("Installing fdl {}", source_version);
        }
    } else {
        println!("Switching from dev symlink to installed copy ({})", source_version);
    }

    // Copy
    if let Err(e) = std::fs::copy(&source_path, &dest) {
        eprintln!("error: {}", e);
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
            println!("  echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc && source ~/.bashrc");
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
    let parse = |v: &str| -> Vec<u32> {
        v.split('.').filter_map(|s| s.parse().ok()).collect()
    };
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
        if cfg!(target_os = "macos") { "arm64" } else { "aarch64" }
    } else {
        return Err("unsupported architecture".into());
    };

    let ext = if cfg!(target_os = "windows") { ".exe" } else { "" };
    let artifact = format!("flodl-cli-{}-{}{}", os, arch, ext);
    let url = format!(
        "https://github.com/fab2s/floDl/releases/download/{}/{}",
        tag, artifact
    );

    let tmp = home.join(".flodl").join("tmp");
    std::fs::create_dir_all(&tmp)
        .map_err(|e| format!("cannot create temp dir: {}", e))?;
    let dest = tmp.join(format!("fdl-{}{}", tag, ext));

    println!("Downloading fdl {} from GitHub...", tag);
    util::http::download_file(&url, &dest)?;

    Ok(dest)
}


// ---------------------------------------------------------------------------
// fdl.yaml dispatch
// ---------------------------------------------------------------------------

const BUILTINS: &[(&str, &str)] = &[
    ("setup", "Interactive guided setup"),
    ("libtorch", "Manage libtorch installations"),
    ("init", "Scaffold a new floDl project"),
    ("diagnose", "System and GPU diagnostics"),
    ("install", "Install or update fdl globally"),
    ("skill", "Manage AI coding assistant skills"),
    ("api-ref", "Generate flodl API reference"),
];

fn load_project_config(cwd: &std::path::Path) -> Option<(config::ProjectConfig, std::path::PathBuf)> {
    let config_path = config::find_config(cwd)?;
    let root = config_path.parent()?.to_path_buf();
    let project = config::load_project(&config_path).ok()?;
    Some((project, root))
}

/// Try to dispatch an unknown command via fdl.yaml scripts and commands.
fn dispatch_config(cmd: &str, args: &[String]) -> ExitCode {
    let cwd = env::current_dir().unwrap_or_default();
    let (project, root) = match load_project_config(&cwd) {
        Some(pair) => pair,
        None => {
            eprintln!("unknown command: {cmd}");
            eprintln!();
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    // Check scripts.
    if let Some(script) = project.scripts.get(cmd) {
        return run::exec_script(script.command(), script.docker_service(), &root);
    }

    // Check commands.
    for cmd_path in &project.commands {
        let short = config::command_name(cmd_path);
        if short != cmd {
            continue;
        }

        let cmd_dir = root.join(cmd_path);
        let cmd_config = match config::load_command(&cmd_dir) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };

        // --help for the sub-command.
        if args.get(2).map(String::as_str) == Some("--help")
            || args.get(2).map(String::as_str) == Some("-h")
        {
            run::print_command_help(&cmd_config, short);
            return ExitCode::SUCCESS;
        }

        // Check if first sub-arg matches a job name.
        let first_sub = args.get(2).map(String::as_str);
        let (job_name, extra_start) = match first_sub {
            Some(name) if cmd_config.jobs.contains_key(name) => (Some(name), 3),
            _ => (None, 2),
        };

        // Recursive help: fdl ddp-bench <job> --help
        if let Some(jn) = &job_name {
            let has_help = args[extra_start..]
                .iter()
                .any(|a| a == "--help" || a == "-h");
            if has_help {
                run::print_job_help(&cmd_config, short, jn);
                return ExitCode::SUCCESS;
            }
        }

        let extra = args[extra_start..].to_vec();
        return run::exec_command(&cmd_config, job_name, &extra, &cmd_dir, &root);
    }

    // Not found.
    eprintln!("unknown command: {cmd}");
    eprintln!();
    run::print_project_help(&project, &root, BUILTINS);
    ExitCode::FAILURE
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
    println!("    help               Show this help");
    println!("    version            Show version");
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
            "-vvv" => level = Some(4),  // Trace
            "-vv" => level = Some(3),   // Debug
            "-v" => level = Some(2),    // Verbose
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
