//! flodl-cli: command-line tool for the floDl deep learning framework.
//!
//! Provides hardware diagnostics, libtorch management, and project scaffolding.
//! Pure Rust binary with no libtorch dependency (GPU detection via nvidia-smi).
//!
//! Works both inside a floDl project and standalone. When standalone, libtorch
//! is managed under `~/.flodl/` (override with `$FLODL_HOME`).

pub mod context;
mod diagnose;
mod init;
mod libtorch;
mod setup;
mod util;

use std::env;
use std::process::ExitCode;

use context::Context;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
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
        "help" | "--help" | "-h" => {
            print_usage();
            ExitCode::SUCCESS
        }
        "version" | "--version" | "-V" => {
            println!("flodl-cli {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command: {}", other);
            eprintln!();
            print_usage();
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
// Usage
// ---------------------------------------------------------------------------

fn print_usage() {
    println!("flodl-cli {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("libtorch manager and GPU diagnostic tool for Rust deep learning.");
    println!("Works inside floDl projects and standalone (~/.flodl).");
    println!();
    println!("USAGE:");
    println!("    fdl <command> [options]");
    println!();
    println!("COMMANDS:");
    println!("    setup              Interactive guided setup");
    println!("    libtorch           Manage libtorch installations");
    println!("    init <name>        Scaffold a new floDl project");
    println!("        --docker       Generate Docker-based scaffold (libtorch baked in)");
    println!("    diagnose           System and GPU diagnostics");
    println!("        --json         Output as JSON");
    println!("    help               Show this help");
    println!("    version            Show version");
    println!();
    println!("INSTALL:");
    println!("    cargo install flodl-cli    # installs 'fdl' binary");
    println!();
    println!("EXAMPLES:");
    println!("    fdl setup                  # first-time setup");
    println!("    fdl libtorch download      # download pre-built libtorch");
    println!("    fdl libtorch list          # show installed variants");
    println!("    fdl init my-model          # scaffold with mounted libtorch");
    println!("    fdl diagnose               # hardware + compatibility report");
    println!("    fdl diagnose --json        # machine-readable output");
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
