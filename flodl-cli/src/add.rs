//! `fdl add flodl-hf` -- scaffold a flodl-hf playground inside the current flodl project.
//!
//! Drops `./flodl-hf/` as a standalone cargo crate: pinned `flodl` +
//! `flodl-hf` deps, a one-file `AutoModel` example, `fdl.yml` with
//! runnable commands, and a README documenting feature flavors and
//! the convert workflow.
//!
//! Scope contract: no mutation of the user's root `Cargo.toml` or
//! `fdl.yml`. The playground is a side crate the user runs for
//! discovery; wiring flodl-hf into their main code stays their call,
//! documented in the generated README.
//!
//! Targets accepted: `flodl-hf` and its alias `hf`. Other targets
//! surface a loud error listing the supported set.

use std::fs;
use std::path::{Path, PathBuf};

/// Scaffold templates baked into the binary at compile time. Paths are
/// relative to `flodl-cli/src/add.rs`, which resolves to
/// `<workspace>/flodl-hf/scaffold/...` in the rdl workspace. Publishing
/// `flodl-cli` standalone to crates.io currently depends on these files
/// being present alongside; documented as a known gap.
const TEMPLATE_CARGO_TOML: &str = include_str!("../../flodl-hf/scaffold/Cargo.toml");
const TEMPLATE_MAIN_RS: &str = include_str!("../../flodl-hf/scaffold/src/main.rs");
const TEMPLATE_FDL_YML: &str = include_str!("../../flodl-hf/scaffold/fdl.yml.example");
const TEMPLATE_README: &str = include_str!("../../flodl-hf/scaffold/README.md");
const TEMPLATE_GITIGNORE: &str = include_str!("../../flodl-hf/scaffold/.gitignore");

pub fn run(target: Option<&str>) -> Result<(), String> {
    let target = target.ok_or(
        "usage: fdl add <target>\n\nSupported targets:\n    flodl-hf    HuggingFace integration (pre-built BERT / RoBERTa / DistilBERT, Hub loader, tokenizer)",
    )?;
    let cwd = std::env::current_dir()
        .map_err(|e| format!("cannot read current directory: {e}"))?;
    match target {
        "flodl-hf" | "hf" => add_flodl_hf_at(&cwd),
        other => Err(format!(
            "unknown target: {other:?}\n\n\
             Supported targets:\n    \
             flodl-hf    HuggingFace integration\n\n\
             (More targets land as the flodl ecosystem grows.)",
        )),
    }
}

/// Scaffold `flodl-hf/` under `base`. Entry point for `fdl add flodl-hf`
/// (with `base = cwd`) and `fdl init --with-hf` / interactive-mode
/// follow-up (with `base = the freshly-scaffolded project dir`). The
/// base dir must contain a `Cargo.toml` with a pinnable `flodl`
/// dependency.
pub fn add_flodl_hf_at(cwd: &Path) -> Result<(), String> {
    // Must be run from a flodl project root (Cargo.toml with flodl dep).
    let cargo_toml = cwd.join("Cargo.toml");
    if !cargo_toml.exists() {
        return Err(format!(
            "no Cargo.toml in {}.\n\n\
             fdl add flodl-hf must run from a flodl project root.\n\
             Start with `fdl init <name>` if you don't have one yet.",
            cwd.display(),
        ));
    }

    // flodl-hf makes no sense without a functioning flodl project: every
    // runnable command assumes a Cargo.toml + fdl.yml pair are already
    // present. Enforce that invariant loudly so the user isn't left
    // with a dead sub-crate.
    if !has_fdl_config(cwd) {
        return Err(format!(
            "no fdl.yml (nor fdl.yml.example) in {}.\n\n\
             fdl add flodl-hf expects an initialised flodl project: \
             Docker or native mode already chosen, fdl.yml present. \
             Run `fdl init <name>` first, or cd into an existing flodl project.",
            cwd.display(),
        ));
    }

    let flodl_version = detect_flodl_version(&cargo_toml)?;
    let mode = detect_project_mode(cwd);

    // Refuse to overwrite an existing flodl-hf/ dir.
    let dest = cwd.join("flodl-hf");
    if dest.exists() {
        return Err(format!(
            "{} already exists.\n\n\
             Remove it first, or keep it. `fdl add flodl-hf` does not overwrite.",
            dest.display(),
        ));
    }

    // Scaffold.
    fs::create_dir_all(dest.join("src"))
        .map_err(|e| format!("cannot create {}: {e}", dest.join("src").display()))?;

    write_file(
        &dest.join("Cargo.toml"),
        &substitute_version(TEMPLATE_CARGO_TOML, &flodl_version),
    )?;
    write_file(&dest.join("src/main.rs"), TEMPLATE_MAIN_RS)?;
    let fdl_yml = render_fdl_yml(TEMPLATE_FDL_YML, mode);
    write_file(&dest.join("fdl.yml.example"), &fdl_yml)?;
    write_file(&dest.join("fdl.yml"), &fdl_yml)?;
    write_file(
        &dest.join("README.md"),
        &substitute_version(TEMPLATE_README, &flodl_version),
    )?;
    write_file(&dest.join(".gitignore"), TEMPLATE_GITIGNORE)?;

    print_next_steps(&flodl_version, mode);
    Ok(())
}

/// Host-project execution mode, inferred from file presence.
///
/// `fdl init` writes `docker-compose.yml` for its Mounted and Docker
/// modes, and omits it for Native. Scaffolded commands follow the
/// same convention: Docker modes dispatch to the `dev` service,
/// Native runs directly on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectMode {
    Docker,
    Native,
}

fn has_fdl_config(cwd: &Path) -> bool {
    cwd.join("fdl.yml").exists() || cwd.join("fdl.yml.example").exists()
}

fn detect_project_mode(cwd: &Path) -> ProjectMode {
    if cwd.join("docker-compose.yml").exists() {
        ProjectMode::Docker
    } else {
        ProjectMode::Native
    }
}

/// In Native mode, strip the `    docker: dev` lines from the scaffold
/// `fdl.yml` so cargo commands run directly on the host instead of
/// trying to dispatch into a non-existent Docker service. Matches the
/// indentation produced by the template exactly; anything else is left
/// alone.
fn render_fdl_yml(template: &str, mode: ProjectMode) -> String {
    match mode {
        ProjectMode::Docker => template.to_string(),
        ProjectMode::Native => template
            .lines()
            .filter(|l| l.trim() != "docker: dev")
            .collect::<Vec<&str>>()
            .join("\n")
            + "\n",
    }
}

/// Parse `Cargo.toml` for the `flodl` dependency version.
///
/// Recognises three forms:
/// - `flodl = "0.5.1"` — plain version string
/// - `flodl = { version = "0.5.1", ... }` — table form
/// - `flodl = { workspace = true }` — workspace inheritance (reads from
///   the workspace root's `Cargo.toml`)
///
/// Errors on: no flodl dep found, git-only dep (no pinnable version),
/// or path-only dep outside this repo (no version to pin against).
fn detect_flodl_version(cargo_toml: &Path) -> Result<String, String> {
    let content = fs::read_to_string(cargo_toml)
        .map_err(|e| format!("cannot read {}: {e}", cargo_toml.display()))?;

    if let Some(v) = parse_flodl_dep(&content)? {
        return Ok(v);
    }

    // Workspace inheritance: climb to find the workspace root.
    if let Some(ws_root) = find_workspace_root(cargo_toml) {
        let ws_content = fs::read_to_string(&ws_root)
            .map_err(|e| format!("cannot read workspace {}: {e}", ws_root.display()))?;
        if let Some(v) = parse_flodl_dep(&ws_content)? {
            return Ok(v);
        }
    }

    Err(format!(
        "no flodl dependency found in {}.\n\n\
         fdl add flodl-hf needs to pin flodl-hf to the same version as \
         flodl. Add `flodl = \"X.Y.Z\"` to [dependencies] first, or run \
         `fdl init <name>` to scaffold a flodl project.",
        cargo_toml.display(),
    ))
}

/// Extract the flodl version from a Cargo.toml's textual content.
///
/// Returns `Ok(Some(version))` on a pinnable version, `Ok(None)` when
/// no flodl dep is present, and `Err(...)` when the dep exists but is
/// git-only / path-only (no version to pin against).
fn parse_flodl_dep(content: &str) -> Result<Option<String>, String> {
    let lines: Vec<&str> = content.lines().collect();

    // Find a line that declares `flodl = ...` under a [dependencies]
    // or [workspace.dependencies] table. We accept any form whose LHS
    // matches `flodl`; the inline value on the RHS tells us the shape.
    let mut in_dep_table = false;
    for line in &lines {
        let t = line.trim();
        if t.starts_with('[') {
            // Only consider tables that declare dependencies.
            in_dep_table = matches!(
                t,
                "[dependencies]" | "[workspace.dependencies]" | "[dev-dependencies]",
            );
            continue;
        }
        if !in_dep_table {
            continue;
        }
        // Match `flodl = ...` exactly (not flodl-hf, flodl-sys, ...).
        let after_key = match t.strip_prefix("flodl") {
            Some(rest) => rest.trim_start(),
            None => continue,
        };
        let Some(rhs) = after_key.strip_prefix('=') else {
            continue;
        };
        let rhs = rhs.trim();

        // Three RHS shapes: "X.Y.Z", { version = "...", ... }, { workspace = true }
        if let Some(v) = rhs.strip_prefix('"').and_then(|r| r.strip_suffix('"')) {
            return Ok(Some(v.to_string()));
        }
        if let Some(v) = extract_version_from_table(rhs) {
            return Ok(Some(v));
        }
        if rhs.contains("workspace") && rhs.contains("true") {
            // Caller resolves workspace inheritance.
            return Ok(None);
        }
        if rhs.contains("git =") || rhs.contains("git=") {
            return Err(
                "flodl is declared as a git dependency. \
                 fdl add flodl-hf needs a pinnable crates.io version. \
                 Switch to `flodl = \"X.Y.Z\"` first."
                    .into(),
            );
        }
        if rhs.contains("path =") || rhs.contains("path=") {
            // Path-only dep: read version from the referenced Cargo.toml.
            // For MVP, error with guidance.
            return Err(
                "flodl is declared as a path dependency only. \
                 Add an explicit `version = \"X.Y.Z\"` so fdl add can \
                 pin the matching flodl-hf release."
                    .into(),
            );
        }
    }
    Ok(None)
}

/// Extract `version = "X.Y.Z"` from an inline table like
/// `{ version = "0.5.1", features = [...] }`. Returns `None` when the
/// string doesn't look like a table or carries no `version` key.
fn extract_version_from_table(rhs: &str) -> Option<String> {
    let rhs = rhs.strip_prefix('{')?.strip_suffix('}')?;
    for part in rhs.split(',') {
        let part = part.trim();
        let Some(after) = part.strip_prefix("version") else {
            continue;
        };
        let after = after.trim_start();
        let Some(after) = after.strip_prefix('=') else {
            continue;
        };
        let after = after.trim_start();
        let Some(v) = after.strip_prefix('"').and_then(|r| r.strip_suffix('"')) else {
            continue;
        };
        return Some(v.to_string());
    }
    None
}

/// Climb the directory tree looking for a Cargo.toml with a
/// `[workspace]` table. Returns the path when found, else None.
fn find_workspace_root(from: &Path) -> Option<PathBuf> {
    let mut dir = from.parent()?.parent()?.to_path_buf();
    loop {
        let candidate = dir.join("Cargo.toml");
        if candidate.exists() {
            if let Ok(content) = fs::read_to_string(&candidate) {
                if content.lines().any(|l| l.trim() == "[workspace]") {
                    return Some(candidate);
                }
            }
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn substitute_version(template: &str, version: &str) -> String {
    template.replace("{{FLODL_VERSION}}", version)
}

fn write_file(path: &Path, content: &str) -> Result<(), String> {
    fs::write(path, content).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

fn print_next_steps(version: &str, mode: ProjectMode) {
    println!();
    println!(
        "Scaffolded flodl-hf/ playground (flodl {version}, {} mode).",
        match mode {
            ProjectMode::Docker => "Docker",
            ProjectMode::Native => "native",
        },
    );
    println!();
    println!("Next steps:");
    println!("  cd flodl-hf");
    println!("  fdl classify                 # run with the default RoBERTa sentiment checkpoint");
    println!("  fdl classify -- bert-base-uncased   # or any other BERT-family repo id");
    println!();
    println!("See flodl-hf/README.md for feature flavors (offline / vision-only),");
    println!("`.bin` to safetensors conversion for older checkpoints, and how to wire");
    println!("flodl-hf into your main crate when you're ready.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plain_version_string() {
        let c = r#"
[dependencies]
flodl = "0.6.0"
other = "1.0"
"#;
        assert_eq!(parse_flodl_dep(c).unwrap(), Some("0.6.0".into()));
    }

    #[test]
    fn parse_table_version() {
        let c = r#"
[dependencies]
flodl = { version = "0.5.1", features = ["cuda"] }
"#;
        assert_eq!(parse_flodl_dep(c).unwrap(), Some("0.5.1".into()));
    }

    #[test]
    fn parse_workspace_inheritance_returns_none() {
        let c = r#"
[dependencies]
flodl = { workspace = true }
"#;
        // Workspace inheritance returns None; caller climbs to workspace root.
        assert_eq!(parse_flodl_dep(c).unwrap(), None);
    }

    #[test]
    fn parse_git_dep_errors() {
        let c = r#"
[dependencies]
flodl = { git = "https://github.com/fab2s/floDl" }
"#;
        let err = parse_flodl_dep(c).unwrap_err();
        assert!(err.contains("git dependency"), "got: {err}");
    }

    #[test]
    fn parse_no_flodl_returns_none() {
        let c = r#"
[dependencies]
other = "1.0"
"#;
        assert_eq!(parse_flodl_dep(c).unwrap(), None);
    }

    #[test]
    fn parse_ignores_flodl_hf_and_flodl_sys() {
        // `flodl = ...` must match exactly — neighbouring crate names
        // (flodl-hf, flodl-sys) must not false-positive.
        let c = r#"
[dependencies]
flodl-hf = "0.6.0"
flodl-sys = "0.6.0"
"#;
        assert_eq!(parse_flodl_dep(c).unwrap(), None);
    }

    #[test]
    fn parse_ignores_non_dep_tables() {
        let c = r#"
[package]
flodl = "0.6.0"   # not actually a dep; this is bogus but must not match
"#;
        assert_eq!(parse_flodl_dep(c).unwrap(), None);
    }

    #[test]
    fn substitute_version_replaces_all_occurrences() {
        let t = "flodl = \"={{FLODL_VERSION}}\"\nflodl-hf = \"={{FLODL_VERSION}}\"";
        let out = substitute_version(t, "0.6.0");
        assert_eq!(out, "flodl = \"=0.6.0\"\nflodl-hf = \"=0.6.0\"");
    }

    #[test]
    fn render_fdl_yml_docker_preserves_docker_lines() {
        let t = "commands:\n  classify:\n    run: cargo run --release\n    docker: dev\n";
        assert_eq!(render_fdl_yml(t, ProjectMode::Docker), t);
    }

    #[test]
    fn render_fdl_yml_native_strips_docker_lines() {
        let t = "commands:\n  classify:\n    run: cargo run --release\n    docker: dev\n  check:\n    run: cargo check\n    docker: dev\n";
        let out = render_fdl_yml(t, ProjectMode::Native);
        assert!(
            !out.contains("docker: dev"),
            "native output must not contain docker: dev lines: {out}"
        );
        // Non-docker lines stay in place — `cargo run --release` and
        // `cargo check` must both survive.
        assert!(out.contains("cargo run --release"));
        assert!(out.contains("cargo check"));
    }

    #[test]
    fn render_fdl_yml_native_only_strips_exact_docker_line() {
        // Indentation-sensitive: lines like `    docker: hf-parity` or
        // `description: docker: dev stuff` must NOT be stripped.
        let t = "\
commands:
  classify:
    run: cargo run
    docker: dev
  other:
    description: docker: dev isn't a literal directive here
    docker: hf-parity
";
        let out = render_fdl_yml(t, ProjectMode::Native);
        assert!(!out.contains("    docker: dev\n"), "exact match stripped: {out}");
        assert!(out.contains("hf-parity"), "other services preserved: {out}");
        assert!(
            out.contains("docker: dev isn't a literal"),
            "description text preserved: {out}",
        );
    }
}
