//! Pure classifier for the `Path`-kind step of `dispatch_config`.
//!
//! The main loop in `main.rs::dispatch_config` walks an arbitrarily
//! nested `commands:` graph. The `Path` arm has the most branches:
//! descend into a child fdl.yml, render `--help`, refresh the schema
//! cache, or forward the tail to the child's entry. [`classify_path_step`]
//! inspects the load result + tail and returns a [`PathOutcome`] variant
//! describing *which* of those four paths applies. All impure actions
//! (printing help, spawning processes) stay in the caller, so this
//! function is straight-line and unit-testable against tempdir fixtures.

use std::path::{Path, PathBuf};

use crate::config::{self, CommandConfig, CommandSpec};

/// What a single `Path`-kind step resolved to. Every variant holds the
/// loaded `child` config when applicable, so the caller doesn't re-load.
pub enum PathOutcome {
    /// Failed to load the child `fdl.yml`. The string is the
    /// underlying error message.
    LoadFailed(String),
    /// Next tail token is a known sub-command of the child — descend.
    Descend {
        child: Box<CommandConfig>,
        new_dir: PathBuf,
        new_name: String,
    },
    /// Tail carries `--help` / `-h` at this level.
    ShowHelp { child: Box<CommandConfig> },
    /// Tail carries `--refresh-schema`.
    RefreshSchema {
        child: Box<CommandConfig>,
        child_dir: PathBuf,
    },
    /// Forward the tail to the child's entry.
    Exec {
        child: Box<CommandConfig>,
        child_dir: PathBuf,
    },
}

/// Classify a `Path`-kind step. Pure: loads the child config, inspects
/// the tail, and returns the matching [`PathOutcome`]. The caller owns
/// every side effect (printing, spawning).
pub fn classify_path_step(
    spec: &CommandSpec,
    name: &str,
    current_dir: &Path,
    tail: &[String],
    env: Option<&str>,
) -> PathOutcome {
    let child_dir = spec.resolve_path(name, current_dir);
    let child_cfg = match config::load_command_with_env(&child_dir, env) {
        Ok(c) => c,
        Err(e) => return PathOutcome::LoadFailed(e),
    };

    // Descent check runs first: `--help` / `--refresh-schema` apply to
    // the level the user is asking about, not to the parent. If the
    // next token names a nested entry, we descend before reading flags.
    if let Some(next) = tail.first() {
        if child_cfg.commands.contains_key(next) {
            return PathOutcome::Descend {
                child: Box::new(child_cfg),
                new_dir: child_dir,
                new_name: next.clone(),
            };
        }
    }

    if tail.iter().any(|a| a == "--help" || a == "-h") {
        return PathOutcome::ShowHelp {
            child: Box::new(child_cfg),
        };
    }

    if tail.iter().any(|a| a == "--refresh-schema") {
        return PathOutcome::RefreshSchema {
            child: Box::new(child_cfg),
            child_dir,
        };
    }

    PathOutcome::Exec {
        child: Box::new(child_cfg),
        child_dir,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal tempdir helper — avoids pulling in the `tempfile` crate.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let base = std::env::temp_dir();
            let unique = format!(
                "flodl-dispatch-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            );
            let dir = base.join(unique);
            std::fs::create_dir_all(&dir).expect("tempdir creation");
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Write a sub-command fdl.yml at `dir/sub/fdl.yml` with the given body.
    fn mkcmd(base: &Path, sub: &str, body: &str) -> PathBuf {
        let dir = base.join(sub);
        std::fs::create_dir_all(&dir).expect("mkcmd dir");
        std::fs::write(dir.join("fdl.yml"), body).expect("mkcmd write");
        dir
    }

    fn path_spec() -> CommandSpec {
        // Convention-default Path: no fields set, `kind()` returns Path.
        CommandSpec::default()
    }

    #[test]
    fn classify_descends_when_tail_names_nested_command() {
        let tmp = TempDir::new();
        mkcmd(
            tmp.path(),
            "ddp-bench",
            "entry: echo\ncommands:\n  quick:\n    options: { model: linear }\n",
        );
        let spec = path_spec();
        let tail = vec!["quick".to_string()];
        let out = classify_path_step(&spec, "ddp-bench", tmp.path(), &tail, None);
        match out {
            PathOutcome::Descend { new_name, .. } => assert_eq!(new_name, "quick"),
            _ => panic!("expected Descend, got something else"),
        }
    }

    #[test]
    fn classify_show_help_when_tail_has_flag() {
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "sub", "entry: echo\n");
        let spec = path_spec();
        let tail = vec!["--help".to_string()];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::ShowHelp { .. }));
    }

    #[test]
    fn classify_show_help_short_flag() {
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "sub", "entry: echo\n");
        let spec = path_spec();
        let tail = vec!["-h".to_string()];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::ShowHelp { .. }));
    }

    #[test]
    fn classify_refresh_schema() {
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "sub", "entry: echo\n");
        let spec = path_spec();
        let tail = vec!["--refresh-schema".to_string()];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::RefreshSchema { .. }));
    }

    #[test]
    fn classify_exec_when_tail_has_no_known_token() {
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "sub", "entry: echo\n");
        let spec = path_spec();
        let tail = vec!["--model".to_string(), "linear".to_string()];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::Exec { .. }));
    }

    #[test]
    fn classify_exec_when_tail_is_empty() {
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "sub", "entry: echo\n");
        let spec = path_spec();
        let tail: Vec<String> = vec![];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::Exec { .. }));
    }

    #[test]
    fn classify_descend_wins_over_help_at_same_level() {
        // `fdl sub quick --help` must render help for `quick` (handled
        // one level deeper), not for `sub`. Descent wins over help at
        // the current step.
        let tmp = TempDir::new();
        mkcmd(
            tmp.path(),
            "sub",
            "entry: echo\ncommands:\n  quick:\n    options: { x: 1 }\n",
        );
        let spec = path_spec();
        let tail = vec!["quick".to_string(), "--help".to_string()];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::Descend { .. }));
    }

    #[test]
    fn classify_load_failed_when_no_child_fdl_yml() {
        let tmp = TempDir::new();
        let spec = path_spec();
        let tail: Vec<String> = vec![];
        let out = classify_path_step(&spec, "missing", tmp.path(), &tail, None);
        match out {
            PathOutcome::LoadFailed(msg) => assert!(msg.contains("no fdl.yml")),
            _ => panic!("expected LoadFailed, got something else"),
        }
    }

    #[test]
    fn classify_uses_explicit_path() {
        // Explicit `path:` overrides the convention default. Drop the
        // child fdl.yml under `actual/` and point `path:` there.
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "actual", "entry: echo\n");
        let spec = CommandSpec {
            path: Some("actual".into()),
            ..Default::default()
        };
        let tail: Vec<String> = vec![];
        // `name` here is the command's label, not where we load from —
        // `actual/` is the real dir courtesy of `path:`.
        let out = classify_path_step(&spec, "label", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::Exec { .. }));
    }
}
