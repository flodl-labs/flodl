//! fdl.yaml configuration loading and discovery.
//!
//! Walks up from CWD to find the project manifest, parses YAML/JSON,
//! and loads sub-command configs from registered command directories.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

// ── Root project config ─────────────────────────────────────────────────

/// Root fdl.yaml at project root.
#[derive(Debug, Default, Deserialize)]
pub struct ProjectConfig {
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub scripts: BTreeMap<String, Script>,
    #[serde(default)]
    pub commands: Vec<String>,
}

/// Script: either a bare string or {description, run, docker}.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Script {
    Short(String),
    Long {
        #[serde(default)]
        description: Option<String>,
        run: String,
        #[serde(default)]
        docker: Option<String>,
    },
}

impl Script {
    pub fn command(&self) -> &str {
        match self {
            Script::Short(s) => s,
            Script::Long { run, .. } => run,
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Script::Short(s) => s.as_str(),
            Script::Long {
                description: Some(d),
                ..
            } => d.as_str(),
            Script::Long { run, .. } => run.as_str(),
        }
    }

    pub fn docker_service(&self) -> Option<&str> {
        match self {
            Script::Short(_) => None,
            Script::Long { docker, .. } => docker.as_deref(),
        }
    }
}

// ── Sub-command config ──────────────────────────────────────────────────

/// Sub-command fdl.yaml (e.g., ddp-bench/fdl.yaml).
#[derive(Debug, Default, Deserialize)]
pub struct CommandConfig {
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub entry: Option<String>,
    /// Docker compose service name. When set, entry is wrapped in
    /// `docker compose run --rm <service> bash -c "cd <workdir> && <entry> <args>"`.
    #[serde(default)]
    pub docker: Option<String>,
    #[serde(default)]
    pub ddp: Option<DdpConfig>,
    #[serde(default)]
    pub training: Option<TrainingConfig>,
    #[serde(default)]
    pub output: Option<OutputConfig>,
    #[serde(default)]
    pub jobs: BTreeMap<String, Job>,
}

// ── Structured config sections ──────────────────────────────────────────

/// DDP configuration. Maps 1:1 to flodl DdpConfig / DdpRunConfig.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DdpConfig {
    pub mode: Option<String>,
    pub policy: Option<String>,
    pub backend: Option<String>,
    /// "auto" or integer.
    pub anchor: Option<serde_json::Value>,
    pub max_anchor: Option<u32>,
    pub overhead_target: Option<f64>,
    pub divergence_threshold: Option<f64>,
    /// null (unlimited) or integer.
    pub max_batch_diff: Option<serde_json::Value>,
    pub speed_hint: Option<SpeedHint>,
    pub partition_ratios: Option<Vec<f64>>,
    /// "auto" or bool.
    pub progressive: Option<serde_json::Value>,
    pub max_grad_norm: Option<f64>,
    pub lr_scale_ratio: Option<f64>,
    pub snapshot_timeout: Option<u32>,
    pub checkpoint_every: Option<u32>,
    pub timeline: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SpeedHint {
    pub slow_rank: usize,
    pub ratio: f64,
}

/// Training scalars.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TrainingConfig {
    pub epochs: Option<u32>,
    pub batch_size: Option<u32>,
    pub batches_per_epoch: Option<u32>,
    pub lr: Option<f64>,
    pub seed: Option<u64>,
}

/// Output settings.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OutputConfig {
    pub dir: Option<String>,
    pub timeline: Option<bool>,
    pub monitor: Option<u16>,
}

/// A named job (preset) within a sub-command.
#[derive(Debug, Default, Deserialize)]
pub struct Job {
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub ddp: Option<DdpConfig>,
    #[serde(default)]
    pub training: Option<TrainingConfig>,
    #[serde(default)]
    pub output: Option<OutputConfig>,
    #[serde(default)]
    pub options: BTreeMap<String, serde_json::Value>,
}

// ── Config discovery ────────────────────────────────────────────────────

const CONFIG_NAMES: &[&str] = &["fdl.yaml", "fdl.yml", "fdl.json"];
const EXAMPLE_SUFFIXES: &[&str] = &[".example", ".dist"];

/// Walk up from `start` looking for fdl.yaml.
///
/// If only an `.example` (or `.dist`) variant exists, offers to copy it
/// to the real config path. This lets the repo commit `fdl.yaml.example`
/// while `.gitignore`-ing `fdl.yaml` so users can customize locally.
pub fn find_config(start: &Path) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        // First pass: look for the real config.
        for name in CONFIG_NAMES {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        // Second pass: look for .example/.dist variants.
        for name in CONFIG_NAMES {
            for suffix in EXAMPLE_SUFFIXES {
                let example = dir.join(format!("{name}{suffix}"));
                if example.is_file() {
                    let target = dir.join(name);
                    if try_copy_example(&example, &target) {
                        return Some(target);
                    }
                    // User declined: use the example directly.
                    return Some(example);
                }
            }
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Prompt the user to copy an example config to the real path.
/// Returns true if the copy succeeded.
fn try_copy_example(example: &Path, target: &Path) -> bool {
    let example_name = example.file_name().unwrap_or_default().to_string_lossy();
    let target_name = target.file_name().unwrap_or_default().to_string_lossy();
    eprintln!(
        "fdl: found {example_name} but no {target_name}. \
         Copy it to create your local config? [Y/n] "
    );
    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return false;
    }
    let answer = input.trim().to_lowercase();
    if answer.is_empty() || answer == "y" || answer == "yes" {
        match std::fs::copy(example, target) {
            Ok(_) => {
                eprintln!("fdl: created {target_name} (edit to customize)");
                true
            }
            Err(e) => {
                eprintln!("fdl: failed to copy: {e}");
                false
            }
        }
    } else {
        false
    }
}

/// Load a project config from a specific path.
pub fn load_project(path: &Path) -> Result<ProjectConfig, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    parse(&content, path)
}

/// Load a command config from a sub-directory.
///
/// Applies the same `.example`/`.dist` fallback as [`find_config`].
pub fn load_command(dir: &Path) -> Result<CommandConfig, String> {
    for name in CONFIG_NAMES {
        let path = dir.join(name);
        if path.is_file() {
            let content = std::fs::read_to_string(&path)
                .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
            return parse(&content, &path);
        }
    }
    // Fallback: try .example/.dist variants.
    for name in CONFIG_NAMES {
        for suffix in EXAMPLE_SUFFIXES {
            let example = dir.join(format!("{name}{suffix}"));
            if example.is_file() {
                let target = dir.join(name);
                if try_copy_example(&example, &target) {
                    let content = std::fs::read_to_string(&target)
                        .map_err(|e| format!("cannot read {}: {}", target.display(), e))?;
                    return parse(&content, &target);
                }
                // User declined: load example directly.
                let content = std::fs::read_to_string(&example)
                    .map_err(|e| format!("cannot read {}: {}", example.display(), e))?;
                return parse(&content, &example);
            }
        }
    }
    Err(format!("no fdl.yml found in {}", dir.display()))
}

/// Resolve a command path string (e.g. "ddp-bench/") to its short name.
pub fn command_name(path: &str) -> &str {
    let trimmed = path.trim_end_matches('/');
    Path::new(trimmed)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(trimmed)
}

fn parse<T: serde::de::DeserializeOwned>(content: &str, path: &Path) -> Result<T, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("yaml");
    match ext {
        "json" => {
            serde_json::from_str(content).map_err(|e| format!("{}: {}", path.display(), e))
        }
        _ => serde_yaml::from_str(content).map_err(|e| format!("{}: {}", path.display(), e)),
    }
}

// ── Merge ───────────────────────────────────────────────────────────────

/// Merge root defaults with per-job overrides. Job values win.
pub fn merge_job(root: &CommandConfig, job: &Job) -> ResolvedConfig {
    ResolvedConfig {
        ddp: merge_ddp(&root.ddp, &job.ddp),
        training: merge_training(&root.training, &job.training),
        output: merge_output(&root.output, &job.output),
        options: job.options.clone(),
    }
}

/// Resolved config from root defaults only (no job).
pub fn defaults_only(root: &CommandConfig) -> ResolvedConfig {
    ResolvedConfig {
        ddp: root.ddp.clone().unwrap_or_default(),
        training: root.training.clone().unwrap_or_default(),
        output: root.output.clone().unwrap_or_default(),
        options: BTreeMap::new(),
    }
}

/// Fully resolved configuration ready for arg translation.
pub struct ResolvedConfig {
    pub ddp: DdpConfig,
    pub training: TrainingConfig,
    pub output: OutputConfig,
    pub options: BTreeMap<String, serde_json::Value>,
}

macro_rules! merge_field {
    ($base:expr, $over:expr, $field:ident) => {
        $over
            .as_ref()
            .and_then(|o| o.$field.clone())
            .or_else(|| $base.as_ref().and_then(|b| b.$field.clone()))
    };
}

fn merge_ddp(base: &Option<DdpConfig>, over: &Option<DdpConfig>) -> DdpConfig {
    DdpConfig {
        mode: merge_field!(base, over, mode),
        policy: merge_field!(base, over, policy),
        backend: merge_field!(base, over, backend),
        anchor: merge_field!(base, over, anchor),
        max_anchor: merge_field!(base, over, max_anchor),
        overhead_target: merge_field!(base, over, overhead_target),
        divergence_threshold: merge_field!(base, over, divergence_threshold),
        max_batch_diff: merge_field!(base, over, max_batch_diff),
        speed_hint: merge_field!(base, over, speed_hint),
        partition_ratios: merge_field!(base, over, partition_ratios),
        progressive: merge_field!(base, over, progressive),
        max_grad_norm: merge_field!(base, over, max_grad_norm),
        lr_scale_ratio: merge_field!(base, over, lr_scale_ratio),
        snapshot_timeout: merge_field!(base, over, snapshot_timeout),
        checkpoint_every: merge_field!(base, over, checkpoint_every),
        timeline: merge_field!(base, over, timeline),
    }
}

fn merge_training(base: &Option<TrainingConfig>, over: &Option<TrainingConfig>) -> TrainingConfig {
    TrainingConfig {
        epochs: merge_field!(base, over, epochs),
        batch_size: merge_field!(base, over, batch_size),
        batches_per_epoch: merge_field!(base, over, batches_per_epoch),
        lr: merge_field!(base, over, lr),
        seed: merge_field!(base, over, seed),
    }
}

fn merge_output(base: &Option<OutputConfig>, over: &Option<OutputConfig>) -> OutputConfig {
    OutputConfig {
        dir: merge_field!(base, over, dir),
        timeline: merge_field!(base, over, timeline),
        monitor: merge_field!(base, over, monitor),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolve the project root (where fdl.yml / fdl.yml.example live) starting
    /// from CARGO_MANIFEST_DIR. The CLI crate sits one level down.
    fn project_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("flodl-cli parent must be project root")
            .to_path_buf()
    }

    fn load_example() -> ProjectConfig {
        let path = project_root().join("fdl.yml.example");
        assert!(
            path.is_file(),
            "fdl.yml.example missing at {} -- the CLI depends on it as the canonical config template",
            path.display()
        );
        load_project(&path).expect("fdl.yml.example must parse as a valid ProjectConfig")
    }

    /// Regression guard: fdl.yml.example must keep a working `doc` script.
    /// The fdl.doc pipeline (api-ref for the port skill, rustdoc warning
    /// enforcement in CI) depends on this entry existing and producing output.
    #[test]
    fn fdl_yml_example_has_doc_script() {
        let cfg = load_example();
        let doc = cfg.scripts.get("doc").unwrap_or_else(|| {
            panic!("fdl.yml.example is missing a `doc` script; the rustdoc pipeline \
                    depends on `fdl doc` being defined")
        });
        let cmd = doc.command();
        assert!(!cmd.trim().is_empty(),
            "fdl.yml.example `doc` script has an empty `run:` command");
        assert!(cmd.contains("cargo doc"),
            "fdl.yml.example `doc` script must invoke `cargo doc`, got: {cmd}");
        // Must assert some output was produced -- otherwise rustdoc can
        // silently succeed without writing anything useful (e.g. when the
        // target crate fails to resolve). Keeping the exact check liberal:
        // any mention of target/doc as a produced artifact counts.
        assert!(
            cmd.contains("target/doc"),
            "fdl.yml.example `doc` script must verify output was produced \
             (expected a `test -f target/doc/...` check), got: {cmd}"
        );
    }
}
