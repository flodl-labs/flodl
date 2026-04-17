//! fdl.yaml configuration loading and discovery.
//!
//! Walks up from CWD to find the project manifest, parses YAML/JSON,
//! and loads sub-command configs from registered command directories.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ── Root project config ─────────────────────────────────────────────────

/// Root fdl.yaml at project root.
#[derive(Debug, Default, Deserialize)]
pub struct ProjectConfig {
    #[serde(default)]
    pub description: Option<String>,
    /// Commands defined at this level. Each value is a [`CommandSpec`] that
    /// encodes the kind of command (inline `run` script, `path` pointer to
    /// a child fdl.yml, or inline preset reusing the parent entry).
    #[serde(default)]
    pub commands: BTreeMap<String, CommandSpec>,
}

// ── Sub-command config ──────────────────────────────────────────────────

/// Sub-command fdl.yaml (e.g., ddp-bench/fdl.yaml).
///
/// Identical shape to [`ProjectConfig`] but with an executable `entry:`
/// and optional structured config sections (ddp/training/output) that
/// inline preset commands can override.
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
    /// Nested commands — inline presets of this config's entry, standalone
    /// `run` scripts, or `path` pointers to child fdl.yml files.
    #[serde(default)]
    pub commands: BTreeMap<String, CommandSpec>,
    /// Help-only placeholder name for the first-positional slot when
    /// `commands:` holds presets. Defaults to "preset". Pure UX — it
    /// does not affect dispatch (presets are always looked up by name).
    /// Useful to match domain vocabulary, e.g. `arg-name: recipe` or
    /// `arg-name: target`.
    #[serde(default, rename = "arg-name")]
    pub arg_name: Option<String>,
    /// Inline interim schema (before `<entry> --fdl-schema` is implemented).
    /// Drives help rendering, validation, and completions.
    #[serde(default)]
    pub schema: Option<Schema>,
}

// ── Unified command specification ───────────────────────────────────────

/// A command at any nesting level. Three mutually-exclusive kinds are
/// recognised at resolve time:
///
/// - **Path** (`path` set, or by default when the map is empty/null): the
///   command is a pointer to a child `fdl.yml`. By convention the path is
///   `./<command-name>/` when omitted.
/// - **Run** (`run` set): the command is a self-contained shell script
///   that is executed as-is. Optional `docker:` service routes it through
///   `docker compose`.
/// - **Preset**: neither `path` nor `run` is set. The command merges its
///   `ddp` / `training` / `output` / `options` fields over the enclosing
///   `CommandConfig` defaults and invokes that config's `entry:`.
#[derive(Debug, Default, Clone)]
pub struct CommandSpec {
    pub description: Option<String>,
    /// Inline shell command. Mutex with `path`.
    pub run: Option<String>,
    /// Pointer to a child directory containing its own `fdl.yml`. Absolute
    /// or relative to the declaring config's directory. Mutex with `run`.
    /// `None` + no other fields = "use the convention path
    /// `./<command-name>/`".
    pub path: Option<String>,
    /// Docker compose service for `run`-kind commands.
    pub docker: Option<String>,
    /// Preset overrides. Only consulted when neither `run` nor `path` is set.
    pub ddp: Option<DdpConfig>,
    pub training: Option<TrainingConfig>,
    pub output: Option<OutputConfig>,
    pub options: BTreeMap<String, serde_json::Value>,
}

/// What kind of command is this, resolved from a [`CommandSpec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandKind {
    /// `run: "…"` — execute the inline shell command (optionally in Docker).
    Run,
    /// `path: "…"` or convention default — load `<path>/fdl.yml` and
    /// recurse.
    Path,
    /// Neither `run` nor `path`. Merges preset fields onto the enclosing
    /// `CommandConfig` defaults and invokes that config's `entry:`.
    Preset,
}

impl CommandSpec {
    /// Classify this command. Returns an error when both `run` and `path`
    /// are declared — always a mistake, caught loudly rather than silently
    /// picking one.
    pub fn kind(&self) -> Result<CommandKind, String> {
        match (self.run.as_deref(), self.path.as_deref()) {
            (Some(_), Some(_)) => Err(
                "command declares both `run:` and `path:`; \
                 only one is allowed"
                    .to_string(),
            ),
            (Some(_), None) => Ok(CommandKind::Run),
            (None, Some(_)) => Ok(CommandKind::Path),
            (None, None) => {
                // No kind-selecting field. If preset fields are present,
                // treat as Preset; otherwise, fall through to Path (the
                // convention-default: `./<name>/fdl.yml`).
                if self.ddp.is_some()
                    || self.training.is_some()
                    || self.output.is_some()
                    || !self.options.is_empty()
                {
                    Ok(CommandKind::Preset)
                } else {
                    Ok(CommandKind::Path)
                }
            }
        }
    }

    /// Resolve the effective directory for a `Path`-kind command declared
    /// in `parent_dir`. Applies the `./<name>/` convention when `path` is
    /// unset.
    pub fn resolve_path(&self, name: &str, parent_dir: &Path) -> PathBuf {
        match &self.path {
            Some(p) => parent_dir.join(p),
            None => parent_dir.join(name),
        }
    }
}

// Custom Deserialize so that `commands: { name: ~ }` (YAML null) and
// `commands: { name: }` (empty value) both deserialize to a default
// `CommandSpec`. Without this, serde_yaml errors on null because a
// struct expects a map.
impl<'de> Deserialize<'de> for CommandSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Inner {
            #[serde(default)]
            description: Option<String>,
            #[serde(default)]
            run: Option<String>,
            #[serde(default)]
            path: Option<String>,
            #[serde(default)]
            docker: Option<String>,
            #[serde(default)]
            ddp: Option<DdpConfig>,
            #[serde(default)]
            training: Option<TrainingConfig>,
            #[serde(default)]
            output: Option<OutputConfig>,
            #[serde(default)]
            options: BTreeMap<String, serde_json::Value>,
        }

        let raw = serde_yaml::Value::deserialize(deserializer)?;
        if matches!(raw, serde_yaml::Value::Null) {
            return Ok(Self::default());
        }
        let inner: Inner =
            serde_yaml::from_value(raw).map_err(serde::de::Error::custom)?;
        Ok(Self {
            description: inner.description,
            run: inner.run,
            path: inner.path,
            docker: inner.docker,
            ddp: inner.ddp,
            training: inner.training,
            output: inner.output,
            options: inner.options,
        })
    }
}

// ── Schema (interim hand-written, future `<entry> --fdl-schema`) ────────

/// The schema declared inline in a sub-command's fdl.yaml. Maps 1:1 to
/// what `<entry> --fdl-schema` will later emit as JSON.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Schema {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<ArgSpec>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub options: BTreeMap<String, OptionSpec>,
    /// When true, the fdl layer rejects options not declared in the schema.
    /// Consumed once schema-aware argv validation lands (rollout step 3).
    #[serde(default, skip_serializing_if = "is_false")]
    #[allow(dead_code)]
    pub strict: bool,
}

/// A flag option, `--name` / `-x`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OptionSpec {
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choices: Option<Vec<serde_json::Value>>,
    /// Single-letter short alias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub short: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<String>,
    /// Shell snippet producing completion values.
    /// Consumed by `fdl completions <shell>` (follow-up rollout task).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[allow(dead_code)]
    pub completer: Option<String>,
}

/// A positional argument.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ArgSpec {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default = "default_required")]
    pub required: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub variadic: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choices: Option<Vec<serde_json::Value>>,
    /// Shell snippet producing completion values.
    /// Consumed by `fdl completions <shell>` (follow-up rollout task).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[allow(dead_code)]
    pub completer: Option<String>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

fn default_required() -> bool {
    true
}

/// Flags reserved at the fdl level — no sub-command option may shadow them.
/// Kept in sync with main.rs dispatch.
const RESERVED_LONGS: &[&str] = &[
    "help", "version", "quiet", "env",
];
const RESERVED_SHORTS: &[&str] = &[
    "h", "V", "q", "v", "e",
];
const VALID_TYPES: &[&str] = &[
    "string", "int", "float", "bool", "path",
    "list[string]", "list[int]", "list[float]", "list[path]",
];

/// Check a schema for collisions and structural issues.
///
/// Loud-at-load-time: ambiguity caught here is cheaper to fix than mysterious
/// pass-through behavior at runtime.
pub fn validate_schema(schema: &Schema) -> Result<(), String> {
    // Options: check types, shorts, reserved flags.
    let mut short_seen: BTreeMap<String, String> = BTreeMap::new();
    for (long, spec) in &schema.options {
        if !VALID_TYPES.contains(&spec.ty.as_str()) {
            return Err(format!(
                "option --{}: unknown type '{}' (valid: {})",
                long,
                spec.ty,
                VALID_TYPES.join(", ")
            ));
        }
        if RESERVED_LONGS.contains(&long.as_str()) {
            return Err(format!(
                "option --{long} shadows a reserved fdl-level flag"
            ));
        }
        if let Some(s) = &spec.short {
            if s.chars().count() != 1 {
                return Err(format!(
                    "option --{long}: `short: \"{s}\"` must be a single character"
                ));
            }
            if RESERVED_SHORTS.contains(&s.as_str()) {
                return Err(format!(
                    "option --{long}: short -{s} shadows a reserved fdl-level flag"
                ));
            }
            if let Some(prev) = short_seen.insert(s.clone(), long.clone()) {
                return Err(format!(
                    "options --{prev} and --{long} both declare short -{s}"
                ));
            }
        }
    }

    // Args: check types, variadic-only-at-end, no-required-after-optional.
    let mut seen_optional = false;
    let mut name_seen: BTreeMap<String, ()> = BTreeMap::new();
    for (i, arg) in schema.args.iter().enumerate() {
        if !VALID_TYPES.contains(&arg.ty.as_str()) {
            return Err(format!(
                "arg <{}>: unknown type '{}' (valid: {})",
                arg.name,
                arg.ty,
                VALID_TYPES.join(", ")
            ));
        }
        if name_seen.insert(arg.name.clone(), ()).is_some() {
            return Err(format!("duplicate positional name <{}>", arg.name));
        }
        if arg.variadic && i != schema.args.len() - 1 {
            return Err(format!(
                "arg <{}>: variadic positional must be the last one",
                arg.name
            ));
        }
        let is_optional = !arg.required || arg.default.is_some();
        if arg.required && arg.default.is_some() {
            return Err(format!(
                "arg <{}>: `required: true` with a default is a contradiction",
                arg.name
            ));
        }
        if seen_optional && arg.required && arg.default.is_none() {
            return Err(format!(
                "arg <{}>: required positional cannot follow an optional one",
                arg.name
            ));
        }
        if is_optional {
            seen_optional = true;
        }
    }

    Ok(())
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
    load_project_with_env(path, None)
}

/// Load a project config with an optional environment overlay.
///
/// When `env` is `Some`, looks for a sibling `fdl.<env>.{yml,yaml,json}` next
/// to `base_path` and deep-merges it over the base before deserialization.
/// Missing overlay files are a hard error — the user asked for this env, so
/// silently ignoring it would be worse than a clear message.
pub fn load_project_with_env(
    base_path: &Path,
    env: Option<&str>,
) -> Result<ProjectConfig, String> {
    let merged = load_merged_value(base_path, env)?;
    serde_yaml::from_value::<ProjectConfig>(merged)
        .map_err(|e| format!("{}: {}", base_path.display(), e))
}

/// Load the raw merged [`serde_yaml::Value`] for a config + optional env
/// overlay. Exposed so callers like `fdl config show` can inspect the
/// resolved view before it is deserialized into a strongly-typed struct.
pub fn load_merged_value(
    base_path: &Path,
    env: Option<&str>,
) -> Result<serde_yaml::Value, String> {
    let mut layers = Vec::with_capacity(2);
    layers.push(crate::overlay::load_value(base_path)?);
    if let Some(name) = env {
        match crate::overlay::find_env_file(base_path, name) {
            Some(p) => layers.push(crate::overlay::load_value(&p)?),
            None => {
                return Err(format!(
                    "environment `{name}` not found (expected fdl.{name}.yml next to {})",
                    base_path.display()
                ));
            }
        }
    }
    Ok(crate::overlay::merge_layers(layers))
}

/// Source path list for a base config + env overlay, in merge order. Used
/// by `fdl config show` to annotate which layer a value came from.
pub fn config_layer_sources(base_path: &Path, env: Option<&str>) -> Vec<PathBuf> {
    let mut out = vec![base_path.to_path_buf()];
    if let Some(name) = env {
        if let Some(p) = crate::overlay::find_env_file(base_path, name) {
            out.push(p);
        }
    }
    out
}

/// Load a command config from a sub-directory.
///
/// Applies the same `.example`/`.dist` fallback as [`find_config`]. If a
/// `schema:` block is present, validates it before returning.
pub fn load_command(dir: &Path) -> Result<CommandConfig, String> {
    load_command_with_env(dir, None)
}

/// Load a sub-command config with an optional environment overlay.
///
/// Applies the same `.example`/`.dist` fallback as [`find_config`] to locate
/// the base file, then deep-merges a sibling `fdl.<env>.yml` overlay if one
/// exists. A *missing* overlay is silently accepted here (different from
/// [`load_project_with_env`]) — envs declared at the project root don't
/// have to exist for every sub-command.
pub fn load_command_with_env(dir: &Path, env: Option<&str>) -> Result<CommandConfig, String> {
    // Resolve the base config path (with .example fallback, same as before).
    let mut base_path: Option<PathBuf> = None;
    for name in CONFIG_NAMES {
        let path = dir.join(name);
        if path.is_file() {
            base_path = Some(path);
            break;
        }
    }
    if base_path.is_none() {
        for name in CONFIG_NAMES {
            for suffix in EXAMPLE_SUFFIXES {
                let example = dir.join(format!("{name}{suffix}"));
                if example.is_file() {
                    let target = dir.join(name);
                    let src = if try_copy_example(&example, &target) {
                        target
                    } else {
                        example
                    };
                    base_path = Some(src);
                    break;
                }
            }
            if base_path.is_some() {
                break;
            }
        }
    }
    let base_path = base_path
        .ok_or_else(|| format!("no fdl.yml found in {}", dir.display()))?;

    // Layered load: base + optional sibling env overlay.
    let mut layers = vec![crate::overlay::load_value(&base_path)?];
    if let Some(name) = env {
        if let Some(p) = crate::overlay::find_env_file(&base_path, name) {
            layers.push(crate::overlay::load_value(&p)?);
        }
    }
    let merged = crate::overlay::merge_layers(layers);
    let mut cfg: CommandConfig = serde_yaml::from_value(merged)
        .map_err(|e| format!("{}: {}", base_path.display(), e))?;

    if let Some(schema) = &cfg.schema {
        validate_schema(schema)
            .map_err(|e| format!("schema error in {}/fdl.yml: {e}", dir.display()))?;
    }

    // Cache precedence: a valid, fresh cached schema (written by `fdl <cmd>
    // --refresh-schema`) wins over the inline YAML schema. This lets a
    // binary become the source of truth for its own surface once it opts
    // into the `--fdl-schema` contract. A cache that is older than the
    // command's fdl.yml is treated as stale and skipped — the inline
    // schema (if any) reasserts until the user refreshes.
    let cmd_name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("_");
    let cache = crate::schema_cache::cache_path(dir, cmd_name);
    // Reference mtimes: config files that, when edited, might invalidate
    // the cached schema (e.g. changing `entry:` to point somewhere else).
    let refs: Vec<std::path::PathBuf> = CONFIG_NAMES
        .iter()
        .map(|n| dir.join(n))
        .filter(|p| p.exists())
        .collect();
    if !crate::schema_cache::is_stale(&cache, &refs) {
        if let Some(cached) = crate::schema_cache::read_cache(&cache) {
            cfg.schema = Some(cached);
        }
    }

    Ok(cfg)
}

// ── Merge ───────────────────────────────────────────────────────────────

/// Merge the enclosing `CommandConfig` defaults with a named preset's
/// overrides. Preset values win. Used when dispatching an inline preset
/// command (neither `run` nor `path`).
pub fn merge_preset(root: &CommandConfig, preset: &CommandSpec) -> ResolvedConfig {
    ResolvedConfig {
        ddp: merge_ddp(&root.ddp, &preset.ddp),
        training: merge_training(&root.training, &preset.training),
        output: merge_output(&root.output, &preset.output),
        options: preset.options.clone(),
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

    fn opt(ty: &str) -> OptionSpec {
        OptionSpec {
            ty: ty.into(),
            description: None,
            default: None,
            choices: None,
            short: None,
            env: None,
            completer: None,
        }
    }

    fn arg(name: &str, ty: &str) -> ArgSpec {
        ArgSpec {
            name: name.into(),
            ty: ty.into(),
            description: None,
            required: true,
            variadic: false,
            default: None,
            choices: None,
            completer: None,
        }
    }

    #[test]
    fn validate_schema_accepts_minimal_valid() {
        let mut s = Schema::default();
        s.options.insert("model".into(), opt("string"));
        s.options.insert("epochs".into(), opt("int"));
        s.args.push(arg("run-id", "string"));
        validate_schema(&s).expect("minimal valid schema must pass");
    }

    #[test]
    fn validate_schema_rejects_unknown_option_type() {
        let mut s = Schema::default();
        s.options.insert("bad".into(), opt("integer"));
        let err = validate_schema(&s).expect_err("unknown type should fail");
        assert!(err.contains("unknown type"), "err was: {err}");
    }

    #[test]
    fn validate_schema_rejects_reserved_long() {
        let mut s = Schema::default();
        s.options.insert("help".into(), opt("bool"));
        let err = validate_schema(&s).expect_err("reserved --help must fail");
        assert!(err.contains("reserved"), "err was: {err}");
    }

    #[test]
    fn validate_schema_rejects_reserved_short() {
        let mut s = Schema::default();
        let mut o = opt("string");
        o.short = Some("h".into());
        s.options.insert("host".into(), o);
        let err = validate_schema(&s).expect_err("short -h must fail");
        assert!(err.contains("reserved"), "err was: {err}");
    }

    #[test]
    fn validate_schema_rejects_duplicate_short() {
        let mut s = Schema::default();
        let mut a = opt("string");
        a.short = Some("m".into());
        let mut b = opt("string");
        b.short = Some("m".into());
        s.options.insert("model".into(), a);
        s.options.insert("mode".into(), b);
        let err = validate_schema(&s).expect_err("duplicate -m must fail");
        assert!(err.contains("both declare short"), "err was: {err}");
    }

    #[test]
    fn validate_schema_rejects_non_last_variadic() {
        let mut s = Schema::default();
        let mut first = arg("files", "string");
        first.variadic = true;
        s.args.push(first);
        s.args.push(arg("trailer", "string"));
        let err = validate_schema(&s).expect_err("variadic-not-last must fail");
        assert!(err.contains("variadic"), "err was: {err}");
    }

    #[test]
    fn validate_schema_rejects_required_after_optional() {
        let mut s = Schema::default();
        let mut first = arg("maybe", "string");
        first.required = false;
        s.args.push(first);
        s.args.push(arg("need", "string"));
        let err = validate_schema(&s).expect_err("required-after-optional must fail");
        assert!(err.contains("cannot follow"), "err was: {err}");
    }

    #[test]
    fn validate_schema_rejects_required_with_default() {
        let mut s = Schema::default();
        let mut a = arg("x", "string");
        a.default = Some(serde_json::json!("foo"));
        s.args.push(a);
        let err = validate_schema(&s).expect_err("required+default must fail");
        assert!(err.contains("contradiction"), "err was: {err}");
    }

    /// Regression guard: fdl.yml.example must keep a working `doc` command.
    /// The fdl.doc pipeline (api-ref for the port skill, rustdoc warning
    /// enforcement in CI) depends on this entry existing and producing output.
    #[test]
    fn fdl_yml_example_has_doc_script() {
        let cfg = load_example();
        let doc = cfg.commands.get("doc").unwrap_or_else(|| {
            panic!(
                "fdl.yml.example is missing a `doc` command; the rustdoc pipeline \
                 depends on `fdl doc` being defined"
            )
        });
        let cmd = doc
            .run
            .as_deref()
            .expect("fdl.yml.example `doc` command must be a `run:` entry");
        assert!(
            !cmd.trim().is_empty(),
            "fdl.yml.example `doc` command has an empty `run:` command"
        );
        assert!(
            cmd.contains("cargo doc"),
            "fdl.yml.example `doc` command must invoke `cargo doc`, got: {cmd}"
        );
        // Must assert some output was produced -- otherwise rustdoc can
        // silently succeed without writing anything useful (e.g. when the
        // target crate fails to resolve). Keeping the exact check liberal:
        // any mention of target/doc as a produced artifact counts.
        assert!(
            cmd.contains("target/doc"),
            "fdl.yml.example `doc` command must verify output was produced \
             (expected a `test -f target/doc/...` check), got: {cmd}"
        );
    }

    #[test]
    fn command_spec_kind_mutex_run_and_path() {
        let spec = CommandSpec {
            run: Some("echo".into()),
            path: Some("x/".into()),
            ..Default::default()
        };
        let err = spec.kind().expect_err("run + path must fail");
        assert!(err.contains("both"), "err was: {err}");
    }

    #[test]
    fn command_spec_kind_path_convention() {
        let spec = CommandSpec::default();
        assert_eq!(spec.kind().unwrap(), CommandKind::Path);
    }

    #[test]
    fn command_spec_kind_preset_when_preset_fields_set() {
        let spec = CommandSpec {
            training: Some(TrainingConfig {
                epochs: Some(1),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(spec.kind().unwrap(), CommandKind::Preset);
    }

    #[test]
    fn command_spec_deserialize_from_null() {
        let yaml = "cmd: ~";
        let map: BTreeMap<String, CommandSpec> =
            serde_yaml::from_str(yaml).expect("null must deserialize to default");
        let spec = map.get("cmd").expect("cmd missing");
        assert!(spec.run.is_none() && spec.path.is_none());
        assert_eq!(spec.kind().unwrap(), CommandKind::Path);
    }

    #[test]
    fn command_config_arg_name_deserializes_kebab_case() {
        // YAML uses `arg-name:`, Rust field is `arg_name`.
        let yaml = "arg-name: recipe\nentry: echo\n";
        let cfg: CommandConfig =
            serde_yaml::from_str(yaml).expect("arg-name must parse");
        assert_eq!(cfg.arg_name.as_deref(), Some("recipe"));
    }

    #[test]
    fn command_config_arg_name_defaults_to_none() {
        let cfg: CommandConfig =
            serde_yaml::from_str("entry: echo\n").expect("minimal cfg must parse");
        assert!(cfg.arg_name.is_none());
    }
}
