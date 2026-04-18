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
    /// picking one. Also rejects `docker:` without `run:`: the docker
    /// service wraps the inline run-script, so pairing it with a `path:`
    /// pointer or a preset entry is always silent-noop territory.
    pub fn kind(&self) -> Result<CommandKind, String> {
        if self.docker.is_some() && self.run.is_none() {
            return Err(
                "command declares `docker:` without `run:`; \
                 `docker:` only wraps inline run-scripts"
                    .to_string(),
            );
        }
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
    /// When true, the fdl layer rejects options not declared in the
    /// schema before the sub-command's entry ever runs. Two validation
    /// points:
    ///
    /// 1. *Load time* — preset `options:` maps are checked against the
    ///    enclosing `schema.options` (see [`validate_presets_strict`]).
    ///    A typo like `options: { batchsize: 32 }` when the schema
    ///    declares `batch-size` is a loud load error.
    /// 2. *Dispatch time* — the user's extra argv tail is tokenized
    ///    against the schema (see [`validate_tail`]). Unknown flags
    ///    error out with a "did you mean" suggestion instead of being
    ///    silently forwarded.
    ///
    /// **Validation NOT gated by `strict`** — always-on for declared
    /// items, so positive assertions from the schema always hold:
    /// - `choices:` on options: the user's value and any preset YAML
    ///   value must be in the list.
    /// - `choices:` on positional args: the user's value must be in
    ///   the list (when strict is off, this may mis-fire if unknown
    ///   flags push orphan values into positional slots — opt into
    ///   strict for clean positional handling).
    ///
    /// `strict` is purely about **unknown** options/args, not about
    /// validating declared contracts.
    #[serde(default, skip_serializing_if = "is_false")]
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
    let layers = resolve_config_layers(base_path, env)?;
    Ok(crate::overlay::merge_layers(
        layers.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
    ))
}

/// Resolve every layer contributing to a config, in merge order, with
/// `inherit-from:` chains expanded. Paired with the base file + optional
/// env overlay, the result is `[chain(base)..., chain(env_overlay)...]`
/// de-duplicated by canonical path (kept-first).
///
/// Used by `fdl config show` for per-leaf source annotation, and
/// internally by [`load_merged_value`] / [`load_command_with_env`] so
/// every consumer picks up `inherit-from:` uniformly.
pub fn resolve_config_layers(
    base_path: &Path,
    env: Option<&str>,
) -> Result<Vec<(PathBuf, serde_yaml::Value)>, String> {
    let mut layers = crate::overlay::resolve_chain(base_path)?;
    if let Some(name) = env {
        match crate::overlay::find_env_file(base_path, name) {
            Some(p) => {
                let env_chain = crate::overlay::resolve_chain(&p)?;
                layers.extend(env_chain);
            }
            None => {
                return Err(format!(
                    "environment `{name}` not found (expected fdl.{name}.yml next to {})",
                    base_path.display()
                ));
            }
        }
    }
    // Dedup by canonical path, keeping first occurrence. An env overlay
    // whose chain loops back to a file already in the base chain (same
    // file via a different inheritance route) collapses cleanly.
    let mut seen = std::collections::HashSet::new();
    layers.retain(|(path, _)| seen.insert(path.clone()));
    Ok(layers)
}

/// Source path list for a base config + env overlay, in merge order. Used
/// by `fdl config show` to annotate which layer a value came from.
pub fn config_layer_sources(base_path: &Path, env: Option<&str>) -> Vec<PathBuf> {
    resolve_config_layers(base_path, env)
        .map(|ls| ls.into_iter().map(|(p, _)| p).collect())
        .unwrap_or_else(|_| vec![base_path.to_path_buf()])
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

    // Layered load: base chain + optional env overlay chain. Both sides
    // run through `resolve_chain` so `inherit-from:` composes the same
    // way for nested commands as for the project root.
    let mut layers = crate::overlay::resolve_chain(&base_path)?;
    if let Some(name) = env {
        if let Some(p) = crate::overlay::find_env_file(&base_path, name) {
            layers.extend(crate::overlay::resolve_chain(&p)?);
        }
    }
    let mut seen = std::collections::HashSet::new();
    layers.retain(|(path, _)| seen.insert(path.clone()));
    let merged = crate::overlay::merge_layers(
        layers.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
    );
    let mut cfg: CommandConfig = serde_yaml::from_value(merged)
        .map_err(|e| format!("{}: {}", base_path.display(), e))?;

    if let Some(schema) = &cfg.schema {
        validate_schema(schema)
            .map_err(|e| format!("schema error in {}/fdl.yml: {e}", dir.display()))?;
        // Preset validation (choice values + strict unknown-key rejection)
        // is intentionally deferred to the exec path. Load-time validation
        // would block `fdl <cmd> --help` whenever ANY preset in the config
        // has a typo — worse UX than letting help render and erroring only
        // when the broken preset is actually invoked.
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

// ── Strict-mode validation ──────────────────────────────────────────────

/// Reserved flags that strict mode always tolerates in the user's tail.
/// These are fdl-level universals (help/version) or opt-ins every
/// FdlArgs-derived binary exposes (--fdl-schema) — keeping them out of
/// the `schema.options` map means strict mode has to allowlist them
/// separately or spuriously reject legal invocations.
const STRICT_UNIVERSAL_LONGS: &[(&str, Option<char>, bool)] = &[
    // (long, short, takes_value)
    ("help", Some('h'), false),
    ("version", Some('V'), false),
    ("fdl-schema", None, false),
    ("refresh-schema", None, false),
];

/// Convert a [`Schema`] into an [`ArgsSpec`](crate::args::parser::ArgsSpec) suitable for strict-mode
/// tail validation. Positional `required` flags are intentionally
/// dropped: the binary itself will enforce them after parsing, and
/// treating them as required here would turn "missing positional" into
/// a double-errored mess.
pub fn schema_to_args_spec(schema: &Schema) -> crate::args::parser::ArgsSpec {
    use crate::args::parser::{ArgsSpec, OptionDecl, PositionalDecl};

    let mut options: Vec<OptionDecl> = schema
        .options
        .iter()
        .map(|(long, spec)| OptionDecl {
            long: long.clone(),
            short: spec
                .short
                .as_deref()
                .and_then(|s| s.chars().next()),
            takes_value: spec.ty != "bool",
            // Every value-taking option is allowed to appear bare in
            // strict mode. fdl does not second-guess whether the binary
            // would accept a bare `--foo`; that stays in the binary's
            // court.
            allows_bare: true,
            repeatable: spec.ty.starts_with("list["),
            choices: spec
                .choices
                .as_ref()
                .map(|cs| strict_choices_to_strings(cs)),
        })
        .collect();

    // Always-allowed universals — help/version/fdl-schema/refresh-schema
    // are not in the user's schema but must not trigger "unknown flag".
    for (long, short, takes_value) in STRICT_UNIVERSAL_LONGS {
        options.push(OptionDecl {
            long: (*long).to_string(),
            short: *short,
            takes_value: *takes_value,
            allows_bare: true,
            repeatable: false,
            choices: None,
        });
    }

    // Positionals: drop the `required` bit. Strict mode is scoped to
    // option names/values only; arity is the binary's concern.
    let positionals: Vec<PositionalDecl> = schema
        .args
        .iter()
        .map(|a| PositionalDecl {
            name: a.name.clone(),
            required: false,
            variadic: a.variadic,
            choices: a
                .choices
                .as_ref()
                .map(|cs| strict_choices_to_strings(cs)),
        })
        .collect();

    ArgsSpec {
        options,
        positionals,
        // Non-strict schemas accept user-forwarded flags the author
        // didn't declare — the binary re-parses the tail anyway.
        // Strict schemas reject anything not declared.
        lenient_unknowns: !schema.strict,
    }
}

fn strict_choices_to_strings(cs: &[serde_json::Value]) -> Vec<String> {
    cs.iter()
        .map(|v| match v {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .collect()
}

/// Validate the user's extra argv tail against a schema. Always called
/// before `run::exec_command` — the parser's lenient-unknowns mode is
/// keyed off `schema.strict` so choice validation on declared flags
/// fires regardless, while unknown-flag rejection stays opt-in.
///
/// The tokenizer from [`crate::args::parser`] is reused so "did you
/// mean" suggestions, cluster, and equals handling come for free.
pub fn validate_tail(tail: &[String], schema: &Schema) -> Result<(), String> {
    let spec = schema_to_args_spec(schema);
    let mut argv = Vec::with_capacity(tail.len() + 1);
    argv.push("fdl".to_string());
    argv.extend(tail.iter().cloned());
    crate::args::parser::parse(&spec, &argv).map(|_| ())
}

/// Validate a single preset that's about to be invoked. Combines the
/// always-on `choices:` check and, if `schema.strict`, the unknown-key
/// rejection — scoped to just this preset, not the whole `commands:`
/// map. Called from the exec path so typos in a sibling preset don't
/// block `--help` for a correct one.
pub fn validate_preset_for_exec(
    preset_name: &str,
    spec: &CommandSpec,
    schema: &Schema,
) -> Result<(), String> {
    for (key, value) in &spec.options {
        let Some(opt) = schema.options.get(key) else {
            if schema.strict {
                return Err(format!(
                    "preset `{preset_name}` pins option `{key}` which is not declared in schema.options"
                ));
            }
            continue;
        };
        let Some(choices) = &opt.choices else {
            continue;
        };
        if !choices.iter().any(|c| values_equal(c, value)) {
            let allowed: Vec<String> = choices
                .iter()
                .map(|c| match c {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .collect();
            return Err(format!(
                "preset `{preset_name}` sets option `{key}` to `{}` -- allowed: {}",
                display_json(value),
                allowed.join(", "),
            ));
        }
    }
    Ok(())
}

/// Always-on: validate preset YAML `options:` values against declared
/// `choices:` in the schema. An option YAML value whose key matches a
/// declared option with a `choices:` list must be one of those choices.
/// Keys not declared in the schema are ignored here — those are the
/// concern of [`validate_presets_strict`] (opt-in).
///
/// Used for whole-map validation (e.g. from a future `fdl config lint`
/// subcommand). The dispatch path uses [`validate_preset_for_exec`] so
/// sibling-preset typos don't block correct invocations.
pub fn validate_preset_values(
    commands: &BTreeMap<String, CommandSpec>,
    schema: &Schema,
) -> Result<(), String> {
    for (preset_name, spec) in commands {
        match spec.kind() {
            Ok(CommandKind::Preset) => {}
            _ => continue,
        }
        for (key, value) in &spec.options {
            let Some(opt) = schema.options.get(key) else {
                continue; // unknown key — strict's problem, not ours
            };
            let Some(choices) = &opt.choices else {
                continue; // no choices declared — anything goes
            };
            if !choices.iter().any(|c| values_equal(c, value)) {
                let allowed: Vec<String> = choices
                    .iter()
                    .map(|c| match c {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .collect();
                return Err(format!(
                    "preset `{preset_name}` sets option `{key}` to `{}` -- allowed: {}",
                    display_json(value),
                    allowed.join(", "),
                ));
            }
        }
    }
    Ok(())
}

/// Compare two JSON values for equality, treating YAML's loose-typed
/// representation (a preset might write `batch-size: 32` as an int
/// while the schema's choices list contains `"32"` as a string).
fn values_equal(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    if a == b {
        return true;
    }
    // Cross-type string ↔ number comparison for YAML-friendly matching.
    match (a, b) {
        (serde_json::Value::String(s), other) | (other, serde_json::Value::String(s)) => {
            s == &other.to_string()
        }
        _ => false,
    }
}

fn display_json(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// At load time, reject preset `options:` keys that are not declared in
/// the enclosing schema. Runs only when `schema.strict == true`, and
/// only against entries resolved to [`CommandKind::Preset`] — `run:` and
/// `path:` kinds don't share the parent schema.
pub fn validate_presets_strict(
    commands: &BTreeMap<String, CommandSpec>,
    schema: &Schema,
) -> Result<(), String> {
    for (preset_name, spec) in commands {
        match spec.kind() {
            Ok(CommandKind::Preset) => {}
            _ => continue,
        }
        for key in spec.options.keys() {
            if !schema.options.contains_key(key) {
                return Err(format!(
                    "preset `{preset_name}` pins option `{key}` which is not declared in schema.options"
                ));
            }
        }
    }
    Ok(())
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

    // ── Tail validation (always-on) + strict unknown-rejection ─────

    fn schema_with_model_option(strict: bool) -> Schema {
        let mut s = Schema {
            strict,
            ..Schema::default()
        };
        let mut model = opt("string");
        model.short = Some("m".into());
        model.choices = Some(vec![
            serde_json::json!("mlp"),
            serde_json::json!("resnet"),
        ]);
        s.options.insert("model".into(), model);
        s.options.insert("epochs".into(), opt("int"));
        // A bool flag, no value.
        s.options.insert("validate".into(), opt("bool"));
        s
    }

    fn strict_schema_with_model_option() -> Schema {
        schema_with_model_option(true)
    }

    #[test]
    fn validate_tail_accepts_known_long_flag() {
        let schema = strict_schema_with_model_option();
        let tail = vec!["--epochs".into(), "3".into()];
        validate_tail(&tail, &schema).expect("known flag must pass");
    }

    #[test]
    fn validate_tail_accepts_known_short_flag() {
        let schema = strict_schema_with_model_option();
        let tail = vec!["-m".into(), "mlp".into()];
        validate_tail(&tail, &schema).expect("known short must pass");
    }

    #[test]
    fn validate_tail_accepts_bool_flag() {
        let schema = strict_schema_with_model_option();
        let tail = vec!["--validate".into()];
        validate_tail(&tail, &schema).expect("bool flag must pass");
    }

    #[test]
    fn validate_tail_strict_rejects_unknown_long_flag() {
        let schema = strict_schema_with_model_option();
        let tail = vec!["--nope".into()];
        let err = validate_tail(&tail, &schema)
            .expect_err("unknown long flag must error in strict mode");
        assert!(err.contains("--nope"), "err was: {err}");
    }

    #[test]
    fn validate_tail_strict_suggests_did_you_mean() {
        // "--epoch" is one char off "--epochs" — edit distance ≤ 2.
        let schema = strict_schema_with_model_option();
        let tail = vec!["--epoch".into(), "3".into()];
        let err = validate_tail(&tail, &schema).expect_err("typo must error");
        assert!(err.contains("did you mean"), "err was: {err}");
        assert!(err.contains("--epochs"), "suggestion missing: {err}");
    }

    #[test]
    fn validate_tail_strict_rejects_unknown_short_flag() {
        let schema = strict_schema_with_model_option();
        let tail = vec!["-z".into()];
        let err = validate_tail(&tail, &schema)
            .expect_err("unknown short must error in strict mode");
        assert!(err.contains("-z"), "err was: {err}");
    }

    #[test]
    fn validate_tail_rejects_bad_choice_always_strict() {
        let schema = strict_schema_with_model_option();
        let tail = vec!["--model".into(), "lenet".into()];
        let err = validate_tail(&tail, &schema)
            .expect_err("out-of-set choice must error");
        assert!(err.contains("lenet"), "err was: {err}");
        assert!(err.contains("allowed"), "err should list allowed values: {err}");
    }

    #[test]
    fn validate_tail_rejects_bad_choice_even_when_not_strict() {
        // The main change in this rollout: `choices:` is a positive
        // assertion by the author, so it must be enforced regardless
        // of `schema.strict`. Only *unknown* flags relax without
        // strict.
        let schema = schema_with_model_option(false);
        let tail = vec!["--model".into(), "lenet".into()];
        let err = validate_tail(&tail, &schema)
            .expect_err("out-of-set choice must error without strict");
        assert!(err.contains("lenet"), "err was: {err}");
        assert!(err.contains("allowed"), "err should list allowed values: {err}");
    }

    #[test]
    fn validate_tail_non_strict_tolerates_unknown_flag() {
        // Without strict, unknown flags are legitimate pass-through
        // candidates (the binary handles them itself).
        let schema = schema_with_model_option(false);
        let tail = vec!["--fancy-passthrough".into(), "value".into()];
        validate_tail(&tail, &schema)
            .expect("unknown flag must be tolerated when strict is off");
    }

    #[test]
    fn validate_tail_non_strict_still_checks_known_short_choices() {
        // The declared short `-m` has choices; a bad value fails even
        // when strict is off. Unknown options would be tolerated, but
        // once the user reaches a declared option, its contract holds.
        let schema = schema_with_model_option(false);
        let tail = vec!["-m".into(), "lenet".into()];
        let err = validate_tail(&tail, &schema)
            .expect_err("out-of-set choice via short must error");
        assert!(err.contains("lenet"), "err was: {err}");
    }

    #[test]
    fn validate_tail_allows_reserved_help() {
        // Reserved universal flags must pass even though they are not
        // declared in the schema. Defense-in-depth against edge cases
        // where `--help` somehow reaches dispatch.
        let schema = strict_schema_with_model_option();
        let tail = vec!["--help".into()];
        validate_tail(&tail, &schema).expect("--help must be allowed");
    }

    #[test]
    fn validate_tail_allows_reserved_fdl_schema() {
        // `fdl ddp-bench --fdl-schema` is forwarded to the binary.
        let schema = strict_schema_with_model_option();
        let tail = vec!["--fdl-schema".into()];
        validate_tail(&tail, &schema).expect("--fdl-schema must be allowed");
    }

    #[test]
    fn validate_tail_passthrough_after_double_dash() {
        // `--` terminates flag parsing. Tokens after it are positionals
        // and must never trigger "unknown flag" errors.
        let schema = strict_schema_with_model_option();
        let tail = vec!["--".into(), "--arbitrary".into(), "anything".into()];
        validate_tail(&tail, &schema).expect("passthrough must work");
    }

    #[test]
    fn validate_presets_strict_rejects_unknown_option() {
        let schema = strict_schema_with_model_option();
        let mut commands = BTreeMap::new();
        let mut bad_options = BTreeMap::new();
        bad_options.insert("batchsize".into(), serde_json::json!(32));
        commands.insert(
            "quick".into(),
            CommandSpec {
                options: bad_options,
                ..Default::default()
            },
        );
        let err = validate_presets_strict(&commands, &schema)
            .expect_err("preset pinning undeclared option must error");
        assert!(err.contains("quick"), "err should name the preset: {err}");
        assert!(err.contains("batchsize"), "err should name the key: {err}");
    }

    #[test]
    fn validate_presets_strict_accepts_known_options() {
        let schema = strict_schema_with_model_option();
        let mut commands = BTreeMap::new();
        let mut good_options = BTreeMap::new();
        good_options.insert("model".into(), serde_json::json!("mlp"));
        good_options.insert("epochs".into(), serde_json::json!(5));
        commands.insert(
            "quick".into(),
            CommandSpec {
                options: good_options,
                ..Default::default()
            },
        );
        validate_presets_strict(&commands, &schema)
            .expect("presets with declared options must pass");
    }

    #[test]
    fn validate_presets_strict_ignores_run_and_path_kinds() {
        // Only Preset-kind entries share the parent schema. Run/Path
        // siblings are independent, so strict must not touch them.
        let schema = strict_schema_with_model_option();
        let mut commands = BTreeMap::new();
        commands.insert(
            "helper".into(),
            CommandSpec {
                run: Some("echo hi".into()),
                ..Default::default()
            },
        );
        commands.insert(
            "nested".into(),
            CommandSpec {
                path: Some("./nested/".into()),
                ..Default::default()
            },
        );
        validate_presets_strict(&commands, &schema)
            .expect("run/path siblings must be ignored by preset strict check");
    }

    // ── Preset value validation (always-on `choices:`) ──────────────

    #[test]
    fn validate_preset_values_rejects_bad_choice_even_without_strict() {
        // Schema has `choices:` on model; a preset pinning model to
        // something outside the list must fail at load, strict or not.
        let schema = schema_with_model_option(false);
        let mut commands = BTreeMap::new();
        let mut opts = BTreeMap::new();
        opts.insert("model".into(), serde_json::json!("lenet"));
        commands.insert(
            "quick".into(),
            CommandSpec {
                options: opts,
                ..Default::default()
            },
        );
        let err = validate_preset_values(&commands, &schema)
            .expect_err("out-of-choices preset must error");
        assert!(err.contains("quick"), "preset name missing: {err}");
        assert!(err.contains("model"), "option name missing: {err}");
        assert!(err.contains("lenet"), "bad value missing: {err}");
        assert!(err.contains("allowed"), "allowed list missing: {err}");
    }

    #[test]
    fn validate_preset_values_accepts_in_choices_preset() {
        let schema = schema_with_model_option(false);
        let mut commands = BTreeMap::new();
        let mut opts = BTreeMap::new();
        opts.insert("model".into(), serde_json::json!("mlp"));
        commands.insert(
            "quick".into(),
            CommandSpec {
                options: opts,
                ..Default::default()
            },
        );
        validate_preset_values(&commands, &schema)
            .expect("in-choices preset must pass");
    }

    #[test]
    fn validate_preset_values_ignores_undeclared_keys() {
        // Unknown keys aren't our concern here — that's for
        // `validate_presets_strict`, which only runs under strict.
        let schema = schema_with_model_option(false);
        let mut commands = BTreeMap::new();
        let mut opts = BTreeMap::new();
        opts.insert("extra".into(), serde_json::json!("whatever"));
        commands.insert(
            "quick".into(),
            CommandSpec {
                options: opts,
                ..Default::default()
            },
        );
        validate_preset_values(&commands, &schema)
            .expect("undeclared key must be ignored by value validator");
    }

    #[test]
    fn validate_preset_values_ignores_options_without_choices() {
        // `epochs` is declared as int with no `choices:`, so any value
        // passes the choice check (type validation is a separate pass).
        let schema = schema_with_model_option(false);
        let mut commands = BTreeMap::new();
        let mut opts = BTreeMap::new();
        opts.insert("epochs".into(), serde_json::json!(999));
        commands.insert(
            "quick".into(),
            CommandSpec {
                options: opts,
                ..Default::default()
            },
        );
        validate_preset_values(&commands, &schema)
            .expect("no-choices option must accept any value");
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
    fn command_spec_kind_preset_when_only_options_set() {
        // `options:` alone is enough to make a preset — not every preset
        // overrides the structured ddp/training/output blocks.
        let mut options = BTreeMap::new();
        options.insert("model".into(), serde_json::json!("linear"));
        let spec = CommandSpec {
            options,
            ..Default::default()
        };
        assert_eq!(spec.kind().unwrap(), CommandKind::Preset);
    }

    #[test]
    fn command_spec_kind_path_explicit() {
        // Explicit `path:` is a Path even if preset fields are also set;
        // the presence of `path:` is the kind-selecting field.
        let spec = CommandSpec {
            path: Some("./sub/".into()),
            ..Default::default()
        };
        assert_eq!(spec.kind().unwrap(), CommandKind::Path);
    }

    #[test]
    fn command_spec_kind_rejects_docker_without_run() {
        // `docker:` is meaningful only as a wrapper around an inline
        // `run:` script. Pairing it with path/preset is a silent noop
        // at dispatch time, so we reject at load.
        let spec = CommandSpec {
            docker: Some("cuda".into()),
            ..Default::default()
        };
        let err = spec
            .kind()
            .expect_err("docker without run must fail");
        assert!(err.contains("docker"), "err was: {err}");
    }

    #[test]
    fn command_spec_kind_allows_docker_with_run() {
        let spec = CommandSpec {
            run: Some("cargo test".into()),
            docker: Some("dev".into()),
            ..Default::default()
        };
        assert_eq!(spec.kind().unwrap(), CommandKind::Run);
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

    // ── resolve_config_layers: inherit-from + env composition ────────────
    //
    // Integration coverage for how `inherit-from:` chains compose with env
    // overlays at the config-module boundary. The overlay module already
    // tests `resolve_chain` in isolation; here we verify the concat+dedup
    // behaviour that config.rs layers on top.

    /// Minimal tempdir helper — matches the pattern used across the crate.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "fdl-cfg-test-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn filenames(layers: &[(PathBuf, serde_yaml::Value)]) -> Vec<String> {
        layers
            .iter()
            .map(|(p, _)| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?")
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn resolve_config_layers_base_only() {
        let tmp = TempDir::new();
        let base = tmp.0.join("fdl.yml");
        std::fs::write(&base, "a: 1\n").unwrap();
        let layers = resolve_config_layers(&base, None).unwrap();
        assert_eq!(filenames(&layers), vec!["fdl.yml"]);
    }

    #[test]
    fn resolve_config_layers_base_with_env_overlay() {
        let tmp = TempDir::new();
        let base = tmp.0.join("fdl.yml");
        let env = tmp.0.join("fdl.ci.yml");
        std::fs::write(&base, "a: 1\n").unwrap();
        std::fs::write(&env, "b: 2\n").unwrap();
        let layers = resolve_config_layers(&base, Some("ci")).unwrap();
        assert_eq!(filenames(&layers), vec!["fdl.yml", "fdl.ci.yml"]);
    }

    #[test]
    fn resolve_config_layers_env_inherits_from_mixin() {
        // fdl.ci.yml inherits from fdl.cloud.yml (standalone mix-in, not
        // derived from base). Combined chain: [base, cloud, ci].
        let tmp = TempDir::new();
        let base = tmp.0.join("fdl.yml");
        let cloud = tmp.0.join("fdl.cloud.yml");
        let ci = tmp.0.join("fdl.ci.yml");
        std::fs::write(&base, "a: 1\n").unwrap();
        std::fs::write(&cloud, "b: 2\n").unwrap();
        std::fs::write(&ci, "inherit-from: fdl.cloud.yml\nc: 3\n").unwrap();
        let layers = resolve_config_layers(&base, Some("ci")).unwrap();
        assert_eq!(
            filenames(&layers),
            vec!["fdl.yml", "fdl.cloud.yml", "fdl.ci.yml"]
        );
    }

    #[test]
    fn resolve_config_layers_dedups_when_env_inherits_from_base() {
        // fdl.ci.yml inherits from fdl.yml directly. Base is already in
        // the layer list, so env's chain collapses into it — the final
        // list must not have fdl.yml twice.
        let tmp = TempDir::new();
        let base = tmp.0.join("fdl.yml");
        let ci = tmp.0.join("fdl.ci.yml");
        std::fs::write(&base, "a: 1\n").unwrap();
        std::fs::write(&ci, "inherit-from: fdl.yml\nb: 2\n").unwrap();
        let layers = resolve_config_layers(&base, Some("ci")).unwrap();
        assert_eq!(filenames(&layers), vec!["fdl.yml", "fdl.ci.yml"]);
    }

    #[test]
    fn resolve_config_layers_merged_value_matches_chain() {
        // End-to-end: the merge result should reflect the chain order
        // (base < cloud < ci), with each subsequent layer overriding.
        let tmp = TempDir::new();
        let base = tmp.0.join("fdl.yml");
        let cloud = tmp.0.join("fdl.cloud.yml");
        let ci = tmp.0.join("fdl.ci.yml");
        std::fs::write(&base, "value: base\nkeep_base: yes\n").unwrap();
        std::fs::write(&cloud, "value: cloud\nkeep_cloud: yes\n").unwrap();
        std::fs::write(
            &ci,
            "inherit-from: fdl.cloud.yml\nvalue: ci\nkeep_ci: yes\n",
        )
        .unwrap();
        let merged = load_merged_value(&base, Some("ci")).unwrap();
        let m = merged.as_mapping().unwrap();
        // Last writer wins on `value`.
        assert_eq!(
            m.get(serde_yaml::Value::String("value".into())).unwrap(),
            &serde_yaml::Value::String("ci".into())
        );
        // Each layer's unique key survives.
        assert!(m.contains_key(serde_yaml::Value::String("keep_base".into())));
        assert!(m.contains_key(serde_yaml::Value::String("keep_cloud".into())));
        assert!(m.contains_key(serde_yaml::Value::String("keep_ci".into())));
    }

    #[test]
    fn resolve_config_layers_missing_env_errors() {
        let tmp = TempDir::new();
        let base = tmp.0.join("fdl.yml");
        std::fs::write(&base, "a: 1\n").unwrap();
        let err = resolve_config_layers(&base, Some("nope")).unwrap_err();
        assert!(err.contains("nope"));
        assert!(err.contains("not found"));
    }

    #[test]
    fn resolve_config_layers_base_inherit_from_chain() {
        // Base itself uses inherit-from: shared-defaults.yml. The
        // defaults live in a sibling file and are merged UNDER the base.
        let tmp = TempDir::new();
        let defaults = tmp.0.join("shared.yml");
        let base = tmp.0.join("fdl.yml");
        std::fs::write(&defaults, "policy: default\n").unwrap();
        std::fs::write(&base, "inherit-from: shared.yml\npolicy: override\n").unwrap();
        let layers = resolve_config_layers(&base, None).unwrap();
        assert_eq!(filenames(&layers), vec!["shared.yml", "fdl.yml"]);
    }
}
