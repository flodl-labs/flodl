//! Shell completion generation and one-shot setup.
//!
//! `fdl completions bash`  outputs a bash completion script to stdout.
//! `fdl completions zsh`   outputs a zsh  completion script to stdout.
//! `fdl completions fish`  outputs a fish completion script to stdout.
//! `fdl autocomplete`      detects shell, installs completions, reloads.
//!
//! When a sub-command declares a `schema:` block in its fdl.yaml, completion
//! is option-aware: `choices:` drive value lists, `type: path` drives file
//! completion, `completer:` (opt-in) lets authors ship arbitrary shell
//! snippets (e.g. `ls runs/`).

use std::path::Path;

use crate::config::{self, CommandConfig, OptionSpec, ProjectConfig};
use crate::style;

/// Built-in commands that are always available.
/// `help` is intentionally omitted: use `--help`/`-h` instead.
const BUILTINS: &[&str] = &[
    "setup",
    "libtorch",
    "diagnose",
    "init",
    "install",
    "skill",
    "api-ref",
    "completions",
    "autocomplete",
    "version",
];

/// Hard-coded libtorch sub-commands (not schema-driven yet).
const LIBTORCH_SUBS: &[&str] = &[
    "download", "build", "list", "activate", "remove", "info",
];

/// Reserved top-level flags, always offered at every position.
const TOP_FLAGS: &[&str] = &[
    "--help", "-h", "--version", "-V",
    "-v", "-vv", "-vvv", "--quiet", "-q",
];

/// The marker comment we use to detect our block in rc files.
const MARKER: &str = "# fdl shell completions";

// ── Intermediate model ──────────────────────────────────────────────────

/// Everything a completion emitter needs, shell-agnostic.
struct CompletionData {
    /// Top-level words: builtins + scripts + command names.
    top_level: Vec<String>,
    /// Per-command info for sub-commands declared in the root fdl.yaml.
    commands: Vec<CommandData>,
}

struct CommandData {
    name: String,
    jobs: Vec<String>,
    options: Vec<OptionCompletion>,
}

struct OptionCompletion {
    long: String,
    short: Option<String>,
    takes_value: bool,
    value: ValueKind,
    description: Option<String>,
}

enum ValueKind {
    /// bool flag, no value expected.
    None,
    /// Fixed list of choices.
    Choices(Vec<String>),
    /// File path completion.
    Path,
    /// Custom shell snippet (emitted verbatim).
    Completer(String),
    /// Free-form value, no hints.
    Any,
}

impl CompletionData {
    fn from_project(project: Option<(&ProjectConfig, &Path)>) -> Self {
        let mut top_level: Vec<String> =
            BUILTINS.iter().map(|s| s.to_string()).collect();
        let mut commands = Vec::new();

        if let Some((proj, root)) = project {
            for name in proj.scripts.keys() {
                top_level.push(name.clone());
            }
            for cmd_path in &proj.commands {
                let short = config::command_name(cmd_path).to_string();
                top_level.push(short.clone());

                let cmd_dir = root.join(cmd_path);
                if let Ok(cmd_config) = config::load_command(&cmd_dir) {
                    commands.push(CommandData::from_config(short, &cmd_config));
                } else {
                    commands.push(CommandData {
                        name: short,
                        jobs: Vec::new(),
                        options: Vec::new(),
                    });
                }
            }
        }

        Self {
            top_level,
            commands,
        }
    }
}

impl CommandData {
    fn from_config(name: String, cfg: &CommandConfig) -> Self {
        let jobs: Vec<String> = cfg.jobs.keys().cloned().collect();
        let options = cfg
            .schema
            .as_ref()
            .map(|s| {
                s.options
                    .iter()
                    .map(|(long, spec)| OptionCompletion::from_spec(long, spec))
                    .collect()
            })
            .unwrap_or_default();
        Self {
            name,
            jobs,
            options,
        }
    }
}

impl OptionCompletion {
    fn from_spec(long: &str, spec: &OptionSpec) -> Self {
        let takes_value = spec.ty != "bool";
        let value = if !takes_value {
            ValueKind::None
        } else if let Some(choices) = &spec.choices {
            ValueKind::Choices(choices.iter().map(value_as_str).collect())
        } else if let Some(c) = &spec.completer {
            ValueKind::Completer(c.clone())
        } else if spec.ty == "path" || spec.ty == "list[path]" {
            ValueKind::Path
        } else {
            ValueKind::Any
        };
        Self {
            long: long.to_string(),
            short: spec.short.clone(),
            takes_value,
            value,
            description: spec.description.clone(),
        }
    }

    fn flag_tokens(&self) -> Vec<String> {
        let mut out = vec![format!("--{}", self.long)];
        if let Some(s) = &self.short {
            out.push(format!("-{s}"));
        }
        out
    }
}

fn value_as_str(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

// ── Public API ──────────────────────────────────────────────────────────

/// Generate a completion script for the given shell.
pub fn generate(shell: &str, project: Option<(&ProjectConfig, &Path)>) {
    let data = CompletionData::from_project(project);
    match shell {
        "bash" => print!("{}", emit_bash(&data)),
        "zsh" => print!("{}", emit_zsh(&data)),
        "fish" => print!("{}", emit_fish(&data)),
        other => {
            eprintln!("unsupported shell: {other}");
            eprintln!("supported: bash, zsh, fish");
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  eval \"$(fdl completions bash)\"   # bash");
            eprintln!("  eval \"$(fdl completions zsh)\"    # zsh");
            eprintln!("  fdl completions fish | source     # fish");
            eprintln!("  fdl autocomplete                  # auto-detect and install");
        }
    }
}

/// One-shot setup: detect shell, write to rc file, reload.
pub fn autocomplete(project: Option<(&ProjectConfig, &Path)>) {
    let shell = detect_shell();
    let (rc_path, shell_name) = match shell.as_str() {
        "fish" => {
            let home = std::env::var("HOME").unwrap_or_default();
            // Fish uses per-command files in ~/.config/fish/completions/.
            // A single rc-file block keeps the install story uniform here;
            // users who prefer the native approach can redirect manually.
            (format!("{home}/.config/fish/config.fish"), "fish")
        }
        "zsh" => {
            let home = std::env::var("HOME").unwrap_or_default();
            (format!("{home}/.zshrc"), "zsh")
        }
        _ => {
            let home = std::env::var("HOME").unwrap_or_default();
            (format!("{home}/.bashrc"), "bash")
        }
    };

    eprintln!(
        "{}  Detected shell: {}",
        style::green("*"),
        style::bold(shell_name)
    );
    eprintln!(
        "{}  Target: {}",
        style::green("*"),
        style::bold(&rc_path)
    );

    // Check if already installed.
    if let Ok(content) = std::fs::read_to_string(&rc_path) {
        if content.contains(MARKER) {
            eprintln!(
                "{}  Completions already installed. Updating...",
                style::yellow("*")
            );
            let cleaned = remove_completion_block(&content);
            if let Err(e) = std::fs::write(&rc_path, cleaned) {
                eprintln!("error: cannot write {rc_path}: {e}");
                return;
            }
        }
    }

    // Ensure target dir exists (needed for fish's config.fish).
    if let Some(parent) = std::path::Path::new(&rc_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let script = match shell_name {
        "fish" => "eval (fdl completions fish | string collect)\n".to_string(),
        "zsh" => "eval \"$(fdl completions zsh)\"\n".to_string(),
        _ => "eval \"$(fdl completions bash)\"\n".to_string(),
    };
    let _ = project; // project is only needed by generate(); install shells re-run fdl at load time

    let block = format!("\n{MARKER}\n{script}{MARKER} end\n");
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&rc_path)
    {
        Ok(mut file) => {
            use std::io::Write;
            if let Err(e) = file.write_all(block.as_bytes()) {
                eprintln!("error: cannot append to {rc_path}: {e}");
                return;
            }
        }
        Err(e) => {
            eprintln!("error: cannot open {rc_path}: {e}");
            return;
        }
    }

    eprintln!(
        "{}  Completions installed.",
        style::green("*")
    );
    eprintln!();
    eprintln!(
        "  Reload with: {}",
        style::dim(&match shell_name {
            "fish" => format!("source {rc_path}"),
            _ => format!("source {rc_path}"),
        })
    );
    eprintln!("  Or restart your terminal for completions to take effect.");
}

fn detect_shell() -> String {
    if let Ok(shell) = std::env::var("SHELL") {
        if shell.contains("fish") {
            return "fish".into();
        }
        if shell.contains("zsh") {
            return "zsh".into();
        }
        if shell.contains("bash") {
            return "bash".into();
        }
    }
    "bash".into()
}

fn remove_completion_block(content: &str) -> String {
    let marker_end = format!("{MARKER} end");
    let mut result = String::new();
    let mut skipping = false;
    for line in content.lines() {
        if line.trim() == MARKER {
            skipping = true;
            continue;
        }
        if line.trim() == marker_end {
            skipping = false;
            continue;
        }
        if !skipping {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

// ── Bash ────────────────────────────────────────────────────────────────

fn emit_bash(data: &CompletionData) -> String {
    let mut s = String::new();
    s.push_str("# fdl bash completion (generated)\n");
    s.push_str("# eval \"$(fdl completions bash)\"\n");
    s.push_str("_fdl_completions() {\n");
    s.push_str("    local cur prev cmd\n");
    s.push_str("    cur=\"${COMP_WORDS[COMP_CWORD]}\"\n");
    s.push_str("    prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n");
    s.push_str("    cmd=\"${COMP_WORDS[1]}\"\n");
    s.push('\n');

    // Position 1: top-level.
    let top = join_for_shell(&data.top_level);
    let top_with_flags = format!(
        "{top} {}",
        TOP_FLAGS.join(" ")
    );
    s.push_str("    if [[ $COMP_CWORD -eq 1 ]]; then\n");
    s.push_str(&format!(
        "        COMPREPLY=($(compgen -W \"{top_with_flags}\" -- \"$cur\"))\n"
    ));
    s.push_str("        return\n");
    s.push_str("    fi\n");

    // Sub-commands with schema / jobs.
    for cmd in &data.commands {
        s.push_str(&format!("\n    if [[ \"$cmd\" == \"{name}\" ]]; then\n", name = cmd.name));

        // Value completion for options that take a value.
        s.push_str("        case \"$prev\" in\n");
        for opt in &cmd.options {
            if !opt.takes_value {
                continue;
            }
            let flags = opt.flag_tokens().join("|");
            let line = match &opt.value {
                ValueKind::Choices(cs) => format!(
                    "            {flags}) COMPREPLY=($(compgen -W \"{}\" -- \"$cur\")); return ;;\n",
                    cs.join(" ")
                ),
                ValueKind::Path => format!(
                    "            {flags}) COMPREPLY=($(compgen -f -- \"$cur\")); return ;;\n",
                ),
                ValueKind::Completer(c) => format!(
                    "            {flags}) COMPREPLY=($(compgen -W \"$({c})\" -- \"$cur\")); return ;;\n",
                ),
                ValueKind::Any => format!(
                    "            {flags}) return ;;\n",
                ),
                ValueKind::None => continue,
            };
            s.push_str(&line);
        }
        s.push_str("        esac\n");

        // At position 2, offer jobs + option flags.
        // Beyond position 2, offer option flags only (prev-value already handled).
        let option_flags: Vec<String> = cmd
            .options
            .iter()
            .flat_map(|o| o.flag_tokens())
            .collect();
        let cmd_flags_str = {
            let mut v = option_flags.clone();
            v.push("--help".into());
            v.push("-h".into());
            v.join(" ")
        };
        let jobs_str = cmd.jobs.join(" ");

        s.push_str("        if [[ $COMP_CWORD -eq 2 ]]; then\n");
        if cmd.jobs.is_empty() {
            s.push_str(&format!(
                "            COMPREPLY=($(compgen -W \"{cmd_flags_str}\" -- \"$cur\"))\n"
            ));
        } else {
            s.push_str(&format!(
                "            COMPREPLY=($(compgen -W \"{jobs_str} {cmd_flags_str}\" -- \"$cur\"))\n"
            ));
        }
        s.push_str("            return\n");
        s.push_str("        fi\n");

        s.push_str(&format!(
            "        COMPREPLY=($(compgen -W \"{cmd_flags_str}\" -- \"$cur\"))\n"
        ));
        s.push_str("        return\n");
        s.push_str("    fi\n");
    }

    // libtorch sub-commands.
    s.push_str("\n    if [[ \"$cmd\" == \"libtorch\" && $COMP_CWORD -eq 2 ]]; then\n");
    s.push_str(&format!(
        "        COMPREPLY=($(compgen -W \"{}\" -- \"$cur\"))\n",
        LIBTORCH_SUBS.join(" ")
    ));
    s.push_str("        return\n");
    s.push_str("    fi\n");

    s.push_str("}\n");
    s.push_str("complete -F _fdl_completions fdl\n");
    s
}

fn join_for_shell(v: &[String]) -> String {
    v.join(" ")
}

// ── Zsh ─────────────────────────────────────────────────────────────────

fn emit_zsh(data: &CompletionData) -> String {
    let mut s = String::new();
    s.push_str("#compdef fdl\n");
    s.push_str("# fdl zsh completion (generated)\n");
    s.push_str("# eval \"$(fdl completions zsh)\"\n");
    s.push_str("_fdl() {\n");
    s.push_str("    local -a commands\n");
    let top_level_with_flags = {
        let mut v: Vec<String> = data.top_level.clone();
        for f in TOP_FLAGS {
            v.push((*f).to_string());
        }
        v
    };
    s.push_str(&format!(
        "    commands=({})\n",
        top_level_with_flags.join(" ")
    ));
    s.push('\n');

    s.push_str("    if (( CURRENT == 2 )); then\n");
    s.push_str("        _describe 'command' commands\n");
    s.push_str("        return\n");
    s.push_str("    fi\n");

    s.push_str("\n    case $words[2] in\n");
    for cmd in &data.commands {
        s.push_str(&format!("        {name})\n", name = cmd.name));

        // Value completion when the previous word is a flag with choices/path/completer.
        if cmd.options.iter().any(|o| o.takes_value) {
            s.push_str("            case $words[CURRENT-1] in\n");
            for opt in &cmd.options {
                if !opt.takes_value {
                    continue;
                }
                let flags = opt.flag_tokens().join("|");
                let body = match &opt.value {
                    ValueKind::Choices(cs) => format!(
                        "                {flags}) _values 'value' {}; return ;;\n",
                        cs.join(" ")
                    ),
                    ValueKind::Path => format!(
                        "                {flags}) _files; return ;;\n",
                    ),
                    ValueKind::Completer(c) => format!(
                        "                {flags}) local -a vals; vals=(${{(f)\"$({c})\"}}); _describe 'value' vals; return ;;\n",
                    ),
                    ValueKind::Any => format!(
                        "                {flags}) return ;;\n",
                    ),
                    ValueKind::None => continue,
                };
                s.push_str(&body);
            }
            s.push_str("            esac\n");
        }

        // Position 3 (after the command name): offer jobs + options.
        let option_flags: Vec<String> = cmd
            .options
            .iter()
            .flat_map(|o| o.flag_tokens())
            .collect();
        let mut all_flags = option_flags.clone();
        all_flags.push("--help".into());
        all_flags.push("-h".into());
        let flags_joined = all_flags.join(" ");

        if !cmd.jobs.is_empty() {
            s.push_str("            if (( CURRENT == 3 )); then\n");
            s.push_str("                local -a jobs\n");
            s.push_str(&format!(
                "                jobs=({})\n",
                cmd.jobs.join(" ")
            ));
            s.push_str("                _describe 'job' jobs\n");
            s.push_str("            fi\n");
        }
        s.push_str(&format!(
            "            _values 'option' {flags_joined}\n"
        ));
        s.push_str("            ;;\n");
    }

    // libtorch sub-commands.
    s.push_str("        libtorch)\n");
    s.push_str("            if (( CURRENT == 3 )); then\n");
    s.push_str("                local -a subcmds\n");
    s.push_str(&format!(
        "                subcmds=({})\n",
        LIBTORCH_SUBS.join(" ")
    ));
    s.push_str("                _describe 'subcommand' subcmds\n");
    s.push_str("            fi\n");
    s.push_str("            ;;\n");
    s.push_str("    esac\n");

    s.push_str("}\n");
    s.push_str("_fdl\n");
    s
}

// ── Fish ────────────────────────────────────────────────────────────────

fn emit_fish(data: &CompletionData) -> String {
    let mut s = String::new();
    s.push_str("# fdl fish completion (generated)\n");
    s.push_str("# fdl completions fish | source\n");
    s.push_str("complete -c fdl -f\n\n");

    // Top-level commands available when no sub-command chosen yet.
    s.push_str("# Top-level commands\n");
    for word in &data.top_level {
        s.push_str(&format!(
            "complete -c fdl -n '__fish_use_subcommand' -a '{word}'\n"
        ));
    }
    for flag in TOP_FLAGS {
        if let Some(long) = flag.strip_prefix("--") {
            s.push_str(&format!(
                "complete -c fdl -n '__fish_use_subcommand' -l '{long}'\n"
            ));
        } else if let Some(short) = flag.strip_prefix('-') {
            s.push_str(&format!(
                "complete -c fdl -n '__fish_use_subcommand' -s '{short}'\n"
            ));
        }
    }
    s.push('\n');

    // Sub-command-specific completions.
    for cmd in &data.commands {
        s.push_str(&format!("# {name}\n", name = cmd.name));
        let cond = format!("__fish_seen_subcommand_from {}", cmd.name);

        for job in &cmd.jobs {
            s.push_str(&format!(
                "complete -c fdl -n '{cond}' -a '{job}' -d 'job'\n"
            ));
        }

        for opt in &cmd.options {
            let long = &opt.long;

            // Base flag declaration.
            let mut line = format!("complete -c fdl -n '{cond}' -l '{long}'");
            if let Some(short_flag) = &opt.short {
                line.push_str(&format!(" -s '{short_flag}'"));
            }

            // Value completion.
            match &opt.value {
                ValueKind::None => {}
                ValueKind::Choices(cs) => {
                    line.push_str(" -r -f -a '");
                    line.push_str(&cs.join(" "));
                    line.push('\'');
                }
                ValueKind::Path => {
                    line.push_str(" -r -F");
                }
                ValueKind::Completer(c) => {
                    line.push_str(&format!(" -r -f -a '({c})'"));
                }
                ValueKind::Any => {
                    line.push_str(" -r");
                }
            }

            // Fish shows -d descriptions inline in the completion menu.
            if let Some(desc) = &opt.description {
                let safe = desc.replace('\'', "\\'");
                line.push_str(&format!(" -d '{safe}'"));
            }
            line.push('\n');
            s.push_str(&line);
        }
        s.push('\n');
    }

    // libtorch sub-commands.
    s.push_str("# libtorch\n");
    for sub in LIBTORCH_SUBS {
        s.push_str(&format!(
            "complete -c fdl -n '__fish_seen_subcommand_from libtorch' -a '{sub}'\n"
        ));
    }

    s
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ArgSpec, OptionSpec, Schema};
    use std::collections::BTreeMap;

    fn make_cmd_with_schema() -> CommandData {
        let mut options = BTreeMap::new();
        options.insert(
            "model".into(),
            OptionSpec {
                ty: "string".into(),
                description: None,
                default: None,
                choices: Some(vec![
                    serde_json::json!("mlp"),
                    serde_json::json!("resnet"),
                ]),
                short: Some("m".into()),
                env: None,
                completer: None,
            },
        );
        options.insert(
            "baseline".into(),
            OptionSpec {
                ty: "path".into(),
                description: None,
                default: None,
                choices: None,
                short: None,
                env: None,
                completer: None,
            },
        );
        options.insert(
            "validate".into(),
            OptionSpec {
                ty: "bool".into(),
                description: None,
                default: None,
                choices: None,
                short: None,
                env: None,
                completer: None,
            },
        );
        let _: Vec<ArgSpec> = Vec::new(); // args intentionally empty here

        let schema = Schema {
            args: vec![],
            options,
            strict: false,
        };
        let cfg = CommandConfig {
            schema: Some(schema),
            ..Default::default()
        };
        CommandData::from_config("ddp-bench".into(), &cfg)
    }

    fn data_with_one_cmd() -> CompletionData {
        CompletionData {
            top_level: vec!["ddp-bench".into(), "libtorch".into()],
            commands: vec![make_cmd_with_schema()],
        }
    }

    #[test]
    fn option_completion_types_resolve() {
        let cmd = make_cmd_with_schema();
        let model = cmd
            .options
            .iter()
            .find(|o| o.long == "model")
            .expect("model option present");
        assert!(matches!(&model.value, ValueKind::Choices(cs) if cs == &vec!["mlp".to_string(), "resnet".into()]));
        let baseline = cmd
            .options
            .iter()
            .find(|o| o.long == "baseline")
            .expect("baseline option present");
        assert!(matches!(baseline.value, ValueKind::Path));
        let validate = cmd
            .options
            .iter()
            .find(|o| o.long == "validate")
            .expect("validate option present");
        assert!(!validate.takes_value);
        assert!(matches!(validate.value, ValueKind::None));
    }

    #[test]
    fn bash_contains_choices_and_path_completion() {
        let data = data_with_one_cmd();
        let out = emit_bash(&data);
        assert!(out.contains("--model|-m"), "model|short flag present");
        assert!(out.contains("mlp resnet"), "choice values inlined");
        assert!(out.contains("--baseline) COMPREPLY=($(compgen -f"), "path → file completion");
        assert!(!out.contains("--validate)"),
            "bool flags must not appear in value case (no value to complete)");
    }

    #[test]
    fn zsh_contains_choices_and_files() {
        let data = data_with_one_cmd();
        let out = emit_zsh(&data);
        assert!(out.contains("_values 'value' mlp resnet"));
        assert!(out.contains("_files"));
    }

    #[test]
    fn fish_contains_choices_and_path_flag() {
        let data = data_with_one_cmd();
        let out = emit_fish(&data);
        assert!(out.contains("-l 'model' -s 'm'"));
        assert!(out.contains("-a 'mlp resnet'"));
        assert!(out.contains("-l 'baseline' -r -F"));
    }

    #[test]
    fn top_level_flags_present_in_all_shells() {
        let data = data_with_one_cmd();
        // Each shell uses its own flag syntax: bash/zsh carry `--help` tokens
        // literally in word lists, while fish maps them to `-l 'help'`.
        let bash = emit_bash(&data);
        assert!(bash.contains("--help"), "bash: --help missing");
        assert!(bash.contains("--version"), "bash: --version missing");

        let zsh = emit_zsh(&data);
        assert!(zsh.contains("--help"), "zsh: --help missing");
        assert!(zsh.contains("--version"), "zsh: --version missing");

        let fish = emit_fish(&data);
        assert!(fish.contains("-l 'help'"), "fish: help long flag missing");
        assert!(fish.contains("-l 'version'"), "fish: version long flag missing");
        assert!(fish.contains("-s 'h'"), "fish: -h short missing");
    }
}
