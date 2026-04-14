//! Shell completion generation and one-shot setup.
//!
//! `fdl completions bash`  outputs a bash completion script to stdout.
//! `fdl completions zsh`   outputs a zsh  completion script to stdout.
//! `fdl autocomplete`      detects shell, installs completions, reloads.

use std::path::Path;

use crate::config::{self, ProjectConfig};
use crate::style;

/// Built-in commands that are always available.
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
    "help",
    "version",
];

/// The marker comment we use to detect our block in rc files.
const MARKER: &str = "# fdl shell completions";

/// Generate a completion script for the given shell.
pub fn generate(shell: &str, project: Option<(&ProjectConfig, &Path)>) {
    match shell {
        "bash" => gen_bash(project),
        "zsh" => gen_zsh(project),
        other => {
            eprintln!("unsupported shell: {other}");
            eprintln!("supported: bash, zsh");
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  eval \"$(fdl completions bash)\"   # bash");
            eprintln!("  eval \"$(fdl completions zsh)\"    # zsh");
            eprintln!("  fdl autocomplete                  # auto-detect and install");
        }
    }
}

/// One-shot setup: detect shell, write to rc file, reload.
pub fn autocomplete(project: Option<(&ProjectConfig, &Path)>) {
    let shell = detect_shell();
    let (rc_path, shell_name) = match shell.as_str() {
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
            // Remove existing block and re-add.
            let cleaned = remove_completion_block(&content);
            if let Err(e) = std::fs::write(&rc_path, cleaned) {
                eprintln!("error: cannot write {rc_path}: {e}");
                return;
            }
        }
    }

    // Generate the completion script.
    let script = capture_completion_script(shell_name, project);

    // Append to rc file.
    let block = format!("\n{MARKER}\n{script}{MARKER} end\n");
    match std::fs::OpenOptions::new().append(true).open(&rc_path) {
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
        style::dim(&format!("source {rc_path}"))
    );
    eprintln!(
        "  Or restart your terminal for completions to take effect.",
    );
}

fn detect_shell() -> String {
    // Check $SHELL env var.
    if let Ok(shell) = std::env::var("SHELL") {
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

/// Capture the completion script as a string instead of printing to stdout.
fn capture_completion_script(
    shell: &str,
    project: Option<(&ProjectConfig, &Path)>,
) -> String {
    // We reuse the same generation logic but capture to a string.
    let words = collect_words(project);
    let command_jobs = collect_command_jobs(project);

    match shell {
        "zsh" => build_zsh_script(&words, &command_jobs),
        _ => build_bash_script(&words, &command_jobs),
    }
}

fn collect_words(project: Option<(&ProjectConfig, &Path)>) -> Vec<String> {
    let mut words: Vec<String> = BUILTINS.iter().map(|s| s.to_string()).collect();
    if let Some((proj, _)) = project {
        for name in proj.scripts.keys() {
            words.push(name.clone());
        }
        for cmd_path in &proj.commands {
            words.push(config::command_name(cmd_path).to_string());
        }
    }
    words
}

fn collect_command_jobs(
    project: Option<(&ProjectConfig, &Path)>,
) -> Vec<(String, Vec<String>)> {
    let mut command_jobs = Vec::new();
    if let Some((proj, root)) = project {
        for cmd_path in &proj.commands {
            let short = config::command_name(cmd_path).to_string();
            let cmd_dir = root.join(cmd_path);
            if let Ok(cmd_config) = config::load_command(&cmd_dir) {
                let jobs: Vec<String> = cmd_config.jobs.keys().cloned().collect();
                if !jobs.is_empty() {
                    command_jobs.push((short, jobs));
                }
            }
        }
    }
    command_jobs
}

fn build_bash_script(
    words: &[String],
    command_jobs: &[(String, Vec<String>)],
) -> String {
    let top_level = words.join(" ");
    let mut s = String::new();
    s.push_str("eval \"$(fdl completions bash)\"\n");
    // The actual completion function is emitted by gen_bash via stdout.
    // For the rc file, we use eval to stay dynamic.
    let _ = (top_level, command_jobs);
    s
}

fn build_zsh_script(
    words: &[String],
    command_jobs: &[(String, Vec<String>)],
) -> String {
    let _ = (words, command_jobs);
    "eval \"$(fdl completions zsh)\"\n".to_string()
}

fn gen_bash(project: Option<(&ProjectConfig, &Path)>) {
    // Collect all top-level completions.
    let mut words: Vec<String> = BUILTINS.iter().map(|s| s.to_string()).collect();
    let mut command_jobs: Vec<(String, Vec<String>)> = Vec::new();

    if let Some((proj, root)) = project {
        for name in proj.scripts.keys() {
            words.push(name.clone());
        }
        for cmd_path in &proj.commands {
            let short = config::command_name(cmd_path).to_string();
            words.push(short.clone());

            // Collect jobs for this command.
            let cmd_dir = root.join(cmd_path);
            if let Ok(cmd_config) = config::load_command(&cmd_dir) {
                let jobs: Vec<String> = cmd_config.jobs.keys().cloned().collect();
                if !jobs.is_empty() {
                    command_jobs.push((short, jobs));
                }
            }
        }
    }

    let top_level = words.join(" ");

    println!("# fdl bash completion (generated)");
    println!("# eval \"$(fdl completions bash)\"");
    println!("_fdl_completions() {{");
    println!("    local cur prev commands");
    println!("    cur=\"${{COMP_WORDS[COMP_CWORD]}}\"");
    println!("    prev=\"${{COMP_WORDS[1]}}\"");
    println!();
    println!("    commands=\"{top_level}\"");
    println!();
    println!("    if [[ $COMP_CWORD -eq 1 ]]; then");
    println!("        COMPREPLY=($(compgen -W \"$commands\" -- \"$cur\"))");
    println!("        return");
    println!("    fi");

    // Sub-command job completions.
    for (cmd, jobs) in &command_jobs {
        let job_list = jobs.join(" ");
        println!();
        println!("    if [[ \"$prev\" == \"{cmd}\" && $COMP_CWORD -eq 2 ]]; then");
        println!("        COMPREPLY=($(compgen -W \"{job_list}\" -- \"$cur\"))");
        println!("        return");
        println!("    fi");
    }

    // libtorch sub-commands.
    println!();
    println!("    if [[ \"$prev\" == \"libtorch\" && $COMP_CWORD -eq 2 ]]; then");
    println!(
        "        COMPREPLY=($(compgen -W \"download build list activate remove info\" -- \"$cur\"))"
    );
    println!("        return");
    println!("    fi");

    println!("}}");
    println!("complete -F _fdl_completions fdl");
}

fn gen_zsh(project: Option<(&ProjectConfig, &Path)>) {
    let mut words: Vec<String> = BUILTINS.iter().map(|s| s.to_string()).collect();
    let mut command_jobs: Vec<(String, Vec<String>)> = Vec::new();

    if let Some((proj, root)) = project {
        for name in proj.scripts.keys() {
            words.push(name.clone());
        }
        for cmd_path in &proj.commands {
            let short = config::command_name(cmd_path).to_string();
            words.push(short.clone());

            let cmd_dir = root.join(cmd_path);
            if let Ok(cmd_config) = config::load_command(&cmd_dir) {
                let jobs: Vec<String> = cmd_config.jobs.keys().cloned().collect();
                if !jobs.is_empty() {
                    command_jobs.push((short, jobs));
                }
            }
        }
    }

    let top_level = words.join(" ");

    println!("#compdef fdl");
    println!("# fdl zsh completion (generated)");
    println!("# eval \"$(fdl completions zsh)\"");
    println!("_fdl() {{");
    println!("    local -a commands");
    println!("    commands=({top_level})");
    println!();
    println!("    if (( CURRENT == 2 )); then");
    println!("        _describe 'command' commands");
    println!("        return");
    println!("    fi");

    // Sub-command jobs.
    if !command_jobs.is_empty() {
        println!();
        println!("    case $words[2] in");
        for (cmd, jobs) in &command_jobs {
            let job_list = jobs.join(" ");
            println!("        {cmd})");
            println!("            if (( CURRENT == 3 )); then");
            println!("                local -a jobs");
            println!("                jobs=({job_list})");
            println!("                _describe 'job' jobs");
            println!("            fi");
            println!("            ;;");
        }
        println!("        libtorch)");
        println!("            if (( CURRENT == 3 )); then");
        println!("                local -a subcmds");
        println!("                subcmds=(download build list activate remove info)");
        println!("                _describe 'subcommand' subcmds");
        println!("            fi");
        println!("            ;;");
        println!("    esac");
    }

    println!("}}");
    println!("_fdl");
}
