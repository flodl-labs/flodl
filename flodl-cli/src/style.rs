//! Terminal colors and formatting.
//!
//! Auto-detects TTY. Falls back to plain text when piped.

use std::io::IsTerminal;

/// Whether stderr supports color (help output goes to stderr).
pub fn color_enabled() -> bool {
    std::io::stderr().is_terminal()
}

// ANSI escape helpers. Return empty strings when color is disabled.

pub fn green(s: &str) -> String {
    if color_enabled() {
        format!("\x1b[32m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn yellow(s: &str) -> String {
    if color_enabled() {
        format!("\x1b[33m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn bold(s: &str) -> String {
    if color_enabled() {
        format!("\x1b[1m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn dim(s: &str) -> String {
    if color_enabled() {
        format!("\x1b[2m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
