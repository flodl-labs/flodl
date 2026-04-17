//! flodl-cli — library side of the `fdl` binary.
//!
//! This crate is both a library and a binary. The binary (`fdl`) is the
//! user-facing driver; the library exposes the pieces that other crates
//! (e.g. a flodl-based training binary) need to integrate with the
//! `fdl` ecosystem:
//!
//! - [`FdlArgs`] — derive macro + trait for argv parsing and schema emission
//! - [`parse_or_schema`] — intercepts `--fdl-schema` / `--help` and dispatches
//! - [`Schema`], [`OptionSpec`], [`ArgSpec`] — the canonical schema shape
//!
//! # Example
//!
//! ```ignore
//! use flodl_cli::{FdlArgs, parse_or_schema};
//!
//! /// My training binary.
//! #[derive(FdlArgs, Debug)]
//! struct Cli {
//!     /// Model to run.
//!     #[option(short = 'm', default = "all")]
//!     model: String,
//!
//!     /// Write a report instead of training.
//!     #[option(default = "runs/report.md")]
//!     report: Option<String>,
//! }
//!
//! fn main() {
//!     let cli: Cli = parse_or_schema();
//!     // ... use cli.model, cli.report, etc.
//! }
//! ```

// Internal modules — shared by lib consumers and the fdl binary.
pub mod api_ref;
pub mod args;
pub mod completions;
pub mod config;
pub mod context;
pub mod dispatch;
pub mod diagnose;
pub mod init;
pub mod libtorch;
pub mod overlay;
pub mod run;
pub mod schema;
pub mod schema_cache;
pub mod setup;
pub mod skill;
pub mod style;
pub mod util;

/// Print a red-prefixed `error: <formatted>` line to stderr.
///
/// Takes standard `format!` arguments. Coloring follows the `--ansi` /
/// `--no-ansi` / `NO_COLOR` / `FORCE_COLOR` chain via
/// [`style::color_enabled`], so pipes stay plain automatically.
#[macro_export]
macro_rules! cli_error {
    ($($arg:tt)*) => {
        $crate::style::print_cli_error(format_args!($($arg)*))
    };
}

// ── Public API for binary authors ──────────────────────────────────────

/// Parse argv into `T`, intercepting `--fdl-schema` and `--help`.
pub use args::parse_or_schema;

/// Slice-based variant of [`parse_or_schema`] — parses from an explicit
/// `&[String]` rather than `std::env::args()`. Used by the `fdl` driver to
/// dispatch per-sub-command arg tails.
pub use args::parse_or_schema_from;

/// Trait implemented by `#[derive(FdlArgs)]` structs. Binary authors do
/// not typically implement this manually — the derive emits it.
pub use args::FdlArgsTrait;

/// Derive macro for `FdlArgs`. Generates argv parsing, `--fdl-schema`
/// emission, and `--help` rendering from a single struct definition.
pub use flodl_cli_macros::FdlArgs;

/// Schema types — mirror the JSON shape emitted by `--fdl-schema` and
/// consumed by the fdl driver.
pub use config::{ArgSpec, OptionSpec, Schema};

/// Re-exported dependencies the derive macro needs to reference by path.
/// Users should not depend on these directly — they are only stable as
/// an implementation detail of the derive.
#[doc(hidden)]
pub use serde_json;
