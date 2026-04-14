//! Verbosity-gated output for flodl.
//!
//! Normal and Verbose levels go to **stdout**.
//! Debug (`-vv`) and Trace (`-vvv`) go to **stderr** (unbuffered, visible
//! in Docker non-TTY mode where stdout is block-buffered).
//! Errors always go to **stderr** via standard `eprintln!` -- never gated.
//!
//! ```ignore
//! // Simple: just print (Normal level, suppressed by --quiet)
//! flodl::msg!("Epoch {}: loss={:.4}", epoch, loss);
//!
//! // With explicit level
//! flodl::msg!(@Verbose, "AllReduce: {:.1}ms", elapsed);
//! flodl::msg!(@Debug, "per-batch timing: {}ms", ms);
//! flodl::msg!(@Trace, "{:?}", my_tensor);
//! ```
//!
//! The level is a single global atomic -- no module filtering, no log targets.
//!
//! **Zero-code setup:** set the `FLODL_VERBOSITY` environment variable
//! (`0`..`4`, or `quiet`/`normal`/`verbose`/`debug`/`trace`).
//! The env var is read once on first access.
//!
//! ```bash
//! FLODL_VERBOSITY=verbose cargo run    # -v equivalent, no code needed
//! ```
//!
//! **Programmatic override:** [`set_verbosity`] takes precedence over the env var.

use std::sync::atomic::{AtomicU8, Ordering};

/// Verbosity level for `flodl` output.
///
/// Higher levels include everything from lower levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Verbosity {
    /// `--quiet` -- errors only (via `eprintln!`), all macros suppressed.
    Quiet = 0,
    /// Default -- epoch summaries, progress, setup, milestones.
    Normal = 1,
    /// `-v` -- DDP sync, cadence changes, data loading detail.
    Verbose = 2,
    /// `-vv` -- heavy loop internals, per-batch timing.
    Debug = 3,
    /// `-vvv` -- extreme granularity.
    Trace = 4,
}

/// Environment variable used to set verbosity without code.
pub const ENV_VAR: &str = "FLODL_VERBOSITY";

/// Sentinel: 255 means "not yet initialized from env".
const UNINIT: u8 = 255;

static LEVEL: AtomicU8 = AtomicU8::new(UNINIT);

/// Read `FLODL_VERBOSITY` from the environment.
fn level_from_env() -> u8 {
    match std::env::var(ENV_VAR) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "0" | "quiet" => Verbosity::Quiet as u8,
            "1" | "normal" => Verbosity::Normal as u8,
            "2" | "verbose" => Verbosity::Verbose as u8,
            "3" | "debug" => Verbosity::Debug as u8,
            "4" | "trace" => Verbosity::Trace as u8,
            _ => Verbosity::Normal as u8,
        },
        Err(_) => Verbosity::Normal as u8,
    }
}

/// Load the level, initializing from env on first access.
#[inline]
fn load_or_init() -> u8 {
    let v = LEVEL.load(Ordering::Relaxed);
    if v != UNINIT {
        return v;
    }
    let from_env = level_from_env();
    // CAS: if still UNINIT, set to env value. If someone called
    // set_verbosity() concurrently, their value wins.
    let _ = LEVEL.compare_exchange(UNINIT, from_env, Ordering::Relaxed, Ordering::Relaxed);
    LEVEL.load(Ordering::Relaxed)
}

/// Set the global verbosity level.
///
/// Overrides the `FLODL_VERBOSITY` env var. Typically called once at startup.
pub fn set_verbosity(level: Verbosity) {
    LEVEL.store(level as u8, Ordering::Relaxed);
}

/// Get the current global verbosity level.
pub fn verbosity() -> Verbosity {
    match load_or_init() {
        0 => Verbosity::Quiet,
        2 => Verbosity::Verbose,
        3 => Verbosity::Debug,
        4 => Verbosity::Trace,
        _ => Verbosity::Normal,
    }
}

/// Check whether a given verbosity level is enabled.
#[inline]
pub fn enabled(level: Verbosity) -> bool {
    load_or_init() >= level as u8
}

/// Print to stdout, gated by verbosity level.
///
/// ```ignore
/// // Default (Normal): suppressed by --quiet, shown otherwise
/// flodl::msg!("Epoch {}: loss={:.4}", epoch, loss);
///
/// // With explicit level (@ prefix)
/// flodl::msg!(@Verbose, "AllReduce: {:.1}ms", elapsed);
/// flodl::msg!(@Debug, "per-batch: {}ms", ms);
/// flodl::msg!(@Trace, "{:?}", tensor);
/// ```
#[macro_export]
macro_rules! msg {
    // With explicit level: flodl::msg!(@Verbose, "msg", arg)
    (@$level:expr, $($arg:tt)+) => {
        if $crate::log::enabled($level) {
            println!($($arg)+)
        }
    };
    // Default (Normal): flodl::msg!("msg", arg)
    ($($arg:tt)+) => {
        if $crate::log::enabled($crate::log::Verbosity::Normal) {
            println!($($arg)+)
        }
    };
}

/// Prints to stdout at `-v` (Verbose) and above.
///
/// Internal shortcut. Equivalent to `flodl::msg!(@Verbose, ...)`.
#[macro_export]
macro_rules! verbose {
    ($($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Verbosity::Verbose) {
            println!($($arg)*)
        }
    };
}

/// Prints to stderr at `-vv` (Debug) and above.
///
/// Uses stderr so output is unbuffered and visible immediately in Docker
/// non-TTY mode (stdout is block-buffered there).
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Verbosity::Debug) {
            eprintln!($($arg)*)
        }
    };
}

/// Prints to stderr at `-vvv` (Trace) and above.
///
/// Uses stderr so output is unbuffered and visible immediately in Docker
/// non-TTY mode (stdout is block-buffered there).
#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Verbosity::Trace) {
            eprintln!($($arg)*)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verbosity_ordering() {
        assert!(Verbosity::Quiet < Verbosity::Normal);
        assert!(Verbosity::Normal < Verbosity::Verbose);
        assert!(Verbosity::Verbose < Verbosity::Debug);
        assert!(Verbosity::Debug < Verbosity::Trace);
    }

    #[test]
    fn test_set_and_get() {
        let orig = verbosity();

        set_verbosity(Verbosity::Debug);
        assert_eq!(verbosity(), Verbosity::Debug);

        set_verbosity(Verbosity::Trace);
        assert_eq!(verbosity(), Verbosity::Trace);

        set_verbosity(Verbosity::Quiet);
        assert_eq!(verbosity(), Verbosity::Quiet);

        set_verbosity(Verbosity::Normal);
        assert_eq!(verbosity(), Verbosity::Normal);

        set_verbosity(orig);
    }

    #[test]
    fn test_enabled() {
        let orig = verbosity();

        set_verbosity(Verbosity::Verbose);
        assert!(enabled(Verbosity::Normal));
        assert!(enabled(Verbosity::Verbose));
        assert!(!enabled(Verbosity::Debug));
        assert!(!enabled(Verbosity::Trace));

        set_verbosity(Verbosity::Trace);
        assert!(enabled(Verbosity::Normal));
        assert!(enabled(Verbosity::Verbose));
        assert!(enabled(Verbosity::Debug));
        assert!(enabled(Verbosity::Trace));

        set_verbosity(Verbosity::Quiet);
        assert!(!enabled(Verbosity::Normal));
        assert!(!enabled(Verbosity::Verbose));

        set_verbosity(orig);
    }

    #[test]
    fn test_msg_with_level() {
        let orig = verbosity();

        set_verbosity(Verbosity::Normal);
        // Verify the macro compiles with both forms
        crate::msg!("normal message");
        crate::msg!(@Verbosity::Verbose, "verbose message");
        crate::msg!(@Verbosity::Debug, "debug: {}", 42);

        set_verbosity(orig);
    }
}
