//! One process per cell.
//!
//! A cell is one (model, mode) pair. A sweep over several of them runs each
//! in a child process of its own, for two reasons. A cluster training
//! session can run once per process: the rendezvous, relays and coordinator
//! behind it are built for one session and shut down with it. And rank
//! children are started with their launcher's arguments, so a launcher
//! holding a sweep would hand every rank the whole sweep; a launcher
//! holding one cell hands them that cell.
//!
//! The sweeping process keeps the decisions that need the whole picture
//! (which cells this box can run, the exit code) and trains nothing itself.

use std::process::Command;

/// Carries the sweeping process's pid to the cell it started, so the cell
/// can tell when that process is gone.
const DRIVER_PID_ENV: &str = "DDP_BENCH_DRIVER_PID";

/// The arguments for one cell: `args` with every `--model` and `--mode`
/// removed (both `--flag value` and `--flag=value`), and the cell's own pair
/// appended. Everything else rides through untouched, so the cell trains
/// with exactly the options the sweep was given.
pub fn cell_args(args: impl IntoIterator<Item = String>, model: &str, mode: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--model" || arg == "--mode" {
            args.next();
        } else if !(arg.starts_with("--model=") || arg.starts_with("--mode=")) {
            out.push(arg);
        }
    }
    out.extend(["--model", model, "--mode", mode].map(String::from));
    out
}

/// Train one cell in a child process and wait for it. `Err` describes a
/// cell that did not exit 0; its own output says why.
pub fn run(model: &str, mode: &str) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate this binary: {e}"))?;
    let status = Command::new(&exe)
        .args(cell_args(std::env::args().skip(1), model, mode))
        .env(DRIVER_PID_ENV, std::process::id().to_string())
        .status()
        .map_err(|e| format!("cannot start {}: {e}", exe.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("its process ended with {status}"))
    }
}

/// In a cell started by a sweep, exit when the sweeping process is gone.
///
/// A sweep stopped from outside (a dropped SSH session, a stopped
/// container) takes a signal the cell never sees, and the cell would train
/// on with nothing left to collect it. The cell holds the coordinator of
/// its run, so its exit ends its ranks too.
///
/// The variable is inherited by the cell's own rank children, whose parent
/// is the cell, so only a process whose parent IS the recorded pid arms the
/// watch.
#[cfg(unix)]
pub fn exit_with_driver() {
    let Some(driver) = std::env::var(DRIVER_PID_ENV).ok().and_then(|v| v.parse::<u32>().ok())
    else {
        return;
    };
    if std::os::unix::process::parent_id() != driver {
        return;
    }
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
        if std::os::unix::process::parent_id() != driver {
            eprintln!("ddp-bench: the sweep that started this cell is gone, stopping");
            std::process::exit(1);
        }
    });
}

#[cfg(not(unix))]
pub fn exit_with_driver() {}

/// Whether the active cluster forms through a join window, read from the
/// launcher's envelope. Each cell would open a window of its own there and
/// wait for the fleet to dial in again.
pub fn cluster_has_join_window() -> bool {
    std::env::var("FLODL_INTERNAL_FULL_CLUSTER_JSON")
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .is_some_and(|v| !v["controller"]["join"].is_null())
}

#[cfg(test)]
mod tests {
    use super::cell_args;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_cell_replaces_every_spelling_of_the_sweep_selectors() {
        let got = cell_args(
            args(&["--model", "all", "--epochs", "5", "--mode=all", "--model=lenet", "--quick"]),
            "mlp",
            "cpu-sync",
        );
        assert_eq!(got, args(&["--epochs", "5", "--quick", "--model", "mlp", "--mode", "cpu-sync"]));
    }

    #[test]
    fn a_sweep_given_no_selector_still_names_its_cell() {
        let got = cell_args(args(&["--seed", "42"]), "lenet", "solo-0");
        assert_eq!(got, args(&["--seed", "42", "--model", "lenet", "--mode", "solo-0"]));
    }

    #[test]
    fn options_that_merely_start_like_a_selector_are_kept() {
        let got = cell_args(args(&["--model-dir", "x", "--modes", "y"]), "lenet", "solo-0");
        assert_eq!(
            got,
            args(&["--model-dir", "x", "--modes", "y", "--model", "lenet", "--mode", "solo-0"])
        );
    }
}
