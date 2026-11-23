//! Stage timers for the CPU-averaging reduce path, printed at `-vv`.
//!
//! A reduce moves a model-sized frame rank → relay → controller and back,
//! and every hop reads, authenticates, combines and writes it. These
//! timers attribute a reduce's wall time to those stages. Each thread
//! accumulates its own totals and prints them at the end of its part of a
//! round, so a line names one hop: a relay's rank reader, the relay's
//! uplink writer, a controller reader, the controller's reduce loop.
//!
//! Times are exclusive: a stage timed inside another (the MAC inside a
//! parse) is subtracted from the outer one, so a line's stages add up.
//! Below `-vv` a timed call runs its closure and nothing else.

use std::cell::{Cell, RefCell};
use std::time::Instant;

use crate::log::{Verbosity, enabled};

/// What a stretch of reduce-path time was spent on.
#[derive(Clone, Copy)]
pub(crate) enum Stage {
    /// Reading a frame's bytes off a socket.
    Recv,
    /// Writing a frame's bytes to a socket.
    Send,
    /// HMAC-SHA256 over a frame, computed or verified.
    Mac,
    /// Copying a frame into a new buffer (MAC input, framing prefix).
    Copy,
    /// Parsing a frame into tensors.
    Parse,
    /// Encoding tensors into a frame.
    Encode,
    /// Element-wise accumulation of frames (relay fold, controller sum).
    Sum,
    /// The outer optimizer's step on the consensus.
    Outer,
}

const STAGES: [(Stage, &str); 8] = [
    (Stage::Recv, "recv"),
    (Stage::Send, "send"),
    (Stage::Mac, "mac"),
    (Stage::Copy, "copy"),
    (Stage::Parse, "parse"),
    (Stage::Encode, "encode"),
    (Stage::Sum, "sum"),
    (Stage::Outer, "outer"),
];

#[derive(Clone, Copy, Default)]
struct Total {
    nanos: u64,
    bytes: u64,
    calls: u32,
}

thread_local! {
    static TOTALS: RefCell<[Total; STAGES.len()]> = const { RefCell::new([Total { nanos: 0, bytes: 0, calls: 0 }; STAGES.len()]) };
    /// Time spent in stages nested inside the one currently being timed.
    static NESTED: Cell<u64> = const { Cell::new(0) };
}

/// Run `f` and charge its time to `stage` on this thread, less any time
/// charged to stages timed inside it. `bytes` is the frame size the stage
/// worked on (0 when it has none), summed for the report.
pub(crate) fn timed<R>(stage: Stage, bytes: usize, f: impl FnOnce() -> R) -> R {
    if !enabled(Verbosity::Debug) {
        return f();
    }
    charge(stage, bytes, f)
}

fn charge<R>(stage: Stage, bytes: usize, f: impl FnOnce() -> R) -> R {
    let outer_nested = NESTED.replace(0);
    let start = Instant::now();
    let out = f();
    let elapsed = start.elapsed().as_nanos() as u64;
    let inner = NESTED.get();
    NESTED.set(outer_nested + elapsed);
    TOTALS.with(|t| {
        let total = &mut t.borrow_mut()[stage as usize];
        total.nanos += elapsed.saturating_sub(inner);
        total.bytes += bytes as u64;
        total.calls += 1;
    });
    out
}

/// Print this thread's totals as one `-vv` line and reset them. Stages
/// that did not run are left out; nothing prints when none took a
/// millisecond, which keeps small control frames out of the log.
pub(crate) fn report(label: &str) {
    if !enabled(Verbosity::Debug) {
        return;
    }
    let totals = TOTALS.with(|t| std::mem::take(&mut *t.borrow_mut()));
    let sum: u64 = totals.iter().map(|t| t.nanos).sum();
    if sum < 1_000_000 {
        return;
    }
    let mut line = format!("[stage-prof] {label}: {:.0}ms", sum as f64 / 1e6);
    for (stage, name) in STAGES {
        let t = totals[stage as usize];
        if t.calls == 0 {
            continue;
        }
        line.push_str(&format!(" | {name} {:.0}ms", t.nanos as f64 / 1e6));
        if t.bytes > 0 {
            line.push_str(&format!("/{:.0}MB", t.bytes as f64 / (1024.0 * 1024.0)));
        }
        if t.calls > 1 {
            line.push_str(&format!(" x{}", t.calls));
        }
    }
    crate::debug!("{line}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nested_stage_is_subtracted_from_the_outer_one() {
        // `charge` is `timed` without the verbosity gate, which reads
        // process state a test must not depend on.
        let nap = std::time::Duration::from_millis(20);
        TOTALS.with(|t| *t.borrow_mut() = Default::default());
        let start = Instant::now();
        charge(Stage::Parse, 0, || {
            std::thread::sleep(nap);
            charge(Stage::Mac, 5, || std::thread::sleep(nap));
        });
        let wall = start.elapsed().as_nanos() as u64;
        let totals = TOTALS.with(|t| std::mem::take(&mut *t.borrow_mut()));
        let (parse, mac) = (totals[Stage::Parse as usize], totals[Stage::Mac as usize]);
        assert!(mac.nanos >= 20_000_000 && mac.bytes == 5 && mac.calls == 1);
        assert!(parse.nanos >= 20_000_000, "outer lost its own time");
        // Exclusive times add up to the time that passed. Were the inner
        // stage still counted in the outer one, the sum would exceed the
        // wall time by the inner stage. A sleep may overrun by any
        // amount, so no bound is put on either stage alone.
        assert!(
            parse.nanos + mac.nanos <= wall,
            "outer {} ns + inner {} ns exceed the {wall} ns that passed",
            parse.nanos,
            mac.nanos
        );
    }

    #[test]
    fn below_debug_a_timed_call_only_runs_its_closure() {
        if enabled(Verbosity::Debug) {
            return;
        }
        assert_eq!(timed(Stage::Mac, 10, || 7), 7);
        TOTALS.with(|t| assert_eq!(t.borrow()[Stage::Mac as usize].calls, 0));
    }
}
