//! Unit tests for the dropped best-effort broadcast trace: a failed
//! control broadcast to a live rank bumps `lost_broadcasts` and emits a
//! `LostBroadcast` timeline event, while a failed `Shutdown` (an expected
//! teardown race, not lost live coordination) is exempt — as is anything
//! sent once shutdown has already been broadcast.

use crate::distributed::ddp::ElChe;
use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};
use crate::monitor::{EventKind, Timeline};

use super::super::{ClusterCoordinator, ClusterCoordinatorConfig, RunPhase};
use super::ControlMsgWire;

fn coord_with_timeline(world_size: usize, tl: std::sync::Arc<Timeline>) -> ClusterCoordinator {
    ClusterCoordinator::for_test(
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Cadence,
            AverageBackend::Cpu,
            world_size,
            ElChe::new(world_size, 4),
        )
        .no_divergence_guard()
        .progressive(true)
        .timeline(tl),
    )
}

#[test]
fn failed_broadcast_records_counter_and_timeline_event() {
    let tl = Timeline::new(1000);
    let mut coord = coord_with_timeline(2, tl.clone());

    // `for_test` leaves `control_streams` empty, so every `send_control`
    // fails -> both live ranks miss the SyncNow broadcast.
    let err = coord
        .broadcast_control(&ControlMsgWire::SyncNow)
        .expect_err("send to a streamless coord must fail");
    assert!(err.to_string().contains("broadcast_control failed"));

    assert_eq!(coord.lost_broadcasts, 1, "one dropped broadcast counted");

    let (_samples, events) = tl.drain();
    let lost: Vec<(String, usize)> = events
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::LostBroadcast { control, failures } => Some((control.clone(), *failures)),
            _ => None,
        })
        .collect();
    assert_eq!(lost, vec![("SyncNow".to_string(), 2)]);
}

/// A heartbeat that misses ranks AFTER shutdown was broadcast is a teardown
/// race, not lost coordination — the ranks are supposed to be gone.
///
/// From the rig: a 3-rank run that completed perfectly raised `[critical]
/// control_drop root — CoordHeartbeat did not reach 2 live rank(s)` one log
/// line after both of that host's ranks exited cleanly. `is_dead` only knows
/// heartbeat staleness, so a cleanly finished rank still counts as live and
/// its closed socket read as a dropped signal. Wiring twin of the predicate
/// matrix in `lifecycle::tests` — this pins the call site, those pin the rule.
#[test]
fn broadcast_failure_after_shutdown_initiated_is_exempt() {
    let tl = Timeline::new(1000);
    let mut coord = coord_with_timeline(2, tl.clone());
    coord.run_phase = RunPhase::ShutdownInitiated;

    let _ = coord.broadcast_control(&ControlMsgWire::CoordHeartbeat);
    assert_eq!(
        coord.lost_broadcasts, 0,
        "a teardown-phase heartbeat miss must not be traced",
    );

    let (_samples, events) = tl.drain();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e.kind, EventKind::LostBroadcast { .. })),
        "no LostBroadcast event once shutdown has been broadcast",
    );
}

/// The guard is phase-scoped, not message-scoped: the SAME heartbeat missing
/// mid-run is still a real dropped signal and must still be traced. Without
/// this, "fix the false positive" could silently become "mute the lane".
#[test]
fn the_same_heartbeat_miss_before_shutdown_is_traced() {
    let tl = Timeline::new(1000);
    let mut coord = coord_with_timeline(2, tl.clone());

    let _ = coord.broadcast_control(&ControlMsgWire::CoordHeartbeat);
    assert_eq!(coord.lost_broadcasts, 1, "mid-run heartbeat miss is traced");

    let (_samples, events) = tl.drain();
    assert!(
        events.iter().any(
            |e| matches!(&e.kind, EventKind::LostBroadcast { control, .. }
                              if control == "CoordHeartbeat")
        ),
        "mid-run heartbeat miss emits a LostBroadcast event",
    );
}

#[test]
fn failed_shutdown_broadcast_is_exempt() {
    let tl = Timeline::new(1000);
    let mut coord = coord_with_timeline(2, tl.clone());

    // The Shutdown send also fails (no streams) but must NOT be traced.
    let _ = coord.broadcast_control(&ControlMsgWire::Shutdown);
    assert_eq!(
        coord.lost_broadcasts, 0,
        "Shutdown is exempt from the trace"
    );

    let (_samples, events) = tl.drain();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e.kind, EventKind::LostBroadcast { .. })),
        "no LostBroadcast event for a failed Shutdown",
    );
}

/// A rank that latched a clean `Exiting` left by protocol: it must not be
/// addressed at all, so its closed relay can neither fail the broadcast nor
/// bump the trace. From the rig: every probe run's final `EvalBroadcast`
/// reported `failed for 2 of 3 ranks (Broken pipe)` — the two pascal workers
/// had already exited cleanly and the coordinator knew it.
#[test]
fn an_exited_rank_is_not_addressed_and_cannot_fail_a_broadcast() {
    let tl = Timeline::new(1000);
    let mut coord = coord_with_timeline(2, tl.clone());
    coord.exited[1] = true;

    // Rank 0 is live and streamless, so the broadcast still fails — but
    // only for rank 0. The exited rank contributes nothing.
    let err = coord
        .broadcast_control(&ControlMsgWire::SyncNow)
        .expect_err("live streamless rank still fails the send");
    let msg = err.to_string();
    assert!(msg.contains("rank 0"), "live rank named: {msg}");
    assert!(
        !msg.contains("rank 1"),
        "exited rank must not appear: {msg}"
    );

    let (_samples, events) = tl.drain();
    let failures: Vec<usize> = events
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::LostBroadcast { failures, .. } => Some(*failures),
            _ => None,
        })
        .collect();
    assert_eq!(failures, vec![1], "one live miss traced, not two");
}

/// The observed teardown shape end to end: every worker already exited
/// cleanly when the final `EvalBroadcast` fires. Nothing is addressed, so
/// the broadcast SUCCEEDS (vacuously) instead of reporting a scary
/// multi-rank failure out of a perfect run.
#[test]
fn a_broadcast_to_a_fully_exited_cohort_is_a_clean_no_op() {
    let tl = Timeline::new(1000);
    let mut coord = coord_with_timeline(2, tl.clone());
    coord.exited[0] = true;
    coord.exited[1] = true;

    coord
        .broadcast_control(&ControlMsgWire::EvalBroadcast {
            epoch: 1,
            metric: 8.19,
        })
        .expect("no live addressee means nothing can fail");
    assert_eq!(coord.lost_broadcasts, 0);

    let (_samples, events) = tl.drain();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e.kind, EventKind::LostBroadcast { .. })),
        "no LostBroadcast event for a fully-exited cohort",
    );
}

/// A connected socket pair on loopback: `(coordinator side, host side)`.
fn socket_pair() -> (std::net::TcpStream, std::net::TcpStream) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let host = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (coord, _) = listener.accept().unwrap();
    (coord, host)
}

/// A failed write ends the connection for good: it is shut, and later
/// sends fail at once without touching the socket. A host that stopped
/// reading would otherwise cost the coordinator a full write-stall timeout
/// on every frame, on the thread that keeps the healthy ranks' beacon
/// going, until those ranks give the coordinator up for dead.
#[test]
fn a_failed_write_closes_the_connection_and_is_never_retried() {
    let mut coord = coord_with_timeline(2, Timeline::new(100));
    let (stream, host) = socket_pair();
    coord.control_streams = vec![stream];
    coord.rank_to_conn = vec![Some(0), Some(0)];

    // The host goes away. A first write can still land in the kernel
    // buffer; the reset it provokes fails one of the next.
    drop(host);
    let mut failed = false;
    for _ in 0..200 {
        if coord
            .send_control(0, &ControlMsgWire::CoordHeartbeat)
            .is_err()
        {
            failed = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(failed, "writing to a closed peer must fail eventually");
    assert_eq!(coord.conn_closed, vec![true]);

    // Every rank on that connection now fails fast, with the latch named.
    for rank in 0..2 {
        let err = coord
            .send_control(rank, &ControlMsgWire::CoordHeartbeat)
            .expect_err("a closed connection is not written again");
        assert!(err.to_string().contains("connection closed"), "got: {err}");
    }
}

/// The short write bound applies only once EVERY rank a connection
/// carries is dead: a host with one live rank keeps the normal timeout,
/// since its relay is alive and a slow drain must not cost that rank its
/// connection.
#[test]
fn writes_are_bounded_only_on_a_connection_whose_ranks_are_all_dead() {
    let ledger = crate::distributed::controller::DeadRanks::new(3);
    let mut coord = ClusterCoordinator::for_test(
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Cadence,
            AverageBackend::Cpu,
            3,
            ElChe::new(3, 4),
        )
        .no_divergence_guard()
        .dead_ranks(std::sync::Arc::clone(&ledger)),
    );
    let (stream_a, _host_a) = socket_pair();
    let (stream_b, _host_b) = socket_pair();
    // Host A carries rank 0; host B carries ranks 1 and 2.
    coord.control_streams = vec![stream_a, stream_b];
    coord.rank_to_conn = vec![Some(0), Some(1), Some(1)];
    let normal = coord.control_streams[1].write_timeout().unwrap();

    ledger.declare_dead(1);
    coord.bound_writes_to_dead_conn(1);
    assert_eq!(
        coord.control_streams[1].write_timeout().unwrap(),
        normal,
        "rank 2 still lives on that host",
    );

    ledger.declare_dead(2);
    coord.bound_writes_to_dead_conn(2);
    let bounded = coord.control_streams[1].write_timeout().unwrap();
    assert!(
        bounded.is_some_and(|t| t <= std::time::Duration::from_secs(1)),
        "a connection with no live rank gets a short write bound, got {bounded:?}",
    );
    assert_eq!(
        coord.control_streams[0].write_timeout().unwrap(),
        normal,
        "the other host is untouched",
    );
}
