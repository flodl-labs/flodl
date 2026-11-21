//! Per-host relay tier.
//!
//! A relay agent runs once per host and collapses the cluster's
//! one-TCP-connection-per-rank topology into one connection per host per
//! channel. Toward its local ranks it presents the existing controller
//! protocol over loopback; upstream toward the real controller it speaks
//! the muxed protocol in [`mux`]. On the control channel it forwards,
//! tagging each frame with its originating rank; on the data channel it
//! is the first fold tier — local `RoundFrame`s are summed into ONE
//! `HostFrame` per round (see [`agent`]), so uplink bytes scale per host,
//! not per rank.
//!
//! See [`mux`] for the wire format and [`agent`] for the byte-router +
//! fold station.
//!
//! # Port map
//!
//! The controller process accepts every cross-host channel — NCCL
//! rendezvous, CPU-reduce data, coordinator control — on ONE port
//! (`controller.port`, bound `0.0.0.0`), demuxed by each connection's
//! channel-select magic (see `port_mux`).
//!
//! Each host's relay binds two LOOPBACK ports its local ranks dial, and
//! forwards upstream to the controller's mux port. With `r` the host's
//! first global rank:
//!
//! ```text
//! 127.0.0.1:+4+2r  relay data    loopback  → controller.host:port
//! 127.0.0.1:+5+2r  relay control loopback  → controller.host:port
//! ```
//!
//! The pair is keyed by the host's first rank because two hosts of one
//! run can share a network namespace: two containers on the host
//! network of a box that holds a card from each vendor, each with its
//! own libtorch. Global ranks are unique across hosts, so the pairs
//! never overlap, and a relay and its ranks derive the same pair from
//! the rank list they both hold
//! ([`crate::distributed::relay::relay_loopback_ports`]). The host
//! carrying rank 0 gets `+4`/`+5`, which never collide with the
//! controller's own port even when the relay shares its host.

pub mod agent;
pub mod mux;

/// Loopback port offset (from `controller.port`) the per-host relay binds
/// for the CPU-averaging data channel.
pub const RELAY_DATA_LOOPBACK_OFFSET: u16 = 4;

/// Loopback port offset (from `controller.port`) the per-host relay binds
/// for the coordinator control channel.
pub const RELAY_CONTROL_LOOPBACK_OFFSET: u16 = 5;

/// The loopback ports `(data, control)` of the relay serving the host
/// that carries `host_ranks`, derived from the controller's port and the
/// host's first global rank (see the module's port map).
///
/// Errors when the pair would fall past the port range, which a
/// saturating add would have turned into every such host sharing 65535.
pub fn relay_loopback_ports(
    controller_port: u16,
    host_ranks: impl IntoIterator<Item = usize>,
) -> crate::tensor::Result<(u16, u16)> {
    let first = host_ranks.into_iter().min().unwrap_or(0);
    let port = |offset: u16| -> Option<u16> {
        let p = usize::from(controller_port)
            .checked_add(usize::from(offset))?
            .checked_add(first.checked_mul(2)?)?;
        u16::try_from(p).ok()
    };
    match (
        port(RELAY_DATA_LOOPBACK_OFFSET),
        port(RELAY_CONTROL_LOOPBACK_OFFSET),
    ) {
        (Some(data), Some(control)) => Ok((data, control)),
        _ => Err(crate::tensor::TensorError::new(&format!(
            "relay: the loopback ports of the host whose first rank is {first} \
             fall past 65535 from controller port {controller_port}; use a \
             lower controller port"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_host_carrying_rank_zero_keeps_the_base_pair() {
        assert_eq!(relay_loopback_ports(1337, [0]).unwrap(), (1341, 1342));
        assert_eq!(relay_loopback_ports(1337, [1, 0]).unwrap(), (1341, 1342));
    }

    // Two hosts of one run can share a network namespace, and every
    // port of every host must then be distinct, whatever the split of
    // ranks between hosts.
    #[test]
    fn hosts_of_one_run_never_share_a_port() {
        let hosts: [&[usize]; 4] = [&[0], &[1, 2], &[3], &[4, 5, 6]];
        let mut seen = std::collections::BTreeSet::new();
        for ranks in hosts {
            let (data, control) = relay_loopback_ports(29500, ranks.iter().copied()).unwrap();
            assert!(seen.insert(data), "data port {data} reused");
            assert!(seen.insert(control), "control port {control} reused");
        }
        assert!(!seen.contains(&29500), "a relay took the controller's port");
    }

    #[test]
    fn a_pair_past_the_port_range_is_an_error_not_a_shared_port() {
        let err = relay_loopback_ports(65530, [3]).unwrap_err().to_string();
        assert!(
            err.contains("65535") && err.contains("first rank is 3"),
            "{err}"
        );
        assert!(relay_loopback_ports(65530, [0]).is_ok());
    }
}
