# Cloud DDP — Communication-Efficient Distributed Training

The next DDP iteration targets **cloud and cross-datacenter** training,
where network latency (not GPU compute) dominates wall time. Current
flodl DDP is a single-host story: NCCL AllReduce over PCIe / NVLink
finishes in microseconds and `overhead_target=0.10` is trivial. On a
cross-region link, one AllReduce can cost *seconds*, so sync-every-step
becomes impractical and cadence-every-10-batches still wastes time.

## The key unlock: outer optimizer on pseudo-gradients

flodl's current `ElChe` cadence averages parameters directly (Local SGD
/ FedAvg semantics). This produces the implicit-regularization boost
already observed empirically (nccl-cadence reaches higher test accuracy
than solo on ResNet-20 CIFAR-10 at the same seed) but leaves the
communication-efficiency half on the table.

[DiLoCo (DeepMind, 2023)](https://arxiv.org/abs/2311.08105) closes that
gap with a two-level optimizer: inner AdamW for H local steps per
worker, outer Nesterov momentum running on the **parameter-space drift**
between sync rounds. The pseudo-gradient (`Δ_k = θ_global - θ_k`) is
treated as a gradient by the outer optimizer; its momentum buffer
smooths the consensus direction across rounds. This unlocks H up to
~500 without divergence, an order of magnitude wider than plain
parameter averaging tolerates.

Every round saved is one AllReduce not performed on the internet.

## Proposed API

```rust
Ddp::builder(...)
    .policy(ApplyPolicy::Cadence)
    .outer_optimizer(|| NesterovMomentum::new(lr = 0.7, mu = 0.9))
    .max_anchor(500)                    // H, widen aggressively
    .run()?;
```

Mechanically, the outer optimizer hooks in at the coordinator, between
the AllReduce (or CPU snapshot average) and the parameter broadcast
back to workers. flodl already snapshots parameters at averaging events
(CPU backend) and computes cross-worker deltas for divergence
monitoring, so the outer-optimizer state buffer is a small addition on
top.

## Design targets

- **`OuterOptimizer` trait**: `step(&mut self, global: &[Tensor], worker_deltas: &[Vec<Tensor>]) -> Vec<Tensor>`.
  Returns the new global parameters.
- **Built-ins**: `NesterovMomentum` (DiLoCo default), `OuterSgd`
  (momentum=0 baseline for ablation), `OuterAdam` (sanity-check).
- **Stateless variant**: `OuterAvg` exactly replicates today's
  weighted-AllReduce behavior so existing code is unchanged when no
  outer optimizer is set.
- **Checkpoints**: the outer-optimizer state (momentum buffer) persists
  across training runs through the existing checkpoint path.

## Cloud-specific primitives (follow-on)

Once the outer optimizer is in place, the rest of the cloud DDP stack
is additive:

- **Gradient / delta compression**: top-K sparsification, 1-bit
  quantization, error-feedback accumulators. Works cleanly on
  pseudo-gradients because the outer optimizer absorbs the quantization
  noise.
- **Asynchronous rounds**: workers submit deltas to a parameter server
  / coordinator without a global barrier. Staleness bound instead of
  lockstep. Pairs with the divergence guard to keep stragglers from
  poisoning the consensus.
- **Hierarchical ElChe**: intra-host NCCL (tight cadence) plus
  inter-host DiLoCo-style outer loop (loose cadence). One heterogeneous
  cluster, two levels of sync policy.
- **Byzantine-tolerant aggregation** (optional): trimmed mean / median
  instead of weighted mean for untrusted workers (federated / open-
  contribution training).
- **Bandwidth-aware scheduling**: measure link latency per worker pair
  (we already measure per-worker compute throughput for ElChe, same
  mechanism), let `ApplyPolicy::Cadence` anchor on the slowest *link*
  instead of the slowest GPU when bandwidth is the binding constraint.

## Why this matters for flodl

Most DDP framework docs sell "faster multi-GPU." The empirical data
already gathered shows flodl's cadence mode is *already* winning on
generalization at the single-host scale, which is the same mechanism
frontier labs are reaching for at the trillion-parameter scale with
DiLoCo and its descendants. Closing the outer-optimizer gap positions
flodl's `ElChe` as the complete pattern: heterogeneous-hardware-aware,
network-efficient, generalization-improving DDP out of the box.
