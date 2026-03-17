# Observation and Early Stopping

Full observation workflow: tag graph nodes, collect per-batch metrics,
flush to epoch history, and query trends for early stopping.

Demonstrates:
- `graph.collect` / `graph.flush` for metric collection
- `graph.record_scalar` for external metrics
- `graph.trend(tag)` — `slope`, `stalled`, `improving`, `converged`
- `graph.trends(tags)` — `all_improving`, `any_stalled`, `mean_slope`
- Early stopping driven by convergence detection

```bash
cargo run --example observation
```
