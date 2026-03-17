# LR Scheduler Composition

Demonstrates composable learning rate scheduling: linear warmup into
cosine annealing, with a plateau scheduler as reactive fallback.

Demonstrates:
- `WarmupScheduler` wrapping `CosineScheduler`
- `PlateauScheduler` for metric-driven LR reduction
- Combining schedule-based and metric-based strategies
- `optimizer.set_lr` to apply the effective rate

```bash
cargo run --example schedulers
```
