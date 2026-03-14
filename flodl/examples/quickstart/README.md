# Quickstart

Train a model in 30 lines.

Builds a small graph with a residual connection, trains it on random data
using Adam, and logs progress with the training monitor.

```sh
cargo run --example quickstart
```

## What it covers

- `FlowBuilder::from` / `through` / `also` / `build`
- `Adam` optimizer with `clip_grad_norm`
- `record_scalar` / `flush` for graph observation
- `Monitor::log(&model)` for training progress
