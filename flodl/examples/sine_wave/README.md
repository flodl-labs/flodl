# Sine Wave Regression

Trains a small network to learn sin(x) on [-2pi, 2pi], prints a
prediction-vs-actual comparison table, and verifies checkpoint save/load
round-trips.

```sh
cargo run --example sine_wave
```

## What it covers

- `FlowBuilder` with `also` (residual connection) and `LayerNorm`
- `Adam` optimizer with `CosineScheduler`
- `record_scalar` / `flush` with `Monitor::log(&model)`
- `no_grad` for evaluation
- `save_parameters_file` / `load_parameters_file` checkpoint round-trip
