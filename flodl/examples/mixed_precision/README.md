# Mixed Precision Training

Trains a model with float16 parameters and `GradScaler` for dynamic loss scaling.

Demonstrates:
- `cast_parameters` to switch parameter dtype
- `GradScaler::scale` before backward
- `GradScaler::step` with inf/nan detection
- `GradScaler::update` for dynamic scale adjustment
- Casting back to float32 for export

```bash
cargo run --example mixed_precision
```
