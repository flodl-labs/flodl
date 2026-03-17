# Transfer Learning

Trains an encoder, saves a checkpoint, loads it into a different architecture,
freezes the pretrained layers, and fine-tunes the new head.

Demonstrates:
- `save_checkpoint_file` / `load_checkpoint_file` with partial matching
- `LoadReport` showing loaded vs skipped parameters
- `Parameter::freeze` / `unfreeze` for layer freezing
- Progressive unfreezing (head first, then full model)

```bash
cargo run --example transfer_learning
```
