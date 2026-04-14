# flodl Roadmap

flodl is developed one thing at a time. Whatever has `[started]` is the
current focus; everything else is a candidate for next. The pull
happens when the current item ships, with the priority chosen against
what's known *that day*, not what was written when the line was added.

For the historical record of shipped phases and individual changes, see
[CHANGELOG.md](CHANGELOG.md). For the project philosophy, see
[the trajectory thesis](https://flodl.dev/thesis).

---

## Shipped

- **Phase 1..11** complete: tensor + autograd + nn + graph + data + multi-GPU DDP
- **ResNet-20 200e** validated across all DDP modes on a heterogeneous
  RTX 5060 Ti + GTX 1060 box. NCCL cadence reaches 0.9242 in 44m10s,
  beating the solo-1 baseline (0.918 in 61m42s) on both axes.
- **AccumulateGrad stream fix**: three undocumented libtorch facts
  resolved, no more "stream does not match" warning on DDP backward.
- **NCCL init-on-main + split**: dead worker recovery without CUDA
  context corruption.
- **ElChe cadence battle-tested** across six architecture families
  (logistic, MLP, LeNet, char-RNN, GPT-nano, conv autoencoder) plus
  ResNet-20 with deep residuals.

See [CHANGELOG.md](CHANGELOG.md) for the full per-version detail.

---

## In progress

- **[started] fdl maturity pass**: `flodl-args` crate + `--fdl-schema`
  contract + multi-env overlay. Landed first because it unblocks both
  Phase 12 (needs `fdl.cloud.yml`) and ddp-bench expansion (needs
  schema-driven help). See
  [docs/design/run-config.md](docs/design/run-config.md) for the full
  spec.

(Length-1 by design. The next item pulls from Possibilities when this
ships.)

---

## Possibilities

Unordered. Any line can become the next In Progress. Adding a line is
not a commitment; only moving one to In Progress is.

- **Phase 12 Cloud DDP**: `OuterOptimizer` trait, DiLoCo baseline,
  hierarchical ElChe (intra-host NCCL + inter-host DiLoCo). Real
  feature PR with design doc and cloud test budget. See
  [docs/design/cloud-ddp.md](docs/design/cloud-ddp.md).
- **ddp-bench: next published model**. Class-level pull, repeats as
  models land. Top candidates: small transformer (GPT-2 tiny / BERT-
  small) for the attention family, ViT, UNet for multi-scale skip
  connections, MoE as a routing pressure test, PPO for the RL data
  loop.
- **JEPA exploration**, with BYOL or MoCo as a stepping stone. The
  EMA target encoder + two-tower training infra needed for I-JEPA /
  V-JEPA is also what BYOL needs, so it's a natural progression.
- **Zero-dispatch training**: full train-step CUDA Graph capture
  (forward + backward + optimizer as one replay) plus double-buffered
  static I/O tensors. Resident and streaming DataLoader modes already
  ship; this is the Graph integration layer on top. See
  [docs/design/resident-training.md](docs/design/resident-training.md).
- **Graph serialization**: save/load graph topology, ONNX import/export.
- **Model parallelism**: tensor / pipeline parallelism for models that
  exceed single-GPU VRAM.
- **Higher-order gradients**: differentiate through backward.

---

## Considered and set aside

(Ideas that were on the table and consciously deferred. Not blocked,
just not currently a priority. Easy to reactivate if context shifts.)

- *(empty for now)*

---

## How this list moves

- A Possibility moves to `In Progress` only when `In Progress` is empty.
  The decision is made fresh each time, with whatever is true that day.
- Shipped items get a one-line entry here when notable; routine work is
  in CHANGELOG only.
- Ideas that turn out to be the wrong priority move to `Considered and
  set aside` rather than being silently deleted. They can move back if
  the context changes.
- No dates. Ordinal commitments only. Slippage shows up as "took
  longer than expected" rather than "broken promise."
