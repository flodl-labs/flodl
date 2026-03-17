# Benchmark: flodl vs PyTorch

Head-to-head comparison on a real training workload — same model, same data,
same hardware, same hyperparameters. No synthetic micro-benchmarks.

## Results

| Metric | PyTorch 2.5.1 | flodl | Delta |
|--------|--------------|-------|-------|
| Avg epoch | 50.1s | 42.1s | **-16%** |
| Total | 83m 34s | 70m 14s | **-16%** |
| GPU utilization | ~80% (spiky) | 88-92% (flat) | more stable |
| VRAM (device) | 2,805 MB | 2,977 MB | +6%* |
| RAM (RSS peak) | 2,091 MB | 2,345 MB | +12%* |
| Startup | minutes | seconds | - |
| Live dashboard | no | yes | - |

\* Higher memory on the Rust side is from libtorch static linking, the monitor
thread, and gzip checkpoint compression buffers. The GPU computation itself
dispatches identical CUDA kernels.

## What was measured

**Model:** FBRL letter recognition — a non-trivial recurrent attention
architecture with 1 scan + 6 read glimpses, GRU controller, CNN encoder,
deconv decoder, and a 9-component loss stack (cross-entropy, MSE, BCE,
attention guide, diversity, void repulsion, recode, content).

**Data:** 11,440 grayscale 128x128 letter images (11 fonts, A-Z upper+lower).

**Hardware:** NVIDIA GeForce GTX 1060 6GB (Pascal), single GPU.

**Config (identical both sides):**

| Parameter | Value |
|-----------|-------|
| latent_dim | 256 |
| batch_size | 52 |
| epochs | 100 |
| optimizer | Adam (lr=0.001) |
| scheduler | Cosine annealing (min_lr=0.0) |
| max_grad_norm | 5.0 |
| checkpoint | every 50 epochs |

Both runs converge to the same final metrics: 100% letter accuracy, 100% case
accuracy, ~0.0038 reconstruction MSE.

## Asymmetries (honest accounting)

The two implementations are not byte-identical. These are the differences,
and why they make the flodl numbers *more* impressive, not less:

| Aspect | PyTorch | flodl |
|--------|---------|-------|
| Checkpoints | synchronous, flat `.pth` (134 MB) | async, gzip `.fdl.gz` (124 MB) |
| Monitor | none | live dashboard thread (SSE server + resource sampling) |
| Autograd | Python dispatch | libtorch native autograd |
| Optimizer | Python Adam | batched Adam (single C++ call) |
| Data loading | Python DataLoader with shuffle | native PNG decode, pre-batched in memory |

The async checkpoint design is the one asymmetry a skeptic might question.
The rebuttal: flodl is doing **more work** per checkpoint (gzip compression)
and also running a live dashboard thread that PyTorch doesn't have. The
checkpoint sizes differ by only 10 MB — the I/O savings from async are
minimal. The async design doesn't skip work; it just doesn't block the
training loop while doing it.

## Epoch time distribution

flodl's epoch times are tighter, especially in the second half:

| | PyTorch | flodl |
|---|---------|-------|
| Epoch 1 (cold) | 51.7s | 48.2s |
| Epochs 80-100 range | 49.1 - 50.8s | 41.3 - 41.6s |
| Std dev (last 20) | ~0.5s | ~0.1s |

The flat GPU utilization (88-92%) vs PyTorch's spiky 80% explains the
consistency — less host-side stalling between ops.

## Raw data and reproduction

All artifacts are committed in the
[fbrl](https://github.com/fab2s/fbrl) repository at commit
[`102225b`](https://github.com/fab2s/fbrl/tree/102225b):

| Side | Path | Contents |
|------|------|----------|
| flodl | [`letter/runs/v1-benchmark/`](https://github.com/fab2s/fbrl/tree/102225b/letter/runs/v1-benchmark/) | benchmark.json, training.md, training.log, training.csv, training_pseudocode.md, dashboard.html, manifest.json, model |
| PyTorch | [`python/runs/letters/v7-benchmark/`](https://github.com/fab2s/fbrl/tree/102225b/python/runs/letters/v7-benchmark/) | benchmark.json, training.md, training.log, training_pseudocode.md, config.yaml, info.txt, training_metrics.png, model |

The `training_pseudocode.md` files show the training loop side-by-side —
identical algorithm, identical loss weights, identical hyperparameters. The
`training.md` files include the exact commands used to run each benchmark.

The `benchmark.json` files contain per-epoch timings (100 data points each),
VRAM/RAM measurements, and full config — machine-parseable for independent
analysis.

## Reproduce

```bash
git clone https://github.com/fab2s/fbrl.git
cd fbrl
git checkout 102225b
```

**PyTorch:**
```bash
cd python
make generate                                                    # training data
make train CONFIG=runs/letters/v7-void-repulsion/config.yaml CKPT=50
```

**flodl:**
```bash
cd python && make generate && cd ..                              # shared data
make train-letter DATA=../python/data/letters EPOCHS=100 SAVE=training MONITOR=3000
```

See each run's `training.md` for full details and environment info.
