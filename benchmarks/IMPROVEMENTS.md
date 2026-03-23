# Benchmark Improvements

Review of the benchmark suite with specific issues found and actionable fixes.
Each section includes the problem, evidence, and what to change.

---

## 1. Add CUDA Graph benchmarks

None of the current benchmarks use CUDA Graphs. Since this feature landed in
flodl, the benchmarks should demonstrate it — especially on the static-graph
workloads where it eliminates CPU dispatch overhead entirely.

**What to do:**

Add a `tier1/mlp_cuda_graph.rs` (and matching Python `torch.cuda.CUDAGraph`
equivalent) that wraps the existing MLP training step in a CUDA Graph
capture/replay cycle:

```
// Pseudocode for the Rust side:
// 1. Run one warmup forward+backward+step (to allocate all tensors)
// 2. Capture: begin capture → forward → loss → backward → step → end capture
// 3. Measured epochs: just replay the captured graph, swapping input data
```

The Python equivalent uses `torch.cuda.CUDAGraph` with `torch.cuda.graph()`:

```python
# Warmup
pred = model(static_x)
loss = loss_fn(pred, static_y)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    pred = model(static_x)
    loss = loss_fn(pred, static_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Replay
for batch in batches:
    static_x.copy_(batch_x)
    static_y.copy_(batch_y)
    g.replay()
```

Also add `tier2/residual_tower_cuda_graph.rs` — residual tower is the best
tier 2 candidate (fully static, no dynamic control flow, already -26% without
CUDA Graphs).

This is the single highest-impact addition. GPU utilization on the
dispatch-bound benchmarks (feedback_loop 29%, gated_routing 39%, gru_seq 40%)
shows the GPU is starving for work — CUDA Graphs eliminate that starvation for
static workloads.

---

## 2. Track iteration count in feedback_loop

The feedback loop uses `until_cond` (flodl) / `halt_signal.item() > 0` (Python)
with `MAX_ITER=10`. After warmup training, the halt network may have learned
to fire at different iteration counts on each side. If Python halts after 2
iterations but flodl after 8, the +116% delta is measuring different workloads,
not framework overhead.

**What to do:**

Rust (`feedback_loop.rs`): track average iterations per forward pass. If the
graph exposes loop iteration count via a tag or trace, read it. Otherwise,
add a counter.

Python (`feedback_loop.py`): count iterations in the forward loop:

```python
def forward(self, x):
    x = self.encoder(x)
    iters = 0
    for _ in range(MAX_ITER):
        x = self.refine(x)
        iters += 1
        halt_signal = self.halt(x)
        if halt_signal.item() > 0:
            break
    self._last_iters = iters  # expose for logging
    return self.decoder(x)
```

Log average iterations per epoch to stderr on both sides. If they differ
significantly, the benchmark is not measuring what we think it's measuring.

**Alternative:** use `for_n(MAX_ITER)` instead of `until_cond` in the flodl
benchmark to force both sides to always run the full 10 iterations. This makes
the comparison fair (same compute) at the cost of not testing adaptive halt.
Could offer both variants: `feedback_loop` (adaptive) and `feedback_loop_fixed`
(fixed 10 iterations).

---

## 3. Fix GRU seq narrow+reshape overhead

In `gru_seq.rs`, each timestep does:

```rust
let x_t = input.narrow(1, t, 1)?.reshape(&[batch, INPUT_DIM])?;
```

That's 2 tensor ops per timestep. Python does:

```python
h = self.gru(x[:, t, :], h)
```

Single indexing operation. Over 50 timesteps × 50 batches × 20 epochs = 50,000
extra ops.

**What to do:**

Replace `narrow + reshape` with a single `select` operation:

```rust
let x_t = input.select(1, t)?;  // [B, INPUT_DIM] — no reshape needed
```

If `Variable::select` doesn't exist yet, `input.narrow(1, t, 1)?.squeeze(1)?`
should also work and is closer to what Python's indexing does (returns a view
without the singleton dimension).

Also verify that Python isn't secretly using `nn.GRU` (sequence-level cuDNN)
instead of `nn.GRUCell`. Currently both use cell-level unrolling, which is
correct for a fair comparison. But consider adding a separate `gru_seq_cudnn`
benchmark that uses `nn.GRU` on the Python side to show the gap — this
motivates the sequence-level GRU feature on the flodl roadmap.

---

## 4. Reduce halt check overhead in feedback_loop

The `until_cond` halt evaluation requires a GPU→CPU sync to read the scalar
halt signal. With a small loop body (one Linear + GELU + LayerNorm), this
sync dominates the iteration time.

Python has the exact same problem (`halt_signal.item()` syncs), but Python's
per-iteration overhead is lower because it's a direct `for` loop calling
modules, while flodl's graph engine does state setup, tag capture, and
condition evaluation per iteration.

**What to do:**

Profile the graph engine's per-iteration overhead in `until_cond` loops. The
overhead should be negligible compared to the compute — if it's not, the loop
hot path needs optimization. Specifically check:

- Is `tagged_outputs` being written/read every iteration?
- Is there unnecessary state cloning in the loop body?
- Can the halt condition check be batched or deferred?

If the overhead is inherent to the graph engine's generality, consider a
fast-path for simple `until_cond` loops where the body is a single Module
(not a full Graph with tags/observation).

---

## 5. Verify VRAM measurement accuracy

The user reported that previous runs showed incorrect VRAM numbers for flodl.
The current harness looks correct:

- `cuda_empty_cache()` + `cuda_reset_peak_stats()` before benchmark
- `cuda_peak_active_bytes()` / `cuda_peak_reserved_bytes()` after all runs

Both call into the same libtorch caching allocator that PyTorch uses. But
verify that:

1. `cuda_reset_peak_stats()` actually resets the stats (check the FFI function
   calls the right C++ method: `c10::cuda::CUDACachingAllocator::resetPeakStats()`).
2. `cuda_peak_active_bytes()` reads `StatType::AGGREGATE` / `Stat::peak` for
   the allocated stat (not reserved).
3. The numbers match when running an identical model on both sides. A simple
   test: create a known tensor of size N on GPU, read peak active bytes,
   verify it matches N × sizeof(float).

---

## 6. Add a fixed-iteration feedback loop variant

To isolate graph engine overhead from adaptive halt behavior, add
`feedback_loop_fixed` that uses `for_n(10)` instead of `until_cond`:

```rust
// flodl
.loop_body(loop_body)
.for_n(MAX_ITER)
```

```python
# Python
for _ in range(MAX_ITER):
    x = self.refine(x)
# no halt check, no .item() sync
```

This removes the CUDA sync per iteration on both sides and ensures identical
compute. The delta measures pure framework overhead for looping.

---

## 7. Consider per-benchmark VRAM reset

Currently, VRAM peak stats are reset once at the start of each benchmark, then
read after all runs complete. The peak therefore includes the warmup run
(which is discarded for timing). If the warmup run triggers higher peak
allocation (e.g., JIT compilation buffers, cuDNN workspace autotuning), the
VRAM numbers might not reflect steady-state usage.

**What to do:**

Reset peak stats after the warmup run, before the measured runs. This way VRAM
numbers reflect steady-state allocation only:

```rust
// Warmup run
run_single_pass(config, &mut run_epoch)?;

// Reset VRAM after warmup (steady-state measurement)
if flodl::cuda_available() {
    flodl::cuda_reset_peak_stats();
}

// Measured runs
for r in 0..measured_runs { ... }
```

Same in Python:

```python
# Warmup run
_run_single_pass(...)

# Reset after warmup
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

# Measured runs
for r in range(measured_runs): ...
```
