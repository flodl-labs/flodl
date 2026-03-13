# Tutorial 9: Training Monitor

The training monitor provides human-readable progress output, system resource
tracking, and an optional live web dashboard — all with zero external
dependencies.

> **Prerequisites**: [Training](04-training.md) covers the training loop.
> [Utilities](08-utilities.md) covers observation and trends.

## Basic Usage

The monitor wraps your training loop with timing, ETA, and resource sampling:

```rust
use flodl::*;
use flodl::monitor::Monitor;

let model = FlowBuilder::from(Linear::new(2, 16)?)
    .through(GELU)
    .through(Linear::new(16, 2)?)
    .build()?;

let params = model.parameters();
let mut optimizer = Adam::new(&params, 0.01);
model.set_training(true);

let num_epochs = 100;
let mut monitor = Monitor::new(num_epochs);

for epoch in 0..num_epochs {
    let t = std::time::Instant::now();
    let mut epoch_loss = 0.0;

    for (input_t, target_t) in &batches {
        let input = Variable::new(input_t.clone(), true);
        let target = Variable::new(target_t.clone(), false);
        let pred = model.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;

        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;
        epoch_loss += loss.item()?;
    }

    let avg_loss = epoch_loss / batches.len() as f64;
    monitor.log(epoch, t.elapsed(), &[("loss", avg_loss)]);
}

monitor.finish();
```

Each `log` call prints a one-liner to stderr:

```
  epoch   1/100  loss=1.5264  [49ms  ETA 4.8s]
  epoch   2/100  loss=1.1020  [28ms  ETA 3.6s]
  epoch  50/100  loss=0.0023  [24ms  ETA 1.2s]  VRAM: 2.1/6.0 GB (82%)
  epoch 100/100  loss=0.0012  [23ms]
  training complete in 2.8s  | loss: 0.0012
```

The ETA adapts its format automatically: `3h 12m`, `4m 32s`, `12s`, `420ms`.

GPU metrics (VRAM usage and utilization) appear automatically when CUDA is
available. On CPU-only builds they are silently omitted.

## Multiple Metrics

Pass any number of `(name, value)` pairs to `log`:

```rust
monitor.log(epoch, t.elapsed(), &[
    ("loss", avg_loss),
    ("lr", scheduler.lr(epoch)),
    ("grad_norm", norm),
]);
// epoch  42/100  loss=0.0023  lr=0.0008  grad_norm=0.4521  [1.2s  ETA 1m 10s]
```

## Live Dashboard

Start an embedded HTTP server to get a real-time web dashboard:

```rust
let mut monitor = Monitor::new(num_epochs);
monitor.serve(3000)?;  // http://localhost:3000
```

Open `http://localhost:3000` in a browser. The dashboard shows:

- **Header**: epoch counter, progress bar, ETA, elapsed time
- **Training metrics chart**: live-updating canvas chart of all logged metrics
- **Resource chart**: CPU%, GPU%, RAM%, VRAM% over time
- **Resource bars**: current values with percentage fill
- **Epoch log table**: all epochs, newest first
- **Graph SVG**: collapsible architecture diagram (if provided)

### How it works

The server uses raw TCP sockets and [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-Sent_Events) (SSE) — no HTTP framework, no WebSocket library, no JavaScript dependencies.

1. `monitor.serve(port)` spawns a background listener thread
2. Each browser connection gets its own handler thread
3. `GET /` serves the dashboard HTML (all JS/CSS inline, ~16 KB)
4. `GET /events` holds the connection open as an SSE stream
5. Each `monitor.log(...)` pushes a JSON event to all connected clients
6. The browser JS updates the charts and log in real time

### Late join

If you open the dashboard mid-training, it catches up instantly. The SSE
handler replays all past epoch events before switching to live streaming.

### Embedding the graph

To show the graph architecture in the dashboard:

```rust
let mut monitor = Monitor::new(num_epochs);
monitor.serve(3000)?;
monitor.watch(&model);  // generates SVG, sends to dashboard
```

The SVG appears in a collapsible section at the bottom of the dashboard.

For a timing-annotated heat map (green/yellow/red by execution time), enable
profiling during training and use `finish_with`:

```rust
model.enable_profiling();

for epoch in 0..num_epochs {
    // ... training (profiling records timing on each forward pass) ...
}

monitor.finish_with(&model);  // final SVG with steady-state timing heat map
```

`finish_with` generates the profiled SVG at the end of training — when the
last forward pass timing is representative of steady-state performance. The
heat map is pushed to the live dashboard and baked into the HTML archive.

You can also update the SVG mid-training with `watch_profiled(&model)`.

Both methods require Graphviz (`dot`) to be installed. If `dot` is not
available, they silently fall back or do nothing.

## Resource Tracking

The monitor samples system resources on every `log` call:

| Metric | Source | When available |
|--------|--------|----------------|
| CPU % | `/proc/stat` (delta) | Linux |
| RAM used/total | `/proc/meminfo` | Linux |
| GPU utilization % | NVML (dynamic load) | NVIDIA GPU + driver |
| VRAM used/total | `cudaMemGetInfo` | CUDA feature enabled |

Resources that aren't available are silently omitted from both the terminal
output and the dashboard.

### Accessing resource data

```rust
for record in monitor.history() {
    if let Some(vram) = record.resources.vram_used_bytes {
        println!("epoch {}: VRAM {} bytes", record.epoch, vram);
    }
}
```

## Export

### Dashboard archive

Save the full dashboard as a self-contained HTML file — all charts, resource
graphs, epoch log, and graph SVG baked in. Open it in any browser, no server.

```rust
monitor.save_html("training_report.html");  // set before training
// ... training loop ...
monitor.finish();  // writes the archive
```

The archive is written automatically when `finish()` is called. It's the same
dashboard you see live, but frozen at the final state with all data pre-loaded.

### Training log

```rust
monitor.write_log("training.log")?;
```

Produces:

```
# flodl training log
epoch   1/100  loss=1.5264  [49ms]
epoch   2/100  loss=1.1020  [28ms]
...
# total: 2.8s
```

### CSV

```rust
monitor.export_csv("training.csv")?;
```

Produces:

```csv
epoch,duration_s,loss,cpu_pct,ram_used,gpu_pct,vram_used
1,0.049,1.5264,45.2,3221225472,82.0,2254857830
2,0.028,1.1020,43.8,3221225472,81.5,2254857830
...
```

## Monitor vs. Graph Observation

The monitor and the graph's built-in observation system serve different roles:

| | Graph (`collect`/`flush`/`trend`) | Monitor |
|---|---|---|
| **Scope** | Per-node metrics within the graph | Whole training loop |
| **Resources** | No | CPU, RAM, GPU, VRAM |
| **ETA** | `g.eta(total)` (basic) | Adaptive formatting |
| **Dashboard** | Static HTML (`plot_html`) | Live web page |
| **Trend analysis** | `slope`, `stalled`, `converged` | Raw history |
| **Training decisions** | Yes (early stopping, LR decay) | No |

They complement each other. Use the graph's observation for metrics that
drive training decisions (loss plateau detection, convergence checks). Use
the monitor for human-facing output and system health.

### Using both together

```rust
let mut monitor = Monitor::new(num_epochs);
monitor.serve(3000)?;
monitor.watch(&model);

for epoch in 0..num_epochs {
    let t = std::time::Instant::now();

    for (input, target) in &batches {
        let pred = model.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;

        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;

        model.record_scalar("loss", loss.item()?);
    }
    model.flush(&["loss"]);

    // Graph trends drive training decisions
    if model.trend("loss").stalled(10, 1e-4) {
        optimizer.set_lr(scheduler.lr(epoch));
    }

    // Monitor handles display and resource tracking
    let loss_val = model.trend("loss").latest().unwrap_or(0.0);
    monitor.log(epoch, t.elapsed(), &[
        ("loss", loss_val),
        ("lr", scheduler.lr(epoch)),
    ]);
}

monitor.finish_with(&model);  // final SVG with profiling heat map
```

## Complete Example

See [`flodl/examples/quickstart.rs`](../../flodl/examples/quickstart.rs) for
a runnable example with the monitor.

---

Previous tutorials: [08-Utilities](08-utilities.md) |
[07-Visualization](07-visualization.md) |
[06-Advanced Graphs](06-advanced-graphs.md) |
[05-Graph Builder](05-graph-builder.md) |
[04-Training](04-training.md) |
[03-Modules](03-modules.md) |
[02-Autograd](02-autograd.md) |
[01-Tensors](01-tensors.md)
