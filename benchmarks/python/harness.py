"""Benchmark harness for PyTorch — mirrors the Rust harness."""

import json
import resource
import statistics
import sys
import time

import torch


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"device: CUDA ({name})", file=sys.stderr)
    else:
        dev = torch.device("cpu")
        print("device: CPU", file=sys.stderr)
    print(file=sys.stderr)
    return dev


def _run_single_pass(model, batches, device, optimizer, loss_fn,
                     warmup_epochs, measured_epochs):
    """Run one pass: warmup + measured epochs. Returns (epoch_times, final_loss)."""
    # Warmup
    for _ in range(warmup_epochs):
        for x, y in batches:
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measured epochs
    epoch_times = []
    final_loss = 0.0

    for _ in range(measured_epochs):
        start = time.perf_counter()
        epoch_loss = 0.0

        for x, y in batches:
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000
        epoch_times.append(elapsed_ms)
        final_loss = epoch_loss / len(batches)

    return epoch_times, final_loss


def run_benchmark(name, model, batches, device, optimizer,
                  loss_fn=torch.nn.MSELoss(),
                  warmup_epochs=3, measured_epochs=20, runs=4):
    """Run a benchmark and return a result dict.

    Runs `runs` passes (first is warmup, rest measured). Reports the best
    (lowest median) run to filter process-level noise.
    """
    param_count = sum(p.numel() for p in model.parameters())

    measured_runs = max(runs, 2) - 1  # first run is warmup

    # Run 0: warmup run (discarded)
    print(f"    run 0/{measured_runs} (warmup)", file=sys.stderr)
    _run_single_pass(model, batches, device, optimizer, loss_fn,
                     warmup_epochs, measured_epochs)

    # Reset VRAM peak tracking AFTER warmup so steady-state is measured,
    # not cuDNN autotuning / JIT workspace spikes.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Measured runs
    best_times = None
    best_median = float("inf")
    run_medians = []
    final_loss = 0.0

    for r in range(measured_runs):
        print(f"    run {r + 1}/{measured_runs}", file=sys.stderr)
        times, loss = _run_single_pass(
            model, batches, device, optimizer, loss_fn,
            warmup_epochs, measured_epochs)
        med = statistics.median(times)
        run_medians.append(med)
        final_loss = loss

        if med < best_median:
            best_median = med
            best_times = times

    epoch_times = best_times
    sorted_times = sorted(epoch_times)
    median = statistics.median(epoch_times)
    mean = statistics.mean(epoch_times)

    # VRAM — report both allocated (active tensors) and reserved (allocator pool)
    vram_mb = None
    vram_reserved_mb = None
    if device.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        vram_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

    # RSS
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: kB

    return {
        "name": name,
        "device": f"CUDA ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "CPU",
        "runs": measured_runs,
        "warmup_epochs": warmup_epochs,
        "measured_epochs": measured_epochs,
        "batches_per_epoch": len(batches),
        "batch_size": batches[0][0].shape[0],
        "param_count": param_count,
        "epoch_times_ms": epoch_times,
        "median_epoch_ms": median,
        "mean_epoch_ms": mean,
        "min_epoch_ms": sorted_times[0],
        "max_epoch_ms": sorted_times[-1],
        "run_medians_ms": run_medians,
        "final_loss": final_loss,
        "vram_mb": vram_mb,
        "vram_reserved_mb": vram_reserved_mb,
        "rss_mb": rss_mb,
    }


def print_result(r):
    print(f"  {r['name']}", file=sys.stderr)
    print(f"    device:     {r['device']}", file=sys.stderr)
    print(f"    params:     {format_count(r['param_count'])}", file=sys.stderr)
    print(f"    batches:    {r['batches_per_epoch']} x {r['batch_size']}", file=sys.stderr)
    print(f"    runs:       1 warmup + {r['runs']} measured", file=sys.stderr)
    print(f"    epochs:     {r['warmup_epochs']} warmup + {r['measured_epochs']} measured (per run)", file=sys.stderr)
    print(f"    median:     {r['median_epoch_ms']:.1f} ms/epoch (best run)", file=sys.stderr)
    print(f"    mean:       {r['mean_epoch_ms']:.1f} ms/epoch", file=sys.stderr)
    print(f"    range:      {r['min_epoch_ms']:.1f} - {r['max_epoch_ms']:.1f} ms", file=sys.stderr)
    print(f"    final loss: {r['final_loss']:.6f}", file=sys.stderr)
    if r["vram_mb"] is not None:
        print(f"    VRAM alloc: {r['vram_mb']:.0f} MB", file=sys.stderr)
        print(f"    VRAM rsrvd: {r['vram_reserved_mb']:.0f} MB", file=sys.stderr)
    print(f"    RSS:        {r['rss_mb']:.0f} MB", file=sys.stderr)
    print(file=sys.stderr)


def format_count(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def print_summary(results):
    print("=== Summary ===", file=sys.stderr)
    print(file=sys.stderr)
    print(f"  {'benchmark':<20} {'median':>10} {'mean':>10} {'params':>10} {'alloc':>10} {'reserved':>10}", file=sys.stderr)
    print(f"  {'-' * 74}", file=sys.stderr)
    for r in results:
        alloc = f"{r['vram_mb']:.0f} MB" if r["vram_mb"] is not None else "—"
        rsrvd = f"{r['vram_reserved_mb']:.0f} MB" if r.get("vram_reserved_mb") is not None else "—"
        print(
            f"  {r['name']:<20} {r['median_epoch_ms']:>8.1f}ms {r['mean_epoch_ms']:>8.1f}ms "
            f"{format_count(r['param_count']):>10} {alloc:>10} {rsrvd:>10}",
            file=sys.stderr,
        )
    print(file=sys.stderr)
