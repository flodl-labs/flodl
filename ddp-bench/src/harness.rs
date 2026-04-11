//! Training harness: ties Timeline + Monitor + DDP together for each (model, mode) combo.

use std::sync::Arc;
use std::time::{Duration, Instant};

use flodl::autograd::Variable;
use flodl::distributed::{ApplyPolicy, AverageBackend, Ddp, DdpConfig};
use flodl::monitor::{Monitor, Timeline};
use flodl::nn::{Adam, Module, Optimizer, Parameter};
use flodl::tensor::{Device, Result, Tensor, TensorError};

use crate::config::{DdpMode, RunConfig};
use crate::models::ModelDef;

/// Result of a single (model, mode) benchmark run.
#[derive(Clone)]
pub struct RunResult {
    pub model_name: String,
    pub mode: String,
    pub final_loss: f64,
    pub epoch_times_ms: Vec<f64>,
    pub total_ms: f64,
    pub timeline_summary: flodl::monitor::TimelineSummary,
    /// Training config for baseline generation.
    pub epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
}

/// Run a single (model, mode) combination.
pub fn run_combo(model_def: &ModelDef, mode: &DdpMode, config: &RunConfig) -> Result<RunResult> {
    let mode_str = mode.to_string();
    let run_dir = format!("{}/{}/{}", config.output_dir, model_def.name, mode_str);
    std::fs::create_dir_all(&run_dir)
        .map_err(|e| TensorError::new(&format!("failed to create {run_dir}: {e}")))?;

    let lr_note = if (config.lr - model_def.defaults.lr).abs() > 1e-10 {
        format!(", lr={:.1e} ({:.2}x)", config.lr, config.lr / model_def.defaults.lr)
    } else {
        String::new()
    };
    eprintln!(
        "\n=== {} / {} ({} epochs, {} batches x {}{}) ===",
        model_def.name, mode_str, config.epochs, config.batches_per_epoch, config.batch_size, lr_note,
    );

    // Create dataset.  Virtual length = batches * batch_size so DataLoader
    // sees the right epoch size.  Physical pool is much smaller (POOL_MUL x
    // batch_size); get_batch wraps indices via modulo.
    let load_start = Instant::now();
    let virtual_len = config.batches_per_epoch * config.batch_size;
    let pool_size = (config.batch_size * crate::data::POOL_MUL).min(virtual_len);
    let dataset = (model_def.dataset)(config.seed, virtual_len, pool_size)?;
    let load_ms = load_start.elapsed().as_millis();
    let preload_tag = if mode.requires_multi_gpu() { "cpu" } else { "gpu-preload" };
    eprintln!("  data: pool={pool_size}, virtual={virtual_len}, mode={preload_tag} ({load_ms}ms)");

    // Start timeline AFTER data loading so measurements reflect training only.
    let timeline = Timeline::new(100);
    timeline.start();

    // Create monitor
    let mut monitor = Monitor::new(config.epochs);
    if let Some(port) = config.monitor_port {
        monitor
            .serve(port)
            .map_err(|e| TensorError::new(&format!("monitor serve: {e}")))?;
    }

    let start = Instant::now();
    let result = match mode {
        DdpMode::Solo(gpu_idx) => run_solo(
            model_def,
            *gpu_idx,
            dataset,
            config,
            &timeline,
            &mut monitor,
        ),
        DdpMode::Sync => run_sync(model_def, dataset, config, &timeline, &mut monitor),
        DdpMode::Builder { policy, backend } => run_builder(
            model_def,
            *policy,
            *backend,
            dataset,
            config,
            &timeline,
            &mut monitor,
        ),
    };
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;

    timeline.stop();

    // Save artifacts
    let _ = timeline.save_json(&format!("{run_dir}/timeline.json"));
    let _ = timeline.save_csv(&format!("{run_dir}/timeline.csv"));
    let _ = timeline.save_html(&format!("{run_dir}/timeline.html"));
    monitor.finish();

    let (final_loss, epoch_times) = result?;

    // Clean up CUDA state between runs. NCCL communicators and cached
    // allocator blocks from the previous run can fragment VRAM or leave
    // stale stream state that interferes with the next NCCL init.
    #[cfg(feature = "cuda")]
    {
        let gpu_count = flodl::tensor::cuda_device_count();
        for i in 0..gpu_count {
            flodl::tensor::cuda_synchronize(i as u8);
            flodl::tensor::cuda_empty_cache();
        }
    }

    let summary = timeline.summary();
    eprintln!(
        "  done: loss={:.6}, total={:.1}s, syncs={}, idle=[{}]",
        final_loss,
        total_ms / 1000.0,
        summary.sync_count,
        summary
            .gpu_idle_pct
            .iter()
            .enumerate()
            .map(|(i, p)| format!("gpu{i}:{p:.1}%"))
            .collect::<Vec<_>>()
            .join(", "),
    );

    Ok(RunResult {
        model_name: model_def.name.to_string(),
        mode: mode_str,
        final_loss,
        epoch_times_ms: epoch_times,
        total_ms,
        timeline_summary: summary,
        epochs: config.epochs,
        batches_per_epoch: config.batches_per_epoch,
        batch_size: config.batch_size,
    })
}

/// Preload a small pool of batches to GPU.  Returns `POOL_MUL` batches;
/// the training loop cycles through them via `batch_idx % pool.len()`.
fn preload_gpu_batches(
    dataset: &dyn flodl::data::BatchDataSet,
    device: Device,
    batch_size: usize,
) -> Result<Vec<Vec<Tensor>>> {
    let n = crate::data::POOL_MUL;
    let mut pool = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * batch_size;
        let indices: Vec<usize> = (start..start + batch_size).collect();
        let batch = dataset.get_batch(&indices)?;
        let gpu_batch: Vec<Tensor> = batch
            .into_iter()
            .map(|t| t.to_device(device))
            .collect::<Result<Vec<_>>>()?;
        pool.push(gpu_batch);
    }
    Ok(pool)
}

/// Solo GPU: no DDP, standard training loop.
fn run_solo(
    model_def: &ModelDef,
    gpu_idx: usize,
    dataset: Arc<dyn flodl::data::BatchDataSet>,
    config: &RunConfig,
    timeline: &Arc<Timeline>,
    monitor: &mut Monitor,
) -> Result<(f64, Vec<f64>)> {
    let device = Device::CUDA(gpu_idx as u8);
    let model = (model_def.build)(device)?;
    let params = model.parameters();
    let mut optimizer = Adam::new(&params, config.lr);
    model.train();

    // Preload batches to GPU -- training loop has zero data overhead.
    let gpu_pool = preload_gpu_batches(dataset.as_ref(), device, config.batch_size)?;
    let pool_len = gpu_pool.len();

    let mut epoch_times = Vec::with_capacity(config.epochs);
    let mut final_loss = 0.0;

    for epoch in 0..config.epochs {
        timeline.event(flodl::monitor::EventKind::EpochStart { epoch });
        let epoch_start = Instant::now();
        let mut total_loss = 0.0;

        for batch_idx in 0..config.batches_per_epoch {
            let batch = &gpu_pool[batch_idx % pool_len];

            let loss = (model_def.train_fn)(model.as_ref(), batch)?;
            let loss_val = loss.item()?;

            optimizer.zero_grad();
            loss.backward()?;
            optimizer.step()?;

            total_loss += loss_val;
        }

        let epoch_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;
        final_loss = total_loss / config.batches_per_epoch as f64;
        timeline.event(flodl::monitor::EventKind::EpochEnd {
            epoch,
            loss: final_loss,
        });

        monitor.log(epoch, epoch_start.elapsed(), &[("loss", final_loss)]);
        epoch_times.push(epoch_ms);
    }

    Ok((final_loss, epoch_times))
}

/// Sync mode: Ddp::setup_with() with Graph.
fn run_sync(
    model_def: &ModelDef,
    dataset: Arc<dyn flodl::data::BatchDataSet>,
    config: &RunConfig,
    timeline: &Arc<Timeline>,
    monitor: &mut Monitor,
) -> Result<(f64, Vec<f64>)> {
    // Build as Graph (the build fn returns Box<dyn Module>, we need Graph)
    let device = Device::CUDA(0);
    let model = (model_def.build)(device)?;

    // Try to get the Graph reference for setup
    let graph = model
        .as_graph()
        .ok_or_else(|| TensorError::new("sync mode requires a Graph-based model"))?;

    let build_fn = model_def.build;
    let lr = config.lr;

    Ddp::setup_with(
        graph,
        move |dev| -> Result<Box<dyn Module>> { build_fn(dev) },
        move |params: &[Parameter]| Adam::new(params, lr),
        DdpConfig::new().timeline(Arc::clone(timeline)),
    )?;

    monitor.watch(graph);

    // Preload batches to GPU -- training loop has zero data overhead.
    let gpu_pool = preload_gpu_batches(dataset.as_ref(), device, config.batch_size)?;
    let pool_len = gpu_pool.len();

    let mut epoch_times = Vec::with_capacity(config.epochs);
    let mut final_loss = 0.0;

    for epoch in 0..config.epochs {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0;

        for batch_idx in 0..config.batches_per_epoch {
            let batch = &gpu_pool[batch_idx % pool_len];

            let loss = (model_def.train_fn)(graph, batch)?;
            let loss_val = loss.item()?;

            loss.backward()?;
            graph.step()?;

            total_loss += loss_val;
        }

        let epoch_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;
        final_loss = total_loss / config.batches_per_epoch as f64;

        graph.record_scalar("loss", final_loss);
        graph.flush(&[]);
        monitor.log(epoch, epoch_start.elapsed(), graph);
        epoch_times.push(epoch_ms);
    }

    Ok((final_loss, epoch_times))
}

/// Builder mode: Ddp::builder() with thread-per-GPU.
fn run_builder(
    model_def: &ModelDef,
    policy: ApplyPolicy,
    backend: AverageBackend,
    dataset: Arc<dyn flodl::data::BatchDataSet>,
    config: &RunConfig,
    timeline: &Arc<Timeline>,
    monitor: &mut Monitor,
) -> Result<(f64, Vec<f64>)> {
    let build_fn = model_def.build;
    let train_fn_ptr = model_def.train_fn;
    let lr = config.lr;

    let handle = Ddp::builder(
        move |dev| build_fn(dev),
        move |params: &[Parameter]| Adam::new(params, lr),
        move |model: &Box<dyn Module>, batch: &[Tensor]| -> Result<Variable> {
            train_fn_ptr(model.as_ref(), batch)
        },
    )
    .dataset(dataset)
    .batch_size(config.batch_size)
    .num_epochs(config.epochs)
    .policy(policy)
    .backend(backend)
    .timeline(Arc::clone(timeline))
    .run()?;

    let mut epoch_times = Vec::new();
    let mut final_loss = 0.0;

    while let Some(metrics) = handle.next_metrics() {
        final_loss = metrics.avg_loss;
        epoch_times.push(metrics.epoch_ms);
        monitor.log(
            metrics.epoch,
            Duration::from_millis(metrics.epoch_ms as u64),
            &metrics,
        );
    }

    let _state = handle.join()?;
    Ok((final_loss, epoch_times))
}
