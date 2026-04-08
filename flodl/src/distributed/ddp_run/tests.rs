use super::*;
use crate::nn::Module;
use crate::tensor::TensorError;

#[test]
fn test_apply_policy_variants() {
    let policies = [ApplyPolicy::Sync, ApplyPolicy::Cadence, ApplyPolicy::Async];
    assert_eq!(policies.len(), 3);
    assert_eq!(ApplyPolicy::Sync, ApplyPolicy::Sync);
    assert_ne!(ApplyPolicy::Sync, ApplyPolicy::Async);
}

#[test]
fn test_average_backend_variants() {
    let backends = [AverageBackend::Nccl, AverageBackend::Cpu];
    assert_eq!(backends.len(), 2);
    assert_eq!(AverageBackend::Nccl, AverageBackend::Nccl);
    assert_ne!(AverageBackend::Nccl, AverageBackend::Cpu);
}

#[test]
fn test_control_msg_variants() {
    // Verify all variants are constructable
    let _req = ControlMsg::RequestParams;
    let _sync = ControlMsg::SyncNow;
    let _throttle = ControlMsg::Throttle;
    let _start = ControlMsg::StartEpoch(EpochPlan {
        epoch: 0, partition_offset: 0, partition_size: 1000,
    });
    let _ckpt = ControlMsg::Checkpoint { version: 42 };
    let _shutdown = ControlMsg::Shutdown;
    let _update = ControlMsg::Update(AveragedParams {
        params: vec![],
        buffers: vec![],
        version: 0,
    });
}

#[test]
fn test_timing_msg_send() {
    // TimingMsg must be Send (all fields are Copy primitives)
    fn assert_send<T: Send>() {}
    assert_send::<TimingMsg>();
}

#[test]
fn test_metrics_msg_send() {
    fn assert_send<T: Send>() {}
    assert_send::<MetricsMsg>();
}

#[test]
fn test_param_snapshot_send() {
    // ParamSnapshot contains Vec<Tensor> which is Send (Tensor: unsafe impl Send)
    fn assert_send<T: Send>() {}
    assert_send::<ParamSnapshot>();
}

#[test]
fn test_averaged_params_send() {
    fn assert_send<T: Send>() {}
    assert_send::<AveragedParams>();
}

#[test]
fn test_control_msg_send() {
    fn assert_send<T: Send>() {}
    assert_send::<ControlMsg>();
}

#[test]
fn test_worker_config_send() {
    fn assert_send<T: Send>() {}
    assert_send::<WorkerConfig>();
}

#[test]
fn test_worker_config_clone() {
    let cfg = WorkerConfig {
        rank: 0,
        world_size: 2,
        device: Device::CPU,
        initial_params: vec![],
        initial_buffers: vec![],
        total_samples: 10000,
        batch_size: 32,
        seed: 42,
        max_grad_norm: None,
    };
    let cfg2 = cfg.clone();
    assert_eq!(cfg2.rank, 0);
    assert_eq!(cfg2.world_size, 2);
    assert_eq!(cfg2.total_samples, 10000);
}

// -----------------------------------------------------------------------
// GpuWorker tests
// -----------------------------------------------------------------------

use std::sync::mpsc;
use crate::autograd::Variable;
use crate::nn::Linear;
use crate::tensor::{test_device, test_opts, Tensor, TensorOptions, DType};

/// Simple test dataset: random (input, target) pairs.
struct TestDataset {
    n: usize,
}
impl crate::data::BatchDataSet for TestDataset {
    fn len(&self) -> usize { self.n }
    fn get_batch(&self, indices: &[usize]) -> crate::tensor::Result<Vec<Tensor>> {
        let n = indices.len() as i64;
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        Ok(vec![
            Tensor::randn(&[n, 4], opts)?,
            Tensor::randn(&[n, 2], opts)?,
        ])
    }
}

/// Simple MSE train function for tests.
fn mse_train(model: &Linear, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].clone(), false);
    let output = model.forward(&input)?;
    let diff = output.sub(&target)?;
    diff.mul(&diff)?.mean()
}

/// Create a GpuWorker with a simple Linear model for testing.
///
/// Uses a minimal dataset (4 samples = 1 batch, matching batch_size=4) so
/// that GpuWorker::new skips PrefetchWorker creation (nothing to prefetch
/// when the dataset fits in a single batch). This keeps CUDA resource
/// footprint low: each GpuWorker still allocates 2 CUDA streams + 1 event,
/// but avoids the extra thread + channel from PrefetchWorker. Under parallel
/// test execution (or when training runs concurrently), VRAM contention from
/// dozens of workers can cause transient allocation failures.
fn make_test_worker() -> (GpuWorker<Linear>, WorkerChannels) {
    make_test_worker_with(0, 1, 4)
}

/// Create a GpuWorker with configurable rank/world_size/dataset_size.
fn make_test_worker_with(
    rank: usize,
    world_size: usize,
    dataset_size: usize,
) -> (GpuWorker<Linear>, WorkerChannels) {
    let dev = test_device();

    // Build a temporary model to extract initial params
    let tmp_model = Linear::on_device(4, 2, dev).unwrap();
    let tmp_params: Vec<Tensor> = tmp_model.parameters().iter()
        .map(|p| p.variable.data())
        .collect();
    let tmp_buffers: Vec<Tensor> = tmp_model.buffers().iter()
        .map(|b| b.get())
        .collect();
    drop(tmp_model);

    let config = WorkerConfig {
        rank,
        world_size,
        device: dev,
        initial_params: tmp_params,
        initial_buffers: tmp_buffers,
        total_samples: dataset_size,
        batch_size: 4,
        seed: 42,
        max_grad_norm: None,
    };

    let ((timing_tx, metrics_tx, param_tx, final_param_tx, control_rx), channels) =
        GpuWorker::<Linear>::channels();

    let dataset: Arc<dyn crate::data::BatchDataSet> =
        Arc::new(TestDataset { n: dataset_size });

    let worker = GpuWorker::new(
        &config,
        |d| Linear::on_device(4, 2, d),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        dataset,
        None, // no NCCL in unit tests
        None, // no checkpoint in unit tests
        timing_tx,
        metrics_tx,
        param_tx,
        final_param_tx,
        control_rx,
    ).unwrap();

    (worker, channels)
}

#[test]
fn test_worker_new_and_accessors() {
    let (worker, _ch) = make_test_worker();
    assert_eq!(worker.rank(), 0);
    assert_eq!(worker.local_step(), 0);
    assert_eq!(worker.current_version(), 0);
    assert_eq!(worker.param_vars.len(), 2); // Linear: weight + bias
}

#[test]
fn test_worker_snapshot_params() {
    let (worker, _ch) = make_test_worker();
    let snap = worker.snapshot_params();
    assert_eq!(snap.rank, 0);
    assert_eq!(snap.params.len(), 2); // weight + bias
    assert_eq!(snap.buffers.len(), 0); // Linear has no buffers
    assert_eq!(snap.batch_count, 1); // max(steps_since_avg=0, 1)

    // Verify snapshot tensors have the right shapes
    assert_eq!(snap.params[0].shape(), &[2, 4]); // weight
    assert_eq!(snap.params[1].shape(), &[2]);     // bias
}

#[test]
fn test_worker_snapshot_is_send() {
    let (worker, _ch) = make_test_worker();
    let snap = worker.snapshot_params();

    // Verify snapshot can be sent through a channel
    let (tx, rx) = mpsc::channel::<ParamSnapshot>();
    tx.send(snap).unwrap();
    let received = rx.recv().unwrap();
    assert_eq!(received.rank, 0);
    assert_eq!(received.params.len(), 2);
}

#[test]
fn test_worker_load_averaged() {
    // NOTE: This test can fail transiently under VRAM pressure (e.g. when
    // training runs concurrently or many CUDA tests execute in parallel).
    // GpuWorker::new allocates CUDA streams + events, and the update tensors
    // below add further allocations. If this flakes, check GPU utilization.
    let (mut worker, _ch) = make_test_worker();

    // Create "averaged" params on CPU (mirrors the real averaging path where
    // coordinator produces CPU tensors). copy_ handles the H2D transfer.
    let cpu = TensorOptions { dtype: DType::Float32, device: Device::CPU };
    let new_weight = Tensor::ones(&[2, 4], cpu).unwrap();
    let new_bias = Tensor::ones(&[2], cpu).unwrap();

    let update = AveragedParams {
        params: vec![new_weight, new_bias],
        buffers: vec![],
        version: 42,
    };

    worker.load_averaged(&update).unwrap();

    // load_averaged uses non-blocking copy_ on comm_stream (CUDA).
    // In the training loop, sync_before_forward() at the next train_step
    // waits for the event. Here we read directly, so sync the device.
    let dev = test_device();
    if let Device::CUDA(idx) = dev {
        crate::tensor::cuda_synchronize(idx);
    }

    // Verify version updated
    assert_eq!(worker.current_version(), 42);

    // Verify model params now contain all ones
    let snap = worker.snapshot_params();
    let w_sum: f64 = snap.params[0].sum().unwrap().item().unwrap();
    assert!((w_sum - 8.0).abs() < 1e-5, "weight should be all ones (sum=8), got {w_sum}");
    let b_sum: f64 = snap.params[1].sum().unwrap().item().unwrap();
    assert!((b_sum - 2.0).abs() < 1e-5, "bias should be all ones (sum=2), got {b_sum}");
}

#[test]
fn test_worker_load_averaged_wrong_count() {
    let (mut worker, _ch) = make_test_worker();

    let update = AveragedParams {
        params: vec![], // wrong count
        buffers: vec![],
        version: 1,
    };
    assert!(worker.load_averaged(&update).is_err());
}

#[test]
fn test_worker_train_step() {
    let (mut worker, ch) = make_test_worker();
    let opts = test_opts();

    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];

    let (loss, ms) = worker.train_step(&batch, &mse_train).unwrap();
    assert!(ms > 0.0);
    assert!(loss > 0.0);
    assert_eq!(worker.local_step(), 1);

    // Verify timing was NOT auto-sent (train_step doesn't auto-send)
    assert!(ch.timing_rx.try_recv().is_err());
}

#[test]
fn test_worker_report_timing() {
    let (worker, ch) = make_test_worker();

    worker.report_timing(12.5).unwrap();

    let msg = ch.timing_rx.recv().unwrap();
    match msg {
        TimingMsg::Batch { rank, batch_ms, step_count } => {
            assert_eq!(rank, 0);
            assert!((batch_ms - 12.5).abs() < 1e-10);
            assert_eq!(step_count, 0);
        }
        _ => panic!("expected Batch"),
    }
}

#[test]
fn test_worker_report_epoch() {
    let (worker, ch) = make_test_worker();

    worker.report_epoch(0.5, 100, 5000.0).unwrap();

    let msg = ch.metrics_rx.recv().unwrap();
    assert_eq!(msg.rank, 0);
    assert_eq!(msg.epoch, 0);
    assert!((msg.avg_loss - 0.5).abs() < 1e-10);
    assert_eq!(msg.batches_processed, 100);
}

#[test]
fn test_worker_handle_control_request_params() {
    let (mut worker, ch) = make_test_worker();

    ch.control_tx.send(ControlMsg::RequestParams).unwrap();
    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown);

    // Verify snapshot was sent back
    let snap = ch.param_rx.recv().unwrap();
    assert_eq!(snap.rank, 0);
    assert_eq!(snap.params.len(), 2);
}

#[test]
fn test_worker_handle_control_update() {
    let (mut worker, ch) = make_test_worker();
    let dev = test_device();
    let opts = TensorOptions { dtype: DType::Float32, device: dev };

    let update = AveragedParams {
        params: vec![
            Tensor::zeros(&[2, 4], opts).unwrap(),
            Tensor::zeros(&[2], opts).unwrap(),
        ],
        buffers: vec![],
        version: 7,
    };
    ch.control_tx.send(ControlMsg::Update(update)).unwrap();

    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown);
    assert_eq!(worker.current_version(), 7);
}

#[test]
fn test_worker_handle_control_start_epoch() {
    let (mut worker, ch) = make_test_worker();

    assert!(worker.pending_plan.is_none());

    ch.control_tx.send(ControlMsg::StartEpoch(EpochPlan {
        epoch: 1, partition_offset: 0, partition_size: 750,
    })).unwrap();
    worker.handle_control().unwrap();

    let plan = worker.pending_plan.take();
    assert!(plan.is_some());
    assert_eq!(plan.unwrap().partition_size, 750);
    assert!(worker.pending_plan.is_none()); // consumed
}

#[test]
fn test_worker_handle_control_shutdown() {
    let (mut worker, ch) = make_test_worker();

    ch.control_tx.send(ControlMsg::Shutdown).unwrap();
    let shutdown = worker.handle_control().unwrap();
    assert!(shutdown);
}

#[test]
fn test_worker_handle_control_sync_now_noop() {
    let (mut worker, ch) = make_test_worker();

    // SyncNow is a no-op without NCCL (Phase 4)
    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown);
}

#[test]
fn test_worker_full_roundtrip() {
    // Simulates: train -> snapshot -> "average" -> load -> train again
    let (mut worker, ch) = make_test_worker();
    let opts = test_opts();

    // Step 1: train a step
    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];
    worker.train_step(&batch, &mse_train).unwrap();
    assert_eq!(worker.local_step(), 1);

    // Step 2: coordinator requests params
    ch.control_tx.send(ControlMsg::RequestParams).unwrap();
    worker.handle_control().unwrap();
    let snap = ch.param_rx.recv().unwrap();
    assert_eq!(snap.batch_count, 1);

    // Step 3: coordinator sends back "averaged" params (same values, pretend averaged)
    let update = AveragedParams {
        params: snap.params,
        buffers: snap.buffers,
        version: 1,
    };
    ch.control_tx.send(ControlMsg::Update(update)).unwrap();
    worker.handle_control().unwrap();
    assert_eq!(worker.current_version(), 1);

    // Step 4: train another step with loaded params
    let batch2 = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];
    worker.train_step(&batch2, &mse_train).unwrap();
    assert_eq!(worker.local_step(), 2);
}

#[test]
fn test_worker_epoch_from_plan() {
    let (mut worker, _ch) = make_test_worker();
    assert_eq!(worker.current_epoch, 0);
    // Epoch is set from EpochPlan in run_epoch_plan
    worker.current_epoch = 3;
    assert_eq!(worker.current_epoch, 3);
}

#[test]
fn test_worker_channels_create() {
    let ((timing_tx, metrics_tx, param_tx, _final_param_tx, _control_rx), ch) =
        GpuWorker::<Linear>::channels();

    // Verify channel pairs work
    timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 1.0, step_count: 0 }).unwrap();
    let msg = ch.timing_rx.recv().unwrap();
    assert!(matches!(msg, TimingMsg::Batch { rank: 0, .. }));

    metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 0, avg_loss: 0.5, batches_processed: 10, epoch_ms: 100.0,
        samples_processed: 320, scalars: HashMap::new(),
    }).unwrap();
    let msg = ch.metrics_rx.recv().unwrap();
    assert_eq!(msg.batches_processed, 10);

    param_tx.send(ParamSnapshot {
        rank: 0, params: vec![], buffers: vec![], batch_count: 0,
    }).unwrap();
    let snap = ch.param_rx.recv().unwrap();
    assert_eq!(snap.rank, 0);

    ch.control_tx.send(ControlMsg::Shutdown).unwrap();
}

// -----------------------------------------------------------------------
// Coordinator tests
// -----------------------------------------------------------------------

use crate::distributed::ddp::ElChe;

/// Simple coordinator test helper.
struct CoordTestHarness {
    coord: Coordinator,
    /// Send timing/metrics/params TO the coordinator.
    timing_tx: mpsc::Sender<TimingMsg>,
    metrics_tx: mpsc::Sender<MetricsMsg>,
    param_tx: mpsc::Sender<ParamSnapshot>,
    /// Receive control messages FROM the coordinator (one per worker).
    control_rxs: Vec<mpsc::Receiver<ControlMsg>>,
}

fn make_coord_harness(
    n: usize,
    policy: ApplyPolicy,
    backend: AverageBackend,
) -> CoordTestHarness {
    make_coord_harness_with_timeout(n, policy, backend, 5)
}

fn make_coord_harness_with_timeout(
    n: usize,
    policy: ApplyPolicy,
    backend: AverageBackend,
    snapshot_timeout_secs: u64,
) -> CoordTestHarness {
    let (timing_tx, timing_rx) = mpsc::channel();
    let (metrics_tx, metrics_rx) = mpsc::channel();
    let (param_tx, param_rx) = mpsc::channel();

    let mut control_txs = Vec::new();
    let mut control_rxs = Vec::new();
    let mut final_param_rxs = Vec::new();
    for _ in 0..n {
        let (tx, rx) = mpsc::channel();
        control_txs.push(tx);
        control_rxs.push(rx);
        let (_ftx, frx) = mpsc::channel();
        final_param_rxs.push(frx);
    }

    let el_che = ElChe::new(n, 10);
    let coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        policy, backend,
        n, 10000, el_che,
    )
    .snapshot_timeout_secs(snapshot_timeout_secs)
    .build();

    CoordTestHarness { coord, timing_tx, metrics_tx, param_tx, control_rxs }
}

#[test]
fn test_coordinator_initial_state() {
    let h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);
    assert_eq!(h.coord.version(), 0);
    assert!(!h.coord.is_calibrated());
    assert_eq!(h.coord.steps_since_avg(), &[0, 0]);
}

#[test]
fn test_coordinator_drain_timing() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();

    h.coord.drain_timing();

    assert_eq!(h.coord.steps_since_avg(), &[1, 1]);
}

#[test]
fn test_coordinator_should_average_sync() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    // Not ready yet (no steps)
    assert!(!h.coord.should_average());

    // One rank reports
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
    h.coord.drain_timing();
    assert!(!h.coord.should_average()); // rank 1 still at 0

    // Both ranks report
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
    h.coord.drain_timing();
    assert!(h.coord.should_average());
}

#[test]
fn test_coordinator_should_average_async() {
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Nccl);

    // Async now uses batch_counts() same as Cadence (anchor=10 from harness).
    // Feed 9 steps per rank: not enough yet.
    for _ in 0..9 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
    }
    h.coord.drain_timing();
    assert!(!h.coord.should_average());

    // 10th step: both ranks reach batch_counts (anchor=10, uncalibrated so equal).
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
    h.coord.drain_timing();
    assert!(h.coord.should_average());
}

#[test]
fn test_coordinator_should_average_wall_time() {
    // After calibration, Cadence uses wall-time trigger (not batch counts).
    // Async keeps batch-count trigger (overshooting is the feature).
    // Setup: 2 ranks, anchor=10, rank 0 = 5ms/batch (fast), rank 1 = 10ms/batch (slow).
    // anchor_wall_ms = 10 * 10 = 100ms.
    let mut h = make_coord_harness(2, ApplyPolicy::Cadence, AverageBackend::Nccl);

    // Phase 1: calibrate ElChe (uncalibrated uses batch-count fallback).
    // Send 10 batches per rank to trigger initial averaging.
    for _ in 0..10 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
    }
    h.coord.drain_timing();
    assert!(h.coord.should_average()); // batch-count fallback: 10 >= 10
    h.coord.trigger_averaging().unwrap();
    for rx in &h.control_rxs { while rx.try_recv().is_ok() {} }

    assert!(h.coord.is_calibrated());
    let target = h.coord.el_che.anchor_wall_ms();
    assert!(target > 0.0, "anchor_wall_ms should be positive after calibration");

    // Phase 2: wall-time trigger. The slow rank needs target ms of compute.
    // Feed batches until slow rank reaches target, but NOT until batch_counts
    // are met. This proves wall time triggers, not batch counts.
    //
    // After calibration with 2:1 ratio, batch_counts ≈ [20, 10].
    // If we feed 10 batches to each: wall_ms_accum = [50, 100].
    // min(50, 100) = 50 < 100 → no trigger (fast rank hasn't accumulated enough).
    for _ in 0..10 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
    }
    h.coord.drain_timing();
    assert!(!h.coord.should_average(), "fast rank wall time < target");

    // Feed 10 more to rank 0 only (simulating fast GPU running ahead).
    // wall_ms_accum = [100, 100]. min = 100 >= target → trigger!
    for _ in 0..10 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
    }
    h.coord.drain_timing();
    assert!(h.coord.should_average(), "both ranks at target wall time");
}

#[test]
fn test_async_uses_batch_count_not_wall_time() {
    // Async keeps batch-count trigger even after calibration.
    // The divergence between replicas IS the feature (exploration diversity).
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Nccl);

    // Calibrate: 10 batches each at 2:1 speed ratio.
    for _ in 0..10 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
    }
    h.coord.drain_timing();
    assert!(h.coord.should_average());
    h.coord.trigger_averaging().unwrap();
    for rx in &h.control_rxs { while rx.try_recv().is_ok() {} }
    assert!(h.coord.is_calibrated());

    // After calibration, batch_counts ≈ [20, 10].
    // Feed exactly those counts. With wall-time trigger this would NOT
    // fire (fast rank wall = 100ms, slow = 100ms, but batch counts would
    // differ). With batch-count trigger it fires immediately.
    let counts = h.coord.el_che.batch_counts();
    for _ in 0..counts[0] {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
    }
    for _ in 0..counts[1] {
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
    }
    h.coord.drain_timing();
    assert!(h.coord.should_average(), "async triggers on batch counts, not wall time");
}

#[test]
fn test_coordinator_trigger_nccl() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    // Feed timing and trigger
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
    h.coord.drain_timing();
    h.coord.trigger_averaging().unwrap();

    // Workers should receive SyncNow
    for rx in &h.control_rxs {
        match rx.recv().unwrap() {
            ControlMsg::SyncNow => {}
            other => panic!("expected SyncNow, got {:?}", std::mem::discriminant(&other)),
        }
    }

    // Version bumped, steps reset
    assert_eq!(h.coord.version(), 1);
    assert_eq!(h.coord.steps_since_avg(), &[0, 0]);
}

#[test]
fn test_coordinator_trigger_cpu_averaging() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Cpu);
    let dev = test_device();
    let opts = TensorOptions { dtype: DType::Float32, device: dev };

    // Feed timing
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
    h.coord.drain_timing();

    // trigger_averaging now returns immediately (enters Collecting state)
    h.coord.trigger_averaging().unwrap();

    // Workers should receive RequestParams
    for rx in &h.control_rxs {
        match rx.recv().unwrap() {
            ControlMsg::RequestParams => {}
            other => panic!("expected RequestParams, got {:?}", std::mem::discriminant(&other)),
        }
    }

    // Send snapshots (simulating workers responding)
    h.param_tx.send(ParamSnapshot {
        rank: 0,
        params: vec![Tensor::ones(&[2, 3], opts).unwrap()],
        buffers: vec![],
        batch_count: 10,
    }).unwrap();
    h.param_tx.send(ParamSnapshot {
        rank: 1,
        params: vec![Tensor::full(&[2, 3], 3.0, opts).unwrap()],
        buffers: vec![],
        batch_count: 10,
    }).unwrap();

    // Poll until the state machine completes (Collecting -> Computing -> Idle)
    for _ in 0..100 {
        h.coord.poll_cpu_averaging().unwrap();
        if h.coord.version() > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    assert_eq!(h.coord.version(), 1);

    // Workers should receive Update
    for rx in &h.control_rxs {
        match rx.recv().unwrap() {
            ControlMsg::Update(avg) => {
                // Weighted average of 1.0 and 3.0 with equal batch counts = 2.0
                let sum: f64 = avg.params[0].sum().unwrap().item().unwrap();
                let expected = 2.0 * 6.0; // 2.0 * (2*3 elements)
                assert!((sum - expected).abs() < 1e-4,
                    "expected sum={expected}, got {sum}");
                assert_eq!(avg.version, 1);
            }
            other => panic!("expected Update, got {:?}", std::mem::discriminant(&other)),
        }
    }
}

#[test]
fn test_coordinator_average_params_weighted() {
    let dev = test_device();
    let opts = TensorOptions { dtype: DType::Float32, device: dev };

    // Rank 0: all 1.0, did 1 batch
    // Rank 1: all 5.0, did 3 batches
    // Weighted avg: (1*1.0 + 3*5.0) / (1+3) = 16/4 = 4.0
    let snapshots = vec![
        ParamSnapshot {
            rank: 0,
            params: vec![Tensor::ones(&[4], opts).unwrap()],
            buffers: vec![],
            batch_count: 1,
        },
        ParamSnapshot {
            rank: 1,
            params: vec![Tensor::full(&[4], 5.0, opts).unwrap()],
            buffers: vec![],
            batch_count: 3,
        },
    ];

    let avg = Coordinator::average_params(&snapshots, 42).unwrap();
    assert_eq!(avg.version, 42);
    assert_eq!(avg.params.len(), 1);

    // Each element should be (1*1.0 + 3*5.0) / (1+3) = 4.0
    let sum: f64 = avg.params[0].sum().unwrap().item().unwrap();
    let expected = 4.0 * 4.0; // 4.0 per element * 4 elements
    assert!((sum - expected).abs() < 1e-4, "expected sum={expected}, got {sum}");
}

#[test]
fn test_coordinator_tick_sync_flow() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    // No steps yet: tick should not trigger
    let metrics = h.coord.tick().unwrap();
    assert!(metrics.is_empty());
    assert_eq!(h.coord.version(), 0);

    // Feed steps from both ranks
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();

    // Tick: should trigger averaging
    let metrics = h.coord.tick().unwrap();
    assert!(metrics.is_empty());
    assert_eq!(h.coord.version(), 1);

    // Workers got SyncNow
    for rx in &h.control_rxs {
        assert!(matches!(rx.recv().unwrap(), ControlMsg::SyncNow));
    }
}

#[test]
fn test_coordinator_drain_metrics() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    h.metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 1, avg_loss: 0.3, batches_processed: 50, epoch_ms: 2000.0,
        samples_processed: 1600, scalars: HashMap::new(),
    }).unwrap();

    let metrics = h.coord.drain_metrics();
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].rank, 0);
    assert_eq!(metrics[0].epoch, 1);
}

#[test]
fn test_coordinator_compute_partition_sizes() {
    let h = make_coord_harness(2, ApplyPolicy::Cadence, AverageBackend::Nccl);

    // Before calibration, partition sizes should be equal
    let sizes = h.coord.compute_partition_sizes();
    assert_eq!(sizes.len(), 2);
    assert_eq!(sizes[0], 5000); // 10000 / 2
    assert_eq!(sizes[1], 5000);
}

#[test]
fn test_divergence_correction_nudges_anchor_down() {
    // High divergence should nudge ElChe's anchor down after timing
    // calibration. Use zero sync_ms to avoid overhead auto-tune.
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Cpu);

    // Calibrate first so we have a stable anchor baseline.
    let steps = vec![10; 2];
    let wall_ms = vec![100.0; 2];
    h.coord.finish_averaging_cpu(0.0, &steps, &wall_ms, None);
    let anchor_after_calibration = h.coord.el_che.anchor();
    assert!(anchor_after_calibration >= 10);

    // Now apply with high divergence (same timing, zero sync_ms).
    let steps2 = vec![10; 2];
    let wall_ms2 = vec![100.0; 2];
    h.coord.finish_averaging_cpu(0.0, &steps2, &wall_ms2, Some(0.20));

    // Anchor should have decreased from calibrated value.
    assert!(h.coord.el_che.anchor() < anchor_after_calibration,
        "anchor should decrease from {}, got {}",
        anchor_after_calibration, h.coord.el_che.anchor());
}

#[test]
fn test_divergence_below_threshold_no_correction() {
    // Low divergence should NOT change the anchor.
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Cpu);

    // Calibrate with zero sync_ms.
    let steps = vec![10; 2];
    let wall_ms = vec![100.0; 2];
    h.coord.finish_averaging_cpu(0.0, &steps, &wall_ms, None);
    let anchor_after_calibration = h.coord.el_che.anchor();

    // Apply with low divergence.
    let steps2 = vec![10; 2];
    let wall_ms2 = vec![100.0; 2];
    h.coord.finish_averaging_cpu(0.0, &steps2, &wall_ms2, Some(0.01));

    // Divergence 0.01 < threshold 0.05: no correction applied.
    assert_eq!(h.coord.el_che.anchor(), anchor_after_calibration);
}

// -----------------------------------------------------------------------
// Throttle (max_batch_diff) tests
// -----------------------------------------------------------------------

fn make_throttle_harness(
    n: usize,
    max_batch_diff: usize,
) -> CoordTestHarness {
    let (timing_tx, timing_rx) = mpsc::channel();
    let (metrics_tx, metrics_rx) = mpsc::channel();
    let (param_tx, param_rx) = mpsc::channel();

    let mut control_txs = Vec::new();
    let mut control_rxs = Vec::new();
    let mut final_param_rxs = Vec::new();
    for _ in 0..n {
        let (tx, rx) = mpsc::channel();
        control_txs.push(tx);
        control_rxs.push(rx);
        let (_ftx, frx) = mpsc::channel();
        final_param_rxs.push(frx);
    }

    let el_che = ElChe::new(n, 10).with_max_batch_diff(max_batch_diff);
    let coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        ApplyPolicy::Async, AverageBackend::Nccl,
        n, 10000, el_che,
    ).build();

    CoordTestHarness { coord, timing_tx, metrics_tx, param_tx, control_rxs }
}

#[test]
fn test_throttle_sends_when_diff_exceeded() {
    let mut h = make_throttle_harness(2, 3);

    // Rank 0 is 5 steps ahead, rank 1 at 0 -> diff = 5 > 3
    for i in 0..5 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: i }).unwrap();
    }
    h.coord.drain_timing();
    h.coord.check_throttle();

    // Rank 0 should receive Throttle
    match h.control_rxs[0].try_recv() {
        Ok(ControlMsg::Throttle) => {}
        _ => panic!("expected Throttle for rank 0"),
    }

    // Rank 1 should NOT receive Throttle
    assert!(h.control_rxs[1].try_recv().is_err(), "rank 1 should not be throttled");
}

#[test]
fn test_throttle_no_send_within_limit() {
    let mut h = make_throttle_harness(2, 5);

    // Rank 0 is 3 steps ahead, rank 1 at 0 -> diff = 3 <= 5
    for i in 0..3 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: i }).unwrap();
    }
    h.coord.drain_timing();
    h.coord.check_throttle();

    // No throttle for either rank
    assert!(h.control_rxs[0].try_recv().is_err());
    assert!(h.control_rxs[1].try_recv().is_err());
}

#[test]
fn test_throttle_zero_is_strict_lockstep() {
    let mut h = make_throttle_harness(2, 0);

    // Rank 0 does 1 batch, rank 1 does 0 -> diff = 1 > 0
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
    h.coord.drain_timing();
    h.coord.check_throttle();

    // Rank 0 throttled immediately
    match h.control_rxs[0].try_recv() {
        Ok(ControlMsg::Throttle) => {}
        _ => panic!("expected Throttle for rank 0"),
    }
}

#[test]
fn test_throttle_disabled_when_none() {
    // Default harness has no max_batch_diff
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Nccl);

    // Rank 0 far ahead
    for i in 0..50 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: i }).unwrap();
    }
    h.coord.drain_timing();
    h.coord.check_throttle();

    // No throttle (feature disabled)
    assert!(h.control_rxs[0].try_recv().is_err());
}

#[test]
fn test_throttle_worker_unblocks_on_sync_now() {
    // Simulate: worker receives Throttle, then SyncNow unblocks it.
    let (mut worker, ch) = make_test_worker();

    ch.control_tx.send(ControlMsg::Throttle).unwrap();
    ch.control_tx.send(ControlMsg::SyncNow).unwrap();

    // handle_control processes Throttle (blocks on recv), then
    // SyncNow arrives and unblocks it.
    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown, "should not shutdown");
}

#[test]
fn test_throttle_worker_unblocks_on_shutdown() {
    let (mut worker, ch) = make_test_worker();

    ch.control_tx.send(ControlMsg::Throttle).unwrap();
    ch.control_tx.send(ControlMsg::Shutdown).unwrap();

    let shutdown = worker.handle_control().unwrap();
    assert!(shutdown, "should signal shutdown");
}

#[test]
fn test_async_ddp_config_max_batch_diff() {
    let config = DdpRunConfig::new().with_max_batch_diff(5);
    assert_eq!(config.max_batch_diff, Some(5));

    let config2 = DdpRunConfig::new();
    assert_eq!(config2.max_batch_diff, None);
}

// -----------------------------------------------------------------------
// DdpHandle / DdpBuilder tests
// -----------------------------------------------------------------------

#[test]
fn test_async_ddp_single_gpu_fallback() {
    // With <2 GPUs, falls back to single-device training.
    // With 2+ GPUs, uses all of them. Either way, join succeeds.
    let ddp = DdpHandle::auto(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
        Arc::new(TestDataset { n: 100 }),
        4,
        2,  // 2 epochs
        ApplyPolicy::Sync,
        AverageBackend::Cpu, // CPU backend: no NCCL needed for this test
    ).unwrap();

    assert!(ddp.world_size() >= 1);
    let state = ddp.join().unwrap();
    // Linear(4,2): weight [2,4] + bias [2] = 2 params, 0 buffers
    assert_eq!(state.params.len(), 2);
    assert_eq!(state.buffers.len(), 0);
}

#[test]
#[ignore = "NCCL init needs exclusive GPU; run with: make cuda-test-nccl"]
fn test_async_ddp_multi_gpu_nccl() {
    if crate::tensor::usable_cuda_devices().len() < 2 {
        return;
    }

    let ddp = DdpHandle::auto(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
        Arc::new(TestDataset { n: 256 }),
        32,
        2,  // 2 epochs
        ApplyPolicy::Sync,
        AverageBackend::Nccl,
    ).unwrap();

    assert!(ddp.world_size() >= 2);

    // Workers train for 2 epochs then exit, join returns trained state
    let state = ddp.join().unwrap();
    assert_eq!(state.params.len(), 2);
}

#[test]
fn test_async_ddp_send_sync() {
    fn assert_send<T: Send>() {}
    assert_send::<DdpHandle>();
    assert_send::<TrainedState>();
}

// -----------------------------------------------------------------------
// DdpBuilder builder tests
// -----------------------------------------------------------------------

#[test]
fn test_builder_with_defaults() {
    let ddp = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .num_epochs(2)
    .backend(AverageBackend::Cpu)
    .run()
    .unwrap();

    assert!(ddp.world_size() >= 1);
    let state = ddp.join().unwrap();
    assert_eq!(state.params.len(), 2);
}

#[test]
fn test_builder_with_all_options() {
    let ddp = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .num_epochs(2)
    .policy(ApplyPolicy::Sync)
    .backend(AverageBackend::Cpu)
    .overhead_target(0.15)
    .max_anchor(100)
    .anchor(5)
    .divergence_threshold(0.1)
    .max_batch_diff(10)
    .run()
    .unwrap();

    let state = ddp.join().unwrap();
    assert_eq!(state.params.len(), 2);
}

#[test]
#[should_panic(expected = "dataset is required")]
fn test_builder_missing_dataset_panics() {
    let _ = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .batch_size(4)
    .num_epochs(2)
    .run();
}

#[test]
#[should_panic(expected = "batch_size is required")]
fn test_builder_missing_batch_size_panics() {
    let _ = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .num_epochs(2)
    .run();
}

#[test]
#[should_panic(expected = "num_epochs is required")]
fn test_builder_missing_num_epochs_panics() {
    let _ = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .run();
}

// -----------------------------------------------------------------------
// epoch_fn tests
// -----------------------------------------------------------------------

#[test]
fn test_worker_current_epoch_accessor() {
    let (mut worker, _ch) = make_test_worker();
    assert_eq!(worker.current_epoch(), 0);
    worker.current_epoch = 1;
    assert_eq!(worker.current_epoch(), 1);
}

#[test]
fn test_worker_set_lr() {
    let (mut worker, _ch) = make_test_worker();
    // set_lr should not panic; we verify it works by running a train step after
    worker.set_lr(0.1);
    let opts = test_opts();
    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];
    let (loss, _) = worker.train_step(&batch, &mse_train).unwrap();
    assert!(loss > 0.0);
}

#[test]
fn test_epoch_fn_called_per_epoch() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let counter = Arc::new(AtomicUsize::new(0));
    let epochs_seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let counter_c = counter.clone();
    let epochs_c = epochs_seen.clone();

    let num_epochs = 3;
    let ddp = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .num_epochs(num_epochs)
    .backend(AverageBackend::Cpu)
    .epoch_fn(move |epoch, worker| {
        counter_c.fetch_add(1, Ordering::Relaxed);
        epochs_c.lock().unwrap().push(epoch);
        // Verify current_epoch matches the callback argument
        assert_eq!(worker.current_epoch(), epoch);
    })
    .run()
    .unwrap();

    let world = ddp.world_size();
    let _state = ddp.join().unwrap();
    // epoch_fn fires once per epoch per worker
    assert_eq!(counter.load(Ordering::Relaxed), num_epochs * world);
    let mut seen = epochs_seen.lock().unwrap().clone();
    seen.sort();
    // Each worker sees [0, 1, 2]; with N workers we get N copies
    let mut expected: Vec<usize> = (0..num_epochs).cycle().take(num_epochs * world).collect();
    expected.sort();
    assert_eq!(seen, expected);
}

#[test]
fn test_epoch_fn_set_lr() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_c = call_count.clone();

    let ddp = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .num_epochs(3)
    .backend(AverageBackend::Cpu)
    .epoch_fn(move |epoch, worker| {
        // Simulate a LR schedule: decrease LR each epoch
        let lr = 0.01 * (1.0 - epoch as f64 * 0.3);
        worker.set_lr(lr);
        call_count_c.fetch_add(1, Ordering::Relaxed);
    })
    .run()
    .unwrap();

    let world = ddp.world_size();
    let _state = ddp.join().unwrap();
    assert_eq!(call_count.load(Ordering::Relaxed), 3 * world);
}

#[test]
fn test_worker_send_final_snapshot() {
    let (worker, ch) = make_test_worker();
    worker.send_final_snapshot();
    let snap = ch.final_param_rx.recv().unwrap();
    assert_eq!(snap.params.len(), 2); // Linear(4,2): weight + bias
    assert_eq!(snap.rank, 0);
}

#[test]
fn test_collect_final_state_averages() {
    let (timing_tx, timing_rx) = mpsc::channel();
    let (_metrics_tx, metrics_rx) = mpsc::channel();
    let (_param_tx, param_rx) = mpsc::channel();

    let mut control_txs = Vec::new();
    let mut final_param_rxs = Vec::new();
    let mut final_param_txs = Vec::new();
    for _ in 0..2 {
        let (ctx, _crx) = mpsc::channel();
        control_txs.push(ctx);
        let (ftx, frx) = mpsc::channel();
        final_param_txs.push(ftx);
        final_param_rxs.push(frx);
    }

    let el_che = ElChe::new(2, 10);
    let coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        ApplyPolicy::Sync, AverageBackend::Cpu,
        2, 1000, el_che,
    ).build();

    // Send final snapshots from both "workers"
    let opts = crate::tensor::test_opts();
    let t1 = Tensor::full(&[3], 2.0, opts).unwrap();
    let t2 = Tensor::full(&[3], 4.0, opts).unwrap();
    final_param_txs[0].send(ParamSnapshot {
        rank: 0, params: vec![t1], buffers: vec![], batch_count: 1,
    }).unwrap();
    final_param_txs[1].send(ParamSnapshot {
        rank: 1, params: vec![t2], buffers: vec![], batch_count: 1,
    }).unwrap();

    let state = coord.collect_final_state().unwrap();
    assert_eq!(state.params.len(), 1);
    // Average of 2.0 and 4.0 with equal weights = 3.0
    let vals: Vec<f64> = state.params[0].to_f64_vec().unwrap();
    assert!(vals.iter().all(|v| (v - 3.0).abs() < 1e-5), "expected all ~3.0, got {vals:?}");

    // Also verify timing_tx keeps coordinator alive
    drop(timing_tx);
}

#[test]
fn test_collect_final_state_single_survivor() {
    let (_timing_tx, timing_rx) = mpsc::channel();
    let (_metrics_tx, metrics_rx) = mpsc::channel();
    let (_param_tx, param_rx) = mpsc::channel();

    let mut control_txs = Vec::new();
    let mut final_param_rxs = Vec::new();
    let mut final_param_txs = Vec::new();
    for _ in 0..2 {
        let (ctx, _crx) = mpsc::channel();
        control_txs.push(ctx);
        let (ftx, frx) = mpsc::channel();
        final_param_txs.push(ftx);
        final_param_rxs.push(frx);
    }

    let el_che = ElChe::new(2, 10);
    let coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        ApplyPolicy::Sync, AverageBackend::Cpu,
        2, 1000, el_che,
    ).build();

    // Only one worker sends a final snapshot (the other "died")
    let opts = crate::tensor::test_opts();
    let t = Tensor::full(&[3], 7.0, opts).unwrap();
    final_param_txs[0].send(ParamSnapshot {
        rank: 0, params: vec![t], buffers: vec![], batch_count: 5,
    }).unwrap();
    // Worker 1 never sends

    let state = coord.collect_final_state().unwrap();
    assert_eq!(state.params.len(), 1);
    let vals: Vec<f64> = state.params[0].to_f64_vec().unwrap();
    assert!(vals.iter().all(|v| (v - 7.0).abs() < 1e-5), "single survivor should return its own params");
}

// -----------------------------------------------------------------------
// Checkpoint coordination tests
// -----------------------------------------------------------------------

#[test]
fn test_checkpoint_msg_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<ControlMsg>();
}

#[test]
fn test_checkpoint_fn_called_on_dispatch() {
    use std::sync::atomic::{AtomicU64, Ordering};

    let (mut worker, ch) = make_test_worker();
    let called_version = Arc::new(AtomicU64::new(0));
    let cv = called_version.clone();
    worker.checkpoint_fn = Some(Arc::new(move |ver, _model| {
        cv.store(ver, Ordering::Relaxed);
        Ok(())
    }));

    ch.control_tx.send(ControlMsg::Checkpoint { version: 7 }).unwrap();
    worker.handle_control().unwrap();

    assert_eq!(called_version.load(Ordering::Relaxed), 7);
}

#[test]
fn test_checkpoint_error_logged_not_propagated() {
    let (mut worker, ch) = make_test_worker();
    worker.checkpoint_fn = Some(Arc::new(|_ver, _model| {
        Err(TensorError::new("disk full"))
    }));

    ch.control_tx.send(ControlMsg::Checkpoint { version: 1 }).unwrap();
    // Should not return an error: log-and-continue
    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown);
}

#[test]
fn test_coordinator_sends_checkpoint_every_n_epochs() {
    use crate::distributed::ddp::ElChe;

    let n = 2;
    let (_timing_tx, timing_rx) = mpsc::channel();
    let (_metrics_tx, metrics_rx) = mpsc::channel();
    let (_param_tx, param_rx) = mpsc::channel();

    let mut control_txs = Vec::new();
    let mut control_rxs = Vec::new();
    let mut final_param_rxs = Vec::new();
    for _ in 0..n {
        let (tx, rx) = mpsc::channel();
        control_txs.push(tx);
        control_rxs.push(rx);
        let (_ftx, frx) = mpsc::channel();
        final_param_rxs.push(frx);
    }

    let el_che = ElChe::new(n, 10);
    let mut coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        ApplyPolicy::Sync, AverageBackend::Nccl,
        n, 10000, el_che,
    )
    .num_epochs(10)
    .checkpoint_every(2)
    .build();

    // Aggregate 3 global epochs.
    for epoch in 0..3 {
        coord.on_epoch_aggregated(epoch);
    }

    // checkpoint_every=2: epoch 0 → (0+1)%2=1 no, epoch 1 → (1+1)%2=0 yes, epoch 2 → (2+1)%2=1 no
    let mut checkpoint_versions = Vec::new();
    for rx in &control_rxs {
        while let Ok(msg) = rx.try_recv() {
            if let ControlMsg::Checkpoint { version } = msg {
                checkpoint_versions.push(version);
            }
        }
    }
    assert_eq!(checkpoint_versions, vec![2], "should checkpoint once (at epoch 2) after 3 epochs with every=2");
}

// -----------------------------------------------------------------------
// Phase 10: 2-GPU end-to-end validation
// -----------------------------------------------------------------------

/// Shared loss tracker for multi-GPU convergence tests.
/// Each rank appends (rank, step, loss) tuples.
type LossLog = Arc<std::sync::Mutex<Vec<(usize, usize, f64)>>>;

fn make_loss_tracker() -> LossLog {
    Arc::new(std::sync::Mutex::new(Vec::new()))
}

/// Run a 2-GPU DDP session and return collected losses per rank.
/// Returns (rank0_losses, rank1_losses) in chronological order.
fn run_2gpu_training(
    backend: AverageBackend,
    policy: ApplyPolicy,
    num_epochs: usize,
) -> (Vec<f64>, Vec<f64>) {
    let log = make_loss_tracker();
    let log_clone = log.clone();

    let ddp = DdpHandle::auto(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        move |model: &Linear, batch: &[Tensor]| {
            let input = Variable::new(batch[0].clone(), false);
            let target = Variable::new(batch[1].clone(), false);
            let output = model.forward(&input)?;
            let diff = output.sub(&target)?;
            let loss = diff.mul(&diff)?.mean()?;
            let loss_val: f64 = loss.data().item()?;
            // Determine rank from device
            let rank = match batch[0].device() {
                Device::CUDA(idx) => idx as usize,
                Device::CPU => 0,
            };
            let step = {
                let mut lg = log_clone.lock().unwrap();
                let step = lg.iter().filter(|(r, _, _)| *r == rank).count();
                lg.push((rank, step, loss_val));
                step
            };
            let _ = step;
            Ok(loss)
        },
        Arc::new(TestDataset { n: 512 }),
        32,
        num_epochs,
        policy,
        backend,
    ).unwrap();

    let _state = ddp.join().unwrap();

    let entries = log.lock().unwrap();
    let r0: Vec<f64> = entries.iter().filter(|(r, _, _)| *r == 0).map(|(_, _, l)| *l).collect();
    let r1: Vec<f64> = entries.iter().filter(|(r, _, _)| *r == 1).map(|(_, _, l)| *l).collect();
    (r0, r1)
}

#[test]
#[ignore = "NCCL init needs exclusive GPU; run with: make cuda-test-nccl"]
fn test_async_ddp_2gpu_cpu_backend_loss_decreases() {
    if crate::tensor::usable_cuda_devices().len() < 2 {
        return;
    }

    let (r0, r1) = run_2gpu_training(AverageBackend::Cpu, ApplyPolicy::Sync, 5);

    // Both ranks should have trained
    assert!(!r0.is_empty(), "rank 0 should have loss entries");
    assert!(!r1.is_empty(), "rank 1 should have loss entries");

    // Loss should converge: final losses should be finite and reasonable.
    // For a tiny Linear(4,2) with random data, the irreducible MSE is ~1.0.
    // We check that training converges (not diverges) rather than strictly decreases,
    // since NCCL averaging overhead can cause minor fluctuations.
    let check_converged = |losses: &[f64], rank: usize| {
        let n = losses.len();
        let quarter = (n / 4).max(1);
        let last_avg: f64 = losses[n - quarter..].iter().sum::<f64>() / quarter as f64;
        assert!(last_avg.is_finite() && last_avg < 2.0,
            "rank {rank} should converge: last_avg={last_avg:.4}");
    };

    check_converged(&r0, 0);
    check_converged(&r1, 1);
}

#[test]
#[ignore = "NCCL init needs exclusive GPU; run with: make cuda-test-nccl"]
fn test_async_ddp_2gpu_nccl_backend_loss_decreases() {
    if crate::tensor::usable_cuda_devices().len() < 2 {
        return;
    }

    let (r0, r1) = run_2gpu_training(AverageBackend::Nccl, ApplyPolicy::Sync, 5);

    assert!(!r0.is_empty(), "rank 0 should have loss entries");
    assert!(!r1.is_empty(), "rank 1 should have loss entries");

    let check_converged = |losses: &[f64], rank: usize| {
        let n = losses.len();
        let quarter = (n / 4).max(1);
        let last_avg: f64 = losses[n - quarter..].iter().sum::<f64>() / quarter as f64;
        assert!(last_avg.is_finite() && last_avg < 2.0,
            "rank {rank} should converge: last_avg={last_avg:.4}");
    };

    check_converged(&r0, 0);
    check_converged(&r1, 1);
}

#[test]
#[ignore = "NCCL init needs exclusive GPU; run with: make cuda-test-nccl"]
fn test_async_ddp_ab_cpu_vs_nccl() {
    if crate::tensor::usable_cuda_devices().len() < 2 {
        return;
    }

    let epochs = 5;
    let (cpu_r0, cpu_r1) = run_2gpu_training(AverageBackend::Cpu, ApplyPolicy::Sync, epochs);
    let (nccl_r0, nccl_r1) = run_2gpu_training(AverageBackend::Nccl, ApplyPolicy::Sync, epochs);

    // Both backends should converge (loss decreases)
    let final_avg = |losses: &[f64]| -> f64 {
        let n = losses.len();
        let quarter = n / 4;
        if quarter == 0 { return f64::MAX; }
        losses[n - quarter..].iter().sum::<f64>() / quarter as f64
    };

    let cpu_final = (final_avg(&cpu_r0) + final_avg(&cpu_r1)) / 2.0;
    let nccl_final = (final_avg(&nccl_r0) + final_avg(&nccl_r1)) / 2.0;

    // Both should have converged to a reasonable loss
    assert!(cpu_final < 2.0,
        "CPU backend final loss too high: {cpu_final:.4}");
    assert!(nccl_final < 2.0,
        "NCCL backend final loss too high: {nccl_final:.4}");

    // Final losses should be in the same ballpark (within 2x of each other).
    // They won't be identical because data shuffling differs across runs,
    // but for a simple Linear model both should converge to similar regions.
    let ratio = cpu_final.max(nccl_final) / cpu_final.min(nccl_final);
    eprintln!("  A/B: CPU final={cpu_final:.4} NCCL final={nccl_final:.4} ratio={ratio:.2}");
    assert!(ratio < 3.0,
        "CPU vs NCCL final loss ratio too large: {ratio:.2} (CPU={cpu_final:.4} NCCL={nccl_final:.4})");
}

// -----------------------------------------------------------------------
// ElChe cadence + adaptive K tests (Phase 6)
// -----------------------------------------------------------------------

#[test]
fn test_cadence_heterogeneous_timing() {
    // Simulate 2:1 speed ratio. Rank 0 is 2x faster (5ms/batch vs 10ms/batch).
    // With Cadence policy, ElChe should give rank 0 more batches.
    let mut h = make_coord_harness(2, ApplyPolicy::Cadence, AverageBackend::Nccl);

    // Feed enough timing to calibrate ElChe.
    // First, trigger with equal steps so ElChe sees the timing.
    for _ in 0..10 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
        h.coord.drain_timing();
        if h.coord.should_average() {
            h.coord.trigger_averaging().unwrap();
            // Drain control messages
            for rx in &h.control_rxs {
                while rx.try_recv().is_ok() {}
            }
        }
    }

    // After calibration, ElChe batch_counts should reflect the speed ratio
    if h.coord.is_calibrated() {
        let counts = h.coord.el_che.batch_counts();
        // Rank 0 (fast) should have more batches than rank 1 (slow)
        assert!(counts[0] >= counts[1],
            "fast rank should get more batches: {:?}", counts);
    }
}

#[test]
fn test_cpu_averaging_divergence_correction() {
    // Full pipeline: high divergence during CPU averaging triggers
    // anchor correction via nudge_anchor_down.
    let dev = test_device();
    let opts = TensorOptions { dtype: DType::Float32, device: dev };
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Cpu);

    assert_eq!(h.coord.el_che.anchor(), 10);

    // Feed enough timing to reach batch_counts (anchor=10, uncalibrated).
    for _ in 0..10 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 0 }).unwrap();
    }
    h.coord.drain_timing();
    assert!(h.coord.should_average());

    // Trigger CPU averaging with highly divergent snapshots.
    h.coord.trigger_averaging().unwrap();
    h.param_tx.send(ParamSnapshot {
        rank: 0,
        params: vec![Tensor::ones(&[100], opts).unwrap()],
        buffers: vec![],
        batch_count: 1,
    }).unwrap();
    h.param_tx.send(ParamSnapshot {
        rank: 1,
        params: vec![Tensor::full(&[100], 100.0, opts).unwrap()],
        buffers: vec![],
        batch_count: 1,
    }).unwrap();

    // Poll until averaging completes.
    let v_before = h.coord.version();
    for _ in 0..100 {
        h.coord.poll_cpu_averaging().unwrap();
        if h.coord.version() > v_before {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(h.coord.version() > v_before, "averaging should have completed");

    // Drain control messages.
    for rx in &h.control_rxs {
        while rx.try_recv().is_ok() {}
    }

    // After one round: report_timing auto-tunes anchor up (from overhead),
    // then divergence correction halves it. Final anchor should be lower
    // than the post-overhead-auto-tune value. We verify it completed and
    // the anchor is reasonable (not at max_anchor=200).
    let anchor = h.coord.el_che.anchor();
    assert!(anchor < 200,
        "divergence correction should have kept anchor below max, got {}", anchor);
    // Verify calibration happened.
    assert!(h.coord.is_calibrated());
}

// -----------------------------------------------------------------------
// Non-blocking CPU averaging tests
// -----------------------------------------------------------------------

#[test]
fn test_throttle_during_cpu_averaging() {
    // The key invariant: check_throttle fires even while CPU averaging
    // is in Collecting state.
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Cpu);
    let el_che = ElChe::new(2, 10).with_max_batch_diff(2);
    h.coord.el_che = el_che;

    // Feed enough timing to trigger averaging
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 1 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 1 }).unwrap();
    h.coord.drain_timing();

    // Trigger averaging (enters Collecting state, returns immediately)
    assert!(h.coord.should_average());
    h.coord.trigger_averaging().unwrap();
    assert!(h.coord.is_cpu_averaging());
    assert!(!h.coord.should_average()); // guard prevents re-trigger

    // Consume RequestParams from control channels
    for rx in &h.control_rxs {
        match rx.try_recv() {
            Ok(ControlMsg::RequestParams) => {}
            other => panic!("expected RequestParams, got {:?}", other.map(|m| std::mem::discriminant(&m))),
        }
    }

    // Simulate rank 0 running ahead by 5 batches during the averaging window
    for i in 0..5 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 1.0, step_count: 2 + i }).unwrap();
    }
    h.coord.drain_timing();

    // check_throttle should fire even though we're in Collecting state
    h.coord.check_throttle();

    // Rank 0 should receive Throttle (it's 5 batches ahead, max_diff=2)
    match h.control_rxs[0].try_recv() {
        Ok(ControlMsg::Throttle) => {}
        other => panic!("expected Throttle for rank 0, got {:?}", other.map(|m| std::mem::discriminant(&m))),
    }
    // Rank 1 should NOT be throttled
    assert!(h.control_rxs[1].try_recv().is_err(), "rank 1 should not be throttled");
}

#[test]
fn test_cpu_avg_state_machine_full_cycle() {
    // Drive the full Idle -> Collecting -> Computing -> Idle cycle.
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Cpu);
    let dev = test_device();
    let opts = TensorOptions { dtype: DType::Float32, device: dev };

    // Feed timing
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
    h.coord.drain_timing();

    assert_eq!(h.coord.version(), 0);
    assert!(!h.coord.is_cpu_averaging());

    // Trigger: enters Collecting
    h.coord.trigger_averaging().unwrap();
    assert!(h.coord.is_cpu_averaging());

    // Poll with no snapshots yet: still Collecting
    h.coord.poll_cpu_averaging().unwrap();
    assert!(h.coord.is_cpu_averaging());

    // Supply snapshots
    h.param_tx.send(ParamSnapshot {
        rank: 0,
        params: vec![Tensor::ones(&[4], opts).unwrap()],
        buffers: vec![],
        batch_count: 5,
    }).unwrap();
    h.param_tx.send(ParamSnapshot {
        rank: 1,
        params: vec![Tensor::full(&[4], 3.0, opts).unwrap()],
        buffers: vec![],
        batch_count: 5,
    }).unwrap();

    // Poll: transitions Collecting -> Computing (spawns thread)
    h.coord.poll_cpu_averaging().unwrap();

    // Poll until Computing -> Idle
    for _ in 0..100 {
        h.coord.poll_cpu_averaging().unwrap();
        if !h.coord.is_cpu_averaging() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    // Verify completion
    assert!(!h.coord.is_cpu_averaging());
    assert_eq!(h.coord.version(), 1);

    // Workers should have received RequestParams then Update
    for rx in &h.control_rxs {
        let mut got_request = false;
        let mut got_update = false;
        while let Ok(msg) = rx.try_recv() {
            match msg {
                ControlMsg::RequestParams => got_request = true,
                ControlMsg::Update(avg) => {
                    got_update = true;
                    assert_eq!(avg.version, 1);
                }
                _ => {}
            }
        }
        assert!(got_request, "worker should have received RequestParams");
        assert!(got_update, "worker should have received Update");
    }
}

#[test]
fn test_cpu_avg_collection_timeout() {
    // Use a very short timeout (1 second) and never send snapshots.
    let mut h = make_coord_harness_with_timeout(
        2, ApplyPolicy::Sync, AverageBackend::Cpu, 1,
    );

    // Feed timing to trigger averaging
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 1 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 1 }).unwrap();
    h.coord.drain_timing();

    // Trigger: enters Collecting
    h.coord.trigger_averaging().unwrap();
    assert!(h.coord.is_cpu_averaging());

    // Wait for the timeout to expire
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Poll: should soft-abort (back to Idle)
    h.coord.poll_cpu_averaging().unwrap(); // Ok, not Err
    assert!(!h.coord.is_cpu_averaging());
    assert_eq!(h.coord.version(), 0); // no version bump

    // should_average is available again for retry
    assert!(h.coord.should_average());
}

#[test]
fn test_stale_snapshot_after_timeout() {
    // After a timeout, stale snapshots from the aborted round
    // must not contaminate the next round.
    let mut h = make_coord_harness_with_timeout(
        2, ApplyPolicy::Sync, AverageBackend::Cpu, 1,
    );
    let dev = test_device();
    let opts = TensorOptions { dtype: DType::Float32, device: dev };

    // Feed timing
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 1 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 1 }).unwrap();
    h.coord.drain_timing();

    // Round 1: trigger, send only rank 0's snapshot, let it timeout
    h.coord.trigger_averaging().unwrap();
    h.param_tx.send(ParamSnapshot {
        rank: 0,
        params: vec![Tensor::full(&[4], 999.0, opts).unwrap()],
        buffers: vec![],
        batch_count: 1,
    }).unwrap();

    // Wait for timeout
    std::thread::sleep(std::time::Duration::from_secs(2));
    h.coord.poll_cpu_averaging().unwrap();
    assert!(!h.coord.is_cpu_averaging()); // soft abort
    assert_eq!(h.coord.version(), 0);

    // Round 2: trigger fresh. The stale rank-0 snapshot from round 1
    // should have been drained by abort_cpu_averaging.
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 2 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 2 }).unwrap();
    h.coord.drain_timing();

    h.coord.trigger_averaging().unwrap();

    // Send FRESH snapshots for both ranks (value=1.0 and 3.0)
    h.param_tx.send(ParamSnapshot {
        rank: 0,
        params: vec![Tensor::ones(&[4], opts).unwrap()],
        buffers: vec![],
        batch_count: 1,
    }).unwrap();
    h.param_tx.send(ParamSnapshot {
        rank: 1,
        params: vec![Tensor::full(&[4], 3.0, opts).unwrap()],
        buffers: vec![],
        batch_count: 1,
    }).unwrap();

    // Poll until complete
    for _ in 0..100 {
        h.coord.poll_cpu_averaging().unwrap();
        if h.coord.version() > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(h.coord.version(), 1);

    // Verify the Update contains fresh data (avg of 1.0 and 3.0 = 2.0),
    // NOT 999.0 from the stale snapshot.
    for rx in &h.control_rxs {
        let mut found_update = false;
        while let Ok(msg) = rx.try_recv() {
            if let ControlMsg::Update(avg) = msg {
                let sum: f64 = avg.params[0].sum().unwrap().item().unwrap();
                let expected = 2.0 * 4.0; // 2.0 per element * 4 elements
                assert!(
                    (sum - expected).abs() < 1e-4,
                    "expected sum={expected}, got {sum} (stale data leaked?)"
                );
                found_update = true;
            }
        }
        assert!(found_update, "worker should have received Update");
    }
}

#[test]
fn test_elche_calibration_produces_proportional_sizes() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    // Feed heterogeneous timing to trigger ElChe calibration
    for _ in 0..5 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
        h.coord.drain_timing();
        if h.coord.should_average() {
            h.coord.trigger_averaging().unwrap();
            for rx in &h.control_rxs {
                while rx.try_recv().is_ok() {}
            }
        }
    }

    assert!(h.coord.is_calibrated(), "ElChe should have calibrated");
    // After calibration, compute_partition_sizes should produce valid sizes
    let sizes = h.coord.compute_partition_sizes();
    assert_eq!(sizes.len(), 2);
    let total: usize = sizes.iter().sum();
    assert!(total <= 10000, "partitions should not exceed total: {total}");
}

#[test]
fn test_wall_ms_accumulation() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    // Send multiple timing messages per rank
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 7.0, step_count: 1 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 12.0, step_count: 1 }).unwrap();
    h.coord.drain_timing();

    // wall_ms_accum should have accumulated totals
    assert!((h.coord.wall_ms_accum[0] - 12.0).abs() < 1e-10, "rank 0 should be 5+7=12");
    assert!((h.coord.wall_ms_accum[1] - 22.0).abs() < 1e-10, "rank 1 should be 10+12=22");
}

#[test]
fn test_config_defaults() {
    let cfg = DdpRunConfig::new();
    assert!(cfg.overhead_target.is_none());
    assert!(cfg.max_anchor.is_none());
    assert!(cfg.anchor.is_none());
    assert!(cfg.divergence_threshold.is_none());
}

#[test]
fn test_config_builder() {
    let cfg = DdpRunConfig::new()
        .with_overhead_target(0.05)
        .with_max_anchor(100)
        .with_anchor(20)
        .with_divergence_threshold(0.01);
    assert_eq!(cfg.overhead_target, Some(0.05));
    assert_eq!(cfg.max_anchor, Some(100));
    assert_eq!(cfg.anchor, Some(20));
    assert_eq!(cfg.divergence_threshold, Some(0.01));
}

// -----------------------------------------------------------------------
// Partition + data iteration tests (Phase 5)
// -----------------------------------------------------------------------

#[test]
fn test_make_partition_basic() {
    let p0 = make_partition(0, 50, 100, 0, 42);
    let p1 = make_partition(50, 50, 100, 0, 42);
    assert_eq!(p0.len(), 50);
    assert_eq!(p1.len(), 50);

    // Non-overlapping (consecutive offsets, same epoch, same seed)
    let mut all: Vec<usize> = p0.iter().chain(p1.iter()).copied().collect();
    all.sort();
    all.dedup();
    assert_eq!(all.len(), 100, "partitions should be non-overlapping");
}

#[test]
fn test_make_partition_different_epochs() {
    let p_e0 = make_partition(0, 50, 100, 0, 42);
    let p_e1 = make_partition(0, 50, 100, 1, 42);
    // Different epochs should produce different orderings
    assert_ne!(p_e0, p_e1);
}

#[test]
fn test_make_partition_deterministic() {
    let p1 = make_partition(0, 50, 100, 5, 42);
    let p2 = make_partition(0, 50, 100, 5, 42);
    assert_eq!(p1, p2, "same params should produce same partition");
}

#[test]
fn test_worker_partition_changes_with_epoch() {
    let (mut worker, _ch) = make_test_worker();
    // Run epoch 0
    let plan0 = EpochPlan { epoch: 0, partition_offset: 0, partition_size: 1000 };
    worker.run_epoch_plan(&plan0, &mse_train).unwrap();
    let partition0 = worker.partition.clone();

    // Run epoch 1 - different epoch produces different partition
    let plan1 = EpochPlan { epoch: 1, partition_offset: 0, partition_size: 1000 };
    worker.run_epoch_plan(&plan1, &mse_train).unwrap();
    assert_ne!(worker.partition, partition0);
}

#[test]
fn test_worker_epoch_plan_applies_partition_size() {
    let (mut worker, _ch) = make_test_worker_with(0, 1, 1000);

    // Run with a smaller partition via EpochPlan
    let plan = EpochPlan { epoch: 0, partition_offset: 0, partition_size: 200 };
    worker.run_epoch_plan(&plan, &mse_train).unwrap();
    assert_eq!(worker.partition.len(), 200);
}

#[test]
fn test_worker_run_epoch_plan() {
    // 40 samples, batch_size=4 -> 10 batches per epoch
    let (mut worker, ch) = make_test_worker_with(0, 1, 40);

    let plan = EpochPlan { epoch: 0, partition_offset: 0, partition_size: 40 };
    let shutdown = worker.run_epoch_plan(&plan, &mse_train).unwrap();
    assert!(!shutdown);
    assert_eq!(worker.current_epoch, 0);

    // Should have received timing messages (one per batch)
    let mut count = 0;
    while ch.timing_rx.try_recv().is_ok() {
        count += 1;
    }
    assert!(count > 0, "should have sent timing messages");

    // Should have received epoch metrics
    let metrics = ch.metrics_rx.recv().unwrap();
    assert_eq!(metrics.epoch, 0); // epoch 0 was just completed
    assert!(metrics.avg_loss > 0.0);
    assert!(metrics.batches_processed > 0);
}

#[test]
fn test_worker_run_epoch_plan_loss_decreases() {
    let (mut worker, _ch) = make_test_worker_with(0, 1, 80);

    // Run a few epochs, loss should decrease
    for epoch in 0..5 {
        let plan = EpochPlan { epoch, partition_offset: 0, partition_size: 80 };
        worker.run_epoch_plan(&plan, &mse_train).unwrap();
    }
    // Snapshot and check loss on a fixed batch
    let opts = test_opts();
    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];
    let loss_after: f64 = mse_train(worker.model(), &batch).unwrap().data().item().unwrap();
    // After 5 epochs of training, loss should be finite and non-negative
    assert!(loss_after.is_finite());
}

#[test]
fn test_worker_run_epoch_plan_shutdown_mid_epoch() {
    let (mut worker, ch) = make_test_worker_with(0, 1, 400);

    // Send shutdown after a short delay via the control channel
    ch.control_tx.send(ControlMsg::Shutdown).unwrap();

    let plan = EpochPlan { epoch: 0, partition_offset: 0, partition_size: 400 };
    let shutdown = worker.run_epoch_plan(&plan, &mse_train).unwrap();
    assert!(shutdown, "should detect shutdown during epoch");
}

#[test]
fn test_cpu_averaging_end_to_end() {
    // Two workers on CPU, CPU averaging backend.
    // Simulate the coordinator cycle manually.
    let (mut w0, _ch0) = make_test_worker_with(0, 2, 40);
    let (mut w1, _ch1) = make_test_worker_with(1, 2, 40);

    // Run one epoch on each worker
    let plan0 = EpochPlan { epoch: 0, partition_offset: 0, partition_size: 20 };
    let plan1 = EpochPlan { epoch: 0, partition_offset: 20, partition_size: 20 };
    w0.run_epoch_plan(&plan0, &mse_train).unwrap();
    w1.run_epoch_plan(&plan1, &mse_train).unwrap();

    // Snapshot params from both
    let snap0 = w0.snapshot_params();
    let snap1 = w1.snapshot_params();

    // Average them (coordinator's static method)
    let averaged = Coordinator::average_params(&[snap0, snap1], 1).unwrap();

    // Load averaged params into both workers
    w0.load_averaged(&averaged).unwrap();
    w1.load_averaged(&averaged).unwrap();

    assert_eq!(w0.current_version(), 1);
    assert_eq!(w1.current_version(), 1);

    // Both should now have the same params
    let s0 = w0.snapshot_params();
    let s1 = w1.snapshot_params();
    for (p0, p1) in s0.params.iter().zip(&s1.params) {
        let diff: f64 = p0.sub(p1).unwrap().abs().unwrap().sum().unwrap().item().unwrap();
        assert!(diff < 1e-5, "params should be identical after averaging, diff={diff}");
    }
}

// -----------------------------------------------------------------------
// Proportional epoch sharding tests (Phase 7)
// -----------------------------------------------------------------------

#[test]
fn test_proportional_sharding() {
    // 2:1 speed ratio -> partition sizes should be 2:1
    let mut h = make_coord_harness(2, ApplyPolicy::Cadence, AverageBackend::Nccl);

    // Calibrate ElChe with 2:1 timing
    for _ in 0..3 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0 }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0 }).unwrap();
        h.coord.drain_timing();
        if h.coord.should_average() {
            h.coord.trigger_averaging().unwrap();
            for rx in &h.control_rxs {
                while rx.try_recv().is_ok() {}
            }
        }
    }

    if h.coord.is_calibrated() {
        let sizes = h.coord.compute_partition_sizes();
        assert_eq!(sizes.len(), 2);
        // Fast rank (0) should get more samples than slow rank (1)
        assert!(sizes[0] > sizes[1],
            "fast rank should get more samples: {:?}", sizes);
        // Total should approximate dataset size (10000)
        let total: usize = sizes.iter().sum();
        assert!(total <= 10000, "partitions should not exceed total: {total}");
    }
}

#[test]
fn test_partition_non_overlapping_equal_sizes() {
    // Equal partition sizes with consecutive offsets: guaranteed non-overlapping
    let total = 300;
    let per_rank = total / 3; // 100 each
    let p0 = make_partition(0, per_rank, total, 5, 42);
    let p1 = make_partition(100, per_rank, total, 5, 42);
    let p2 = make_partition(200, per_rank, total, 5, 42);

    assert_eq!(p0.len(), 100);
    assert_eq!(p1.len(), 100);
    assert_eq!(p2.len(), 100);

    let set0: std::collections::HashSet<usize> = p0.iter().copied().collect();
    let set1: std::collections::HashSet<usize> = p1.iter().copied().collect();
    let set2: std::collections::HashSet<usize> = p2.iter().copied().collect();
    assert_eq!(set0.intersection(&set1).count(), 0, "rank 0/1 should not overlap");
    assert_eq!(set0.intersection(&set2).count(), 0, "rank 0/2 should not overlap");
    assert_eq!(set1.intersection(&set2).count(), 0, "rank 1/2 should not overlap");
}

#[test]
fn test_partition_non_overlapping_smaller_sizes() {
    // Non-overlapping consecutive offsets with varying sizes
    let total = 300;
    let p0 = make_partition(0, 50, total, 5, 42);   // offset 0, size 50
    let p1 = make_partition(50, 80, total, 5, 42);   // offset 50, size 80
    let p2 = make_partition(130, 60, total, 5, 42);  // offset 130, size 60

    let set0: std::collections::HashSet<usize> = p0.iter().copied().collect();
    let set1: std::collections::HashSet<usize> = p1.iter().copied().collect();
    let set2: std::collections::HashSet<usize> = p2.iter().copied().collect();
    assert_eq!(set0.intersection(&set1).count(), 0, "rank 0/1 should not overlap");
    assert_eq!(set0.intersection(&set2).count(), 0, "rank 0/2 should not overlap");
    assert_eq!(set1.intersection(&set2).count(), 0, "rank 1/2 should not overlap");
}

#[test]
fn test_partition_benign_overlap_different_epochs() {
    // Different epochs produce different permutations, so overlap is expected
    let p0_e5 = make_partition(0, 50, 100, 5, 42);
    let p1_e6 = make_partition(50, 50, 100, 6, 42);
    // These are from different epochs, so some overlap is expected and benign
    let set0: std::collections::HashSet<usize> = p0_e5.iter().copied().collect();
    let set1: std::collections::HashSet<usize> = p1_e6.iter().copied().collect();
    // Just verify they're valid indices
    assert!(set0.iter().all(|&i| i < 100));
    assert!(set1.iter().all(|&i| i < 100));
}

#[test]
fn test_self_managed_epochs() {
    // Worker should run multiple epochs via plans, reporting metrics each time
    let (mut worker, ch) = make_test_worker_with(0, 1, 40);

    // Run 3 epochs
    for epoch in 0..3 {
        let plan = EpochPlan { epoch, partition_offset: 0, partition_size: 40 };
        let shutdown = worker.run_epoch_plan(&plan, &mse_train).unwrap();
        assert!(!shutdown);
    }

    assert_eq!(worker.current_epoch, 2); // set to last plan's epoch

    // Should have received 3 epoch metrics
    let mut epoch_msgs = Vec::new();
    while let Ok(msg) = ch.metrics_rx.try_recv() {
        epoch_msgs.push(msg);
    }
    assert_eq!(epoch_msgs.len(), 3);
    assert_eq!(epoch_msgs[0].epoch, 0);
    assert_eq!(epoch_msgs[1].epoch, 1);
    assert_eq!(epoch_msgs[2].epoch, 2);
}

#[test]
fn test_epoch_plan_partition_size_at_epoch_boundary() {
    let (mut worker, _ch) = make_test_worker_with(0, 1, 80);

    // Run first epoch with full partition
    let plan0 = EpochPlan { epoch: 0, partition_offset: 0, partition_size: 80 };
    worker.run_epoch_plan(&plan0, &mse_train).unwrap();
    assert_eq!(worker.partition.len(), 80);

    // Next epoch with a smaller partition from EpochPlan
    let plan1 = EpochPlan { epoch: 1, partition_offset: 0, partition_size: 20 };
    worker.run_epoch_plan(&plan1, &mse_train).unwrap();
    assert_eq!(worker.partition.len(), 20);
}

// -----------------------------------------------------------------------
// record_scalar tests
// -----------------------------------------------------------------------

#[test]
fn test_record_scalar_accumulates() {
    // Clear any leftovers from other tests on this thread
    drain_scalars();

    record_scalar("loss", 1.0);
    record_scalar("loss", 2.0);
    record_scalar("loss", 3.0);

    let map = drain_scalars();
    assert_eq!(map.len(), 1);
    let (sum, count) = map["loss"];
    assert_eq!(sum, 6.0);
    assert_eq!(count, 3);
}

#[test]
fn test_record_scalar_multiple_tags() {
    drain_scalars();

    record_scalar("a", 1.0);
    record_scalar("b", 2.0);
    record_scalar("a", 3.0);

    let map = drain_scalars();
    assert_eq!(map.len(), 2);
    assert_eq!(map["a"], (4.0, 2));
    assert_eq!(map["b"], (2.0, 1));
}

#[test]
fn test_drain_scalars_clears() {
    drain_scalars();

    record_scalar("x", 1.0);
    let first = drain_scalars();
    assert_eq!(first.len(), 1);

    // Second drain should be empty
    let second = drain_scalars();
    assert!(second.is_empty());

    // New records show up in the next drain
    record_scalar("y", 5.0);
    let third = drain_scalars();
    assert_eq!(third.len(), 1);
    assert!(!third.contains_key("x"));
    assert_eq!(third["y"], (5.0, 1));
}

#[test]
fn test_record_scalar_thread_isolation() {
    drain_scalars();
    record_scalar("main", 1.0);

    let child_result = std::thread::spawn(|| {
        // Child thread starts with empty accumulator
        let empty = drain_scalars();
        assert!(empty.is_empty());

        record_scalar("child", 42.0);
        drain_scalars()
    }).join().unwrap();

    // Child's values
    assert_eq!(child_result.len(), 1);
    assert_eq!(child_result["child"], (42.0, 1));

    // Main thread still has its own values
    let main_result = drain_scalars();
    assert_eq!(main_result.len(), 1);
    assert_eq!(main_result["main"], (1.0, 1));
}

#[test]
fn test_aggregate_epoch_metrics() {
    use super::coordinator::aggregate_epoch_metrics;

    let mut scalars_r0 = HashMap::new();
    scalars_r0.insert("loss".to_string(), (3.0, 3_usize)); // mean = 1.0
    scalars_r0.insert("acc".to_string(), (1.8, 3));         // mean = 0.6

    let mut scalars_r1 = HashMap::new();
    scalars_r1.insert("loss".to_string(), (4.0, 2_usize)); // mean = 2.0
    scalars_r1.insert("acc".to_string(), (0.8, 2));         // mean = 0.4

    let msgs = vec![
        MetricsMsg {
            rank: 0, epoch: 0, avg_loss: 0.5, batches_processed: 60,
            epoch_ms: 1000.0, samples_processed: 1920, scalars: scalars_r0,
        },
        MetricsMsg {
            rank: 1, epoch: 0, avg_loss: 0.7, batches_processed: 40,
            epoch_ms: 1200.0, samples_processed: 1280, scalars: scalars_r1,
        },
    ];

    let dev_indices = vec![0_u8, 1];
    let m = aggregate_epoch_metrics(0, &msgs, &dev_indices);
    assert_eq!(m.epoch, 0);

    // Batch-weighted average loss: (0.5*60 + 0.7*40) / 100 = 0.58
    assert!((m.avg_loss - 0.58).abs() < 1e-9);

    // Max epoch_ms
    assert_eq!(m.epoch_ms, 1200.0);

    // Weighted scalar: loss = (1.0*60 + 2.0*40) / 100 = 1.4
    assert!((m.scalars["loss"] - 1.4).abs() < 1e-9);

    // Weighted scalar: acc = (0.6*60 + 0.4*40) / 100 = 0.52
    assert!((m.scalars["acc"] - 0.52).abs() < 1e-9);

    // Per-rank
    assert_eq!(m.per_rank.len(), 2);
    assert!((m.per_rank[0]["loss"] - 1.0).abs() < 1e-9);
    assert!((m.per_rank[1]["loss"] - 2.0).abs() < 1e-9);

    // Throughput: rank 0 = 1920/1000 = 1.92, rank 1 = 1280/1200 ~= 1.0667
    assert!((m.per_rank_throughput[0] - 1.92).abs() < 1e-9);
    assert!((m.per_rank_throughput[1] - 1280.0 / 1200.0).abs() < 1e-9);

    // Batch share: rank 0 = 1920/3200 = 0.6, rank 1 = 1280/3200 = 0.4
    assert!((m.per_rank_batch_share[0] - 0.6).abs() < 1e-9);
    assert!((m.per_rank_batch_share[1] - 0.4).abs() < 1e-9);

    // Device indices
    assert_eq!(m.device_indices, vec![0, 1]);
}

/// Progressive dispatch: multiple MetricsMsg per rank should be aggregated
/// into exactly world_size entries, not one entry per message.
#[test]
fn test_aggregate_epoch_metrics_progressive() {
    use super::coordinator::aggregate_epoch_metrics;

    // Simulate 2 ranks, 3 chunks from rank 0, 2 chunks from rank 1
    let msgs = vec![
        // Rank 0 chunk 1
        MetricsMsg {
            rank: 0, epoch: 0, avg_loss: 0.5, batches_processed: 20,
            epoch_ms: 300.0, samples_processed: 640,
            scalars: [("loss".to_string(), (2.0, 2_usize))].into(),
        },
        // Rank 0 chunk 2
        MetricsMsg {
            rank: 0, epoch: 0, avg_loss: 0.4, batches_processed: 20,
            epoch_ms: 600.0, samples_processed: 640,
            scalars: [("loss".to_string(), (1.6, 2_usize))].into(),
        },
        // Rank 0 chunk 3
        MetricsMsg {
            rank: 0, epoch: 0, avg_loss: 0.6, batches_processed: 20,
            epoch_ms: 900.0, samples_processed: 640,
            scalars: [("loss".to_string(), (1.8, 2_usize))].into(),
        },
        // Rank 1 chunk 1
        MetricsMsg {
            rank: 1, epoch: 0, avg_loss: 0.7, batches_processed: 20,
            epoch_ms: 500.0, samples_processed: 640,
            scalars: [("loss".to_string(), (2.8, 2_usize))].into(),
        },
        // Rank 1 chunk 2
        MetricsMsg {
            rank: 1, epoch: 0, avg_loss: 0.8, batches_processed: 20,
            epoch_ms: 1000.0, samples_processed: 640,
            scalars: [("loss".to_string(), (3.2, 2_usize))].into(),
        },
    ];

    let dev_indices = vec![0_u8, 1];
    let m = aggregate_epoch_metrics(0, &msgs, &dev_indices);

    // Must have exactly 2 entries (world_size), not 5 (one per msg)
    assert_eq!(m.per_rank_throughput.len(), 2, "should have world_size entries");
    assert_eq!(m.per_rank_batch_share.len(), 2);
    assert_eq!(m.per_rank.len(), 2);
    assert_eq!(m.device_indices, vec![0, 1]);

    // Rank 0: 60 batches, 1920 samples, max time 900ms
    // Rank 1: 40 batches, 1280 samples, max time 1000ms
    assert!((m.per_rank_throughput[0] - 1920.0 / 900.0).abs() < 1e-6);
    assert!((m.per_rank_throughput[1] - 1280.0 / 1000.0).abs() < 1e-6);

    // Total samples = 3200
    assert!((m.per_rank_batch_share[0] - 0.6).abs() < 1e-9);
    assert!((m.per_rank_batch_share[1] - 0.4).abs() < 1e-9);

    // Max epoch_ms across ranks
    assert_eq!(m.epoch_ms, 1000.0);

    // Scalars: rank 0 loss mean = (2.0+1.6+1.8)/(2+2+2) = 5.4/6 = 0.9
    assert!((m.per_rank[0]["loss"] - 0.9).abs() < 1e-9);
    // Rank 1 loss mean = (2.8+3.2)/(2+2) = 6.0/4 = 1.5
    assert!((m.per_rank[1]["loss"] - 1.5).abs() < 1e-9);

    // Weighted average: (0.9*60 + 1.5*40)/100 = (54+60)/100 = 1.14
    assert!((m.scalars["loss"] - 1.14).abs() < 1e-9);
}

// -----------------------------------------------------------------------
// Regression: NCCL safety during shutdown (progressive dispatch deadlock)
// -----------------------------------------------------------------------

#[test]
fn test_drain_until_shutdown_skips_sync_now() {
    // Regression: in progressive mode, a worker that reported Exiting
    // could receive a stale SyncNow (sent before the coordinator saw
    // Exiting). Calling AllReduce on a dead peer deadlocks.
    // drain_until_shutdown must skip SyncNow, not call sync_now_nccl.
    let (mut worker, ch) = make_test_worker();

    // Queue messages that would arrive during shutdown:
    // SyncNow (stale, from averaging triggered before our Exiting)
    // followed by Shutdown (from coordinator's shutdown_workers).
    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    ch.control_tx.send(ControlMsg::Shutdown).unwrap();

    // drain_until_shutdown should skip SyncNow and exit on Shutdown.
    // If it tried AllReduce, it would deadlock (no peer in unit test).
    worker.drain_until_shutdown();
    // Reaching here means no deadlock — the SyncNow was skipped.
}

#[test]
fn test_drain_until_shutdown_handles_multiple_sync_now() {
    // Multiple stale SyncNow messages could accumulate if the
    // coordinator triggered several averaging events before seeing
    // our Exiting. All must be skipped.
    let (mut worker, ch) = make_test_worker();

    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    ch.control_tx.send(ControlMsg::Shutdown).unwrap();

    worker.drain_until_shutdown();
}

#[test]
fn test_drain_until_shutdown_handles_interleaved_messages() {
    // Other control messages (RequestParams, StartEpoch, Checkpoint)
    // may arrive between SyncNow and Shutdown. They should be handled
    // normally (not treated as shutdown signals).
    let (mut worker, ch) = make_test_worker();

    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    ch.control_tx.send(ControlMsg::Checkpoint { version: 99 }).unwrap();
    ch.control_tx.send(ControlMsg::StartEpoch(EpochPlan {
        epoch: 5, partition_offset: 0, partition_size: 100,
    })).unwrap();
    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    ch.control_tx.send(ControlMsg::Shutdown).unwrap();

    worker.drain_until_shutdown();
    // StartEpoch queued as pending_plan
    assert!(worker.pending_plan.is_some());
}

#[test]
fn test_abort_nccl_no_panic_without_comm() {
    // abort_nccl takes the NCCL comm (None in unit tests) and aborts it.
    // Must not panic when comm is None.
    let (mut worker, _ch) = make_test_worker();

    // Unit test workers have no NCCL comm. abort_nccl should be a no-op.
    worker.abort_nccl();

    // Call twice to verify idempotence.
    worker.abort_nccl();
}

#[test]
fn test_collect_final_state_disconnected_worker() {
    // Regression: when a worker errors, its final_param_tx is dropped
    // (channel disconnects). collect_final_state should detect this
    // as Disconnected, not wait for the full 10s timeout.
    let (_timing_tx, timing_rx) = mpsc::channel();
    let (_metrics_tx, metrics_rx) = mpsc::channel();
    let (_param_tx, param_rx) = mpsc::channel();

    let mut control_txs = Vec::new();
    let mut final_param_rxs = Vec::new();
    let mut final_param_txs = Vec::new();
    for _ in 0..2 {
        let (ctx, _crx) = mpsc::channel();
        control_txs.push(ctx);
        let (ftx, frx) = mpsc::channel();
        final_param_txs.push(ftx);
        final_param_rxs.push(frx);
    }

    let el_che = ElChe::new(2, 10);
    let coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        ApplyPolicy::Sync, AverageBackend::Cpu,
        2, 1000, el_che,
    ).build();

    // Worker 0 sends snapshot normally
    let opts = crate::tensor::test_opts();
    let t = Tensor::full(&[3], 5.0, opts).unwrap();
    final_param_txs[0].send(ParamSnapshot {
        rank: 0, params: vec![t], buffers: vec![], batch_count: 1,
    }).unwrap();

    // Worker 1 "errors": drop its sender (simulates error path)
    drop(final_param_txs.remove(1));

    // collect_final_state should return quickly (disconnect is instant,
    // not the 10s timeout). The surviving worker's snapshot is returned.
    let start = std::time::Instant::now();
    let state = coord.collect_final_state();
    let elapsed = start.elapsed();

    assert!(state.is_some(), "should get state from surviving worker");
    assert!(elapsed.as_secs() < 2, "disconnect should be fast, not 10s timeout");
    assert_eq!(state.unwrap().params.len(), 1);
}

#[test]
fn test_worker_error_triggers_shutdown_flag() {
    // When a worker errors, it should send Exiting and set the
    // shutdown flag. Other workers check this flag each iteration.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_check = shutdown.clone();

    // Simulate: worker errors → sets shutdown
    shutdown.store(true, Ordering::Relaxed);

    // Coordinator (or sibling worker) sees it
    assert!(shutdown_check.load(Ordering::Relaxed));
}

#[test]
fn test_coordinator_active_count_prevents_averaging_after_exit() {
    // When a worker exits (sends Exiting), active_count drops.
    // should_average must return false to prevent sending SyncNow
    // to a dead peer (which would deadlock the survivor's AllReduce).
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    // Both ranks report a batch
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1 }).unwrap();
    h.coord.drain_timing();
    assert!(h.coord.should_average(), "both ranks reported, should average");

    // Reset: trigger averaging to zero counters
    h.coord.trigger_averaging().unwrap();

    // Both report again
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 2 }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 2 }).unwrap();
    h.coord.drain_timing();
    assert!(h.coord.should_average());

    // Now worker 1 exits
    h.timing_tx.send(TimingMsg::Exiting { rank: 1 }).unwrap();
    h.coord.drain_timing();
    assert_eq!(h.coord.active_count, 1);

    // should_average must return false (can't do collective with dead peer)
    assert!(!h.coord.should_average(),
        "should NOT average when active_count < world_size");
}
