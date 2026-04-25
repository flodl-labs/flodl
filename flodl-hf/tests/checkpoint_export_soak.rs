//! End-to-end soak for the `--checkpoint` re-export pipeline.
//!
//! Validates that the queue #3 (sidecar emission via
//! `AutoModel::from_pretrained_for_export` → `set_source_config` →
//! `save_checkpoint` writes `<stem>.config.json`) and queue #4
//! (`build_for_export` + `load_checkpoint` + `export_hf_dir` reads the
//! sidecar to rebuild the matching topology) paths actually round-trip:
//!
//! 1. `AutoModel::from_pretrained_for_export("bert-base-uncased")`
//!    fetches the Hub config, sets it on the returned graph as
//!    `source_config`.
//! 2. `graph.save_checkpoint(<tmp>.fdl)` emits both the `.fdl` and the
//!    `<tmp>.config.json` sidecar (queue #3 contract).
//! 3. Read the sidecar, parse via `AutoConfig::from_json_str`, instantiate
//!    a fresh graph via `build_for_export`, then load the `.fdl` weights
//!    into it (queue #4 contract — same shape as the example binary's
//!    `run_checkpoint`).
//! 4. `export_hf_dir` writes `model.safetensors` + `config.json`.
//! 5. Compare `model.safetensors` bytes bit-exact against a parallel
//!    direct-from-Hub export — proves --hub and --checkpoint produce
//!    equivalent output.
//!
//! `_live` because step 1 hits the Hub. Run with `fdl test-live`.

use std::path::PathBuf;

use flodl::{checkpoint_keys, Device, Graph};
use flodl_hf::export::{build_for_export, export_hf_dir, keys_have_pooler};
use flodl_hf::models::auto::{AutoConfig, AutoModel};

fn unique_tmp_root(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "flodl_hf_checkpoint_soak_{tag}_{}",
        std::process::id(),
    ))
}

fn sidecar_for_checkpoint(checkpoint: &str) -> PathBuf {
    // Mirrors flodl's pub(crate) `sidecar_config_path`: strip optional
    // `.gz`, then replace the trailing extension with `config.json`.
    let mut p = PathBuf::from(checkpoint);
    if p.extension().and_then(|e| e.to_str()) == Some("gz") {
        p.set_extension("");
    }
    p.set_extension("config.json");
    p
}

#[test]
#[ignore = "live: HF checkpoint-mode soak; requires network + hf-hub cache"]
fn bert_checkpoint_export_pipeline_soak_live() {
    let repo = "bert-base-uncased";
    let root = unique_tmp_root("bert");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    // Step 1: Hub-fetch with source_config attached (queue #3).
    let graph_hub: Graph = AutoModel::from_pretrained_for_export(repo)
        .unwrap_or_else(|e| panic!("AutoModel::from_pretrained_for_export({repo}): {e}"));
    assert!(
        graph_hub.source_config().is_some(),
        "queue #3 contract: AutoModel::from_pretrained_for_export must set source_config",
    );

    // Step 2: save_checkpoint emits the sidecar (queue #3 contract).
    let ckpt = root.join("bert.fdl");
    let ckpt_str = ckpt.to_string_lossy().to_string();
    graph_hub
        .save_checkpoint(&ckpt_str)
        .unwrap_or_else(|e| panic!("save_checkpoint({ckpt_str}): {e}"));
    assert!(ckpt.exists(), "save_checkpoint produced no .fdl");
    let sidecar = sidecar_for_checkpoint(&ckpt_str);
    assert!(
        sidecar.exists(),
        "queue #3 contract: save_checkpoint did not emit sidecar at {}",
        sidecar.display(),
    );

    // Step 3: Read sidecar → parse → build_for_export → load_checkpoint
    // (queue #4 contract — what `run_checkpoint` does in the example
    // binary, here exercised at the library level).
    let sidecar_str = std::fs::read_to_string(&sidecar)
        .unwrap_or_else(|e| panic!("read sidecar {}: {e}", sidecar.display()));
    let config = AutoConfig::from_json_str(&sidecar_str).unwrap();
    assert_eq!(config.model_type(), "bert");
    let keys = checkpoint_keys(&ckpt_str)
        .unwrap_or_else(|e| panic!("checkpoint_keys({ckpt_str}): {e}"));
    let has_pooler = keys_have_pooler(&keys);
    let graph_rebuilt: Graph =
        build_for_export(&config, has_pooler, Device::CPU).unwrap_or_else(|e| {
            panic!("build_for_export from sidecar: {e}");
        });
    let report = graph_rebuilt
        .load_checkpoint(&ckpt_str)
        .unwrap_or_else(|e| panic!("load_checkpoint({ckpt_str}): {e}"));
    assert!(
        report.missing.is_empty(),
        "checkpoint-mode rebuild missed {} keys: {:?}",
        report.missing.len(),
        &report.missing[..report.missing.len().min(5)],
    );

    // Step 4: Export the rebuilt graph (checkpoint mode output).
    let ckpt_export = root.join("checkpoint_export");
    export_hf_dir(&graph_rebuilt, &config.to_json_str(), &ckpt_export).unwrap();

    // Parallel: export the original Hub graph (hub mode reference).
    let hub_export = root.join("hub_export");
    export_hf_dir(&graph_hub, &config.to_json_str(), &hub_export).unwrap();

    // Step 5: Bit-exact compare model.safetensors between the two
    // pipelines. Both should serialise identical parameter sets.
    let ckpt_bytes = std::fs::read(ckpt_export.join("model.safetensors")).unwrap();
    let hub_bytes = std::fs::read(hub_export.join("model.safetensors")).unwrap();
    assert_eq!(
        ckpt_bytes.len(),
        hub_bytes.len(),
        "checkpoint-mode and hub-mode exports differ in size: {} vs {} bytes",
        ckpt_bytes.len(),
        hub_bytes.len(),
    );
    assert!(
        ckpt_bytes == hub_bytes,
        "checkpoint-mode and hub-mode exports differ bit-exact (same length, different bytes)",
    );

    eprintln!(
        "checkpoint soak: {} keys round-tripped; --checkpoint and --hub produced \
         bit-identical {}-byte model.safetensors",
        report.loaded.len(),
        ckpt_bytes.len(),
    );

    let _ = std::fs::remove_dir_all(&root);
}
