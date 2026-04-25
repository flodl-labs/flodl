//! Export a HuggingFace-compatible model directory from a Hub repo id.
//!
//! Fetches `config.json` + `model.safetensors` for `<repo_id>`, rebuilds
//! the matching flodl graph via `AutoModel::from_pretrained`, then writes
//! `model.safetensors` + `config.json` to `<out_dir>` via
//! [`flodl_hf::export::export_hf_dir`]. The output directory is ready for
//! HF Python's `AutoModel.from_pretrained("<out_dir>")`.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf export bert-base-uncased /tmp/bert-exported
//! # or, inside the dev container directly:
//! cargo run --release --example export_hf -- bert-base-uncased /tmp/bert-exported
//! ```
//!
//! Supported families: bert, roberta, distilbert, xlm-roberta, albert,
//! deberta-v2 (dispatched via `AutoConfig::from_json_str` on the repo's
//! `model_type`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use flodl_hf::export::export_hf_dir;
use flodl_hf::models::auto::{AutoConfig, AutoModel};

/// Anchor a relative `out_dir` path against `FDL_PROJECT_ROOT` when set.
///
/// `fdl` injects this env var inside docker-compose-managed services so
/// argv paths resolve from the host shell's invocation root regardless
/// of the container's `working_dir` (which `fdl` overrides per-task to
/// keep `cd flodl-hf/scripts` etc. working). Without this, a user typing
/// `flodl-hf/tests/.exports/bert` from the repo root lands the export
/// at `<container-cwd>/flodl-hf/tests/.exports/bert` instead of
/// `<workspace-root>/flodl-hf/tests/.exports/bert`. Absolute paths and
/// host-side runs are unaffected.
fn resolve_out_dir(arg: &str) -> PathBuf {
    let p = Path::new(arg);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    if let Some(root) = std::env::var_os("FDL_PROJECT_ROOT") {
        return PathBuf::from(root).join(p);
    }
    p.to_path_buf()
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: export_hf <hf_repo_id> <out_dir>\n\
         \n\
         example: export_hf bert-base-uncased /tmp/bert-exported\n\
         \n\
         Writes <out_dir>/model.safetensors + <out_dir>/config.json.\n\
         Supported families: bert, roberta, distilbert, xlm-roberta, albert, deberta-v2.",
    );
    ExitCode::from(64) // EX_USAGE
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(repo_id) = args.next() else {
        return usage();
    };
    let Some(out_dir) = args.next() else {
        return usage();
    };
    if args.next().is_some() {
        eprintln!("error: unexpected extra arguments");
        return usage();
    }
    let out_dir = resolve_out_dir(&out_dir);

    eprintln!("fetching config.json for {repo_id} ...");
    let config = match AutoConfig::from_pretrained(&repo_id) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("detected family: {}", config.model_type());

    eprintln!("loading weights for {repo_id} ...");
    // `_for_export` keeps the family pooler so HF AutoModel.from_pretrained
    // gets every weight it expects on reload. Default `from_pretrained`
    // strips the pooler for cross-family `last_hidden_state` consistency,
    // which is wrong for export.
    let graph = match AutoModel::from_pretrained_for_export(&repo_id) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    eprintln!("exporting to {} ...", out_dir.display());
    if let Err(e) = export_hf_dir(&graph, &config.to_json_str(), &out_dir) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }

    println!(
        "exported {} → {}\n  model.safetensors + config.json ready for AutoModel.from_pretrained",
        repo_id,
        out_dir.display(),
    );
    ExitCode::SUCCESS
}
