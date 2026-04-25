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
//! Argv parsing + `--help` + `--fdl-schema` are handled by `flodl-cli`'s
//! [`FdlArgs`] derive, so help is identical whether the user types
//! `fdl flodl-hf export -h` (rendered by fdl from the cached schema) or
//! `fdl flodl-hf export` with missing args (rendered by this binary
//! through the same renderer).
//!
//! Supported families: bert, roberta, distilbert, xlm-roberta, albert,
//! deberta-v2 (dispatched via `AutoConfig::from_json_str` on the repo's
//! `model_type`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use flodl_cli::{parse_or_schema, FdlArgs};
use flodl_hf::export::export_hf_dir;
use flodl_hf::models::auto::{AutoConfig, AutoModel};

/// Export a Hub repo as a HuggingFace-compatible directory (model.safetensors + config.json) using flodl's own writer. Auto-detects family (bert/roberta/distilbert/xlm-roberta/albert/deberta-v2).
#[derive(FdlArgs, Debug)]
struct ExportArgs {
    /// HuggingFace repo id (e.g. `bert-base-uncased`).
    #[arg]
    hf_repo_id: String,
    /// Output directory; written as
    /// `<out_dir>/model.safetensors` + `<out_dir>/config.json`.
    #[arg]
    out_dir: String,
}

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

fn main() -> ExitCode {
    let cli: ExportArgs = parse_or_schema();
    let out_dir = resolve_out_dir(&cli.out_dir);

    eprintln!("fetching config.json for {} ...", cli.hf_repo_id);
    let config = match AutoConfig::from_pretrained(&cli.hf_repo_id) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("detected family: {}", config.model_type());

    eprintln!("loading weights for {} ...", cli.hf_repo_id);
    // `_for_export` keeps the family pooler so HF AutoModel.from_pretrained
    // gets every weight it expects on reload. Default `from_pretrained`
    // strips the pooler for cross-family `last_hidden_state` consistency,
    // which is wrong for export.
    let graph = match AutoModel::from_pretrained_for_export(&cli.hf_repo_id) {
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
        cli.hf_repo_id,
        out_dir.display(),
    );
    ExitCode::SUCCESS
}
