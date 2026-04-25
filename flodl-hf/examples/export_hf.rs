//! Export a HuggingFace-compatible model directory.
//!
//! Two input modes (mutually exclusive, exactly one required):
//!
//! - `--hub <repo_id>`: fetch `config.json` + `model.safetensors` from
//!   the HuggingFace Hub, rebuild the matching flodl graph via
//!   `AutoModel::from_pretrained_for_export`, and write the export.
//! - `--checkpoint <file.fdl>`: load weights from a local flodl
//!   checkpoint, rebuild the matching architecture from the sidecar
//!   `<stem>.config.json` (or an explicit `--config <file>`) and write
//!   the export.
//!
//! In both modes the output is `<out_dir>/model.safetensors` +
//! `<out_dir>/config.json`, ready for HF Python's
//! `AutoModel.from_pretrained("<out_dir>")`.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf export --hub bert-base-uncased --out /tmp/bert-exported
//! fdl flodl-hf export --checkpoint /tmp/my.fdl --out /tmp/my-exported
//! # or, inside the dev container directly:
//! cargo run --release --example export_hf --features hub -- --hub bert-base-uncased --out /tmp/bert-exported
//! ```
//!
//! Argv parsing + `--help` + `--fdl-schema` are handled by `flodl-cli`'s
//! [`FdlArgs`] derive, so help is identical whether the user types
//! `fdl flodl-hf export -h` (rendered by fdl from the cached schema) or
//! `fdl flodl-hf export` with missing args (rendered by this binary
//! through the same renderer).
//!
//! Supported families: bert, roberta, distilbert, xlm-roberta, albert,
//! deberta-v2 (dispatched via `AutoConfig::from_json_str` on the
//! checkpoint's `model_type`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use flodl::{Device, Graph};
use flodl_cli::{parse_or_schema, FdlArgs, FdlArgsTrait};
use flodl_hf::export::{build_for_export, export_hf_dir, keys_have_pooler};
use flodl_hf::models::auto::{AutoConfig, AutoModel};

/// Export a Hub repo or local flodl checkpoint as a HuggingFace-compatible directory (model.safetensors + config.json) using flodl's own writer. Auto-detects family (bert/roberta/distilbert/xlm-roberta/albert/deberta-v2).
#[derive(FdlArgs, Debug)]
struct ExportArgs {
    /// HuggingFace repo id to fetch (e.g. `bert-base-uncased`).
    /// Mutex with `--checkpoint`; exactly one is required.
    #[option]
    hub: Option<String>,
    /// Local flodl `.fdl` checkpoint to load. Reads the matching
    /// architecture from the sidecar `<stem>.config.json` (or `--config`).
    /// Mutex with `--hub`; exactly one is required.
    #[option]
    checkpoint: Option<String>,
    /// Output directory; written as
    /// `<out>/model.safetensors` + `<out>/config.json`. Required.
    #[option]
    out: Option<String>,
    /// Override the source `config.json` (checkpoint mode only).
    /// Defaults to reading the sidecar next to `--checkpoint`.
    #[option]
    config: Option<String>,
    /// Overwrite an existing `<out>` directory's `model.safetensors` /
    /// `config.json` without prompting.
    #[option]
    force: bool,
    /// Write the loaded source config verbatim to `<out>/config.json`
    /// instead of regenerating via the family's `to_json_str`.
    /// Checkpoint mode only.
    #[option]
    preserve_source_config: bool,
}

/// Anchor a relative path against `FDL_PROJECT_ROOT` when set.
///
/// `fdl` injects this env var inside docker-compose-managed services so
/// argv paths resolve from the host shell's invocation root regardless
/// of the container's `working_dir`. Absolute paths and host-side runs
/// are unaffected.
fn resolve_path(arg: &str) -> PathBuf {
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
    match dispatch(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(DispatchError::Usage(msg)) => {
            // Mirror parse_or_schema_from's error UX: print the message
            // followed by the rendered help so the user sees BOTH the
            // specific complaint and the full flag list.
            eprintln!("error: {msg}");
            eprintln!();
            eprintln!("{}", ExportArgs::render_help());
            ExitCode::from(2)
        }
        Err(DispatchError::Runtime(msg)) => {
            // Hub fetch failures, checkpoint IO, family-build errors —
            // the help text wouldn't add value, so just surface the
            // error.
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// Two-tone error type to keep the help-on-usage-error UX surgical:
/// only render the full help when the user's argv shape is wrong, not
/// for every runtime failure mid-export.
enum DispatchError {
    /// Argv-shape problem (missing flag, mutex violation, ...).
    Usage(String),
    /// Runtime failure (network, IO, parse, model build, ...).
    Runtime(String),
}

impl From<flodl::TensorError> for DispatchError {
    fn from(e: flodl::TensorError) -> Self {
        DispatchError::Runtime(format!("{e}"))
    }
}

fn dispatch(cli: &ExportArgs) -> Result<(), DispatchError> {
    // Mutex enforcement: exactly one of --hub / --checkpoint required.
    match (cli.hub.is_some(), cli.checkpoint.is_some()) {
        (true, true) => {
            return Err(DispatchError::Usage(
                "--hub and --checkpoint are mutually exclusive; pass exactly one."
                    .into(),
            ));
        }
        (false, false) => {
            return Err(DispatchError::Usage(
                "missing required input: pass --hub <repo_id> or --checkpoint <file.fdl>."
                    .into(),
            ));
        }
        _ => {}
    }

    if cli.preserve_source_config && cli.hub.is_some() {
        return Err(DispatchError::Usage(
            "--preserve-source-config requires --checkpoint (Hub mode regenerates config from to_json_str)."
                .into(),
        ));
    }
    if cli.config.is_some() && cli.hub.is_some() {
        return Err(DispatchError::Usage(
            "--config is only meaningful with --checkpoint.".into(),
        ));
    }

    let out_arg = cli
        .out
        .as_deref()
        .ok_or_else(|| DispatchError::Usage("missing required --out <dir>.".into()))?;
    let out_dir = resolve_path(out_arg);
    if !cli.force {
        let model_path = out_dir.join("model.safetensors");
        let config_path = out_dir.join("config.json");
        if model_path.exists() || config_path.exists() {
            return Err(DispatchError::Runtime(format!(
                "{} already contains model.safetensors or config.json. Pass --force to overwrite.",
                out_dir.display(),
            )));
        }
    }

    if let Some(repo_id) = cli.hub.as_deref() {
        run_hub(repo_id, &out_dir)?;
    } else if let Some(checkpoint_path) = cli.checkpoint.as_deref() {
        run_checkpoint(
            checkpoint_path,
            cli.config.as_deref(),
            &out_dir,
            cli.preserve_source_config,
        )?;
    }

    Ok(())
}

/// Hub mode: fetch and rebuild via `AutoModel::from_pretrained_for_export`.
fn run_hub(repo_id: &str, out_dir: &Path) -> flodl::Result<()> {
    eprintln!("fetching config.json for {repo_id} ...");
    let config = AutoConfig::from_pretrained(repo_id)?;
    eprintln!("detected family: {}", config.model_type());

    eprintln!("loading weights for {repo_id} ...");
    let graph = AutoModel::from_pretrained_for_export(repo_id)?;

    eprintln!("exporting to {} ...", out_dir.display());
    export_hf_dir(&graph, &config.to_json_str(), out_dir)?;

    println!(
        "exported {repo_id} → {}\n  model.safetensors + config.json ready for AutoModel.from_pretrained",
        out_dir.display(),
    );
    Ok(())
}

/// Checkpoint mode: read sidecar (or --config), build matching graph,
/// load `.fdl`, write export.
fn run_checkpoint(
    checkpoint_path: &str,
    config_override: Option<&str>,
    out_dir: &Path,
    preserve_source_config: bool,
) -> flodl::Result<()> {
    let checkpoint_path = resolve_path(checkpoint_path);
    let checkpoint_str = checkpoint_path.to_string_lossy();

    // Resolve config source: --config wins; fall back to sidecar.
    let config_str = if let Some(cfg) = config_override {
        let cfg_path = resolve_path(cfg);
        eprintln!("reading config from {} ...", cfg_path.display());
        std::fs::read_to_string(&cfg_path).map_err(|e| {
            flodl::TensorError::new(&format!(
                "cannot read --config {}: {e}",
                cfg_path.display()
            ))
        })?
    } else {
        let sidecar = sidecar_for(&checkpoint_path);
        if !sidecar.exists() {
            return Err(flodl::TensorError::new(&format!(
                "no sidecar config at {}; pass --config <file> to override (or save the checkpoint via flodl-hf so the sidecar is emitted automatically)",
                sidecar.display(),
            )));
        }
        eprintln!("reading sidecar from {} ...", sidecar.display());
        std::fs::read_to_string(&sidecar).map_err(|e| {
            flodl::TensorError::new(&format!(
                "cannot read sidecar {}: {e}",
                sidecar.display()
            ))
        })?
    };

    let config = AutoConfig::from_json_str(&config_str)?;
    eprintln!("detected family: {}", config.model_type());

    // Peek the checkpoint's parameter names to decide pooler-presence.
    let keys = flodl::checkpoint_keys(&checkpoint_str)?;
    let has_pooler = keys_have_pooler(&keys);
    eprintln!(
        "checkpoint declares {} keys, with_pooler={has_pooler}",
        keys.len()
    );

    let graph: Graph = build_for_export(&config, has_pooler, Device::CPU)?;
    let report = graph.load_checkpoint(&checkpoint_str)?;
    eprintln!(
        "loaded {} param(s)/buffer(s); {} skipped, {} missing",
        report.loaded.len(),
        report.skipped.len(),
        report.missing.len(),
    );

    let out_config = if preserve_source_config {
        config_str
    } else {
        config.to_json_str()
    };

    eprintln!("exporting to {} ...", out_dir.display());
    export_hf_dir(&graph, &out_config, out_dir)?;

    println!(
        "exported {} → {}\n  model.safetensors + config.json ready for AutoModel.from_pretrained",
        checkpoint_path.display(),
        out_dir.display(),
    );
    Ok(())
}

/// Resolve the sidecar config path for a checkpoint path. Mirrors
/// `flodl::Graph::save_checkpoint`'s sidecar naming (strip `.fdl` and
/// optional `.gz`, then add `.config.json`).
fn sidecar_for(checkpoint: &Path) -> PathBuf {
    let mut p = checkpoint.to_path_buf();
    if p.extension().and_then(|e| e.to_str()) == Some("gz") {
        p.set_extension("");
    }
    p.set_extension("config.json");
    p
}
