//! libtorch installation detection and .arch metadata parsing.

use std::fs;
use std::path::Path;

use crate::util::system::GpuInfo;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Metadata about an installed libtorch variant (from `.arch` file).
pub struct LibtorchInfo {
    /// Relative path from project root (e.g. "precompiled/cu128", "builds/sm61-sm120").
    pub path: String,
    pub torch_version: Option<String>,
    pub cuda_version: Option<String>,
    pub archs: Option<String>,
    pub source: Option<String>,
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// Read the active libtorch variant from `<root>/libtorch/.active` and parse
/// its `.arch` metadata.
pub fn read_active(root: &Path) -> Option<LibtorchInfo> {
    let active_path = root.join("libtorch/.active");
    let active = fs::read_to_string(active_path).ok()?;
    let path = active.trim().to_string();
    if path.is_empty() {
        return None;
    }

    let arch_path = root.join(format!("libtorch/{}/.arch", path));
    let mut info = LibtorchInfo {
        path,
        torch_version: None,
        cuda_version: None,
        archs: None,
        source: None,
    };

    if let Ok(content) = fs::read_to_string(arch_path) {
        for line in content.lines() {
            if let Some(val) = line.strip_prefix("torch=") {
                info.torch_version = Some(val.to_string());
            } else if let Some(val) = line.strip_prefix("cuda=") {
                info.cuda_version = Some(val.to_string());
            } else if let Some(val) = line.strip_prefix("archs=") {
                info.archs = Some(val.to_string());
            } else if let Some(val) = line.strip_prefix("source=") {
                info.source = Some(val.to_string());
            }
        }
    }

    Some(info)
}

/// List all installed libtorch variants under `<root>/libtorch/`.
///
/// Scans `precompiled/` and `builds/` subdirectories.
pub fn list_variants(root: &Path) -> Vec<String> {
    let mut variants = Vec::new();
    let lt_dir = root.join("libtorch");

    for subdir in ["precompiled", "builds"] {
        let dir = lt_dir.join(subdir);
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if entry.path().join("lib").is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        variants.push(format!("{}/{}", subdir, name));
                    }
                }
            }
        }
    }

    variants.sort();
    variants
}

/// Check whether a GPU's compute capability is covered by the libtorch
/// variant's compiled architectures (from the .arch file).
pub fn arch_compatible(gpu: &GpuInfo, archs: &str) -> bool {
    let exact = format!("{}.{}", gpu.sm_major, gpu.sm_minor);
    archs.contains(&exact) || archs.contains(&format!("{}", gpu.sm_major))
}

/// Check whether a libtorch variant directory looks valid (has lib/).
pub fn is_valid_variant(root: &Path, variant: &str) -> bool {
    root.join(format!("libtorch/{}/lib", variant)).is_dir()
}

/// Set the active libtorch variant by writing `<root>/libtorch/.active`.
pub fn set_active(root: &Path, variant: &str) -> Result<(), String> {
    let lt_dir = root.join("libtorch");
    fs::create_dir_all(&lt_dir)
        .map_err(|e| format!("cannot create libtorch/: {}", e))?;
    fs::write(lt_dir.join(".active"), format!("{}\n", variant))
        .map_err(|e| format!("cannot write libtorch/.active: {}", e))
}
