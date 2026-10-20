//! Cross-platform system detection (CPU, RAM, OS, Docker, GPU).

use std::path::Path;
use std::process::Command;

// ---------------------------------------------------------------------------
// GPU detection
// ---------------------------------------------------------------------------
//
// The implementation lives in the dependency-free `flodl-hw` crate, which
// `flodl` depends on too. It does NOT pull libtorch, so fdl still builds and
// runs before libtorch is installed. `GpuInfo` + the nvidia-smi parse used to
// be hand-copied between here and `flodl::sys`, kept aligned by a comment;
// there is now one source.
//
// Note the mapping: fdl's `detect_gpus` never honored `CUDA_VISIBLE_DEVICES`,
// and that is correct for the questions fdl asks ("which libtorch variant
// covers this box"), which a container mask must not change the answer to. It
// is therefore `detect_gpus_physical` upstream. `flodl_hw::detect_gpus` is the
// mask-honoring runtime view, used by `flodl`.

pub use flodl_hw::{
    GpuInfo, GpuVendor, cpu_model, cpu_threads, detect_gpus_physical as detect_gpus,
    nvidia_driver_version,
};

// ---------------------------------------------------------------------------
// RAM
// ---------------------------------------------------------------------------

/// Total physical RAM in whole GiB, `0` when the host cannot be read.
pub fn ram_total_gb() -> u64 {
    flodl_hw::mem_info().map_or(0, |m| m.total_gb())
}

// ---------------------------------------------------------------------------
// OS
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
pub fn os_version() -> Option<String> {
    let uname = Command::new("uname").arg("-r").output().ok()?;
    let kernel = String::from_utf8_lossy(&uname.stdout).trim().to_string();
    let wsl = if kernel.contains("WSL") || kernel.contains("microsoft") {
        " (WSL2)"
    } else {
        ""
    };
    Some(format!("Linux {}{}", kernel, wsl))
}

#[cfg(target_os = "macos")]
pub fn os_version() -> Option<String> {
    let out = Command::new("sw_vers")
        .args(["-productVersion"])
        .output()
        .ok()?;
    let ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if ver.is_empty() {
        return None;
    }
    let arch = Command::new("uname")
        .arg("-m")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    if arch.is_empty() {
        Some(format!("macOS {}", ver))
    } else {
        Some(format!("macOS {} ({})", ver, arch))
    }
}

#[cfg(target_os = "windows")]
pub fn os_version() -> Option<String> {
    let out = Command::new("cmd").args(["/C", "ver"]).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

// ---------------------------------------------------------------------------
// Docker
// ---------------------------------------------------------------------------

pub fn is_inside_docker() -> bool {
    Path::new("/.dockerenv").exists()
}

pub fn docker_version() -> Option<String> {
    let out = Command::new("docker").arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    s.split("version ")
        .nth(1)
        .and_then(|v| v.split(',').next())
        .map(|v| v.trim().to_string())
}

/// Check whether cargo is available on the host.
#[allow(dead_code)]
pub fn has_cargo() -> bool {
    Command::new("cargo")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check whether a command exists on PATH.
#[allow(dead_code)]
pub fn has_command(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

/// Platform string for download URLs (e.g. "linux-x86_64", "macos-arm64").
#[allow(dead_code)]
pub fn platform_tag() -> Option<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    match (os, arch) {
        ("linux", "x86_64") => Some("linux-x86_64".into()),
        ("macos", "aarch64") => Some("macos-arm64".into()),
        ("windows", "x86_64") => Some("windows-x86_64".into()),
        _ => None,
    }
}

/// Escape a string for embedding in a JSON string literal. Complete per
/// RFC 8259: backslash, quote, and every control char below 0x20. The
/// hand-rolled predecessors missed `\t` / `\r` — one control character in
/// a GPU name or mount path produced invalid JSON, which broke cluster
/// probe fan-in.
pub fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = std::fmt::Write::write_fmt(&mut out, format_args!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Convert a `;`-separated arch list into a variant directory name.
/// Shared by the libtorch and NCCL source builders so their variant
/// paths cannot drift.
///
/// NVIDIA capabilities take the `sm` prefix and lose their dot:
/// `"6.1;12.0"` -> `"sm61-sm120"`. AMD tokens are already their own
/// name and pass through: `"gfx1030;gfx1100"` -> `"gfx1030-gfx1100"`.
/// Prefixing those would produce `smgfx1030`, which
/// `detect::variant_vendor` would then fail to recognise as AMD.
pub fn arch_dir_name(archs: &str) -> String {
    archs
        .split(';')
        .map(|tok| {
            let tok = tok.trim();
            // Only the AMD parse accepts a `gfx…` token, so it doubles
            // as the discriminator (and normalises case + any
            // `:sramecc±:xnack±` suffix on the way through).
            match flodl_hw::GpuArch::parse(flodl_hw::GpuVendor::Amd, tok) {
                Some(arch) => arch.to_string(),
                None => format!("sm{}", tok.replace('.', "")),
            }
        })
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(test)]
mod tests {
    use super::{arch_dir_name, escape_json};

    #[test]
    fn escape_json_passes_through_plain_ascii() {
        assert_eq!(escape_json("hello world 123"), "hello world 123");
    }

    #[test]
    fn escape_json_escapes_quotes_and_backslashes() {
        // A Windows-style path with quotes is the realistic hazard for the
        // probe/diagnose JSON output this feeds.
        assert_eq!(escape_json(r#"C:\a\b"#), r#"C:\\a\\b"#);
        assert_eq!(escape_json(r#"say "hi""#), r#"say \"hi\""#);
    }

    #[test]
    fn escape_json_escapes_named_control_chars() {
        assert_eq!(escape_json("a\nb\rc\td"), "a\\nb\\rc\\td");
        assert_eq!(escape_json("\u{08}\u{0C}"), "\\b\\f");
    }

    #[test]
    fn escape_json_uescapes_other_control_chars() {
        // < 0x20 with no short form -> \uXXXX (lowercase, 4 hex digits).
        assert_eq!(escape_json("\u{01}"), "\\u0001");
        assert_eq!(escape_json("\u{1f}"), "\\u001f");
    }

    #[test]
    fn escape_json_leaves_non_ascii_unescaped() {
        // >= 0x20 passes through verbatim, including multibyte UTF-8.
        assert_eq!(escape_json("café — 日本"), "café — 日本");
    }

    #[test]
    fn arch_dir_name_single() {
        assert_eq!(arch_dir_name("12.0"), "sm120");
    }

    #[test]
    fn arch_dir_name_multi() {
        assert_eq!(arch_dir_name("6.1;12.0"), "sm61-sm120");
    }

    #[test]
    fn arch_dir_name_strips_all_dots() {
        // A three-component cap and a two-digit minor both flatten correctly.
        assert_eq!(arch_dir_name("7.5"), "sm75");
        assert_eq!(arch_dir_name("8.0;8.6;9.0"), "sm80-sm86-sm90");
    }
}
