//! Host CPU identity: model name, usable thread count, package count.

/// The CPU's marketing name, such as `AMD Ryzen 9 7950X 16-Core
/// Processor` or `Apple M2 Pro`.
///
/// - **Linux**: `model name` in `/proc/cpuinfo`. ARM kernels print no
///   such line, only implementer and part codes, so the fallback is
///   `lscpu`, which decodes them (`Neoverse-V2`). A chip with two core
///   designs reads as both, `Cortex-A55 + Cortex-A76`.
/// - **macOS**: the `machdep.cpu.brand_string` sysctl.
/// - **Windows**: `ProcessorNameString` under the registry's
///   `CentralProcessor\0` key, read with `reg query`.
///
/// `None` when nothing names the CPU. Read once per process (a summary
/// line, `fdl setup`), so a subprocess is fine here.
pub fn cpu_model() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        crate::darwin::sysctl_string(c"machdep.cpu.brand_string")
    }
    #[cfg(windows)]
    {
        windows_cpu_model()
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        linux_cpu_model()
    }
}

/// Logical CPUs this process may run on.
///
/// This is `std::thread::available_parallelism`, so on Linux it honors
/// the CPU affinity mask and a cgroup quota: a container limited to 8
/// CPUs on a 64-core host reports 8, which is what a training process
/// can actually use. On an unconstrained host it is every logical CPU.
/// `1` when the platform cannot tell.
pub fn cpu_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

#[cfg(not(any(target_os = "macos", windows)))]
fn linux_cpu_model() -> Option<String> {
    if let Some(name) = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .as_deref()
        .and_then(parse_cpuinfo_model)
    {
        return Some(name);
    }
    // lscpu translates its labels, so ask for the untranslated ones.
    let out = std::process::Command::new("lscpu")
        .env("LC_ALL", "C")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_lscpu_model(&String::from_utf8_lossy(&out.stdout))
}

/// The first `model name : <name>` line of `/proc/cpuinfo`.
#[cfg(any(not(any(target_os = "macos", windows)), test))]
fn parse_cpuinfo_model(text: &str) -> Option<String> {
    text.lines()
        .find_map(|l| {
            l.strip_prefix("model name")?
                .split_once(':')
                .map(|(_, v)| v.trim())
        })
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

/// Every distinct `Model name:` in `lscpu` output, joined with ` + `.
/// lscpu prints `-` when it cannot decode a part.
#[cfg(any(not(any(target_os = "macos", windows)), test))]
fn parse_lscpu_model(text: &str) -> Option<String> {
    let mut names: Vec<&str> = Vec::new();
    for line in text.lines() {
        if let Some((label, value)) = line.split_once(':')
            && label.trim() == "Model name"
        {
            let v = value.trim();
            if !v.is_empty() && v != "-" && !names.contains(&v) {
                names.push(v);
            }
        }
    }
    (!names.is_empty()).then(|| names.join(" + "))
}

#[cfg(windows)]
fn windows_cpu_model() -> Option<String> {
    let out = std::process::Command::new("reg")
        .args([
            "query",
            r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            "/v",
            "ProcessorNameString",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_reg_query(&String::from_utf8_lossy(&out.stdout))
}

/// The value line of `reg query ... /v ProcessorNameString`, which reads
/// `    ProcessorNameString    REG_SZ    <name>`.
#[cfg(any(windows, test))]
fn parse_reg_query(text: &str) -> Option<String> {
    text.lines()
        .find_map(|l| {
            l.trim_start()
                .strip_prefix("ProcessorNameString")?
                .split_once("REG_SZ")
                .map(|(_, v)| v.trim())
        })
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

/// How many CPU packages (sockets) the kernel reports, or `None` off
/// Linux / when the topology is not exposed.
///
/// Matters for one specific case: a **multi-package APU**. Each package
/// carries its own memory and its own GPU aperture, while [`crate::mem_info`]
/// reads `/proc/meminfo`, which is SYSTEM-WIDE. So on such a machine
/// "subtract this GPU's memory from host RAM" is wrong in both
/// directions — every rank would either subtract its own package's share
/// from the whole-system total (collapsing the budget to nothing) or
/// treat the whole-system total as its own (over-committing by the
/// package count).
///
/// There is no per-package `MemAvailable` to fall back on: the kernel
/// only computes that estimate system-wide, so
/// `/sys/devices/system/node/N/` offers `MemFree` and nothing
/// equivalent. Hence detect-and-refuse rather than
/// detect-and-approximate.
///
/// # Why packages and not NUMA nodes
///
/// They are not the same count, and reading nodes here refuses runs that
/// are perfectly sizeable. AMD's NPS and Intel's Sub-NUMA Clustering
/// partition **one** package's memory controllers into several NUMA
/// nodes: an MI300A node booted NPS4 reports four nodes on a single
/// socket. The memory is still one physical pool with one aperture
/// carved out of it, so the subtraction is exact — but a node-count test
/// sees 4 > 1 and refuses. Packages are what actually multiply the
/// aperture, and unlike a NUMA layout the count does not move with a
/// firmware setting.
///
/// A count of 1 (the overwhelmingly common case, including every
/// consumer APU) means the system-wide figures ARE this package's
/// figures and the budget math is sound.
pub fn cpu_package_count() -> Option<usize> {
    let dir = std::fs::read_dir("/sys/devices/system/cpu").ok()?;
    let mut ids: Vec<u32> = Vec::new();
    for entry in dir.flatten() {
        let name = entry.file_name();
        let Some(rest) = name.to_str().and_then(|s| s.strip_prefix("cpu")) else {
            continue;
        };
        if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        // Offline CPUs and the `cpuidle` / `cpufreq` pseudo-entries have
        // no topology dir; skipping them is correct, since a package
        // with no online CPU holds no rank either.
        let Ok(id) = std::fs::read_to_string(entry.path().join("topology/physical_package_id"))
        else {
            continue;
        };
        let Ok(id) = id.trim().parse::<u32>() else {
            continue;
        };
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    (!ids.is_empty()).then_some(ids.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_x86_cpuinfo() {
        let text = "processor\t: 0\n\
                    vendor_id\t: AuthenticAMD\n\
                    model name\t: AMD Ryzen 9 7950X 16-Core Processor\n\
                    cpu MHz\t\t: 4500.000\n";
        assert_eq!(
            parse_cpuinfo_model(text).as_deref(),
            Some("AMD Ryzen 9 7950X 16-Core Processor")
        );
    }

    #[test]
    fn arm_cpuinfo_names_no_model() {
        // Why the lscpu fallback exists: an aarch64 kernel lists the
        // part as a hex code and no `model name` line at all.
        let text = "processor\t: 0\n\
                    BogoMIPS\t: 50.00\n\
                    Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n\
                    CPU implementer\t: 0x41\n\
                    CPU architecture: 8\n\
                    CPU variant\t: 0x3\n\
                    CPU part\t: 0xd0c\n\
                    CPU revision\t: 1\n";
        assert_eq!(parse_cpuinfo_model(text), None);
    }

    #[test]
    fn parses_lscpu_model() {
        let arm = "Architecture:             aarch64\n\
                   CPU(s):                   4\n\
                   Vendor ID:                ARM\n  \
                   Model name:             Neoverse-N1\n    \
                   Model:                1\n";
        assert_eq!(parse_lscpu_model(arm).as_deref(), Some("Neoverse-N1"));

        let big_little = "Vendor ID:                ARM\n  \
                          Model name:             Cortex-A55\n  \
                          Model name:             Cortex-A76\n  \
                          Model name:             Cortex-A55\n";
        assert_eq!(
            parse_lscpu_model(big_little).as_deref(),
            Some("Cortex-A55 + Cortex-A76")
        );

        let undecoded = "BIOS Model name:          To Be Filled\n  \
                         Model name:             -\n";
        assert_eq!(parse_lscpu_model(undecoded), None);
    }

    #[test]
    fn parses_reg_query() {
        let text = "\r\n\
                    HKEY_LOCAL_MACHINE\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0\r\n    \
                    ProcessorNameString    REG_SZ    Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz\r\n\r\n";
        assert_eq!(
            parse_reg_query(text).as_deref(),
            Some("Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz")
        );
        assert_eq!(parse_reg_query(""), None);
    }

    #[test]
    fn names_and_counts_the_live_cpu() {
        // These platforms must answer: a None here is an "Unknown CPU"
        // line on a machine that could have been named.
        if cfg!(any(target_os = "linux", target_os = "macos", windows)) {
            let name = cpu_model().expect("no CPU name on a platform that has a probe");
            assert!(!name.is_empty());
            assert!(
                std::thread::available_parallelism().is_ok(),
                "thread count fell back to 1"
            );
        }
        assert!(cpu_threads() >= 1);
    }

    #[test]
    fn package_count_is_at_least_one_where_topology_is_exposed() {
        // Every machine that answers at all has one socket or more. The
        // value gates a hard refusal in the data-plane budget, so a
        // bogus 0 would be worse than no answer.
        if let Some(n) = cpu_package_count() {
            assert!(n >= 1);
        }
    }
}
