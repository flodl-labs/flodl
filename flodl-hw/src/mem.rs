//! Host RAM detection.

/// Host RAM snapshot: total and currently-available bytes.
///
/// `available_bytes` is memory a new workload can take without pushing
/// the system into swap, reclaimable page cache included. Each platform
/// states it differently (see [`mem_info`]), but that is the promise.
/// It makes the figure the honest baseline for sizing staging buffers,
/// since it already accounts for every other process on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemInfo {
    /// Total physical RAM in bytes.
    pub total_bytes: u64,
    /// RAM a new workload can take without swapping, in bytes.
    pub available_bytes: u64,
}

impl MemInfo {
    /// Total physical RAM in whole GiB, rounded down. `0` on a host
    /// whose totals could not be read.
    pub fn total_gb(&self) -> u64 {
        self.total_bytes / (1024 * 1024 * 1024)
    }
}

/// Read host RAM totals.
///
/// - **Linux**: `MemTotal` and `MemAvailable` from `/proc/meminfo`.
/// - **macOS** has no single equivalent, so the figure is assembled
///   from the kernel's page counters (the ones `vm_stat` prints) and
///   `hw.memsize`: free pages, plus file-backed pages (page cache,
///   dropped without writing anything to swap), plus purgeable pages
///   (memory its owner marked discardable). Inactive anonymous pages
///   are left out on purpose: reclaiming them means compressing or
///   swapping, which is what the figure promises not to cause.
///   Speculative pages are left out too, since they may already sit in
///   the file-backed count, so any error is on the low side.
/// - **Windows**: `GlobalMemoryStatusEx`, whose available figure (the
///   standby, free and zeroed page lists) is that same promise.
///
/// A file read or a few system calls, no subprocess, so it is cheap
/// enough for a sampler polling several times a second.
///
/// Returns `None` on other platforms, or when the source cannot be
/// read. Callers then keep nothing in their RAM tiers and give flow
/// buffers a small fixed size, rather than trusting a guess.
pub fn mem_info() -> Option<MemInfo> {
    #[cfg(target_os = "macos")]
    {
        macos::mem_info()
    }
    #[cfg(windows)]
    {
        windows::mem_info()
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        parse_meminfo(&std::fs::read_to_string("/proc/meminfo").ok()?)
    }
}

/// macOS available RAM from page counts, by the rule at [`mem_info`].
#[cfg(any(target_os = "macos", test))]
fn macos_mem_from_pages(
    free: u64,
    file_backed: u64,
    purgeable: u64,
    page_size: u64,
    total_bytes: u64,
) -> Option<MemInfo> {
    let pages = free.checked_add(file_backed)?.checked_add(purgeable)?;
    Some(MemInfo {
        total_bytes,
        available_bytes: pages.saturating_mul(page_size).min(total_bytes),
    })
}

#[cfg(target_os = "macos")]
mod macos {
    use super::MemInfo;
    use std::ffi::{c_char, c_int, c_void};
    use std::sync::OnceLock;

    /// `struct vm_statistics64` from `<mach/vm_statistics.h>`
    /// (`natural_t` is 32-bit). Only four counters are read; the rest
    /// are here to keep the layout.
    #[repr(C, align(8))]
    #[derive(Default)]
    #[allow(dead_code)]
    struct VmStatistics64 {
        free_count: u32,
        active_count: u32,
        inactive_count: u32,
        wire_count: u32,
        zero_fill_count: u64,
        reactivations: u64,
        pageins: u64,
        pageouts: u64,
        faults: u64,
        cow_faults: u64,
        lookups: u64,
        hits: u64,
        purges: u64,
        purgeable_count: u32,
        speculative_count: u32,
        decompressions: u64,
        compressions: u64,
        swapins: u64,
        swapouts: u64,
        compressor_page_count: u32,
        throttled_count: u32,
        external_page_count: u32,
        internal_page_count: u32,
        total_uncompressed_pages_in_compressor: u64,
    }

    const HOST_VM_INFO64: c_int = 4;
    const KERN_SUCCESS: c_int = 0;

    unsafe extern "C" {
        fn mach_host_self() -> u32;
        fn host_page_size(host: u32, page_size: *mut usize) -> c_int;
        fn host_statistics64(host: u32, flavor: c_int, info: *mut c_int, count: *mut u32) -> c_int;
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// The host port, fetched once: each `mach_host_self` call adds a
    /// reference to it, and a sampler would add ten a second.
    fn host() -> u32 {
        static HOST: OnceLock<u32> = OnceLock::new();
        // SAFETY: takes nothing, returns a port name.
        *HOST.get_or_init(|| unsafe { mach_host_self() })
    }

    pub(super) fn mem_info() -> Option<MemInfo> {
        let mut total: u64 = 0;
        let mut len = size_of::<u64>();
        // SAFETY: `hw.memsize` is a u64, `len` says so, and nothing is
        // written back to the kernel.
        let rc = unsafe {
            sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&raw mut total).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 || len != size_of::<u64>() || total == 0 {
            return None;
        }

        let mut page_size: usize = 0;
        // SAFETY: a single out-parameter of type `vm_size_t`.
        if unsafe { host_page_size(host(), &mut page_size) } != KERN_SUCCESS || page_size == 0 {
            return None;
        }

        let mut stats = VmStatistics64::default();
        let mut count = (size_of::<VmStatistics64>() / size_of::<c_int>()) as u32;
        // SAFETY: `count` is the buffer's size in `integer_t` units, so
        // the kernel writes at most that much into `stats`.
        let rc = unsafe {
            host_statistics64(host(), HOST_VM_INFO64, (&raw mut stats).cast(), &mut count)
        };
        if rc != KERN_SUCCESS {
            return None;
        }

        // `free_count` includes the speculative pages; `vm_stat`'s
        // "Pages free" subtracts them, and so does the rule.
        super::macos_mem_from_pages(
            u64::from(stats.free_count.saturating_sub(stats.speculative_count)),
            u64::from(stats.external_page_count),
            u64::from(stats.purgeable_count),
            page_size as u64,
            total,
        )
    }
}

#[cfg(windows)]
mod windows {
    use super::MemInfo;

    /// `MEMORYSTATUSEX` from `<sysinfoapi.h>`.
    #[repr(C)]
    #[derive(Default)]
    #[allow(dead_code)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    pub(super) fn mem_info() -> Option<MemInfo> {
        let mut status = MemoryStatusEx {
            length: size_of::<MemoryStatusEx>() as u32,
            ..Default::default()
        };
        // SAFETY: a correctly sized `MEMORYSTATUSEX` with `dwLength`
        // set, which is all the call requires.
        if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 || status.total_phys == 0 {
            return None;
        }
        Some(MemInfo {
            total_bytes: status.total_phys,
            available_bytes: status.avail_phys.min(status.total_phys),
        })
    }
}

/// Parse `MemTotal:` / `MemAvailable:` lines (values are in kB).
#[cfg(any(not(any(target_os = "macos", windows)), test))]
fn parse_meminfo(text: &str) -> Option<MemInfo> {
    let mut total = None;
    let mut available = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = parse_meminfo_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available = parse_meminfo_kb(rest);
        }
        if total.is_some() && available.is_some() {
            break;
        }
    }
    Some(MemInfo {
        total_bytes: total? * 1024,
        available_bytes: available? * 1024,
    })
}

#[cfg(any(not(any(target_os = "macos", windows)), test))]
fn parse_meminfo_kb(rest: &str) -> Option<u64> {
    rest.trim().strip_suffix("kB")?.trim().parse().ok()
}

/// How many CPU packages (sockets) the kernel reports, or `None` off
/// Linux / when the topology is not exposed.
///
/// Matters for one specific case: a **multi-package APU**. Each package
/// carries its own memory and its own GPU aperture, while [`mem_info`]
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
    fn parses_meminfo_format() {
        let text = "MemTotal:       131781120 kB\n\
                    MemFree:         8123456 kB\n\
                    MemAvailable:   98765432 kB\n\
                    Buffers:          123456 kB\n";
        let m = parse_meminfo(text).unwrap();
        assert_eq!(m.total_bytes, 131_781_120 * 1024);
        assert_eq!(m.available_bytes, 98_765_432 * 1024);
        assert_eq!(m.total_gb(), 125);
    }

    #[test]
    fn missing_memavailable_is_none_not_a_guess() {
        // Pre-3.14 kernels lack MemAvailable. Refusing to answer beats
        // inventing a number the budget logic would trust.
        assert!(parse_meminfo("MemTotal: 100 kB\nMemFree: 50 kB\n").is_none());
        assert!(parse_meminfo("").is_none());
    }

    /// Parse `vm_stat` output by the same rule as the macOS probe.
    ///
    /// `vm_stat` prints the counters the probe reads through a
    /// hand-written C struct, so on a Mac it is the reference that
    /// catches a misplaced field: that reads a neighbouring counter and
    /// yields a plausible wrong number, not a crash. The header names
    /// the page size and every counter line reads `<label>: <pages>.`.
    fn parse_vm_stat(text: &str, total_bytes: u64) -> Option<MemInfo> {
        let mut lines = text.lines();
        let page_size: u64 = lines
            .next()?
            .split("page size of ")
            .nth(1)?
            .split_whitespace()
            .next()?
            .parse()
            .ok()?;
        let (mut free, mut file_backed, mut purgeable) = (None, None, None);
        for line in lines {
            let Some((label, value)) = line.split_once(':') else {
                continue;
            };
            let pages = value.trim().trim_end_matches('.').parse::<u64>().ok();
            match label.trim() {
                "Pages free" => free = pages,
                "File-backed pages" => file_backed = pages,
                "Pages purgeable" => purgeable = pages,
                _ => {}
            }
        }
        macos_mem_from_pages(free?, file_backed?, purgeable?, page_size, total_bytes)
    }

    /// `vm_stat`'s layout on Apple Silicon (16 KiB pages). The counts
    /// are illustrative; inactive and speculative are deliberately large
    /// so that counting either one would change the result.
    const VM_STAT: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
        Pages free:                                3000.\n\
        Pages active:                            400000.\n\
        Pages inactive:                          390000.\n\
        Pages speculative:                        20000.\n\
        Pages throttled:                              0.\n\
        Pages wired down:                        150000.\n\
        Pages purgeable:                           1000.\n\
        \"Translation faults\":                  9876543.\n\
        Pages copy-on-write:                     123456.\n\
        Pages zero filled:                      2345678.\n\
        Pages reactivated:                        12345.\n\
        Pages purged:                              1234.\n\
        File-backed pages:                       300000.\n\
        Anonymous pages:                         492000.\n\
        Pages stored in compressor:               50000.\n\
        Pages occupied by compressor:             20000.\n\
        Decompressions:                           45678.\n\
        Compressions:                             56789.\n\
        Pageins:                                  67890.\n\
        Pageouts:                                     0.\n\
        Swapins:                                      0.\n\
        Swapouts:                                     0.\n";

    const SIXTEEN_GIB: u64 = 16 * 1024 * 1024 * 1024;

    #[test]
    fn parses_vm_stat_format() {
        let m = parse_vm_stat(VM_STAT, SIXTEEN_GIB).unwrap();
        assert_eq!(m.total_bytes, SIXTEEN_GIB);
        // free + file-backed + purgeable, in 16 KiB pages.
        assert_eq!(m.available_bytes, (3000 + 300_000 + 1000) * 16384);
    }

    #[test]
    fn macos_available_never_exceeds_total() {
        let one_gib = 1024 * 1024 * 1024;
        let m = macos_mem_from_pages(3000, 300_000, 1000, 16384, one_gib).unwrap();
        assert_eq!(m.available_bytes, one_gib);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_probe_agrees_with_vm_stat() {
        let ours = mem_info().expect("macOS probe");
        let memsize = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .expect("sysctl");
        let memsize: u64 = String::from_utf8_lossy(&memsize.stdout)
            .trim()
            .parse()
            .expect("hw.memsize");
        assert_eq!(ours.total_bytes, memsize);
        let vm_stat = std::process::Command::new("vm_stat")
            .output()
            .expect("vm_stat");
        let theirs = parse_vm_stat(&String::from_utf8_lossy(&vm_stat.stdout), memsize)
            .expect("vm_stat output parses");
        // Two snapshots a few milliseconds apart: free pages move a
        // little between them, a misread counter moves a lot.
        let drift = ours.available_bytes.abs_diff(theirs.available_bytes);
        assert!(
            drift <= memsize / 20,
            "probe {} vs vm_stat {} bytes available",
            ours.available_bytes,
            theirs.available_bytes
        );
    }

    #[test]
    fn reads_live_host() {
        // These platforms must answer: there a None silently turns off
        // every RAM tier, which is the failure this probe exists to
        // prevent. Elsewhere None is the contract.
        let m = mem_info();
        if cfg!(any(target_os = "linux", target_os = "macos", windows)) {
            let m = m.expect("no host RAM reading on a platform that has a probe");
            assert!(m.total_bytes > 0);
            assert!(m.available_bytes > 0);
            assert!(m.available_bytes <= m.total_bytes);
        } else {
            assert!(m.is_none());
        }
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
