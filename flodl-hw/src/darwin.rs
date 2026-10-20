//! macOS `sysctlbyname`, shared by the RAM and CPU probes.

use std::ffi::{CStr, c_char, c_int, c_void};

unsafe extern "C" {
    fn sysctlbyname(
        name: *const c_char,
        oldp: *mut c_void,
        oldlenp: *mut usize,
        newp: *mut c_void,
        newlen: usize,
    ) -> c_int;
}

/// A 64-bit integer sysctl, such as `hw.memsize`.
pub(crate) fn sysctl_u64(name: &CStr) -> Option<u64> {
    let mut value: u64 = 0;
    let mut len = size_of::<u64>();
    // SAFETY: `value` is a u64, `len` says so, and nothing is written
    // back to the kernel.
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == size_of::<u64>()).then_some(value)
}

/// A string sysctl, such as `machdep.cpu.brand_string`. `None` when it
/// is missing or empty.
pub(crate) fn sysctl_string(name: &CStr) -> Option<String> {
    let mut len = 0usize;
    // SAFETY: a null buffer asks only for the length.
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len];
    // SAFETY: `buf` holds `len` bytes and `len` says so.
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    buf.truncate(len);
    let s = CStr::from_bytes_until_nul(&buf).ok()?.to_str().ok()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}
