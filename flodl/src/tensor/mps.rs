//! Apple MPS queries. The shim is compiled on every host. Availability
//! is a runtime answer: libtorch reports false wherever it has no MPS
//! backend.

use flodl_sys as ffi;

/// Returns true when this process can allocate on the Apple GPU.
pub fn mps_available() -> bool {
    unsafe { ffi::flodl_mps_is_available() != 0 }
}

/// Wait until queued MPS work has finished.
///
/// A no-op when MPS is not available, so a caller that raced the device
/// disappearing does not hit the default hook's throw.
pub fn mps_synchronize() {
    if mps_available() {
        unsafe { ffi::flodl_mps_synchronize() };
    }
}
