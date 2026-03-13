use std::cell::Cell;

thread_local! {
    static NO_GRAD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Returns true if gradient computation is enabled.
pub fn is_grad_enabled() -> bool {
    NO_GRAD_DEPTH.with(|d| d.get() == 0)
}

/// RAII guard that disables gradient computation while alive.
///
/// Analogous to PyTorch's `torch.no_grad()` context manager.
/// Gradients are re-enabled when the guard is dropped — including
/// on panic unwind, so a panic inside a no-grad block cannot
/// permanently disable gradients on the thread.
///
/// Prefer the [`no_grad`] function for simple cases. Use `NoGradGuard`
/// directly when you need scoped no-grad without a closure:
///
/// ```ignore
/// {
///     let _guard = NoGradGuard::new();
///     let pred = model.forward(&x)?;
///     // gradients re-enabled when _guard drops
/// }
/// ```
pub struct NoGradGuard {
    // prevent construction outside this module
    _private: (),
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl NoGradGuard {
    /// Disable gradient computation until this guard is dropped.
    pub fn new() -> Self {
        NO_GRAD_DEPTH.with(|d| d.set(d.get() + 1));
        NoGradGuard { _private: () }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        NO_GRAD_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

/// Execute a closure with gradient computation disabled.
///
/// Operations inside `no_grad` will not build a backward graph,
/// reducing memory usage for inference and parameter updates.
/// Safe across panics — the guard is dropped even if `f` panics.
///
/// ```ignore
/// let pred = no_grad(|| model.forward(&x).unwrap());
/// assert!(!pred.requires_grad());
/// ```
///
/// For scoped no-grad without a closure, see [`NoGradGuard`].
///
/// Per-thread: each thread has its own depth counter. This matches
/// Rust's Rc-based autograd (single-threaded by design).
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGradGuard::new();
    f()
}
