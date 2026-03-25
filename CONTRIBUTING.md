# Contributing to floDl

Thank you for your interest in floDl. Contributions are welcome and appreciated.

## Getting Started

floDl builds against libtorch via FFI, so all development happens inside Docker:

```bash
git clone https://github.com/fab2s/floDl.git
cd floDl
make image      # build dev container (Rust + libtorch)
make shell      # interactive shell inside the container
make test       # run all tests
make clippy     # lint
```

You do **not** need Rust or libtorch installed on the host machine.

## Development Workflow

1. Fork the repository and create your branch from `main`.
2. Make your changes inside the dev container (`make shell`).
3. Run `make test` to verify all tests pass.
4. Run `make clippy` to ensure zero warnings.
5. Open a pull request.

## Code Style

- Standard Rust conventions: `rustfmt`, zero clippy warnings.
- Keep the API consistent with existing patterns.
- Every fallible operation returns `Result<T>` — use `?` for propagation.
- Every differentiable operation needs a backward function and a numerical
  gradient check in the autograd tests.
- Public types and methods should have `///` doc comments.

## What We're Looking For

**High value contributions:**
- New NN modules (with forward, backward, parameter collection, and gradient checks)
- New autograd operations (with backward and numerical verification)
- Performance improvements to the FFI dispatch path
- Bug fixes with reproducing tests
- **Backend support**: AMD ROCm, Apple MPS, Intel XPU — the architecture is
  ready (libtorch supports them), but the `Device` enum, FFI shim, and resource
  monitoring need extending. If you have hardware we don't, this is a great way
  to contribute. See the [architecture section](README.md#architecture) for context.

**Also welcome:**
- Documentation improvements and examples
- Doc tests for public APIs
- CI improvements

**Please discuss first:**
- Changes to public API signatures
- New dependencies
- Architecture changes

Open an issue to discuss before investing significant effort on these.

## Testing

Every PR should pass the existing test suite on **both CPU and CUDA**:

```bash
make test          # CPU tests
make cuda-test     # CUDA tests (requires NVIDIA GPU + Container Toolkit)
make test-all      # CPU first, then CUDA if a GPU is available
```

All tests use `test_device()` / `test_opts()` from `tensor.rs` so the same
test code runs on whichever device is available. When writing new tests:

- Use `test_device()` instead of `Device::CPU` for device selection
- Use `test_opts()` instead of `TensorOptions::default()` or `Default::default()`
- Use `on_device(..., test_device())` constructors instead of `::new()` for modules
- Tests that are inherently CPU-only (e.g. RSS-based leak checks) should guard
  with `if test_device() != Device::CPU { return; }` at the top

**Test template:**
```rust
#[test]
fn test_my_feature() {
    let dev = test_device();
    let opts = test_opts();

    let input = Tensor::randn(&[2, 4], opts).unwrap();
    let layer = Linear::on_device(4, 2, dev).unwrap();
    let x = Variable::new(input, true);
    let y = layer.forward(&x).unwrap();

    assert_eq!(y.data().shape(), vec![2, 2]);
}
```

If you add new functionality:

- **Tensor ops**: add tests in `tensor.rs`
- **Autograd ops**: add a numerical gradient check
- **NN modules**: add both a functional test and a gradient check
- **Graph features**: add a test in the graph module
- **Module constructors**: always provide an `on_device()` variant alongside `new()`

## Before Publishing to crates.io

Always validate the docs.rs build locally before publishing. docs.rs uses nightly
Rust with `--cfg docsrs` and no libtorch — things that build fine in the dev
container can fail there.

```bash
make docs-rs    # simulates docs.rs build in a disposable container
```

This catches:
- Broken intra-doc links (`rustdoc::broken_intra_doc_links`)
- Dependencies that don't compile on nightly with `--cfg docsrs`
- Example scraping failures (examples need libtorch)
- Missing `#[cfg(docsrs)]` gates on FFI code

crates.io is immutable — a broken publish means bumping the version. Run this
before every `cargo publish`.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).
