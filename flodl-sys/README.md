# flodl-sys

Raw FFI bindings to [libtorch](https://pytorch.org/cppdocs/) via a thin C++ shim (`shim.h` / `shim.cpp`). This crate is the foundation layer for [flodl](https://flodl.dev).

**You should not use this crate directly.** Use the [`flodl`](https://crates.io/crates/flodl) crate instead, which provides a safe Rust API for tensors, autograd, neural network modules, and graph-based model composition.

## Build requirements

The `build.rs` script expects libtorch to be available. Set `LIBTORCH_PATH` to the path of your libtorch installation, or use the Docker-based build provided by the main repository.

## License

floDl is open-sourced software licensed under the [MIT license](https://github.com/flodl-labs/flodl/blob/main/LICENSE).

Repository: <https://github.com/flodl-labs/flodl>
