# flodl-sys

Raw FFI bindings to [libtorch](https://pytorch.org/cppdocs/) via a thin C++ shim (`shim.h` / `shim.cpp`). This crate is the foundation layer for [floDl](https://flodl.dev), a Rust deep learning framework. The main entry point is the [`flodl`](https://crates.io/crates/flodl) crate; see [flodl.dev](https://flodl.dev) and the [main README](https://github.com/flodl-labs/flodl#readme) for the full framework.

**You should not use this crate directly.** Use the [`flodl`](https://crates.io/crates/flodl) crate instead, which provides a safe Rust API for tensors, autograd, neural network modules, and graph-based model composition.

## Build requirements

The `build.rs` script expects libtorch to be available. The easy path is the [`flodl-cli`](https://crates.io/crates/flodl-cli) (`fdl`) tool, which downloads the right libtorch variant for your hardware:

```sh
curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl
./fdl libtorch download
```

Or set `LIBTORCH_PATH` to a libtorch installation you manage yourself. The Docker-based build in the [main repository](https://github.com/flodl-labs/flodl) wires this up automatically.

## License

floDl is open-sourced software licensed under the [MIT license](https://github.com/flodl-labs/flodl/blob/main/LICENSE).

Repository: <https://github.com/flodl-labs/flodl>
