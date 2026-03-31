use std::env;
use std::path::PathBuf;

fn main() {
    // docs.rs builds without libtorch — skip C++ compilation entirely.
    // cargo doc does not link, so unresolved extern symbols are fine.
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    let libtorch = env::var("LIBTORCH_PATH")
        .unwrap_or_else(|_| "/usr/local/libtorch".to_string());
    let libtorch = PathBuf::from(&libtorch);

    // Compile shim.cpp as C++17
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("shim.cpp")
        .include(".")
        .include(libtorch.join("include"))
        .include(libtorch.join("include/torch/csrc/api/include"))
        .warnings(false);

    if cfg!(feature = "cuda") {
        build.define("FLODL_BUILD_CUDA", "1");
        // CUDA toolkit headers
        let cuda_home = env::var("CUDA_HOME")
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        build.include(format!("{}/include", cuda_home));
    }

    build.compile("flodl_shim");

    // Link libtorch shared libraries
    println!("cargo:rustc-link-search=native={}", libtorch.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    if cfg!(feature = "cuda") {
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
        println!("cargo:rustc-link-lib=dylib=c10_cuda");

        let cuda_home = env::var("CUDA_HOME")
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
        println!("cargo:rustc-link-lib=dylib=cudart");

        // dlopen for NVML GPU utilization queries
        println!("cargo:rustc-link-lib=dylib=dl");

        // NCCL for multi-GPU collective operations
        println!("cargo:rustc-link-lib=dylib=nccl");
    }

    // Rerun if sources change
    println!("cargo:rerun-if-changed=shim.cpp");
    println!("cargo:rerun-if-changed=shim.h");
    println!("cargo:rerun-if-env-changed=LIBTORCH_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
