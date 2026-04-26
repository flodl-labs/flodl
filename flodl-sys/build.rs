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

    // Preflight: confirm libtorch is actually present before cc::Build
    // launches a multi-minute compile that would otherwise fail with a
    // cryptic `fatal error: torch/torch.h: No such file or directory`
    // deep in the C++ output. Pointing users at `fdl setup` is the
    // canonical fix; the manual override is documented for users who
    // are bypassing fdl on purpose.
    // Match the same header file cc::Build's include path resolves
    // (`include/torch/csrc/api/include`); presence here is the
    // canonical "libtorch is installed" sentinel for both the
    // pre-built and source-built variants.
    let torch_header = libtorch
        .join("include/torch/csrc/api/include/torch/torch.h");
    if !torch_header.exists() {
        eprintln!(
            "\nflodl-sys: libtorch not found at `{}`\n\
             (expected `{}` to exist).\n\n\
             Recommended fix: install `flodl-cli` and run `fdl setup` from\n\
             your project root. It auto-detects your hardware, downloads or\n\
             builds the matching libtorch variant, and points LIBTORCH_PATH\n\
             at it for you.\n\n\
             Manual override: set LIBTORCH_PATH=/path/to/libtorch where the\n\
             directory contains both `include/torch/csrc/api/include/torch/torch.h`\n\
             and `lib/libtorch.so` (or the platform equivalent).\n",
            libtorch.display(),
            torch_header.display(),
        );
        std::process::exit(1);
    }

    // Unity build: shim.cpp #includes the topic-focused ops_*.cpp files so the
    // C++ compiler parses torch/torch.h exactly once. Splitting into separate
    // TUs would multiply torch.h parse cost (~17s/TU) since cc::Build rebuilds
    // every TU on any change.
    //
    // Files listed for cargo:rerun-if-changed below; only shim.cpp is compiled.
    let shim_includes = [
        "shim.h",
        "helpers.h",
        "ops_tensor.cpp",
        "ops_nn.cpp",
        "ops_math_ext.cpp",
        "ops_training.cpp",
        "ops_cuda.cpp",
    ];

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

    // Rerun if sources change (shim.cpp + every #included unit + headers).
    println!("cargo:rerun-if-changed=shim.cpp");
    for src in &shim_includes {
        println!("cargo:rerun-if-changed={}", src);
    }
    println!("cargo:rerun-if-env-changed=LIBTORCH_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
