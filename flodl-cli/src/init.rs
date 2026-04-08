//! `fdl init <name>` -- scaffold a new floDl project.
//!
//! Two modes:
//! - Default: mounted libtorch (same model as the main flodl repo)
//! - `--docker`: standalone Docker scaffold (libtorch baked into images)

use std::fs;
use std::path::Path;
use std::process::Command;

pub fn run(name: Option<&str>, docker: bool) -> Result<(), String> {
    let name = name.ok_or("usage: fdl init <project-name>")?;
    validate_name(name)?;

    if Path::new(name).exists() {
        return Err(format!("'{}' already exists", name));
    }

    let crate_name = name.replace('-', "_");
    let flodl_dep = resolve_flodl_dep();

    fs::create_dir_all(format!("{}/src", name))
        .map_err(|e| format!("cannot create directory: {}", e))?;

    if docker {
        scaffold_docker(name, &crate_name, &flodl_dep)?;
    } else {
        scaffold_mounted(name, &crate_name, &flodl_dep)?;
    }

    // Shared files
    write_file(
        &format!("{}/src/main.rs", name),
        &main_rs_template(),
    )?;
    write_file(
        &format!("{}/.gitignore", name),
        &gitignore_template(docker),
    )?;

    println!();
    println!("Project '{}' created. Next steps:", name);
    println!();
    println!("  cd {}", name);
    if docker {
        println!("  make build    # first build (downloads libtorch, ~5 min)");
        println!("  make test     # run tests");
        println!("  make run      # train the model");
    } else {
        println!("  ./fdl setup   # detect hardware + download libtorch");
        println!("  make test     # run tests");
        println!("  make run      # train the model");
    }
    println!("  make shell    # interactive shell");
    println!();
    println!("Edit src/main.rs to build your model.");
    println!();
    println!("Guides:");
    println!("  Tutorials:         https://flodl.dev/guide/tensors");
    println!("  Graph Tree:        https://flodl.dev/guide/graph-tree");
    println!("  PyTorch migration: https://flodl.dev/guide/migration");
    println!("  Troubleshooting:   https://flodl.dev/guide/troubleshooting");

    Ok(())
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("project name cannot be empty".into());
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err("project name must contain only letters, digits, hyphens, underscores".into());
    }
    Ok(())
}

fn resolve_flodl_dep() -> String {
    // Try crates.io for the latest version
    if let Some(version) = crates_io_version() {
        format!("flodl = \"{}\"", version)
    } else {
        "flodl = { git = \"https://github.com/fab2s/floDl.git\" }".into()
    }
}

fn crates_io_version() -> Option<String> {
    let output = Command::new("curl")
        .args(["-sL", "https://crates.io/api/v1/crates/flodl"])
        .output()
        .ok()?;
    let body = String::from_utf8_lossy(&output.stdout);
    // Extract "max_stable_version":"X.Y.Z"
    let marker = "\"max_stable_version\":\"";
    let start = body.find(marker)? + marker.len();
    let end = start + body[start..].find('"')?;
    let version = &body[start..end];
    if version.is_empty() { None } else { Some(version.to_string()) }
}

// ---------------------------------------------------------------------------
// Docker scaffold (standalone, libtorch baked into images)
// ---------------------------------------------------------------------------

fn scaffold_docker(name: &str, crate_name: &str, flodl_dep: &str) -> Result<(), String> {
    write_file(
        &format!("{}/Cargo.toml", name),
        &cargo_toml_template(crate_name, flodl_dep),
    )?;
    write_file(
        &format!("{}/Dockerfile.cpu", name),
        DOCKERFILE_CPU,
    )?;
    write_file(
        &format!("{}/Dockerfile.cuda", name),
        DOCKERFILE_CUDA,
    )?;
    write_file(
        &format!("{}/docker-compose.yml", name),
        &docker_compose_template(crate_name, true),
    )?;
    write_file(
        &format!("{}/Makefile", name),
        MAKEFILE_DOCKER,
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Mounted scaffold (libtorch from host, like the main repo)
// ---------------------------------------------------------------------------

fn scaffold_mounted(name: &str, crate_name: &str, flodl_dep: &str) -> Result<(), String> {
    write_file(
        &format!("{}/Cargo.toml", name),
        &cargo_toml_template(crate_name, flodl_dep),
    )?;
    write_file(
        &format!("{}/Dockerfile", name),
        DOCKERFILE_MOUNTED,
    )?;
    write_file(
        &format!("{}/Dockerfile.cuda", name),
        DOCKERFILE_CUDA_MOUNTED,
    )?;
    write_file(
        &format!("{}/docker-compose.yml", name),
        &docker_compose_template(crate_name, false),
    )?;
    write_file(
        &format!("{}/Makefile", name),
        MAKEFILE_MOUNTED,
    )?;

    // Copy fdl bootstrap into the project for self-contained setup
    let fdl_script = include_str!("../assets/fdl");
    write_file(&format!("{}/fdl", name), fdl_script)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(
            format!("{}/fdl", name),
            fs::Permissions::from_mode(0o755),
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------

fn cargo_toml_template(crate_name: &str, flodl_dep: &str) -> String {
    format!(
        r#"[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2024"

[dependencies]
{flodl_dep}

# Optimize floDl in dev builds -- your code stays fast to compile.
# After the first build, only your graph code recompiles (~2s).
[profile.dev.package.flodl]
opt-level = 3

[profile.dev.package.flodl-sys]
opt-level = 3

# Release: cross-crate optimization for maximum throughput.
[profile.release]
lto = "thin"
codegen-units = 1
"#
    )
}

fn main_rs_template() -> String {
    r#"//! floDl training template.
//!
//! This is a starting point for your model. Edit the architecture,
//! data loading, and training loop to fit your task.
//!
//! New to Rust? Read: https://flodl.dev/guide/rust-primer
//! Stuck?       Read: https://flodl.dev/guide/troubleshooting

use flodl::*;
use flodl::monitor::Monitor;

fn main() -> Result<()> {
    // --- Model ---
    let model = FlowBuilder::from(Linear::new(4, 32)?)
        .through(GELU)
        .through(LayerNorm::new(32)?)
        .also(Linear::new(32, 32)?)       // residual connection
        .through(Linear::new(32, 1)?)
        .build()?;

    // --- Optimizer ---
    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 0.001);
    let scheduler = CosineScheduler::new(0.001, 1e-6, 100);
    model.train();

    // --- Data ---
    // Replace this with your data loading.
    let opts = TensorOptions::default();
    let batches: Vec<(Tensor, Tensor)> = (0..32)
        .map(|_| {
            let x = Tensor::randn(&[16, 4], opts).unwrap();
            let y = Tensor::randn(&[16, 1], opts).unwrap();
            (x, y)
        })
        .collect();

    // --- Training loop ---
    let num_epochs = 100usize;
    let mut monitor = Monitor::new(num_epochs);
    // monitor.serve(3000)?;              // uncomment for live dashboard
    // monitor.watch(&model);             // uncomment to show graph SVG
    // monitor.save_html("report.html");  // uncomment to save HTML report

    for epoch in 0..num_epochs {
        let t = std::time::Instant::now();
        let mut epoch_loss = 0.0;

        for (input_t, target_t) in &batches {
            let input = Variable::new(input_t.clone(), true);
            let target = Variable::new(target_t.clone(), false);

            optimizer.zero_grad();
            let pred = model.forward(&input)?;
            let loss = mse_loss(&pred, &target)?;
            loss.backward()?;
            clip_grad_norm(&params, 1.0)?;
            optimizer.step()?;

            epoch_loss += loss.item()?;
        }

        let avg_loss = epoch_loss / batches.len() as f64;
        let lr = scheduler.lr(epoch);
        optimizer.set_lr(lr);
        monitor.log(epoch, t.elapsed(), &[("loss", avg_loss), ("lr", lr)]);
    }

    monitor.finish();
    Ok(())
}
"#
    .into()
}

fn gitignore_template(docker: bool) -> String {
    let mut s = String::from(
        "/target
*.fdl
*.log
*.csv
*.html
",
    );
    if docker {
        s.push_str(
            ".cargo-cache/
.cargo-git/
.cargo-cache-cuda/
.cargo-git-cuda/
",
        );
    } else {
        s.push_str(
            ".cargo-cache/
.cargo-git/
.cargo-cache-cuda/
.cargo-git-cuda/
libtorch/
",
        );
    }
    s
}

fn docker_compose_template(crate_name: &str, baked: bool) -> String {
    if baked {
        format!(
            r#"services:
  dev:
    build:
      context: .
      dockerfile: Dockerfile.cpu
    image: {crate_name}-dev
    user: "${{UID:-1000}}:${{GID:-1000}}"
    volumes:
      - .:/workspace
      - ./.cargo-cache:/usr/local/cargo/registry
      - ./.cargo-git:/usr/local/cargo/git
    working_dir: /workspace
    stdin_open: true
    tty: true

  cuda:
    build:
      context: .
      dockerfile: Dockerfile.cuda
    image: {crate_name}-cuda
    user: "${{UID:-1000}}:${{GID:-1000}}"
    volumes:
      - .:/workspace
      - ./.cargo-cache-cuda:/usr/local/cargo/registry
      - ./.cargo-git-cuda:/usr/local/cargo/git
    working_dir: /workspace
    stdin_open: true
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"#
        )
    } else {
        format!(
            r#"services:
  dev:
    build:
      context: .
      dockerfile: Dockerfile
    image: {crate_name}-dev
    user: "${{UID:-1000}}:${{GID:-1000}}"
    volumes:
      - .:/workspace
      - ./.cargo-cache:/usr/local/cargo/registry
      - ./.cargo-git:/usr/local/cargo/git
      - ${{LIBTORCH_CPU_PATH:-./libtorch/precompiled/cpu}}:/usr/local/libtorch:ro
    working_dir: /workspace
    stdin_open: true
    tty: true

  cuda:
    build:
      context: .
      dockerfile: Dockerfile.cuda
      args:
        CUDA_VERSION: ${{CUDA_VERSION:-12.8.0}}
    image: {crate_name}-cuda:${{CUDA_TAG:-12.8}}
    user: "${{UID:-1000}}:${{GID:-1000}}"
    volumes:
      - .:/workspace
      - ./.cargo-cache-cuda:/usr/local/cargo/registry
      - ./.cargo-git-cuda:/usr/local/cargo/git
      - ${{LIBTORCH_HOST_PATH:-./libtorch/precompiled/cu128}}:/usr/local/libtorch:ro
    working_dir: /workspace
    stdin_open: true
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"#
        )
    }
}

// ---------------------------------------------------------------------------
// Dockerfile templates
// ---------------------------------------------------------------------------

// Docker mode: libtorch baked into images
const DOCKERFILE_CPU: &str = r#"# CPU-only dev image for floDl projects.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip ca-certificates git gcc g++ pkg-config graphviz \
    && rm -rf /var/lib/apt/lists/*

# Rust
ENV CARGO_HOME="/usr/local/cargo"
ENV RUSTUP_HOME="/usr/local/rustup"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && chmod -R a+rwx "$CARGO_HOME" "$RUSTUP_HOME"
ENV PATH="${CARGO_HOME}/bin:${PATH}"

# libtorch (CPU-only, ~200MB)
ARG LIBTORCH_VERSION=2.10.0
RUN wget -q https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip \
    && unzip -q libtorch-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip -d /usr/local \
    && rm libtorch-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip

ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib"

WORKDIR /workspace
"#;

const DOCKERFILE_CUDA: &str = r#"# CUDA dev image for floDl projects.
# Requires: docker run --gpus all ...
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip ca-certificates git gcc g++ pkg-config graphviz \
    && rm -rf /var/lib/apt/lists/*

# Rust
ENV CARGO_HOME="/usr/local/cargo"
ENV RUSTUP_HOME="/usr/local/rustup"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && chmod -R a+rwx "$CARGO_HOME" "$RUSTUP_HOME"
ENV PATH="${CARGO_HOME}/bin:${PATH}"

# libtorch (CUDA 12.8)
ARG LIBTORCH_VERSION=2.10.0
RUN wget -q "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu128.zip" \
    && unzip -q "libtorch-shared-with-deps-${LIBTORCH_VERSION}+cu128.zip" -d /usr/local \
    && rm "libtorch-shared-with-deps-${LIBTORCH_VERSION}+cu128.zip"

ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib:/usr/local/cuda/lib64"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib:/usr/local/cuda/lib64"
ENV CUDA_HOME="/usr/local/cuda"

WORKDIR /workspace
"#;

// Mounted mode: libtorch provided at runtime via volume mount
const DOCKERFILE_MOUNTED: &str = r#"# CPU dev image for floDl projects (libtorch mounted at runtime).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip ca-certificates git gcc g++ pkg-config graphviz \
    && rm -rf /var/lib/apt/lists/*

# Rust
ENV CARGO_HOME="/usr/local/cargo"
ENV RUSTUP_HOME="/usr/local/rustup"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && chmod -R a+rwx "$CARGO_HOME" "$RUSTUP_HOME"
ENV PATH="${CARGO_HOME}/bin:${PATH}"

ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib"

WORKDIR /workspace
"#;

const DOCKERFILE_CUDA_MOUNTED: &str = r#"# CUDA dev image for floDl projects (libtorch mounted at runtime).
# Requires: docker run --gpus all ...
ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip ca-certificates git gcc g++ pkg-config graphviz \
    && rm -rf /var/lib/apt/lists/*

# Rust
ENV CARGO_HOME="/usr/local/cargo"
ENV RUSTUP_HOME="/usr/local/rustup"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && chmod -R a+rwx "$CARGO_HOME" "$RUSTUP_HOME"
ENV PATH="${CARGO_HOME}/bin:${PATH}"

ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib:/usr/local/cuda/lib64"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib:/usr/local/cuda/lib64"
ENV CUDA_HOME="/usr/local/cuda"

WORKDIR /workspace
"#;

// ---------------------------------------------------------------------------
// Makefile templates
// ---------------------------------------------------------------------------

const MAKEFILE_DOCKER: &str = r#"# Development commands -- all builds run inside Docker.
#
# Quick start:
#   make build   -- compile (CPU)
#   make test    -- run tests
#   make run     -- cargo run
#   make shell   -- interactive shell in container
#
# GPU (requires NVIDIA Container Toolkit):
#   make cuda-build / cuda-test / cuda-run / cuda-shell

COMPOSE = docker compose
RUN     = $(COMPOSE) run --rm dev
RUN_GPU = $(COMPOSE) run --rm cuda

.PHONY: build test run check clippy shell image clean \
        cuda-image cuda-build cuda-test cuda-run cuda-shell

# --- CPU targets ---

image:
	@mkdir -p .cargo-cache .cargo-git
	$(COMPOSE) build dev

build: image
	$(RUN) cargo build

test: image
	$(RUN) cargo test -- --nocapture

run: image
	$(RUN) cargo run

check: image
	$(RUN) cargo check

clippy: image
	$(RUN) cargo clippy -- -W clippy::all

shell: image
	$(COMPOSE) run --rm dev bash

# --- CUDA targets ---

cuda-image:
	@mkdir -p .cargo-cache-cuda .cargo-git-cuda
	$(COMPOSE) build cuda

cuda-build: cuda-image
	$(RUN_GPU) cargo build --features cuda

cuda-test: cuda-image
	$(RUN_GPU) cargo test --features cuda -- --nocapture

cuda-run: cuda-image
	$(RUN_GPU) cargo run --features cuda

cuda-shell: cuda-image
	$(COMPOSE) run --rm cuda bash

# --- Cleanup ---

clean:
	$(COMPOSE) down -v --rmi local
"#;

const MAKEFILE_MOUNTED: &str = r#"# Development commands -- all builds run inside Docker.
# libtorch is mounted from the host libtorch/ directory.
#
# Quick start:
#   make setup   -- detect hardware, download libtorch, build image
#   make build   -- compile (CPU)
#   make test    -- run tests
#   make run     -- cargo run
#   make shell   -- interactive shell
#
# GPU (requires NVIDIA Container Toolkit):
#   make cuda-build / cuda-test / cuda-run / cuda-shell

COMPOSE = docker compose

# --- libtorch auto-detection ---
LIBTORCH_ACTIVE := $(shell cat libtorch/.active 2>/dev/null | tr -d '[:space:]')
LIBTORCH_HOST_PATH := $(if $(LIBTORCH_ACTIVE),./libtorch/$(LIBTORCH_ACTIVE),)
ARCH_FILE := $(if $(LIBTORCH_HOST_PATH),$(LIBTORCH_HOST_PATH)/.arch,)
ARCH_CUDA := $(shell grep '^cuda=' $(ARCH_FILE) 2>/dev/null | cut -d= -f2)

ifeq ($(ARCH_CUDA),none)
  _CUDA_VER :=
else ifneq ($(ARCH_CUDA),)
  _CUDA_VER := $(ARCH_CUDA).0
else
  _CUDA_VER := 12.8.0
endif
CUDA_VERSION ?= $(_CUDA_VER)
CUDA_TAG     ?= $(shell echo "$(CUDA_VERSION)" | cut -d. -f1,2)

LIBTORCH_CPU_PATH := ./libtorch/precompiled/cpu

export LIBTORCH_HOST_PATH
export LIBTORCH_CPU_PATH
export CUDA_VERSION
export CUDA_TAG

RUN     = $(COMPOSE) run --rm dev
RUN_GPU = $(COMPOSE) run --rm cuda

.PHONY: build test run check clippy shell image clean \
        cuda-image cuda-build cuda-test cuda-run cuda-shell \
        setup _require-libtorch _require-libtorch-cuda

# --- libtorch guards ---

_require-libtorch:
	@if [ ! -d "$(LIBTORCH_CPU_PATH)/lib" ]; then \
		echo ""; \
		echo "ERROR: No CPU libtorch found."; \
		echo "  Run: make setup"; \
		echo "  Or:  ./fdl libtorch download --cpu"; \
		echo ""; \
		exit 1; \
	fi

_require-libtorch-cuda:
	@if [ -z "$(LIBTORCH_HOST_PATH)" ] || [ ! -d "$(LIBTORCH_HOST_PATH)/lib" ]; then \
		echo ""; \
		echo "ERROR: No active CUDA libtorch found."; \
		echo "  Run: make setup"; \
		echo "  Or:  ./fdl libtorch download --cuda 12.8"; \
		echo ""; \
		exit 1; \
	fi

# --- CPU targets ---

image:
	@mkdir -p .cargo-cache .cargo-git
	@if ! docker image inspect $$(basename $$(pwd))-dev:latest >/dev/null 2>&1; then \
		$(COMPOSE) build dev; \
	fi

build: image _require-libtorch
	$(RUN) cargo build

test: image _require-libtorch
	$(RUN) cargo test -- --nocapture

run: image _require-libtorch
	$(RUN) cargo run

check: image _require-libtorch
	$(RUN) cargo check

clippy: image _require-libtorch
	$(RUN) cargo clippy -- -W clippy::all

shell: image
	$(COMPOSE) run --rm dev bash

# --- CUDA targets ---

cuda-image:
	@mkdir -p .cargo-cache-cuda .cargo-git-cuda
	$(COMPOSE) build cuda

cuda-build: cuda-image _require-libtorch-cuda
	$(RUN_GPU) cargo build --features cuda

cuda-test: cuda-image _require-libtorch-cuda
	$(RUN_GPU) cargo test --features cuda -- --nocapture

cuda-run: cuda-image _require-libtorch-cuda
	$(RUN_GPU) cargo run --features cuda

cuda-shell: cuda-image
	$(COMPOSE) run --rm cuda bash

# --- Setup ---

setup:
	./fdl setup --non-interactive

# --- Cleanup ---

clean:
	$(COMPOSE) down -v --rmi local
"#;

// ---------------------------------------------------------------------------
// File writing helper
// ---------------------------------------------------------------------------

fn write_file(path: &str, content: &str) -> Result<(), String> {
    fs::write(path, content).map_err(|e| format!("cannot write {}: {}", path, e))
}
