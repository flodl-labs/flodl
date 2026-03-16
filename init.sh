#!/bin/sh
# floDl project scaffolding — generates a ready-to-build Rust DL project.
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/fab2s/floDl/main/init.sh | sh -s my-project
#   cd my-project && make test
#
# Or run locally:
#   sh init.sh my-project

set -e

# --- Argument validation ---

PROJECT_NAME="${1:?Usage: sh init.sh <project-name>}"

# Validate project name (alphanumeric, hyphens, underscores).
case "$PROJECT_NAME" in
    *[!a-zA-Z0-9_-]*)
        echo "error: project name must contain only letters, digits, hyphens, underscores" >&2
        exit 1
        ;;
esac

if [ -e "$PROJECT_NAME" ]; then
    echo "error: '$PROJECT_NAME' already exists" >&2
    exit 1
fi

# Rust crate names use underscores.
CRATE_NAME=$(echo "$PROJECT_NAME" | tr '-' '_')

echo "Creating floDl project: $PROJECT_NAME"

# --- Directory structure ---

mkdir -p "$PROJECT_NAME/src"
cd "$PROJECT_NAME"

# --- Cargo.toml ---

cat > Cargo.toml << 'CARGO_EOF'
[package]
name = "CRATE_PLACEHOLDER"
version = "0.1.0"
edition = "2024"

[dependencies]
flodl = { git = "https://github.com/fab2s/floDl.git" }

# Optimize floDl in dev builds — your code stays fast to compile.
# After the first build, only your graph code recompiles (~2s).
[profile.dev.package.flodl]
opt-level = 3

[profile.dev.package.flodl-sys]
opt-level = 3

# Release: cross-crate optimization for maximum throughput.
[profile.release]
lto = "thin"
codegen-units = 1
CARGO_EOF

# Portable sed: replace the placeholder.
sed -i.bak "s/CRATE_PLACEHOLDER/$CRATE_NAME/" Cargo.toml && rm -f Cargo.toml.bak

# --- Dockerfile.cpu ---

cat > Dockerfile.cpu << 'DOCKERFILE_CPU_EOF'
# CPU-only dev image for floDl projects.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip ca-certificates git gcc g++ pkg-config graphviz \
    && rm -rf /var/lib/apt/lists/*

# Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# libtorch (CPU-only, ~200MB)
ARG LIBTORCH_VERSION=2.10.0
RUN wget -q https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip \
    && unzip -q libtorch-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip -d /usr/local \
    && rm libtorch-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip

ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib"

# Allow non-root users to access Rust toolchain (for user: mapping in compose)
RUN chmod a+rx /root && chmod -R a+rwx /root/.cargo /root/.rustup

WORKDIR /workspace
DOCKERFILE_CPU_EOF

# --- Dockerfile.cuda ---

cat > Dockerfile.cuda << 'DOCKERFILE_CUDA_EOF'
# CUDA dev image for floDl projects.
# Requires: docker run --gpus all ...
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip ca-certificates git gcc g++ pkg-config graphviz \
    && rm -rf /var/lib/apt/lists/*

# Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# libtorch (CUDA 12.6)
ARG LIBTORCH_VERSION=2.10.0
RUN wget -q "https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu126.zip" \
    && unzip -q "libtorch-shared-with-deps-${LIBTORCH_VERSION}+cu126.zip" -d /usr/local \
    && rm "libtorch-shared-with-deps-${LIBTORCH_VERSION}+cu126.zip"

ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib:/usr/local/cuda/lib64"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib:/usr/local/cuda/lib64"
ENV CUDA_HOME="/usr/local/cuda"

# Allow non-root users to access Rust toolchain (for user: mapping in compose)
RUN chmod a+rx /root && chmod -R a+rwx /root/.cargo /root/.rustup

WORKDIR /workspace
DOCKERFILE_CUDA_EOF

# --- docker-compose.yml ---

cat > docker-compose.yml << 'COMPOSE_EOF'
services:
  dev:
    build:
      context: .
      dockerfile: Dockerfile.cpu
    image: CRATE_PLACEHOLDER-dev
    user: "${UID:-1000}:${GID:-1000}"
    environment:
      - HOME=/root
    volumes:
      - .:/workspace
      - cargo-registry:/root/.cargo/registry
      - cargo-git:/root/.cargo/git
    working_dir: /workspace
    stdin_open: true
    tty: true

  cuda:
    build:
      context: .
      dockerfile: Dockerfile.cuda
    image: CRATE_PLACEHOLDER-cuda
    user: "${UID:-1000}:${GID:-1000}"
    environment:
      - HOME=/root
    volumes:
      - .:/workspace
      - cargo-registry-cuda:/root/.cargo/registry
      - cargo-git-cuda:/root/.cargo/git
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

volumes:
  cargo-registry:
  cargo-git:
  cargo-registry-cuda:
  cargo-git-cuda:
COMPOSE_EOF

sed -i.bak "s/CRATE_PLACEHOLDER/$CRATE_NAME/g" docker-compose.yml && rm -f docker-compose.yml.bak

# --- Makefile ---

cat > Makefile << 'MAKEFILE_EOF'
# Development commands — all builds run inside Docker.
#
# Quick start:
#   make build   — compile (CPU)
#   make test    — run tests
#   make run     — cargo run
#   make shell   — interactive shell in container
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
MAKEFILE_EOF

# --- src/main.rs ---

cat > src/main.rs << 'MAIN_EOF'
//! floDl training template.
//!
//! This is a starting point for your model. Edit the architecture,
//! data loading, and training loop to fit your task.
//!
//! New to Rust? Read: https://flodl.dev/guide/rust-primer
//! Stuck?       Read: https://flodl.dev/guide/troubleshooting

use flodl::*;                         // import torch / import torch.nn as nn
use flodl::monitor::Monitor;

fn main() -> Result<()> {
    // --- Model ---
    // PyTorch equivalent:
    //   model = nn.Sequential(
    //       nn.Linear(4, 32), nn.GELU(), nn.LayerNorm(32),
    //       ResidualBlock(nn.Linear(32, 32)),  # also() = skip connection
    //       nn.Linear(32, 1),
    //   )
    let model = FlowBuilder::from(Linear::new(4, 32)?)   // input: 4 features
        .through(GELU)                                     // activation
        .through(LayerNorm::new(32)?)                      // normalization
        .also(Linear::new(32, 32)?)                        // residual: output = input + Linear(input)
        .through(Linear::new(32, 1)?)                      // output: 1 value
        .build()?;

    // --- Optimizer ---
    // PyTorch: optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 0.001);
    let scheduler = CosineScheduler::new(0.001, 1e-6, 100);
    model.set_training(true);                              // model.train()

    // --- Data ---
    // Replace this with your data loading.
    let opts = TensorOptions::default();
    let batches: Vec<(Tensor, Tensor)> = (0..32)
        .map(|_| {
            let x = Tensor::randn(&[16, 4], opts).unwrap();   // [batch=16, features=4]
            let y = Tensor::randn(&[16, 1], opts).unwrap();   // [batch=16, target=1]
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
            // PyTorch: input = torch.tensor(input_t, requires_grad=True)
            let input = Variable::new(input_t.clone(), true);
            let target = Variable::new(target_t.clone(), false);

            // PyTorch: optimizer.zero_grad()
            optimizer.zero_grad();
            // PyTorch: pred = model(input)
            let pred = model.forward(&input)?;
            // PyTorch: loss = F.mse_loss(pred, target)
            let loss = mse_loss(&pred, &target)?;
            // PyTorch: loss.backward()
            loss.backward()?;
            // PyTorch: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            clip_grad_norm(&params, 1.0)?;
            // PyTorch: optimizer.step()
            optimizer.step()?;

            epoch_loss += loss.item()?;                    // loss.item()
        }

        let avg_loss = epoch_loss / batches.len() as f64;
        let lr = scheduler.lr(epoch);
        optimizer.set_lr(lr);                              // scheduler.step()
        monitor.log(epoch, t.elapsed(), &[("loss", avg_loss), ("lr", lr)]);
    }

    monitor.finish();

    // --- Save checkpoint ---
    // PyTorch: torch.save(model.state_dict(), "model.fdl")
    // save_checkpoint_file("model.fdl", &model.named_parameters(), &model.named_buffers())?;

    Ok(())
}
MAIN_EOF

# --- .gitignore ---

cat > .gitignore << 'GITIGNORE_EOF'
/target
*.fdl
*.log
*.csv
*.html
GITIGNORE_EOF

# --- Done ---

echo ""
echo "Project '$PROJECT_NAME' created. Next steps:"
echo ""
echo "  cd $PROJECT_NAME"
echo "  make build    # first build (downloads libtorch, ~5 min)"
echo "  make test     # run tests"
echo "  make run      # train the model"
echo "  make shell    # interactive shell"
echo ""
echo "Edit src/main.rs to build your model."
echo ""
echo "Guides:"
echo "  Rust primer:       https://flodl.dev/guide/rust-primer"
echo "  Troubleshooting:   https://flodl.dev/guide/troubleshooting"
echo "  PyTorch migration: https://flodl.dev/guide/migration"
