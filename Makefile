# flodl development commands
#
# All commands run inside Docker containers via docker compose.
# libtorch is mounted from the host libtorch/ directory (not baked into images).
#
# Quick start:
#   make setup        # detect hardware, download libtorch, build Docker image
#   make test         # run CPU tests
#   make cuda-test    # run CUDA tests (parallel)
#   make cuda-test-all  # full suite: parallel + NCCL (isolated) + serial

COMPOSE   = docker compose

# --- libtorch auto-detection ---
# Read the active libtorch variant from libtorch/.active
LIBTORCH_ACTIVE := $(shell cat libtorch/.active 2>/dev/null | tr -d '[:space:]')
LIBTORCH_HOST_PATH := $(if $(LIBTORCH_ACTIVE),./libtorch/$(LIBTORCH_ACTIVE),)

# Read .arch properties from the active variant
ARCH_FILE := $(if $(LIBTORCH_HOST_PATH),$(LIBTORCH_HOST_PATH)/.arch,)
ARCH_CUDA := $(shell grep '^cuda=' $(ARCH_FILE) 2>/dev/null | cut -d= -f2)

# Determine CUDA version for Docker image. Override: CUDA_VERSION=12.6.0 make cuda-test
ifeq ($(ARCH_CUDA),none)
  _CUDA_VER :=
else ifneq ($(ARCH_CUDA),)
  _CUDA_VER := $(ARCH_CUDA).0
else
  _CUDA_VER := 12.8.0
endif
CUDA_VERSION ?= $(_CUDA_VER)
CUDA_TAG     ?= $(shell echo "$(CUDA_VERSION)" | cut -d. -f1,2)

# CPU libtorch is always the precompiled CPU variant
LIBTORCH_CPU_PATH := ./libtorch/precompiled/cpu

export LIBTORCH_HOST_PATH
export LIBTORCH_CPU_PATH
export CUDA_VERSION
export CUDA_TAG

# Docker run shortcuts
RUN       = $(COMPOSE) run --rm dev
RUN_GPU   = $(COMPOSE) run --rm cuda
RUN_BENCH = $(COMPOSE) run --rm bench

.PHONY: build test test-release check clippy clippy-all doc shell clean \
        cuda-build cuda-test cuda-test-nccl cuda-test-serial cuda-test-graph cuda-test-all \
        cuda-clippy cuda-clippy-all cuda-shell \
        image cuda-image \
        test-all setup build-libtorch \
        bench-image bench bench-cpu bench-compare bench-publish \
        docs-rs site site-stop test-init \
        cli \
        _require-libtorch _require-libtorch-cuda

# --- libtorch guards ---

_require-libtorch:
	@if [ ! -d "$(LIBTORCH_CPU_PATH)/lib" ]; then \
		echo ""; \
		echo "[flodl] ERROR: No CPU libtorch found at $(LIBTORCH_CPU_PATH)."; \
		echo "[flodl] Run:  make setup"; \
		echo "[flodl]   or: fdl libtorch download --cpu"; \
		echo ""; \
		exit 1; \
	fi

_require-libtorch-cuda:
	@if [ -z "$(LIBTORCH_HOST_PATH)" ] || [ ! -d "$(LIBTORCH_HOST_PATH)/lib" ]; then \
		echo ""; \
		echo "[flodl] ERROR: No active CUDA libtorch found."; \
		echo "[flodl] Run:  make setup        (auto-detect and download)"; \
		echo "[flodl]   or: fdl libtorch download --cuda 12.8"; \
		echo ""; \
		exit 1; \
	fi; \
	if [ "$(ARCH_CUDA)" = "none" ]; then \
		echo ""; \
		echo "[flodl] ERROR: Active libtorch is CPU-only ($(LIBTORCH_HOST_PATH))."; \
		echo "[flodl] Run:  make setup        (to get a CUDA variant)"; \
		echo ""; \
		exit 1; \
	fi

# --- CPU targets ---

# Build the Docker image (skips if already exists)
image:
	@mkdir -p .cargo-cache .cargo-git
	@if ! docker image inspect flodl-dev:latest >/dev/null 2>&1; then \
		$(COMPOSE) build dev; \
	fi

# Build the project (debug)
build: image _require-libtorch
	$(RUN) cargo build

# Run all tests
test: image _require-libtorch
	$(RUN) cargo test -- --nocapture

# Run tests in release mode
test-release: image _require-libtorch
	$(RUN) cargo test --release -- --nocapture

# Type check without building
check: image _require-libtorch
	$(RUN) cargo check

# Lint
clippy: image _require-libtorch
	$(RUN) cargo clippy -- -W clippy::all

# Lint including test code
clippy-all: image _require-libtorch
	$(RUN) cargo clippy --tests -- -W clippy::all

# Generate API docs
doc: image _require-libtorch
	$(RUN) cargo doc --no-deps --document-private-items

# Interactive shell
shell: image
	$(COMPOSE) run --rm dev bash

# --- CUDA targets ---

# Build the CUDA Docker image (skips if already exists)
cuda-image:
	@mkdir -p .cargo-cache-cuda .cargo-git-cuda
	@if ! docker image inspect flodl-cuda:$(CUDA_TAG) >/dev/null 2>&1; then \
		$(COMPOSE) build cuda; \
	fi

# Build with CUDA feature
cuda-build: cuda-image _require-libtorch-cuda
	$(RUN_GPU) cargo build --features cuda

# Run all tests with CUDA (parallel, excludes NCCL/DDP/Graph)
cuda-test: cuda-image _require-libtorch-cuda
	$(RUN_GPU) cargo test --features cuda -- --nocapture

# Run NCCL/DDP tests (NCCL init poisons CUBLAS -- each group in isolated process)
cuda-test-nccl: cuda-image _require-libtorch-cuda
	$(RUN_GPU) cargo test --features cuda -- --nocapture --ignored --test-threads=1 nccl
	$(RUN_GPU) cargo test --features cuda -- --nocapture --ignored --test-threads=1 graph_distribute

# Run remaining serial tests (Graphs, manual_seed, etc.) -- separate process, no NCCL poison
cuda-test-serial: cuda-image _require-libtorch-cuda
	$(RUN_GPU) cargo test --features cuda -- --nocapture --ignored --test-threads=1 --skip nccl --skip graph_distribute

# Run CUDA Graph tests only (need exclusive GPU, single-threaded)
cuda-test-graph: cuda-image _require-libtorch-cuda
	$(RUN_GPU) cargo test --features cuda -- --nocapture --ignored --test-threads=1 cuda_graph

# Full CUDA test suite: parallel + NCCL (isolated) + remaining serial
cuda-test-all: cuda-test cuda-test-nccl cuda-test-serial

# Lint with CUDA feature
cuda-clippy: cuda-image _require-libtorch-cuda
	$(RUN_GPU) cargo clippy --features cuda -- -W clippy::all

# Lint with CUDA feature including test code
cuda-clippy-all: cuda-image _require-libtorch-cuda
	$(RUN_GPU) cargo clippy --features cuda --tests -- -W clippy::all

# Interactive shell (CUDA)
cuda-shell: cuda-image
	$(COMPOSE) run --rm cuda bash

# --- Combined ---

# Run CPU tests, then CUDA tests if a GPU is available
test-all: test
	@if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then \
		echo ""; \
		echo "=== GPU detected -- running CUDA tests ==="; \
		$(MAKE) cuda-test-all; \
	else \
		echo ""; \
		echo "=== No GPU available -- skipping CUDA tests ==="; \
	fi

# --- Setup ---

# Detect hardware, download/build libtorch, build Docker image.
setup: cli
	./target/release/fdl setup --non-interactive

# Build libtorch from PyTorch source for custom GPU architectures.
# Auto-detects compute capabilities from installed GPUs.
# Takes 2-6 hours. Run overnight: make build-libtorch
build-libtorch: cli
	./target/release/fdl libtorch build

# --- CLI (pure Rust, no libtorch needed) ---

cli:
	cargo build --release -p flodl-cli

# --- Benchmarks ---

bench-image:
	@mkdir -p .cargo-cache-bench .cargo-git-bench
	@if ! docker image inspect flodl-bench:latest >/dev/null 2>&1; then \
		$(COMPOSE) build bench; \
	fi

# Run CUDA benchmarks: flodl vs PyTorch comparison
bench: bench-image _require-libtorch-cuda
	$(RUN_BENCH) benchmarks/run.sh $(ARGS)

# Run CPU-only benchmarks
bench-cpu: bench-image _require-libtorch
	$(RUN_BENCH) benchmarks/run.sh --cpu $(ARGS)

# Publication benchmarks: interleaved rounds, locked clocks, long warmup.
ROUNDS ?= 10
CLOCK  ?= 2407
OUTPUT ?= benchmarks/report.txt
bench-publish: bench-image _require-libtorch-cuda
	$(RUN_BENCH) benchmarks/run.sh --rounds $(ROUNDS) --lock-clocks $(CLOCK) --warmup-secs 15 --output $(OUTPUT) $(ARGS)

bench-compare: bench

# --- docs.rs validation ---

docs-rs:
	@mkdir -p .cargo-cache-docsrs .cargo-git-docsrs .target-docsrs
	$(COMPOSE) run --rm docs-rs bash -c "\
		rustup install nightly 2>&1 | tail -1 && \
		cargo +nightly rustdoc --lib \
			--no-default-features \
			--config 'build.rustflags=[\"--cfg\", \"docsrs\"]' \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\"]' \
			-Zrustdoc-scrape-examples"

# --- Site ---

site:
	@python3 site/build_guide.py
	$(COMPOSE) up jekyll

site-stop:
	$(COMPOSE) down jekyll

# --- Smoke test: init.sh end-to-end ---

test-init:
	@echo "=== Testing init.sh scaffold ==="
	@cd /tmp && rm -rf flodl-init-test && sh $(CURDIR)/init.sh flodl-init-test
	@cd /tmp/flodl-init-test && make image
	@cd /tmp/flodl-init-test && docker compose run --rm dev \
		sh -c "touch \$$CARGO_HOME/registry/.write-test && rm \$$CARGO_HOME/registry/.write-test && echo 'write ok'"
	@rm -rf /tmp/flodl-init-test
	@echo "=== init.sh smoke test passed ==="

# --- Cleanup ---

clean:
	$(COMPOSE) down -v --rmi local
