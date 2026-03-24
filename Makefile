# flodl development commands
#
# All commands run inside the Docker container via docker compose.
# Use the cuda-* targets for GPU builds (requires NVIDIA Container Toolkit).

COMPOSE   = docker compose
RUN       = $(COMPOSE) run --rm dev
RUN_GPU   = $(COMPOSE) run --rm cuda
RUN_BENCH = $(COMPOSE) run --rm bench

.PHONY: build test test-release check clippy doc shell clean image \
        cuda-image cuda-build cuda-test cuda-shell test-all \
        bench-image bench bench-cpu bench-compare bench-publish \
        site site-stop test-init

# --- CPU targets ---

# Build the Docker image (skips if already exists)
image:
	@mkdir -p .cargo-cache .cargo-git
	@if ! docker image inspect flodl-dev:latest >/dev/null 2>&1; then \
		$(COMPOSE) build dev; \
	fi

# Build the project (debug)
build: image
	$(RUN) cargo build

# Run all tests
test: image
	$(RUN) cargo test -- --nocapture

# Run tests in release mode
test-release: image
	$(RUN) cargo test --release -- --nocapture

# Type check without building
check: image
	$(RUN) cargo check

# Lint
clippy: image
	$(RUN) cargo clippy -- -W clippy::all

# Generate API docs → target/doc/flodl/index.html
doc: image
	$(RUN) cargo doc --no-deps --document-private-items

# Interactive shell
shell: image
	$(COMPOSE) run --rm dev bash

# --- CUDA targets ---

# Build the CUDA Docker image (skips if already exists)
cuda-image:
	@mkdir -p .cargo-cache-cuda .cargo-git-cuda
	@if ! docker image inspect flodl-cuda:latest >/dev/null 2>&1; then \
		$(COMPOSE) build cuda; \
	fi

# Build with CUDA feature
cuda-build: cuda-image
	$(RUN_GPU) cargo build --features cuda

# Run all tests with CUDA
cuda-test: cuda-image
	$(RUN_GPU) cargo test --features cuda -- --nocapture

# Run CUDA Graph tests (need exclusive GPU — single-threaded)
cuda-test-graph: cuda-image
	$(RUN_GPU) cargo test --features cuda -- --nocapture --ignored --test-threads=1 cuda_graph

# Lint with CUDA feature
cuda-clippy: cuda-image
	$(RUN_GPU) cargo clippy --features cuda -- -W clippy::all

# Interactive shell (CUDA)
cuda-shell: cuda-image
	$(COMPOSE) run --rm cuda bash

# --- Combined ---

# Run CPU tests, then CUDA tests if a GPU is available
test-all: test
	@if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then \
		echo ""; \
		echo "=== GPU detected — running CUDA tests ==="; \
		$(MAKE) cuda-test; \
	else \
		echo ""; \
		echo "=== No GPU available — skipping CUDA tests ==="; \
	fi

# --- Benchmarks ---

# Build the benchmark Docker image (Rust + Python + PyTorch)
# Skips rebuild if flodl-bench image already exists (use `make clean` to force)
bench-image:
	@mkdir -p .cargo-cache-bench .cargo-git-bench
	@if ! docker image inspect flodl-bench:latest >/dev/null 2>&1; then \
		$(COMPOSE) build bench; \
	else \
		echo "flodl-bench image exists, skipping build (docker compose down -v --rmi local to force)"; \
	fi

# Run CUDA benchmarks: flodl vs PyTorch comparison
# Pass extra args: make bench ARGS="--tier1"
bench: bench-image
	$(RUN_BENCH) benchmarks/run.sh $(ARGS)

# Run CPU-only benchmarks: flodl vs PyTorch comparison
bench-cpu: bench-image
	$(RUN_BENCH) benchmarks/run.sh --cpu $(ARGS)

# Publication benchmarks: 10 interleaved rounds, locked clocks, long warmup.
# Override rounds/freq/output: make bench-publish ROUNDS=20 CLOCK=2400 OUTPUT=report.txt
ROUNDS ?= 10
CLOCK  ?= 2407
OUTPUT ?= benchmarks/report.txt
bench-publish: bench-image
	$(RUN_BENCH) benchmarks/run.sh --rounds $(ROUNDS) --lock-clocks $(CLOCK) --warmup-secs 15 --output $(OUTPUT) $(ARGS)

# Run flodl + PyTorch benchmarks and compare (alias)
bench-compare: bench

# --- Site ---

# Preview site locally at http://localhost:4000 (Ctrl-C to stop)
site:
	@python3 site/build_guide.py
	$(COMPOSE) up jekyll

# Stop the site preview
site-stop:
	$(COMPOSE) down jekyll

# --- Smoke test: init.sh end-to-end ---

# Test that init.sh produces a working project scaffold
test-init:
	@echo "=== Testing init.sh scaffold ==="
	@cd /tmp && rm -rf flodl-init-test && sh $(CURDIR)/init.sh flodl-init-test
	@cd /tmp/flodl-init-test && make image
	@cd /tmp/flodl-init-test && docker compose run --rm dev \
		sh -c "touch \$$CARGO_HOME/registry/.write-test && rm \$$CARGO_HOME/registry/.write-test && echo 'write ok'"
	@rm -rf /tmp/flodl-init-test
	@echo "=== init.sh smoke test passed ==="

# --- Cleanup ---

# Clean build artifacts
clean:
	$(COMPOSE) down -v --rmi local
