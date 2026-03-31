# flodl development commands
#
# All commands run inside the Docker container via docker compose.
# Use the cuda-* targets for GPU builds (requires NVIDIA Container Toolkit).

COMPOSE   = docker compose
RUN       = $(COMPOSE) run --rm dev
RUN_GPU   = $(COMPOSE) run --rm cuda
RUN_BENCH = $(COMPOSE) run --rm bench

.PHONY: build test test-release check clippy doc shell clean image \
        cuda-image cuda-build cuda-test cuda-shell test-all setup build-libtorch \
        bench-image bench bench-cpu bench-compare bench-publish \
        docs-rs site site-stop test-init

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

# --- docs.rs validation ---

# Simulate docs.rs build (nightly, no libtorch, --cfg docsrs)
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

# --- GPU-aware setup ---

# Detect GPUs and build CUDA image with the best libtorch variant.
# Automatically selects cu124 (broadest compat) or cu128 (best perf)
# based on the lowest compute capability across all installed GPUs.
setup:
	@echo "[flodl] Detecting GPUs..."
	@if ! command -v nvidia-smi >/dev/null 2>&1; then \
		echo "[flodl] No nvidia-smi found. Building CPU-only."; \
		$(MAKE) image; \
		exit 0; \
	fi
	@CAPS=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); \
	if [ -z "$$CAPS" ]; then \
		echo "[flodl] No GPUs detected. Building CPU-only."; \
		$(MAKE) image; \
		exit 0; \
	fi; \
	echo "[flodl] GPUs found:"; \
	nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader 2>/dev/null \
		| while IFS= read -r line; do echo "  $$line"; done; \
	LO_MAJOR=$$(echo "$$CAPS" | sort -t. -k1,1n | head -1 | cut -d. -f1); \
	HI_MAJOR=$$(echo "$$CAPS" | sort -t. -k1,1n | tail -1 | cut -d. -f1); \
	if [ "$$LO_MAJOR" -lt 7 ] && [ "$$HI_MAJOR" -ge 10 ]; then \
		echo ""; \
		echo "[flodl] NOTE: Your GPUs span sm_$$LO_MAJOR.x to sm_$$HI_MAJOR.x."; \
		echo "[flodl] No pre-built libtorch covers both."; \
		echo "[flodl]   cu126 supports sm_50-sm_90 (pre-Blackwell)"; \
		echo "[flodl]   cu128 supports sm_70-sm_120 (Volta through Blackwell)"; \
		echo "[flodl] Selecting cu128 for the most capable GPU."; \
		echo "[flodl] Older GPUs will be auto-excluded at runtime with diagnostics."; \
		echo "[flodl] To support all GPUs, build libtorch from source with:"; \
		echo "[flodl]   TORCH_CUDA_ARCH_LIST=\"$$LO_MAJOR.x;$$HI_MAJOR.x\""; \
		echo ""; \
		$(COMPOSE) build cuda; \
	elif [ "$$LO_MAJOR" -lt 7 ]; then \
		echo "[flodl] Lowest compute capability: sm_$$LO_MAJOR.x (< sm_70)"; \
		echo "[flodl] Selecting libtorch cu126 for broadest GPU compatibility."; \
		CUDA_VARIANT=cu126 $(COMPOSE) build cuda; \
	else \
		echo "[flodl] All GPUs are sm_70+. Using libtorch cu128 for best performance."; \
		$(COMPOSE) build cuda; \
	fi; \
	echo "[flodl] Setup complete. Run 'make cuda-test' to verify."

# Build libtorch from PyTorch source for custom GPU architectures.
# Auto-detects compute capabilities from installed GPUs.
# Takes 2-6 hours. Run overnight: make build-libtorch
build-libtorch:
	@echo "[flodl] Building libtorch from source..."
	@if ! command -v nvidia-smi >/dev/null 2>&1; then \
		echo "[flodl] ERROR: nvidia-smi not found. Cannot detect GPUs."; \
		exit 1; \
	fi
	@mkdir -p .cargo-cache-cuda .cargo-git-cuda
	@ARCHS=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
		| sort -u | tr -d ' ' | tr '\n' ';' | sed 's/;$$//'); \
	echo "[flodl] GPUs detected:"; \
	nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader 2>/dev/null \
		| while IFS= read -r line; do echo "  $$line"; done; \
	echo "[flodl] Building libtorch with TORCH_CUDA_ARCH_LIST=\"$$ARCHS\""; \
	echo "[flodl] This will take 2-6 hours. Go to sleep."; \
	echo ""; \
	TORCH_CUDA_ARCH_LIST="$$ARCHS" $(COMPOSE) build cuda-source; \
	echo ""; \
	echo "[flodl] libtorch build complete!"; \
	echo "[flodl] All GPUs supported. Run 'make cuda-test' to verify."

# --- Cleanup ---

# Clean build artifacts
clean:
	$(COMPOSE) down -v --rmi local
