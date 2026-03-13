# flodl development commands
#
# All commands run inside the Docker container via docker compose.
# Use the cuda-* targets for GPU builds (requires NVIDIA Container Toolkit).

COMPOSE = docker compose
RUN     = $(COMPOSE) run --rm dev
RUN_GPU = $(COMPOSE) run --rm cuda

.PHONY: build test test-release check clippy doc shell clean image \
        cuda-image cuda-build cuda-test cuda-shell

# --- CPU targets ---

# Build the Docker image
image:
	$(COMPOSE) build dev

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

# Build the CUDA Docker image
cuda-image:
	$(COMPOSE) build cuda

# Build with CUDA feature
cuda-build: cuda-image
	$(RUN_GPU) cargo build --features cuda

# Run all tests with CUDA
cuda-test: cuda-image
	$(RUN_GPU) cargo test --features cuda -- --nocapture

# Lint with CUDA feature
cuda-clippy: cuda-image
	$(RUN_GPU) cargo clippy --features cuda -- -W clippy::all

# Interactive shell (CUDA)
cuda-shell: cuda-image
	$(COMPOSE) run --rm cuda bash

# --- Cleanup ---

# Clean build artifacts
clean:
	$(COMPOSE) down -v --rmi local
