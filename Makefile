# flodl development -- legacy Makefile
#
# All development commands are available via fdl (see fdl.yml.example).
# This Makefile retains only host-side tasks that fdl can't handle.
#
# Quick start:
#   fdl setup             # detect hardware, download libtorch, build Docker image
#   fdl test              # run CPU tests
#   fdl cuda-test-all     # full CUDA suite

COMPOSE = docker compose

.PHONY: docs-rs site site-stop test-init clean

# --- docs.rs validation (host-side mkdir + nightly toolchain) ---

docs-rs:
	@mkdir -p .cargo-cache-docsrs .cargo-git-docsrs .target-docsrs
	$(COMPOSE) run --rm docs-rs bash -c "\
		rustup install nightly 2>&1 | tail -1 && \
		cargo +nightly rustdoc --lib -p flodl \
			--no-default-features --features rng \
			--config 'build.rustflags=[\"--cfg\", \"docsrs\"]' \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\"]' && \
		cargo +nightly rustdoc --lib -p flodl-cli \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\"]' && \
		cargo +nightly rustdoc --lib -p flodl-cli-macros \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\"]'"

# --- Site (host python + docker compose up/down) ---

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
