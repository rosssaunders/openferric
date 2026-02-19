# Coverage targets using cargo-llvm-cov (LLVM source-based instrumentation)
# Works on macOS/ARM unlike tarpaulin

COVERAGE_DIR := target/llvm-cov
HTML_REPORT := $(COVERAGE_DIR)/html

.PHONY: install-coverage coverage-test coverage-bench coverage-bench-parallel \
        coverage-all coverage-lcov coverage-clean

## Install cargo-llvm-cov and llvm-tools-preview
install-coverage:
	cargo install cargo-llvm-cov
	rustup component add llvm-tools-preview

## Run test coverage with HTML output
coverage-test:
	cargo llvm-cov test --html --output-dir $(HTML_REPORT)
	@echo "HTML report: $(HTML_REPORT)/index.html"

## Run benchmark coverage (single pass, all benches, no extra features)
coverage-bench:
	cargo llvm-cov bench --html --output-dir $(HTML_REPORT) -- --quick
	@echo "HTML report: $(HTML_REPORT)/index.html"

## Run benchmark coverage with parallel feature
coverage-bench-parallel:
	cargo llvm-cov bench --features parallel --html --output-dir $(HTML_REPORT) -- --quick
	@echo "HTML report: $(HTML_REPORT)/index.html"

## Combined test + bench coverage (merged report)
coverage-all:
	cargo llvm-cov clean
	cargo llvm-cov test --no-report
	cargo llvm-cov bench --no-report -- --quick
	cargo llvm-cov bench --no-report --features parallel -- --quick
	cargo llvm-cov report --html --output-dir $(HTML_REPORT)
	@echo "HTML report: $(HTML_REPORT)/index.html"

## Output lcov format for CI integration
coverage-lcov:
	cargo llvm-cov clean
	cargo llvm-cov test --no-report
	cargo llvm-cov bench --no-report -- --quick
	cargo llvm-cov bench --no-report --features parallel -- --quick
	cargo llvm-cov report --lcov --output-path $(COVERAGE_DIR)/lcov.info
	@echo "LCOV report: $(COVERAGE_DIR)/lcov.info"

## Clean profraw data and coverage artifacts
coverage-clean:
	cargo llvm-cov clean
