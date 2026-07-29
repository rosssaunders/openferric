#!/usr/bin/env bash
set -euo pipefail

# Build one of OpenFerric's browser WebAssembly distributions:
#
#   ./scripts/build-wasm.sh baseline  # portable fallback
#   ./scripts/build-wasm.sh simd      # requires WebAssembly SIMD128
#   ./scripts/build-wasm.sh threads   # requires WebAssembly threads
#   ./scripts/build-wasm.sh all
#
# One-time toolchain setup:
#
#   rustup target add --toolchain 1.94.0 wasm32-unknown-unknown
#   rustup toolchain install nightly-2025-11-15 \
#       --component rust-src --target wasm32-unknown-unknown
#
# Output defaults to www/pkg (baseline), www/pkg-simd, and www/pkg-threads. Set
# WASM_OUT_ROOT to use another parent directory. Select between distributions
# in JavaScript with `simd()` / `threads()` from `wasm-feature-detect`; always
# retain the baseline package as the fallback.
#
# The threaded package uses Web Workers and SharedArrayBuffer. It must be served
# in a cross-origin-isolated context, normally with these response headers:
#
#   Cross-Origin-Opener-Policy: same-origin
#   Cross-Origin-Embedder-Policy: require-corp
#
# After module initialization, await
# `initThreadPool(navigator.hardwareConcurrency)` before parallel work. The
# threaded build uses wasm-bindgen-rayon's tested nightly by default; override
# WASM_THREADS_TOOLCHAIN only when deliberately validating another nightly.

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly WASM_CRATE="${REPO_ROOT}/wasm"
readonly OUTPUT_ROOT="${WASM_OUT_ROOT:-${REPO_ROOT}/www}"
readonly STABLE_TOOLCHAIN="${WASM_STABLE_TOOLCHAIN:-1.94.0}"
readonly THREADS_TOOLCHAIN="${WASM_THREADS_TOOLCHAIN:-nightly-2025-11-15}"
readonly GETRANDOM_CFG='--cfg getrandom_backend="wasm_js"'
readonly TOOLS_TARGET_DIR="${WASM_TOOLS_TARGET_DIR:-${TMPDIR:-/tmp}/openferric-wasm-tools-target}"

usage() {
    echo "usage: $0 {baseline|simd|threads|all}" >&2
    exit 2
}

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required tool '$1' is not installed" >&2
        exit 1
    fi
}

verify_features() {
    local profile="$1"
    local module="$2"

    CARGO_TARGET_DIR="${TOOLS_TARGET_DIR}" \
        rustup run "${STABLE_TOOLCHAIN}" cargo run \
        --quiet \
        --locked \
        --manifest-path "${WASM_CRATE}/Cargo.toml" \
        --example verify_wasm_features \
        -- \
        "${module}" \
        "${profile}"
}

build_baseline() {
    echo "Building portable WebAssembly package with ${STABLE_TOOLCHAIN}"
    rustup run "${STABLE_TOOLCHAIN}" env \
        RUSTFLAGS="${GETRANDOM_CFG}" \
        wasm-pack build "${WASM_CRATE}" \
        --release \
        --target web \
        --out-dir "${OUTPUT_ROOT}/pkg" \
        -- \
        --locked
    verify_features baseline "${OUTPUT_ROOT}/pkg/openferric_wasm_bg.wasm"
}

build_simd() {
    echo "Building SIMD128 WebAssembly package with ${STABLE_TOOLCHAIN}"
    rustup run "${STABLE_TOOLCHAIN}" env \
        RUSTFLAGS="${GETRANDOM_CFG} -C target-feature=+simd128" \
        wasm-pack build "${WASM_CRATE}" \
        --release \
        --target web \
        --out-dir "${OUTPUT_ROOT}/pkg-simd" \
        -- \
        --locked \
        --features simd
    verify_features simd "${OUTPUT_ROOT}/pkg-simd/openferric_wasm_bg.wasm"
}

build_threads() {
    echo "Building threaded WebAssembly package with ${THREADS_TOOLCHAIN}"
    rustup run "${THREADS_TOOLCHAIN}" env \
        RUSTFLAGS="${GETRANDOM_CFG} -C target-feature=+atomics,+bulk-memory \
        -C link-arg=--shared-memory \
        -C link-arg=--max-memory=1073741824 \
        -C link-arg=--import-memory \
        -C link-arg=--export=__wasm_init_tls \
        -C link-arg=--export=__tls_size \
        -C link-arg=--export=__tls_align \
        -C link-arg=--export=__tls_base" \
        wasm-pack build "${WASM_CRATE}" \
        --release \
        --target web \
        --out-dir "${OUTPUT_ROOT}/pkg-threads" \
        -- \
        --locked \
        --features threads \
        -Z build-std=panic_abort,std
    verify_features threads "${OUTPUT_ROOT}/pkg-threads/openferric_wasm_bg.wasm"
}

require_tool wasm-pack
require_tool rustup

case "${1:-}" in
    baseline)
        build_baseline
        ;;
    simd)
        build_simd
        ;;
    threads)
        build_threads
        ;;
    all)
        build_baseline
        build_simd
        build_threads
        ;;
    *)
        usage
        ;;
esac
