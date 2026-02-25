#!/usr/bin/env bash
# =============================================================================
# run.sh — Build and run all targets for SALR3YUCUDA
#
# Usage:
#   ./run.sh              # build + run main simulation + run tests
#   ./run.sh build        # build only
#   ./run.sh sim          # build + run main simulation only
#   ./run.sh tests        # build + run test targets only
#   ./run.sh clean        # remove build directory
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
CONFIG="$PROJECT_ROOT/configs/default.cfg"

# ── colour helpers ────────────────────────────────────────────────────────────
_green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
_cyan()   { printf '\033[0;36m%s\033[0m\n' "$*"; }
_red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
_bold()   { printf '\033[1m%s\033[0m\n'   "$*"; }

# ── steps ─────────────────────────────────────────────────────────────────────
do_build() {
    _bold ">>> Building project (CMake)"
    cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -Wno-dev
    cmake --build "$BUILD_DIR" --parallel "$(nproc)"
    _green "Build complete."
}

do_sim() {
    _bold ">>> Running simulation: salr_dft"
    _cyan "    config: $CONFIG"
    # Clear previous results so stale snapshots from a longer run are not mixed
    # with the new one.  Preserve the output directory itself.
    local out_dir
    out_dir="$PROJECT_ROOT/output"
    if compgen -G "$out_dir/density_species*.dat" > /dev/null 2>&1 || \
       [[ -f "$out_dir/convergence.dat" ]]; then
        _cyan "    clearing previous output in $out_dir/ ..."
        rm -f "$out_dir"/density_species*.dat "$out_dir"/convergence.dat
    fi
    "$BUILD_DIR/salr_dft" "$CONFIG"
    _green "Simulation finished."
}

do_tests() {
    _bold ">>> Running tests"

    local failed=0

    for target in test_solver test_potential; do
        exe="$BUILD_DIR/$target"
        if [[ ! -x "$exe" ]]; then
            _red "  [SKIP] $target — binary not found"
            continue
        fi
        _cyan "  Running $target ..."
        if "$exe"; then
            _green "  [PASS] $target"
        else
            _red   "  [FAIL] $target (exit code $?)"
            failed=$((failed + 1))
        fi
    done

    if [[ $failed -gt 0 ]]; then
        _red "$failed test target(s) failed."
        return 1
    else
        _green "All tests passed."
    fi
}

do_clean() {
    _bold ">>> Cleaning build directory"
    rm -rf "$BUILD_DIR"
    _green "Cleaned."
}

# ── dispatch ──────────────────────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    build)  do_build ;;
    sim)    do_build; do_sim ;;
    tests)  do_build; do_tests ;;
    clean)  do_clean ;;
    all)    do_build; do_sim; do_tests ;;
    *)
        _red "Unknown mode: $MODE"
        echo "Usage: $0 [all|build|sim|tests|clean]"
        exit 1
        ;;
esac
