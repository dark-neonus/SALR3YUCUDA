#!/usr/bin/env bash
# =============================================================================
# run_cuda.sh — Build and run the CUDA GPU targets for SALR3YUCUDA
#
# Usage:
#   ./run_cuda.sh              # build + run CUDA simulation (default)
#   ./run_cuda.sh build        # build only
#   ./run_cuda.sh sim          # build + run CUDA simulation
#   ./run_cuda.sh tests        # build + run CPU test targets
#   ./run_cuda.sh all          # build + CUDA simulation + tests
#   ./run_cuda.sh clean        # remove build directory
#   ./run_cuda.sh gpuinfo      # print GPU info and exit
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
CONFIG="$PROJECT_ROOT/configs/default.cfg"

# ── colour helpers ────────────────────────────────────────────────────────────
_green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
_cyan()   { printf '\033[0;36m%s\033[0m\n' "$*"; }
_red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
_yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
_bold()   { printf '\033[1m%s\033[0m\n'   "$*"; }

# ── GPU sanity check ──────────────────────────────────────────────────────────
check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        _red "nvidia-smi not found — is the NVIDIA driver installed?"
        exit 1
    fi
    if ! nvidia-smi &>/dev/null; then
        _red "nvidia-smi failed — no GPU available or driver not loaded."
        exit 1
    fi
}

do_gpuinfo() {
    _bold ">>> GPU information"
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
               --format=csv,noheader 2>/dev/null \
    | while IFS=',' read -r name drv mem cc; do
        _cyan "  Name:              $name"
        _cyan "  Driver:            $drv"
        _cyan "  Memory:            $mem"
        _cyan "  Compute cap:       $cc"
      done
    echo ""
    # nvcc version
    if command -v nvcc &>/dev/null; then
        _cyan "  nvcc:              $(nvcc --version | tail -1)"
    fi
}

# ── steps ─────────────────────────────────────────────────────────────────────
do_build() {
    _bold ">>> Building project (CMake, CUDA enabled)"
    cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -Wno-dev
    cmake --build "$BUILD_DIR" --parallel "$(nproc)"

    if [[ ! -x "$BUILD_DIR/salr_dft_cuda" ]]; then
        _red "salr_dft_cuda was not built — CUDA toolkit may be missing."
        exit 1
    fi
    _green "Build complete."
}

do_sim() {
    check_gpu
    _bold ">>> Running CUDA simulation: salr_dft_cuda"
    _cyan "    config: $CONFIG"

    local out_dir="$PROJECT_ROOT/output"
    local data_dir="$out_dir/data"
    mkdir -p "$data_dir"

    # Clear previous output so stale snapshots don't mix with new results
    if compgen -G "$data_dir/density_species*.dat" > /dev/null 2>&1 || \
       [[ -f "$out_dir/convergence.dat" ]] || \
       [[ -f "$out_dir/density_species1_final.dat" ]]; then
        _cyan "    clearing previous output ..."
        rm -f "$data_dir"/density_species*.dat \
              "$out_dir"/convergence.dat \
              "$out_dir"/density_species*_final.dat
    fi

    # Select GPU via CUDA_VISIBLE_DEVICES (default: use all / device 0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    _cyan "    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    time "$BUILD_DIR/salr_dft_cuda" "$CONFIG"
    _green "CUDA simulation finished."
}

do_tests() {
    _bold ">>> Running CPU tests"
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

    [[ $failed -eq 0 ]] && _green "All tests passed." || { _red "$failed test(s) failed."; return 1; }
}

do_clean() {
    _bold ">>> Cleaning build directory"
    rm -rf "$BUILD_DIR"
    _green "Cleaned."
}

# ── dispatch ──────────────────────────────────────────────────────────────────
MODE="${1:-sim}"

case "$MODE" in
    build)    do_build ;;
    sim)      do_build; do_sim ;;
    tests)    do_build; do_tests ;;
    all)      do_build; do_sim; do_tests ;;
    clean)    do_clean ;;
    gpuinfo)  do_gpuinfo ;;
    *)
        _red "Unknown mode: $MODE"
        echo "Usage: $0 [sim|build|tests|all|clean|gpuinfo]"
        exit 1
        ;;
esac
