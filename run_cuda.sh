#!/usr/bin/env bash
# =============================================================================
# run_cuda.sh — Build and run CUDA GPU targets for SALR3YUCUDA with database
#
# Usage:
#   ./run_cuda.sh              # build + run CUDA simulation with DB (default)
#   ./run_cuda.sh build        # build only
#   ./run_cuda.sh sim          # build + run CUDA simulation with database
#   ./run_cuda.sh sim-nodb     # build + run CUDA simulation without database
#   ./run_cuda.sh resume <id>  # resume from a previous run
#   ./run_cuda.sh list         # list all runs in the database
#   ./run_cuda.sh tests        # build + run test targets
#   ./run_cuda.sh all          # build + CUDA simulation + tests
#   ./run_cuda.sh clean        # remove build directory
#   ./run_cuda.sh gpuinfo      # print GPU info and exit
#
# Database:
#   Results are stored in project_root/database/ with HDF5 snapshots
#   Use 'resume' mode to continue from a checkpoint
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
CONFIG="$PROJECT_ROOT/configs/default.cfg"
DATABASE_DIR="$PROJECT_ROOT/database"

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
    _bold ">>> Building project (CMake, CUDA + database engine enabled)"
    cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_DB_ENGINE=ON \
        -Wno-dev
    cmake --build "$BUILD_DIR" --parallel "$(nproc)"

    if [[ ! -x "$BUILD_DIR/salr_dft_cuda" ]]; then
        _red "salr_dft_cuda was not built — CUDA toolkit may be missing."
        exit 1
    fi
    _green "Build complete."
}

do_sim() {
    check_gpu
    local use_db="${1:-yes}"
    local resume_id="${2:-}"
    local resume_iter="${3:-}"

    # Determine which executable to use
    local exe
    if [[ "$use_db" == "yes" && -x "$BUILD_DIR/salr_dft_cuda_db" ]]; then
        exe="$BUILD_DIR/salr_dft_cuda_db"
        _bold ">>> Running CUDA simulation: salr_dft_cuda_db (with HDF5 database)"
    else
        exe="$BUILD_DIR/salr_dft_cuda"
        _bold ">>> Running CUDA simulation: salr_dft_cuda (no database)"
    fi

    _cyan "    config: $CONFIG"

    # Create database directory and export path for the executable
    mkdir -p "$DATABASE_DIR"
    export SALR_DB_PATH="$DATABASE_DIR"

    # Set up output directory
    local out_dir="$PROJECT_ROOT/output"
    local data_dir="$out_dir/data"
    mkdir -p "$data_dir"

    # Clear previous ASCII output (HDF5 data is preserved in database/)
    if [[ "$use_db" != "yes" ]]; then
        if compgen -G "$data_dir/density_species*.dat" > /dev/null 2>&1 || \
           [[ -f "$out_dir/convergence.dat" ]] || \
           [[ -f "$out_dir/density_species1_final.dat" ]]; then
            _cyan "    clearing previous output ..."
            rm -f "$data_dir"/density_species*.dat \
                  "$out_dir"/convergence.dat \
                  "$out_dir"/density_species*_final.dat
        fi
    fi

    # Select GPU via CUDA_VISIBLE_DEVICES (default: use device 0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    _cyan "    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    if [[ "$use_db" == "yes" ]]; then
        _cyan "    database: $DATABASE_DIR"
    fi

    # Build command with optional resume args
    local cmd=("$exe" "$CONFIG")
    if [[ -n "$resume_id" ]]; then
        cmd+=("--resume" "$resume_id")
        if [[ -n "$resume_iter" ]]; then
            cmd+=("$resume_iter")
        fi
    fi

    time "${cmd[@]}"
    _green "CUDA simulation finished."

    if [[ "$use_db" == "yes" ]]; then
        _cyan "    HDF5 snapshots saved to: $DATABASE_DIR/"
    fi
}

do_resume() {
    local run_id="${1:-}"
    local iter="${2:-}"

    if [[ -z "$run_id" ]]; then
        _red "Error: resume requires a run_id"
        echo "Usage: $0 resume <run_id> [iteration]"
        echo ""
        echo "List available runs with: $0 list"
        exit 1
    fi

    do_build
    do_sim "yes" "$run_id" "$iter"
}

do_list() {
    _bold ">>> Listing runs in database"

    if [[ ! -d "$DATABASE_DIR" ]]; then
        _yellow "Database directory not found: $DATABASE_DIR"
        echo "No runs yet. Run a simulation first with: $0 sim"
        return 0
    fi

    # Check for registry.db
    if [[ ! -f "$DATABASE_DIR/registry.db" ]]; then
        _yellow "No registry.db found in $DATABASE_DIR"
        echo "No runs yet. Run a simulation first with: $0 sim"
        return 0
    fi

    # Use sqlite3 to list runs
    if command -v sqlite3 &>/dev/null; then
        _cyan "Runs in $DATABASE_DIR/registry.db:"
        echo ""
        sqlite3 -header -column "$DATABASE_DIR/registry.db" \
            "SELECT run_id, nickname, created_at, temperature, rho1_bulk, rho2_bulk, snapshot_count, converged
             FROM runs ORDER BY created_at DESC LIMIT 20;"
    else
        _yellow "sqlite3 not found. Install with: sudo apt install sqlite3"
        echo ""
        echo "Run directories in $DATABASE_DIR:"
        ls -1d "$DATABASE_DIR"/run_* 2>/dev/null || echo "  (none)"
    fi
}

do_tests() {
    _bold ">>> Running tests"
    local failed=0

    for target in test_solver test_potential test_db_integration test_hdf5; do
        exe="$BUILD_DIR/$target"
        # test_db_integration and test_hdf5 are in database_engine/
        if [[ ! -x "$exe" ]]; then
            exe="$BUILD_DIR/database_engine/$target"
        fi
        if [[ ! -x "$exe" ]]; then
            _yellow "  [SKIP] $target — binary not found"
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

show_help() {
    cat <<EOF
SALR DFT Solver (CUDA) - Run Script

Usage: $0 [command] [options]

Commands:
  sim           Build and run CUDA simulation with HDF5 database (default)
  sim-nodb      Build and run CUDA simulation without database
  resume <id>   Resume from a previous run (optional: specify iteration)
  list          List all runs in the database
  build         Build only (no simulation)
  tests         Build and run test suite
  all           Build, run simulation, and run tests
  clean         Remove build directory
  gpuinfo       Display GPU information

Examples:
  $0                           # Run CUDA simulation with database
  $0 sim                       # Same as above
  $0 resume run_20260323_1430  # Resume from a specific run
  $0 resume run_20260323_1430 100  # Resume from iteration 100
  $0 list                      # Show all previous runs
  $0 gpuinfo                   # Show GPU info

Database:
  Results are stored in: $DATABASE_DIR/
  Each run creates HDF5 snapshots that can be resumed later.

EOF
}

# ── dispatch ──────────────────────────────────────────────────────────────────
MODE="${1:-sim}"

case "$MODE" in
    build)      do_build ;;
    sim)        do_build; do_sim "yes" ;;
    sim-nodb)   do_build; do_sim "no" ;;
    resume)     do_resume "${2:-}" "${3:-}" ;;
    list)       do_list ;;
    tests)      do_build; do_tests ;;
    all)        do_build; do_sim "yes"; do_tests ;;
    clean)      do_clean ;;
    gpuinfo)    do_gpuinfo ;;
    help|-h|--help) show_help ;;
    *)
        _red "Unknown mode: $MODE"
        echo "Usage: $0 [sim|sim-nodb|resume|list|build|tests|all|clean|gpuinfo|help]"
        exit 1
        ;;
esac
