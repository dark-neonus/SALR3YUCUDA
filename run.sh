#!/usr/bin/env bash
# =============================================================================
# run.sh — Build and run all targets for SALR3YUCUDA with database support
#
# Usage:
#   ./run.sh              # build + run main simulation with DB (default)
#   ./run.sh build        # build only
#   ./run.sh sim          # build + run simulation with database
#   ./run.sh sim-nodb     # build + run simulation without database
#   ./run.sh resume <id>  # resume from a previous run
#   ./run.sh list         # list all runs in the database
#   ./run.sh tests        # build + run test targets only
#   ./run.sh all          # build + run simulation + run tests
#   ./run.sh clean        # remove build directory
#
# Database:
#   Results are stored in project_root/database/ with HDF5 snapshots
#   Use 'resume' mode to continue from a checkpoint
#
# Multithreading:
#   Uses OMP_NUM_THREADS (default: all cores) for OpenMP parallelization
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

# ── steps ─────────────────────────────────────────────────────────────────────
do_build() {
    _bold ">>> Building project (CMake with database engine)"
    cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_DB_ENGINE=ON \
        -Wno-dev
    cmake --build "$BUILD_DIR" --parallel "$(nproc)"
    _green "Build complete."
}

do_sim() {
    local use_db="${1:-yes}"
    local resume_id="${2:-}"
    local resume_iter="${3:-}"

    # Determine which executable to use
    local exe
    if [[ "$use_db" == "yes" && -x "$BUILD_DIR/salr_dft_db" ]]; then
        exe="$BUILD_DIR/salr_dft_db"
        _bold ">>> Running simulation: salr_dft_db (with HDF5 database)"
    else
        exe="$BUILD_DIR/salr_dft"
        _bold ">>> Running simulation: salr_dft (no database)"
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
            _cyan "    clearing previous output in $data_dir/ and $out_dir/ ..."
            rm -f "$data_dir"/density_species*.dat \
                  "$out_dir"/convergence.dat \
                  "$out_dir"/density_species*_final.dat
        fi
    fi

    # Set OpenMP threads for multithreading (use all available cores)
    local num_threads
    num_threads="${OMP_NUM_THREADS:-$(nproc)}"
    export OMP_NUM_THREADS="$num_threads"
    _cyan "    using $num_threads OpenMP threads"

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

    "${cmd[@]}"
    _green "Simulation finished."

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

    # Check for sessions.db
    if [[ ! -f "$DATABASE_DIR/sessions.db" ]]; then
        _yellow "No sessions.db found in $DATABASE_DIR"
        echo "No runs yet. Run a simulation first with: $0 sim"
        return 0
    fi

    # Use sqlite3 to list runs
    if command -v sqlite3 &>/dev/null; then
        _cyan "Runs in $DATABASE_DIR/sessions.db:"
        echo ""
        sqlite3 -header -column "$DATABASE_DIR/sessions.db" \
            "SELECT run_id, nickname, created_at, temperature, rho1_bulk, rho2_bulk, snapshot_count, converged
             FROM sessions ORDER BY created_at DESC LIMIT 20;"
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

show_help() {
    cat <<EOF
SALR DFT Solver - Run Script

Usage: $0 [command] [options]

Commands:
  sim           Build and run simulation with HDF5 database (default)
  sim-nodb      Build and run simulation without database
  resume <id>   Resume from a previous run (optional: specify iteration)
  list          List all runs in the database
  build         Build only (no simulation)
  tests         Build and run test suite
  all           Build, run simulation, and run tests
  clean         Remove build directory

Examples:
  $0                           # Run simulation with database
  $0 sim                       # Same as above
  $0 resume run_20260323_1430  # Resume from a specific run
  $0 resume run_20260323_1430 100  # Resume from iteration 100
  $0 list                      # Show all previous runs

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
    clean)      do_clean ;;
    all)        do_build; do_sim "yes"; do_tests ;;
    help|-h|--help) show_help ;;
    *)
        _red "Unknown mode: $MODE"
        echo "Usage: $0 [sim|sim-nodb|resume|list|build|tests|all|clean|help]"
        exit 1
        ;;
esac
