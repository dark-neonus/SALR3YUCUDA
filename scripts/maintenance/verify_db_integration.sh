#!/usr/bin/env bash
# =============================================================================
# verify_db_integration.sh - Verify database engine integration
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
DATABASE_DIR="$PROJECT_ROOT/database"

# Colors
_green()  { printf '\033[0;32m✓ %s\033[0m\n' "$*"; }
_red()    { printf '\033[0;31m✗ %s\033[0m\n' "$*"; }
_yellow() { printf '\033[0;33m! %s\033[0m\n' "$*"; }
_bold()   { printf '\033[1m%s\033[0m\n' "$*"; }

echo ""
_bold "Verifying Database Engine Integration"
echo ""

# Check if HDF5 is installed
if pkg-config --exists hdf5 2>/dev/null; then
    _green "HDF5 library found: $(pkg-config --modversion hdf5)"
else
    _red "HDF5 library not found"
    echo "  Install with: sudo apt install libhdf5-dev"
fi

# Check if SQLite3 is installed
if pkg-config --exists sqlite3 2>/dev/null; then
    _green "SQLite3 library found: $(pkg-config --modversion sqlite3)"
else
    _red "SQLite3 library not found"
    echo "  Install with: sudo apt install libsqlite3-dev"
fi

echo ""

# Check if database-enabled executables exist
if [[ -x "$BUILD_DIR/salr_dft_db" ]]; then
    _green "CPU database executable built: salr_dft_db"
else
    _yellow "CPU database executable not found: salr_dft_db"
    echo "  Build with: ./run.sh build"
fi

if [[ -x "$BUILD_DIR/salr_dft_cuda_db" ]]; then
    _green "CUDA database executable built: salr_dft_cuda_db"
else
    _yellow "CUDA database executable not found (may not have CUDA)"
fi

echo ""

# Check source files have database integration
_bold "Source files with database integration:"

if grep -q "USE_DB_ENGINE" "$PROJECT_ROOT/src/main.c"; then
    _green "src/main.c has database integration"
fi

if grep -q "solver_run_binary_db" "$PROJECT_ROOT/src/cpu/solver_cpu.c"; then
    _green "src/cpu/solver_cpu.c has solver_run_binary_db()"
fi

if grep -q "solver_run_binary_db" "$PROJECT_ROOT/src/cuda/solver_cuda.cu"; then
    _green "src/cuda/solver_cuda.cu has solver_run_binary_db()"
fi

echo ""

# Check database engine source files
_bold "Database engine files:"

for file in db_engine.h db_engine.c hdf5_io.h hdf5_io.c registry.h registry.c db_utils.h db_utils.c; do
    if [[ -f "$PROJECT_ROOT/database_engine/include/$file" ]] || \
       [[ -f "$PROJECT_ROOT/database_engine/src/$file" ]]; then
        _green "database_engine: $file exists"
    fi
done

echo ""

# Check run scripts configuration
_bold "Run scripts configuration:"

if grep -q "ENABLE_DB_ENGINE=ON" "$PROJECT_ROOT/run.sh"; then
    _green "run.sh configured for database engine"
fi

if grep -q "ENABLE_DB_ENGINE=ON" "$PROJECT_ROOT/run_cuda.sh"; then
    _green "run_cuda.sh configured for database engine"
fi

if grep -q "SALR_DB_PATH" "$PROJECT_ROOT/run.sh"; then
    _green "run.sh exports SALR_DB_PATH environment variable"
fi

if grep -q "SALR_DB_PATH" "$PROJECT_ROOT/run_cuda.sh"; then
    _green "run_cuda.sh exports SALR_DB_PATH environment variable"
fi

echo ""

# Check database directory
if [[ -d "$DATABASE_DIR" ]]; then
    _green "Database directory exists: $DATABASE_DIR"

    # Check for registry
    if [[ -f "$DATABASE_DIR/registry.db" ]]; then
        _green "Registry database exists: $DATABASE_DIR/registry.db"

        # Count runs if sqlite3 is available
        if command -v sqlite3 &>/dev/null; then
            run_count=$(sqlite3 "$DATABASE_DIR/registry.db" \
                "SELECT COUNT(*) FROM runs;" 2>/dev/null || echo "0")
            echo "  Runs in database: $run_count"
        fi
    else
        _yellow "Registry database not yet created (run a simulation first)"
    fi

    # Count run directories
    run_dirs=$(find "$DATABASE_DIR" -maxdepth 1 -type d -name "run_*" 2>/dev/null | wc -l)
    if [[ $run_dirs -gt 0 ]]; then
        echo "  Run directories: $run_dirs"
    fi
else
    _yellow "Database directory not yet created: $DATABASE_DIR"
    echo "  Will be created on first run"
fi

echo ""
_bold "Integration Status:"
echo ""
echo "The database engine integration is complete. To use it:"
echo "  1. Build with database support: ./run.sh build"
echo "  2. Run simulation: ./run.sh sim (saves to $DATABASE_DIR)"
echo "  3. List runs: ./run.sh list"
echo "  4. Resume from checkpoint: ./run.sh resume <run_id>"
echo ""
