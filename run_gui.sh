#!/usr/bin/env bash
# =============================================================================
# run_gui.sh — Build and run the Qt Visualization GUI for SALR3YUCUDA
#
# Usage:
#   ./run_gui.sh              # build + run GUI (default)
#   ./run_gui.sh build        # build only
#   ./run_gui.sh run          # run only (fails if not built)
#   ./run_gui.sh clean        # remove build directory
#
# Dependencies required for GUI (Debian/Ubuntu):
#   sudo apt install qt6-base-dev libqt6opengl6-dev libgl1-mesa-dev
#   sudo apt install libhdf5-dev libsqlite3-dev
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
DATABASE_DIR="$PROJECT_ROOT/database"
GUI_EXE="$BUILD_DIR/visualization_gui/salr_gui"

# ── colour helpers ────────────────────────────────────────────────────────────
_green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
_cyan()   { printf '\033[0;36m%s\033[0m\n' "$*"; }
_red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
_yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
_bold()   { printf '\033[1m%s\033[0m\n'   "$*"; }

# ── steps ─────────────────────────────────────────────────────────────────────
do_build() {
    _bold ">>> Building GUI project (CMake with GUI and database engine)"
    
    # Configure CMake with GUI and Database turned on
    cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_DB_ENGINE=ON \
        -DBUILD_GUI=ON \
        -Wno-dev
    
    # Build it
    cmake --build "$BUILD_DIR" --parallel "$(nproc)"
    
    if [[ ! -x "$GUI_EXE" ]]; then
        _red "Build failed or salr_gui was not generated. Check if Qt6 is installed."
        exit 1
    fi
    _green "Build complete."
}

do_run() {
    if [[ ! -x "$GUI_EXE" ]]; then
        _red "GUI executable not found! Building first..."
        do_build
    fi

    # Ensure database dir exists so we can pass it safely
    mkdir -p "$DATABASE_DIR"

    _bold ">>> Running SALR Visualization GUI"
    _cyan "    Database path: $DATABASE_DIR"
    
    # Run the GUI
    "$GUI_EXE" --database "$DATABASE_DIR"
}

do_clean() {
    _bold ">>> Cleaning build directory"
    rm -rf "$BUILD_DIR"
    _green "Cleaned."
}

show_help() {
    cat <<EOF
SALR DFT Solver - GUI Run Script

Usage: $0 [command]

Commands:
  all           Build and run the GUI (default)
  build         Build the GUI only
  run           Run the GUI (expects it to be built)
  clean         Remove the build directory

Examples:
  $0            # Build and run
  $0 build      # Just build
EOF
}

# ── dispatch ──────────────────────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    build)      do_build ;;
    run)        do_run ;;
    all)        do_build; do_run ;;
    clean)      do_clean ;;
    help|-h|--help) show_help ;;
    *)
        _red "Unknown mode: $MODE"
        echo "Usage: $0 [all|build|run|clean|help]"
        exit 1
        ;;
esac
