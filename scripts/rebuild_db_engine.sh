#!/bin/bash
#
# rebuild_db_engine.sh - Rebuild the database engine and related executables
#
# Usage:
#   ./scripts/rebuild_db_engine.sh           # Rebuild with database engine
#   ./scripts/rebuild_db_engine.sh --clean   # Clean and rebuild
#   ./scripts/rebuild_db_engine.sh --no-cuda # Build without CUDA support
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

# Parse arguments
CLEAN=0
CUDA_OPT=""
BUILD_TYPE="Release"

for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN=1
            ;;
        --no-cuda)
            CUDA_OPT="-DENABLE_CUDA=OFF"
            ;;
        --debug)
            BUILD_TYPE="Debug"
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --clean     Clean build directory before building"
            echo "  --no-cuda   Build without CUDA support"
            echo "  --debug     Build with debug symbols"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Clean if requested
if [ "$CLEAN" -eq 1 ]; then
    echo "==> Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "==> Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DENABLE_DB_ENGINE=ON \
    $CUDA_OPT

# Build
echo "==> Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "==> Build complete!"
echo ""
echo "Available executables:"
ls -1 "$BUILD_DIR"/salr_dft* 2>/dev/null || echo "  (none found)"
echo ""
echo "Database engine tests:"
ls -1 "$BUILD_DIR"/test_* 2>/dev/null || echo "  (none found)"
echo ""

# Check if database engine was built
if [ -f "$BUILD_DIR/salr_dft_db" ]; then
    echo "Database engine enabled: YES"
    echo "  Run with: ./build/salr_dft_db configs/default.cfg"
    echo "  Resume:   ./build/salr_dft_db configs/default.cfg --resume <run_id> [iteration]"
else
    echo "Database engine enabled: NO (missing HDF5 or SQLite3 libraries)"
    echo "  Install: sudo apt install libhdf5-dev libsqlite3-dev"
fi
