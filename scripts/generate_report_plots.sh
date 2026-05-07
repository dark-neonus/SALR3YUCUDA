#!/usr/bin/env bash
# =============================================================================
# generate_report_plots.sh — Run CUDA solver + generate scientific plots for
#                             the SALR3YUCUDA IEEE report.
#
# Produces the six placeholder figures required by SALR3YUCUDA.tex:
#   pbc-random.png       pbc-sinusoidal.png
#   w2-random.png        w2-sinusoidal.png
#   w4-random.png        w4-sinusoidal.png
#
# All output goes to  docs/report/SALR3YUCUDA/src/
#
# Usage:
#   ./scripts/generate_report_plots.sh            # run everything
#   ./scripts/generate_report_plots.sh --no-build # skip CMake rebuild
#   ./scripts/generate_report_plots.sh --cpu      # use CPU solver (no GPU)
#   ./scripts/generate_report_plots.sh --dry-run  # print commands, do not run
#
# Requirements:
#   - CUDA toolkit + NVIDIA GPU   (or use --cpu flag for the OpenMP solver)
#   - Python 3 with numpy + matplotlib
#   - CMake, make / ninja
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
DO_BUILD=1
USE_CUDA=1
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        --no-build)  DO_BUILD=0 ;;
        --cpu)       USE_CUDA=0 ;;
        --dry-run)   DRY_RUN=1  ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ── paths ─────────────────────────────────────────────────────────────────────
BUILD_DIR="$PROJECT_ROOT/build"
BASE_CFG="$PROJECT_ROOT/configs/default.cfg"
WORKDIR="$PROJECT_ROOT/output/report_runs"    # per-scenario output dirs live here
TMPDIR_CFG="$PROJECT_ROOT/output/report_configs"
DEST_DIR="$PROJECT_ROOT/docs/report/SALR3YUCUDA/src"
PLOT_SCRIPT="$PROJECT_ROOT/scripts/generate_report_plots.py"
DATABASE_DIR="$PROJECT_ROOT/database"

# ── colour helpers ─────────────────────────────────────────────────────────────
_bold()   { printf '\033[1m%s\033[0m\n'   "$*"; }
_green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
_cyan()   { printf '\033[0;36m%s\033[0m\n' "$*"; }
_yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
_red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }

# ── helpers ───────────────────────────────────────────────────────────────────
run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [dry-run] $*"
    else
        "$@"
    fi
}

# Write a per-scenario config file by patching the base config.
# Args: <dest_cfg> <boundary_mode> <init_mode> <nx> <ny> <dx> <dy>
#       <temperature> <rho1> <rho2> <output_dir>
write_config() {
    local dest="$1"
    local boundary="$2" init="$3"
    local nx="$4" ny="$5" dx="$6" dy="$7"
    local temp="$8" rho1="$9" rho2="${10}"
    local out_dir="${11}"

    python3 - <<PYEOF
import re, sys

src = open("$BASE_CFG").read()

def replace(txt, key, val):
    # replace 'key = <anything>' in the first occurrence
    return re.sub(r'(?m)^(\s*' + re.escape(key) + r'\s*=\s*).*$',
                  r'\g<1>' + str(val), txt, count=1)

overrides = {
    "boundary_mode": "$boundary",
    "init_mode":     "$init",
    "nx":            "$nx",
    "ny":            "$ny",
    "dx":            "$dx",
    "dy":            "$dy",
    "temperature":   "$temp",
    "rho1":          "$rho1",
    "rho2":          "$rho2",
    "output_dir":    "$out_dir/",
    "save_every":    "9999999",   # don't save snapshots, only final
    "max_iterations": "5000",
}

for k, v in overrides.items():
    src = replace(src, k, v)

open("$dest", "w").write(src)
print("  Config written: $dest")
PYEOF
}

# ── build ─────────────────────────────────────────────────────────────────────
if [[ "$DO_BUILD" -eq 1 ]]; then
    _bold ">>> Building project"
    run cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_DB_ENGINE=ON \
        -Wno-dev
    run cmake --build "$BUILD_DIR" --parallel "$(nproc)"
    _green "Build complete."
fi

# Select the solver executable
if [[ "$USE_CUDA" -eq 1 ]]; then
    if [[ -x "$BUILD_DIR/salr_dft_cuda" ]]; then
        SOLVER_EXE="$BUILD_DIR/salr_dft_cuda"
    elif [[ -x "$BUILD_DIR/salr_dft_cuda_db" ]]; then
        SOLVER_EXE="$BUILD_DIR/salr_dft_cuda_db"
    else
        _red "CUDA executable not found. Re-run with --cpu or rebuild."
        exit 1
    fi
    _cyan "Solver: $SOLVER_EXE (CUDA)"
    # Verify GPU is available
    if ! nvidia-smi &>/dev/null 2>&1; then
        _red "nvidia-smi failed — no GPU available. Use --cpu flag."
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
else
    if [[ -x "$BUILD_DIR/salr_dft" ]]; then
        SOLVER_EXE="$BUILD_DIR/salr_dft"
    elif [[ -x "$BUILD_DIR/salr_dft_db" ]]; then
        SOLVER_EXE="$BUILD_DIR/salr_dft_db"
    else
        _red "CPU executable not found. Rebuild with: ./run.sh build"
        exit 1
    fi
    _cyan "Solver: $SOLVER_EXE (CPU/OpenMP)"
fi

# ── scenario definitions ──────────────────────────────────────────────────────
# Each entry: "name|boundary|init|nx|ny|dx|dy|T|rho1|rho2"
declare -a SCENARIOS=(
    "pbc-random|PBC|random|160|160|0.1|0.1|8.0|0.2|0.2"
    "pbc-sinusoidal|PBC|sinusoids|160|160|0.1|0.1|8.0|0.2|0.2"
    "w2-random|W2|random|160|160|0.1|0.1|8.0|0.2|0.2"
    "w2-sinusoidal|W2|sinusoids|160|160|0.1|0.1|8.0|0.2|0.2"
    "w4-random|W4|random|160|160|0.1|0.1|8.0|0.2|0.2"
    "w4-sinusoidal|W4|sinusoids|160|160|0.1|0.1|8.0|0.2|0.2"
)

# Friendly titles for each scenario (used as figure title in the PNG)
declare -A TITLES=(
    ["pbc-random"]="Periodic Boundary Conditions — Random Initial Density"
    ["pbc-sinusoidal"]="Periodic Boundary Conditions — Sinusoidal Initial Density"
    ["w2-random"]="Two-Wall Confinement (W2) — Random Initial Density"
    ["w2-sinusoidal"]="Two-Wall Confinement (W2) — Sinusoidal Initial Density"
    ["w4-random"]="Four-Wall Confinement (W4) — Random Initial Density"
    ["w4-sinusoidal"]="Four-Wall Confinement (W4) — Sinusoidal Initial Density"
)

# ── prepare directories ───────────────────────────────────────────────────────
run mkdir -p "$WORKDIR" "$TMPDIR_CFG" "$DEST_DIR"
export SALR_DB_PATH="$DATABASE_DIR"
run mkdir -p "$DATABASE_DIR"

# ── main loop ─────────────────────────────────────────────────────────────────
FAILED_SCENARIOS=()

for entry in "${SCENARIOS[@]}"; do
    IFS='|' read -r name boundary init nx ny dx dy temp rho1 rho2 <<< "$entry"

    echo ""
    _bold "══════════════════════════════════════════════════════════════"
    _bold "  Scenario: $name"
    _cyan "  Boundary: $boundary   Init: $init"
    _cyan "  Grid: ${nx}x${ny}  dx=${dx}  T=${temp}  rho1=${rho1}  rho2=${rho2}"
    _bold "══════════════════════════════════════════════════════════════"

    out_dir="$WORKDIR/$name"
    cfg_file="$TMPDIR_CFG/${name}.cfg"

    run mkdir -p "$out_dir/data"

    # Write scenario-specific config
    if [[ "$DRY_RUN" -eq 0 ]]; then
        write_config "$cfg_file" \
            "$boundary" "$init" \
            "$nx" "$ny" "$dx" "$dy" \
            "$temp" "$rho1" "$rho2" \
            "$out_dir"
    else
        echo "  [dry-run] write_config $cfg_file ..."
    fi

    # Run the solver (non-zero exit = non-convergence warning, data still written)
    _cyan "  Running solver..."
    solver_ok=1
    run "$SOLVER_EXE" "$cfg_file" || solver_ok=0
    if [[ "$solver_ok" -eq 0 ]]; then
        _yellow "  Solver did not converge for $name — will plot partial result."
    else
        _green "  Solver finished."
    fi

    # Check output exists
    if [[ "$DRY_RUN" -eq 0 ]]; then
        if [[ ! -f "$out_dir/data/density_species1_final.dat" ]]; then
            _red "  Output data not found for $name — skipping plot."
            FAILED_SCENARIOS+=("$name")
            continue
        fi
    fi

    # Build parameter annotation string for the figure
    params_text="T=${temp},  ρ₁=${rho1},  ρ₂=${rho2},  rᶜ=8.0,  grid ${nx}×${ny},  Δx=Δy=${dx},  ξ=0.02,  ε=1×10⁻⁸"
    plot_title="${TITLES[$name]}"
    [[ "$solver_ok" -eq 0 ]] && plot_title="${plot_title} (partial, not converged)"

    # Use high-contrast diverging colormaps for wall boundary modes
    # and trim boundary cells that have forced BC values
    cmap_args=()
    if [[ "$boundary" == W* ]]; then
        local trim_cells=5
        [[ "$boundary" == "W4" ]] && trim_cells=6
        cmap_args=(--cmap1 RdBu_r --cmap2 PuOr --trim "$trim_cells")
    fi

    # Generate the scientific figure
    _cyan "  Generating figure..."
    run python3 "$PLOT_SCRIPT" \
        "$out_dir" \
        "$name" \
        --dest "$DEST_DIR" \
        --title "$plot_title" \
        --params "$params_text" \
        --dpi 200 \
        "${cmap_args[@]}"

    if [[ "$DRY_RUN" -eq 0 && -f "$DEST_DIR/${name}.png" ]]; then
        _green "  Output: $DEST_DIR/${name}.png"
    fi
done

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
_bold "══════════════════════════════════════════════════════════════"
_bold "  Done"
_bold "══════════════════════════════════════════════════════════════"

if [[ "${#FAILED_SCENARIOS[@]}" -eq 0 ]]; then
    _green "All scenarios completed successfully."
else
    _yellow "Failed scenarios (check solver output above):"
    for s in "${FAILED_SCENARIOS[@]}"; do
        _yellow "  - $s"
    done
fi

echo ""
_cyan "Report images are in: $DEST_DIR"
ls -lh "$DEST_DIR"/*.png 2>/dev/null | awk '{print "  " $NF "  (" $5 ")"}'
