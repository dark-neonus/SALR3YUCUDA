# SALR3YUCUDA

Numerical solver for mean-field Density Functional Theory (DFT) equations describing spatial density distributions in two-component SALR (Short-Range Attraction, Long-Range Repulsion) colloidal systems with Triple Yukawa pair potentials. Developed for the "Architecture of Computer Systems" course.

## Overview

The program solves a system of nonlinear integral equations via Picard iteration to obtain equilibrium density profiles ρ₁(r) and ρ₂(r) for a binary mixture confined in a two-dimensional domain with periodic or hard-wall boundary conditions.

**Features:**
- CPU solver with OpenMP parallelization
- CUDA GPU acceleration
- Qt6-based visualization GUI with real-time heatmaps and 3D scatter plots
- HDF5-based session database for experiment management
- Professional benchmarking tools

## Project Structure

```
configs/            Configuration files (.cfg)
include/            C headers
src/
  main.c            Entry point
  core/             Configuration parser, grid initialization
  cpu/              CPU implementations (solver, potential, math)
  cuda/             GPU kernels
  utils/            File I/O helpers
visualization_gui/  Qt6 visualization application
  include/          Widget headers
  src/              Widget implementations
database_engine/    HDF5-based session storage
tests/              Validation and test programs
scripts/            Python visualization and benchmarking
output/             Simulation output (not tracked)
docs/               Documentation
```

## Requirements

- CMake >= 3.18
- C11 compiler (GCC or Clang)
- OpenMP support
- CUDA Toolkit (optional, for GPU acceleration)
- Qt6 with OpenGL (for GUI)
- HDF5 library
- Python 3 with matplotlib, numpy (for visualization)

```bash
pip install -r requirements.txt
```

## Build

### Using the provided scripts

```bash
./run.sh              # build, run simulation, and execute all tests
./run.sh build        # configure and compile only
./run.sh sim          # build and run the main simulation
./run.sh tests        # build and run all test binaries
./run.sh clean        # remove the build/ directory

./run_cuda.sh         # build and run with CUDA support
./run_gui.sh          # build and run the visualization GUI
```

### Manual build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

./build/salr_dft configs/default.cfg
./build/salr_dft_gui   # visualization GUI
```

See [docs/BUILD.md](docs/BUILD.md) for full build instructions.

## Visualization GUI

The Qt6 GUI provides:

- **Session Browser**: View and manage simulation runs stored in the database
- **Snapshot Browser**: Navigate through iteration snapshots
- **2D Heatmap**: Density visualization with customizable colors
- **3D Scatter Plot**: Interactive 3D density point cloud
- **Parameter Controls**: 
  - Collapsible parameter sections (Grid, Physics, Solver)
  - Initial distribution selection (Random, Sinusoidal, Uniform)
  - Species color customization
  - Linear/logarithmic axis scaling
- **Simulation Control**: Start CPU/CUDA runs, resume from snapshots

Launch with:
```bash
./run_gui.sh
# or
./build/salr_dft_gui
```

## Benchmarking

Generate performance comparison plots:

```bash
# Run benchmarks (collects timing data)
python scripts/benchmarking/run_benchmarks.py --grid-sizes 32 64 128 256 --threads 1 2 4 8

# Generate publication-quality plots
python scripts/benchmarking/benchmark_cuda_cpu.py

# Convergence-focused comparison on real default.cfg parameters
python scripts/benchmarking/performance_real_params.py
```

Output plots (PNG + SVG at 300 DPI) are saved to `output/benchmark_plots/`:
- `performance_comparison`: Execution time vs grid size
- `speedup_analysis`: CUDA speedup and CPU parallel scaling
- `parallel_efficiency`: OpenMP efficiency analysis
- `scaling_analysis`: Computational complexity

## Configuration

Simulations are controlled through `.cfg` files in INI format:

| Section | Key Parameters | Description |
|---------|---------------|-------------|
| `[grid]` | `nx`, `ny`, `dx`, `dy` | Grid dimensions and spacing |
| `[physics]` | `temperature`, `rho1`, `rho2`, `cutoff_radius` | Thermodynamic parameters |
| `[interaction]` | `A_ij_m`, `a_ij_m` | 3-Yukawa potential coefficients |
| `[solver]` | `max_iterations`, `tolerance`, `xi1`, `xi2` | Picard iteration settings |
| `[output]` | `output_dir`, `save_every` | Output configuration |

See `configs/default.cfg` for a reference configuration.

## Output

| File | Description |
|------|-------------|
| `parameters.cfg` | Snapshot of simulation parameters |
| `convergence.dat` | L2 error per iteration |
| `density_species{1,2}_final.dat` | Converged density profiles |
| `data/density_species{1,2}_iter_*.dat` | Intermediate snapshots |

## License

See [LICENSE](LICENSE) for details.
