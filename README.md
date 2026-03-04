# SALR3YUCUDA

Numerical solver for the mean-field Density Functional Theory (DFT) equations describing spatial density distributions in two-component SALR (Short-Range Attraction, Long-Range Repulsion) colloidal systems with Triple Yukawa pair potentials. Developed as part of the "Architecture of Computer Systems" course.

## Overview

The program solves a system of nonlinear integral equations via Picard iteration to obtain equilibrium density profiles $\rho_1(\mathbf{r})$ and $\rho_2(\mathbf{r})$ for a binary mixture confined in a two-dimensional domain with periodic or hard-wall boundary conditions.

**Current state:** the CPU (pure C) implementation is complete and functional. CUDA GPU acceleration is planned as the next development phase.

## Project Structure

```
configs/            Configuration files (.cfg)
include/            C headers (shared between CPU and future CUDA builds)
src/
  main.c            Entry point
  core/             Configuration parser, grid initialisation
  cpu/              CPU implementations (solver, potential, math utilities)
  cuda/             GPU kernels (planned)
  math/             Specialised math routines (planned)
  utils/            File I/O helpers
tests/              Validation and test programs
scripts/            Python and gnuplot visualisation scripts
output/             Simulation output (not tracked in version control)
docs/               Build instructions, bug reports, project documentation
```

## Requirements

- CMake >= 3.18
- C11 compiler (GCC or Clang)
- OpenMP support
- Python 3 with matplotlib and numpy (for visualisation); install via:
  ```bash
  pip install -r requirements.txt
  ```

## Build and Run

### Using the provided script

```bash
./run.sh              # build, run simulation, and execute all tests
./run.sh build        # configure and compile only
./run.sh sim          # build and run the main simulation
./run.sh tests        # build and run all test binaries
./run.sh clean        # remove the build/ directory
```

### Manual build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

./build/salr_dft configs/default.cfg
./build/test_solver
./build/test_potential
```

See [docs/BUILD.md](docs/BUILD.md) for full build instructions.

### Visualisation

After a completed simulation run:

```bash
pip install -r requirements.txt   # if not already installed
python3 scripts/plot_joint_heatmap.py output/
python3 scripts/plot_density.py output/
```

The first command produces a joint two-colour heatmap (`output/joint_heatmap_final.png`). The second generates individual species heatmaps and a convergence plot via gnuplot.

## Output

The `output/` directory contains the following files after a simulation:

| File | Description |
|------|-------------|
| `parameters.cfg` | Snapshot of all simulation parameters used in the run |
| `convergence.dat` | L2 error per iteration (columns: iteration, error) |
| `density_species1_final.dat` | Converged density profile for species 1 (columns: x, y, rho) |
| `density_species2_final.dat` | Converged density profile for species 2 (columns: x, y, rho) |
| `data/density_species{1,2}_iter_NNNNNN.dat` | Intermediate snapshots at intervals defined by `save_every` |

## Configuration

Simulations are controlled through `.cfg` files in INI format. The main sections are:

| Section | Key Parameters | Description |
|---------|---------------|-------------|
| `[grid]` | `nx`, `ny`, `Lx`, `Ly`, `dx`, `dy` | Computational grid dimensions and physical domain size |
| `[physics]` | `temperature`, `rho1`, `rho2`, `cutoff_radius` | Thermodynamic state and interaction cutoff |
| `[interaction]` | `A_ij_m`, `a_ij_m` | 3-Yukawa amplitude and decay-rate coefficient matrices |
| `[solver]` | `max_iterations`, `tolerance`, `xi1`, `xi2` | Picard iteration parameters |
| `[output]` | `output_dir`, `save_every` | Output directory and snapshot frequency |

See `configs/default.cfg` for a reference configuration.

## Development Roadmap

### Completed

- Project structure and header interfaces
- Configuration file parser (all 3-Yukawa parameters)
- Grid initialisation (cell-centred, periodic and walled boundaries)
- 3-Yukawa potential evaluation (CPU)
- DFT solver with Picard iteration (CPU, O(N^2) convolution)
- Density output (ASCII, gnuplot-compatible format)
- Convergence logging
- Python and gnuplot visualisation scripts

### Planned

- CUDA port of the convolution kernel and solver
- CPU vs GPU performance benchmarking
- Validation tests against known analytical limits
