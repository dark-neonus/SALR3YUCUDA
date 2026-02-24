# SALR3YUCUDA
Short-Range Attraction and Long-Range Repulsion Colloids with 3 Yukawa potentials calculation using CUDA , developed as a part of the "Architecture of Computer Systems" course.
## What this project does

Numerically solves a system of nonlinear integral equations (mean-field DFT) describing the spatial density distribution of SALR particles confined between two walls.

**Development plan:** pure C (CPU) first → CUDA GPU acceleration later.

## Project structure

```
├── configs/            # Simulation parameters (.cfg files)
│   └── default.cfg     # 2D test case
├── include/            # C headers (shared between CPU and future CUDA)
│   ├── config.h        #   config parser interface
│   ├── grid.h          #   computational grid
│   ├── potential.h     #   SALR potential + Aij interaction matrix
│   ├── solver.h        #   iterative DFT solver
│   ├── math_utils.h    #   vector/matrix math
│   └── io.h            #   file I/O helpers
├── src/
│   ├── main.c          # Entry point
│   ├── core/           # Config parser, grid initialisation
│   ├── cpu/            # CPU implementations (active development)
│   ├── cuda/           # GPU kernels (future — will mirror cpu/)
│   ├── math/           # Specialised math routines (future)
│   └── utils/          # I/O, timing, helpers
├── tests/              # Validation programs
├── scripts/            # Python plotting / post-processing
├── output/             # Runtime data (git-ignored)
├── studying/           # Standalone CUDA learning examples (separate)
├── docs/BUILD.md       # Build instructions
├── run.sh              # One-shot build-and-run script (see below)
├── CMakeLists.txt      # CMake build (pure C)
└── Makefile            # Alternative Make build (pure C)
```

## Build & run

Requires: **CMake ≥ 3.12** and a **C11 compiler** (gcc / clang).

### Quick start — `run.sh`

```bash
./run.sh              # build + run simulation + run all tests (default)
./run.sh build        # configure & compile only
./run.sh sim          # build + run main simulation
./run.sh tests        # build + run all test binaries
./run.sh clean        # delete the build/ directory
```

The script always (re-)runs CMake before building, so it is safe to call after any source or config change.

### Manual build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

./build/salr_dft configs/default.cfg
./build/test_solver
./build/test_potential
```

> Full instructions: [docs/BUILD.md](docs/BUILD.md)

## Configuration

The simulation is controlled via `.cfg` files. Key sections:

| Section | Parameters | Description |
|---------|-----------|-------------|
| `[grid]` | `nx`, `ny`, `Lx`, `Ly`, `dx`, `dy` | Computational grid size and physical domain |
| `[physics]` | `temperature`, `rho1`, `rho2`, `cutoff_radius` | Thermodynamic state and interaction cutoff |
| `[interaction]` | `A_ij_m`, `a_ij_m` (i,j∈{1,2}, m∈{1,2,3}) | 3-Yukawa amplitude and decay-rate matrices |
| `[solver]` | `max_iterations`, `tolerance`, `xi1`, `xi2` | Picard iteration control |
| `[output]` | `output_dir`, `save_every` | Output file settings |

## Roadmap

- [x] Project structure and headers
- [ ] Config file parser
- [ ] Grid initialisation
- [ ] SALR potential evaluation (CPU)
- [ ] DFT solver — Picard iteration (CPU)
- [ ] Validation tests
- [ ] Output & plotting
- [ ] CUDA port of solver kernels
- [ ] CPU vs GPU benchmarking
