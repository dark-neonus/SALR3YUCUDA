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
├── CMakeLists.txt      # CMake build (pure C)
└── Makefile            # Alternative Make build (pure C)
```

## Build & run

Requires: **CMake ≥ 3.12** (or GNU Make) and a **C11 compiler** (gcc).

```bash
# CMake build
mkdir build && cd build
cmake ..
make

# — or — simple Make build
make

# Run
./build/salr_dft configs/default.cfg

# Run tests
./build/test_solver
./build/test_potential
```

> Full instructions: [docs/BUILD.md](docs/BUILD.md)

## Configuration

The simulation is controlled via `.cfg` files. Key sections:

| Section | Parameters | Description |
|---------|-----------|-------------|
| `[grid]` | `nx`, `ny`, `Lx`, `Ly` | Computational grid size and physical domain |
| `[potential]` | `epsilon_attract`, `sigma_attract`, `epsilon_repulse`, `sigma_repulse` | SALR pair potential parameters |
| `[interaction]` | `A11`, `A12`, `A22` | Direct correlation function matrix (coupling coefficients) |
| `[physics]` | `temperature`, `bulk_density` | Thermodynamic state |
| `[solver]` | `max_iterations`, `tolerance`, `mixing_param` | Picard iteration control |
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
