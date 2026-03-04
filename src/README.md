# src/

Main source code directory.

| Path | Description |
|------|-------------|
| `main.c` | Program entry point: parses configuration, initialises the density fields, invokes the solver, and writes output |
| `core/` | Platform-independent infrastructure (configuration parsing, grid creation) |
| `cpu/` | CPU reference implementations of the solver, potential evaluation, and vector math |
| `cuda/` | CUDA GPU implementations (planned; will mirror `cpu/` interfaces) |
| `math/` | Specialised math routines (planned) |
| `utils/` | File I/O utilities |
