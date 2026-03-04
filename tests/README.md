# tests/

Validation and test programs.

| File | Description |
|------|-------------|
| `test_solver.c` | Verifies the DFT solver convergence and output correctness |
| `test_potential.c` | Validates 3-Yukawa potential evaluation against known values |
| `benchmark_integral.c` | CPU integral computation benchmark |
| `benchmark_integral_cuda.cu` | CUDA integral computation benchmark (planned) |

All test binaries are built automatically by CMake and can be executed via `./run.sh tests`.
