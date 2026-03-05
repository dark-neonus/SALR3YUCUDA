# src/cuda/

Planned CUDA GPU implementations of the DFT solver, vector operations, and potential evaluation. The files in this directory are intended to mirror the CPU interfaces defined in `src/cpu/` and `include/`.

| File | Purpose |
|---|---|
| `solver_cuda.cu` | GPU-accelerated Picard iteration (convolution, Euler-Lagrange operator, mixing, reduction) |
| `math_utils_cuda.cu` | Device-side vector arithmetic (`vec_add`, `vec_scale`, `vec_dot`, `vec_norm`, `vec_add_scaled`) |
| `potential_cuda.cu` | Host and device 3-Yukawa potential evaluation |

The primary optimisation target is the O(N^2) interaction-field convolution, which is the dominant computational cost in the CPU solver and is well suited to massive parallelisation on GPU hardware.

Status: not yet implemented. The CPU reference implementation in `src/cpu/` is the current working solver.
