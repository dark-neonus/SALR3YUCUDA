# src/cuda/

CUDA GPU implementations of the DFT solver, vector operations, and potential evaluation.

| File | Description |
|---|---|
| `solver_cuda.cu` | Picard iteration with all GPU kernels (Φ convolution, K operator, mixing, smoothing, reduction) |
| `math_utils_cuda.cu` | `vec_add`, `vec_scale`, `vec_dot`, `vec_norm`, `vec_add_scaled` on device arrays |
| `potential_cuda.cu` | Host + device 3-Yukawa potential U(r) |

The Φ convolution kernel (`k_compute_Phi`) is the main compute bottleneck — O(N²) fully parallelised over N GPU threads with `__ldg()` read-only cache hints for the potential table.
