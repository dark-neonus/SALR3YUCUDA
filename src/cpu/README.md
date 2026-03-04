# src/cpu/

CPU (pure C) reference implementations of the physics modules. These serve as the baseline that will be ported to CUDA for GPU acceleration.

---

## potential_cpu.c

Evaluates the 3-Yukawa pair potential:

$$U_{ij}(r) = \sum_{m=1}^{3} A_{ij}^{(m)} \frac{e^{-\alpha_{ij}^{(m)} r}}{r}$$

Returns 0 for $r \le 0$ or $r > r_c$ (cutoff radius).

Principal function: `potential_u(i, j, r, p)` for species pair $(i, j) \in \{0,1\}^2$ at distance $r$.

---

## solver_cpu.c

Picard iteration for the 2-component mean-field DFT equations. The solver computes interaction fields $\Phi_{ij}(\mathbf{r})$ via O(N^2) discrete convolution and updates density profiles according to the Euler-Lagrange operator with chemical-potential renormalisation for mass conservation.

Algorithm outline:

1. Precompute potential tables U11, U12, U22 indexed by circular grid offset (O(N), executed once).
2. For each Picard iteration:
   - Compute interaction fields $\Phi_{ij}$ via 2D convolution (O(N^2)).
   - Evaluate the Euler-Lagrange operator $K_i[\rho]$.
   - Renormalise to conserve mean density.
   - Apply Picard mixing: $\rho_i^{(t+1)} = \xi_i K_i + (1 - \xi_i) \rho_i^{(t)}$.
   - Enforce boundary conditions (periodic, W2, or W4).
   - Apply anti-checkerboard smoothing (5-point Laplacian, epsilon = 0.01).
   - Check convergence via per-point L2 norm.
3. Write convergence log and intermediate snapshots.

The O(N^2) convolution is kept in direct-sum form to provide a clear baseline for comparison with the planned CUDA implementation.

Principal function: `solver_run_binary(rho1, rho2, cfg)` -- returns 0 (converged), 1 (iteration limit), or -1 (allocation error).

See `MATH.md` in this directory for a detailed mathematical derivation mapping each code section to the corresponding equations.

---

## math_utils_cpu.c

Element-wise operations on double-precision flat arrays:

| Function | Operation |
|---|---|
| `vec_add(a, b, c, n)` | $c_i = a_i + b_i$ |
| `vec_scale(a, s, c, n)` | $c_i = s \cdot a_i$ |
| `vec_dot(a, b, n)` | $\sum_i a_i b_i$ |
| `vec_norm(a, n)` | $\sqrt{\sum_i a_i^2}$ |
| `vec_add_scaled(a, s_a, b, s_b, c, n)` | $c_i = s_a \cdot a_i + s_b \cdot b_i$ |

All functions are O(n) and branchless, written to permit auto-vectorisation at `-O2` or higher.
