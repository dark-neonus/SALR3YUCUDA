# src/cpu/

CPU (pure C) implementations of the physics modules.
These are the reference implementations that will later be ported to CUDA.

---

## potential_cpu.c

Evaluates the 3-Yukawa pair potential:

$$U_{ij}(r) = \sum_{m=0}^{2} A_{ij}^{(m)} \frac{e^{-\alpha_{ij}^{(m)} r}}{r}$$

Returns 0 for $r \le 0$ or $r > r_c$ so that the cutoff is respected
without branching inside the solver's inner loop.

**Key function**

`potential_u(i, j, r, p)` \u2014 species pair $(i, j) \in \{0,1\}^2$, distance $r$.

---

## solver_cpu.c

Picard iteration for the 2-component mean-field DFT equation:

$$\rho_i(\mathbf{r}) = \rho_{i,0} \cdot
  \frac{N \exp(-\beta\,\varphi_i(\mathbf{r}))}{\sum_{\mathbf{r}'} \exp(-\beta\,\varphi_i(\mathbf{r}'))}
$$

where the interaction field is the 2-D discrete convolution:

$$\varphi_i[i_x, i_y] = \Delta A \sum_{j}\sum_{j_x, j_y}
  V_{ij}[(i_x - j_x)\bmod n_x,\;(i_y - j_y)\bmod n_y]\,\rho_j[j_x, j_y]$$

**Algorithm overview**

1. **Precompute potential tables** `V00`, `V01`, `V11` \u2014 one call per pair
   ($O(N)$ floating-point operations, done once before the loop).  Because
   the potential only depends on the separation $r$, the table is indexed
   by the circular offset $(d_{ix},\, d_{iy})$ with minimum-image distances.

2. **Picard loop** (up to `max_iterations` steps):
   - `compute_fields` \u2014 fills `phi1` and `phi2` via $O(N^2)$ convolution.
     The inner x-loop is split at `jx` to replace modulo arithmetic with
     two contiguous range operations, enabling auto-vectorisation.
   - `update_density` \u2014 applies the mean-field equation with a min-shift
     for numerical stability, then rescales to conserve the average density.
   - Picard mixing: $\rho \leftarrow (1-\xi)\rho_{\rm old} + \xi\rho_{\rm new}$.
   - Convergence check: per-grid-point L2 norm of $(\rho_{\rm new} - \rho_{\rm old})$.
   - Logs error to `convergence.dat`; saves snapshots every `save_every` steps.

**Performance note**

The $O(N^2)$ convolution is intentionally kept simple to make the
comparison with the CUDA implementation instructive.  For $N = 6400$
(80\xd780 grid) and typical convergence at a few hundred iterations,
runtime is on the order of minutes on a modern desktop.

**Key function**

`solver_run_binary(rho1, rho2, cfg)` \u2014 runs the full solver in-place;
returns 0 (converged), 1 (iteration limit), or -1 (allocation error).

---

## math_utils_cpu.c

Element-wise operations on double-precision flat arrays:

| Function | Operation |
|---|---|
| `vec_add(a, b, c, n)` | $c_i = a_i + b_i$ |
| `vec_scale(a, s, c, n)` | $c_i = s\,a_i$ |
| `vec_dot(a, b, n)` | $\sum_i a_i b_i$ |
| `vec_norm(a, n)` | $\sqrt{\sum_i a_i^2}$ |
| `vec_add_scaled(a, s_a, b, s_b, c, n)` | $c_i = s_a a_i + s_b b_i$ |

All functions are O(n) and branchless; the loops are written to allow
auto-vectorisation with `-O2` or higher.
