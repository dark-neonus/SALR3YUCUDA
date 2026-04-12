# Solver Notes for `solver_cpu.c`

This note summarizes the discrete solver behavior that is implemented in `src/cpu/solver_cpu.c`.

## 1. Grid and State

The solver works on a cell-centred 2D grid with flat row-major arrays:

$$k = i_y N_x + i_x$$

The physical densities are `rho1[k]` and `rho2[k]`.

## 2. Boundary Conditions

The code supports three modes:

- `BC_PBC`: periodic in both directions
- `BC_W2`: hard walls on the left and right edges, periodic in `y`
- `BC_W4`: hard walls on all four edges

For wall modes, the boundary cells are projected to zero after every update:

$$\rho_1 = \rho_2 = 0 \quad \text{on wall cells}$$

That projection is applied to both the physical densities and the Picard candidate fields, so the wall values do not drift during iteration.

## 3. Interaction Fields

For each species pair, the solver builds a lookup table for the truncated Yukawa interaction and evaluates the discrete convolution

$$\Phi_{ij}(\mathbf r) = \Delta x\,\Delta y \sum_{\mathbf r' \neq \mathbf r} \rho_j(\mathbf r')\,U_{ij}(|\mathbf r - \mathbf r'|)$$

The bulk reference fields are computed from the same tables, so the discrete implementation stays consistent with the actual grid and cutoff.

## 4. Euler-Lagrange Update

The candidate density is computed as

$$K_i(\mathbf r) = \rho_{i,b}\exp\big[-\beta(\Phi_{i1}(\mathbf r)+\Phi_{i2}(\mathbf r)-\Phi_{i1,b}-\Phi_{i2,b})\big]$$

Then the code rescales only the physical interior cells so that the total interior mass matches the target bulk density:

$$K_i \leftarrow K_i\,\frac{N_{\mathrm{int}}\rho_{i,b}}{\sum_{\Omega_{\mathrm{int}}} K_i}$$

This is the step that prevents drift of the total density during Picard iteration.

## 5. Picard Mixing

The updated density is

$$\rho_i^{(t+1)} = (1-\xi_i)\rho_i^{(t)} + \xi_i K_i$$

The mixing coefficients `xi1` and `xi2` are taken from the solver configuration.

## 6. Convergence Measure

Convergence is monitored with the RMS difference on the physical cells only:

$$\varepsilon_i = \xi_i\sqrt{\frac{1}{N_{\mathrm{int}}}\sum_{\Omega_{\mathrm{int}}} \left(K_i - \rho_i^{(t)}\right)^2}$$

The run stops when

$$\max(\varepsilon_1, \varepsilon_2) < \varepsilon_{\text{tol}}$$

Wall cells are excluded because they are fixed by the boundary projection and should not contribute to the convergence metric.

## 7. Smoothing

A weak 5-point Laplacian smoother is applied after each Picard step to suppress the grid-scale checkerboard mode:

$$\rho \leftarrow \rho + \epsilon\,(\text{neighbor average} - \text{local value})$$

The smoothing parameter is small enough to damp the Nyquist artefact without changing the large-scale structure appreciably.
