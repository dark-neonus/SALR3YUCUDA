/*
 * solver_cpu.c — Iterative DFT solver for a 2-component SALR mixture (CPU)
 *
 * Variable naming matches the slide notation directly so every block of code
 * can be traced back to a specific formula on the slides.
 *
 * ── Pair potential (slide "Triple Yukawa pair potential") ────────────────
 *
 *   U_ij(r) = sum_{m=1}^{3}  A_ij^(m) * exp(-alpha_ij^(m) * r) / r
 *
 *   Tables:  U11[diy*Nx+dix] = U_{11}(r),  U12 = U_{12}(r),  U22 = U_{22}(r)
 *   (species indices are 1-based in names; 0-based internally in potential_u)
 *
 * ── Interaction fields (slide "Euler-Lagrange equations", Phi_ij defs) ───
 *
 *   Phi11(r) = dA * sum_{r'} rho1(r') * U11(|r-r'|)   [rho1 conv U11]
 *   Phi12(r) = dA * sum_{r'} rho2(r') * U12(|r-r'|)   [rho2 conv U12]
 *   Phi21(r) = dA * sum_{r'} rho1(r') * U12(|r-r'|)   [rho1 conv U12]
 *   Phi22(r) = dA * sum_{r'} rho2(r') * U22(|r-r'|)   [rho2 conv U22]
 *
 * ── Bulk fields (slide "Euler-Lagrange equations", Phi_ij,b defs) ────────
 *
 *   Phi11b = dA * rho1_b * sum_{r'} U11(r')
 *   Phi12b = dA * rho2_b * sum_{r'} U12(r')
 *   Phi21b = dA * rho1_b * sum_{r'} U12(r')
 *   Phi22b = dA * rho2_b * sum_{r'} U22(r')
 *
 * ── Euler-Lagrange equations (slide "Euler-Lagrange equations") ──────────
 *
 *   K1(r) = rho1_b * exp( -beta*(Phi11(r)+Phi12(r) - Phi11b-Phi12b) )
 *   K2(r) = rho2_b * exp( -beta*(Phi21(r)+Phi22(r) - Phi21b-Phi22b) )
 *
 * ── Picard iteration (slide "Numerical computation method") ──────────────
 *
 *   rho1^(t+1)(r) = xi1 * K1(r) + (1-xi1) * rho1^(t)(r)
 *   rho2^(t+1)(r) = xi2 * K2(r) + (1-xi2) * rho2^(t)(r)
 *
 * ── Convergence (slide "Numerical computation method") ───────────────────
 *
 *   ||rho_i^(t+1) - rho_i^(t)|| = xi_i * ||K_i - rho_i^(t)|| < epsilon
 *
 * ── Boundary modes (slide "Spatial density distributions") ──────────────
 *   BC_PBC — periodic both axes  (PBC)
 *   BC_W2  — walls at x=0, x=Lx (W2: "two parallel walls")
 *   BC_W4  — walls on all sides  (W4: "square box")
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "../../include/solver.h"
#include "../../include/config.h"
#include "../../include/potential.h"
#include "../../include/math_utils.h"
#include "../../include/io.h"

/* ── Potential table precomputation ─────────────────────────────────────── */

/*
 * build_potential_table - Fill table[Ny*Nx] with U_ij(r) for every grid
 * offset (dix, diy).  Corresponds to slide "Triple Yukawa pair potential":
 *
 *   U_ij(r) = sum_{m=1}^{3} A_ij^(m) * exp(-alpha_ij^(m)*r) / r
 *
 * Called three times to produce tables U11, U12, U22 (species indices are
 * 1-based in the table names; 0-based internally: si=0→species1, si=1→species2).
 *
 * wall_x / wall_y: when 1, use actual displacement in that axis instead of
 * the minimum-image convention.  Required for W2/W4 boundary modes so that
 * the Yukawa tail does not wrap around the hard-wall edge.
 *
 * Slide "Two-dimensional case": r = |r_ij - r_mn| = sqrt(dx^2 + dy^2),
 * summed only for r <= r_c (cutoff enforced inside potential_u).
 */
static void build_potential_table(double *table, int si, int sj,
                                  const PotentialParams *p,
                                  int nx, int ny, double dx, double dy,
                                  int wall_x, int wall_y) {
    for (int diy = 0; diy < ny; ++diy) {
        /* Slide "Two-dimensional case": minimum-image or actual displacement */
        double dry = wall_y ? diy * dy
                            : ((diy <= ny / 2) ? diy * dy : (diy - ny) * dy);
        for (int dix = 0; dix < nx; ++dix) {
            double drx = wall_x ? dix * dx
                                : ((dix <= nx / 2) ? dix * dx : (dix - nx) * dx);
            double r   = sqrt(drx * drx + dry * dry);  /* |r_ij - r_mn| */
            table[diy * nx + dix] = potential_u(si, sj, r, p);
        }
    }
}

/* ── 2-D discrete convolution → Phi_ij fields ───────────────────────────── */

/*
 * compute_Phi - Compute all four Phi_ij fields for one Picard iteration.
 *
 * Slide "Two-dimensional case" (Phi_ij definitions):
 *
 *   Phi11(xi,yj) = dA * sum_{r'} rho1(r') * U11(|r_ij - r'|)
 *   Phi12(xi,yj) = dA * sum_{r'} rho2(r') * U12(|r_ij - r'|)
 *   Phi21(xi,yj) = dA * sum_{r'} rho1(r') * U12(|r_ij - r'|)
 *   Phi22(xi,yj) = dA * sum_{r'} rho2(r') * U22(|r_ij - r'|)
 *
 * All four fields are accumulated in a single pass over source cells (r') to
 * minimise memory traffic.  The inner x-loop is split into two contiguous
 * ranges to eliminate the modulo operation in the hot path:
 *
 *   For source column jx:
 *     ix in [jx, nx-1]  =>  tab_offset k = ix - jx        (k = 0 .. nx-1-jx)
 *     ix in [0,  jx-1]  =>  tab_offset k = nx - jx + ix   (k = nx-jx .. nx-1)
 *
 * dA is applied once to all four arrays after the full O(N^2) sum.
 * Only 3 tables are needed because U12 is shared by Phi12 and Phi21
 * (U_{12} == U_{21} by symmetry of the pair potential).
 */
static void compute_Phi(
    double       *Phi11,  double       *Phi12,
    double       *Phi21,  double       *Phi22,
    const double *rho1,   const double *rho2,
    const double *U11,    const double *U12,   const double *U22,
    int nx, int ny, double dA)
{
    size_t Ntot = (size_t)(nx * ny);
    memset(Phi11, 0, Ntot * sizeof(double));
    memset(Phi12, 0, Ntot * sizeof(double));
    memset(Phi21, 0, Ntot * sizeof(double));
    memset(Phi22, 0, Ntot * sizeof(double));

    for (int jy = 0; jy < ny; ++jy) {
        for (int jx = 0; jx < nx; ++jx) {
            /* Source densities at r' = (jx, jy) */
            double s1 = rho1[jy * nx + jx];  /* rho1(r') */
            double s2 = rho2[jy * nx + jx];  /* rho2(r') */

            for (int iy = 0; iy < ny; ++iy) {
                int diy = (iy - jy + ny) % ny;
                const double *tU11 = U11 + (size_t)diy * nx;
                const double *tU12 = U12 + (size_t)diy * nx;
                const double *tU22 = U22 + (size_t)diy * nx;
                double *p11 = Phi11 + (size_t)iy * nx;
                double *p12 = Phi12 + (size_t)iy * nx;
                double *p21 = Phi21 + (size_t)iy * nx;
                double *p22 = Phi22 + (size_t)iy * nx;

                /* ix in [jx, nx-1]: table index k = 0 .. nx-1-jx */
                int len1 = nx - jx;
                for (int k = 0; k < len1; ++k) {
                    p11[jx + k] += s1 * tU11[k];  /* Phi11 += rho1*U11 */
                    p12[jx + k] += s2 * tU12[k];  /* Phi12 += rho2*U12 */
                    p21[jx + k] += s1 * tU12[k];  /* Phi21 += rho1*U12 */
                    p22[jx + k] += s2 * tU22[k];  /* Phi22 += rho2*U22 */
                }
                /* ix in [0, jx-1]: table index k = nx-jx .. nx-1 */
                int off = nx - jx;
                for (int k = 0; k < jx; ++k) {
                    p11[k] += s1 * tU11[off + k];
                    p12[k] += s2 * tU12[off + k];
                    p21[k] += s1 * tU12[off + k];
                    p22[k] += s2 * tU22[off + k];
                }
            }
        }
    }

    /* Apply dA once — equivalent to the dA prefactor in the slide integral */
    for (size_t k = 0; k < Ntot; ++k) {
        Phi11[k] *= dA;  Phi12[k] *= dA;
        Phi21[k] *= dA;  Phi22[k] *= dA;
    }
}

/* ── Euler-Lagrange operator K_i ─────────────────────────────────────────── */

/*
 * compute_K - Apply the mean-field Euler-Lagrange equation for one species.
 *
 * Slide "Euler-Lagrange equations" / "Numerical computation method":
 *
 *   K_i[rho_i^(t)](r) = rho_{i,b} * exp( -beta*(Phi_ia(r)+Phi_ib(r)
 *                                               - Phi_iab - Phi_ibb) )
 *
 * where Phi_ia and Phi_ib are the two interaction field components for
 * species i (e.g. Phi11+Phi12 for i=1, Phi21+Phi22 for i=2), and
 * Phi_iab, Phi_ibb are their corresponding bulk scalar values.
 *
 * rho0b  — bulk density rho_{i,b} (prefactor, ensures K_i → rho_{i,b} in bulk)
 * beta   — inverse temperature 1/T
 */
static void compute_K(const double *Phi_a,  const double *Phi_b,
                      double Phi_ab,        double Phi_bb,
                      double rho0b,         double beta,
                      double *K,            size_t N) {
    for (size_t k = 0; k < N; ++k)
        K[k] = rho0b * exp(-beta * (Phi_a[k] + Phi_b[k] - Phi_ab - Phi_bb));
}

/* ── Boundary mask ───────────────────────────────────────────────────────── */

/*
 * apply_boundary_mask - Zero density at wall nodes.
 *
 * Slide "Spatial density distributions" (boundary conditions):
 *   PBC: rho(xi ± Lx, yj ± Ly) = rho(xi, yj)  — no wall nodes, no-op.
 *   W2:  rho(xi, yj) = 0  for xi < 0 or xi >= Lx  (ix=0, ix=Nx-1)
 *   W4:  additionally rho(xi, yj) = 0  for yj < 0 or yj >= Ly (iy=0, iy=Ny-1)
 */
static void apply_boundary_mask(double *rho1, double *rho2,
                                 int nx, int ny, int mode) {
    if (mode == BC_PBC) return;

    /* W2 and W4: hard walls at x=0 (ix=0) and x=Lx (ix=Nx-1) */
    for (int iy = 0; iy < ny; ++iy) {
        rho1[iy * nx + 0]        = 0.0;
        rho1[iy * nx + (nx - 1)] = 0.0;
        rho2[iy * nx + 0]        = 0.0;
        rho2[iy * nx + (nx - 1)] = 0.0;
    }

    /* W4 only: hard walls at y=0 (iy=0) and y=Ly (iy=Ny-1) */
    if (mode == BC_W4) {
        for (int ix = 0; ix < nx; ++ix) {
            rho1[0 * nx + ix]        = 0.0;
            rho1[(ny - 1) * nx + ix] = 0.0;
            rho2[0 * nx + ix]        = 0.0;
            rho2[(ny - 1) * nx + ix] = 0.0;
        }
    }
}

/* ── Anti-checkerboard density smoothing ────────────────────────────────── */

/*
 * smooth_density - Apply one step of 5-point Laplacian smoothing to kill
 * the grid-Nyquist (checkerboard) mode that arises from discrete convolution.
 *
 * Root cause: U_ij(r) diverges as 1/r, so U_ij(dx) >> U_ij(2*dx).  The four
 * face-adjacent neighbours have the largest potential weight.  On a square grid
 * their combined Fourier component at the Nyquist wavevector k=(π/dx, π/dx) is
 *
 *   Ũ_cb = dA * Σ_{r≠0} U(r) · cos(π(ix+iy))  <  0
 *
 * For U11 at the working parameters:  Ũ_cb ≈ −12.22, beta·|Ũ_cb|·rho1_b ≈ 0.815.
 * The linearised Picard contraction factor for the checkerboard mode is
 * 1 − xi·(1 − 0.815) ≈ 0.963 per iteration.  In the nonlinear regime the
 * physical SALR modulation couples back into this mode and re-excites it,
 * preventing it from converging to zero in finite iterations.
 *
 * The 5-point smoothing step has eigenvalue
 *   s(k) = (1−4ε) + 4ε·cos(k_x·dx)·cos(k_y·dx)
 * At the Nyquist:  s = 1 − 8ε = 0.92  (with ε=0.01)
 * At physical SALR wavelength λ ≈ 4 (k = 2π/4):  s ≈ 0.999997
 * → The smoothing eliminates checkerboard with negligible effect on physics.
 *
 * For non-PBC modes only the interior is smoothed; wall nodes are re-zeroed
 * by apply_boundary_mask immediately after this call.
 */
#define SMOOTH_EPS 0.01   /* Laplacian smoothing strength; keep well below 0.125 */

static void smooth_density(double *rho, int nx, int ny)
{
    /* Use a second buffer to avoid in-place aliasing */
    double *tmp = malloc((size_t)(nx * ny) * sizeof(double));
    if (!tmp) return;   /* silently skip if allocation fails */

    for (int iy = 0; iy < ny; ++iy) {
        int ym = (iy - 1 + ny) % ny;
        int yp = (iy + 1)      % ny;
        for (int ix = 0; ix < nx; ++ix) {
            int xm = (ix - 1 + nx) % nx;
            int xp = (ix + 1)      % nx;
            tmp[iy*nx + ix] =
                (1.0 - 4.0*SMOOTH_EPS) * rho[iy*nx + ix]
                + SMOOTH_EPS * (rho[iy*nx + xm] + rho[iy*nx + xp]
                              + rho[ym*nx + ix] + rho[yp*nx + ix]);
        }
    }
    memcpy(rho, tmp, (size_t)(nx * ny) * sizeof(double));
    free(tmp);
}

/* ── L2 difference ───────────────────────────────────────────────────────── */

double solver_l2_diff(const double *a, const double *b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum / (double)n);   /* normalised to be grid-size independent */
}

/* ── Output helpers ──────────────────────────────────────────────────────── */

/*
 * save_snapshot - Write both density profiles for a given iteration to disk.
 * Filenames: <output_dir>/density_species{1,2}_iter_<NNNNNN>.dat
 */
static void save_snapshot(const double *rho1, const double *rho2,
                           const double *xs,   const double *ys,
                           int nx, int ny, int iter,
                           const char *output_dir) {
    char path[512];
    snprintf(path, sizeof(path),
             "%s/density_species1_iter_%06d.dat", output_dir, iter);
    io_save_density_2d(path, xs, ys, rho1, (size_t)nx, (size_t)ny);

    snprintf(path, sizeof(path),
             "%s/density_species2_iter_%06d.dat", output_dir, iter);
    io_save_density_2d(path, xs, ys, rho2, (size_t)nx, (size_t)ny);
}

/* ── Main solver entry point ─────────────────────────────────────────────── */

int solver_run_binary(double *rho1, double *rho2, struct SimConfig *cfg) {
    /*
     * Slide "Parameters used in solving the problem":
     *   Nx = Lx/dx,  Ny = Ly/dy  (grid dimensions)
     *   dA = dx*dy               (cell area, slide "Two-dimensional case")
     *   beta = 1/T               (slide "Grand thermodynamic potential")
     */
    const int    Nx   = cfg->grid.nx;         /* number of grid cells in x  */
    const int    Ny   = cfg->grid.ny;         /* number of grid cells in y  */
    const double dx   = cfg->grid.dx;         /* Delta_x                    */
    const double dy   = cfg->grid.dy;         /* Delta_y                    */
    const size_t N    = (size_t)(Nx * Ny);    /* total grid points Nx*Ny    */
    const double dA   = dx * dy;              /* cell area Delta_x*Delta_y  */
    const double beta = 1.0 / cfg->temperature; /* beta = 1/T               */
    const int    mode = cfg->boundary_mode;

    /* wall_x/wall_y: actual (non-minimum-image) distances for walled axes */
    const int wall_x = (mode == BC_W2 || mode == BC_W4);
    const int wall_y = (mode == BC_W4);

    /*
     * Potential tables — slide "Triple Yukawa pair potential":
     *   U11[diy*Nx+dix] = U_{11}(r),  species pair (1,1)  si=0,sj=0
     *   U12[diy*Nx+dix] = U_{12}(r),  species pair (1,2)  si=0,sj=1
     *   U22[diy*Nx+dix] = U_{22}(r),  species pair (2,2)  si=1,sj=1
     */
    double *U11   = malloc(N * sizeof(double));
    double *U12   = malloc(N * sizeof(double));
    double *U22   = malloc(N * sizeof(double));

    /*
     * Interaction field arrays — slide "Euler-Lagrange equations":
     *   Phi11(r), Phi12(r), Phi21(r), Phi22(r)
     */
    double *Phi11 = malloc(N * sizeof(double));
    double *Phi12 = malloc(N * sizeof(double));
    double *Phi21 = malloc(N * sizeof(double));
    double *Phi22 = malloc(N * sizeof(double));

    /*
     * K1, K2 — results of the Euler-Lagrange operator K_i
     * (slide "Numerical computation method"):
     *   K1(r) = rho1_b * exp(-beta*(Phi11+Phi12 - Phi11b-Phi12b))
     *   K2(r) = rho2_b * exp(-beta*(Phi21+Phi22 - Phi21b-Phi22b))
     */
    double *K1    = malloc(N * sizeof(double));
    double *K2    = malloc(N * sizeof(double));

    /* Cell-centre coordinate arrays for output (not part of physics) */
    double *xs    = malloc((size_t)Nx * sizeof(double));
    double *ys    = malloc((size_t)Ny * sizeof(double));

    if (!U11 || !U12 || !U22 ||
        !Phi11 || !Phi12 || !Phi21 || !Phi22 ||
        !K1 || !K2 || !xs || !ys) {
        free(U11); free(U12); free(U22);
        free(Phi11); free(Phi12); free(Phi21); free(Phi22);
        free(K1); free(K2); free(xs); free(ys);
        return -1;
    }

    /* Build potential tables once — O(N) evaluations of U_ij(r) */
    build_potential_table(U11, 0, 0, &cfg->potential, Nx, Ny, dx, dy, wall_x, wall_y);
    build_potential_table(U12, 0, 1, &cfg->potential, Nx, Ny, dx, dy, wall_x, wall_y);
    build_potential_table(U22, 1, 1, &cfg->potential, Nx, Ny, dx, dy, wall_x, wall_y);

    /*
     * Bulk interaction fields — slide "Euler-Lagrange equations" (Phi_ij,b):
     *
     *   Phi11b = dA * rho1_b * sum_{r'} U11(r')
     *   Phi12b = dA * rho2_b * sum_{r'} U12(r')
     *   Phi21b = dA * rho1_b * sum_{r'} U12(r')   (same sum_U12, diff. rho_b)
     *   Phi22b = dA * rho2_b * sum_{r'} U22(r')
     *
     * These scalars fix the chemical potential so rho_i → rho_{i,b} in bulk.
     * cfg->rho1 = rho_{1,b},  cfg->rho2 = rho_{2,b}.
     */
    double sum_U11 = 0.0, sum_U12 = 0.0, sum_U22 = 0.0;
    for (size_t k = 0; k < N; ++k) {
        sum_U11 += U11[k];
        sum_U12 += U12[k];
        sum_U22 += U22[k];
    }
    const double rho1_b = cfg->rho1;  /* rho_{1,b}: bulk density species 1 */
    const double rho2_b = cfg->rho2;  /* rho_{2,b}: bulk density species 2 */

    const double Phi11b = dA * rho1_b * sum_U11;
    const double Phi12b = dA * rho2_b * sum_U12;
    const double Phi21b = dA * rho1_b * sum_U12;
    const double Phi22b = dA * rho2_b * sum_U22;

    /* Cell-centre coordinates xi = (i+0.5)*dx, yj = (j+0.5)*dy */
    for (int i = 0; i < Nx; ++i) xs[i] = (i + 0.5) * dx;
    for (int j = 0; j < Ny; ++j) ys[j] = (j + 0.5) * dy;

    /* Convergence log: create fresh file with header */
    char log_path[512];
    snprintf(log_path, sizeof(log_path), "%s/convergence.dat", cfg->output_dir);
    {
        FILE *lf = fopen(log_path, "w");
        if (lf) { fprintf(lf, "# iter  L2_error\n"); fclose(lf); }
    }

    /* Slide "Numerical computation method": xi1, xi2 — Picard mixing params */
    const int    max_iter = cfg->solver.max_iterations;
    const double tol      = cfg->solver.tolerance;  /* epsilon */
    const double xi1      = cfg->solver.xi1;        /* xi_1 */
    const double xi2      = cfg->solver.xi2;        /* xi_2 */
    const int    save_ev  = cfg->save_every;

    /* Enforce initial boundary mask (slide "Spatial density distributions") */
    apply_boundary_mask(rho1, rho2, Nx, Ny, mode);

    int converged = 0;

    for (int iter = 0; iter < max_iter; ++iter) {

        /*
         * --- Step 1: Compute Phi_ij fields (slide "Two-dimensional case") ---
         *   Phi11(r) = dA*sum_{r'} rho1(r')*U11(|r-r'|)
         *   Phi12(r) = dA*sum_{r'} rho2(r')*U12(|r-r'|)
         *   Phi21(r) = dA*sum_{r'} rho1(r')*U12(|r-r'|)
         *   Phi22(r) = dA*sum_{r'} rho2(r')*U22(|r-r'|)
         */
        compute_Phi(Phi11, Phi12, Phi21, Phi22,
                    rho1, rho2, U11, U12, U22,
                    Nx, Ny, dA);

        /*
         * --- Step 2: Apply Euler-Lagrange operator K_i ---
         * Slide "Numerical computation method":
         *   K1(r) = rho1_b * exp(-beta*(Phi11(r)+Phi12(r) - Phi11b-Phi12b))
         *   K2(r) = rho2_b * exp(-beta*(Phi21(r)+Phi22(r) - Phi21b-Phi22b))
         */
        compute_K(Phi11, Phi12, Phi11b, Phi12b, rho1_b, beta, K1, N);
        compute_K(Phi21, Phi22, Phi21b, Phi22b, rho2_b, beta, K2, N);

        /*
         * --- Step 2b: Renormalise K_i to enforce global mass conservation ---
         *
         * The EL operator K_i = rho_{i,b} * exp(-beta*(Phi_i - Phi_i^b)) fixes
         * the chemical potential using the BULK reference value Phi_i^b.  Once
         * the density field becomes inhomogeneous this is only an approximation:
         * by Jensen's inequality  <exp(x)> >= exp(<x>), so
         *
         *   (1/N) * sum_r K_i(r)  >=  rho_{i,b}                             (*)
         *
         * Every iteration the total mass therefore drifts upward, the
         * exponential amplifies the inhomogeneity, and the field blows up
         * (observed as near-zero density almost everywhere plus isolated spikes).
         *
         * The thermodynamically correct fix is to determine the chemical
         * potential mu_i dynamically each iteration so that the density
         * constraint  (1/N)*sum rho_i = rho_{i,b}  is satisfied exactly.
         * In practice this means rescaling K_i after the exponential:
         *
         *   K_i(r) → K_i(r) * ( N * rho_{i,b} / sum_r K_i(r) )
         *
         * which is equivalent to:
         *   mu_i = (1/beta) * log( rho_{i,b} / <exp(-beta*Phi_i)> )
         *
         * In the bulk (uniform rho) the rescaling factor equals 1 and the two
         * formulations are identical.  Away from the bulk the dynamic mu_i
         * keeps the total mass pinned at its physical value and prevents the
         * exponential blow-up described above.
         */
        {
            double s1 = 0.0, s2 = 0.0;
            for (size_t k = 0; k < N; ++k) { s1 += K1[k]; s2 += K2[k]; }
            double norm1 = ((double)N * rho1_b) / s1;
            double norm2 = ((double)N * rho2_b) / s2;
            for (size_t k = 0; k < N; ++k) { K1[k] *= norm1; K2[k] *= norm2; }
        }

        /*
         * --- Step 3: Convergence check (slide "Numerical computation method")
         * Must be done BEFORE mixing while rho_i^(t) is still in rho1/rho2:
         *   ||rho_i^(t+1) - rho_i^(t)|| = xi_i * ||K_i - rho_i^(t)||  < epsilon
         */
        double err = xi1 * solver_l2_diff(K1, rho1, N);
        double e2  = xi2 * solver_l2_diff(K2, rho2, N);
        if (e2 > err) err = e2;

        /*
         * --- Step 4: Picard mixing (slide "Numerical computation method") ---
         *   rho1^(t+1)(r) = xi1 * K1(r)  +  (1-xi1) * rho1^(t)(r)
         *   rho2^(t+1)(r) = xi2 * K2(r)  +  (1-xi2) * rho2^(t)(r)
         */
        vec_add_scaled(rho1, 1.0 - xi1, K1, xi1, rho1, N);
        vec_add_scaled(rho2, 1.0 - xi2, K2, xi2, rho2, N);

        /* --- Step 5: Enforce boundary conditions on mixed density --- */
        apply_boundary_mask(rho1, rho2, Nx, Ny, mode);

        /*
         * --- Step 5b: Anti-checkerboard smoothing ---
         * One step of 5-point Laplacian (strength SMOOTH_EPS = 0.01) per
         * iteration damps the grid-Nyquist mode factor ~0.92 per step while
         * leaving physical SALR wavelengths (λ ≫ dx) essentially unchanged.
         * See smooth_density() doc-comment for the full derivation.
         */
        smooth_density(rho1, Nx, Ny);
        smooth_density(rho2, Nx, Ny);
        /* Re-enforce walls after smoothing (wrapping from periodic BCs is
         * harmless for PBC mode; needed for W2/W4 to keep wall nodes at 0) */
        apply_boundary_mask(rho1, rho2, Nx, Ny, mode);

        /* --- Step 6: log and optional snapshot --- */

        io_log_convergence(log_path, iter, err);

        if ((iter + 1) % save_ev == 0) {
            printf("  iter %6d   err = %.6e\n", iter + 1, err);
            save_snapshot(rho1, rho2, xs, ys, Nx, Ny, iter + 1, cfg->output_dir);
        }

        if (err < tol) {
            printf("  Converged at iteration %d  (err = %.3e < %.3e)\n",
                   iter + 1, err, tol);
            save_snapshot(rho1, rho2, xs, ys, Nx, Ny, iter + 1, cfg->output_dir);
            converged = 1;
            break;
        }
    }

    if (!converged)
        fprintf(stderr,
                "Warning: solver did not converge in %d iterations.\n",
                max_iter);

    free(U11); free(U12); free(U22);
    free(Phi11); free(Phi12); free(Phi21); free(Phi22);
    free(K1); free(K2);
    free(xs); free(ys);

    return converged ? 0 : 1;
}
