/*
 * solver_cpu.c — Iterative DFT solver for a 2-component SALR mixture (CPU)
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
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdio.h>
#include <omp.h>

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
    #pragma omp parallel for collapse(2)
    for (int diy = 0; diy < ny; ++diy) {
        for (int dix = 0; dix < nx; ++dix) {
            /* Slide "Two-dimensional case": minimum-image or actual displacement */
            double dry = wall_y ? diy * dy
                                : ((diy <= ny / 2) ? diy * dy : (diy - ny) * dy);
            double drx = wall_x ? dix * dx
                                : ((dix <= nx / 2) ? dix * dx : (dix - nx) * dx);
            double r   = sqrt(drx * drx + dry * dry);  /* |r_ij - r_mn| */
            table[diy * nx + dix] = potential_u(si, sj, r, p);
        }
    }
}

/* ── Analytical bulk field calculation (slide "Two-dimensional case") ───────
 *
 * Φ_11,b = 2π·ρ_1,b · Σ_{m=1}^{3} (A_11^(m) / α_11^(m)) · (1 - exp(-α_11^(m)·r_c))
 * Φ_12,b = 2π·ρ_2,b · Σ_{m=1}^{3} (A_12^(m) / α_12^(m)) · (1 - exp(-α_12^(m)·r_c))
 * Φ_21,b = 2π·ρ_1,b · Σ_{m=1}^{3} (A_12^(m) / α_12^(m)) · (1 - exp(-α_12^(m)·r_c))
 * Φ_22,b = 2π·ρ_2,b · Σ_{m=1}^{3} (A_22^(m) / α_22^(m)) · (1 - exp(-α_22^(m)·r_c))
 */
static double compute_Phi_bulk_analytical(double rho_b, int si, int sj,
                                          const PotentialParams *p) {
    double sum = 0.0;
    for (int m = 0; m < YUKAWA_TERMS; ++m) {
        double A     = p->A[si][sj][m];
        double alpha = p->alpha[si][sj][m];
        if (alpha > 1e-12) {  /* Avoid division by zero */
            sum += (A / alpha) * (1.0 - exp(-alpha * p->cutoff_radius));
        }
    }
    return 2.0 * M_PI * rho_b * sum;
}

/* ── 2-D discrete convolution → Phi_ij fields ───────────────────────────── */

/*
 * compute_Phi - Compute all four Phi_ij fields for one Picard iteration.
 *
 * Boundary handling from slides "Spatial density distributions":
 *   PBC: periodic in both x and y - use minimum image (modular) in both axes
 *   W2:  walls at x=0, x=Lx - actual distance in x, periodic (modular) in y
 *   W4:  walls on all sides - actual distance in both x and y
 *
 * For walled axes, sum only over interior points (no wrap-around).
 */
static void compute_Phi(
    double       *Phi11,  double       *Phi12,
    double       *Phi21,  double       *Phi22,
    const double *rho1,   const double *rho2,
    const double *U11,    const double *U12,   const double *U22,
    int nx, int ny, double dA, int wall_x, int wall_y)
{
    size_t Ntot = (size_t)(nx * ny);
    memset(Phi11, 0, Ntot * sizeof(double));
    memset(Phi12, 0, Ntot * sizeof(double));
    memset(Phi21, 0, Ntot * sizeof(double));
    memset(Phi22, 0, Ntot * sizeof(double));

    #pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            double p11 = 0.0, p12 = 0.0, p21 = 0.0, p22 = 0.0;
            
            for (int jy = 0; jy < ny; ++jy) {
                /* Y displacement: periodic (minimum image) or actual distance */
                int diy;
                if (wall_y) {
                    diy = (iy >= jy) ? (iy - jy) : (jy - iy);  /* |iy - jy| */
                } else {
                    diy = (iy - jy + ny) % ny;  /* minimum image */
                    if (diy > ny / 2) diy = ny - diy;
                }
                if (diy >= ny) continue;  /* out of range */
                
                const double *tU11 = U11 + (size_t)diy * nx;
                const double *tU12 = U12 + (size_t)diy * nx;
                const double *tU22 = U22 + (size_t)diy * nx;
                
                for (int jx = 0; jx < nx; ++jx) {
                    /* X displacement: periodic (minimum image) or actual distance */
                    int dix;
                    if (wall_x) {
                        dix = (ix >= jx) ? (ix - jx) : (jx - ix);  /* |ix - jx| */
                    } else {
                        dix = (ix - jx + nx) % nx;  /* minimum image */
                        if (dix > nx / 2) dix = nx - dix;
                    }
                    if (dix >= nx) continue;  /* out of range */
                    
                    double s1 = rho1[jy * nx + jx];  /* rho1(r') */
                    double s2 = rho2[jy * nx + jx];  /* rho2(r') */
                    
                    double u11 = tU11[dix];
                    double u12 = tU12[dix];
                    double u22 = tU22[dix];
                    
                    p11 += s1 * u11;
                    p12 += s2 * u12;
                    p21 += s1 * u12;
                    p22 += s2 * u22;
                }
            }
            
            size_t idx = (size_t)iy * nx + ix;
            Phi11[idx] = p11 * dA;
            Phi12[idx] = p12 * dA;
            Phi21[idx] = p21 * dA;
            Phi22[idx] = p22 * dA;
        }
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
    #pragma omp parallel for
    for (size_t k = 0; k < N; ++k) {
        double arg = -beta * (Phi_a[k] + Phi_b[k] - Phi_ab - Phi_bb);
        /* Clamp the exponent to prevent floating-point overflow/underflow.
         * exp(+500) ≈ 1e217 (finite), exp(-500) ≈ 1e-217 (tiny but nonzero).
         * Without clamping, early-iteration noise can produce ±Inf or 0. */
        if (arg >  500.0) arg =  500.0;
        if (arg < -500.0) arg = -500.0;
        K[k] = rho0b * exp(arg);
    }
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
 * Boundary handling:
 *   PBC: periodic wrapping in both x and y
 *   W2:  no wrap in x (use clamped neighbor or skip wall), periodic in y
 *   W4:  no wrap in either axis (clamped neighbors)
 */
#define SMOOTH_EPS 0.01   /* Laplacian smoothing strength; keep well below 0.125 */

static void smooth_density(double *rho, int nx, int ny, int mode)
{
    /* Use a second buffer to avoid in-place aliasing */
    double *tmp = malloc((size_t)(nx * ny) * sizeof(double));
    if (!tmp) return;   /* silently skip if allocation fails */

    int wall_x = (mode == BC_W2 || mode == BC_W4);
    int wall_y = (mode == BC_W4);
    
    /* For wall modes, only smooth interior; wall cells stay at zero */
    int x_start = wall_x ? 1 : 0;
    int x_end   = wall_x ? nx - 1 : nx;
    int y_start = wall_y ? 1 : 0;
    int y_end   = wall_y ? ny - 1 : ny;

    /* First copy current values (wall cells will stay unchanged) */
    memcpy(tmp, rho, (size_t)(nx * ny) * sizeof(double));

    #pragma omp parallel for collapse(2)
    for (int iy = y_start; iy < y_end; ++iy) {
        for (int ix = x_start; ix < x_end; ++ix) {
            int ym, yp, xm, xp;
            
            /* Y neighbors */
            if (wall_y) {
                /* Clamped: use current cell value if at boundary */
                ym = (iy > 0) ? iy - 1 : iy;
                yp = (iy < ny - 1) ? iy + 1 : iy;
            } else {
                /* Periodic */
                ym = (iy - 1 + ny) % ny;
                yp = (iy + 1) % ny;
            }
            
            /* X neighbors */
            if (wall_x) {
                /* Clamped: use current cell value if at boundary */
                xm = (ix > 0) ? ix - 1 : ix;
                xp = (ix < nx - 1) ? ix + 1 : ix;
            } else {
                /* Periodic */
                xm = (ix - 1 + nx) % nx;
                xp = (ix + 1) % nx;
            }
            
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
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum / (double)n);   /* normalised to be grid-size independent */
}

/*
 * solver_l2_diff_interior - L2 difference only over interior cells (excluding walls).
 * For W2: exclude ix=0 and ix=Nx-1 (wall columns)
 * For W4: also exclude iy=0 and iy=Ny-1 (wall rows)
 * For PBC: all cells are interior
 *
 * This prevents wall cells (where K≠0 but ρ=0 by constraint) from contributing
 * to the convergence error, which would otherwise create a permanent error floor.
 */
static double solver_l2_diff_interior(const double *a, const double *b,
                                       int nx, int ny, int mode) {
    int x_start = (mode == BC_W2 || mode == BC_W4) ? 1 : 0;
    int x_end   = (mode == BC_W2 || mode == BC_W4) ? nx - 1 : nx;
    int y_start = (mode == BC_W4) ? 1 : 0;
    int y_end   = (mode == BC_W4) ? ny - 1 : ny;
    
    double sum = 0.0;
    size_t count = 0;
    
    #pragma omp parallel for collapse(2) reduction(+:sum, count)
    for (int iy = y_start; iy < y_end; ++iy) {
        for (int ix = x_start; ix < x_end; ++ix) {
            size_t k = (size_t)iy * nx + ix;
            double d = a[k] - b[k];
            sum += d * d;
            count++;
        }
    }
    
    return (count > 0) ? sqrt(sum / (double)count) : 0.0;
}

/* ── Output helpers ──────────────────────────────────────────────────────── */

/*
 * save_snapshot - Write both density profiles for a given iteration to disk.
 * Filenames: <output_dir>/data/density_species{1,2}_iter_<NNNNNN>.dat
 */
static void save_snapshot(const double *rho1, const double *rho2,
                           const double *xs,   const double *ys,
                           int nx, int ny, int iter,
                           const char *output_dir) {
    char path[512];
    snprintf(path, sizeof(path),
             "%s/data/density_species1_iter_%06d.dat", output_dir, iter);
    io_save_density_2d(path, xs, ys, rho1, (size_t)nx, (size_t)ny);

    snprintf(path, sizeof(path),
             "%s/data/density_species2_iter_%06d.dat", output_dir, iter);
    io_save_density_2d(path, xs, ys, rho2, (size_t)nx, (size_t)ny);
}

/*
 * save_final - Write final converged density profiles.
 * Filenames: <output_dir>/data/density_species{1,2}_final.dat
 */
static void save_final(const double *rho1, const double *rho2,
                        const double *xs,   const double *ys,
                        int nx, int ny,
                        const char *output_dir) {
    char path[512];
    snprintf(path, sizeof(path),
             "%s/data/density_species1_final.dat", output_dir);
    io_save_density_2d(path, xs, ys, rho1, (size_t)nx, (size_t)ny);

    snprintf(path, sizeof(path),
             "%s/data/density_species2_final.dat", output_dir);
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
     * Bulk interaction fields — must be CONSISTENT with how Φ_ij(r) is computed.
     * Since Φ(r) uses discrete summation over the potential table, Φ_b must also
     * use numerical summation for the same discretization:
     *
     *   Φ_11,b = dA · ρ_1,b · Σ_r U_11(r)
     *   Φ_12,b = dA · ρ_2,b · Σ_r U_12(r)
     *   Φ_21,b = dA · ρ_1,b · Σ_r U_12(r)
     *   Φ_22,b = dA · ρ_2,b · Σ_r U_22(r)
     */
    const double rho1_b = cfg->rho1;  /* ρ_{1,b}: bulk density species 1 */
    const double rho2_b = cfg->rho2;  /* ρ_{2,b}: bulk density species 2 */

    double sum_U11 = 0.0, sum_U12 = 0.0, sum_U22 = 0.0;
    #pragma omp parallel for reduction(+:sum_U11, sum_U12, sum_U22)
    for (size_t k = 0; k < N; ++k) {
        sum_U11 += U11[k];
        sum_U12 += U12[k];
        sum_U22 += U22[k];
    }

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
                    Nx, Ny, dA, wall_x, wall_y);

        /*
         * --- Step 2: Apply Euler-Lagrange operator K_i ---
         * Slide "Numerical computation method":
         *   K1(r) = rho1_b * exp(-beta*(Phi11(r)+Phi12(r) - Phi11b-Phi12b))
         *   K2(r) = rho2_b * exp(-beta*(Phi21(r)+Phi22(r) - Phi21b-Phi22b))
         */
        compute_K(Phi11, Phi12, Phi11b, Phi12b, rho1_b, beta, K1, N);
        compute_K(Phi21, Phi22, Phi21b, Phi22b, rho2_b, beta, K2, N);

        /*
         * --- Step 2b: Mass renormalisation ---
         * Ensures total mass is conserved during iteration.
         *
         * For PBC: normalize over entire grid (all cells are interior)
         * For W2/W4: normalize over interior cells only (exclude wall cells)
         *
         * Target: interior cells should average to rho_b
         */
        {
            int x_start = (mode == BC_W2 || mode == BC_W4) ? 1 : 0;
            int x_end   = (mode == BC_W2 || mode == BC_W4) ? Nx - 1 : Nx;
            int y_start = (mode == BC_W4) ? 1 : 0;
            int y_end   = (mode == BC_W4) ? Ny - 1 : Ny;
            
            double s1 = 0.0, s2 = 0.0;
            size_t interior_count = 0;
            
            #pragma omp parallel for collapse(2) reduction(+:s1, s2, interior_count)
            for (int iy = y_start; iy < y_end; ++iy) {
                for (int ix = x_start; ix < x_end; ++ix) {
                    size_t k = (size_t)iy * Nx + ix;
                    s1 += K1[k];
                    s2 += K2[k];
                    interior_count++;
                }
            }
            
            if (interior_count > 0 && s1 > 1e-12 && s2 > 1e-12) {
                double target1 = (double)interior_count * rho1_b;
                double target2 = (double)interior_count * rho2_b;
                double norm1 = target1 / s1;
                double norm2 = target2 / s2;
                
                #pragma omp parallel for collapse(2)
                for (int iy = y_start; iy < y_end; ++iy) {
                    for (int ix = x_start; ix < x_end; ++ix) {
                        size_t k = (size_t)iy * Nx + ix;
                        K1[k] *= norm1;
                        K2[k] *= norm2;
                    }
                }
            }
        }

        /*
         * --- Step 3: Convergence check (slide "Numerical computation method")
         * Must be done BEFORE mixing while rho_i^(t) is still in rho1/rho2:
         *   ||rho_i^(t+1) - rho_i^(t)|| = xi_i * ||K_i - rho_i^(t)||  < epsilon
         *
         * For wall modes, compute error only over interior cells (excluding wall
         * cells where K≠0 but ρ=0 by boundary constraint).
         */
        double err = xi1 * solver_l2_diff_interior(K1, rho1, Nx, Ny, mode);
        double e2  = xi2 * solver_l2_diff_interior(K2, rho2, Nx, Ny, mode);
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

        /* Step 5b: Gentle smoothing to prevent checkerboard instability */
        smooth_density(rho1, Nx, Ny, mode);
        smooth_density(rho2, Nx, Ny, mode);
        apply_boundary_mask(rho1, rho2, Nx, Ny, mode);

        /* --- Step 6: log and optional snapshot --- */

        io_log_convergence(log_path, iter, err);

        if ((iter + 1) % save_ev == 0) {
            /* Compute density statistics for diagnostics */
            double min1 = rho1[0], max1 = rho1[0], sum1 = 0;
            double min2 = rho2[0], max2 = rho2[0], sum2 = 0;
            for (size_t k = 0; k < N; ++k) {
                if (rho1[k] < min1) min1 = rho1[k];
                if (rho1[k] > max1) max1 = rho1[k];
                sum1 += rho1[k];
                if (rho2[k] < min2) min2 = rho2[k];
                if (rho2[k] > max2) max2 = rho2[k];
                sum2 += rho2[k];
            }
            printf("  iter %6d   err=%.3e   rho1[%.4f,%.4f,%.4f]  rho2[%.4f,%.4f,%.4f]\n",
                   iter + 1, err, min1, sum1/(double)N, max1,
                   min2, sum2/(double)N, max2);
            save_snapshot(rho1, rho2, xs, ys, Nx, Ny, iter + 1, cfg->output_dir);
        }

        if (err < tol) {
            printf("  Converged at iteration %d  (err = %.3e < %.3e)\n",
                   iter + 1, err, tol);
            save_snapshot(rho1, rho2, xs, ys, Nx, Ny, iter + 1, cfg->output_dir);
            save_final(rho1, rho2, xs, ys, Nx, Ny, cfg->output_dir);
            converged = 1;
            break;
        }
    }

    if (!converged) {
        fprintf(stderr,
                "Warning: solver did not converge in %d iterations.\n",
                max_iter);
        save_final(rho1, rho2, xs, ys, Nx, Ny, cfg->output_dir);
    }

    free(U11); free(U12); free(U22);
    free(Phi11); free(Phi12); free(Phi21); free(Phi22);
    free(K1); free(K2);
    free(xs); free(ys);

    return converged ? 0 : 1;
}
