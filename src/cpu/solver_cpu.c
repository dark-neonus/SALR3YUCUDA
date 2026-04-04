/* solver_cpu.c - Picard DFT solver for 2-component SALR mixture (CPU/OpenMP) */

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

/* Build potential table U_ij(r) for all grid offsets */
static void build_potential_table(double *table, int si, int sj,
                                  const PotentialParams *p,
                                  int nx, int ny, double dx, double dy,
                                  int wall_x, int wall_y) {
    #pragma omp parallel for collapse(2)
    for (int diy = 0; diy < ny; ++diy) {
        for (int dix = 0; dix < nx; ++dix) {
            /* Compute displacement with minimum-image for PBC or actual for walls */
            double dry = wall_y ? diy * dy
                                : ((diy <= ny / 2) ? diy * dy : (diy - ny) * dy);
            double drx = wall_x ? dix * dx
                                : ((dix <= nx / 2) ? dix * dx : (dix - nx) * dx);
            double r   = sqrt(drx * drx + dry * dry);
            table[diy * nx + dix] = potential_u(si, sj, r, p);
        }
    }
}

/* Compute bulk field analytically: 2*pi*rho_b * sum_m (A_m/alpha_m) * (1 - exp(-alpha_m*rc)) */
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

/* Convolve densities with potential tables to compute Phi fields */
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

/* Euler-Lagrange operator: K_i = rho_b * exp(-beta*(Phi_a + Phi_b - Phi_ab - Phi_bb)) */
static void compute_K(const double *Phi_a,  const double *Phi_b,
                      double Phi_ab,        double Phi_bb,
                      double rho0b,         double beta,
                      double *K,            size_t N) {
    #pragma omp parallel for
    for (size_t k = 0; k < N; ++k) {
        double arg = -beta * (Phi_a[k] + Phi_b[k] - Phi_ab - Phi_bb);
        /* Clamp exponent to prevent overflow */
        if (arg >  500.0) arg =  500.0;
        if (arg < -500.0) arg = -500.0;
        K[k] = rho0b * exp(arg);
    }
}

/* Zero density at wall boundary cells */
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

/* 5-point Laplacian smoothing to suppress checkerboard instability */
#define SMOOTH_EPS 0.01

static void smooth_density(double *rho, int nx, int ny, int mode)
{
    double *tmp = malloc((size_t)(nx * ny) * sizeof(double));
    if (!tmp) return;

    int wall_x = (mode == BC_W2 || mode == BC_W4);
    int wall_y = (mode == BC_W4);
    int x_start = wall_x ? 1 : 0;
    int x_end   = wall_x ? nx - 1 : nx;
    int y_start = wall_y ? 1 : 0;
    int y_end   = wall_y ? ny - 1 : ny;

    memcpy(tmp, rho, (size_t)(nx * ny) * sizeof(double));

    #pragma omp parallel for collapse(2)
    for (int iy = y_start; iy < y_end; ++iy) {
        for (int ix = x_start; ix < x_end; ++ix) {
            int ym, yp, xm, xp;
            if (wall_y) {
                ym = (iy > 0) ? iy - 1 : iy;
                yp = (iy < ny - 1) ? iy + 1 : iy;
            } else {
                ym = (iy - 1 + ny) % ny;
                yp = (iy + 1) % ny;
            }
            if (wall_x) {
                xm = (ix > 0) ? ix - 1 : ix;
                xp = (ix < nx - 1) ? ix + 1 : ix;
            } else {
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

/* L2 norm of difference (a - b) */
double solver_l2_diff(const double *a, const double *b, size_t n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum / (double)n);
}

/* L2 difference over interior cells only (excludes wall cells) */
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

/* Save snapshot to disk */
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

/* Save final density profiles */
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

/* Boltzmann constant in SI units (J/K) */
#define BOLTZMANN_CONSTANT 1.380649e-23

/* ── Main solver entry point ─────────────────────────────────────────────── */

int solver_run_binary(double *rho1, double *rho2, struct SimConfig *cfg) {
    /*
     * Slide "Parameters used in solving the problem":
     *   Nx = Lx/dx,  Ny = Ly/dy  (grid dimensions)
     *   dA = dx*dy               (cell area, slide "Two-dimensional case")
     *   beta = 1/(k_B*T)         (slide "Grand thermodynamic potential")
     */
    const int    Nx   = cfg->grid.nx;         /* number of grid cells in x  */
    const int    Ny   = cfg->grid.ny;         /* number of grid cells in y  */
    const double dx   = cfg->grid.dx;         /* Delta_x                    */
    const double dy   = cfg->grid.dy;         /* Delta_y                    */
    const size_t N    = (size_t)(Nx * Ny);    /* total grid points Nx*Ny    */
    const double dA   = dx * dy;              /* cell area Delta_x*Delta_y  */
    const double beta = 1.0 / (BOLTZMANN_CONSTANT * cfg->temperature); /* beta = 1/(k_B*T) */
    const int    mode = cfg->boundary_mode;

    /* wall_x/wall_y: actual (non-minimum-image) distances for walled axes */
    const int wall_x = (mode == BC_W2 || mode == BC_W4);
    const int wall_y = (mode == BC_W4);

    /* Potential tables U_ij(r) */
    double *U11   = malloc(N * sizeof(double));
    double *U12   = malloc(N * sizeof(double));
    double *U22   = malloc(N * sizeof(double));

    /* Interaction field arrays */
    double *Phi11 = malloc(N * sizeof(double));
    double *Phi12 = malloc(N * sizeof(double));
    double *Phi21 = malloc(N * sizeof(double));
    double *Phi22 = malloc(N * sizeof(double));

    /* Euler-Lagrange operator results */
    double *K1    = malloc(N * sizeof(double));
    double *K2    = malloc(N * sizeof(double));

    /* Cell-centre coordinates for output */
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

    /* Build potential tables */
    build_potential_table(U11, 0, 0, &cfg->potential, Nx, Ny, dx, dy, wall_x, wall_y);
    build_potential_table(U12, 0, 1, &cfg->potential, Nx, Ny, dx, dy, wall_x, wall_y);
    build_potential_table(U22, 1, 1, &cfg->potential, Nx, Ny, dx, dy, wall_x, wall_y);

    /* Compute bulk interaction fields numerically */
    const double rho1_b = cfg->rho1;
    const double rho2_b = cfg->rho2;

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

    /* Cell-centre coordinates */
    for (int i = 0; i < Nx; ++i) xs[i] = (i + 0.5) * dx;
    for (int j = 0; j < Ny; ++j) ys[j] = (j + 0.5) * dy;

    /* Convergence log */
    char log_path[512];
    snprintf(log_path, sizeof(log_path), "%s/convergence.dat", cfg->output_dir);
    {
        FILE *lf = fopen(log_path, "w");
        if (lf) { fprintf(lf, "# iter  L2_error\n"); fclose(lf); }
    }

    /* Save simulation parameters to output directory */
    char param_path[512];
    snprintf(param_path, sizeof(param_path), "%s/parameters.cfg", cfg->output_dir);
    io_save_parameters(param_path, cfg);

    const int    max_iter   = cfg->solver.max_iterations;
    const double tol        = cfg->solver.tolerance;
    double       xi1        = cfg->solver.xi1;
    double       xi2        = cfg->solver.xi2;
    const int    save_ev    = cfg->save_every;
    const double err_thresh = cfg->solver.error_change_threshold;
    const double xi_damp    = cfg->solver.xi_damping_factor;

    /* Enforce boundary mask */
    apply_boundary_mask(rho1, rho2, Nx, Ny, mode);

    /* Save initial density snapshot before first iteration */
    save_snapshot(rho1, rho2, xs, ys, Nx, Ny, 0, cfg->output_dir);

    int converged = 0;
    double prev_err = -1.0;  /* previous iteration error (negative = uninitialized) */

    for (int iter = 0; iter < max_iter; ++iter) {

        /* Step 1: Compute interaction fields Phi_ij */
        compute_Phi(Phi11, Phi12, Phi21, Phi22,
                    rho1, rho2, U11, U12, U22,
                    Nx, Ny, dA, wall_x, wall_y);

        /* Step 2: Apply Euler-Lagrange operator */
        compute_K(Phi11, Phi12, Phi11b, Phi12b, rho1_b, beta, K1, N);
        compute_K(Phi21, Phi22, Phi21b, Phi22b, rho2_b, beta, K2, N);

        /* Step 2b: Mass renormalisation */
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

        /* Step 3: Convergence check */
        double err = xi1 * solver_l2_diff_interior(K1, rho1, Nx, Ny, mode);
        double e2  = xi2 * solver_l2_diff_interior(K2, rho2, Nx, Ny, mode);
        if (e2 > err) err = e2;

        /* Step 3b: Adaptive damping - reduce mixing when error change is small */
        double err_delta = (prev_err >= 0.0) ? fabs(prev_err - err) : -1.0;
        if (prev_err >= 0.0) {
            double err_change = err_delta;
            if (err_change < err_thresh && err_change > 0.0) {
                xi1 *= xi_damp;
                xi2 *= xi_damp;
            }
        }
        prev_err = err;

        /* Step 4: Picard mixing */
        vec_add_scaled(rho1, 1.0 - xi1, K1, xi1, rho1, N);
        vec_add_scaled(rho2, 1.0 - xi2, K2, xi2, rho2, N);

        /* Step 5: Apply boundary conditions */
        apply_boundary_mask(rho1, rho2, Nx, Ny, mode);

        /* Step 5b: Anti-checkerboard smoothing */
        smooth_density(rho1, Nx, Ny, mode);
        smooth_density(rho2, Nx, Ny, mode);
        apply_boundary_mask(rho1, rho2, Nx, Ny, mode);

        /* Step 6: Log and snapshot */
        io_log_convergence(log_path, iter, err);

        if ((iter + 1) % save_ev == 0) {
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
            printf("  iter %6d   err=%.3e   err_delta=%.3e   xi1=%.4f   xi2=%.4f   rho1[%.4f,%.4f,%.4f]  rho2[%.4f,%.4f,%.4f]\n",
                   iter + 1, err, err_delta, xi1, xi2, min1, sum1/(double)N, max1,
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

#ifdef USE_DB_ENGINE
#include "db_engine.h"

/* ── Solver with HDF5 database support ───────────────────────────────────── */

int solver_run_binary_db(double *rho1, double *rho2, struct SimConfig *cfg,
                         struct DbRun *run, int start_iter, double *final_error_out) {
    /*
     * Same as solver_run_binary, but with HDF5 snapshot support
     * and ability to resume from a checkpoint.
     */
    const int    Nx   = cfg->grid.nx;
    const int    Ny   = cfg->grid.ny;
    const double dx   = cfg->grid.dx;
    const double dy   = cfg->grid.dy;
    const size_t N    = (size_t)(Nx * Ny);
    const double dA   = dx * dy;
    const double beta = 1.0 / (BOLTZMANN_CONSTANT * cfg->temperature);
    const int    mode = cfg->boundary_mode;

    const int wall_x = (mode == BC_W2 || mode == BC_W4);
    const int wall_y = (mode == BC_W4);

    /* Potential tables U_ij(r) */
    double *U11   = malloc(N * sizeof(double));
    double *U12   = malloc(N * sizeof(double));
    double *U22   = malloc(N * sizeof(double));

    /* Interaction field arrays */
    double *Phi11 = malloc(N * sizeof(double));
    double *Phi12 = malloc(N * sizeof(double));
    double *Phi21 = malloc(N * sizeof(double));
    double *Phi22 = malloc(N * sizeof(double));

    /* Euler-Lagrange operator results */
    double *K1    = malloc(N * sizeof(double));
    double *K2    = malloc(N * sizeof(double));

    /* Cell-centre coordinates for output */
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

    /* Build potential tables */
    build_potential_table(U11, 0, 0, &cfg->potential, Nx, Ny, dx, dy, wall_x, wall_y);
    build_potential_table(U12, 0, 1, &cfg->potential, Nx, Ny, dx, dy, wall_x, wall_y);
    build_potential_table(U22, 1, 1, &cfg->potential, Nx, Ny, dx, dy, wall_x, wall_y);

    /* Compute bulk interaction fields */
    const double rho1_b = cfg->rho1;
    const double rho2_b = cfg->rho2;

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

    /* Cell-centre coordinates */
    for (int i = 0; i < Nx; ++i) xs[i] = (i + 0.5) * dx;
    for (int j = 0; j < Ny; ++j) ys[j] = (j + 0.5) * dy;

    /* Convergence log (append mode for resumption) */
    char log_path[512];
    snprintf(log_path, sizeof(log_path), "%s/convergence.dat", cfg->output_dir);
    if (start_iter == 0) {
        FILE *lf = fopen(log_path, "w");
        if (lf) { fprintf(lf, "# iter  L2_error\n"); fclose(lf); }
    }

    /* Save simulation parameters */
    char param_path[512];
    snprintf(param_path, sizeof(param_path), "%s/parameters.cfg", cfg->output_dir);
    io_save_parameters(param_path, cfg);

    const int    max_iter   = cfg->solver.max_iterations;
    const double tol        = cfg->solver.tolerance;
    double       xi1        = cfg->solver.xi1;
    double       xi2        = cfg->solver.xi2;
    const int    save_ev    = cfg->save_every;
    const double err_thresh = cfg->solver.error_change_threshold;
    const double xi_damp    = cfg->solver.xi_damping_factor;

    /* Enforce boundary mask */
    apply_boundary_mask(rho1, rho2, Nx, Ny, mode);

    /* Save initial snapshot (HDF5) if new run */
    if (start_iter == 0) {
        db_snapshot_save(run, rho1, rho2, 0, 1.0, -1.0, cfg);
        /* Also save ASCII for compatibility */
        save_snapshot(rho1, rho2, xs, ys, Nx, Ny, 0, cfg->output_dir);
    }

    int converged = 0;
    double prev_err = -1.0;
    double final_err = 0.0;

    for (int iter = start_iter; iter < max_iter; ++iter) {

        /* Step 1: Compute interaction fields */
        compute_Phi(Phi11, Phi12, Phi21, Phi22,
                    rho1, rho2, U11, U12, U22,
                    Nx, Ny, dA, wall_x, wall_y);

        /* Step 2: Apply Euler-Lagrange operator */
        compute_K(Phi11, Phi12, Phi11b, Phi12b, rho1_b, beta, K1, N);
        compute_K(Phi21, Phi22, Phi21b, Phi22b, rho2_b, beta, K2, N);

        /* Step 2b: Mass renormalisation */
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

        /* Step 3: Convergence check */
        double err = xi1 * solver_l2_diff_interior(K1, rho1, Nx, Ny, mode);
        double e2  = xi2 * solver_l2_diff_interior(K2, rho2, Nx, Ny, mode);
        if (e2 > err) err = e2;
        final_err = err;

        /* Step 3b: Adaptive damping */
        double err_delta = (prev_err >= 0.0) ? fabs(prev_err - err) : -1.0;
        if (prev_err >= 0.0) {
            double err_change = err_delta;
            if (err_change < err_thresh && err_change > 0.0) {
                xi1 *= xi_damp;
                xi2 *= xi_damp;
            }
        }
        prev_err = err;

        /* Step 4: Picard mixing */
        vec_add_scaled(rho1, 1.0 - xi1, K1, xi1, rho1, N);
        vec_add_scaled(rho2, 1.0 - xi2, K2, xi2, rho2, N);

        /* Step 5: Apply boundary conditions */
        apply_boundary_mask(rho1, rho2, Nx, Ny, mode);

        /* Step 5b: Anti-checkerboard smoothing */
        smooth_density(rho1, Nx, Ny, mode);
        smooth_density(rho2, Nx, Ny, mode);
        apply_boundary_mask(rho1, rho2, Nx, Ny, mode);

        /* Step 6: Log and snapshot */
        io_log_convergence(log_path, iter, err);

        if ((iter + 1) % save_ev == 0) {
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
            printf("  iter %6d   err=%.3e   err_delta=%.3e   xi1=%.4f   xi2=%.4f   rho1[%.4f,%.4f,%.4f]  rho2[%.4f,%.4f,%.4f]\n",
                   iter + 1, err, err_delta, xi1, xi2, min1, sum1/(double)N, max1,
                   min2, sum2/(double)N, max2);

            /* Save HDF5 snapshot */
            db_snapshot_save(run, rho1, rho2, iter + 1, err, err_delta, cfg);

            /* Also save ASCII for compatibility */
            save_snapshot(rho1, rho2, xs, ys, Nx, Ny, iter + 1, cfg->output_dir);
        }

        if (err < tol) {
            printf("  Converged at iteration %d  (err = %.3e < %.3e)\n",
                   iter + 1, err, tol);
            db_snapshot_save(run, rho1, rho2, iter + 1, err, err_delta, cfg);
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

    /* Set output parameter for final error */
    if (final_error_out) {
        *final_error_out = final_err;
    }

    return converged ? 0 : 1;
}
#endif /* USE_DB_ENGINE */
