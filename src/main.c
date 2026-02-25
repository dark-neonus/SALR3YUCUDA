/*
 * main.c — Entry point for the SALR DFT solver
 *
 * Workflow:
 *   1. Parse the config file.
 *   2. Create the computational grid and allocate density arrays.
 *   3. Initialise both species to a uniform (bulk) density.
 *   4. Ensure the output directory exists.
 *   5. Run the Picard solver.
 *   6. Write the final density profiles.
 *
 * Usage:  ./salr_dft <config_file>
 * Example: ./salr_dft configs/default.cfg
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

#include "../include/config.h"
#include "../include/grid.h"
#include "../include/solver.h"
#include "../include/io.h"

/*
 * ensure_dir - Create a directory if it does not already exist.
 * Returns 0 on success, -1 if creation fails for a reason other than
 * the directory already existing.
 */
static int ensure_dir(const char *path) {
    if (mkdir(path, 0755) == 0 || errno == EEXIST)
        return 0;
    perror(path);
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        return 1;
    }

    /* ── Load configuration ─────────────────────────────────────────────── */
    SimConfig cfg;
    if (config_load(argv[1], &cfg) != 0) {
        fprintf(stderr, "Error: failed to load config '%s'\n", argv[1]);
        return 1;
    }
    config_print(&cfg);

    /* ── Prepare output directory ───────────────────────────────────────── */
    if (ensure_dir(cfg.output_dir) != 0)
        return 1;

    /* ── Build coordinate arrays ────────────────────────────────────────── */
    double *xs = grid_create_x(&cfg.grid);
    double *ys = grid_create_y(&cfg.grid);
    if (!xs || !ys) {
        fprintf(stderr, "Error: failed to allocate grid arrays.\n");
        free(xs); free(ys);
        return 1;
    }

    /* ── Allocate and initialise density arrays ─────────────────────────── */
    size_t N   = (size_t)grid_total_points(&cfg.grid);
    double *r1 = malloc(N * sizeof(double));
    double *r2 = malloc(N * sizeof(double));
    if (!r1 || !r2) {
        fprintf(stderr, "Error: failed to allocate density arrays.\n");
        free(xs); free(ys); free(r1); free(r2);
        return 1;
    }

    /* Initial condition: half-space stripe seed to bias towards the
     * phase-separated structured solution (avoids trivial uniform fixed point).
     *
     * Half A (ix < Nx/2): high rho1, low  rho2
     * Half B (ix >= Nx/2): low  rho1, high rho2
     * Global averages are conserved: (rho_hi + rho_lo)/2 = rho_b.
     * Small noise breaks y-symmetry so the stripe can tilt.
     */
    {
        double rho1_hi = cfg.rho1 * 1.25;
        double rho1_lo = cfg.rho1 * 0.75;
        double rho2_hi = cfg.rho2 * 1.25;
        double rho2_lo = cfg.rho2 * 0.75;
        int Nx = cfg.grid.nx, Ny = cfg.grid.ny;
        srand(42);
        for (int iy = 0; iy < Ny; ++iy) {
            for (int ix = 0; ix < Nx; ++ix) {
                double noise = 0.05 * (2.0 * rand()/(double)RAND_MAX - 1.0);
                size_t k = (size_t)(iy * Nx + ix);
                if (ix < Nx / 2) {
                    r1[k] = rho1_hi * (1.0 + noise);
                    r2[k] = rho2_lo * (1.0 + noise);
                } else {
                    r1[k] = rho1_lo * (1.0 + noise);
                    r2[k] = rho2_hi * (1.0 + noise);
                }
                if (r1[k] < 1e-6) r1[k] = 1e-6;
                if (r2[k] < 1e-6) r2[k] = 1e-6;
            }
        }
    }

    /* ── Run Picard solver ──────────────────────────────────────────────── */
    printf("\nStarting Picard iteration  (tol=%.2e, max_iter=%d)\n\n",
           cfg.solver.tolerance, cfg.solver.max_iterations);

    int status = solver_run_binary(r1, r2, &cfg);

    /* ── Save final profiles ────────────────────────────────────────────── */
    char path[512];

    snprintf(path, sizeof(path), "%s/density_species1_final.dat",
             cfg.output_dir);
    io_save_density_2d(path, xs, ys, r1,
                       (size_t)cfg.grid.nx, (size_t)cfg.grid.ny);

    snprintf(path, sizeof(path), "%s/density_species2_final.dat",
             cfg.output_dir);
    io_save_density_2d(path, xs, ys, r2,
                       (size_t)cfg.grid.nx, (size_t)cfg.grid.ny);

    printf("\nFinal profiles saved to '%s'.\n", cfg.output_dir);

    free(xs); free(ys); free(r1); free(r2);
    return (status == 0) ? 0 : 2;
}
