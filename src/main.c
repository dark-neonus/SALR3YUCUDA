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

    /* Initial condition: random density field with exact mass conservation.
     *
     * Each cell gets a uniform random value in (0, 1], then the whole field
     * is rescaled so that its mean equals the target bulk density exactly:
     *
     *   r_i[k] = u_k * (rho_b_i * N / sum(u))
     *
     * This guarantees  sum(r_i) = rho_b_i * N  to floating-point precision
     * regardless of the random seed, while keeping the spatial distribution
     * fully random (no imposed stripe or other geometric bias).
     */
    {
        srand(42);
        double sum1 = 0.0, sum2 = 0.0;
        for (size_t k = 0; k < N; ++k) {
            r1[k] = (double)rand() / ((double)RAND_MAX + 1.0) + 1e-6;
            r2[k] = (double)rand() / ((double)RAND_MAX + 1.0) + 1e-6;
            sum1 += r1[k];
            sum2 += r2[k];
        }
        /* Rescale so mean == bulk density (exact mass conservation). */
        double scale1 = cfg.rho1 * (double)N / sum1;
        double scale2 = cfg.rho2 * (double)N / sum2;
        for (size_t k = 0; k < N; ++k) {
            r1[k] *= scale1;
            r2[k] *= scale2;
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
