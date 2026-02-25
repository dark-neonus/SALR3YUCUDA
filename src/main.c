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
    setbuf(stdout, NULL);  /* unbuffered output for real-time monitoring */
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

    /* Initial condition: random density placement.
     * Each grid point gets a random density value in [0, 2*rho_bulk].
     */
    {
        const int N = cfg.grid.nx * cfg.grid.ny;
        srand(42);
        for (int i = 0; i < N; ++i) {
            r1[i] = cfg.rho1 * (2.0 * rand() / (double)(RAND_MAX + 1.0) + 1e-6);
            r2[i] = cfg.rho2 * (2.0 * rand() / (double)(RAND_MAX + 1.0) + 1e-6);
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
