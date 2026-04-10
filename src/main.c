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
 * Usage:  ./salr_dft <config_file> [--resume <run_id> [iteration]]
 * Example: ./salr_dft configs/default.cfg
 *          ./salr_dft configs/default.cfg --resume run_20260323_143052_a1b2c3d4
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>
#include <math.h>

#include "../include/config.h"
#include "../include/grid.h"
#include "../include/solver.h"
#include "../include/io.h"

#ifdef USE_DB_ENGINE
#include "db_engine.h"
#endif

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
    setbuf(stdout, NULL);

#ifdef USE_DB_ENGINE
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config_file> [--resume <run_id> [iteration]]\n", argv[0]);
        return 1;
    }

    /* Parse command line for --resume flag */
    int resume_mode = 0;
    const char *resume_run_id = NULL;
    int resume_iter = -1;  /* -1 means latest */

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--resume") == 0) {
            resume_mode = 1;
            if (i + 1 < argc) {
                resume_run_id = argv[i + 1];
                i++;
            }
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                resume_iter = atoi(argv[i + 1]);
                i++;
            }
        }
    }

    if (resume_mode && !resume_run_id) {
        fprintf(stderr, "Error: --resume requires a run_id\n");
        return 1;
    }
#else
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        return 1;
    }
#endif

    /* Load config */
    SimConfig cfg;
    if (config_load(argv[1], &cfg) != 0) {
        fprintf(stderr, "Error: failed to load config '%s'\n", argv[1]);
        return 1;
    }
    config_print(&cfg);

    /* Create output directory */
    if (ensure_dir(cfg.output_dir) != 0)
        return 1;

#ifdef USE_DB_ENGINE
    /* Initialize database engine - store runs in project_root/database */
    const char *db_root = getenv("SALR_DB_PATH");
    if (!db_root || db_root[0] == '\0') {
        db_root = "./database";  /* Default: project_root/database */
    }

    if (ensure_dir(db_root) != 0) {
        fprintf(stderr, "Error: failed to create database directory '%s'\n", db_root);
        return 1;
    }
    if (db_init(db_root) != DB_OK) {
        fprintf(stderr, "Error: failed to initialize database engine\n");
        return 1;
    }
    printf("Database initialized at: %s\n", db_root);
#endif

    /* Build coordinate arrays */
    double *xs = grid_create_x(&cfg.grid);
    double *ys = grid_create_y(&cfg.grid);
    if (!xs || !ys) {
        fprintf(stderr, "Error: failed to allocate grid arrays.\n");
        free(xs); free(ys);
#ifdef USE_DB_ENGINE
        db_close();
#endif
        return 1;
    }

    /* Allocate density arrays */
    size_t N   = (size_t)grid_total_points(&cfg.grid);
    double *r1 = malloc(N * sizeof(double));
    double *r2 = malloc(N * sizeof(double));
    if (!r1 || !r2) {
        fprintf(stderr, "Error: failed to allocate density arrays.\n");
        free(xs); free(ys); free(r1); free(r2);
#ifdef USE_DB_ENGINE
        db_close();
#endif
        return 1;
    }

#ifdef USE_DB_ENGINE
    DbRun *run = NULL;
    int start_iter = 0;

    if (resume_mode) {
        /* Resume from checkpoint */
        printf("\nResuming from run '%s'", resume_run_id);
        if (resume_iter >= 0) {
            printf(" at iteration %d", resume_iter);
        } else {
            printf(" (latest snapshot)");
        }
        printf("...\n");

        DbError err = db_resume_state(resume_run_id, resume_iter, &cfg,
                                      r1, r2, &start_iter);
        if (err == DB_ERR_MISMATCH) {
            fprintf(stderr, "Error: grid size mismatch - cannot resume\n");
            free(xs); free(ys); free(r1); free(r2);
            db_close();
            return 1;
        }
        if (err != DB_OK) {
            fprintf(stderr, "Error: failed to resume (error %d)\n", err);
            free(xs); free(ys); free(r1); free(r2);
            db_close();
            return 1;
        }

        /* Open existing run for continuation */
        err = db_run_open(resume_run_id, &run);
        if (err != DB_OK) {
            fprintf(stderr, "Error: failed to open run '%s'\n", resume_run_id);
            free(xs); free(ys); free(r1); free(r2);
            db_close();
            return 1;
        }

        printf("Resumed from iteration %d\n", start_iter);
    } else {
        /* Create new run */
        DbError err = db_run_create(&cfg, &run);
        if (err != DB_OK) {
            fprintf(stderr, "Error: failed to create database run\n");
            free(xs); free(ys); free(r1); free(r2);
            db_close();
            return 1;
        }

        printf("\nCreated run: %s\n", db_run_get_id(run));
        printf("Run directory: %s\n", db_run_get_path(run));

        /* Initialize density based on init_mode */
        const int nx = cfg.grid.nx;
        const int ny = cfg.grid.ny;
        const double Lx = cfg.grid.Lx;
        const double Ly = cfg.grid.Ly;
        const double dx = cfg.grid.dx;
        const double dy = cfg.grid.dy;
        const double PI = 3.14159265358979323846;

        printf("Initializing densities with mode: ");
        if (cfg.init_mode == INIT_SINUSOIDS) {
            printf("SINUSOIDS\n");
        } else if (cfg.init_mode == INIT_TRIVIAL) {
            printf("TRIVIAL (uniform)\n");
        } else {
            printf("RANDOM noise\n");
        }

        srand((unsigned int)67);

        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int idx = iy * nx + ix;
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dy;

                if (cfg.init_mode == INIT_SINUSOIDS) {
                    r1[idx] = cfg.rho1 * (1.0 + sin(x * 2.0 * PI / Lx * 2.0) * sin(y * 2.0 * PI / Ly * 2.0));
                    r2[idx] = cfg.rho2 * (1.0 + cos(x * 2.0 * PI / Lx * 2.0) * cos(y * 2.0 * PI / Ly * 2.0));
                } else if (cfg.init_mode == INIT_TRIVIAL) {
                    r1[idx] = cfg.rho1;
                    r2[idx] = cfg.rho2;
                } else {
                    double noise1 = 1.0 * (rand() / (double)RAND_MAX - 0.5);
                    double noise2 = 1.0 * (rand() / (double)RAND_MAX - 0.5);
                    r1[idx] = cfg.rho1 * (1.0 + noise1);
                    r2[idx] = cfg.rho2 * (1.0 + noise2);
                }

                if (r1[idx] < 1e-6) r1[idx] = 1e-6;
                if (r2[idx] < 1e-6) r2[idx] = 1e-6;
            }
        }

        printf("Density initialization complete. rho1: [%.4f, %.4f], rho2: [%.4f, %.4f]\n",
               r1[0], r1[nx*ny/2], r2[0], r2[nx*ny/2]);
    }
#else
    {
        const int nx = cfg.grid.nx;
        const int ny = cfg.grid.ny;
        const double Lx = cfg.grid.Lx;
        const double Ly = cfg.grid.Ly;
        const double dx = cfg.grid.dx;
        const double dy = cfg.grid.dy;
        const double PI = 3.14159265358979323846;

        srand((unsigned int)67);

        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int idx = iy * nx + ix;
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dy;

                if (cfg.init_mode == INIT_SINUSOIDS) {
                    r1[idx] = cfg.rho1 * (1.0 + sin(x * 2.0 * PI / Lx * 2.0) * sin(y * 2.0 * PI / Ly * 2.0));
                    r2[idx] = cfg.rho2 * (1.0 + cos(x * 2.0 * PI / Lx * 2.0) * cos(y * 2.0 * PI / Ly * 2.0));
                } else if (cfg.init_mode == INIT_TRIVIAL) {
                    r1[idx] = cfg.rho1;
                    r2[idx] = cfg.rho2;
                } else {
                    double noise1 = 1.0 * (rand() / (double)RAND_MAX - 0.5);
                    double noise2 = 1.0 * (rand() / (double)RAND_MAX - 0.5);
                    r1[idx] = cfg.rho1 * (1.0 + noise1);
                    r2[idx] = cfg.rho2 * (1.0 + noise2);
                }

                if (r1[idx] < 1e-6) r1[idx] = 1e-6;
                if (r2[idx] < 1e-6) r2[idx] = 1e-6;
            }
        }
    }
#endif

    /* Run Picard solver */
    printf("\nStarting Picard iteration  (tol=%.2e, max_iter=%d)\n\n",
           cfg.solver.tolerance, cfg.solver.max_iterations);

#ifdef USE_DB_ENGINE
    double final_error = 0.0;
    int status = solver_run_binary_db(r1, r2, &cfg, run, start_iter, &final_error);
#else
    int status = solver_run_binary(r1, r2, &cfg);
#endif

    /* Save final profiles */
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

#ifdef USE_DB_ENGINE
    /* Update registry with final status */
    if (run) {
        int *iters;
        int snapshot_count;
        db_snapshot_list(run, &iters, &snapshot_count);
        free(iters);

        db_registry_update_status(db_run_get_id(run), snapshot_count,
                                  final_error,
                                  (status == 0) ? 1 : 0);

        printf("Run completed: %s (%d snapshots)\n",
               db_run_get_id(run), snapshot_count);

        db_run_close(run);
    }
    db_close();
#endif

    free(xs); free(ys); free(r1); free(r2);
    return (status == 0) ? 0 : 2;
}
