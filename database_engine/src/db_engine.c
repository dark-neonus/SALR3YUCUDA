/*
 * db_engine.c - Main public API implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#include "../include/db_engine.h"
#include "../include/db_utils.h"
#include "../include/hdf5_io.h"
#include "../include/registry.h"
#include "../../include/config.h"
#include "../../include/grid.h"

/* Global state */
static char g_data_root[DB_PATH_MAX] = {0};
static int g_initialized = 0;

/* Run handle structure */
struct DbRun {
    char run_id[DB_RUN_ID_MAX];
    char path[DB_PATH_MAX];
    int snapshot_count;
};

/*============================================================================
 * Database Lifecycle
 *============================================================================*/

DbError db_init(const char *data_root) {
    if (g_initialized) {
        return DB_OK;  /* Already initialized */
    }

    if (!data_root) {
        return DB_ERR_INVALID;
    }

    /* Store data root */
    strncpy(g_data_root, data_root, sizeof(g_data_root) - 1);
    g_data_root[sizeof(g_data_root) - 1] = '\0';

    /* Create data directory if needed */
    if (db_ensure_directory(g_data_root) != 0) {
        fprintf(stderr, "db_init: cannot create data directory '%s'\n", g_data_root);
        return DB_ERR_IO;
    }

    /* Initialize registry */
    char db_path[DB_PATH_MAX];
    if (db_path_join(db_path, sizeof(db_path), g_data_root, "sessions.db") != 0) {
        return DB_ERR_IO;
    }

    RegistryError err = registry_init(db_path);
    if (err != REG_OK) {
        fprintf(stderr, "db_init: cannot initialize registry\n");
        return DB_ERR_SQLITE;
    }

    g_initialized = 1;
    return DB_OK;
}

void db_close(void) {
    if (g_initialized) {
        registry_close();
        g_initialized = 0;
        g_data_root[0] = '\0';
    }
}

int db_is_initialized(void) {
    return g_initialized;
}

/*============================================================================
 * Run Management
 *============================================================================*/

DbError db_run_create(const struct SimConfig *cfg, DbRun **run_out) {
    if (!g_initialized) return DB_ERR_INIT;
    if (!cfg || !run_out) return DB_ERR_INVALID;

    /* Allocate run handle */
    DbRun *run = malloc(sizeof(DbRun));
    if (!run) return DB_ERR_ALLOC;

    /* Generate run ID */
    if (db_generate_run_id(run->run_id, sizeof(run->run_id),
                           cfg->temperature, cfg->rho1, cfg->rho2,
                           cfg->grid.nx, cfg->grid.ny) != 0) {
        free(run);
        return DB_ERR_IO;
    }

    /* Create run directory */
    if (db_path_join(run->path, sizeof(run->path), g_data_root, run->run_id) != 0) {
        free(run);
        return DB_ERR_IO;
    }

    if (db_ensure_directory(run->path) != 0) {
        fprintf(stderr, "db_run_create: cannot create directory '%s'\n", run->path);
        free(run);
        return DB_ERR_IO;
    }

    /* Generate config hash */
    char hash_input[256];
    snprintf(hash_input, sizeof(hash_input), "%.6f:%.6f:%.6f:%d:%d",
             cfg->temperature, cfg->rho1, cfg->rho2, cfg->grid.nx, cfg->grid.ny);
    char config_hash[16];
    db_hash_simple(hash_input, strlen(hash_input), config_hash);

    /* Get creation timestamp */
    char created_at[DB_TIMESTAMP_MAX];
    db_format_timestamp(created_at, sizeof(created_at));

    /* Register in database */
    RegistryError err = registry_insert_run(run->run_id, created_at, cfg, config_hash);
    if (err != REG_OK) {
        fprintf(stderr, "db_run_create: failed to register run\n");
        db_remove_directory(run->path);
        free(run);
        return DB_ERR_SQLITE;
    }

    run->snapshot_count = 0;
    *run_out = run;

    return DB_OK;
}

DbError db_run_open(const char *run_id, DbRun **run_out) {
    if (!g_initialized) return DB_ERR_INIT;
    if (!run_id || !run_out) return DB_ERR_INVALID;

    /* Check if run exists in registry */
    if (!registry_run_exists(run_id)) {
        return DB_ERR_NOT_FOUND;
    }

    /* Allocate run handle */
    DbRun *run = malloc(sizeof(DbRun));
    if (!run) return DB_ERR_ALLOC;

    strncpy(run->run_id, run_id, sizeof(run->run_id) - 1);
    run->run_id[sizeof(run->run_id) - 1] = '\0';

    if (db_path_join(run->path, sizeof(run->path), g_data_root, run_id) != 0) {
        free(run);
        return DB_ERR_IO;
    }

    /* Check directory exists */
    if (!db_file_exists(run->path)) {
        free(run);
        return DB_ERR_NOT_FOUND;
    }

    run->snapshot_count = 0;

    *run_out = run;
    return DB_OK;
}

void db_run_close(DbRun *run) {
    if (run) {
        free(run);
    }
}

const char *db_run_get_id(const DbRun *run) {
    return run ? run->run_id : NULL;
}

const char *db_run_get_path(const DbRun *run) {
    return run ? run->path : NULL;
}

/*============================================================================
 * Snapshot Operations
 *============================================================================*/

DbError db_snapshot_save(DbRun *run,
                         const double *rho1,
                         const double *rho2,
                         int iteration,
                         double error,
                         double delta_error,
                         const struct SimConfig *cfg) {
    if (!run || !rho1 || !rho2 || !cfg) return DB_ERR_INVALID;

    /* Build snapshot path */
    char filename[64];
    snprintf(filename, sizeof(filename), "snapshot_%06d.h5", iteration);

    char filepath[DB_PATH_MAX];
    if (db_path_join(filepath, sizeof(filepath), run->path, filename) != 0) {
        return DB_ERR_IO;
    }

    /* Write HDF5 file */
    HDF5Error err = hdf5_write_snapshot(filepath, rho1, rho2,
                                        cfg->grid.nx, cfg->grid.ny,
                                        iteration, error, delta_error, cfg);
    if (err != HDF5_OK) {
        fprintf(stderr, "db_snapshot_save: HDF5 write failed for '%s'\n", filepath);
        return DB_ERR_HDF5;
    }

    run->snapshot_count++;
    return DB_OK;
}

/*
 * Find the latest snapshot in a run directory
 */
static int find_latest_snapshot(const char *run_path) {
    DIR *dir = opendir(run_path);
    if (!dir) return -1;

    int latest = -1;
    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL) {
        int iter;
        if (sscanf(entry->d_name, "snapshot_%d.h5", &iter) == 1) {
            if (iter > latest) {
                latest = iter;
            }
        }
    }

    closedir(dir);
    return latest;
}

DbError db_snapshot_load(const DbRun *run,
                         int snapshot_iter,
                         double *rho1,
                         double *rho2,
                         SnapshotMeta *meta_out) {
    if (!run || !rho1 || !rho2) return DB_ERR_INVALID;

    /* Find snapshot iteration */
    int iter = snapshot_iter;
    if (iter < 0) {
        iter = find_latest_snapshot(run->path);
        if (iter < 0) {
            return DB_ERR_NOT_FOUND;
        }
    }

    /* Build path */
    char filename[64];
    snprintf(filename, sizeof(filename), "snapshot_%06d.h5", iter);

    char filepath[DB_PATH_MAX];
    if (db_path_join(filepath, sizeof(filepath), run->path, filename) != 0) {
        return DB_ERR_IO;
    }

    /* Check file exists */
    if (!db_file_exists(filepath)) {
        return DB_ERR_NOT_FOUND;
    }

    /* Read HDF5 file */
    HDF5Error err = hdf5_read_snapshot(filepath, rho1, rho2, meta_out);
    if (err != HDF5_OK) {
        return DB_ERR_HDF5;
    }

    return DB_OK;
}

DbError db_snapshot_list(const DbRun *run, int **iters_out, int *count_out) {
    if (!run || !iters_out || !count_out) return DB_ERR_INVALID;

    *iters_out = NULL;
    *count_out = 0;

    /* Count snapshots first */
    DIR *dir = opendir(run->path);
    if (!dir) return DB_ERR_IO;

    int count = 0;
    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL) {
        int iter;
        if (sscanf(entry->d_name, "snapshot_%d.h5", &iter) == 1) {
            count++;
        }
    }
    rewinddir(dir);

    if (count == 0) {
        closedir(dir);
        return DB_OK;
    }

    /* Allocate array */
    int *iters = malloc(count * sizeof(int));
    if (!iters) {
        closedir(dir);
        return DB_ERR_ALLOC;
    }

    /* Fill array */
    int i = 0;
    while ((entry = readdir(dir)) != NULL && i < count) {
        int iter;
        if (sscanf(entry->d_name, "snapshot_%d.h5", &iter) == 1) {
            iters[i++] = iter;
        }
    }
    closedir(dir);

    /* Sort ascending */
    for (int a = 0; a < i - 1; a++) {
        for (int b = a + 1; b < i; b++) {
            if (iters[a] > iters[b]) {
                int tmp = iters[a];
                iters[a] = iters[b];
                iters[b] = tmp;
            }
        }
    }

    *iters_out = iters;
    *count_out = i;

    return DB_OK;
}

DbError db_snapshot_extract_slice(const DbRun *run,
                                  int snapshot_iter,
                                  const char *dataset,
                                  int x_start, int x_count,
                                  int y_start, int y_count,
                                  double *data_out) {
    if (!run || !dataset || !data_out) return DB_ERR_INVALID;

    int iter = snapshot_iter;
    if (iter < 0) {
        iter = find_latest_snapshot(run->path);
        if (iter < 0) return DB_ERR_NOT_FOUND;
    }

    char filename[64];
    snprintf(filename, sizeof(filename), "snapshot_%06d.h5", iter);

    char filepath[DB_PATH_MAX];
    if (db_path_join(filepath, sizeof(filepath), run->path, filename) != 0) {
        return DB_ERR_IO;
    }

    HDF5Error err = hdf5_read_hyperslab(filepath, dataset,
                                        x_start, x_count,
                                        y_start, y_count,
                                        data_out);
    return (err == HDF5_OK) ? DB_OK : DB_ERR_HDF5;
}

/*============================================================================
 * Registry Operations
 *============================================================================*/

DbError db_registry_list(double temp_min, double temp_max,
                         double rho1_min, double rho1_max,
                         RunSummary **runs_out, int *count_out) {
    if (!g_initialized) return DB_ERR_INIT;
    if (!runs_out || !count_out) return DB_ERR_INVALID;

    RegistryFilter filter = {0};

    if (temp_min != 0 || temp_max != 0) {
        filter.use_temp_filter = 1;
        filter.temp_min = temp_min;
        filter.temp_max = (temp_max != 0) ? temp_max : 1e10;
    }

    if (rho1_min != 0 || rho1_max != 0) {
        filter.use_rho1_filter = 1;
        filter.rho1_min = rho1_min;
        filter.rho1_max = (rho1_max != 0) ? rho1_max : 1e10;
    }

    RegistryError err = registry_query_runs(&filter, runs_out, count_out);
    return (err == REG_OK) ? DB_OK : DB_ERR_SQLITE;
}

DbError db_registry_set_nickname(const char *run_id, const char *nickname) {
    if (!g_initialized) return DB_ERR_INIT;
    if (!run_id) return DB_ERR_INVALID;

    RegistryError err = registry_set_nickname(run_id, nickname);
    return (err == REG_OK) ? DB_OK : DB_ERR_SQLITE;
}

DbError db_registry_delete_run(const char *run_id) {
    if (!g_initialized) return DB_ERR_INIT;
    if (!run_id) return DB_ERR_INVALID;

    /* Begin transaction for atomicity */
    registry_begin_transaction();

    /* Build run path */
    char run_path[DB_PATH_MAX];
    if (db_path_join(run_path, sizeof(run_path), g_data_root, run_id) != 0) {
        registry_rollback();
        return DB_ERR_IO;
    }

    /* Delete directory */
    if (db_file_exists(run_path)) {
        if (db_remove_directory(run_path) != 0) {
            fprintf(stderr, "db_registry_delete_run: failed to remove directory '%s'\n", run_path);
            registry_rollback();
            return DB_ERR_IO;
        }
    }

    /* Delete registry entry */
    RegistryError err = registry_delete_run(run_id);
    if (err != REG_OK) {
        registry_rollback();
        return DB_ERR_SQLITE;
    }

    registry_commit();
    return DB_OK;
}

DbError db_registry_update_status(const char *run_id,
                                  int snapshot_count,
                                  double final_error,
                                  int converged) {
    if (!g_initialized) return DB_ERR_INIT;
    if (!run_id) return DB_ERR_INVALID;

    RegistryError err = registry_update_status(run_id, snapshot_count, final_error, converged);
    return (err == REG_OK) ? DB_OK : DB_ERR_SQLITE;
}

/*============================================================================
 * State Resumption
 *============================================================================*/

DbError db_resume_state(const char *run_id,
                        int snapshot_iter,
                        const struct SimConfig *cfg,
                        double *rho1,
                        double *rho2,
                        int *start_iter_out) {
    if (!g_initialized) return DB_ERR_INIT;
    if (!run_id || !cfg || !rho1 || !rho2 || !start_iter_out) {
        return DB_ERR_INVALID;
    }

    /* Open run */
    DbRun *run;
    DbError err = db_run_open(run_id, &run);
    if (err != DB_OK) return err;

    /* Find snapshot */
    int iter = snapshot_iter;
    if (iter < 0) {
        iter = find_latest_snapshot(run->path);
        if (iter < 0) {
            db_run_close(run);
            return DB_ERR_NOT_FOUND;
        }
    }

    /* Build path */
    char filename[64];
    snprintf(filename, sizeof(filename), "snapshot_%06d.h5", iter);

    char filepath[DB_PATH_MAX];
    db_path_join(filepath, sizeof(filepath), run->path, filename);

    /* Validate grid */
    HDF5Error hdf_err = hdf5_validate_grid(filepath, cfg);
    if (hdf_err == HDF5_ERR_MISMATCH) {
        db_run_close(run);
        return DB_ERR_MISMATCH;
    }
    if (hdf_err != HDF5_OK) {
        db_run_close(run);
        return DB_ERR_HDF5;
    }

    /* Load snapshot */
    SnapshotMeta meta;
    hdf_err = hdf5_read_snapshot(filepath, rho1, rho2, &meta);
    if (hdf_err != HDF5_OK) {
        db_run_close(run);
        return DB_ERR_HDF5;
    }

    *start_iter_out = meta.iteration;

    db_run_close(run);
    return DB_OK;
}

/*============================================================================
 * Export Utilities
 *============================================================================*/

DbError db_export_ascii(const DbRun *run,
                        int snapshot_iter,
                        const char *output_path) {
    if (!run || !output_path) return DB_ERR_INVALID;

    /* Find snapshot */
    int iter = snapshot_iter;
    if (iter < 0) {
        iter = find_latest_snapshot(run->path);
        if (iter < 0) return DB_ERR_NOT_FOUND;
    }

    /* Build path */
    char filename[64];
    snprintf(filename, sizeof(filename), "snapshot_%06d.h5", iter);

    char filepath[DB_PATH_MAX];
    db_path_join(filepath, sizeof(filepath), run->path, filename);

    /* Read metadata for grid size */
    SnapshotMeta meta;
    HDF5Error hdf_err = hdf5_read_metadata(filepath, &meta);
    if (hdf_err != HDF5_OK) return DB_ERR_HDF5;

    int nx = meta.nx;
    int ny = meta.ny;
    size_t N = (size_t)nx * ny;

    /* Allocate arrays */
    double *rho1 = malloc(N * sizeof(double));
    double *rho2 = malloc(N * sizeof(double));
    if (!rho1 || !rho2) {
        free(rho1);
        free(rho2);
        return DB_ERR_ALLOC;
    }

    /* Read data */
    hdf_err = hdf5_read_snapshot(filepath, rho1, rho2, NULL);
    if (hdf_err != HDF5_OK) {
        free(rho1);
        free(rho2);
        return DB_ERR_HDF5;
    }

    /* Write ASCII file */
    FILE *f = fopen(output_path, "w");
    if (!f) {
        free(rho1);
        free(rho2);
        return DB_ERR_IO;
    }

    fprintf(f, "# x y rho1 rho2\n");
    fprintf(f, "# iteration=%d, temperature=%.6f, rho1_bulk=%.6f, rho2_bulk=%.6f\n",
            meta.iteration, meta.temperature, meta.rho1_bulk, meta.rho2_bulk);

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            double x = (ix + 0.5) * meta.dx;
            double y = (iy + 0.5) * meta.dy;
            size_t idx = (size_t)iy * nx + ix;
            fprintf(f, "%.10g %.10g %.10g %.10g\n", x, y, rho1[idx], rho2[idx]);
        }
        fputc('\n', f);  /* Blank line between rows for gnuplot */
    }

    fclose(f);
    free(rho1);
    free(rho2);

    return DB_OK;
}
