/*
 * db_engine.h - Main public API for the SALR database engine
 *
 * This is the primary interface for:
 *   - Initializing/closing the database
 *   - Creating and managing simulation runs
 *   - Saving and loading HDF5 snapshots
 *   - Querying the registry
 *   - State resumption
 *
 * Usage:
 *   1. Call db_init() at program start
 *   2. Create a new run with db_run_create() or open existing with db_run_open()
 *   3. Save snapshots with db_snapshot_save()
 *   4. Close run with db_run_close()
 *   5. Call db_close() at program end
 */

#ifndef DB_ENGINE_H
#define DB_ENGINE_H

#include <stddef.h>
#include "../include/hdf5_io.h"      /* For SnapshotMeta */
#include "../include/registry.h"     /* For RunSummary */

/* Forward declarations */
struct SimConfig;

/* Error codes */
typedef enum {
    DB_OK              =  0,
    DB_ERR_INIT        = -1,  /* Initialization error */
    DB_ERR_IO          = -2,  /* File I/O error */
    DB_ERR_HDF5        = -3,  /* HDF5 library error */
    DB_ERR_SQLITE      = -4,  /* SQLite error */
    DB_ERR_NOT_FOUND   = -5,  /* Run or snapshot not found */
    DB_ERR_INVALID     = -6,  /* Invalid parameter or state */
    DB_ERR_MISMATCH    = -7,  /* Grid size mismatch on resume */
    DB_ERR_ALLOC       = -8   /* Memory allocation failure */
} DbError;

/* Opaque run handle */
typedef struct DbRun DbRun;

/*============================================================================
 * Database Lifecycle
 *============================================================================*/

/*
 * db_init - Initialize the database engine.
 *
 * Creates the data directory and registry database if they don't exist.
 * Must be called before any other db_* functions.
 *
 * @param data_root  Path to the data directory (e.g., "./data")
 * @return           DB_OK on success
 */
DbError db_init(const char *data_root);

/*
 * db_close - Close the database engine and release resources.
 *
 * Should be called at program end.
 */
void db_close(void);

/*
 * db_is_initialized - Check if database is initialized.
 *
 * @return  1 if initialized, 0 if not
 */
int db_is_initialized(void);

/*============================================================================
 * Run Management
 *============================================================================*/

/*
 * db_run_create - Create a new simulation run.
 *
 * Creates a new run directory and registers it in the database.
 * The run ID is auto-generated from timestamp + config hash.
 *
 * @param cfg       Simulation configuration
 * @param run_out   Output: run handle (caller must call db_run_close)
 * @return          DB_OK on success
 */
DbError db_run_create(const struct SimConfig *cfg, DbRun **run_out);

/*
 * db_run_open - Open an existing run by ID.
 *
 * @param run_id    Run identifier (e.g., "session_20260323_143052_a1b2c3d4")
 * @param run_out   Output: run handle
 * @return          DB_OK on success, DB_ERR_NOT_FOUND if not exists
 */
DbError db_run_open(const char *run_id, DbRun **run_out);

/*
 * db_run_close - Close a run handle and free resources.
 *
 * @param run  Run handle to close
 */
void db_run_close(DbRun *run);

/*
 * db_run_get_id - Get the run ID string.
 *
 * @param run  Run handle
 * @return     Run ID string (valid until db_run_close)
 */
const char *db_run_get_id(const DbRun *run);

/*
 * db_run_get_path - Get the run directory path.
 *
 * @param run  Run handle
 * @return     Directory path (valid until db_run_close)
 */
const char *db_run_get_path(const DbRun *run);

/*============================================================================
 * Snapshot Operations
 *============================================================================*/

/*
 * db_snapshot_save - Save a density snapshot to HDF5.
 *
 * Creates a compressed HDF5 file with density arrays and metadata.
 * File is named snapshot_{iteration:06d}.h5
 *
 * @param run         Run handle
 * @param rho1        Density array species 1 (nx * ny, row-major)
 * @param rho2        Density array species 2 (nx * ny, row-major)
 * @param iteration   Current Picard iteration number
 * @param error       Current L2 convergence error
 * @param delta_error Error change from previous iteration
 * @param cfg         Simulation configuration (for metadata)
 * @return            DB_OK on success
 */
DbError db_snapshot_save(DbRun *run,
                         const double *rho1,
                         const double *rho2,
                         int iteration,
                         double error,
                         double delta_error,
                         const struct SimConfig *cfg);

/*
 * db_snapshot_load - Load a density snapshot from HDF5.
 *
 * @param run             Run handle
 * @param snapshot_iter   Iteration number to load (-1 for latest)
 * @param rho1            Output: pre-allocated array (nx * ny)
 * @param rho2            Output: pre-allocated array (nx * ny)
 * @param meta_out        Output: snapshot metadata (can be NULL)
 * @return                DB_OK on success
 */
DbError db_snapshot_load(const DbRun *run,
                         int snapshot_iter,
                         double *rho1,
                         double *rho2,
                         SnapshotMeta *meta_out);

/*
 * db_snapshot_list - List all snapshots in a run.
 *
 * @param run         Run handle
 * @param iters_out   Output: array of iteration numbers (caller frees)
 * @param count_out   Output: number of snapshots
 * @return            DB_OK on success
 */
DbError db_snapshot_list(const DbRun *run, int **iters_out, int *count_out);

/*
 * db_snapshot_extract_slice - Extract partial data from a snapshot.
 *
 * Reads a rectangular region without loading the full array.
 *
 * @param run             Run handle
 * @param snapshot_iter   Iteration number (-1 for latest)
 * @param dataset         Dataset name ("rho1" or "rho2")
 * @param x_start         Starting X index
 * @param x_count         Number of X elements
 * @param y_start         Starting Y index
 * @param y_count         Number of Y elements
 * @param data_out        Output: pre-allocated array (x_count * y_count)
 * @return                DB_OK on success
 */
DbError db_snapshot_extract_slice(const DbRun *run,
                                  int snapshot_iter,
                                  const char *dataset,
                                  int x_start, int x_count,
                                  int y_start, int y_count,
                                  double *data_out);

/*============================================================================
 * Registry Operations
 *============================================================================*/

/*
 * db_registry_list - List all runs with optional filters.
 *
 * @param temp_min    Minimum temperature (0 to ignore)
 * @param temp_max    Maximum temperature (0 to ignore)
 * @param rho1_min    Minimum rho1 (0 to ignore)
 * @param rho1_max    Maximum rho1 (0 to ignore)
 * @param runs_out    Output: array of RunSummary (caller frees)
 * @param count_out   Output: number of runs
 * @return            DB_OK on success
 */
DbError db_registry_list(double temp_min, double temp_max,
                         double rho1_min, double rho1_max,
                         RunSummary **runs_out, int *count_out);

/*
 * db_registry_set_nickname - Set nickname for a run.
 *
 * Does NOT rename the directory. Only updates the registry.
 *
 * @param run_id    Run identifier
 * @param nickname  New nickname (or NULL to clear)
 * @return          DB_OK on success
 */
DbError db_registry_set_nickname(const char *run_id, const char *nickname);

/*
 * db_registry_delete_run - Delete a run atomically.
 *
 * Removes both the directory and the registry entry in a transaction.
 * This is a DESTRUCTIVE operation.
 *
 * @param run_id  Run identifier
 * @return        DB_OK on success
 */
DbError db_registry_delete_run(const char *run_id);

/*
 * db_registry_update_status - Update run completion status.
 *
 * Call this when the solver finishes.
 *
 * @param run_id          Run identifier
 * @param snapshot_count  Total number of snapshots saved
 * @param final_error     Final convergence error
 * @param converged       1 if converged, 0 otherwise
 * @return                DB_OK on success
 */
DbError db_registry_update_status(const char *run_id,
                                  int snapshot_count,
                                  double final_error,
                                  int converged);

/*============================================================================
 * State Resumption
 *============================================================================*/

/*
 * db_resume_state - Resume solver state from a checkpoint.
 *
 * Validates grid dimensions, loads density arrays, and returns
 * the iteration number to continue from.
 *
 * @param run_id          Run to resume from
 * @param snapshot_iter   Iteration to resume from (-1 for latest)
 * @param cfg             Current config (for grid validation)
 * @param rho1            Output: pre-allocated array (nx * ny)
 * @param rho2            Output: pre-allocated array (nx * ny)
 * @param start_iter_out  Output: iteration number to continue from
 * @return                DB_OK on success, DB_ERR_MISMATCH if grid differs
 */
DbError db_resume_state(const char *run_id,
                        int snapshot_iter,
                        const struct SimConfig *cfg,
                        double *rho1,
                        double *rho2,
                        int *start_iter_out);

/*============================================================================
 * Export Utilities
 *============================================================================*/

/*
 * db_export_ascii - Export a snapshot to ASCII format.
 *
 * Creates a text file compatible with existing visualization scripts.
 * Format: "x y rho1 rho2" columns, blank lines between rows.
 *
 * @param run             Run handle
 * @param snapshot_iter   Iteration number (-1 for latest)
 * @param output_path     Output file path
 * @return                DB_OK on success
 */
DbError db_export_ascii(const DbRun *run,
                        int snapshot_iter,
                        const char *output_path);

#endif /* DB_ENGINE_H */
