/*
 * registry.h - SQLite registry operations for run management
 *
 * The registry provides:
 *   - Run indexing with metadata
 *   - Fast filtering by temperature, density, etc.
 *   - Concurrent access via SQLite WAL mode
 *   - Atomic transactions for safe operations
 */

#ifndef REGISTRY_H
#define REGISTRY_H

#include <sqlite3.h>

/* Forward declarations */
struct SimConfig;

/* Error codes */
typedef enum {
    REG_OK           =  0,
    REG_ERR_OPEN     = -1,  /* Failed to open database */
    REG_ERR_EXEC     = -2,  /* SQL execution error */
    REG_ERR_PREPARE  = -3,  /* SQL prepare error */
    REG_ERR_NOT_FOUND = -4, /* Run not found */
    REG_ERR_ALLOC    = -5,  /* Memory allocation error */
    REG_ERR_BUSY     = -6   /* Database is busy/locked */
} RegistryError;

/* Run summary structure for queries */
typedef struct {
    char   run_id[64];
    char   nickname[128];
    char   created_at[32];
    double temperature;
    double rho1_bulk;
    double rho2_bulk;
    int    nx;
    int    ny;
    double dx;
    double dy;
    char   boundary_mode[8];
    char   config_hash[16];
    char   source[64];         /* Source session ID (empty if from scratch) */
    int    snapshot_count;
    double final_error;
    int    converged;
} RunSummary;

/* Query filter structure */
typedef struct {
    double temp_min;
    double temp_max;
    double rho1_min;
    double rho1_max;
    double rho2_min;
    double rho2_max;
    int    use_temp_filter;
    int    use_rho1_filter;
    int    use_rho2_filter;
} RegistryFilter;

/*
 * registry_init - Initialize the registry database.
 *
 * Creates tables and indices if they don't exist.
 * Enables WAL mode for better concurrency.
 *
 * @param db_path  Path to the SQLite database file
 * @return         REG_OK on success, error code on failure
 */
RegistryError registry_init(const char *db_path);

/*
 * registry_close - Close the registry database connection.
 */
void registry_close(void);

/*
 * registry_get_handle - Get the raw SQLite handle.
 *
 * For advanced queries. Use with caution.
 *
 * @return  SQLite database handle, or NULL if not initialized
 */
sqlite3 *registry_get_handle(void);

/*
 * registry_insert_run - Insert a new run into the registry.
 *
 * @param run_id        Unique run identifier
 * @param created_at    ISO8601 timestamp
 * @param cfg           Simulation configuration
 * @param config_hash   Hash of the configuration
 * @return              REG_OK on success
 */
RegistryError registry_insert_run(const char *run_id,
                                  const char *created_at,
                                  const struct SimConfig *cfg,
                                  const char *config_hash);

/*
 * registry_update_status - Update run completion status.
 *
 * @param run_id          Run identifier
 * @param snapshot_count  Number of snapshots saved
 * @param final_error     Final convergence error
 * @param converged       1 if converged, 0 otherwise
 * @return                REG_OK on success
 */
RegistryError registry_update_status(const char *run_id,
                                     int snapshot_count,
                                     double final_error,
                                     int converged);

/*
 * registry_set_nickname - Set or update the nickname for a run.
 *
 * @param run_id    Run identifier
 * @param nickname  New nickname (or NULL to clear)
 * @return          REG_OK on success
 */
RegistryError registry_set_nickname(const char *run_id, const char *nickname);

/*
 * registry_delete_run - Delete a run from the registry.
 *
 * NOTE: This only removes the registry entry, not the files.
 *       Use db_registry_delete_run() for atomic deletion.
 *
 * @param run_id  Run identifier
 * @return        REG_OK on success
 */
RegistryError registry_delete_run(const char *run_id);

/*
 * registry_run_exists - Check if a run exists in the registry.
 *
 * @param run_id  Run identifier
 * @return        1 if exists, 0 if not
 */
int registry_run_exists(const char *run_id);

/*
 * registry_query_runs - Query runs with optional filters.
 *
 * @param filter     Filter criteria (or NULL for all runs)
 * @param runs_out   Output: array of RunSummary (caller frees)
 * @param count_out  Output: number of runs returned
 * @return           REG_OK on success
 */
RegistryError registry_query_runs(const RegistryFilter *filter,
                                  RunSummary **runs_out,
                                  int *count_out);

/*
 * registry_get_run - Get a single run by ID.
 *
 * @param run_id   Run identifier
 * @param run_out  Output: run summary
 * @return         REG_OK on success, REG_ERR_NOT_FOUND if not exists
 */
RegistryError registry_get_run(const char *run_id, RunSummary *run_out);

/*
 * Transaction management for atomic operations.
 */
RegistryError registry_begin_transaction(void);
RegistryError registry_commit(void);
RegistryError registry_rollback(void);

#endif /* REGISTRY_H */
