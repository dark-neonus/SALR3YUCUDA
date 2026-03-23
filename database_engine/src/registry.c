/*
 * registry.c - SQLite registry implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/registry.h"
#include "../../include/config.h"

/* Global database connection */
static sqlite3 *g_db = NULL;

/* SQL statements */
static const char *SQL_CREATE_TABLE =
    "CREATE TABLE IF NOT EXISTS runs ("
    "    run_id         TEXT PRIMARY KEY,"
    "    nickname       TEXT DEFAULT NULL,"
    "    created_at     TEXT NOT NULL,"
    "    temperature    REAL NOT NULL,"
    "    rho1_bulk      REAL NOT NULL,"
    "    rho2_bulk      REAL NOT NULL,"
    "    nx             INTEGER NOT NULL,"
    "    ny             INTEGER NOT NULL,"
    "    boundary_mode  TEXT NOT NULL,"
    "    config_hash    TEXT NOT NULL,"
    "    snapshot_count INTEGER DEFAULT 0,"
    "    final_error    REAL DEFAULT NULL,"
    "    converged      INTEGER DEFAULT 0"
    ");";

static const char *SQL_CREATE_INDEX_TEMP =
    "CREATE INDEX IF NOT EXISTS idx_runs_temp ON runs(temperature);";

static const char *SQL_CREATE_INDEX_RHO =
    "CREATE INDEX IF NOT EXISTS idx_runs_rho ON runs(rho1_bulk, rho2_bulk);";

static const char *SQL_CREATE_INDEX_CREATED =
    "CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);";

/*
 * Convert BoundaryMode to string
 */
static const char *boundary_mode_to_str(BoundaryMode mode) {
    switch (mode) {
        case BC_PBC: return "PBC";
        case BC_W2:  return "W2";
        case BC_W4:  return "W4";
        default:     return "PBC";
    }
}

/*
 * Convert string to BoundaryMode
 */
static BoundaryMode str_to_boundary_mode(const char *s) {
    if (strcmp(s, "W2") == 0) return BC_W2;
    if (strcmp(s, "W4") == 0) return BC_W4;
    return BC_PBC;
}

RegistryError registry_init(const char *db_path) {
    if (g_db) {
        /* Already initialized */
        return REG_OK;
    }

    int rc = sqlite3_open(db_path, &g_db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "registry_init: cannot open '%s': %s\n",
                db_path, sqlite3_errmsg(g_db));
        sqlite3_close(g_db);
        g_db = NULL;
        return REG_ERR_OPEN;
    }

    /* Enable WAL mode for better concurrency */
    char *err_msg = NULL;
    rc = sqlite3_exec(g_db, "PRAGMA journal_mode=WAL;", NULL, NULL, &err_msg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "registry_init: WAL mode failed: %s\n", err_msg);
        sqlite3_free(err_msg);
    }

    /* Set busy timeout */
    sqlite3_exec(g_db, "PRAGMA busy_timeout=5000;", NULL, NULL, NULL);

    /* Create tables and indices */
    rc = sqlite3_exec(g_db, SQL_CREATE_TABLE, NULL, NULL, &err_msg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "registry_init: create table failed: %s\n", err_msg);
        sqlite3_free(err_msg);
        sqlite3_close(g_db);
        g_db = NULL;
        return REG_ERR_EXEC;
    }

    sqlite3_exec(g_db, SQL_CREATE_INDEX_TEMP, NULL, NULL, NULL);
    sqlite3_exec(g_db, SQL_CREATE_INDEX_RHO, NULL, NULL, NULL);
    sqlite3_exec(g_db, SQL_CREATE_INDEX_CREATED, NULL, NULL, NULL);

    return REG_OK;
}

void registry_close(void) {
    if (g_db) {
        sqlite3_close(g_db);
        g_db = NULL;
    }
}

sqlite3 *registry_get_handle(void) {
    return g_db;
}

RegistryError registry_begin_transaction(void) {
    if (!g_db) return REG_ERR_OPEN;
    int rc = sqlite3_exec(g_db, "BEGIN EXCLUSIVE", NULL, NULL, NULL);
    return (rc == SQLITE_OK) ? REG_OK : REG_ERR_EXEC;
}

RegistryError registry_commit(void) {
    if (!g_db) return REG_ERR_OPEN;
    int rc = sqlite3_exec(g_db, "COMMIT", NULL, NULL, NULL);
    return (rc == SQLITE_OK) ? REG_OK : REG_ERR_EXEC;
}

RegistryError registry_rollback(void) {
    if (!g_db) return REG_ERR_OPEN;
    int rc = sqlite3_exec(g_db, "ROLLBACK", NULL, NULL, NULL);
    return (rc == SQLITE_OK) ? REG_OK : REG_ERR_EXEC;
}

RegistryError registry_insert_run(const char *run_id,
                                  const char *created_at,
                                  const struct SimConfig *cfg,
                                  const char *config_hash) {
    if (!g_db || !run_id || !created_at || !cfg) {
        return REG_ERR_OPEN;
    }

    const char *sql =
        "INSERT INTO runs (run_id, created_at, temperature, rho1_bulk, rho2_bulk, "
        "nx, ny, boundary_mode, config_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);";

    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "registry_insert_run: prepare failed: %s\n",
                sqlite3_errmsg(g_db));
        return REG_ERR_PREPARE;
    }

    sqlite3_bind_text(stmt, 1, run_id, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, created_at, -1, SQLITE_STATIC);
    sqlite3_bind_double(stmt, 3, cfg->temperature);
    sqlite3_bind_double(stmt, 4, cfg->rho1);
    sqlite3_bind_double(stmt, 5, cfg->rho2);
    sqlite3_bind_int(stmt, 6, cfg->grid.nx);
    sqlite3_bind_int(stmt, 7, cfg->grid.ny);
    sqlite3_bind_text(stmt, 8, boundary_mode_to_str(cfg->boundary_mode), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 9, config_hash ? config_hash : "", -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        fprintf(stderr, "registry_insert_run: insert failed: %s\n",
                sqlite3_errmsg(g_db));
        return REG_ERR_EXEC;
    }

    return REG_OK;
}

RegistryError registry_update_status(const char *run_id,
                                     int snapshot_count,
                                     double final_error,
                                     int converged) {
    if (!g_db || !run_id) return REG_ERR_OPEN;

    const char *sql =
        "UPDATE runs SET snapshot_count = ?, final_error = ?, converged = ? "
        "WHERE run_id = ?;";

    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) return REG_ERR_PREPARE;

    sqlite3_bind_int(stmt, 1, snapshot_count);
    sqlite3_bind_double(stmt, 2, final_error);
    sqlite3_bind_int(stmt, 3, converged);
    sqlite3_bind_text(stmt, 4, run_id, -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return (rc == SQLITE_DONE) ? REG_OK : REG_ERR_EXEC;
}

RegistryError registry_set_nickname(const char *run_id, const char *nickname) {
    if (!g_db || !run_id) return REG_ERR_OPEN;

    const char *sql = "UPDATE runs SET nickname = ? WHERE run_id = ?;";

    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) return REG_ERR_PREPARE;

    if (nickname) {
        sqlite3_bind_text(stmt, 1, nickname, -1, SQLITE_STATIC);
    } else {
        sqlite3_bind_null(stmt, 1);
    }
    sqlite3_bind_text(stmt, 2, run_id, -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return (rc == SQLITE_DONE) ? REG_OK : REG_ERR_EXEC;
}

RegistryError registry_delete_run(const char *run_id) {
    if (!g_db || !run_id) return REG_ERR_OPEN;

    const char *sql = "DELETE FROM runs WHERE run_id = ?;";

    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) return REG_ERR_PREPARE;

    sqlite3_bind_text(stmt, 1, run_id, -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return (rc == SQLITE_DONE) ? REG_OK : REG_ERR_EXEC;
}

int registry_run_exists(const char *run_id) {
    if (!g_db || !run_id) return 0;

    const char *sql = "SELECT 1 FROM runs WHERE run_id = ? LIMIT 1;";

    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) return 0;

    sqlite3_bind_text(stmt, 1, run_id, -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return (rc == SQLITE_ROW) ? 1 : 0;
}

/*
 * Helper to fill RunSummary from a statement row
 */
static void fill_run_summary(sqlite3_stmt *stmt, RunSummary *run) {
    const char *text;

    text = (const char *)sqlite3_column_text(stmt, 0);
    strncpy(run->run_id, text ? text : "", sizeof(run->run_id) - 1);
    run->run_id[sizeof(run->run_id) - 1] = '\0';

    text = (const char *)sqlite3_column_text(stmt, 1);
    if (text) {
        strncpy(run->nickname, text, sizeof(run->nickname) - 1);
        run->nickname[sizeof(run->nickname) - 1] = '\0';
    } else {
        run->nickname[0] = '\0';
    }

    text = (const char *)sqlite3_column_text(stmt, 2);
    strncpy(run->created_at, text ? text : "", sizeof(run->created_at) - 1);
    run->created_at[sizeof(run->created_at) - 1] = '\0';

    run->temperature = sqlite3_column_double(stmt, 3);
    run->rho1_bulk = sqlite3_column_double(stmt, 4);
    run->rho2_bulk = sqlite3_column_double(stmt, 5);
    run->nx = sqlite3_column_int(stmt, 6);
    run->ny = sqlite3_column_int(stmt, 7);

    text = (const char *)sqlite3_column_text(stmt, 8);
    strncpy(run->boundary_mode, text ? text : "PBC", sizeof(run->boundary_mode) - 1);
    run->boundary_mode[sizeof(run->boundary_mode) - 1] = '\0';

    text = (const char *)sqlite3_column_text(stmt, 9);
    strncpy(run->config_hash, text ? text : "", sizeof(run->config_hash) - 1);
    run->config_hash[sizeof(run->config_hash) - 1] = '\0';

    run->snapshot_count = sqlite3_column_int(stmt, 10);
    run->final_error = sqlite3_column_double(stmt, 11);
    run->converged = sqlite3_column_int(stmt, 12);
}

RegistryError registry_get_run(const char *run_id, RunSummary *run_out) {
    if (!g_db || !run_id || !run_out) return REG_ERR_OPEN;

    const char *sql =
        "SELECT run_id, nickname, created_at, temperature, rho1_bulk, rho2_bulk, "
        "nx, ny, boundary_mode, config_hash, snapshot_count, final_error, converged "
        "FROM runs WHERE run_id = ?;";

    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) return REG_ERR_PREPARE;

    sqlite3_bind_text(stmt, 1, run_id, -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        return REG_ERR_NOT_FOUND;
    }

    fill_run_summary(stmt, run_out);
    sqlite3_finalize(stmt);

    return REG_OK;
}

RegistryError registry_query_runs(const RegistryFilter *filter,
                                  RunSummary **runs_out,
                                  int *count_out) {
    if (!g_db || !runs_out || !count_out) return REG_ERR_OPEN;

    *runs_out = NULL;
    *count_out = 0;

    /* Build query with optional filters */
    char sql[1024];
    int offset = 0;

    offset += snprintf(sql + offset, sizeof(sql) - offset,
        "SELECT run_id, nickname, created_at, temperature, rho1_bulk, rho2_bulk, "
        "nx, ny, boundary_mode, config_hash, snapshot_count, final_error, converged "
        "FROM runs WHERE 1=1");

    if (filter) {
        if (filter->use_temp_filter) {
            offset += snprintf(sql + offset, sizeof(sql) - offset,
                " AND temperature >= %f AND temperature <= %f",
                filter->temp_min, filter->temp_max);
        }
        if (filter->use_rho1_filter) {
            offset += snprintf(sql + offset, sizeof(sql) - offset,
                " AND rho1_bulk >= %f AND rho1_bulk <= %f",
                filter->rho1_min, filter->rho1_max);
        }
        if (filter->use_rho2_filter) {
            offset += snprintf(sql + offset, sizeof(sql) - offset,
                " AND rho2_bulk >= %f AND rho2_bulk <= %f",
                filter->rho2_min, filter->rho2_max);
        }
    }

    offset += snprintf(sql + offset, sizeof(sql) - offset, " ORDER BY created_at DESC;");

    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "registry_query_runs: prepare failed: %s\n",
                sqlite3_errmsg(g_db));
        return REG_ERR_PREPARE;
    }

    /* Count rows first */
    int count = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        count++;
    }
    sqlite3_reset(stmt);

    if (count == 0) {
        sqlite3_finalize(stmt);
        return REG_OK;
    }

    /* Allocate result array */
    RunSummary *runs = malloc(count * sizeof(RunSummary));
    if (!runs) {
        sqlite3_finalize(stmt);
        return REG_ERR_ALLOC;
    }

    /* Fill results */
    int i = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW && i < count) {
        fill_run_summary(stmt, &runs[i]);
        i++;
    }

    sqlite3_finalize(stmt);

    *runs_out = runs;
    *count_out = count;

    return REG_OK;
}
