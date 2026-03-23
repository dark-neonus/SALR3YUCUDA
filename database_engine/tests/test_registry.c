/*
 * test_registry.c - Unit tests for SQLite registry operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../include/registry.h"
#include "../include/db_utils.h"
#include "../../include/config.h"

#define TEST_DB_FILE "/tmp/test_registry.db"

static void init_test_config(SimConfig *cfg) {
    memset(cfg, 0, sizeof(SimConfig));
    cfg->grid.nx = 32;
    cfg->grid.ny = 32;
    cfg->grid.Lx = 6.4;
    cfg->grid.Ly = 6.4;
    cfg->grid.dx = 0.2;
    cfg->grid.dy = 0.2;
    cfg->temperature = 2.9;
    cfg->rho1 = 0.4;
    cfg->rho2 = 0.2;
    cfg->boundary_mode = BC_PBC;
    cfg->potential.cutoff_radius = 8.0;
    cfg->solver.xi1 = 0.2;
    cfg->solver.xi2 = 0.2;
}

static void cleanup_test_db(void) {
    remove(TEST_DB_FILE);
    /* Also try to remove WAL files */
    remove(TEST_DB_FILE "-wal");
    remove(TEST_DB_FILE "-shm");
}

static void test_init_close(void) {
    printf("Test: registry init/close... ");

    cleanup_test_db();

    RegistryError err = registry_init(TEST_DB_FILE);
    assert(err == REG_OK);

    registry_close();

    /* Second init should succeed (reopening) */
    err = registry_init(TEST_DB_FILE);
    assert(err == REG_OK);
    registry_close();

    cleanup_test_db();

    printf("PASSED\n");
}

static void test_insert_run(void) {
    printf("Test: insert run... ");

    cleanup_test_db();
    assert(registry_init(TEST_DB_FILE) == REG_OK);

    SimConfig cfg;
    init_test_config(&cfg);

    char created_at[DB_TIMESTAMP_MAX];
    db_format_timestamp(created_at, sizeof(created_at));

    RegistryError err = registry_insert_run("run_test_001", created_at, &cfg, "hash12345678");
    assert(err == REG_OK);

    /* Verify run exists */
    assert(registry_run_exists("run_test_001") == 1);
    assert(registry_run_exists("run_nonexistent") == 0);

    registry_close();
    cleanup_test_db();

    printf("PASSED\n");
}

static void test_query_runs(void) {
    printf("Test: query runs... ");

    cleanup_test_db();
    assert(registry_init(TEST_DB_FILE) == REG_OK);

    SimConfig cfg;
    init_test_config(&cfg);

    char created_at[DB_TIMESTAMP_MAX];
    db_format_timestamp(created_at, sizeof(created_at));

    /* Insert multiple runs with different temperatures */
    cfg.temperature = 2.5;
    registry_insert_run("run_temp_25", created_at, &cfg, "hash1");

    cfg.temperature = 3.0;
    registry_insert_run("run_temp_30", created_at, &cfg, "hash2");

    cfg.temperature = 3.5;
    registry_insert_run("run_temp_35", created_at, &cfg, "hash3");

    /* Query all runs */
    RegistryFilter filter = {0};
    RunSummary *runs;
    int count;

    RegistryError err = registry_query_runs(&filter, &runs, &count);
    assert(err == REG_OK);
    assert(count == 3);
    free(runs);

    /* Query with temperature filter */
    filter.use_temp_filter = 1;
    filter.temp_min = 2.8;
    filter.temp_max = 3.2;

    err = registry_query_runs(&filter, &runs, &count);
    assert(err == REG_OK);
    assert(count == 1);
    assert(runs[0].temperature >= 2.8 && runs[0].temperature <= 3.2);
    free(runs);

    registry_close();
    cleanup_test_db();

    printf("PASSED\n");
}

static void test_set_nickname(void) {
    printf("Test: set nickname... ");

    cleanup_test_db();
    assert(registry_init(TEST_DB_FILE) == REG_OK);

    SimConfig cfg;
    init_test_config(&cfg);

    char created_at[DB_TIMESTAMP_MAX];
    db_format_timestamp(created_at, sizeof(created_at));

    registry_insert_run("run_nick_test", created_at, &cfg, "hash_nick");

    /* Set nickname */
    RegistryError err = registry_set_nickname("run_nick_test", "MyFavoriteRun");
    assert(err == REG_OK);

    /* Query and verify */
    RegistryFilter filter = {0};
    RunSummary *runs;
    int count;

    err = registry_query_runs(&filter, &runs, &count);
    assert(err == REG_OK);
    assert(count == 1);
    assert(strcmp(runs[0].nickname, "MyFavoriteRun") == 0);
    free(runs);

    registry_close();
    cleanup_test_db();

    printf("PASSED\n");
}

static void test_update_status(void) {
    printf("Test: update status... ");

    cleanup_test_db();
    assert(registry_init(TEST_DB_FILE) == REG_OK);

    SimConfig cfg;
    init_test_config(&cfg);

    char created_at[DB_TIMESTAMP_MAX];
    db_format_timestamp(created_at, sizeof(created_at));

    registry_insert_run("run_status_test", created_at, &cfg, "hash_stat");

    /* Update status */
    RegistryError err = registry_update_status("run_status_test", 50, 1e-8, 1);
    assert(err == REG_OK);

    /* Query and verify */
    RegistryFilter filter = {0};
    RunSummary *runs;
    int count;

    err = registry_query_runs(&filter, &runs, &count);
    assert(err == REG_OK);
    assert(count == 1);
    assert(runs[0].snapshot_count == 50);
    assert(runs[0].converged == 1);
    free(runs);

    registry_close();
    cleanup_test_db();

    printf("PASSED\n");
}

static void test_delete_run(void) {
    printf("Test: delete run... ");

    cleanup_test_db();
    assert(registry_init(TEST_DB_FILE) == REG_OK);

    SimConfig cfg;
    init_test_config(&cfg);

    char created_at[DB_TIMESTAMP_MAX];
    db_format_timestamp(created_at, sizeof(created_at));

    registry_insert_run("run_to_delete", created_at, &cfg, "hash_del");

    assert(registry_run_exists("run_to_delete") == 1);

    /* Delete */
    RegistryError err = registry_delete_run("run_to_delete");
    assert(err == REG_OK);

    assert(registry_run_exists("run_to_delete") == 0);

    registry_close();
    cleanup_test_db();

    printf("PASSED\n");
}

static void test_transactions(void) {
    printf("Test: transactions... ");

    cleanup_test_db();
    assert(registry_init(TEST_DB_FILE) == REG_OK);

    SimConfig cfg;
    init_test_config(&cfg);

    char created_at[DB_TIMESTAMP_MAX];
    db_format_timestamp(created_at, sizeof(created_at));

    /* Begin transaction */
    registry_begin_transaction();

    registry_insert_run("run_txn_1", created_at, &cfg, "hash_txn1");
    registry_insert_run("run_txn_2", created_at, &cfg, "hash_txn2");

    /* Rollback */
    registry_rollback();

    /* Runs should not exist */
    assert(registry_run_exists("run_txn_1") == 0);
    assert(registry_run_exists("run_txn_2") == 0);

    /* Now with commit */
    registry_begin_transaction();
    registry_insert_run("run_txn_3", created_at, &cfg, "hash_txn3");
    registry_commit();

    assert(registry_run_exists("run_txn_3") == 1);

    registry_close();
    cleanup_test_db();

    printf("PASSED\n");
}

int main(void) {
    printf("=== SQLite Registry Unit Tests ===\n\n");

    test_init_close();
    test_insert_run();
    test_query_runs();
    test_set_nickname();
    test_update_status();
    test_delete_run();
    test_transactions();

    printf("\n=== All registry tests PASSED ===\n");
    return 0;
}
