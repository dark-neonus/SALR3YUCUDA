/*
 * test_integration.c - Integration test for the database engine
 *
 * Tests the full workflow:
 *   1. Initialize database
 *   2. Create a run
 *   3. Save snapshots
 *   4. Load and verify
 *   5. Delete run
 *   6. Cleanup
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "../include/db_engine.h"
#include "../../include/config.h"

#define TEST_DATA_DIR "/tmp/salr_test_data"

/* Simple test config */
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
    cfg->solver.tolerance = 1e-8;
    cfg->solver.max_iterations = 10000;

    strncpy(cfg->output_dir, "output", sizeof(cfg->output_dir) - 1);
    cfg->save_every = 100;
}

static void test_init_close(void) {
    printf("Test: init/close... ");

    assert(db_init(TEST_DATA_DIR) == DB_OK);
    assert(db_is_initialized() == 1);

    db_close();
    assert(db_is_initialized() == 0);

    printf("PASSED\n");
}

static void test_run_create(void) {
    printf("Test: run create... ");

    assert(db_init(TEST_DATA_DIR) == DB_OK);

    SimConfig cfg;
    init_test_config(&cfg);

    DbRun *run;
    assert(db_run_create(&cfg, &run) == DB_OK);

    const char *run_id = db_run_get_id(run);
    assert(run_id != NULL);
    assert(strlen(run_id) > 0);
    printf("(run_id=%s) ", run_id);

    const char *path = db_run_get_path(run);
    assert(path != NULL);

    db_run_close(run);
    db_close();

    printf("PASSED\n");
}

static void test_snapshot_save_load(void) {
    printf("Test: snapshot save/load... ");

    assert(db_init(TEST_DATA_DIR) == DB_OK);

    SimConfig cfg;
    init_test_config(&cfg);

    DbRun *run;
    assert(db_run_create(&cfg, &run) == DB_OK);

    const int nx = cfg.grid.nx;
    const int ny = cfg.grid.ny;
    const size_t N = (size_t)nx * ny;

    /* Create test density arrays */
    double *rho1 = malloc(N * sizeof(double));
    double *rho2 = malloc(N * sizeof(double));
    double *rho1_read = malloc(N * sizeof(double));
    double *rho2_read = malloc(N * sizeof(double));

    for (size_t i = 0; i < N; i++) {
        rho1[i] = cfg.rho1 + 0.1 * sin(i * 0.1);
        rho2[i] = cfg.rho2 + 0.05 * cos(i * 0.1);
    }

    /* Save snapshot */
    assert(db_snapshot_save(run, rho1, rho2, 100, 1e-5, 1e-6, &cfg) == DB_OK);

    /* List snapshots */
    int *iters;
    int count;
    assert(db_snapshot_list(run, &iters, &count) == DB_OK);
    assert(count == 1);
    assert(iters[0] == 100);
    free(iters);

    /* Load snapshot */
    SnapshotMeta meta;
    assert(db_snapshot_load(run, 100, rho1_read, rho2_read, &meta) == DB_OK);

    /* Verify metadata */
    assert(meta.iteration == 100);
    assert(fabs(meta.current_error - 1e-5) < 1e-10);
    assert(fabs(meta.delta_error - 1e-6) < 1e-15);
    assert(meta.nx == nx);
    assert(meta.ny == ny);

    /* Verify data */
    for (size_t i = 0; i < N; i++) {
        assert(fabs(rho1[i] - rho1_read[i]) < 1e-10);
        assert(fabs(rho2[i] - rho2_read[i]) < 1e-10);
    }

    free(rho1);
    free(rho2);
    free(rho1_read);
    free(rho2_read);

    db_run_close(run);
    db_close();

    printf("PASSED\n");
}

static void test_registry_operations(void) {
    printf("Test: registry operations... ");

    assert(db_init(TEST_DATA_DIR) == DB_OK);

    SimConfig cfg;
    init_test_config(&cfg);

    /* Create runs */
    DbRun *run1, *run2;
    assert(db_run_create(&cfg, &run1) == DB_OK);

    cfg.temperature = 3.0;  /* Different temperature */
    assert(db_run_create(&cfg, &run2) == DB_OK);

    const char *id1 = db_run_get_id(run1);
    const char *id2 = db_run_get_id(run2);
    char id1_copy[64], id2_copy[64];
    strncpy(id1_copy, id1, sizeof(id1_copy) - 1);
    strncpy(id2_copy, id2, sizeof(id2_copy) - 1);

    /* Set nickname */
    assert(db_registry_set_nickname(id1_copy, "TestRun1") == DB_OK);

    /* Query all runs */
    RunSummary *runs;
    int count;
    assert(db_registry_list(0, 0, 0, 0, &runs, &count) == DB_OK);
    assert(count >= 2);

    /* Verify nickname was set */
    int found = 0;
    for (int i = 0; i < count; i++) {
        if (strcmp(runs[i].run_id, id1_copy) == 0) {
            assert(strcmp(runs[i].nickname, "TestRun1") == 0);
            found = 1;
            break;
        }
    }
    assert(found);
    free(runs);

    /* Query with temperature filter */
    assert(db_registry_list(2.8, 3.1, 0, 0, &runs, &count) == DB_OK);
    assert(count >= 2);
    free(runs);

    db_run_close(run1);
    db_run_close(run2);

    /* Delete run1 */
    assert(db_registry_delete_run(id1_copy) == DB_OK);

    /* Verify deletion */
    assert(db_registry_list(0, 0, 0, 0, &runs, &count) == DB_OK);
    for (int i = 0; i < count; i++) {
        assert(strcmp(runs[i].run_id, id1_copy) != 0);
    }
    free(runs);

    /* Clean up run2 */
    assert(db_registry_delete_run(id2_copy) == DB_OK);

    db_close();

    printf("PASSED\n");
}

static void test_hyperslab_extraction(void) {
    printf("Test: hyperslab extraction... ");

    assert(db_init(TEST_DATA_DIR) == DB_OK);

    SimConfig cfg;
    init_test_config(&cfg);

    DbRun *run;
    assert(db_run_create(&cfg, &run) == DB_OK);

    const int nx = cfg.grid.nx;
    const int ny = cfg.grid.ny;
    const size_t N = (size_t)nx * ny;

    /* Create density with known pattern */
    double *rho1 = malloc(N * sizeof(double));
    double *rho2 = malloc(N * sizeof(double));

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            rho1[iy * nx + ix] = iy * 1000 + ix;  /* Unique value */
            rho2[iy * nx + ix] = iy * 2000 + ix;
        }
    }

    assert(db_snapshot_save(run, rho1, rho2, 0, 1.0, -1.0, &cfg) == DB_OK);

    /* Extract slice: row 16, columns 10-19 */
    double slice[10];
    assert(db_snapshot_extract_slice(run, 0, "rho1", 10, 10, 16, 1, slice) == DB_OK);

    /* Verify slice values */
    for (int i = 0; i < 10; i++) {
        double expected = 16 * 1000 + (10 + i);
        assert(fabs(slice[i] - expected) < 1e-10);
    }

    free(rho1);
    free(rho2);

    /* Cleanup */
    const char *id = db_run_get_id(run);
    char id_copy[64];
    strncpy(id_copy, id, sizeof(id_copy) - 1);
    db_run_close(run);
    assert(db_registry_delete_run(id_copy) == DB_OK);

    db_close();

    printf("PASSED\n");
}

int main(void) {
    printf("=== SALR Database Engine Integration Tests ===\n\n");

    test_init_close();
    test_run_create();
    test_snapshot_save_load();
    test_registry_operations();
    test_hyperslab_extraction();

    printf("\n=== All tests PASSED ===\n");
    return 0;
}
