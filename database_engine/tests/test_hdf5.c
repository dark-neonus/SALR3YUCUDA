/*
 * test_hdf5.c - Unit tests for HDF5 I/O operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "../include/hdf5_io.h"
#include "../include/db_utils.h"
#include "../../include/config.h"

#define TEST_FILE "/tmp/test_snapshot.h5"

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

static void test_write_read_snapshot(void) {
    printf("Test: write/read snapshot... ");

    SimConfig cfg;
    init_test_config(&cfg);

    const int nx = cfg.grid.nx;
    const int ny = cfg.grid.ny;
    const size_t N = (size_t)nx * ny;

    double *rho1 = malloc(N * sizeof(double));
    double *rho2 = malloc(N * sizeof(double));

    /* Initialize with known pattern */
    for (size_t i = 0; i < N; i++) {
        rho1[i] = 0.4 + 0.1 * sin(i * 0.1);
        rho2[i] = 0.2 + 0.05 * cos(i * 0.1);
    }

    /* Write snapshot */
    HDF5Error err = hdf5_write_snapshot(TEST_FILE, rho1, rho2, nx, ny,
                                        100, 1e-5, 1e-6, &cfg);
    assert(err == HDF5_OK);

    /* Read back */
    double *rho1_read = malloc(N * sizeof(double));
    double *rho2_read = malloc(N * sizeof(double));
    SnapshotMeta meta;

    err = hdf5_read_snapshot(TEST_FILE, rho1_read, rho2_read, &meta);
    assert(err == HDF5_OK);

    /* Verify metadata */
    assert(meta.iteration == 100);
    assert(fabs(meta.current_error - 1e-5) < 1e-10);
    assert(fabs(meta.delta_error - 1e-6) < 1e-15);
    assert(meta.nx == nx);
    assert(meta.ny == ny);
    assert(fabs(meta.temperature - 2.9) < 1e-10);

    /* Verify data */
    for (size_t i = 0; i < N; i++) {
        assert(fabs(rho1[i] - rho1_read[i]) < 1e-10);
        assert(fabs(rho2[i] - rho2_read[i]) < 1e-10);
    }

    free(rho1);
    free(rho2);
    free(rho1_read);
    free(rho2_read);

    /* Cleanup */
    remove(TEST_FILE);

    printf("PASSED\n");
}

static void test_read_metadata(void) {
    printf("Test: read metadata only... ");

    SimConfig cfg;
    init_test_config(&cfg);

    const int nx = cfg.grid.nx;
    const int ny = cfg.grid.ny;
    const size_t N = (size_t)nx * ny;

    double *rho1 = malloc(N * sizeof(double));
    double *rho2 = malloc(N * sizeof(double));

    for (size_t i = 0; i < N; i++) {
        rho1[i] = 0.4;
        rho2[i] = 0.2;
    }

    hdf5_write_snapshot(TEST_FILE, rho1, rho2, nx, ny, 500, 1e-8, 5e-9, &cfg);

    /* Read only metadata */
    SnapshotMeta meta;
    HDF5Error err = hdf5_read_metadata(TEST_FILE, &meta);
    assert(err == HDF5_OK);

    assert(meta.iteration == 500);
    assert(fabs(meta.current_error - 1e-8) < 1e-15);
    assert(fabs(meta.delta_error - 5e-9) < 1e-15);

    free(rho1);
    free(rho2);
    remove(TEST_FILE);

    printf("PASSED\n");
}

static void test_hyperslab(void) {
    printf("Test: hyperslab extraction... ");

    SimConfig cfg;
    init_test_config(&cfg);

    const int nx = cfg.grid.nx;
    const int ny = cfg.grid.ny;
    const size_t N = (size_t)nx * ny;

    double *rho1 = malloc(N * sizeof(double));
    double *rho2 = malloc(N * sizeof(double));

    /* Create pattern with unique values */
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            rho1[iy * nx + ix] = iy * 1000 + ix;
            rho2[iy * nx + ix] = iy * 2000 + ix;
        }
    }

    hdf5_write_snapshot(TEST_FILE, rho1, rho2, nx, ny, 0, 1.0, -1.0, &cfg);

    /* Extract slice: row 16, columns 10-14 */
    double slice[5];
    HDF5Error err = hdf5_read_hyperslab(TEST_FILE, "rho1", 10, 5, 16, 1, slice);
    assert(err == HDF5_OK);

    for (int i = 0; i < 5; i++) {
        double expected = 16 * 1000 + (10 + i);
        assert(fabs(slice[i] - expected) < 1e-10);
    }

    free(rho1);
    free(rho2);
    remove(TEST_FILE);

    printf("PASSED\n");
}

static void test_grid_validation(void) {
    printf("Test: grid validation... ");

    SimConfig cfg;
    init_test_config(&cfg);

    const int nx = cfg.grid.nx;
    const int ny = cfg.grid.ny;
    const size_t N = (size_t)nx * ny;

    double *rho1 = malloc(N * sizeof(double));
    double *rho2 = malloc(N * sizeof(double));

    for (size_t i = 0; i < N; i++) {
        rho1[i] = 0.4;
        rho2[i] = 0.2;
    }

    hdf5_write_snapshot(TEST_FILE, rho1, rho2, nx, ny, 0, 1.0, -1.0, &cfg);

    /* Validate with matching config */
    HDF5Error err = hdf5_validate_grid(TEST_FILE, &cfg);
    assert(err == HDF5_OK);

    /* Validate with mismatched config */
    SimConfig cfg2 = cfg;
    cfg2.grid.nx = 64;  /* Different size */
    err = hdf5_validate_grid(TEST_FILE, &cfg2);
    assert(err == HDF5_ERR_MISMATCH);

    free(rho1);
    free(rho2);
    remove(TEST_FILE);

    printf("PASSED\n");
}

int main(void) {
    printf("=== HDF5 I/O Unit Tests ===\n\n");

    test_write_read_snapshot();
    test_read_metadata();
    test_hyperslab();
    test_grid_validation();

    printf("\n=== All HDF5 tests PASSED ===\n");
    return 0;
}
