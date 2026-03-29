/*
 * hdf5_io.c - HDF5 snapshot I/O implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/hdf5_io.h"
#include "../include/db_utils.h"
#include "../../include/config.h"

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
 * Helper to write a string attribute
 */
static herr_t write_string_attr(hid_t loc, const char *name, const char *value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t type = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, strlen(value) + 1);
    H5Tset_strpad(type, H5T_STR_NULLTERM);

    hid_t attr = H5Acreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Awrite(attr, type, value);

    H5Aclose(attr);
    H5Tclose(type);
    H5Sclose(space);

    return status;
}

/*
 * Helper to write an int attribute
 */
static herr_t write_int_attr(hid_t loc, const char *name, int value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Awrite(attr, H5T_NATIVE_INT, &value);
    H5Aclose(attr);
    H5Sclose(space);
    return status;
}

/*
 * Helper to write a double attribute
 */
static herr_t write_double_attr(hid_t loc, const char *name, double value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attr);
    H5Sclose(space);
    return status;
}

/*
 * Helper to read a string attribute
 */
static herr_t read_string_attr(hid_t loc, const char *name, char *out, size_t out_size) {
    if (!H5Aexists(loc, name)) {
        out[0] = '\0';
        return 0;
    }

    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) return -1;

    hid_t type = H5Aget_type(attr);
    size_t size = H5Tget_size(type);

    char *buf = malloc(size + 1);
    if (!buf) {
        H5Tclose(type);
        H5Aclose(attr);
        return -1;
    }

    herr_t status = H5Aread(attr, type, buf);
    if (status >= 0) {
        buf[size] = '\0';
        strncpy(out, buf, out_size - 1);
        out[out_size - 1] = '\0';
    }

    free(buf);
    H5Tclose(type);
    H5Aclose(attr);

    return status;
}

/*
 * Helper to read an int attribute
 */
static herr_t read_int_attr(hid_t loc, const char *name, int *out) {
    if (!H5Aexists(loc, name)) {
        *out = 0;
        return 0;
    }

    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) return -1;

    herr_t status = H5Aread(attr, H5T_NATIVE_INT, out);
    H5Aclose(attr);

    return status;
}

/*
 * Helper to read a double attribute
 */
static herr_t read_double_attr(hid_t loc, const char *name, double *out) {
    if (!H5Aexists(loc, name)) {
        *out = 0.0;
        return 0;
    }

    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) return -1;

    herr_t status = H5Aread(attr, H5T_NATIVE_DOUBLE, out);
    H5Aclose(attr);

    return status;
}

/*
 * Create a chunked, compressed 2D dataset
 */
static hid_t create_compressed_dataset(hid_t file, const char *name,
                                        int nx, int ny) {
    hsize_t dims[2] = {(hsize_t)ny, (hsize_t)nx};  /* row-major: [y][x] */
    hsize_t chunk[2] = {
        (ny < HDF5_CHUNK_NY) ? (hsize_t)ny : HDF5_CHUNK_NY,
        (nx < HDF5_CHUNK_NX) ? (hsize_t)nx : HDF5_CHUNK_NX
    };

    hid_t space = H5Screate_simple(2, dims, NULL);
    if (space < 0) return -1;

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    if (dcpl < 0) {
        H5Sclose(space);
        return -1;
    }

    H5Pset_chunk(dcpl, 2, chunk);
    H5Pset_shuffle(dcpl);  /* Improves compression ratio */
    H5Pset_deflate(dcpl, HDF5_COMPRESSION_LEVEL);

    hid_t dset = H5Dcreate2(file, name, H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, dcpl, H5P_DEFAULT);

    H5Pclose(dcpl);
    H5Sclose(space);

    return dset;
}

HDF5Error hdf5_write_snapshot(const char *filepath,
                              const double *rho1,
                              const double *rho2,
                              int nx, int ny,
                              int iteration,
                              double error,
                              double delta_error,
                              const struct SimConfig *cfg) {
    if (!filepath || !rho1 || !rho2 || !cfg) {
        return HDF5_ERR_FILE;
    }

    /* Create file */
    hid_t file = H5Fcreate(filepath, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        fprintf(stderr, "hdf5_write_snapshot: cannot create '%s'\n", filepath);
        return HDF5_ERR_FILE;
    }

    /* Write rho1 dataset */
    hid_t dset1 = create_compressed_dataset(file, "rho1", nx, ny);
    if (dset1 < 0) {
        H5Fclose(file);
        return HDF5_ERR_DATASET;
    }
    H5Dwrite(dset1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rho1);
    H5Dclose(dset1);

    /* Write rho2 dataset */
    hid_t dset2 = create_compressed_dataset(file, "rho2", nx, ny);
    if (dset2 < 0) {
        H5Fclose(file);
        return HDF5_ERR_DATASET;
    }
    H5Dwrite(dset2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rho2);
    H5Dclose(dset2);

    /* Write metadata attributes to root group */
    hid_t root = H5Gopen2(file, "/", H5P_DEFAULT);

    write_int_attr(root, "iteration", iteration);
    write_double_attr(root, "current_error", error);
    write_double_attr(root, "delta_error", delta_error);
    write_double_attr(root, "temperature", cfg->temperature);
    write_double_attr(root, "rho1_bulk", cfg->rho1);
    write_double_attr(root, "rho2_bulk", cfg->rho2);
    write_int_attr(root, "nx", cfg->grid.nx);
    write_int_attr(root, "ny", cfg->grid.ny);
    write_double_attr(root, "Lx", cfg->grid.Lx);
    write_double_attr(root, "Ly", cfg->grid.Ly);
    write_double_attr(root, "dx", cfg->grid.dx);
    write_double_attr(root, "dy", cfg->grid.dy);
    write_string_attr(root, "boundary_mode", boundary_mode_to_str(cfg->boundary_mode));
    write_double_attr(root, "xi1", cfg->solver.xi1);
    write_double_attr(root, "xi2", cfg->solver.xi2);
    write_double_attr(root, "cutoff_radius", cfg->potential.cutoff_radius);

    /* Write creation timestamp */
    char ts[DB_TIMESTAMP_MAX];
    db_format_timestamp(ts, sizeof(ts));
    write_string_attr(root, "created_at", ts);

    H5Gclose(root);
    H5Fclose(file);

    return HDF5_OK;
}

HDF5Error hdf5_read_snapshot(const char *filepath,
                             double *rho1,
                             double *rho2,
                             SnapshotMeta *meta) {
    if (!filepath || !rho1 || !rho2) {
        return HDF5_ERR_FILE;
    }

    /* Open file */
    hid_t file = H5Fopen(filepath, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        fprintf(stderr, "hdf5_read_snapshot: cannot open '%s'\n", filepath);
        return HDF5_ERR_FILE;
    }

    /* Read rho1 dataset */
    hid_t dset1 = H5Dopen2(file, "rho1", H5P_DEFAULT);
    if (dset1 < 0) {
        H5Fclose(file);
        return HDF5_ERR_DATASET;
    }
    H5Dread(dset1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rho1);
    H5Dclose(dset1);

    /* Read rho2 dataset */
    hid_t dset2 = H5Dopen2(file, "rho2", H5P_DEFAULT);
    if (dset2 < 0) {
        H5Fclose(file);
        return HDF5_ERR_DATASET;
    }
    H5Dread(dset2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rho2);
    H5Dclose(dset2);

    /* Read metadata if requested */
    if (meta) {
        /* Initialize struct to zero to ensure all fields have valid defaults */
        memset(meta, 0, sizeof(*meta));

        hid_t root = H5Gopen2(file, "/", H5P_DEFAULT);

        read_int_attr(root, "iteration", &meta->iteration);
        read_double_attr(root, "current_error", &meta->current_error);
        read_double_attr(root, "delta_error", &meta->delta_error);
        read_double_attr(root, "temperature", &meta->temperature);
        read_double_attr(root, "rho1_bulk", &meta->rho1_bulk);
        read_double_attr(root, "rho2_bulk", &meta->rho2_bulk);
        read_int_attr(root, "nx", &meta->nx);
        read_int_attr(root, "ny", &meta->ny);
        read_double_attr(root, "Lx", &meta->Lx);
        read_double_attr(root, "Ly", &meta->Ly);
        read_double_attr(root, "dx", &meta->dx);
        read_double_attr(root, "dy", &meta->dy);
        read_string_attr(root, "boundary_mode", meta->boundary_mode, sizeof(meta->boundary_mode));
        read_double_attr(root, "xi1", &meta->xi1);
        read_double_attr(root, "xi2", &meta->xi2);
        read_double_attr(root, "cutoff_radius", &meta->cutoff_radius);
        read_string_attr(root, "created_at", meta->created_at, sizeof(meta->created_at));

        H5Gclose(root);
    }

    H5Fclose(file);

    return HDF5_OK;
}

HDF5Error hdf5_read_metadata(const char *filepath, SnapshotMeta *meta) {
    if (!filepath || !meta) {
        return HDF5_ERR_FILE;
    }

    /* Initialize struct to zero to ensure all fields have valid defaults */
    memset(meta, 0, sizeof(*meta));

    hid_t file = H5Fopen(filepath, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        return HDF5_ERR_FILE;
    }

    hid_t root = H5Gopen2(file, "/", H5P_DEFAULT);

    read_int_attr(root, "iteration", &meta->iteration);
    read_double_attr(root, "current_error", &meta->current_error);
    read_double_attr(root, "delta_error", &meta->delta_error);
    read_double_attr(root, "temperature", &meta->temperature);
    read_double_attr(root, "rho1_bulk", &meta->rho1_bulk);
    read_double_attr(root, "rho2_bulk", &meta->rho2_bulk);
    read_int_attr(root, "nx", &meta->nx);
    read_int_attr(root, "ny", &meta->ny);
    read_double_attr(root, "Lx", &meta->Lx);
    read_double_attr(root, "Ly", &meta->Ly);
    read_double_attr(root, "dx", &meta->dx);
    read_double_attr(root, "dy", &meta->dy);
    read_string_attr(root, "boundary_mode", meta->boundary_mode, sizeof(meta->boundary_mode));
    read_double_attr(root, "xi1", &meta->xi1);
    read_double_attr(root, "xi2", &meta->xi2);
    read_double_attr(root, "cutoff_radius", &meta->cutoff_radius);
    read_string_attr(root, "created_at", meta->created_at, sizeof(meta->created_at));

    H5Gclose(root);
    H5Fclose(file);

    return HDF5_OK;
}

HDF5Error hdf5_read_hyperslab(const char *filepath,
                              const char *dataset,
                              int x_start, int x_count,
                              int y_start, int y_count,
                              double *data) {
    if (!filepath || !dataset || !data) {
        return HDF5_ERR_FILE;
    }

    hid_t file = H5Fopen(filepath, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        return HDF5_ERR_FILE;
    }

    hid_t dset = H5Dopen2(file, dataset, H5P_DEFAULT);
    if (dset < 0) {
        H5Fclose(file);
        return HDF5_ERR_DATASET;
    }

    hid_t fspace = H5Dget_space(dset);

    /* Define hyperslab in file space */
    hsize_t start[2] = {(hsize_t)y_start, (hsize_t)x_start};
    hsize_t count[2] = {(hsize_t)y_count, (hsize_t)x_count};

    herr_t status = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
    if (status < 0) {
        H5Sclose(fspace);
        H5Dclose(dset);
        H5Fclose(file);
        return HDF5_ERR_SPACE;
    }

    /* Create memory space */
    hid_t mspace = H5Screate_simple(2, count, NULL);

    /* Read data */
    status = H5Dread(dset, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, data);

    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Dclose(dset);
    H5Fclose(file);

    return (status >= 0) ? HDF5_OK : HDF5_ERR_DATASET;
}

HDF5Error hdf5_validate_grid(const char *filepath,
                             const struct SimConfig *cfg) {
    if (!filepath || !cfg) {
        return HDF5_ERR_FILE;
    }

    int nx, ny;
    HDF5Error err = hdf5_get_grid_size(filepath, &nx, &ny);
    if (err != HDF5_OK) {
        return err;
    }

    if (nx != cfg->grid.nx || ny != cfg->grid.ny) {
        fprintf(stderr, "hdf5_validate_grid: grid mismatch (%dx%d vs %dx%d)\n",
                nx, ny, cfg->grid.nx, cfg->grid.ny);
        return HDF5_ERR_MISMATCH;
    }

    return HDF5_OK;
}

HDF5Error hdf5_get_grid_size(const char *filepath, int *nx_out, int *ny_out) {
    if (!filepath || !nx_out || !ny_out) {
        return HDF5_ERR_FILE;
    }

    hid_t file = H5Fopen(filepath, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        return HDF5_ERR_FILE;
    }

    hid_t root = H5Gopen2(file, "/", H5P_DEFAULT);

    read_int_attr(root, "nx", nx_out);
    read_int_attr(root, "ny", ny_out);

    H5Gclose(root);
    H5Fclose(file);

    return HDF5_OK;
}
