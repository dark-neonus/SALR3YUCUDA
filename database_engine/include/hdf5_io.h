/*
 * hdf5_io.h - HDF5 snapshot I/O operations
 *
 * Provides:
 *   - Writing density snapshots with compression
 *   - Reading full snapshots or partial hyperslabs
 *   - Metadata extraction without loading full data
 *   - Grid validation for state resumption
 */

#ifndef HDF5_IO_H
#define HDF5_IO_H

#include <hdf5.h>

/* Forward declarations */
struct SimConfig;

/* Error codes */
typedef enum {
    HDF5_OK           =  0,
    HDF5_ERR_FILE     = -1,  /* File open/create error */
    HDF5_ERR_DATASET  = -2,  /* Dataset operations error */
    HDF5_ERR_ATTR     = -3,  /* Attribute operations error */
    HDF5_ERR_SPACE    = -4,  /* Dataspace error */
    HDF5_ERR_MISMATCH = -5,  /* Grid size mismatch */
    HDF5_ERR_ALLOC    = -6   /* Memory allocation error */
} HDF5Error;

/* Compression level (GZIP deflate) */
#define HDF5_COMPRESSION_LEVEL 5

/* Chunk size for 2D datasets */
#define HDF5_CHUNK_NX 64
#define HDF5_CHUNK_NY 64

/* Snapshot metadata structure */
typedef struct {
    int    iteration;
    double current_error;
    double delta_error;     /* Error change from previous iteration */
    double temperature;
    double rho1_bulk;
    double rho2_bulk;
    int    nx;
    int    ny;
    double Lx;
    double Ly;
    double dx;
    double dy;
    char   boundary_mode[8];
    double xi1;
    double xi2;
    double cutoff_radius;
    char   created_at[32];
} SnapshotMeta;

/*
 * hdf5_write_snapshot - Write density arrays to HDF5 file.
 *
 * Creates chunked, compressed datasets with metadata attributes.
 *
 * @param filepath    Path to the HDF5 file to create
 * @param rho1        Density array for species 1 (nx * ny, row-major)
 * @param rho2        Density array for species 2 (nx * ny, row-major)
 * @param nx          Grid size in X
 * @param ny          Grid size in Y
 * @param iteration   Current iteration number
 * @param error       Current convergence error
 * @param delta_error Error change from previous iteration
 * @param cfg         Simulation configuration (for metadata)
 * @return            HDF5_OK on success
 */
HDF5Error hdf5_write_snapshot(const char *filepath,
                              const double *rho1,
                              const double *rho2,
                              int nx, int ny,
                              int iteration,
                              double error,
                              double delta_error,
                              const struct SimConfig *cfg);

/*
 * hdf5_read_snapshot - Read density arrays from HDF5 file.
 *
 * @param filepath   Path to the HDF5 file
 * @param rho1       Output: density array for species 1 (pre-allocated)
 * @param rho2       Output: density array for species 2 (pre-allocated)
 * @param meta       Output: metadata (can be NULL)
 * @return           HDF5_OK on success
 */
HDF5Error hdf5_read_snapshot(const char *filepath,
                             double *rho1,
                             double *rho2,
                             SnapshotMeta *meta);

/*
 * hdf5_read_metadata - Read only metadata from HDF5 file.
 *
 * Does not load density arrays - fast operation for listing/filtering.
 *
 * @param filepath   Path to the HDF5 file
 * @param meta       Output: metadata
 * @return           HDF5_OK on success
 */
HDF5Error hdf5_read_metadata(const char *filepath, SnapshotMeta *meta);

/*
 * hdf5_read_hyperslab - Read partial data from a dataset.
 *
 * Extracts a rectangular region without loading the full array.
 *
 * @param filepath   Path to the HDF5 file
 * @param dataset    Dataset name ("rho1" or "rho2")
 * @param x_start    Starting X index
 * @param x_count    Number of X elements
 * @param y_start    Starting Y index
 * @param y_count    Number of Y elements
 * @param data       Output: pre-allocated array (x_count * y_count)
 * @return           HDF5_OK on success
 */
HDF5Error hdf5_read_hyperslab(const char *filepath,
                              const char *dataset,
                              int x_start, int x_count,
                              int y_start, int y_count,
                              double *data);

/*
 * hdf5_validate_grid - Validate grid dimensions match config.
 *
 * Used before state resumption to ensure compatibility.
 *
 * @param filepath   Path to the HDF5 file
 * @param cfg        Simulation configuration to validate against
 * @return           HDF5_OK if match, HDF5_ERR_MISMATCH if not
 */
HDF5Error hdf5_validate_grid(const char *filepath,
                             const struct SimConfig *cfg);

/*
 * hdf5_get_grid_size - Read grid dimensions from snapshot.
 *
 * @param filepath   Path to the HDF5 file
 * @param nx_out     Output: grid size X
 * @param ny_out     Output: grid size Y
 * @return           HDF5_OK on success
 */
HDF5Error hdf5_get_grid_size(const char *filepath, int *nx_out, int *ny_out);

#endif /* HDF5_IO_H */
