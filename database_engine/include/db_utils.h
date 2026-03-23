/*
 * db_utils.h - Utility functions for the database engine
 *
 * Provides:
 *   - Run ID generation (timestamp + hash)
 *   - ISO8601 timestamp formatting
 *   - Path utilities (recursive mkdir, path joining)
 *   - Simple hash functions
 */

#ifndef DB_UTILS_H
#define DB_UTILS_H

#include <stddef.h>
#include <time.h>

/* Maximum lengths for various strings */
#define DB_RUN_ID_MAX      64
#define DB_TIMESTAMP_MAX   32
#define DB_PATH_MAX       512
#define DB_HASH_LEN         8   /* Truncated hash length */

/*
 * db_generate_run_id - Generate a unique run ID.
 *
 * Format: session_{YYYYMMDD_HHMMSS}_{hash8}
 * The hash is derived from config parameters + random salt.
 *
 * @param out       Output buffer (at least DB_RUN_ID_MAX bytes)
 * @param out_size  Size of output buffer
 * @param temp      Temperature value (for hash)
 * @param rho1      Bulk density species 1 (for hash)
 * @param rho2      Bulk density species 2 (for hash)
 * @param nx        Grid size X (for hash)
 * @param ny        Grid size Y (for hash)
 * @return          0 on success, -1 on error
 */
int db_generate_run_id(char *out, size_t out_size,
                       double temp, double rho1, double rho2,
                       int nx, int ny);

/*
 * db_format_timestamp - Format current time as ISO8601 string.
 *
 * @param out       Output buffer (at least DB_TIMESTAMP_MAX bytes)
 * @param out_size  Size of output buffer
 * @return          0 on success, -1 on error
 */
int db_format_timestamp(char *out, size_t out_size);

/*
 * db_format_timestamp_compact - Format time as YYYYMMDD_HHMMSS.
 *
 * @param out       Output buffer (at least 16 bytes)
 * @param out_size  Size of output buffer
 * @param t         Time to format (or NULL for current time)
 * @return          0 on success, -1 on error
 */
int db_format_timestamp_compact(char *out, size_t out_size, const time_t *t);

/*
 * db_ensure_directory - Create directory if it doesn't exist.
 *
 * Creates parent directories as needed (like mkdir -p).
 *
 * @param path  Path to create
 * @return      0 on success, -1 on error
 */
int db_ensure_directory(const char *path);

/*
 * db_path_join - Join path components safely.
 *
 * @param out       Output buffer
 * @param out_size  Size of output buffer
 * @param base      Base path
 * @param name      Name to append
 * @return          0 on success, -1 on error (truncation)
 */
int db_path_join(char *out, size_t out_size, const char *base, const char *name);

/*
 * db_file_exists - Check if a file or directory exists.
 *
 * @param path  Path to check
 * @return      1 if exists, 0 if not
 */
int db_file_exists(const char *path);

/*
 * db_remove_directory - Recursively remove a directory and its contents.
 *
 * WARNING: This is destructive and cannot be undone.
 *
 * @param path  Directory to remove
 * @return      0 on success, -1 on error
 */
int db_remove_directory(const char *path);

/*
 * db_hash_simple - Compute a simple hash of input data.
 *
 * Uses djb2 algorithm, returns 8 hex chars.
 *
 * @param data      Input data
 * @param len       Length of input
 * @param out       Output buffer (at least 9 bytes for null terminator)
 */
void db_hash_simple(const void *data, size_t len, char *out);

#endif /* DB_UTILS_H */
