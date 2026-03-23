/*
 * db_utils.c - Utility functions implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>

#include "../include/db_utils.h"

/*
 * djb2 hash function - fast and simple string hash
 */
static unsigned long djb2_hash(const unsigned char *data, size_t len) {
    unsigned long hash = 5381;
    for (size_t i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + data[i]; /* hash * 33 + c */
    }
    return hash;
}

void db_hash_simple(const void *data, size_t len, char *out) {
    unsigned long hash = djb2_hash(data, len);
    snprintf(out, 9, "%08lx", hash & 0xFFFFFFFF);
}

int db_format_timestamp(char *out, size_t out_size) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    if (!tm_info) return -1;

    size_t written = strftime(out, out_size, "%Y-%m-%dT%H:%M:%S", tm_info);
    return (written > 0) ? 0 : -1;
}

int db_format_timestamp_compact(char *out, size_t out_size, const time_t *t) {
    time_t now = t ? *t : time(NULL);
    struct tm *tm_info = localtime(&now);
    if (!tm_info) return -1;

    size_t written = strftime(out, out_size, "%Y%m%d_%H%M%S", tm_info);
    return (written > 0) ? 0 : -1;
}

int db_generate_run_id(char *out, size_t out_size,
                       double temp, double rho1, double rho2,
                       int nx, int ny) {
    /* Create hash input from parameters + random salt */
    char hash_input[256];
    unsigned int salt = (unsigned int)(time(NULL) ^ getpid());
    int n = snprintf(hash_input, sizeof(hash_input),
                     "%.6f:%.6f:%.6f:%d:%d:%u",
                     temp, rho1, rho2, nx, ny, salt);
    if (n < 0 || n >= (int)sizeof(hash_input)) return -1;

    /* Compute hash */
    char hash8[16];
    db_hash_simple(hash_input, strlen(hash_input), hash8);

    /* Format timestamp */
    char ts[20];
    if (db_format_timestamp_compact(ts, sizeof(ts), NULL) != 0) {
        return -1;
    }

    /* Combine into session ID */
    int written = snprintf(out, out_size, "session_%s_%s", ts, hash8);
    return (written > 0 && (size_t)written < out_size) ? 0 : -1;
}

int db_path_join(char *out, size_t out_size, const char *base, const char *name) {
    int written;

    /* Handle trailing slash in base */
    size_t base_len = strlen(base);
    if (base_len > 0 && base[base_len - 1] == '/') {
        written = snprintf(out, out_size, "%s%s", base, name);
    } else {
        written = snprintf(out, out_size, "%s/%s", base, name);
    }

    return (written > 0 && (size_t)written < out_size) ? 0 : -1;
}

int db_file_exists(const char *path) {
    struct stat st;
    return (stat(path, &st) == 0) ? 1 : 0;
}

/*
 * Recursive mkdir implementation
 */
static int mkdir_p(const char *path, mode_t mode) {
    char tmp[DB_PATH_MAX];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);

    /* Remove trailing slash */
    if (tmp[len - 1] == '/') {
        tmp[len - 1] = '\0';
    }

    /* Create parent directories */
    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, mode) != 0 && errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }

    /* Create final directory */
    if (mkdir(tmp, mode) != 0 && errno != EEXIST) {
        return -1;
    }

    return 0;
}

int db_ensure_directory(const char *path) {
    return mkdir_p(path, 0755);
}

/*
 * Recursive directory removal
 */
int db_remove_directory(const char *path) {
    DIR *dir = opendir(path);
    if (!dir) {
        /* Not a directory or doesn't exist */
        return (errno == ENOENT) ? 0 : -1;
    }

    struct dirent *entry;
    char child_path[DB_PATH_MAX];
    int result = 0;

    while ((entry = readdir(dir)) != NULL) {
        /* Skip . and .. */
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        if (db_path_join(child_path, sizeof(child_path), path, entry->d_name) != 0) {
            result = -1;
            break;
        }

        struct stat st;
        if (stat(child_path, &st) != 0) {
            result = -1;
            break;
        }

        if (S_ISDIR(st.st_mode)) {
            /* Recursively remove subdirectory */
            if (db_remove_directory(child_path) != 0) {
                result = -1;
                break;
            }
        } else {
            /* Remove file */
            if (unlink(child_path) != 0) {
                result = -1;
                break;
            }
        }
    }

    closedir(dir);

    /* Remove the directory itself */
    if (result == 0 && rmdir(path) != 0) {
        result = -1;
    }

    return result;
}
