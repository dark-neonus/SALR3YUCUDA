/*
 * io.c — File I/O helpers (CPU, pure C)
 *
 * TODO: implement density profile saving and convergence logging.
 */

#include <stdio.h>
#include "../../include/io.h"

int io_save_density_1d(const char *filename,
                       const double *x, const double *rho, size_t n) {
    /* TODO: write x[i] rho[i] columns to file */
    (void)filename; (void)x; (void)rho; (void)n;
    return -1;
}

int io_save_density_2d(const char *filename,
                       const double *x, const double *y,
                       const double *rho, size_t nx, size_t ny) {
    /* TODO: write x y rho(x,y) columns to file */
    (void)filename; (void)x; (void)y; (void)rho; (void)nx; (void)ny;
    return -1;
}

int io_log_convergence(const char *filename, int iteration, double error) {
    /* TODO: append iteration and error to log file */
    (void)filename; (void)iteration; (void)error;
    return -1;
}
