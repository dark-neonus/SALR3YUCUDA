/*
 * io.h — File I/O helpers
 *
 * Functions for saving/loading density profiles and convergence logs.
 */

#ifndef IO_H
#define IO_H

#include <stddef.h>

/* Save a 1D density profile rho[n] to a text file (x, rho(x) columns) */
int io_save_density_1d(const char *filename,
                       const double *x, const double *rho, size_t n);

/* Save a 2D density profile to a text file (x, y, rho(x,y) columns) */
int io_save_density_2d(const char *filename,
                       const double *x, const double *y,
                       const double *rho, size_t nx, size_t ny);

/* Append one line to a convergence log: iteration, L2 error */
int io_log_convergence(const char *filename, int iteration, double error);

#endif /* IO_H */
