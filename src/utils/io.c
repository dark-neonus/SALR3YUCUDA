/*
 * io.c — File I/O helpers (CPU, pure C)
 *
 * All functions return 0 on success, -1 on failure.
 * 2-D density files use row-major (y-outer, x-inner) traversal so that
 * each contiguous block in the file corresponds to a fixed y value —
 * this matches the memory layout used throughout the solver.
 */

#include <stdio.h>
#include "../../include/io.h"

/*
 * io_save_density_1d - Write a 1-D profile as two-column ASCII.
 * Columns: x  rho(x)
 */
int io_save_density_1d(const char *filename,
                       const double *x, const double *rho, size_t n) {
    FILE *f = fopen(filename, "w");
    if (!f) return -1;
    fprintf(f, "# x rho\n");
    for (size_t i = 0; i < n; ++i)
        fprintf(f, "%.10g %.10g\n", x[i], rho[i]);
    fclose(f);
    return 0;
}

/*
 * io_save_density_2d - Write a 2-D profile as three-column ASCII.
 * Columns: x  y  rho(x,y).  Array layout: rho[iy * nx + ix].
 * A blank line is inserted after each row of constant y to enable
 * gnuplot's pm3d and Python's loadtxt / reshape to work directly.
 */
int io_save_density_2d(const char *filename,
                       const double *x, const double *y,
                       const double *rho, size_t nx, size_t ny) {
    FILE *f = fopen(filename, "w");
    if (!f) return -1;
    fprintf(f, "# x y rho\n");
    for (size_t iy = 0; iy < ny; ++iy) {
        for (size_t ix = 0; ix < nx; ++ix)
            fprintf(f, "%.10g %.10g %.10g\n",
                    x[ix], y[iy], rho[iy * nx + ix]);
        fputc('\n', f);
    }
    fclose(f);
    return 0;
}

/*
 * io_log_convergence - Append one iteration record to a convergence log.
 * File is opened in append mode so successive calls accumulate data.
 * Columns: iteration  L2_error
 */
int io_log_convergence(const char *filename, int iteration, double error) {
    FILE *f = fopen(filename, "a");
    if (!f) return -1;
    fprintf(f, "%d %.14e\n", iteration, error);
    fclose(f);
    return 0;
}
