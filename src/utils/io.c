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
#include "../../include/config.h"

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

/*
 * io_save_parameters - Save all simulation parameters to a text file.
 * Format: key = value pairs, organized by sections for easy parsing by scripts.
 */
int io_save_parameters(const char *filename, const struct SimConfig *config) {
    FILE *f = fopen(filename, "w");
    if (!f) return -1;
    
    /* Header */
    fprintf(f, "# SALR3YUCUDA Simulation Parameters\n");
    fprintf(f, "# Auto-generated parameter file\n\n");
    
    /* Grid parameters */
    fprintf(f, "[grid]\n");
    fprintf(f, "Lx = %.10g\n", config->grid.Lx);
    fprintf(f, "Ly = %.10g\n", config->grid.Ly);
    fprintf(f, "dx = %.10g\n", config->grid.dx);
    fprintf(f, "dy = %.10g\n", config->grid.dy);
    fprintf(f, "nx = %d\n", config->grid.nx);
    fprintf(f, "ny = %d\n", config->grid.ny);
    fprintf(f, "boundary_mode = %s\n", 
            config->boundary_mode == BC_PBC ? "PBC" :
            config->boundary_mode == BC_W2  ? "W2" : "W4");
    fprintf(f, "\n");
    
    /* Physics parameters */
    fprintf(f, "[physics]\n");
    fprintf(f, "temperature = %.10g\n", config->temperature);
    fprintf(f, "rho1 = %.10g\n", config->rho1);
    fprintf(f, "rho2 = %.10g\n", config->rho2);
    fprintf(f, "cutoff_radius = %.10g\n", config->potential.cutoff_radius);
    fprintf(f, "\n");
    
    /* Interaction parameters */
    fprintf(f, "[interaction]\n");
    for (int m = 0; m < YUKAWA_TERMS; ++m) {
        fprintf(f, "A_11_%d = %.10g\n", m+1, config->potential.A[0][0][m]);
    }
    for (int m = 0; m < YUKAWA_TERMS; ++m) {
        fprintf(f, "a_11_%d = %.10g\n", m+1, config->potential.alpha[0][0][m]);
    }
    fprintf(f, "\n");
    for (int m = 0; m < YUKAWA_TERMS; ++m) {
        fprintf(f, "A_12_%d = %.10g\n", m+1, config->potential.A[0][1][m]);
    }
    for (int m = 0; m < YUKAWA_TERMS; ++m) {
        fprintf(f, "a_12_%d = %.10g\n", m+1, config->potential.alpha[0][1][m]);
    }
    fprintf(f, "\n");
    for (int m = 0; m < YUKAWA_TERMS; ++m) {
        fprintf(f, "A_22_%d = %.10g\n", m+1, config->potential.A[1][1][m]);
    }
    for (int m = 0; m < YUKAWA_TERMS; ++m) {
        fprintf(f, "a_22_%d = %.10g\n", m+1, config->potential.alpha[1][1][m]);
    }
    fprintf(f, "\n");
    
    /* Solver parameters */
    fprintf(f, "[solver]\n");
    fprintf(f, "max_iterations = %d\n", config->solver.max_iterations);
    fprintf(f, "tolerance = %.10g\n", config->solver.tolerance);
    fprintf(f, "xi1 = %.10g\n", config->solver.xi1);
    fprintf(f, "xi2 = %.10g\n", config->solver.xi2);
    fprintf(f, "\n");
    
    /* Output parameters */
    fprintf(f, "[output]\n");
    fprintf(f, "output_dir = %s\n", config->output_dir);
    fprintf(f, "save_every = %d\n", config->save_every);
    
    fclose(f);
    return 0;
}
