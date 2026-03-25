/*
 * grid.h — Computational grid definitions
 *
 * Defines the discretised spatial domain for the DFT solver.
 */

#ifndef GRID_H
#define GRID_H

/* Grid parameters */
typedef struct {
    int    nx;       /* number of grid points along x                       */
    int    ny;       /* number of grid points along y                       */
    double Lx;       /* physical size of box along x (derived: Lx = dx*nx)  */
    double Ly;       /* physical size of box along y (derived: Ly = dy*ny)  */
    double dx;       /* discretisation step along x                         */
    double dy;       /* discretisation step along y                         */
} GridParams;

/* Allocate and initialise a uniform grid; returns pointer to coordinate
   array or NULL on failure.  Caller must free. */
double *grid_create_x(const GridParams *g);
double *grid_create_y(const GridParams *g);

/* Total grid points: nx * ny */
int grid_total_points(const GridParams *g);

#endif /* GRID_H */
