/*
 * grid.c — Uniform 2-D grid initialisation (CPU, pure C)
 *
 * Grid points are cell-centred: x[i] = (i + 0.5) * dx, y[j] = (j + 0.5) * dy.
 * This avoids the singularity at r = 0 when evaluating Yukawa potentials and
 * keeps the grid periodic without a duplicated boundary node.
 */

#include <stdlib.h>
#include "../../include/grid.h"

int grid_total_points(const GridParams *g) {
    return g->nx * g->ny;
}

/*
 * grid_create_x - Allocate and return the x-coordinate array (length nx).
 * Caller is responsible for free().
 */
double *grid_create_x(const GridParams *g) {
    double *x = malloc((size_t)g->nx * sizeof(double));
    if (!x) return NULL;
    for (int i = 0; i < g->nx; ++i)
        x[i] = (i + 0.5) * g->dx;
    return x;
}

/*
 * grid_create_y - Allocate and return the y-coordinate array (length ny).
 * Caller is responsible for free().
 */
double *grid_create_y(const GridParams *g) {
    double *y = malloc((size_t)g->ny * sizeof(double));
    if (!y) return NULL;
    for (int j = 0; j < g->ny; ++j)
        y[j] = (j + 0.5) * g->dy;
    return y;
}
