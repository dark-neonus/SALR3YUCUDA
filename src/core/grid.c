/*
 * grid.c — Grid initialisation (CPU, pure C)
 *
 * TODO: implement uniform grid creation and coordinate arrays.
 */

#include <stdlib.h>
#include "../../include/grid.h"

int grid_total_points(const GridParams *g) {
    return g->nx * g->ny;
}

double *grid_create_x(const GridParams *g) {
    /* TODO: allocate and fill x-coordinate array */
    (void)g;
    return NULL;
}

double *grid_create_y(const GridParams *g) {
    /* TODO: allocate and fill y-coordinate array */
    (void)g;
    return NULL;
}
