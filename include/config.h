/*
 * config.h — Configuration file parser
 *
 * Reads .cfg files and populates the SimConfig structure
 * with all simulation parameters.
 */

#ifndef CONFIG_H
#define CONFIG_H

#include "grid.h"
#include "potential.h"
#include "solver.h"

/* ── Boundary condition modes ────────────────────────────────────────────── */
typedef enum {
    BC_PBC,  /* periodic in both x and y                                   */
    BC_W2,   /* hard walls at x=0 and x=Lx; periodic in y                  */
    BC_W4    /* hard walls on all four sides                                */
} BoundaryMode;

/* ── Master configuration structure ─────────────────────────────────────── */
typedef struct SimConfig {
    GridParams      grid;
    PotentialParams potential;
    SolverParams    solver;
    BoundaryMode    boundary_mode;   /* spatial boundary condition type     */
    double          temperature;     /* environment temperature T            */
    double          rho1;            /* average density of species 1         */
    double          rho2;            /* average density of species 2         */
    char            output_dir[256]; /* path for output files                */
    int             save_every;      /* save density every N iterations      */
} SimConfig;

/* Parse a .cfg file and fill config; returns 0 on success, -1 on error */
int config_load(const char *filename, SimConfig *config);

/* Print config to stdout for verification */
void config_print(const SimConfig *config);

#endif /* CONFIG_H */
