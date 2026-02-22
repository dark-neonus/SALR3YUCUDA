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

/* ── Master configuration structure ─────────────────────────────────────── */
typedef struct {
    GridParams      grid;
    PotentialParams potential;
    SolverParams    solver;
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
