/*
 * main.c — Entry point for the SALR DFT solver
 *
 * Usage:  ./salr_dft configs/default.cfg
 *
 * TODO: implement main simulation loop.
 */

#include <stdio.h>
#include <stdlib.h>
#include "../include/config.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        return 1;
    }

    SimConfig config;
    if (config_load(argv[1], &config) != 0) {
        fprintf(stderr, "Error: failed to load config '%s'\n", argv[1]);
        return 1;
    }

    config_print(&config);

    /* TODO:
     * 1. Create grid
     * 2. Initialise density profile (uniform bulk_density)
     * 3. Run solver (Picard iteration)
     * 4. Save results
     */

    printf("Simulation not yet implemented.\n");
    return 0;
}
