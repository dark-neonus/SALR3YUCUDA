/*
 * config.c — Configuration file parser (CPU, pure C)
 *
 * Parses INI-style .cfg files and populates SimConfig.
 * Sections: [grid], [physics], [interaction], [solver], [output].
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "../../include/config.h"

/* ── helpers ────────────────────────────────────────────────────────────── */

static char *trim(char *s) {
    while (isspace((unsigned char)*s)) ++s;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)*(end - 1))) --end;
    *end = '\0';
    return s;
}

/* ── parser ─────────────────────────────────────────────────────────────── */

int config_load(const char *filename, SimConfig *config) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "config_load: cannot open '%s'\n", filename);
        return -1;
    }

    /* sensible defaults */
    memset(config, 0, sizeof(*config));
    snprintf(config->output_dir, sizeof(config->output_dir), "output/");
    config->save_every = 100;
    config->solver.max_iterations = 10000;
    config->solver.tolerance      = 1e-8;
    config->solver.xi1            = 0.2;
    config->solver.xi2            = 0.2;

    char line[512];
    char section[64] = "";

    while (fgets(line, sizeof(line), f)) {
        /* strip comment */
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';

        char *p = trim(line);
        if (*p == '\0') continue;

        /* section header */
        if (*p == '[') {
            char *end = strchr(p, ']');
            if (end) {
                *end = '\0';
                strncpy(section, p + 1, sizeof(section) - 1);
                section[sizeof(section) - 1] = '\0';
            }
            continue;
        }

        /* key = value */
        char *eq = strchr(p, '=');
        if (!eq) continue;
        *eq = '\0';
        char *key = trim(p);
        char *val = trim(eq + 1);

        /* ── [grid] ───────────────────────────────────────────────────── */
        if (strcmp(section, "grid") == 0) {
            if      (strcmp(key, "nx") == 0) config->grid.nx = atoi(val);
            else if (strcmp(key, "ny") == 0) config->grid.ny = atoi(val);
            else if (strcmp(key, "Lx") == 0) config->grid.Lx = atof(val);
            else if (strcmp(key, "Ly") == 0) config->grid.Ly = atof(val);
            else if (strcmp(key, "dx") == 0) config->grid.dx = atof(val);
            else if (strcmp(key, "dy") == 0) config->grid.dy = atof(val);
            else if (strcmp(key, "boundary_mode") == 0) {
                if      (strcmp(val, "W2") == 0) config->boundary_mode = BC_W2;
                else if (strcmp(val, "W4") == 0) config->boundary_mode = BC_W4;
                else                             config->boundary_mode = BC_PBC;
            }
        }
        /* ── [physics] ────────────────────────────────────────────────── */
        else if (strcmp(section, "physics") == 0) {
            if      (strcmp(key, "temperature")   == 0) config->temperature              = atof(val);
            else if (strcmp(key, "rho1")          == 0) config->rho1                     = atof(val);
            else if (strcmp(key, "rho2")          == 0) config->rho2                     = atof(val);
            else if (strcmp(key, "cutoff_radius") == 0) config->potential.cutoff_radius  = atof(val);
        }
        /* ── [interaction] ────────────────────────────────────────────── */
        else if (strcmp(section, "interaction") == 0) {
            /* Keys: A_IJ_M and a_IJ_M where I,J in {1,2}, M in {1,2,3} */
            int si, sj, sm;
            char atype;
            /* format:  A_11_1  or  a_11_1  (uppercase A = amplitude, lowercase a = decay) */
            if (sscanf(key, "%c_%1d%1d_%1d", &atype, &si, &sj, &sm) == 4) {
                int i = si - 1, j = sj - 1, m = sm - 1;
                if (i >= 0 && i < 2 && j >= 0 && j < 2 && m >= 0 && m < YUKAWA_TERMS) {
                    double v = atof(val);
                    if (atype == 'A') {
                        config->potential.A[i][j][m]     = v;
                        config->potential.A[j][i][m]     = v;  /* symmetry */
                    } else if (atype == 'a') {
                        config->potential.alpha[i][j][m] = v;
                        config->potential.alpha[j][i][m] = v;  /* symmetry */
                    }
                }
            }
        }
        /* ── [solver] ─────────────────────────────────────────────────── */
        else if (strcmp(section, "solver") == 0) {
            if      (strcmp(key, "max_iterations") == 0) config->solver.max_iterations = atoi(val);
            else if (strcmp(key, "tolerance")      == 0) config->solver.tolerance      = atof(val);
            else if (strcmp(key, "xi1")            == 0) config->solver.xi1            = atof(val);
            else if (strcmp(key, "xi2")            == 0) config->solver.xi2            = atof(val);
        }
        /* ── [output] ─────────────────────────────────────────────────── */
        else if (strcmp(section, "output") == 0) {
            if      (strcmp(key, "output_dir") == 0) snprintf(config->output_dir, sizeof(config->output_dir), "%s", val);
            else if (strcmp(key, "save_every") == 0) config->save_every = atoi(val);
        }
    }

    fclose(f);
    return 0;
}

void config_print(const SimConfig *config) {
    printf("=== SimConfig ===\n");
    printf("[grid]\n");
    printf("  nx=%d  ny=%d  Lx=%.4g  Ly=%.4g  dx=%.4g  dy=%.4g\n",
           config->grid.nx, config->grid.ny,
           config->grid.Lx, config->grid.Ly,
           config->grid.dx, config->grid.dy);
    const char *bc_names[] = {"PBC", "W2", "W4"};
    printf("  boundary_mode=%s\n", bc_names[config->boundary_mode]);
    printf("[physics]\n");
    printf("  temperature=%.6g  rho1=%.6g  rho2=%.6g  cutoff_radius=%.6g\n",
           config->temperature, config->rho1, config->rho2,
           config->potential.cutoff_radius);
    printf("[interaction] 3-Yukawa parameters (A / alpha):\n");
    const char *pairs[3] = {"11", "12", "22"};
    int pi[3] = {0, 0, 1};
    int pj[3] = {0, 1, 1};
    for (int p = 0; p < 3; ++p) {
        int i = pi[p], j = pj[p];
        printf("  pair %s: ", pairs[p]);
        for (int m = 0; m < YUKAWA_TERMS; ++m)
            printf("A%d=%.7g a%d=%.7g  ", m+1, config->potential.A[i][j][m],
                                          m+1, config->potential.alpha[i][j][m]);
        printf("\n");
    }
    printf("[solver]\n");
    printf("  max_iter=%d  tol=%.2e  xi1=%.4g  xi2=%.4g\n",
           config->solver.max_iterations, config->solver.tolerance,
           config->solver.xi1, config->solver.xi2);
    printf("[output]\n");
    printf("  output_dir=%s  save_every=%d\n",
           config->output_dir, config->save_every);
}
