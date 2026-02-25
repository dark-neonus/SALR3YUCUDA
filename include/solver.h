/*
 * solver.h — Iterative DFT solver interface
 *
 * Picard iteration for the density profile:
 *   rho_new = (1 - alpha) * rho_old  +  alpha * rho_calc
 *
 * Convergence is checked via L2 norm of (rho_new - rho_old).
 */

#ifndef SOLVER_H
#define SOLVER_H

#include <stddef.h>

/* ── Solver parameters ──────────────────────────────────────────────────── */
typedef struct {
    int    max_iterations;   /* maximum Picard iterations                   */
    double tolerance;        /* convergence accuracy epsilon                */
    double xi1;              /* Picard mixing coefficient for species 1     */
    double xi2;              /* Picard mixing coefficient for species 2     */
} SolverParams;

/* Forward declaration — full SimConfig defined in config.h */
struct SimConfig;

/*
 * solver_run_binary - Picard iteration for a 2-component SALR mixture.
 *
 * rho1 and rho2 are flat nx*ny arrays (input: initial guess,
 * output: converged density profile, row-major: index = iy*nx + ix).
 * Returns 0 on convergence, 1 if max_iterations was reached without
 * convergence, -1 on allocation failure.
 * Intermediate snapshots and the convergence log are written to
 * cfg->output_dir according to cfg->save_every.
 */
int solver_run_binary(double *rho1, double *rho2, struct SimConfig *cfg);

/* Compute L2 norm of (a - b) over n elements */
double solver_l2_diff(const double *a, const double *b, size_t n);

#endif /* SOLVER_H */
