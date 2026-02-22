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

/* Run the DFT solver; returns 0 on convergence, -1 on failure.
   rho[] is the density array (input: initial guess, output: result).
   n = total number of grid points. */
int solver_run(double *rho, size_t n, const void *config);

/* Compute L2 norm of difference between two arrays */
double solver_l2_diff(const double *a, const double *b, size_t n);

#endif /* SOLVER_H */
