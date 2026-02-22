/*
 * solver_cpu.c — Iterative DFT solver (CPU, pure C)
 *
 * TODO: implement Picard iteration loop.
 */

#include <stddef.h>
#include <math.h>
#include "../../include/solver.h"
#include "../../include/math_utils.h"

int solver_run(double *rho, size_t n, const void *config) {
    /* TODO: implement Picard iteration:
     *   1. compute rho_calc from current rho via integral equation
     *   2. mix:  rho_new = (1-alpha)*rho_old + alpha*rho_calc
     *   3. check convergence: ||rho_new - rho_old||_2 < tolerance
     *   4. repeat until converged or max_iterations reached
     */
    (void)rho;
    (void)n;
    (void)config;
    return -1;
}

double solver_l2_diff(const double *a, const double *b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum);
}
