/*
 * potential_cpu.c — Interaction potential evaluation (CPU, pure C)
 *
 * U_ij(r) = sum_{m=0}^{2} A[i][j][m] * exp(-alpha[i][j][m] * r) / r
 */

#include <math.h>
#include "../../include/potential.h"

double potential_u(int i, int j, double r, const PotentialParams *p) {
    if (r <= 0.0 || r > p->cutoff_radius)
        return 0.0;
    double result = 0.0;
    for (int m = 0; m < YUKAWA_TERMS; ++m)
        result += p->A[i][j][m] * exp(-p->alpha[i][j][m] * r) / r;
    return result;
}

double potential_dcf(int i, int j, double r, const PotentialParams *p) {
    /* Direct correlation function: c_ij(r) = -beta * U_ij(r) (mean-field approx)
     * TODO: replace with proper OZ / HNC closure if needed */
    (void)i; (void)j; (void)r; (void)p;
    return 0.0;
}
