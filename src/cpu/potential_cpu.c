/*
 * potential_cpu.c — Interaction potential evaluation (CPU, pure C)
 *
 * TODO: implement pair potential and direct correlation function.
 */

#include <math.h>
#include "../../include/potential.h"

double potential_salr(double r, const PotentialParams *p) {
    /* TODO: implement SALR pair potential.
     * Must respect p->cutoff_radius: return 0 if r > r_c. */
    (void)r;
    (void)p;
    return 0.0;
}

double potential_dcf(double r, const PotentialParams *p) {
    /* TODO: implement direct correlation function using A11, A12, A22 */
    (void)r;
    (void)p;
    return 0.0;
}
