/*
 * potential_cuda.cu — 3-Yukawa potential evaluation (host + device)
 *
 * Host functions match the potential.h interface.
 * The __device__ version is used by GPU kernels in solver_cuda.cu.
 */

#include <math.h>
#include <cuda_runtime.h>

extern "C" {
#include "../../include/potential.h"
}

/* Host: U_ij(r) = Σ A_m·exp(−α_m·r)/r, zero outside cutoff */
extern "C" double potential_u(int i, int j, double r, const PotentialParams *p)
{
    if (r <= 0.0 || r > p->cutoff_radius) return 0.0;
    double s = 0.0;
    for (int m = 0; m < YUKAWA_TERMS; ++m)
        s += p->A[i][j][m] * exp(-p->alpha[i][j][m] * r) / r;
    return s;
}

/* Placeholder — not used in symmetric mean-field approximation */
extern "C" double potential_dcf(int i, int j, double r, const PotentialParams *p)
{
    (void)i; (void)j; (void)r; (void)p;
    return 0.0;
}
