/*
 * potential.h — SALR 3-Yukawa interaction potential
 *
 * U_ij(r) = sum_{m=0}^{2}  A[i][j][m] * exp(-alpha[i][j][m] * r) / r
 *
 * Indices i, j in {0,1} correspond to particle species {1,2}.
 * The matrix is symmetric: A[i][j][m] == A[j][i][m].
 */

#ifndef POTENTIAL_H
#define POTENTIAL_H

#define YUKAWA_TERMS 3   /* number of Yukawa terms in the pair potential */

/* ── Potential / interaction parameters ──────────────────────────────────── */
typedef struct {
    double cutoff_radius;                      /* interaction cutoff r_c  */

    /* 3-Yukawa amplitudes and decay rates for each species pair.
     * First two indices: species pair (i,j) with i,j in {0,1}.
     * Third index: Yukawa term m in {0,1,2}.
     * Symmetric: A[0][1] == A[1][0], alpha[0][1] == alpha[1][0].        */
    double A    [2][2][YUKAWA_TERMS];          /* amplitudes A_ij_m       */
    double alpha[2][2][YUKAWA_TERMS];          /* decay rates alpha_ij_m  */
} PotentialParams;

/* Evaluate U_ij(r) for species pair (i,j) in {0,1};
 * returns 0 if r > cutoff_radius or r <= 0.                             */
double potential_u(int i, int j, double r, const PotentialParams *p);

/* Evaluate the direct correlation function c_ij(r) */
double potential_dcf(int i, int j, double r, const PotentialParams *p);

#endif /* POTENTIAL_H */
