/*
 * potential.h — SALR interaction potential
 *
 * u(r) = -eps_a * exp(-r / sig_a)  +  eps_r * exp(-r / sig_r)
 *        ~~~~ attraction ~~~~         ~~~~ repulsion ~~~~
 *
 * The Aij coefficients describe inter-component or mode-coupling
 * relations in the direct correlation function matrix.
 */

#ifndef POTENTIAL_H
#define POTENTIAL_H

/* ── Potential / interaction parameters ──────────────────────────────────── */
typedef struct {
    double cutoff_radius;     /* interaction cutoff r_c                     */

    /* Direct correlation function matrix (symmetric 2×2: A21 = A12).
     * Describes pair coupling between species in a binary mixture.
     *   A11 — species 1 ↔ species 1  (self-interaction)
     *   A12 — species 1 ↔ species 2  (cross-interaction)
     *   A22 — species 2 ↔ species 2  (self-interaction)
     * TODO: compute from Ornstein-Zernike / pair potential               */
    double A11;
    double A12;
    double A22;
} PotentialParams;

/* Evaluate the SALR pair potential u(r) */
double potential_salr(double r, const PotentialParams *p);

/* Evaluate the direct correlation function c(r) (to be implemented) */
double potential_dcf(double r, const PotentialParams *p);

#endif /* POTENTIAL_H */
