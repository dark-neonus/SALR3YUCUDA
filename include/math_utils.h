/*
 * math_utils.h — Vector and matrix operations
 *
 * Building blocks for the DFT solver.
 * CPU implementations first; CUDA versions will be added later.
 */

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <stddef.h>

/* ── Vector operations ──────────────────────────────────────────────────── */

void   vec_add(const double *a, const double *b, double *c, size_t n);
void   vec_scale(const double *a, double alpha, double *c, size_t n);
double vec_dot(const double *a, const double *b, size_t n);
double vec_norm(const double *a, size_t n);

/* ── Matrix operations (row-major flat arrays) ──────────────────────────── */

void vec_add_scaled(const double *a, double alpha,
                    const double *b, double beta,
                    double *c, size_t n);

#endif /* MATH_UTILS_H */
