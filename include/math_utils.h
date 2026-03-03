/*
 * math_utils.h — Vector and matrix operations
 *
 * Building blocks for the DFT solver.
 * CPU implementations first; CUDA versions will be added later.
 */

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <stddef.h>

/* Vector operations */

void   vec_add(const double *a, const double *b, double *c, size_t n);
void   vec_scale(const double *a, double alpha, double *c, size_t n);
double vec_dot(const double *a, const double *b, size_t n);
double vec_norm(const double *a, size_t n);

/* Scaled addition: c[i] = alpha*a[i] + beta*b[i] */
void vec_add_scaled(const double *a, double alpha,
                    const double *b, double beta,
                    double *c, size_t n);

/* Numerical integration */

/* Trapezoidal rule: integral of y over uniform grid with spacing h */
double integrate_trapezoidal(const double *y, size_t n, double h);

/* Simpson's rule: integral of y over uniform grid with spacing h (n must be odd) */
double integrate_simpson(const double *y, size_t n, double h);

/* 2D trapezoidal integral over nx*ny grid with spacings dx,dy */
double integrate_2d_trapezoidal(const double *z, size_t nx, size_t ny, double dx, double dy);

#endif /* MATH_UTILS_H */
