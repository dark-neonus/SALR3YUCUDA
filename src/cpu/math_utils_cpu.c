/*
 * math_utils_cpu.c — Vector math operations (CPU, pure C with OpenMP)
 *
 * All operations are element-wise over flat arrays of length n.
 * Output arrays may alias input arrays only when the operation is safe
 * (e.g. vec_scale(a, s, a, n) is fine; vec_add(a,b,a,n) is fine).
 */

#include <math.h>
#include <stddef.h>
#include <omp.h>
#include "../../include/math_utils.h"

/* c[i] = a[i] + b[i] */
void vec_add(const double *a, const double *b, double *c, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) { 
        c[i] = a[i] + b[i];
    }
}

/* c[i] = alpha * a[i] */
void vec_scale(const double *a, double alpha, double *c, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        c[i] = alpha * a[i];
    }
}

/* Returns sum_i a[i]*b[i] */
double vec_dot(const double *a, const double *b, size_t n) {
    double s = 0.0;
    #pragma omp parallel for reduction(+:s)
    for (size_t i = 0; i < n; ++i) { 
        s += a[i] * b[i];
    }
    return s;
}

/* Returns sqrt( sum_i a[i]^2 ) */
double vec_norm(const double *a, size_t n) {
    return sqrt(vec_dot(a, a, n));
}

/* c[i] = alpha*a[i] + beta*b[i] */
void vec_add_scaled(const double *a, double alpha,
                    const double *b, double beta,
                    double *c, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        c[i] = alpha * a[i] + beta * b[i];
    }
}
