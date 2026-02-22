/*
 * math_utils_cpu.c — Vector/matrix math operations (CPU, pure C)
 *
 * TODO: implement all math utility functions.
 */

#include <math.h>
#include <stddef.h>
#include "../../include/math_utils.h"

void vec_add(const double *a, const double *b, double *c, size_t n) {
    /* TODO: implement c[i] = a[i] + b[i] */
    (void)a; (void)b; (void)c; (void)n;
}

void vec_scale(const double *a, double alpha, double *c, size_t n) {
    /* TODO: implement c[i] = alpha * a[i] */
    (void)a; (void)alpha; (void)c; (void)n;
}

double vec_dot(const double *a, const double *b, size_t n) {
    /* TODO: implement dot product */
    (void)a; (void)b; (void)n;
    return 0.0;
}

double vec_norm(const double *a, size_t n) {
    /* TODO: implement Euclidean norm */
    (void)a; (void)n;
    return 0.0;
}

void vec_add_scaled(const double *a, double alpha,
                    const double *b, double beta,
                    double *c, size_t n) {
    /* TODO: implement c[i] = alpha*a[i] + beta*b[i] */
    (void)a; (void)alpha; (void)b; (void)beta; (void)c; (void)n;
}
