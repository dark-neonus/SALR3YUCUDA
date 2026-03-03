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

/* Trapezoidal rule: integral = h * (y[0]/2 + y[1] + ... + y[n-2] + y[n-1]/2) */
double integrate_trapezoidal(const double *y, size_t n, double h) {
    if (n < 2) return 0.0;
    
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 1; i < n - 1; ++i) {
        sum += y[i];
    }
    
    return h * (0.5 * y[0] + sum + 0.5 * y[n - 1]);
}

/* Simpson's rule: requires odd n (even number of intervals). Uses 1/3 rule */
double integrate_simpson(const double *y, size_t n, double h) {
    if (n < 3 || n % 2 == 0) return integrate_trapezoidal(y, n, h);
    
    double sum_odd = 0.0, sum_even = 0.0;
    
    #pragma omp parallel for reduction(+:sum_odd)
    for (size_t i = 1; i < n - 1; i += 2) {
        sum_odd += y[i];
    }
    
    #pragma omp parallel for reduction(+:sum_even)
    for (size_t i = 2; i < n - 1; i += 2) {
        sum_even += y[i];
    }
    
    return (h / 3.0) * (y[0] + 4.0 * sum_odd + 2.0 * sum_even + y[n - 1]);
}

/* 2D trapezoidal rule: applies trapezoidal in both x and y directions */
double integrate_2d_trapezoidal(const double *z, size_t nx, size_t ny, double dx, double dy) {
    if (nx < 2 || ny < 2) return 0.0;
    
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum)
    for (size_t iy = 0; iy < ny; ++iy) {
        double wy = (iy == 0 || iy == ny - 1) ? 0.5 : 1.0;
        for (size_t ix = 0; ix < nx; ++ix) {
            double wx = (ix == 0 || ix == nx - 1) ? 0.5 : 1.0;
            sum += wx * wy * z[iy * nx + ix];
        }
    }
    
    return dx * dy * sum;
}
