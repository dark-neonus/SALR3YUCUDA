/* benchmark_integral.c - Benchmark integral calculations on CPU with OpenMP */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "../include/math_utils.h"

#define M_PI 3.14159265358979323846

/* Test function: sin(x) over [0, pi] has exact integral = 2 */
static void fill_sin_1d(double *y, int n, double h) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        y[i] = sin(i * h);
    }
}

/* Test function: sin(x)*sin(y) over [0,pi]x[0,pi] has exact integral = 4 */
static void fill_sin_2d(double *z, int nx, int ny, double dx, double dy) {
    #pragma omp parallel for collapse(2)
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            z[iy * nx + ix] = sin(ix * dx) * sin(iy * dy);
        }
    }
}

/* Run benchmark and return average time in ms */
static double benchmark_func(void (*func)(void), int iterations) {
    double start = omp_get_wtime();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    double end = omp_get_wtime();
    return (end - start) * 1000.0 / iterations;
}

/* Benchmark parameters */
static double *g_y1d, *g_z2d;
static int g_n, g_nx, g_ny;
static double g_h, g_dx, g_dy;
static double g_result;

static void run_trap_1d(void) { g_result = integrate_trapezoidal(g_y1d, g_n, g_h); }
static void run_simp_1d(void) { g_result = integrate_simpson(g_y1d, g_n, g_h); }
static void run_trap_2d(void) { g_result = integrate_2d_trapezoidal(g_z2d, g_nx, g_ny, g_dx, g_dy); }

int main(int argc, char **argv) {
    printf("Integral Benchmark (CPU OpenMP)\n");
    printf("================================\n");
    printf("Threads: %d\n\n", omp_get_max_threads());

    /* Output file for Python plotting */
    FILE *outfile = fopen("output/benchmark_cpu.dat", "w");
    if (!outfile) {
        fprintf(stderr, "Cannot create output/benchmark_cpu.dat\n");
        return 1;
    }
    fprintf(outfile, "# n method time_ms error\n");

    /* Test sizes from 1M to 500M points with adaptive iterations for ~60s runtime */
    int sizes[] = {1000001, 10000001, 50000001, 100000001, 200000001, 500000001};
    int iterations_per_size[] = {20000, 2000, 400, 200, 100, 40};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("1D Integration (sin(x) over [0,pi], exact=2.0)\n");
    printf("%-12s %-12s %-12s %-15s %-15s\n", "N", "Trap(ms)", "Simp(ms)", "Trap err", "Simp err");
    printf("----------------------------------------------------------------------\n");

    for (int s = 0; s < num_sizes; ++s) {
        g_n = sizes[s];
        int iters = iterations_per_size[s];
        g_h = M_PI / (g_n - 1);
        g_y1d = (double *)malloc(g_n * sizeof(double));
        fill_sin_1d(g_y1d, g_n, g_h);

        printf("Running N=%d with %d iterations... ", g_n, iters);
        fflush(stdout);

        double trap_time = benchmark_func(run_trap_1d, iters);
        double trap_result = integrate_trapezoidal(g_y1d, g_n, g_h);
        double trap_error = fabs(trap_result - 2.0);

        double simp_time = benchmark_func(run_simp_1d, iters);
        double simp_result = integrate_simpson(g_y1d, g_n, g_h);
        double simp_error = fabs(simp_result - 2.0);
        printf("done\n");

        printf("%-12d %-12.4f %-12.4f %-15.2e %-15.2e\n",
               g_n, trap_time, simp_time, trap_error, simp_error);

        fprintf(outfile, "%d trapezoidal %f %e\n", g_n, trap_time, trap_error);
        fprintf(outfile, "%d simpson %f %e\n", g_n, simp_time, simp_error);

        free(g_y1d);
    }

    /* 2D benchmarks - slower so fewer iterations */
    int sizes_2d[] = {1001, 2001, 4001, 8001, 12001};
    int iterations_2d[] = {2000, 400, 100, 40, 20};
    int num_sizes_2d = sizeof(sizes_2d) / sizeof(sizes_2d[0]);

    printf("\n2D Integration (sin(x)*sin(y) over [0,pi]^2, exact=4.0)\n");
    printf("%-12s %-12s %-15s\n", "NxN", "Trap2D(ms)", "Error");
    printf("----------------------------------------------\n");

    for (int s = 0; s < num_sizes_2d; ++s) {
        g_nx = g_ny = sizes_2d[s];
        int iters_2d = iterations_2d[s];
        g_dx = g_dy = M_PI / (g_nx - 1);
        g_z2d = (double *)malloc(g_nx * g_ny * sizeof(double));
        fill_sin_2d(g_z2d, g_nx, g_ny, g_dx, g_dy);

        printf("Running %dx%d with %d iterations... ", g_nx, g_ny, iters_2d);
        fflush(stdout);

        double trap2d_time = benchmark_func(run_trap_2d, iters_2d);
        double trap2d_result = integrate_2d_trapezoidal(g_z2d, g_nx, g_ny, g_dx, g_dy);
        double trap2d_error = fabs(trap2d_result - 4.0);        printf("done\n");
        printf("%-12d %-12.4f %-15.2e\n", g_nx * g_ny, trap2d_time, trap2d_error);
        fprintf(outfile, "%d trap2d %f %e\n", g_nx * g_ny, trap2d_time, trap2d_error);

        free(g_z2d);
    }

    fclose(outfile);
    printf("\nResults saved to output/benchmark_cpu.dat\n");
    return 0;
}
