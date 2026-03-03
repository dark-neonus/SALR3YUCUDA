/* benchmark_integral_cuda.cu - Benchmark integral calculations on CUDA */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C" {
#include "../include/math_utils.h"
}

#define M_PI 3.14159265358979323846

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* Kernel to fill 1D sin array on device */
__global__ void k_fill_sin_1d(double *y, int n, double h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = sin(i * h);
    }
}

/* Kernel to fill 2D sin*sin array on device */
__global__ void k_fill_sin_2d(double *z, int nx, int ny, double dx, double dy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nx * ny) {
        int ix = idx % nx;
        int iy = idx / nx;
        z[idx] = sin(ix * dx) * sin(iy * dy);
    }
}

int main(int argc, char **argv) {
    /* Print GPU info */
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("Integral Benchmark (CUDA)\n");
    printf("=========================\n");
    printf("GPU: %s (SM %d.%d, %zu MB)\n\n", prop.name, prop.major, prop.minor, prop.totalGlobalMem >> 20);

    /* Output file */
    FILE *outfile = fopen("output/benchmark_cuda.dat", "w");
    if (!outfile) {
        fprintf(stderr, "Cannot create output/benchmark_cuda.dat\n");
        return 1;
    }
    fprintf(outfile, "# n method time_ms error\n");

    /* Timing events */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int block_size = 256;

    /* 1D test sizes with adaptive iterations for ~60s runtime */
    int sizes[] = {1000001, 10000001, 50000001, 100000001, 200000001, 500000001};
    int iterations_per_size[] = {20000, 2000, 400, 200, 100, 40};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("1D Integration (sin(x) over [0,pi], exact=2.0)\n");
    printf("%-12s %-12s %-12s %-15s %-15s\n", "N", "Trap(ms)", "Simp(ms)", "Trap err", "Simp err");
    printf("----------------------------------------------------------------------\n");

    for (int s = 0; s < num_sizes; ++s) {
        int n = sizes[s];
        int iters = iterations_per_size[s];
        double h = M_PI / (n - 1);

        double *d_y;
        CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));

        /* Fill array on GPU */
        int grid_size = (n + block_size - 1) / block_size;
        k_fill_sin_1d<<<grid_size, block_size>>>(d_y, n, h);
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("Running N=%d with %d iterations... ", n, iters);
        fflush(stdout);

        /* Benchmark trapezoidal */
        CUDA_CHECK(cudaEventRecord(start));
        double trap_result = 0;
        for (int i = 0; i < iters; ++i) {
            trap_result = integrate_trapezoidal(d_y, n, h);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float trap_time;
        CUDA_CHECK(cudaEventElapsedTime(&trap_time, start, stop));
        trap_time /= iters;
        double trap_error = fabs(trap_result - 2.0);

        /* Benchmark Simpson's */
        CUDA_CHECK(cudaEventRecord(start));
        double simp_result = 0;
        for (int i = 0; i < iters; ++i) {
            simp_result = integrate_simpson(d_y, n, h);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float simp_time;
        CUDA_CHECK(cudaEventElapsedTime(&simp_time, start, stop));
        simp_time /= iters;
        double simp_error = fabs(simp_result - 2.0);
        printf("done\n");

        printf("%-12d %-12.4f %-12.4f %-15.2e %-15.2e\n",
               n, trap_time, simp_time, trap_error, simp_error);

        fprintf(outfile, "%d trapezoidal %f %e\n", n, trap_time, trap_error);
        fprintf(outfile, "%d simpson %f %e\n", n, simp_time, simp_error);

        CUDA_CHECK(cudaFree(d_y));
    }

    /* 2D benchmarks - slower so fewer iterations */
    int sizes_2d[] = {1001, 2001, 4001, 8001, 12001};
    int iterations_2d[] = {2000, 400, 100, 40, 20};
    int num_sizes_2d = sizeof(sizes_2d) / sizeof(sizes_2d[0]);

    printf("\n2D Integration (sin(x)*sin(y) over [0,pi]^2, exact=4.0)\n");
    printf("%-12s %-12s %-15s\n", "NxN", "Trap2D(ms)", "Error");
    printf("----------------------------------------------\n");

    for (int s = 0; s < num_sizes_2d; ++s) {
        int nx = sizes_2d[s];
        int ny = nx;
        int iters_2d = iterations_2d[s];
        double dx = M_PI / (nx - 1);
        double dy = M_PI / (ny - 1);
        int N = nx * ny;

        double *d_z;
        CUDA_CHECK(cudaMalloc(&d_z, N * sizeof(double)));

        /* Fill array on GPU */
        int grid_size = (N + block_size - 1) / block_size;
        k_fill_sin_2d<<<grid_size, block_size>>>(d_z, nx, ny, dx, dy);
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("Running %dx%d with %d iterations... ", nx, ny, iters_2d);
        fflush(stdout);

        /* Benchmark 2D trapezoidal */
        CUDA_CHECK(cudaEventRecord(start));
        double trap2d_result = 0;
        for (int i = 0; i < iters_2d; ++i) {
            trap2d_result = integrate_2d_trapezoidal(d_z, nx, ny, dx, dy);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float trap2d_time;
        CUDA_CHECK(cudaEventElapsedTime(&trap2d_time, start, stop));
        trap2d_time /= iters_2d;
        double trap2d_error = fabs(trap2d_result - 4.0);
        printf("done\n");

        printf("%-12d %-12.4f %-15.2e\n", N, trap2d_time, trap2d_error);
        fprintf(outfile, "%d trap2d %f %e\n", N, trap2d_time, trap2d_error);

        CUDA_CHECK(cudaFree(d_z));
    }

    fclose(outfile);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\nResults saved to output/benchmark_cuda.dat\n");
    return 0;
}
