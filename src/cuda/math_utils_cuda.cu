/*
 * math_utils_cuda.cu — CUDA vector operations on device arrays
 *
 * Same interface as math_utils.h; pointers must be device memory.
 * Reductions use shared-memory tree + host finalisation.
 */

#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

extern "C" {
#include "../../include/math_utils.h"
}

#define BLK 256

/* ── element-wise kernels ───────────────────────────────────────────────── */

__global__ static void k_add(const double *a, const double *b, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ static void k_scale(const double *a, double alpha, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = alpha * a[i];
}

__global__ static void k_add_scaled(
    const double *a, double alpha, const double *b, double beta,
    double *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = alpha * a[i] + beta * b[i];
}

/* ── dot-product reduction kernel ───────────────────────────────────────── */

__global__ static void k_dot_reduce(
    const double *a, const double *b, double *out, int n)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x * 2 + tid;

    double v = 0.0;
    if (i < n)              v  = a[i] * b[i];
    if (i + blockDim.x < n) v += a[i + blockDim.x] * b[i + blockDim.x];
    sh[tid] = v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sh[0];
}

/* ── host wrappers ──────────────────────────────────────────────────────── */

static int grid(int N) { return (N + BLK - 1) / BLK; }

extern "C" void vec_add(const double *a, const double *b, double *c, size_t n) {
    k_add<<<grid((int)n), BLK>>>(a, b, c, (int)n);
}

extern "C" void vec_scale(const double *a, double alpha, double *c, size_t n) {
    k_scale<<<grid((int)n), BLK>>>(a, alpha, c, (int)n);
}

extern "C" double vec_dot(const double *a, const double *b, size_t n) {
    int nb = ((int)n + BLK * 2 - 1) / (BLK * 2);
    double *d_part, *h_part;
    cudaMalloc(&d_part, nb * sizeof(double));
    h_part = (double *)malloc(nb * sizeof(double));

    k_dot_reduce<<<nb, BLK, BLK * sizeof(double)>>>(a, b, d_part, (int)n);
    cudaMemcpy(h_part, d_part, nb * sizeof(double), cudaMemcpyDeviceToHost);

    double s = 0.0;
    for (int i = 0; i < nb; ++i) s += h_part[i];

    cudaFree(d_part);
    free(h_part);
    return s;
}

extern "C" double vec_norm(const double *a, size_t n) {
    return sqrt(vec_dot(a, a, n));
}

extern "C" void vec_add_scaled(
    const double *a, double alpha,
    const double *b, double beta,
    double *c, size_t n)
{
    k_add_scaled<<<grid((int)n), BLK>>>(a, alpha, b, beta, c, (int)n);
}
