/*
 * math_utils_cuda.cu - CUDA vector operations with persistent workspace
 */

#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

extern "C" {
#include "../../include/math_utils.h"
}

#define BLK 256
#define MAX_BLOCKS 65536

/* Persistent workspace for reductions to avoid cudaMalloc/Free overhead */
static double *g_d_workspace = NULL;
static double *g_h_workspace = NULL;
static int     g_workspace_size = 0;

static void ensure_workspace(int blocks) {
    if (blocks > g_workspace_size) {
        if (g_d_workspace) cudaFree(g_d_workspace);
        if (g_h_workspace) free(g_h_workspace);
        g_workspace_size = (blocks < MAX_BLOCKS) ? MAX_BLOCKS : blocks;
        cudaMalloc(&g_d_workspace, g_workspace_size * sizeof(double));
        g_h_workspace = (double *)malloc(g_workspace_size * sizeof(double));
    }
}

/* Element-wise kernels */

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

/* Dot-product reduction kernel */
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

/* Host wrappers */

static int grid(int N) { return (N + BLK - 1) / BLK; }

extern "C" void vec_add(const double *a, const double *b, double *c, size_t n) {
    k_add<<<grid((int)n), BLK>>>(a, b, c, (int)n);
}

extern "C" void vec_scale(const double *a, double alpha, double *c, size_t n) {
    k_scale<<<grid((int)n), BLK>>>(a, alpha, c, (int)n);
}

extern "C" double vec_dot(const double *a, const double *b, size_t n) {
    int nb = ((int)n + BLK * 2 - 1) / (BLK * 2);
    ensure_workspace(nb);

    k_dot_reduce<<<nb, BLK, BLK * sizeof(double)>>>(a, b, g_d_workspace, (int)n);
    cudaMemcpy(g_h_workspace, g_d_workspace, nb * sizeof(double), cudaMemcpyDeviceToHost);

    double s = 0.0;
    for (int i = 0; i < nb; ++i) s += g_h_workspace[i];
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

/* Reduction kernel for trapezoidal integration with edge weights */
__global__ static void k_trap_reduce(const double *y, double *out, int n) {
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x * 2 + tid;
    int i2  = i + blockDim.x;
    
    double v = 0.0;
    if (i < n) {
        double w = (i == 0 || i == n - 1) ? 0.5 : 1.0;
        v = w * y[i];
    }
    if (i2 < n) {
        double w = (i2 == 0 || i2 == n - 1) ? 0.5 : 1.0;
        v += w * y[i2];
    }
    sh[tid] = v;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sh[0];
}

/* Reduction kernel for Simpson's rule with 1,4,2,4,2,...,4,1 weights */
__global__ static void k_simp_reduce(const double *y, double *out, int n) {
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x * 2 + tid;
    int i2  = i + blockDim.x;
    
    double v = 0.0;
    if (i < n) {
        double w = (i == 0 || i == n - 1) ? 1.0 : ((i % 2 == 1) ? 4.0 : 2.0);
        v = w * y[i];
    }
    if (i2 < n) {
        double w = (i2 == 0 || i2 == n - 1) ? 1.0 : ((i2 % 2 == 1) ? 4.0 : 2.0);
        v += w * y[i2];
    }
    sh[tid] = v;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sh[0];
}

/* 2D trapezoidal reduction kernel */
__global__ static void k_trap2d_reduce(const double *z, double *out, int nx, int ny) {
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int N   = nx * ny;
    int i   = blockIdx.x * blockDim.x * 2 + tid;
    int i2  = i + blockDim.x;
    
    double v = 0.0;
    if (i < N) {
        int ix = i % nx, iy = i / nx;
        double wx = (ix == 0 || ix == nx - 1) ? 0.5 : 1.0;
        double wy = (iy == 0 || iy == ny - 1) ? 0.5 : 1.0;
        v = wx * wy * z[i];
    }
    if (i2 < N) {
        int ix = i2 % nx, iy = i2 / nx;
        double wx = (ix == 0 || ix == nx - 1) ? 0.5 : 1.0;
        double wy = (iy == 0 || iy == ny - 1) ? 0.5 : 1.0;
        v += wx * wy * z[i2];
    }
    sh[tid] = v;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sh[0];
}

/* Trapezoidal rule integration */
extern "C" double integrate_trapezoidal(const double *y, size_t n, double h) {
    if (n < 2) return 0.0;
    
    int nb = ((int)n + BLK * 2 - 1) / (BLK * 2);
    ensure_workspace(nb);
    
    k_trap_reduce<<<nb, BLK, BLK * sizeof(double)>>>(y, g_d_workspace, (int)n);
    cudaMemcpy(g_h_workspace, g_d_workspace, nb * sizeof(double), cudaMemcpyDeviceToHost);
    
    double s = 0.0;
    for (int i = 0; i < nb; ++i) s += g_h_workspace[i];
    return h * s;
}

/* Simpson's rule integration (requires odd n) */
extern "C" double integrate_simpson(const double *y, size_t n, double h) {
    if (n < 3 || n % 2 == 0) return integrate_trapezoidal(y, n, h);
    
    int nb = ((int)n + BLK * 2 - 1) / (BLK * 2);
    ensure_workspace(nb);
    
    k_simp_reduce<<<nb, BLK, BLK * sizeof(double)>>>(y, g_d_workspace, (int)n);
    cudaMemcpy(g_h_workspace, g_d_workspace, nb * sizeof(double), cudaMemcpyDeviceToHost);
    
    double s = 0.0;
    for (int i = 0; i < nb; ++i) s += g_h_workspace[i];
    return (h / 3.0) * s;
}

/* 2D trapezoidal integration */
extern "C" double integrate_2d_trapezoidal(const double *z, size_t nx, size_t ny, double dx, double dy) {
    if (nx < 2 || ny < 2) return 0.0;
    
    int N  = (int)(nx * ny);
    int nb = (N + BLK * 2 - 1) / (BLK * 2);
    ensure_workspace(nb);
    
    k_trap2d_reduce<<<nb, BLK, BLK * sizeof(double)>>>(z, g_d_workspace, (int)nx, (int)ny);
    cudaMemcpy(g_h_workspace, g_d_workspace, nb * sizeof(double), cudaMemcpyDeviceToHost);
    
    double s = 0.0;
    for (int i = 0; i < nb; ++i) s += g_h_workspace[i];
    return dx * dy * s;
}
