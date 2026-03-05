/*
 * math_utils_cuda.cu - CUDA vector operations with persistent workspace
 * 
 * Optimizations:
 * - Grid-stride loops for processing multiple elements per thread
 * - Warp shuffle reductions (__shfl_down_sync) for final 32 threads
 * - __ldg() intrinsic for read-only texture cache access
 * - Unrolled final warp reduction (no __syncthreads needed)
 */

#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

extern "C" {
#include "../../include/math_utils.h"
}

#define BLK 256
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
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

/* Warp-level reduction using shuffle */
__device__ __forceinline__ double warp_reduce_sum(double val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

/* Block-level reduction: shared memory + warp shuffle for final warp */
__device__ __forceinline__ double block_reduce_sum(double val) {
    __shared__ double shared[BLK / WARP_SIZE];  /* One slot per warp */
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    /* Reduce within each warp */
    val = warp_reduce_sum(val);

    /* First thread in each warp writes to shared memory */
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    /* First warp reduces the per-warp results */
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0;
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

/* Element-wise kernels (grid-stride for large arrays) */

__global__ static void k_add(const double *a, const double *b, double *c, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        c[i] = __ldg(&a[i]) + __ldg(&b[i]);
    }
}

__global__ static void k_scale(const double *a, double alpha, double *c, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        c[i] = alpha * __ldg(&a[i]);
    }
}

__global__ static void k_add_scaled(
    const double *a, double alpha, const double *b, double beta,
    double *c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        c[i] = alpha * __ldg(&a[i]) + beta * __ldg(&b[i]);
    }
}

/* Dot-product reduction kernel with grid-stride loop */
__global__ static void k_dot_reduce(const double *a, const double *b, double *out, int n) {
    double sum = 0.0;
    
    /* Grid-stride loop: each thread accumulates multiple elements */
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += __ldg(&a[i]) * __ldg(&b[i]);
    }
    
    /* Block-level reduction */
    sum = block_reduce_sum(sum);
    
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

/* Host wrappers */

/* Calculate optimal grid size - limit blocks for better reduction efficiency */
static int optimal_blocks(int n) {
    int blocks = (n + BLK - 1) / BLK;
    /* Cap at 256 blocks for reductions - more work per block is more efficient */
    return (blocks > 256) ? 256 : blocks;
}

static int grid(int N) { return (N + BLK - 1) / BLK; }

extern "C" void vec_add(const double *a, const double *b, double *c, size_t n) {
    int blocks = (n > 256 * BLK) ? 256 : grid((int)n);
    k_add<<<blocks, BLK>>>(a, b, c, (int)n);
}

extern "C" void vec_scale(const double *a, double alpha, double *c, size_t n) {
    int blocks = (n > 256 * BLK) ? 256 : grid((int)n);
    k_scale<<<blocks, BLK>>>(a, alpha, c, (int)n);
}

extern "C" double vec_dot(const double *a, const double *b, size_t n) {
    int nb = optimal_blocks((int)n);
    ensure_workspace(nb);

    k_dot_reduce<<<nb, BLK>>>(a, b, g_d_workspace, (int)n);
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
    int blocks = (n > 256 * BLK) ? 256 : grid((int)n);
    k_add_scaled<<<blocks, BLK>>>(a, alpha, b, beta, c, (int)n);
}

/* Reduction kernel for trapezoidal integration with edge weights */
__global__ static void k_trap_reduce(const double *y, double *out, int n) {
    double sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    /* Grid-stride loop with __ldg() for cached reads */
    for (int i = idx; i < n; i += stride) {
        double w = (i == 0 || i == n - 1) ? 0.5 : 1.0;
        sum += w * __ldg(&y[i]);
    }
    
    /* Efficient block reduction using warp shuffles */
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

/* Reduction kernel for Simpson's rule with 1,4,2,4,2,...,4,1 weights */
__global__ static void k_simp_reduce(const double *y, double *out, int n) {
    double sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    /* Grid-stride loop with __ldg() for cached reads */
    for (int i = idx; i < n; i += stride) {
        double w = (i == 0 || i == n - 1) ? 1.0 : ((i % 2 == 1) ? 4.0 : 2.0);
        sum += w * __ldg(&y[i]);
    }
    
    /* Efficient block reduction using warp shuffles */
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

/* 2D trapezoidal reduction kernel */
__global__ static void k_trap2d_reduce(const double *z, double *out, int nx, int ny) {
    double sum = 0.0;
    int N = nx * ny;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    /* Grid-stride loop with __ldg() for cached reads */
    for (int i = idx; i < N; i += stride) {
        int ix = i % nx, iy = i / nx;
        double wx = (ix == 0 || ix == nx - 1) ? 0.5 : 1.0;
        double wy = (iy == 0 || iy == ny - 1) ? 0.5 : 1.0;
        sum += wx * wy * __ldg(&z[i]);
    }
    
    /* Efficient block reduction using warp shuffles */
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

/* Trapezoidal rule integration */
extern "C" double integrate_trapezoidal(const double *y, size_t n, double h) {
    if (n < 2) return 0.0;
    
    int nb = optimal_blocks((int)n);
    ensure_workspace(nb);
    
    k_trap_reduce<<<nb, BLK>>>(y, g_d_workspace, (int)n);
    cudaMemcpy(g_h_workspace, g_d_workspace, nb * sizeof(double), cudaMemcpyDeviceToHost);
    
    double s = 0.0;
    for (int i = 0; i < nb; ++i) s += g_h_workspace[i];
    return h * s;
}

/* Simpson's rule integration (requires odd n) */
extern "C" double integrate_simpson(const double *y, size_t n, double h) {
    if (n < 3 || n % 2 == 0) return integrate_trapezoidal(y, n, h);
    
    int nb = optimal_blocks((int)n);
    ensure_workspace(nb);
    
    k_simp_reduce<<<nb, BLK>>>(y, g_d_workspace, (int)n);
    cudaMemcpy(g_h_workspace, g_d_workspace, nb * sizeof(double), cudaMemcpyDeviceToHost);
    
    double s = 0.0;
    for (int i = 0; i < nb; ++i) s += g_h_workspace[i];
    return (h / 3.0) * s;
}

/* 2D trapezoidal integration */
extern "C" double integrate_2d_trapezoidal(const double *z, size_t nx, size_t ny, double dx, double dy) {
    if (nx < 2 || ny < 2) return 0.0;
    
    int N  = (int)(nx * ny);
    int nb = optimal_blocks(N);
    ensure_workspace(nb);
    
    k_trap2d_reduce<<<nb, BLK>>>(z, g_d_workspace, (int)nx, (int)ny);
    cudaMemcpy(g_h_workspace, g_d_workspace, nb * sizeof(double), cudaMemcpyDeviceToHost);
    
    double s = 0.0;
    for (int i = 0; i < nb; ++i) s += g_h_workspace[i];
    return dx * dy * s;
}
