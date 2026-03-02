/*
 * solver_cuda.cu — CUDA Picard DFT solver for 2-component SALR mixture
 *
 * All heavy loops (convolution, K operator, mixing, smoothing, reduction)
 * run as GPU kernels. Data stays on device between iterations; only copied
 * back for periodic snapshots and final output.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C" {
#include "../../include/solver.h"
#include "../../include/config.h"
#include "../../include/potential.h"
#include "../../include/io.h"
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(call) do {                                           \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

#define BLK 256
#define SMOOTH_EPS 0.01

/* grid size helpers */
static inline int divup(int n, int d) { return (n + d - 1) / d; }
static inline int nblk(int N) { return divup(N, BLK); }
static inline int nblk_red(int N) { return divup(N, BLK * 2); }

/* ═══════════════════════════════════════════════════════════════════════════
 *  DEVICE: 3-Yukawa potential U(r) = Σ A_m·exp(−α_m·r)/r
 * ═══════════════════════════════════════════════════════════════════════════ */

__device__ static double dev_potential(
    double r, const double *A, const double *alpha, double rc)
{
    if (r <= 0.0 || r > rc) return 0.0;
    double s = 0.0;
    for (int m = 0; m < 3; ++m)
        s += A[m] * exp(-alpha[m] * r) / r;
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  KERNELS
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Build potential lookup table: one thread per (diy,dix) offset */
__global__ void k_build_pot(
    double *tbl, const double *A, const double *alpha, double rc,
    int nx, int ny, double dx, double dy, int wx, int wy)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) return;

    int diy = id / nx, dix = id % nx;
    double dry = wy ? diy * dy : (diy <= ny / 2 ? diy * dy : (diy - ny) * dy);
    double drx = wx ? dix * dx : (dix <= nx / 2 ? dix * dx : (dix - nx) * dx);
    tbl[id] = dev_potential(sqrt(drx * drx + dry * dry), A, alpha, rc);
}

/* ── Φ convolution — the main O(N²) bottleneck fully parallelised ────────
 *
 *  Each thread computes Φ₁₁, Φ₁₂, Φ₂₁, Φ₂₂ at one grid point by
 *  summing ρ(r')·U(|r−r'|) over all source points r'.
 *  __ldg() directs potential-table reads through the read-only cache.
 */
__global__ void k_compute_Phi(
    double * __restrict__ P11, double * __restrict__ P12,
    double * __restrict__ P21, double * __restrict__ P22,
    const double * __restrict__ r1, const double * __restrict__ r2,
    const double * __restrict__ U11, const double * __restrict__ U12,
    const double * __restrict__ U22,
    int nx, int ny, double dA, int wx, int wy)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) return;

    int iy = id / nx, ix = id % nx;
    double p11 = 0.0, p12 = 0.0, p21 = 0.0, p22 = 0.0;

    for (int jy = 0; jy < ny; ++jy) {
        /* minimum-image (PBC) or absolute (wall) y-displacement */
        int diy;
        if (wy) { diy = abs(iy - jy); }
        else     { diy = (iy - jy + ny) % ny; if (diy > ny / 2) diy = ny - diy; }

        for (int jx = 0; jx < nx; ++jx) {
            int dix;
            if (wx) { dix = abs(ix - jx); }
            else     { dix = (ix - jx + nx) % nx; if (dix > nx / 2) dix = nx - dix; }

            int si = jy * nx + jx;
            int ui = diy * nx + dix;

            double u11 = __ldg(&U11[ui]);
            double u12 = __ldg(&U12[ui]);
            double u22 = __ldg(&U22[ui]);
            double s1  = r1[si];
            double s2  = r2[si];

            p11 += s1 * u11;  p12 += s2 * u12;
            p21 += s1 * u12;  p22 += s2 * u22;
        }
    }

    P11[id] = p11 * dA;  P12[id] = p12 * dA;
    P21[id] = p21 * dA;  P22[id] = p22 * dA;
}

/* K_i = ρ_b · exp(−β·(Φ_a + Φ_b − Φ_ab − Φ_bb)), clamped to ±500 */
__global__ void k_compute_K(
    const double *Pa, const double *Pb, double Pab, double Pbb,
    double rho_b, double beta, double *K, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double a = -beta * (Pa[i] + Pb[i] - Pab - Pbb);
    a = fmin(fmax(a, -500.0), 500.0);
    K[i] = rho_b * exp(a);
}

/* Picard mixing: ρ ← (1−ξ)·ρ + ξ·K */
__global__ void k_mix(double *rho, const double *K, double xi, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    rho[i] = (1.0 - xi) * rho[i] + xi * K[i];
}

/* Zero density at wall nodes (BC_W2=1, BC_W4=2) */
__global__ void k_boundary(double *r1, double *r2, int nx, int ny, int mode)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) return;
    int iy = id / nx, ix = id % nx;
    int z = 0;
    if (mode >= 1 && (ix == 0 || ix == nx - 1)) z = 1;
    if (mode == 2 && (iy == 0 || iy == ny - 1)) z = 1;
    if (z) { r1[id] = 0.0; r2[id] = 0.0; }
}

/* 5-point Laplacian smoothing to suppress checkerboard mode */
__global__ void k_smooth(
    const double *in, double *out, int nx, int ny, int mode, double eps)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) return;
    int iy = id / nx, ix = id % nx;
    int wkx = (mode >= 1), wky = (mode == 2);

    /* wall cells pass through unchanged */
    if ((wkx && (ix == 0 || ix == nx - 1)) ||
        (wky && (iy == 0 || iy == ny - 1))) {
        out[id] = in[id]; return;
    }

    int ym = wky ? max(iy - 1, 0)      : (iy - 1 + ny) % ny;
    int yp = wky ? min(iy + 1, ny - 1)  : (iy + 1) % ny;
    int xm = wkx ? max(ix - 1, 0)      : (ix - 1 + nx) % nx;
    int xp = wkx ? min(ix + 1, nx - 1)  : (ix + 1) % nx;

    out[id] = (1.0 - 4.0 * eps) * in[id]
            + eps * (in[iy * nx + xm] + in[iy * nx + xp]
                   + in[ym * nx + ix] + in[yp * nx + ix]);
}

/* (a−b)² per element, zero at wall cells — feeds into L2 reduction */
__global__ void k_sq_diff(
    const double *a, const double *b, double *out,
    int nx, int ny, int mode)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) return;
    int iy = id / nx, ix = id % nx;
    if (mode >= 1 && (ix == 0 || ix == nx - 1)) { out[id] = 0.0; return; }
    if (mode == 2 && (iy == 0 || iy == ny - 1)) { out[id] = 0.0; return; }
    double d = a[id] - b[id];
    out[id] = d * d;
}

/* ── Block-level tree reduction (shared memory) ─────────────────────────
 *  Each block reduces 2×BLK elements to one partial sum. */
__global__ void k_reduce(const double *in, double *out, int N)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x * 2 + tid;
    sh[tid] = (i < N ? in[i] : 0.0) + (i + blockDim.x < N ? in[i + blockDim.x] : 0.0);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sh[0];
}

/* Reduction summing only interior cells (walls excluded) */
__global__ void k_reduce_interior(
    const double *in, double *out, int nx, int ny, int mode)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int N   = nx * ny;
    int i   = blockIdx.x * blockDim.x * 2 + tid;
    int i2  = i + blockDim.x;

    double v1 = 0.0, v2 = 0.0;
    if (i < N) {
        int iy1 = i / nx, ix1 = i % nx;
        int skip = (mode >= 1 && (ix1 == 0 || ix1 == nx - 1))
                || (mode == 2 && (iy1 == 0 || iy1 == ny - 1));
        if (!skip) v1 = in[i];
    }
    if (i2 < N) {
        int iy2 = i2 / nx, ix2 = i2 % nx;
        int skip = (mode >= 1 && (ix2 == 0 || ix2 == nx - 1))
                || (mode == 2 && (iy2 == 0 || iy2 == ny - 1));
        if (!skip) v2 = in[i2];
    }
    sh[tid] = v1 + v2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sh[0];
}

/* Scale interior cells by a constant factor (mass renormalisation) */
__global__ void k_scale_interior(
    double *data, double factor, int nx, int ny, int mode)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) return;
    int iy = id / nx, ix = id % nx;
    if (mode >= 1 && (ix == 0 || ix == nx - 1)) return;
    if (mode == 2 && (iy == 0 || iy == ny - 1)) return;
    data[id] *= factor;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  HOST HELPERS
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Two-pass GPU sum: kernel reduces to per-block partials, host finalises */
static double gpu_sum(const double *d_data, double *d_part, double *h_part, int N)
{
    int nb = nblk_red(N);
    k_reduce<<<nb, BLK, BLK * sizeof(double)>>>(d_data, d_part, N);
    CUDA_CHECK(cudaMemcpy(h_part, d_part, nb * sizeof(double), cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int i = 0; i < nb; ++i) s += h_part[i];
    return s;
}

/* Same, but only interior cells */
static double gpu_sum_interior(
    const double *d_data, double *d_part, double *h_part,
    int nx, int ny, int mode)
{
    int N  = nx * ny;
    int nb = nblk_red(N);
    k_reduce_interior<<<nb, BLK, BLK * sizeof(double)>>>(d_data, d_part, nx, ny, mode);
    CUDA_CHECK(cudaMemcpy(h_part, d_part, nb * sizeof(double), cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int i = 0; i < nb; ++i) s += h_part[i];
    return s;
}

/* Count interior (non-wall) cells */
static int count_interior(int nx, int ny, int mode)
{
    switch (mode) {
        case BC_W2:  return (nx - 2) * ny;
        case BC_W4:  return (nx - 2) * (ny - 2);
        default:     return nx * ny;
    }
}

/* Save snapshot to disk (host-side helper) */
static void save_snap(const double *r1, const double *r2,
                      const double *xs, const double *ys,
                      int nx, int ny, int iter, const char *dir)
{
    char p1[512], p2[512];
    snprintf(p1, sizeof(p1), "%s/data/density_species1_iter_%06d.dat", dir, iter);
    snprintf(p2, sizeof(p2), "%s/data/density_species2_iter_%06d.dat", dir, iter);
    io_save_density_2d(p1, xs, ys, r1, (size_t)nx, (size_t)ny);
    io_save_density_2d(p2, xs, ys, r2, (size_t)nx, (size_t)ny);
}

static void save_final(const double *r1, const double *r2,
                       const double *xs, const double *ys,
                       int nx, int ny, const char *dir)
{
    char p1[512], p2[512];
    snprintf(p1, sizeof(p1), "%s/data/density_species1_final.dat", dir);
    snprintf(p2, sizeof(p2), "%s/data/density_species2_final.dat", dir);
    io_save_density_2d(p1, xs, ys, r1, (size_t)nx, (size_t)ny);
    io_save_density_2d(p2, xs, ys, r2, (size_t)nx, (size_t)ny);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  PUBLIC API  (extern "C" so main.c can call it)
 * ═══════════════════════════════════════════════════════════════════════════ */

extern "C" double solver_l2_diff(const double *a, const double *b, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) { double d = a[i] - b[i]; sum += d * d; }
    return sqrt(sum / (double)n);
}

extern "C" int solver_run_binary(double *rho1, double *rho2, struct SimConfig *cfg)
{
    /* ── grid / physics constants ───────────────────────────────────────── */
    const int    Nx   = cfg->grid.nx;
    const int    Ny   = cfg->grid.ny;
    const double dx   = cfg->grid.dx;
    const double dy   = cfg->grid.dy;
    const int    N    = Nx * Ny;
    const double dA   = dx * dy;
    const double beta = 1.0 / cfg->temperature;
    const int    mode = cfg->boundary_mode;
    const int    wx   = (mode == BC_W2 || mode == BC_W4);
    const int    wy   = (mode == BC_W4);
    const int    interior = count_interior(Nx, Ny, mode);
    const size_t sz   = (size_t)N * sizeof(double);

    const double rho1_b = cfg->rho1;
    const double rho2_b = cfg->rho2;
    const int    max_it = cfg->solver.max_iterations;
    const double tol    = cfg->solver.tolerance;
    const double xi1    = cfg->solver.xi1;
    const double xi2    = cfg->solver.xi2;
    const int    sav_ev = cfg->save_every;
    const double rc     = cfg->potential.cutoff_radius;

    /* ── print GPU info ─────────────────────────────────────────────────── */
    {
        int dev; cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        printf("  CUDA device: %s  (SM %d.%d, %.0f MHz, %zu MB)\n",
               prop.name, prop.major, prop.minor,
               prop.clockRate / 1000.0, prop.totalGlobalMem >> 20);
        printf("  Grid: %dx%d = %d points, %d interior\n", Nx, Ny, N, interior);
    }

    /* ── device allocations ─────────────────────────────────────────────── */
    double *d_rho1, *d_rho2;
    double *d_U11,  *d_U12,  *d_U22;
    double *d_P11,  *d_P12,  *d_P21,  *d_P22;
    double *d_K1,   *d_K2,   *d_tmp,  *d_part;
    int    nb_r = nblk_red(N);

    CUDA_CHECK(cudaMalloc(&d_rho1, sz));
    CUDA_CHECK(cudaMalloc(&d_rho2, sz));
    CUDA_CHECK(cudaMalloc(&d_U11,  sz));
    CUDA_CHECK(cudaMalloc(&d_U12,  sz));
    CUDA_CHECK(cudaMalloc(&d_U22,  sz));
    CUDA_CHECK(cudaMalloc(&d_P11,  sz));
    CUDA_CHECK(cudaMalloc(&d_P12,  sz));
    CUDA_CHECK(cudaMalloc(&d_P21,  sz));
    CUDA_CHECK(cudaMalloc(&d_P22,  sz));
    CUDA_CHECK(cudaMalloc(&d_K1,   sz));
    CUDA_CHECK(cudaMalloc(&d_K2,   sz));
    CUDA_CHECK(cudaMalloc(&d_tmp,  sz));
    CUDA_CHECK(cudaMalloc(&d_part, nb_r * sizeof(double)));

    double *h_part = (double *)malloc(nb_r * sizeof(double));

    /* Yukawa parameters on device (3 doubles each, 6 small arrays) */
    double *d_A11, *d_a11, *d_A12, *d_a12, *d_A22, *d_a22;
    size_t ysz = YUKAWA_TERMS * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_A11, ysz));  CUDA_CHECK(cudaMalloc(&d_a11, ysz));
    CUDA_CHECK(cudaMalloc(&d_A12, ysz));  CUDA_CHECK(cudaMalloc(&d_a12, ysz));
    CUDA_CHECK(cudaMalloc(&d_A22, ysz));  CUDA_CHECK(cudaMalloc(&d_a22, ysz));

    CUDA_CHECK(cudaMemcpy(d_A11, cfg->potential.A[0][0],     ysz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a11, cfg->potential.alpha[0][0], ysz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A12, cfg->potential.A[0][1],     ysz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a12, cfg->potential.alpha[0][1], ysz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A22, cfg->potential.A[1][1],     ysz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a22, cfg->potential.alpha[1][1], ysz, cudaMemcpyHostToDevice));

    /* ── build potential tables on GPU ──────────────────────────────────── */
    int g = nblk(N);
    k_build_pot<<<g, BLK>>>(d_U11, d_A11, d_a11, rc, Nx, Ny, dx, dy, wx, wy);
    k_build_pot<<<g, BLK>>>(d_U12, d_A12, d_a12, rc, Nx, Ny, dx, dy, wx, wy);
    k_build_pot<<<g, BLK>>>(d_U22, d_A22, d_a22, rc, Nx, Ny, dx, dy, wx, wy);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ── bulk Φ fields (numerical sum of potential tables on GPU) ────────
     *  Φ_11,b = dA·ρ₁_b·Σ U₁₁,  Φ_12,b = dA·ρ₂_b·Σ U₁₂, etc. */
    double sU11 = gpu_sum(d_U11, d_part, h_part, N);
    double sU12 = gpu_sum(d_U12, d_part, h_part, N);
    double sU22 = gpu_sum(d_U22, d_part, h_part, N);

    const double Phi11b = dA * rho1_b * sU11;
    const double Phi12b = dA * rho2_b * sU12;
    const double Phi21b = dA * rho1_b * sU12;
    const double Phi22b = dA * rho2_b * sU22;

    /* ── coordinate arrays for file output ──────────────────────────────── */
    double *xs = (double *)malloc(Nx * sizeof(double));
    double *ys = (double *)malloc(Ny * sizeof(double));
    for (int i = 0; i < Nx; ++i) xs[i] = (i + 0.5) * dx;
    for (int j = 0; j < Ny; ++j) ys[j] = (j + 0.5) * dy;

    /* convergence log */
    char log_path[512];
    snprintf(log_path, sizeof(log_path), "%s/convergence.dat", cfg->output_dir);
    { FILE *lf = fopen(log_path, "w");
      if (lf) { fprintf(lf, "# iter  L2_error\n"); fclose(lf); } }

    /* ── transfer initial densities to GPU ──────────────────────────────── */
    CUDA_CHECK(cudaMemcpy(d_rho1, rho1, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho2, rho2, sz, cudaMemcpyHostToDevice));
    k_boundary<<<g, BLK>>>(d_rho1, d_rho2, Nx, Ny, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* copy back for initial snapshot */
    CUDA_CHECK(cudaMemcpy(rho1, d_rho1, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rho2, d_rho2, sz, cudaMemcpyDeviceToHost));
    save_snap(rho1, rho2, xs, ys, Nx, Ny, 0, cfg->output_dir);

    /* ── CUDA timing events ─────────────────────────────────────────────── */
    cudaEvent_t t_start, t_stop;
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_stop));
    CUDA_CHECK(cudaEventRecord(t_start));

    /* ═══════════════════════════════════════════════════════════════════════
     *  PICARD ITERATION LOOP
     * ═══════════════════════════════════════════════════════════════════════ */
    int converged = 0;

    for (int iter = 0; iter < max_it; ++iter) {

        /* 1. Φ convolution on GPU (O(N²) parallelised over N threads) */
        k_compute_Phi<<<g, BLK>>>(
            d_P11, d_P12, d_P21, d_P22,
            d_rho1, d_rho2, d_U11, d_U12, d_U22,
            Nx, Ny, dA, wx, wy);

        /* 2. Euler-Lagrange operator K_i */
        k_compute_K<<<g, BLK>>>(d_P11, d_P12, Phi11b, Phi12b, rho1_b, beta, d_K1, N);
        k_compute_K<<<g, BLK>>>(d_P21, d_P22, Phi21b, Phi22b, rho2_b, beta, d_K2, N);

        /* 2b. Mass renormalisation: interior mean K → ρ_b */
        {
            double s1 = gpu_sum_interior(d_K1, d_part, h_part, Nx, Ny, mode);
            double s2 = gpu_sum_interior(d_K2, d_part, h_part, Nx, Ny, mode);
            if (interior > 0 && s1 > 1e-12 && s2 > 1e-12) {
                k_scale_interior<<<g, BLK>>>(
                    d_K1, (double)interior * rho1_b / s1, Nx, Ny, mode);
                k_scale_interior<<<g, BLK>>>(
                    d_K2, (double)interior * rho2_b / s2, Nx, Ny, mode);
            }
        }

        /* 3. Convergence: ξ·√(Σ(K−ρ)²/N_int) before mixing */
        double err;
        {
            k_sq_diff<<<g, BLK>>>(d_K1, d_rho1, d_tmp, Nx, Ny, mode);
            double s1 = gpu_sum(d_tmp, d_part, h_part, N);
            k_sq_diff<<<g, BLK>>>(d_K2, d_rho2, d_tmp, Nx, Ny, mode);
            double s2 = gpu_sum(d_tmp, d_part, h_part, N);
            double ic = (interior > 0) ? (double)interior : 1.0;
            double e1 = xi1 * sqrt(s1 / ic);
            double e2 = xi2 * sqrt(s2 / ic);
            err = (e1 > e2) ? e1 : e2;
        }

        /* 4. Picard mixing: ρ ← (1−ξ)ρ + ξK */
        k_mix<<<g, BLK>>>(d_rho1, d_K1, xi1, N);
        k_mix<<<g, BLK>>>(d_rho2, d_K2, xi2, N);

        /* 5. Boundary → smooth → boundary */
        k_boundary<<<g, BLK>>>(d_rho1, d_rho2, Nx, Ny, mode);

        k_smooth<<<g, BLK>>>(d_rho1, d_tmp, Nx, Ny, mode, SMOOTH_EPS);
        CUDA_CHECK(cudaMemcpy(d_rho1, d_tmp, sz, cudaMemcpyDeviceToDevice));
        k_smooth<<<g, BLK>>>(d_rho2, d_tmp, Nx, Ny, mode, SMOOTH_EPS);
        CUDA_CHECK(cudaMemcpy(d_rho2, d_tmp, sz, cudaMemcpyDeviceToDevice));

        k_boundary<<<g, BLK>>>(d_rho1, d_rho2, Nx, Ny, mode);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* 6. Logging and periodic snapshots */
        io_log_convergence(log_path, iter, err);

        if ((iter + 1) % sav_ev == 0) {
            CUDA_CHECK(cudaMemcpy(rho1, d_rho1, sz, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(rho2, d_rho2, sz, cudaMemcpyDeviceToHost));

            double mn1 = rho1[0], mx1 = rho1[0], sm1 = 0;
            double mn2 = rho2[0], mx2 = rho2[0], sm2 = 0;
            for (int k = 0; k < N; ++k) {
                if (rho1[k] < mn1) mn1 = rho1[k]; if (rho1[k] > mx1) mx1 = rho1[k]; sm1 += rho1[k];
                if (rho2[k] < mn2) mn2 = rho2[k]; if (rho2[k] > mx2) mx2 = rho2[k]; sm2 += rho2[k];
            }
            printf("  iter %6d   err=%.3e   rho1[%.4f,%.4f,%.4f]  rho2[%.4f,%.4f,%.4f]\n",
                   iter + 1, err, mn1, sm1 / (double)N, mx1,
                   mn2, sm2 / (double)N, mx2);
            save_snap(rho1, rho2, xs, ys, Nx, Ny, iter + 1, cfg->output_dir);
        }

        if (err < tol) {
            CUDA_CHECK(cudaEventRecord(t_stop));
            CUDA_CHECK(cudaEventSynchronize(t_stop));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t_start, t_stop));
            printf("  Converged at iteration %d  (err=%.3e < %.3e, %.2f s GPU)\n",
                   iter + 1, err, tol, ms / 1000.0f);

            CUDA_CHECK(cudaMemcpy(rho1, d_rho1, sz, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(rho2, d_rho2, sz, cudaMemcpyDeviceToHost));
            save_snap(rho1, rho2, xs, ys, Nx, Ny, iter + 1, cfg->output_dir);
            save_final(rho1, rho2, xs, ys, Nx, Ny, cfg->output_dir);
            converged = 1;
            break;
        }
    }

    if (!converged) {
        CUDA_CHECK(cudaEventRecord(t_stop));
        CUDA_CHECK(cudaEventSynchronize(t_stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t_start, t_stop));
        fprintf(stderr, "Warning: did not converge in %d iterations (%.2f s GPU).\n",
                max_it, ms / 1000.0f);

        CUDA_CHECK(cudaMemcpy(rho1, d_rho1, sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rho2, d_rho2, sz, cudaMemcpyDeviceToHost));
        save_final(rho1, rho2, xs, ys, Nx, Ny, cfg->output_dir);
    }

    /* ── cleanup ────────────────────────────────────────────────────────── */
    CUDA_CHECK(cudaEventDestroy(t_start));
    CUDA_CHECK(cudaEventDestroy(t_stop));

    cudaFree(d_rho1); cudaFree(d_rho2);
    cudaFree(d_U11);  cudaFree(d_U12);  cudaFree(d_U22);
    cudaFree(d_P11);  cudaFree(d_P12);  cudaFree(d_P21);  cudaFree(d_P22);
    cudaFree(d_K1);   cudaFree(d_K2);   cudaFree(d_tmp);  cudaFree(d_part);
    cudaFree(d_A11);  cudaFree(d_a11);
    cudaFree(d_A12);  cudaFree(d_a12);
    cudaFree(d_A22);  cudaFree(d_a22);
    free(h_part); free(xs); free(ys);

    return converged ? 0 : 1;
}
