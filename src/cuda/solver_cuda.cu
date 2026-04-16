/*
 * solver_cuda.cu
 * CUDA Picard-iteration DFT solver for a 2-component SALR mixture.
 *
 * Algorithm overview:
 *   1. Build pair-potential lookup tables U_ij on the GPU.
 *   2. Each Picard step:
 *      a. Convolve densities with U to get mean-field potentials Phi_ij.
 *      b. Compute new trial densities K via the Euler-Lagrange equation.
 *      c. Renormalize K to conserve particle number.
 *      d. Mix K into rho with step size xi (damped if stagnating).
 *      e. Apply Laplacian smoothing and re-enforce wall BCs.
 *   3. Repeat until the L2 error falls below tolerance or max_iterations reached.
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

/* Abort on fatal CUDA errors. */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* Set ret = -1 and jump to cleanup on CUDA errors. */
#define CUDA_TRY(call)                                                          \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            ret = -1;                                                           \
            goto cleanup;                                                       \
        }                                                                       \
    } while (0)

#define BLK        256     /* threads per block */
#define SMOOTH_EPS 0.01    /* Laplacian smoothing coefficient */

/* ── grid helpers ─────────────────────────────────────────────────────────── */

static inline int divup(int n, int d)
{
    return (n + d - 1) / d;
}

static inline int nblk(int N)
{
    return divup(N, BLK);
}

/* Reduction uses pairs of blocks, so halve the block count. */
static inline int nblk_red(int N)
{
    return divup(N, BLK * 2);
}

/* Returns 1 if cell (ix,iy) is a wall (zero-density) cell. */
static __host__ __device__ __forceinline__
int is_wall_cell(int ix, int iy, int nx, int ny, int mode)
{
    if (mode == BC_W2) {
        return (ix == 0 || ix == nx - 1);
    }
    if (mode == BC_W4) {
        return (ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1);
    }
    return 0;
}

/* Number of physical (non-wall) cells. */
static inline int count_interior_cells(int nx, int ny, int mode)
{
    int wall_x = (mode == BC_W2 || mode == BC_W4) ? 2 : 0;
    int wall_y = (mode == BC_W4)                  ? 2 : 0;
    int phys_x = (nx - wall_x > 0) ? nx - wall_x : 0;
    int phys_y = (ny - wall_y > 0) ? ny - wall_y : 0;
    return phys_x * phys_y;
}

/* ── constant memory for 3-Yukawa potential coefficients ─────────────────── */

__constant__ double d_A[2][2][3];
__constant__ double d_alpha[2][2][3];

/* 3-Yukawa potential U(r) = sum_m A_m * exp(-alpha_m * r) / r */
__device__ static double dev_potential(double r, int i, int j, double rc)
{
    if (r <= 0.0 || r > rc) {
        return 0.0;
    }
    double s = 0.0;
    for (int m = 0; m < 3; ++m) {
        s += d_A[i][j][m] * exp(-d_alpha[i][j][m] * r) / r;
    }
    return s;
}

/* ── kernels ──────────────────────────────────────────────────────────────── */

/*
 * k_build_pot
 * Fill the pair-potential lookup table tbl[ny*nx].
 * For wall directions the index is an absolute distance; for periodic
 * directions it is a wrapped distance mapped to [0, N/2].
 */
__global__ void k_build_pot(
    double *tbl, int pair_i, int pair_j, double rc,
    int nx, int ny, double dx, double dy, int wx, int wy)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) {
        return;
    }

    int    diy = id / nx;
    int    dix = id % nx;
    double dry = wy ? diy * dy : (diy <= ny / 2 ? diy * dy : (diy - ny) * dy);
    double drx = wx ? dix * dx : (dix <= nx / 2 ? dix * dx : (dix - nx) * dx);

    tbl[id] = dev_potential(sqrt(drx * drx + dry * dry), pair_i, pair_j, rc);
}

/*
 * k_compute_Phi
 * Direct-space convolution: Phi_ij[r] = dA * sum_{r'} rho_j[r'] * U_ij(|r-r'|).
 * Tiles of source densities are loaded into shared memory to reduce global reads.
 *
 * Bug fix (vs. original): jx/jy are now correctly initialised per-tile element
 * by precomputing the 2-D index of element e inside the tile.
 */
__global__ void k_compute_Phi(
    double * __restrict__ P11, double * __restrict__ P12,
    double * __restrict__ P21, double * __restrict__ P22,
    const double * __restrict__ r1,  const double * __restrict__ r2,
    const double * __restrict__ U11, const double * __restrict__ U12,
    const double * __restrict__ U22,
    int nx, int ny, double dA, int wx, int wy)
{
    int id    = blockIdx.x * blockDim.x + threadIdx.x;
    int N     = nx * ny;
    int valid = (id < N);

    const int half_nx = nx / 2;
    const int half_ny = ny / 2;

    int    iy   = valid ? id / nx : 0;
    int    ix   = valid ? id % nx : 0;
    double p11  = 0.0, p12 = 0.0, p21 = 0.0, p22 = 0.0;

    __shared__ double s_r1[BLK];
    __shared__ double s_r2[BLK];

    int num_tiles = (N + BLK - 1) / BLK;

    for (int t = 0; t < num_tiles; ++t) {
        /* Load one tile of source densities into shared memory. */
        int t_id = t * BLK + threadIdx.x;
        s_r1[threadIdx.x] = (t_id < N) ? __ldg(&r1[t_id]) : 0.0;
        s_r2[threadIdx.x] = (t_id < N) ? __ldg(&r2[t_id]) : 0.0;
        __syncthreads();

        if (valid) {
            int elems  = min(BLK, N - t * BLK);
            int tile0  = t * BLK;            /* linear index of tile element 0 */
            int jy0    = tile0 / nx;
            int jx0    = tile0 % nx;

            for (int e = 0; e < elems; ++e) {
                /* 2-D index of source cell, correctly derived per element. */
                int jy = jy0 + (jx0 + e) / nx;
                int jx = (jx0 + e) % nx;

                /* Wrapped or wall distance index for the lookup table. */
                int diy, dix;
                if (wy) {
                    diy = abs(iy - jy);
                } else {
                    diy = (iy - jy + ny) % ny;
                    if (diy > half_ny) { diy = ny - diy; }
                }
                if (wx) {
                    dix = abs(ix - jx);
                } else {
                    dix = (ix - jx + nx) % nx;
                    if (dix > half_nx) { dix = nx - dix; }
                }

                /* Skip self-interaction (handled via bulk reference). */
                if (diy == 0 && dix == 0) {
                    continue;
                }

                int    ui  = diy * nx + dix;
                double u11 = __ldg(&U11[ui]);
                double u12 = __ldg(&U12[ui]);
                double u22 = __ldg(&U22[ui]);
                double s1  = s_r1[e];
                double s2  = s_r2[e];

                p11 += s1 * u11;
                p12 += s2 * u12;
                p21 += s1 * u12;
                p22 += s2 * u22;
            }
        }
        __syncthreads();
    }

    if (valid) {
        P11[id] = p11 * dA;
        P12[id] = p12 * dA;
        P21[id] = p21 * dA;
        P22[id] = p22 * dA;
    }
}

/*
 * k_compute_K2_boundary
 * Euler-Lagrange trial density:
 *   K_i[r] = rho_i_bulk * exp( -beta * (Phi_i[r] - Phi_i_bulk) )
 * Wall cells are pinned to zero.
 */
__global__ void k_compute_K2_boundary(
    const double *P11, const double *P12,
    const double *P21, const double *P22,
    double Phi11b, double Phi12b, double Phi21b, double Phi22b,
    double rho1_b, double rho2_b, double beta,
    double *K1, double *K2,
    int nx, int ny, int mode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx * ny) {
        return;
    }

    if (is_wall_cell(i % nx, i / nx, nx, ny, mode)) {
        K1[i] = 0.0;
        K2[i] = 0.0;
        return;
    }

    /* Clamp exponent to avoid overflow. */
    double a1 = -beta * (P11[i] + P12[i] - Phi11b - Phi12b);
    double a2 = -beta * (P21[i] + P22[i] - Phi21b - Phi22b);
    a1 = fmin(fmax(a1, -500.0), 500.0);
    a2 = fmin(fmax(a2, -500.0), 500.0);

    K1[i] = rho1_b * exp(a1);
    K2[i] = rho2_b * exp(a2);
}

/*
 * k_mix2_boundary
 * Picard mixing: rho <- (1-xi)*rho + xi*K.
 * Wall cells remain zero.
 */
__global__ void k_mix2_boundary(
    double *rho1, double *rho2,
    const double *K1, const double *K2,
    double xi1, double xi2,
    int nx, int ny, int mode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx * ny) {
        return;
    }

    if (is_wall_cell(i % nx, i / nx, nx, ny, mode)) {
        rho1[i] = 0.0;
        rho2[i] = 0.0;
        return;
    }

    rho1[i] = (1.0 - xi1) * rho1[i] + xi1 * K1[i];
    rho2[i] = (1.0 - xi2) * rho2[i] + xi2 * K2[i];
}

/* Zero out wall cells in both density fields. */
__global__ void k_boundary(double *r1, double *r2, int nx, int ny, int mode)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) {
        return;
    }

    if (is_wall_cell(id % nx, id / nx, nx, ny, mode)) {
        r1[id] = 0.0;
        r2[id] = 0.0;
    }
}

/*
 * k_smooth
 * One step of mass-conserving Laplacian smoothing:
 *   out[i] = in[i] + eps * (sum_nbr in[nbr] - degree * in[i])
 * Periodic or wall boundary conditions are respected per direction.
 */
__global__ void k_smooth(
    const double *in, double *out,
    int nx, int ny, int mode, double eps)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) {
        return;
    }

    int iy  = id / nx;
    int ix  = id % nx;
    int wkx = (mode >= 1);
    int wky = (mode == 2);

    double nbr_sum = 0.0;
    int    degree  = 0;

    /* x-neighbors */
    if (ix > 0) {
        nbr_sum += in[iy * nx + (ix - 1)];
        degree++;
    } else if (!wkx) {
        nbr_sum += in[iy * nx + (nx - 1)];
        degree++;
    }

    if (ix < nx - 1) {
        nbr_sum += in[iy * nx + (ix + 1)];
        degree++;
    } else if (!wkx) {
        nbr_sum += in[iy * nx + 0];
        degree++;
    }

    /* y-neighbors */
    if (iy > 0) {
        nbr_sum += in[(iy - 1) * nx + ix];
        degree++;
    } else if (!wky) {
        nbr_sum += in[(ny - 1) * nx + ix];
        degree++;
    }

    if (iy < ny - 1) {
        nbr_sum += in[(iy + 1) * nx + ix];
        degree++;
    } else if (!wky) {
        nbr_sum += in[0 * nx + ix];
        degree++;
    }

    out[id] = in[id] + eps * (nbr_sum - degree * in[id]);
}

/* Squared pointwise difference, zeroed on wall cells. */
__global__ void k_sq_diff(
    const double *a, const double *b, double *out,
    int nx, int ny, int mode)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) {
        return;
    }

    if (is_wall_cell(id % nx, id / nx, nx, ny, mode)) {
        out[id] = 0.0;
        return;
    }

    double d = a[id] - b[id];
    out[id] = d * d;
}

/*
 * k_reduce
 * Two-element-per-thread parallel sum reduction.
 * Each block reduces BLK*2 input elements to one partial sum in out[blockIdx.x].
 */
__global__ void k_reduce(const double *in, double *out, int N)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x * 2 + tid;

    sh[tid] = (i < N             ? in[i]             : 0.0)
            + (i + blockDim.x < N ? in[i + blockDim.x] : 0.0);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh[tid] += sh[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sh[0];
    }
}

/* Same as k_reduce but silently skips wall cells. */
__global__ void k_reduce_interior(
    const double *in, double *out,
    int nx, int ny, int mode)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int N   = nx * ny;
    int i   = blockIdx.x * blockDim.x * 2 + tid;
    int i2  = i + blockDim.x;

    double v1 = 0.0;
    double v2 = 0.0;

    if (i < N && !is_wall_cell(i % nx, i / nx, nx, ny, mode)) {
        v1 = in[i];
    }
    if (i2 < N && !is_wall_cell(i2 % nx, i2 / nx, nx, ny, mode)) {
        v2 = in[i2];
    }

    sh[tid] = v1 + v2;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh[tid] += sh[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sh[0];
    }
}

/* Multiply all physical cells by factor (wall cells untouched). */
__global__ void k_scale_interior(
    double *data, double factor,
    int nx, int ny, int mode)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) {
        return;
    }

    if (!is_wall_cell(id % nx, id / nx, nx, ny, mode)) {
        data[id] *= factor;
    }
}

/* ── host-side GPU reduction helpers ─────────────────────────────────────── */

/* Sum all N elements of d_data via two-pass reduction. */
static double gpu_sum(
    const double *d_data, double *d_part, double *h_part, int N)
{
    int nb = nblk_red(N);
    k_reduce<<<nb, BLK, BLK * sizeof(double)>>>(d_data, d_part, N);

    cudaError_t err = cudaMemcpy(h_part, d_part, nb * sizeof(double),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n",
                __FILE__, __LINE__, cudaGetErrorString(err));
        return NAN;
    }

    double s = 0.0;
    for (int i = 0; i < nb; ++i) {
        s += h_part[i];
    }
    return s;
}

/* Sum only physical (non-wall) cells. */
static double gpu_sum_interior(
    const double *d_data, double *d_part, double *h_part,
    int nx, int ny, int mode)
{
    int N  = nx * ny;
    int nb = nblk_red(N);
    k_reduce_interior<<<nb, BLK, BLK * sizeof(double)>>>(d_data, d_part, nx, ny, mode);

    cudaError_t err = cudaMemcpy(h_part, d_part, nb * sizeof(double),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n",
                __FILE__, __LINE__, cudaGetErrorString(err));
        return NAN;
    }

    double s = 0.0;
    for (int i = 0; i < nb; ++i) {
        s += h_part[i];
    }
    return s;
}

/* ── snapshot I/O helpers ─────────────────────────────────────────────────── */

static void save_snap(
    const double *r1, const double *r2,
    const double *xs, const double *ys,
    int nx, int ny, int iter, const char *dir)
{
    char p1[512], p2[512];
    snprintf(p1, sizeof(p1), "%s/data/density_species1_iter_%06d.dat", dir, iter);
    snprintf(p2, sizeof(p2), "%s/data/density_species2_iter_%06d.dat", dir, iter);
    io_save_density_2d(p1, xs, ys, r1, (size_t)nx, (size_t)ny);
    io_save_density_2d(p2, xs, ys, r2, (size_t)nx, (size_t)ny);
}

static void save_final(
    const double *r1, const double *r2,
    const double *xs, const double *ys,
    int nx, int ny, const char *dir)
{
    char p1[512], p2[512];
    snprintf(p1, sizeof(p1), "%s/data/density_species1_final.dat", dir);
    snprintf(p2, sizeof(p2), "%s/data/density_species2_final.dat", dir);
    io_save_density_2d(p1, xs, ys, r1, (size_t)nx, (size_t)ny);
    io_save_density_2d(p2, xs, ys, r2, (size_t)nx, (size_t)ny);
}

/* ── public utility ───────────────────────────────────────────────────────── */

extern "C" double solver_l2_diff(const double *a, const double *b, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum / (double)n);
}

/* ── DB engine forward declarations ─────────────────────────────────────────*/

#ifdef USE_DB_ENGINE
extern "C" {
#include "db_engine.h"
}
#endif

/* ── core solver implementation ──────────────────────────────────────────── */

/*
 * solver_run_binary_impl
 * Shared execution path for both the standalone and DB-enabled builds.
 *
 * Returns 0 on convergence, 1 if max_iterations reached, -1 on CUDA error.
 */
static int solver_run_binary_impl(
    double          *rho1,
    double          *rho2,
    struct SimConfig *cfg,
    int              start_iter,
    int              enable_db,
    void            *db_run,
    double          *final_error_out)
{
    int ret = 1;
    int g = 0;
    double sU11 = 0.0, sU12 = 0.0, sU22 = 0.0;
    double Phi11b = 0.0, Phi12b = 0.0, Phi21b = 0.0, Phi22b = 0.0;
    double prev_err = -1.0;
    double final_err = 0.0;
    int converged = 0;

#ifdef USE_DB_ENGINE
    struct DbRun *run = (struct DbRun *)db_run;
#else
    (void)db_run;
#endif

    /* ── simulation constants ── */
    const int    Nx       = cfg->grid.nx;
    const int    Ny       = cfg->grid.ny;
    const double dx       = cfg->grid.dx;
    const double dy       = cfg->grid.dy;
    const int    N        = Nx * Ny;
    const double dA       = dx * dy;
    const double beta     = 1.0 / cfg->temperature;
    const int    mode     = cfg->boundary_mode;
    const int    wx       = (mode == BC_W2 || mode == BC_W4);
    const int    wy       = (mode == BC_W4);
    const int    interior = count_interior_cells(Nx, Ny, mode);
    const size_t sz       = (size_t)N * sizeof(double);

    const double rho1_b    = cfg->rho1;
    const double rho2_b    = cfg->rho2;
    const int    max_it    = cfg->solver.max_iterations;
    const double tol       = cfg->solver.tolerance;
    const int    sav_ev    = cfg->save_every;
    const double rc        = cfg->potential.cutoff_radius;
    const double err_thresh = cfg->solver.error_change_threshold;
    const double xi_damp   = cfg->solver.xi_damping_factor;

    /* Mixing parameters are mutable (adaptive damping). */
    double xi1 = cfg->solver.xi1;
    double xi2 = cfg->solver.xi2;

    /* ── device pointers ── */
    double *d_rho1 = NULL, *d_rho2 = NULL;
    double *d_U11  = NULL, *d_U12  = NULL, *d_U22  = NULL;
    double *d_P11  = NULL, *d_P12  = NULL, *d_P21  = NULL, *d_P22  = NULL;
    double *d_K1   = NULL, *d_K2   = NULL;
    double *d_tmp  = NULL, *d_part = NULL;
    double *d_s1   = NULL, *d_s2   = NULL;
    double *h_part = NULL;
    double *xs     = NULL, *ys = NULL;
    int     nb_r   = nblk_red(N);

    cudaEvent_t t_start = NULL, t_stop = NULL;
    int events_created = 0;

    /* ── print device info ── */
    {
        int            dev;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        printf("  CUDA device: %s  (SM %d.%d, %.0f MHz, %zu MB)\n",
               prop.name, prop.major, prop.minor,
               prop.clockRate / 1000.0, prop.totalGlobalMem >> 20);
        printf("  Grid: %dx%d = %d points, %d interior\n", Nx, Ny, N, interior);
        if (enable_db && start_iter > 0) {
            printf("  Resuming from iteration %d\n", start_iter);
        }
    }

    /* ── allocations ── */
    CUDA_TRY(cudaMalloc(&d_rho1, sz));
    CUDA_TRY(cudaMalloc(&d_rho2, sz));
    CUDA_TRY(cudaMalloc(&d_U11,  sz));
    CUDA_TRY(cudaMalloc(&d_U12,  sz));
    CUDA_TRY(cudaMalloc(&d_U22,  sz));
    CUDA_TRY(cudaMalloc(&d_P11,  sz));
    CUDA_TRY(cudaMalloc(&d_P12,  sz));
    CUDA_TRY(cudaMalloc(&d_P21,  sz));
    CUDA_TRY(cudaMalloc(&d_P22,  sz));
    CUDA_TRY(cudaMalloc(&d_K1,   sz));
    CUDA_TRY(cudaMalloc(&d_K2,   sz));
    CUDA_TRY(cudaMalloc(&d_tmp,  sz));
    CUDA_TRY(cudaMalloc(&d_s1,   sz));
    CUDA_TRY(cudaMalloc(&d_s2,   sz));
    CUDA_TRY(cudaMalloc(&d_part, nb_r * sizeof(double)));
    CUDA_TRY(cudaMallocHost((void **)&h_part, nb_r * sizeof(double)));

    /* Upload potential coefficients to constant memory. */
    CUDA_TRY(cudaMemcpyToSymbol(d_A,     cfg->potential.A,
                                 sizeof(double) * 2 * 2 * 3));
    CUDA_TRY(cudaMemcpyToSymbol(d_alpha, cfg->potential.alpha,
                                 sizeof(double) * 2 * 2 * 3));

    /* ── build potential tables and compute bulk reference values ── */
    g = nblk(N);
    k_build_pot<<<g, BLK>>>(d_U11, 0, 0, rc, Nx, Ny, dx, dy, wx, wy);
    k_build_pot<<<g, BLK>>>(d_U12, 0, 1, rc, Nx, Ny, dx, dy, wx, wy);
    k_build_pot<<<g, BLK>>>(d_U22, 1, 1, rc, Nx, Ny, dx, dy, wx, wy);

    sU11 = gpu_sum(d_U11, d_part, h_part, N);
    sU12 = gpu_sum(d_U12, d_part, h_part, N);
    sU22 = gpu_sum(d_U22, d_part, h_part, N);

    if (isnan(sU11) || isnan(sU12) || isnan(sU22)) {
        ret = -1;
        goto cleanup;
    }

    /* Bulk mean-field potentials: Phi_ij^bulk = dA * rho_j_bulk * sum U_ij */
    Phi11b = dA * rho1_b * sU11;
    Phi12b = dA * rho2_b * sU12;
    Phi21b = dA * rho1_b * sU12;
    Phi22b = dA * rho2_b * sU22;

    /* ── coordinate arrays (cell-centred) ── */
    xs = (double *)malloc(Nx * sizeof(double));
    ys = (double *)malloc(Ny * sizeof(double));
    if (!xs || !ys) {
        ret = -1;
        goto cleanup;
    }
    for (int i = 0; i < Nx; ++i) { xs[i] = (i + 0.5) * dx; }
    for (int j = 0; j < Ny; ++j) { ys[j] = (j + 0.5) * dy; }

    /* ── initialise log files ── */
    {
        char log_path[512];
        snprintf(log_path, sizeof(log_path), "%s/convergence.dat", cfg->output_dir);
        if (!enable_db || start_iter == 0) {
            FILE *lf = fopen(log_path, "w");
            if (lf) {
                fprintf(lf, "# iter  L2_error\n");
                fclose(lf);
            }
        }

        char param_path[512];
        snprintf(param_path, sizeof(param_path), "%s/parameters.cfg", cfg->output_dir);
        io_save_parameters(param_path, cfg);
    }

    /* ── upload initial densities and enforce BCs ── */
    CUDA_TRY(cudaMemcpy(d_rho1, rho1, sz, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_rho2, rho2, sz, cudaMemcpyHostToDevice));
    k_boundary<<<g, BLK>>>(d_rho1, d_rho2, Nx, Ny, mode);

    /* Write back so host arrays are consistent with wall BCs. */
    CUDA_TRY(cudaMemcpy(rho1, d_rho1, sz, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(rho2, d_rho2, sz, cudaMemcpyDeviceToHost));

    if (!enable_db || start_iter == 0) {
        save_snap(rho1, rho2, xs, ys, Nx, Ny, 0, cfg->output_dir);
    }
#ifdef USE_DB_ENGINE
    if (enable_db && start_iter == 0) {
        db_snapshot_save(run, rho1, rho2, 0, 1.0, -1.0, cfg);
    }
#endif

    /* ── Picard iteration ── */
    CUDA_TRY(cudaEventCreate(&t_start));
    CUDA_TRY(cudaEventCreate(&t_stop));
    events_created = 1;
    CUDA_TRY(cudaEventRecord(t_start));

    prev_err = -1.0;
    final_err = 0.0;
    converged = 0;

    for (int iter = start_iter; iter < max_it; ++iter) {

        /* Step 1: mean-field potential Phi = U * rho (direct convolution) */
        k_compute_Phi<<<g, BLK>>>(
            d_P11, d_P12, d_P21, d_P22,
            d_rho1, d_rho2,
            d_U11, d_U12, d_U22,
            Nx, Ny, dA, wx, wy);

        /* Step 2: Euler-Lagrange trial density K */
        k_compute_K2_boundary<<<g, BLK>>>(
            d_P11, d_P12, d_P21, d_P22,
            Phi11b, Phi12b, Phi21b, Phi22b,
            rho1_b, rho2_b, beta,
            d_K1, d_K2,
            Nx, Ny, mode);

        /* Step 3: rescale K to conserve total particle number */
        {
            double s1 = gpu_sum_interior(d_K1, d_part, h_part, Nx, Ny, mode);
            double s2 = gpu_sum_interior(d_K2, d_part, h_part, Nx, Ny, mode);

            if (isnan(s1) || isnan(s2)) {
                ret = -1;
                goto cleanup;
            }

            if (interior > 0 && s1 > 1e-12 && s2 > 1e-12) {
                k_scale_interior<<<g, BLK>>>(
                    d_K1, (double)interior * rho1_b / s1, Nx, Ny, mode);
                k_scale_interior<<<g, BLK>>>(
                    d_K2, (double)interior * rho2_b / s2, Nx, Ny, mode);
            }
        }

        /* Step 4: compute weighted L2 error before mixing */
        double err;
        {
            k_sq_diff<<<g, BLK>>>(d_K1, d_rho1, d_tmp, Nx, Ny, mode);
            double s1 = gpu_sum(d_tmp, d_part, h_part, N);
            k_sq_diff<<<g, BLK>>>(d_K2, d_rho2, d_tmp, Nx, Ny, mode);
            double s2 = gpu_sum(d_tmp, d_part, h_part, N);

            if (isnan(s1) || isnan(s2)) {
                ret = -1;
                goto cleanup;
            }

            double ic = (interior > 0) ? (double)interior : 1.0;
            double e1 = xi1 * sqrt(s1 / ic);
            double e2 = xi2 * sqrt(s2 / ic);
            err = (e1 > e2) ? e1 : e2;
        }
        final_err = err;

        /* Step 5: adaptive xi damping if error stagnates */
        double err_delta = (prev_err >= 0.0) ? fabs(prev_err - err) : -1.0;
        if (prev_err >= 0.0) {
            if (err_delta > 0.0 && err_delta < err_thresh) {
                xi1 *= xi_damp;
                xi2 *= xi_damp;
            }
        }
        prev_err = err;

        /* Step 6: Picard mixing and smoothing */
        k_mix2_boundary<<<g, BLK>>>(
            d_rho1, d_rho2, d_K1, d_K2, xi1, xi2, Nx, Ny, mode);

        /* Laplacian smoothing writes to separate buffers to avoid data races. */
        k_smooth<<<g, BLK>>>(d_rho1, d_s1, Nx, Ny, mode, SMOOTH_EPS);
        k_smooth<<<g, BLK>>>(d_rho2, d_s2, Nx, Ny, mode, SMOOTH_EPS);

        /* Swap smoothed buffers into active density pointers. */
        {
            double *tmp = d_rho1; d_rho1 = d_s1; d_s1 = tmp;
            tmp = d_rho2; d_rho2 = d_s2; d_s2 = tmp;
        }

        k_boundary<<<g, BLK>>>(d_rho1, d_rho2, Nx, Ny, mode);

        /* ── periodic logging and snapshots ── */
        {
            char log_path[512];
            snprintf(log_path, sizeof(log_path),
                     "%s/convergence.dat", cfg->output_dir);
            io_log_convergence(log_path, iter, err);
        }

        if ((iter + 1) % sav_ev == 0) {
            CUDA_TRY(cudaMemcpy(rho1, d_rho1, sz, cudaMemcpyDeviceToHost));
            CUDA_TRY(cudaMemcpy(rho2, d_rho2, sz, cudaMemcpyDeviceToHost));

            /* Compute diagnostics on host. */
            double mn1 = rho1[0], mx1 = rho1[0], sm1 = 0.0;
            double mn2 = rho2[0], mx2 = rho2[0], sm2 = 0.0;
            for (int k = 0; k < N; ++k) {
                if (rho1[k] < mn1) { mn1 = rho1[k]; }
                if (rho1[k] > mx1) { mx1 = rho1[k]; }
                sm1 += rho1[k];
                if (rho2[k] < mn2) { mn2 = rho2[k]; }
                if (rho2[k] > mx2) { mx2 = rho2[k]; }
                sm2 += rho2[k];
            }
            printf("  iter %6d  err=%.3e  Δerr=%.3e  xi=[%.4f,%.4f]"
                   "  ρ1[%.4f,%.4f,%.4f]  ρ2[%.4f,%.4f,%.4f]\n",
                   iter + 1, err, err_delta, xi1, xi2,
                   mn1, sm1 / (double)N, mx1,
                   mn2, sm2 / (double)N, mx2);

#ifdef USE_DB_ENGINE
            if (enable_db) {
                db_snapshot_save(run, rho1, rho2, iter + 1, err, err_delta, cfg);
            }
#endif
            save_snap(rho1, rho2, xs, ys, Nx, Ny, iter + 1, cfg->output_dir);
        }

        /* ── convergence check ── */
        if (err < tol) {
            CUDA_TRY(cudaEventRecord(t_stop));
            CUDA_TRY(cudaEventSynchronize(t_stop));
            float ms = 0.0f;
            CUDA_TRY(cudaEventElapsedTime(&ms, t_start, t_stop));
            printf("  Converged at iteration %d  (err=%.3e < tol=%.3e, %.2f s GPU)\n",
                   iter + 1, err, tol, ms / 1000.0f);

            CUDA_TRY(cudaMemcpy(rho1, d_rho1, sz, cudaMemcpyDeviceToHost));
            CUDA_TRY(cudaMemcpy(rho2, d_rho2, sz, cudaMemcpyDeviceToHost));

#ifdef USE_DB_ENGINE
            if (enable_db) {
                db_snapshot_save(run, rho1, rho2, iter + 1, err, err_delta, cfg);
            }
#endif
            save_snap (rho1, rho2, xs, ys, Nx, Ny, iter + 1, cfg->output_dir);
            save_final(rho1, rho2, xs, ys, Nx, Ny, cfg->output_dir);
            converged = 1;
            break;
        }
    }

    /* ── post-loop cleanup ── */
    if (!converged) {
        CUDA_TRY(cudaEventRecord(t_stop));
        CUDA_TRY(cudaEventSynchronize(t_stop));
        float ms = 0.0f;
        CUDA_TRY(cudaEventElapsedTime(&ms, t_start, t_stop));
        fprintf(stderr,
                "Warning: did not converge in %d iterations (%.2f s GPU).\n",
                max_it, ms / 1000.0f);

        CUDA_TRY(cudaMemcpy(rho1, d_rho1, sz, cudaMemcpyDeviceToHost));
        CUDA_TRY(cudaMemcpy(rho2, d_rho2, sz, cudaMemcpyDeviceToHost));
        save_final(rho1, rho2, xs, ys, Nx, Ny, cfg->output_dir);
    }

    ret = converged ? 0 : 1;

cleanup:
    if (events_created) {
        cudaEventDestroy(t_start);
        cudaEventDestroy(t_stop);
    }

    if (d_rho1)  { cudaFree(d_rho1); }
    if (d_rho2)  { cudaFree(d_rho2); }
    if (d_U11)   { cudaFree(d_U11);  }
    if (d_U12)   { cudaFree(d_U12);  }
    if (d_U22)   { cudaFree(d_U22);  }
    if (d_P11)   { cudaFree(d_P11);  }
    if (d_P12)   { cudaFree(d_P12);  }
    if (d_P21)   { cudaFree(d_P21);  }
    if (d_P22)   { cudaFree(d_P22);  }
    if (d_K1)    { cudaFree(d_K1);   }
    if (d_K2)    { cudaFree(d_K2);   }
    if (d_tmp)   { cudaFree(d_tmp);  }
    if (d_s1)    { cudaFree(d_s1);   }
    if (d_s2)    { cudaFree(d_s2);   }
    if (d_part)  { cudaFree(d_part); }
    if (h_part)  { cudaFreeHost(h_part); }
    free(xs);
    free(ys);

    if (final_error_out) {
        *final_error_out = final_err;
    }

    return ret;
}

/* ── public API ───────────────────────────────────────────────────────────── */

extern "C" int solver_run_binary(
    double *rho1, double *rho2, struct SimConfig *cfg)
{
    return solver_run_binary_impl(rho1, rho2, cfg, 0, 0, NULL, NULL);
}

#ifdef USE_DB_ENGINE
extern "C" int solver_run_binary_db(
    double          *rho1,
    double          *rho2,
    struct SimConfig *cfg,
    struct DbRun    *run,
    int              start_iter,
    double          *final_error_out)
{
    return solver_run_binary_impl(
        rho1, rho2, cfg, start_iter, 1, (void *)run, final_error_out);
}
#endif /* USE_DB_ENGINE */