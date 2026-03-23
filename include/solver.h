/*
 * solver.h — Iterative DFT solver interface
 *
 * Picard iteration for the density profile:
 *   rho_new = (1 - alpha) * rho_old  +  alpha * rho_calc
 *
 * Convergence is checked via L2 norm of (rho_new - rho_old).
 */

#ifndef SOLVER_H
#define SOLVER_H

#include <stddef.h>

/* Solver parameters */
typedef struct {
    int    max_iterations;         /* maximum Picard iterations                   */
    double tolerance;              /* convergence accuracy epsilon                */
    double xi1;                    /* Picard mixing coefficient for species 1     */
    double xi2;                    /* Picard mixing coefficient for species 2     */
    double error_change_threshold; /* threshold for error change to trigger damping */
    double xi_damping_factor;      /* factor to multiply xi by when error stabilizes */
} SolverParams;

/* Forward declaration — full SimConfig defined in config.h */
struct SimConfig;

#ifdef USE_DB_ENGINE
/* Forward declaration for database run handle */
struct DbRun;
#endif

/*
 * solver_run_binary - Picard iteration for a 2-component SALR mixture.
 *
 * rho1 and rho2 are flat nx*ny arrays (input: initial guess,
 * output: converged density profile, row-major: index = iy*nx + ix).
 * Returns 0 on convergence, 1 if max_iterations was reached without
 * convergence, -1 on allocation failure.
 * Intermediate snapshots and the convergence log are written to
 * cfg->output_dir according to cfg->save_every.
 */
int solver_run_binary(double *rho1, double *rho2, struct SimConfig *cfg);

#ifdef USE_DB_ENGINE
/*
 * solver_run_binary_db - Picard iteration with HDF5 snapshot support.
 *
 * Same as solver_run_binary, but saves snapshots to HDF5 files via
 * the database engine. Supports resumption from a checkpoint.
 *
 * run: Database run handle for snapshot storage
 * start_iter: Starting iteration (0 for new run, >0 for resume)
 */
int solver_run_binary_db(double *rho1, double *rho2, struct SimConfig *cfg,
                         struct DbRun *run, int start_iter);
#endif

/* Compute L2 norm of (a - b) over n elements */
double solver_l2_diff(const double *a, const double *b, size_t n);

#endif /* SOLVER_H */
