# src/utils/

File I/O utilities for saving simulation output.

---

## io.c

All functions return 0 on success and -1 if `fopen` fails.

| Function | Description |
|---|---|
| `io_save_density_2d` | Write a flat `rho[iy * nx + ix]` array as a three-column ASCII file (`x  y  rho`). A blank line is inserted after each constant-y row for gnuplot pm3d compatibility. |
| `io_save_density_1d` | Write a two-column file (`x  rho(x)`) for 1D slices. |
| `io_log_convergence` | Append one `iteration  L2_error` record to a convergence log (opened in append mode). |
