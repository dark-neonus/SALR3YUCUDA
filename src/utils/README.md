# src/utils/

File I/O utilities for saving simulation output.

---

## io.c

All functions return 0 on success and -1 if `fopen` fails.

### `io_save_density_2d`

Writes a flat `rho[iy * nx + ix]` array as a three-column ASCII file:
```
# x  y  rho
0.1  0.1  0.402
0.3  0.1  0.398
...
```
A blank line is inserted after each row of constant $y$ so the file is
compatible with **gnuplot** `pm3d` and with
`numpy.loadtxt` / `array.reshape`.

### `io_save_density_1d`

Two-column file `x  rho(x)`.  Used when a 1-D slice is needed.

### `io_log_convergence`

Appends one record `iteration  L2_error` to a convergence log.
The file is opened in append mode (`"a"`) so successive calls accumulate
data without truncating earlier entries.
Utility functions — file I/O (io.c), timing helpers.
