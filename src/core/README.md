# src/core/

Platform-independent infrastructure: configuration parsing and grid creation.

---

## config.c

INI-style `.cfg` parser that populates the `SimConfig` structure.

The file is read line by line. Comments (everything after `#`) are stripped. Section headers (`[grid]`, `[physics]`, `[interaction]`, `[solver]`, `[output]`) set a context variable that routes subsequent `key = value` lines to the corresponding struct member.

The `[interaction]` section decodes keys of the form `A_IJ_M` and `a_IJ_M` (species pair IJ in {11, 12, 22}, term M in {1, 2, 3}) and fills both `cfg.potential.A[i][j][m]` and `cfg.potential.A[j][i][m]` to enforce the symmetry condition $A_{ij}^{(m)} = A_{ji}^{(m)}$.

| Function | Description |
|---|---|
| `config_load(filename, cfg)` | Parse a configuration file and populate `*cfg`. Returns 0 on success. |
| `config_print(cfg)` | Print all parameters to stdout for verification. |

---

## grid.c

Creates cell-centred coordinate arrays for the 2D simulation box.

Grid points are placed at $x_i = (i + 0.5)\,\Delta x$, $y_j = (j + 0.5)\,\Delta y$ for $i = 0, \ldots, N_x - 1$ and $j = 0, \ldots, N_y - 1$. The half-cell offset avoids the $r = 0$ singularity in the Yukawa potential and produces a grid that is naturally periodic without a duplicated boundary node.

| Function | Description |
|---|---|
| `grid_create_x(g)` | Allocate and return `x[nx]`. Caller must `free()`. |
| `grid_create_y(g)` | Allocate and return `y[ny]`. Caller must `free()`. |
| `grid_total_points(g)` | Return `nx * ny`. |
