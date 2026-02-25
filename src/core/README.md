# src/core/

Platform-independent infrastructure: configuration parsing and grid creation.

---

## config.c

INI-style `.cfg` parser that populates `SimConfig`.

**How it works**

The file is read line by line.  Comments (everything after `#`) are stripped.
Section headers (`[grid]`, `[physics]`, `[interaction]`, `[solver]`, `[output]`)
set a context variable that routes subsequent `key = value` lines to the
correct struct member.

The `[interaction]` section uses `sscanf` to decode keys of the form
`A_IJ_M` and `a_IJ_M` (species pair IJ \u2208 {11,12,22}, term M \u2208 {1,2,3}) and
fills both `cfg.potential.A[i][j][m]` and `cfg.potential.A[j][i][m]` to
enforce the symmetry condition $A_{ij}^{(m)} = A_{ji}^{(m)}$.

**Key functions**

| Function | Description |
|---|---|
| `config_load(filename, cfg)` | Parse a file and fill `*cfg`. Returns 0 on success. |
| `config_print(cfg)` | Dump all parameters to stdout for verification. |

---

## grid.c

Creates cell-centred coordinate arrays for the 2-D periodic simulation box.

**Cell-centred layout**

Grid points are placed at $x_i = (i + 0.5)\,dx$, $y_j = (j + 0.5)\,dy$
(for $i = 0 \dots n_x - 1$, $j = 0 \dots n_y - 1$).
The $+0.5$ offset ensures that no point sits at the origin, avoiding the
$r = 0$ singularity in the Yukawa potential, and makes the grid naturally
periodic without a duplicated boundary node.

**Key functions**

| Function | Description |
|---|---|
| `grid_create_x(g)` | Allocate and return `x[nx]`. Caller must `free()`. |
| `grid_create_y(g)` | Allocate and return `y[ny]`. Caller must `free()`. |
| `grid_total_points(g)` | Returns `nx * ny`. |
