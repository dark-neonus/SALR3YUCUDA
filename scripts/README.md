# scripts/

Post-processing and visualisation scripts (Python 3 + gnuplot).

Requirements: **gnuplot ≥ 5.0** and **Python 3** (no external Python packages needed).

---

## plot_density.py

Visualises solver output by generating gnuplot scripts and executing them.
For each plot a `.gp` script is saved alongside the `.png` so plots can be
reproduced or tweaked manually with `gnuplot <file>.gp`.

### Modes

**Directory mode** (recommended after a full run):
```bash
python3 scripts/plot_density.py output/
```
Produces:
- `output/density_species1_final.png` — pm3d heatmap of species 1
- `output/density_species2_final.png` — pm3d heatmap of species 2 (shared colour scale)
- `output/convergence.png` — L2 error vs iteration (log-y)

**Single density file**:
```bash
python3 scripts/plot_density.py output/density_species1_final.dat
```

**Convergence log only**:
```bash
python3 scripts/plot_density.py output/convergence.dat
```

### Gnuplot scripts

Every invocation writes a `.gp` file next to the output PNG.
To re-run a plot without Python:
```bash
gnuplot output/density_species1_final.gp
```

### Data format expected

Density files: three columns `x  y  rho`, comment lines start with `#`,
blank line between each $y$-level block (standard gnuplot `pm3d` format).

Convergence log: two columns `iteration  L2_error`, header line starts with `#`.

---

## plot_density_3d.py

Launches an **interactive gnuplot window** with a 3-D scatter plot that shows
both species in a single view (x, y, ρ).  The window stays open and responds
to mouse input.

Requirements: **gnuplot-qt** (or gnuplot-wxt / gnuplot-x11 as fallback).
The terminal is detected automatically; override with `GNUPLOT_TERM=wxt`.

### Usage

**Directory mode** (plots the final density files):
```bash
python3 scripts/plot_density_3d.py output/
```

**Explicit pair of files** (e.g. a specific iteration):
```bash
python3 scripts/plot_density_3d.py \
    output/density_species1_iter_001000.dat \
    output/density_species2_iter_001000.dat
```

**Single file** (one species only):
```bash
python3 scripts/plot_density_3d.py output/density_species1_final.dat
```

### Mouse controls (qt/wxt terminal)

| Action | Effect |
|---|---|
| Left-drag | Rotate the 3-D view |
| Scroll wheel | Zoom in/out |
| Right-click | Context menu (reset, export, …) |
| `r` key | Reset view to default rotation |
| `q` key | Close window and exit |
