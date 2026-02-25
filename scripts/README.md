# scripts/

Post-processing and visualisation scripts (Python 3 + matplotlib/numpy or gnuplot).

Requirements:
- **Python 3** with **matplotlib** and **numpy** (for `plot_joint_heatmap.py`)
- **gnuplot ≥ 5.0** (for `plot_density.py` and `density_browser.gp`)

⚠️ **Known Issue:** On some systems, gnuplot may crash with:
```
gnuplot: symbol lookup error: libpthread.so.0: undefined symbol: __libc_pthread_init
```
This is a snap package library conflict. Use `plot_joint_heatmap.py` as a workaround.

---

## plot_joint_heatmap.py (NEW - RECOMMENDED)

**Two-color joint visualization** showing both species in a single plot.

### Usage

```bash
python3 scripts/plot_joint_heatmap.py output/
```

### Output

Produces:
- `output/joint_heatmap_final.png` — Three-panel figure:
  1. Species 1 (blue colormap)
  2. Species 2 (red colormap)
  3. Joint overlay (blue=species1, red=species2, purple=both)
- `output/joint_heatmap_last_iter.png` — Same for last iteration snapshot

### Why use this?

- Shows spatial relationship between both species
- Reveals anti-correlation patterns in SALR systems
- Works even when gnuplot has library conflicts
- Professional matplotlib output

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

---

## density_browser.gp

**Interactive frame-by-frame density browser** (pure gnuplot script).

Shows iteration snapshots side-by-side: 3D scatter plots + 2D heatmaps for both species and their sum.

### Usage

```bash
gnuplot scripts/density_browser.gp
```

⚠️ **Known Issues:**
- **Snap library conflict:** May crash with `libpthread.so.0` symbol lookup error  
- **No interactive terminal:** If your gnuplot lacks x11/wxt/qt support, the script will exit with an error

**Workaround:** The script auto-detects available terminals (x11→wxt→qt). If none work, use `plot_joint_heatmap.py` instead.

### Controls

| Key | Action |
|---|---|
| `n` / Right arrow | Next frame (+1) |
| `p` / Left arrow | Previous frame (-1) |
| `]` / Page Down | Jump +10 frames |
| `[` / Page Up | Jump -10 frames |
| `f` / Home | First frame |
| `l` / End | Last frame |
| `c` | Toggle color clipping mode |
| `q` | Quit |

### Display

- **Top row:** 3D scatter plots (species1, species2, sum)
- **Bottom row:** 2D heatmaps (species1, species2, sum)
- **Color modes:**
  - Clipped (default): cbrange = [0, 3×mean] — shows dilute background clearly
  - Auto: cbrange = [min, max] — shows peak values exactly
