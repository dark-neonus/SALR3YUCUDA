# scripts/

Post-processing and visualisation scripts.

## Requirements

- Python 3 with matplotlib and numpy; install via `pip install -r requirements.txt` from the repository root
- gnuplot >= 5.0 (optional, required only for gnuplot-based scripts)

---

## plot_joint_heatmap.py

Two-colour joint visualisation showing both species in a single figure.

```bash
python3 scripts/plot_joint_heatmap.py output/
```

Produces `output/joint_heatmap_final.png` with three panels: species 1 (blue), species 2 (red), and a joint overlay. Also generates `output/joint_heatmap_last_iter.png` for the last saved iteration snapshot.

---

## plot_density.py

Generates gnuplot-based heatmaps and convergence plots. A `.gp` script is saved alongside each `.png` for manual reproduction.

```bash
python3 scripts/plot_density.py output/                              # all outputs
python3 scripts/plot_density.py output/density_species1_final.dat    # single density file
python3 scripts/plot_density.py output/convergence.dat               # convergence log only
```

Expected data format for density files: three columns (`x  y  rho`), blank line between each y-level block (gnuplot pm3d format). Convergence log: two columns (`iteration  L2_error`).

---

## plot_density_3d.py

Interactive 3D scatter plot of both species using gnuplot. Requires a gnuplot build with a graphical terminal (qt, wxt, or x11).
```bash
python3 scripts/plot_density_3d.py output/
python3 scripts/plot_density_3d.py output/density_species1_final.dat
```

---

## performance_real_params.py

Convergence-focused CPU(OpenMP) vs CUDA comparison using physical parameters from `configs/default.cfg`.

What it does:
- Runs CPU solver with all OpenMP threads by default (`OMP_NUM_THREADS = os.cpu_count()`)
- Runs CUDA solver with the same default physics parameters
- Executes each run until convergence or max-iteration stop from config
- Collects wall-time, iterations, final error, and convergence history
- Produces presentation-ready comparison figures and CSV summary

```bash
python3 scripts/performance_real_params.py
```

Output folder: `analysis/results/real_params_cuda_vs_cpu_<timestamp>/`

Key artifacts:
- `performance_real_params.png` and `.svg` (multi-panel professional figure)
- `metrics.csv` (machine-readable results)
- `summary.txt` (human-readable summary)
- `cpu/run_stdout.log`, `cuda/run_stdout.log` (full solver logs)

---

## density_browser.gp

Interactive frame-by-frame gnuplot density browser. Displays iteration snapshots as side-by-side 3D scatter plots and 2D heatmaps for both species and their sum. Requires gnuplot with an interactive terminal (x11, wxt, or qt).

```bash
gnuplot scripts/density_browser.gp
```

Navigation: `n`/`p` for next/previous frame, `]`/`[` to jump by 10, `f`/`l` for first/last, `c` to toggle colour clipping, `q` to quit.

---

## density_browser_merged.gp

Interactive merged-species density browser showing both species in a unified view: a 3D scatter plot (purple for species 1, green for species 2) alongside a 2D heatmap of total density.

```bash
gnuplot scripts/density_browser_merged.gp
gnuplot -e "output_dir='output'" scripts/density_browser_merged.gp
```

Additional controls beyond the standard browser: `t` for transparency mode, `R` for 3D rotation mode, `i` to toggle parameter overlay, `d` to expand/collapse Yukawa coefficient display.
