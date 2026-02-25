#!/usr/bin/env python3
"""
plot_density.py — Visualise SALR DFT solver output using gnuplot.

Usage
-----
  python3 scripts/plot_density.py output/          # final heatmaps + convergence
  python3 scripts/plot_density.py output/density_species1_final.dat
  python3 scripts/plot_density.py output/convergence.dat

Output
------
All plots are saved as PNG files alongside the input data.
Intermediate .gp scripts are written next to the PNG for manual re-use.

Requirements
------------
  gnuplot >= 5.0  (must be on PATH)
"""

import sys
import os
import glob
import subprocess


# ── gnuplot runner ────────────────────────────────────────────────────────────

def run_gnuplot(script: str, gp_path: str):
    """Write script to gp_path and execute it with gnuplot."""
    with open(gp_path, "w") as f:
        f.write(script)
    result = subprocess.run(["gnuplot", gp_path],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"gnuplot error:\n{result.stderr.strip()}")
    return result.returncode


# ── Density range helpers ────────────────────────────────────────────────────

def dat_rho_range(*paths):
    """
    Return (rho_min, rho_max) over all given data files.
    Reads only the third column, skipping comment/blank lines.
    """
    lo, hi = float("inf"), float("-inf")
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    v = float(line.split()[2])
                    if v < lo: lo = v
                    if v > hi: hi = v
                except (IndexError, ValueError):
                    pass
    return lo, hi


# ── Single heatmap ────────────────────────────────────────────────────────────

def plot_density_file(dat_path: str, title: str,
                      png_path: str, gp_path: str,
                      cbmin: float = None, cbmax: float = None):
    """Generate a pm3d heatmap for one density file."""
    cb = ""
    if cbmin is not None and cbmax is not None:
        cb = f"set cbrange [{cbmin:.6g}:{cbmax:.6g}]"

    script = f"""\
set terminal png size 800,700 enhanced font 'Helvetica,13'
set output '{png_path}'

set title '{title}'
set xlabel 'x'
set ylabel 'y'
set cblabel '{chr(961)}(x,y)'

set palette rgbformulae 33,13,10    # viridis-like: dark-blue -> yellow
{cb}

set pm3d map interpolate 0,0
set size ratio -1

splot '{dat_path}' using 1:2:3 with pm3d notitle
"""
    rc = run_gnuplot(script, gp_path)
    if rc == 0:
        print(f"  Saved: {png_path}")


# ── Side-by-side final heatmaps ───────────────────────────────────────────────

def plot_binary_final(output_dir: str):
    """Plot species 1 and species 2 final densities as two separate PNGs."""
    path1 = os.path.join(output_dir, "density_species1_final.dat")
    path2 = os.path.join(output_dir, "density_species2_final.dat")

    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"Final density files not found in '{output_dir}'.")
        return

    cbmin, cbmax = dat_rho_range(path1, path2)

    for dat, label, idx in [(path1, "Species 1", 1), (path2, "Species 2", 2)]:
        png = os.path.join(output_dir, f"density_species{idx}_final.png")
        gp  = os.path.join(output_dir, f"density_species{idx}_final.gp")
        plot_density_file(dat, f"SALR DFT — {label} final density",
                          png, gp, cbmin, cbmax)


# ── Single-file mode ──────────────────────────────────────────────────────────

def plot_single_file(dat_path: str):
    base  = dat_path.replace(".dat", "")
    cbmin, cbmax = dat_rho_range(dat_path)
    plot_density_file(dat_path,
                      title=os.path.basename(dat_path),
                      png_path=base + ".png",
                      gp_path=base + ".gp",
                      cbmin=cbmin, cbmax=cbmax)


# ── Convergence plot ──────────────────────────────────────────────────────────

def plot_convergence(dat_path: str):
    """Plot L2 error vs iteration on a log-y scale."""
    png = dat_path.replace(".dat", ".png")
    gp  = dat_path.replace(".dat", ".gp")

    script = f"""\
set terminal png size 900,500 enhanced font 'Helvetica,13'
set output '{png}'

set title 'Picard iteration convergence'
set xlabel 'Iteration'
set ylabel 'L2 error'
set logscale y
set grid
set key off

plot '{dat_path}' using 1:2 with linespoints pt 7 ps 0.4 lw 1.2 lc rgb '#2171b5' notitle
"""
    rc = run_gnuplot(script, gp)
    if rc == 0:
        print(f"  Saved: {png}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]

    if os.path.isdir(arg):
        plot_binary_final(arg)
        conv = os.path.join(arg, "convergence.dat")
        if os.path.exists(conv):
            plot_convergence(conv)

    elif "convergence" in os.path.basename(arg):
        plot_convergence(arg)

    else:
        plot_single_file(arg)


if __name__ == "__main__":
    main()
