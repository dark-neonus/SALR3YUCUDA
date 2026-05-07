#!/usr/bin/env python3
"""
generate_report_plots.py — Scientific plot generator for SALR3YUCUDA report.

Creates combined 2D heatmap + 3D surface figures for each simulation scenario.
Each figure has four panels:
  [Species 1 — 2D heatmap]  [Species 2 — 2D heatmap]
  [Species 1 — 3D surface]  [Species 2 — 3D surface]

Usage:
    python3 scripts/generate_report_plots.py <output_dir> <plot_name> [--title "..."] [--params "..."]

Arguments:
    output_dir  — directory containing data/density_species{1,2}_final.dat
    plot_name   — base name for output file (e.g. pbc-random)
    --dest      — destination directory for the PNG (default: current dir)
    --title     — figure title (e.g. "PBC, random initial conditions")
    --params    — parameter annotation (e.g. "T=8.0, rho1=0.2, rho2=0.2, N=160x160")
    --dpi       — output resolution in DPI (default: 200)

Output:
    <dest>/<plot_name>.png
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — activates 3D projection


# ── Data loading ──────────────────────────────────────────────────────────────

def load_density_2d(path: str):
    """
    Load a SALR DFT output file (3-column x y rho).
    Returns (x_unique, y_unique, rho_2d) where rho_2d has shape (ny, nx).
    """
    data = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                data.append((float(parts[0]), float(parts[1]), float(parts[2])))

    data = np.array(data)
    xs = np.unique(data[:, 0])
    ys = np.unique(data[:, 1])
    nx, ny = len(xs), len(ys)
    rho = data[:, 2].reshape(ny, nx)
    return xs, ys, rho


# ── Figure builder ────────────────────────────────────────────────────────────

_CMAP1 = "Blues"     # Species 1 — blue palette (default)
_CMAP2 = "Reds"      # Species 2 — red palette (default)


def _plot_2d_panel(ax, xs, ys, rho, cmap, species_label, vmin=None, vmax=None, trim=0):
    """Draw a 2D heatmap panel showing δρ = ρ − ⟨ρ⟩ (density modulation)."""
    idx = "1" if "1" in species_label else "2"

    # Optionally trim boundary cells that have forced BC values (wall modes)
    if trim > 0:
        rho = rho[trim:-trim, trim:-trim]
        xs = xs[trim:-trim]
        ys = ys[trim:-trim]

    rho_mean = rho.mean()
    delta = rho - rho_mean

    # Robust colorscale: use 99th-percentile of |δρ| to suppress any remaining outliers
    dv = np.percentile(np.abs(delta), 99)
    if dv == 0:
        dv = max(abs(delta.min()), abs(delta.max()))
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    im = ax.imshow(
        delta,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="equal",
        vmin=-dv,
        vmax=dv,
        interpolation="bilinear",
    )
    ax.set_title(f"{species_label} — 2D density modulation", fontsize=10, fontweight="bold")
    ax.set_xlabel(r"$x / \sigma$", fontsize=9)
    ax.set_ylabel(r"$y / \sigma$", fontsize=9)
    ax.tick_params(labelsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(rf"$\delta\rho_{idx}(x,y)$", fontsize=9)
    cb.ax.tick_params(labelsize=7)
    cb.formatter.set_powerlimits((-2, 2))
    cb.update_ticks()


def _plot_3d_panel(ax3d, xs, ys, rho, cmap, species_label, vmin=None, vmax=None,
                   trim=0, view_elev=30, view_azim=-60):
    """
    Draw a 3D scatter plot with z = δρ(x,y) = ρ(x,y) − ⟨ρ⟩.
    Using the deviation from mean removes the large DC offset (e.g. 0.2)
    and makes the pattern structure clearly visible on the z-axis.
    """
    idx = "1" if "1" in species_label else "2"

    # Optionally trim boundary cells that have forced BC values (wall modes)
    if trim > 0:
        rho = rho[trim:-trim, trim:-trim]
        xs = xs[trim:-trim]
        ys = ys[trim:-trim]
        if vmin is not None and vmax is not None:
            vmin, vmax = rho.min(), rho.max()

    # Density deviation from spatial mean — physically meaningful modulation
    rho_mean = rho.mean()
    delta = rho - rho_mean

    # Sub-sample so individual points are clearly visible (~35×35 = ~1225 pts)
    target = 35
    sx = max(1, len(xs) // target)
    sy = max(1, len(ys) // target)
    xs_s = xs[::sx]
    ys_s = ys[::sy]

    X, Y = np.meshgrid(xs_s, ys_s)
    xf, yf, zf = X.ravel(), Y.ravel(), delta[::sy, ::sx].ravel()

    # Colour by δρ with a symmetric percentile-clipped normalization so that
    # any remaining boundary-adjacent outliers saturate at the edge colour
    # rather than dominating the full scale.
    delta_s = delta[::sy, ::sx]
    clim = np.percentile(np.abs(delta), 99)
    if clim == 0:
        clim = max(abs(delta.min()), abs(delta.max()))
    norm = plt.Normalize(vmin=-clim, vmax=clim)
    colors = plt.get_cmap(cmap)(norm(delta_s.ravel()))

    ax3d.scatter(xf, yf, zf, c=colors, s=16, marker="o",
                 depthshade=True, alpha=0.85, edgecolors="none")

    ax3d.set_title(f"{species_label} — 3D scatter (δρ)", fontsize=10, fontweight="bold")
    ax3d.set_xlabel(r"$x / \sigma$", fontsize=8, labelpad=2)
    ax3d.set_ylabel(r"$y / \sigma$", fontsize=8, labelpad=2)
    ax3d.set_zlabel(rf"$\delta\rho_{idx}(x,y)$", fontsize=8, labelpad=4)
    ax3d.tick_params(labelsize=7)
    ax3d.view_init(elev=view_elev, azim=view_azim)

    # Symmetric z-axis: use 99th percentile to suppress outliers
    zlim = np.percentile(np.abs(delta), 99)
    if zlim == 0:
        zlim = max(abs(delta.min()), abs(delta.max()))
    pad = zlim * 0.15 if zlim > 0 else 1e-8
    ax3d.set_zlim(-zlim - pad, zlim + pad)

    # Use scientific notation on z-axis if range is tiny
    ax3d.zaxis.get_major_formatter().set_powerlimits((-2, 2))


def make_combined_figure(
    path1: str,
    path2: str,
    title: str,
    params_text: str,
    dest_path: str,
    dpi: int = 200,
    cmap1: str = _CMAP1,
    cmap2: str = _CMAP2,
    trim: int = 0,
):
    """
    Create a 2×2 combined figure:
      [Species 1 — 2D heatmap]  [Species 2 — 2D heatmap]
      [Species 1 — 3D surface]  [Species 2 — 3D surface]

    path1, path2 : density dat files for species 1 and 2
    title        : overall figure title
    params_text  : parameter annotation shown below the title
    dest_path    : output PNG path
    dpi          : output resolution
    """
    xs1, ys1, rho1 = load_density_2d(path1)
    xs2, ys2, rho2 = load_density_2d(path2)

    # Shared colour scale for 3D scatter colours (absolute ρ per species)
    vmin1, vmax1 = rho1.min(), rho1.max()
    vmin2, vmax2 = rho2.min(), rho2.max()

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    if params_text:
        fig.text(
            0.5, 0.955,
            params_text,
            ha="center", va="top",
            fontsize=8.5,
            color="#444444",
            style="italic",
        )

    # -- Row 1: 2D heatmaps (δρ) --
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    _plot_2d_panel(ax1, xs1, ys1, rho1, cmap1, "Species 1", trim=trim)
    _plot_2d_panel(ax2, xs2, ys2, rho2, cmap2, "Species 2", trim=trim)

    # -- Row 2: 3D scatter (δρ) --
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    _plot_3d_panel(ax3, xs1, ys1, rho1, cmap1, "Species 1", vmin1, vmax1, trim=trim)
    _plot_3d_panel(ax4, xs2, ys2, rho2, cmap2, "Species 2", vmin2, vmax2, trim=trim)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(dest_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {dest_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate scientific combined 2D+3D density plots for the report."
    )
    parser.add_argument("output_dir", help="Directory containing data/density_species*_final.dat")
    parser.add_argument("plot_name", help="Output file base name (e.g. pbc-random)")
    parser.add_argument("--dest", default=".", help="Destination directory for the PNG")
    parser.add_argument(
        "--title",
        default="",
        help='Figure title (e.g. "PBC, random initial conditions")',
    )
    parser.add_argument(
        "--params",
        default="",
        help='Parameter annotation text',
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200)")
    parser.add_argument("--cmap1", default=_CMAP1, help="Colormap for species 1 (default: Blues)")
    parser.add_argument("--cmap2", default=_CMAP2, help="Colormap for species 2 (default: Reds)")
    parser.add_argument("--trim", type=int, default=0,
                        help="Trim N boundary cells from each edge (use for wall BC modes)")
    args = parser.parse_args()

    data_dir = os.path.join(args.output_dir, "data")
    path1 = os.path.join(data_dir, "density_species1_final.dat")
    path2 = os.path.join(data_dir, "density_species2_final.dat")

    for p in (path1, path2):
        if not os.path.exists(p):
            sys.exit(f"Error: file not found: {p}")

    os.makedirs(args.dest, exist_ok=True)
    dest_path = os.path.join(args.dest, f"{args.plot_name}.png")

    make_combined_figure(
        path1=path1,
        path2=path2,
        title=args.title or args.plot_name,
        params_text=args.params,
        dest_path=dest_path,
        dpi=args.dpi,
        cmap1=args.cmap1,
        cmap2=args.cmap2,
        trim=args.trim,
    )


if __name__ == "__main__":
    main()
