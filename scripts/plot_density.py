#!/usr/bin/env python3
"""
plot_density.py — Visualise a 2D density profile from simulation output.

Usage:
    python3 scripts/plot_density.py output/density_2d.dat

The data file is expected to have three columns:  x  y  rho(x,y)

Future: extend to 3D visualisation with matplotlib's mplot3d or pyvista.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_density.py <data_file>")
        sys.exit(1)

    fname = sys.argv[1]
    data = np.loadtxt(fname)

    x   = data[:, 0]
    y   = data[:, 1]
    rho = data[:, 2]

    # Determine grid dimensions from unique coordinate values
    xu = np.unique(x)
    yu = np.unique(y)
    nx, ny = len(xu), len(yu)

    # Reshape to 2D grid
    Z = rho.reshape((ny, nx))

    plt.figure(figsize=(8, 6))
    plt.imshow(Z, extent=[xu.min(), xu.max(), yu.min(), yu.max()],
               origin='lower', cmap='viridis', aspect='equal')
    plt.colorbar(label=r'$\rho(x, y)$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Density profile')
    plt.tight_layout()
    plt.savefig(fname.replace('.dat', '.png'), dpi=150)
    plt.show()

if __name__ == '__main__':
    main()
