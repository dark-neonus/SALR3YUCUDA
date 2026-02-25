#!/usr/bin/env python3
"""
plot_joint_heatmap.py — Two-color joint heatmap for binary SALR mixture.

Creates a visualization where:
  - Species 1 is shown in BLUE tones
  - Species 2 is shown in RED tones
  - Overlap regions appear PURPLE

Usage:
  python3 scripts/plot_joint_heatmap.py output/
  python3 scripts/plot_joint_heatmap.py output/density_species1_iter_000500.dat output/density_species2_iter_000500.dat

Output:
  PNG files with joint heatmaps showing both species.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib


def load_density(path: str) -> tuple:
    """Load a 2D density file and return (x, y, rho) as arrays."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    data = np.array(data)
    x = np.unique(data[:, 0])
    y = np.unique(data[:, 1])
    nx, ny = len(x), len(y)
    
    rho = data[:, 2].reshape(ny, nx)
    return x, y, rho


def normalize_density(rho: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """Normalize density to [0, 1] range."""
    if vmin is None:
        vmin = rho.min()
    if vmax is None:
        vmax = rho.max()
    
    if vmax == vmin:
        return np.zeros_like(rho)
    
    return np.clip((rho - vmin) / (vmax - vmin), 0, 1)


def create_rgb_overlay(rho1_norm: np.ndarray, rho2_norm: np.ndarray) -> np.ndarray:
    """
    Create RGB image where:
    - Species 1 contributes to Blue channel
    - Species 2 contributes to Red channel
    - Both contribute to make purple in overlap regions
    """
    ny, nx = rho1_norm.shape
    rgb = np.zeros((ny, nx, 3))
    
    # Species 1 -> Blue (with some green for visibility)
    rgb[:, :, 2] = rho1_norm  # Blue
    rgb[:, :, 1] = 0.2 * rho1_norm  # Small green component
    
    # Species 2 -> Red (with some green for visibility)
    rgb[:, :, 0] = rho2_norm  # Red
    rgb[:, :, 1] = np.maximum(rgb[:, :, 1], 0.2 * rho2_norm)  # Small green component
    
    # Normalize to prevent oversaturation
    max_val = rgb.max()
    if max_val > 1:
        rgb = rgb / max_val
    
    return rgb


def plot_joint_heatmap(path1: str, path2: str, output_path: str):
    """Create a joint heatmap showing both species in different colors."""
    
    # Load data
    x1, y1, rho1 = load_density(path1)
    x2, y2, rho2 = load_density(path2)
    
    # Normalize each species independently
    rho1_norm = normalize_density(rho1)
    rho2_norm = normalize_density(rho2)
    
    # Create RGB overlay
    rgb = create_rgb_overlay(rho1_norm, rho2_norm)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot species 1 (Blue colormap)
    ax1 = axes[0]
    im1 = ax1.imshow(rho1, origin='lower', extent=[x1.min(), x1.max(), y1.min(), y1.max()],
                     cmap='Blues', aspect='equal')
    ax1.set_title(f'Species 1\nmin={rho1.min():.3f}, max={rho1.max():.3f}', fontsize=11)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, label='ρ₁(x,y)')
    
    # Plot species 2 (Red colormap)
    ax2 = axes[1]
    im2 = ax2.imshow(rho2, origin='lower', extent=[x2.min(), x2.max(), y2.min(), y2.max()],
                     cmap='Reds', aspect='equal')
    ax2.set_title(f'Species 2\nmin={rho2.min():.3f}, max={rho2.max():.3f}', fontsize=11)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, label='ρ₂(x,y)')
    
    # Plot combined (RGB overlay)
    ax3 = axes[2]
    ax3.imshow(rgb, origin='lower', extent=[x1.min(), x1.max(), y1.min(), y1.max()],
               aspect='equal')
    ax3.set_title('Joint Distribution\nBlue=Species1, Red=Species2, Purple=Both', fontsize=11)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    
    # Add legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Species 1 (high ρ₁)'),
        Patch(facecolor='red', label='Species 2 (high ρ₂)'),
        Patch(facecolor='purple', label='Both species'),
        Patch(facecolor='black', label='Low density')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_directory(output_dir: str):
    """Plot final density files from a directory."""
    path1 = os.path.join(output_dir, "density_species1_final.dat")
    path2 = os.path.join(output_dir, "density_species2_final.dat")
    
    if os.path.exists(path1) and os.path.exists(path2):
        out_path = os.path.join(output_dir, "joint_heatmap_final.png")
        plot_joint_heatmap(path1, path2, out_path)
    else:
        print(f"Final density files not found in {output_dir}")
        return
    
    # Also plot intermediate snapshots if available
    import glob
    iter_files1 = sorted(glob.glob(os.path.join(output_dir, "density_species1_iter_*.dat")))
    iter_files2 = sorted(glob.glob(os.path.join(output_dir, "density_species2_iter_*.dat")))
    
    if iter_files1 and iter_files2:
        # Plot last iteration
        plot_joint_heatmap(iter_files1[-1], iter_files2[-1],
                          os.path.join(output_dir, "joint_heatmap_last_iter.png"))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    matplotlib.use('Agg')  # Non-interactive backend
    
    arg = sys.argv[1]
    
    if os.path.isdir(arg):
        plot_directory(arg)
    elif len(sys.argv) >= 3:
        # Two files specified
        plot_joint_heatmap(sys.argv[1], sys.argv[2], "joint_heatmap.png")
    else:
        print("Please provide either a directory or two density files")
        sys.exit(1)


if __name__ == "__main__":
    main()
