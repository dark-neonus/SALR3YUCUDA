#!/usr/bin/env python3
"""
benchmark_cuda_cpu.py - Professional benchmark analysis for CUDA vs CPU performance

Generates publication-quality plots comparing:
- CUDA vs CPU solver performance at various grid sizes
- CPU scaling with different OpenMP thread counts
- Speedup analysis and efficiency metrics

Output: SVG and PNG (300 DPI) suitable for presentations and publications.
"""

import os
import sys
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
from pathlib import Path
from datetime import datetime

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})

COLORS = {
    'cuda': '#2196F3',
    'speedup': '#9C27B0',
    'efficiency': '#00BCD4',
}


def get_cpu_color(thread_count, all_threads):
    """
    Generate distinct colors for CPU thread counts using a perceptually uniform colormap.
    
    Args:
        thread_count: The specific thread count to get color for
        all_threads: List of all thread counts being plotted
        
    Returns:
        Hex color string
    """
    import matplotlib.cm as cm
    
    # Predefined colors for common thread counts (green to red gradient)
    base_colors = {
        1: '#4CAF50',   # Green
        2: '#8BC34A',   # Light green
        4: '#FFC107',   # Amber
        8: '#FF9800',   # Orange
        16: '#F44336',  # Red
    }
    
    if thread_count in base_colors:
        return base_colors[thread_count]
    
    # For other thread counts, use a colormap to ensure distinct colors
    sorted_threads = sorted(all_threads)
    if thread_count not in sorted_threads:
        return '#666666'  # Fallback gray
    
    idx = sorted_threads.index(thread_count)
    # Use tab20 colormap which has 20 distinct colors
    cmap = cm.get_cmap('tab20')
    color_idx = idx % 20
    rgba = cmap(color_idx)
    # Convert RGBA to hex
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))



def find_latest_benchmark():
    """
    Find the latest benchmark CSV file in analysis/results/ subdirectories.
    Returns (csv_path, output_dir) or (None, None) if not found.
    """
    results_dir = Path('analysis/results')
    if not results_dir.exists():
        return None, None
    
    # Find all benchmark CSV files
    csv_files = list(results_dir.glob('*/benchmark_*.csv'))
    
    if not csv_files:
        return None, None
    
    # Sort by modification time, get latest
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    # Output directory is same as input directory
    output_dir = latest_csv.parent
    
    return latest_csv, output_dir


def load_benchmark_data(filepath):
    """Load pre-generated benchmark data from CSV file."""
    data = {'grid_sizes': [], 'cuda': {}, 'cpu': {}}
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        header = f.readline().strip().split(',')
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            
            grid_size = int(parts[0])
            data['grid_sizes'].append(grid_size)
            
            for i, col in enumerate(header[1:], 1):
                if i >= len(parts) or not parts[i]:
                    continue
                val = float(parts[i])
                
                if 'cuda' in col.lower():
                    data['cuda'][grid_size] = val
                elif 'cpu' in col.lower():
                    threads_match = re.search(r'(\d+)', col)
                    threads = int(threads_match.group(1)) if threads_match else 1
                    
                    if threads not in data['cpu']:
                        data['cpu'][threads] = {}
                    data['cpu'][threads][grid_size] = val
    
    return data


def plot_performance_comparison(data, output_dir):
    """
    Plot execution time vs grid size for CUDA and CPU configurations.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grid_sizes = np.array(sorted(data['grid_sizes']))
    all_threads = list(data['cpu'].keys())
    
    if data['cuda']:
        cuda_times = [data['cuda'].get(n, np.nan) for n in grid_sizes]
        ax.plot(grid_sizes, cuda_times, 'o-', color=COLORS['cuda'], 
                linewidth=2.5, markersize=8, label='CUDA GPU', zorder=5)
    
    for threads in sorted(all_threads):
        cpu_times = [data['cpu'][threads].get(n, np.nan) for n in grid_sizes]
        color = get_cpu_color(threads, all_threads)
        ax.plot(grid_sizes, cpu_times, 's--', color=color, 
                linewidth=2, markersize=7, label=f'CPU ({threads} thread{"s" if threads > 1 else ""})')
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(grid_sizes)
    ax.set_xticklabels([str(n) for n in grid_sizes])
    
    ax.set_xlabel('Grid Size (N × N)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('SALR DFT Solver Performance: CUDA vs CPU')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'performance_comparison.png', bbox_inches='tight')
    fig.savefig(output_dir / 'performance_comparison.svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: performance_comparison.png/svg")


def plot_speedup(data, output_dir):
    """
    Plot CUDA speedup over CPU configurations.
    """
    if not data['cuda'] or not data['cpu']:
        print("  Skipping speedup plot (missing data)")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    grid_sizes = np.array(sorted(data['grid_sizes']))
    all_threads = list(data['cpu'].keys())
    
    for threads in sorted(all_threads):
        speedups = []
        valid_sizes = []
        
        for n in grid_sizes:
            cuda_time = data['cuda'].get(n)
            cpu_time = data['cpu'][threads].get(n)
            
            if cuda_time and cpu_time and cuda_time > 0:
                speedups.append(cpu_time / cuda_time)
                valid_sizes.append(n)
        
        if speedups:
            color = get_cpu_color(threads, all_threads)
            ax1.plot(valid_sizes, speedups, 'o-', color=color, 
                     linewidth=2.5, markersize=8, 
                     label=f'vs CPU {threads} thread{"s" if threads > 1 else ""}')
    
    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.7, label='Baseline (1×)')
    ax1.set_xscale('log', base=2)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_xticks(grid_sizes)
    ax1.set_xticklabels([str(n) for n in grid_sizes])
    
    ax1.set_xlabel('Grid Size (N × N)')
    ax1.set_ylabel('Speedup (CPU time / CUDA time)')
    ax1.set_title('CUDA Speedup over CPU')
    ax1.legend(loc='upper left', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    if 1 in data['cpu']:
        base_times = {n: data['cpu'][1].get(n) for n in grid_sizes if data['cpu'][1].get(n)}
        
        if base_times:
            mt_threads = [t for t in all_threads if t > 1]
            for threads in sorted(mt_threads):
                speedups = []
                valid_sizes = []
                
                for n in grid_sizes:
                    base = base_times.get(n)
                    curr = data['cpu'][threads].get(n)
                    
                    if base and curr and curr > 0:
                        speedups.append(base / curr)
                        valid_sizes.append(n)
                
                if speedups:
                    color = get_cpu_color(threads, all_threads)
                    ax2.plot(valid_sizes, speedups, 's-', color=color,
                             linewidth=2, markersize=7,
                             label=f'{threads} threads')
            
            if mt_threads:
                max_threads = max(mt_threads)
                ax2.plot(grid_sizes, [max_threads] * len(grid_sizes), 
                         '--', color='gray', alpha=0.5, label=f'Ideal ({max_threads}×)')
    
    ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xscale('log', base=2)
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.set_xticks(grid_sizes)
    ax2.set_xticklabels([str(n) for n in grid_sizes])
    
    ax2.set_xlabel('Grid Size (N × N)')
    ax2.set_ylabel('Speedup vs Single Thread')
    ax2.set_title('CPU Parallel Scaling (OpenMP)')
    ax2.legend(loc='upper left', framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'speedup_analysis.png', bbox_inches='tight')
    fig.savefig(output_dir / 'speedup_analysis.svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: speedup_analysis.png/svg")


def plot_efficiency(data, output_dir):
    """
    Plot parallel efficiency (speedup / threads).
    """
    if 1 not in data['cpu'] or len(data['cpu']) < 2:
        print("  Skipping efficiency plot (insufficient CPU data)")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    grid_sizes = np.array(sorted(data['grid_sizes']))
    base_times = {n: data['cpu'][1].get(n) for n in grid_sizes if data['cpu'][1].get(n)}
    
    thread_list = sorted([t for t in data['cpu'].keys() if t > 1])
    all_threads = list(data['cpu'].keys())
    
    width = 0.15
    x = np.arange(len(grid_sizes))
    
    for i, threads in enumerate(thread_list):
        efficiencies = []
        
        for n in grid_sizes:
            base = base_times.get(n)
            curr = data['cpu'][threads].get(n)
            
            if base and curr and curr > 0:
                speedup = base / curr
                efficiency = speedup / threads * 100
                efficiencies.append(efficiency)
            else:
                efficiencies.append(0)
        
        color = get_cpu_color(threads, all_threads)
        
        offset = (i - len(thread_list)/2 + 0.5) * width
        bars = ax.bar(x + offset, efficiencies, width, label=f'{threads} threads',
                      color=color, edgecolor='white', linewidth=0.5)
    
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Ideal (100%)')
    
    ax.set_xlabel('Grid Size (N × N)')
    ax.set_ylabel('Parallel Efficiency (%)')
    ax.set_title('CPU OpenMP Parallel Efficiency')
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in grid_sizes])
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_ylim(0, 120)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'parallel_efficiency.png', bbox_inches='tight')
    fig.savefig(output_dir / 'parallel_efficiency.svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: parallel_efficiency.png/svg")


def plot_scaling_analysis(data, output_dir):
    """
    Plot computational complexity analysis.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    grid_sizes = np.array(sorted(data['grid_sizes']))
    n_squared = grid_sizes ** 2
    
    if data['cuda']:
        cuda_times = np.array([data['cuda'].get(n, np.nan) for n in grid_sizes])
        valid = ~np.isnan(cuda_times)
        
        if np.sum(valid) > 1:
            ax1.loglog(n_squared[valid], cuda_times[valid], 'o-', color=COLORS['cuda'],
                       linewidth=2.5, markersize=8, label='CUDA measured')
            
            coeffs = np.polyfit(np.log(n_squared[valid]), np.log(cuda_times[valid]), 1)
            fit_times = np.exp(coeffs[1]) * n_squared ** coeffs[0]
            ax1.loglog(n_squared, fit_times, '--', color=COLORS['cuda'], alpha=0.5,
                       label=f'CUDA fit (O(N^{coeffs[0]:.2f}))')
    
    if 1 in data['cpu']:
        cpu_times = np.array([data['cpu'][1].get(n, np.nan) for n in grid_sizes])
        valid = ~np.isnan(cpu_times)
        all_threads = list(data['cpu'].keys())
        
        if np.sum(valid) > 1:
            ax1.loglog(n_squared[valid], cpu_times[valid], 's-', color=get_cpu_color(1, all_threads),
                       linewidth=2, markersize=7, label='CPU (1 thread) measured')
            
            coeffs = np.polyfit(np.log(n_squared[valid]), np.log(cpu_times[valid]), 1)
            fit_times = np.exp(coeffs[1]) * n_squared ** coeffs[0]
            ax1.loglog(n_squared, fit_times, '--', color=get_cpu_color(1, all_threads), alpha=0.5,
                       label=f'CPU fit (O(N^{coeffs[0]:.2f}))')
    
    ax1.set_xlabel('Problem Size (N²)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Computational Complexity Analysis')
    ax1.legend(loc='upper left', framealpha=0.95)
    ax1.grid(True, which='both', alpha=0.3)
    
    if data['cuda'] and 1 in data['cpu']:
        cuda_times = np.array([data['cuda'].get(n, np.nan) for n in grid_sizes])
        cpu_times = np.array([data['cpu'][1].get(n, np.nan) for n in grid_sizes])
        all_threads = list(data['cpu'].keys())
        
        valid = ~(np.isnan(cuda_times) | np.isnan(cpu_times))
        
        if np.sum(valid) > 0:
            throughput_cuda = n_squared[valid] / cuda_times[valid]
            throughput_cpu = n_squared[valid] / cpu_times[valid]
            
            ax2.semilogy(grid_sizes[valid], throughput_cuda, 'o-', color=COLORS['cuda'],
                         linewidth=2.5, markersize=8, label='CUDA')
            ax2.semilogy(grid_sizes[valid], throughput_cpu, 's-', color=get_cpu_color(1, all_threads),
                         linewidth=2, markersize=7, label='CPU (1 thread)')
    
    ax2.set_xlabel('Grid Size (N × N)')
    ax2.set_ylabel('Throughput (grid points / second)')
    ax2.set_title('Computational Throughput')
    ax2.legend(loc='upper left', framealpha=0.95)
    ax2.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'scaling_analysis.png', bbox_inches='tight')
    fig.savefig(output_dir / 'scaling_analysis.svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: scaling_analysis.png/svg")


def generate_summary_table(data, output_dir):
    """Generate a summary table of benchmark results."""
    grid_sizes = sorted(data['grid_sizes'])
    cpu_threads = sorted(data.get('cpu', {}).keys())
    
    # Build header dynamically
    header_parts = ["Grid Size", "CUDA (s)"]
    for t in cpu_threads:
        header_parts.append(f"CPU {t}T (s)")
    header_parts.append("CUDA Speedup")
    
    # Calculate column widths
    col_widths = [max(10, len(h)) for h in header_parts]
    
    # Build header string
    header = " | ".join(f"{h:^{w}}" for h, w in zip(header_parts, col_widths))
    separator = "-" * len(header)
    
    lines = [
        "# SALR DFT Solver Benchmark Summary",
        f"# Generated: {datetime.now().isoformat()}",
        "#",
        "# " + header,
        "# " + separator,
    ]
    
    for n in grid_sizes:
        cuda_t = data['cuda'].get(n, np.nan)
        cpu1_t = data['cpu'].get(1, {}).get(n, np.nan)
        
        # Build row
        row_parts = [f"{n}x{n}"]
        row_parts.append(f"{cuda_t:.3f}" if not np.isnan(cuda_t) else "N/A")
        
        for t in cpu_threads:
            cpu_t = data['cpu'].get(t, {}).get(n, np.nan)
            row_parts.append(f"{cpu_t:.3f}" if not np.isnan(cpu_t) else "N/A")
        
        speedup = cpu1_t / cuda_t if (not np.isnan(cuda_t) and not np.isnan(cpu1_t) and cuda_t > 0) else np.nan
        row_parts.append(f"{speedup:.1f}x" if not np.isnan(speedup) else "N/A")
        
        # Format row with proper spacing
        row = " | ".join(f"{p:^{w}}" for p, w in zip(row_parts, col_widths))
        lines.append("  " + row)
    
    summary = "\n".join(lines)
    
    with open(output_dir / 'benchmark_summary.txt', 'w') as f:
        f.write(summary)
    
    print(f"  Saved: benchmark_summary.txt")
    return summary


def main():
    """Main function with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate benchmark plots from SALR DFT data')
    parser.add_argument('--input', '-i', type=str,
                       help='Input CSV file (auto-detects latest if not specified)')
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Output directory (uses same as input if not specified)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("SALR DFT Benchmark Analysis & Plotting")
    print("=" * 50)
    
    # Find data file
    if args.input:
        data_file = Path(args.input)
        output_dir = Path(args.output_dir) if args.output_dir else data_file.parent
    else:
        data_file, output_dir = find_latest_benchmark()
        if not data_file:
            print("\nNo benchmark data found in analysis/results/")
            print("Run 'python scripts/benchmarking/run_benchmarks.py' first to generate data.")
            return 1
        print(f"\nAuto-detected latest benchmark: {data_file}")
    
    if not data_file.exists():
        print(f"\nError: Data file not found: {data_file}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print(f"\nLoading benchmark data from {data_file}...")
    data = load_benchmark_data(data_file)
    
    if not data or not data['grid_sizes']:
        print("Error: No valid benchmark data available")
        return 1
    
    print(f"\nData summary:")
    print(f"  Grid sizes: {sorted(data['grid_sizes'])}")
    print(f"  CUDA data points: {len(data.get('cuda', {}))}")
    print(f"  CPU configurations: {sorted(data.get('cpu', {}).keys())}")
    
    print("\nGenerating plots...")
    
    plot_performance_comparison(data, output_dir)
    plot_speedup(data, output_dir)
    plot_efficiency(data, output_dir)
    plot_scaling_analysis(data, output_dir)
    
    print("\nGenerating summary...")
    summary = generate_summary_table(data, output_dir)
    print("\n" + summary)
    
    print(f"\nAll plots saved to: {output_dir}/")
    print("Done!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
