#!/usr/bin/env python3
"""
plot_cluster_vs_laptop.py  —  Compare SALR DFT benchmark results across machines.

Inputs (CSV files produced by run_benchmarks.py):
    --laptop   PATH   CSV from laptop benchmark
    --epyc     PATH   CSV from cluster AMD EPYC 9754 benchmark
    --tr       PATH   CSV from cluster Ryzen Threadripper benchmark   (optional)

Each CSV has columns:
    grid_size, [cuda,] cpu_1, cpu_2, cpu_4, ..., cpu_N

Output plots (saved next to the first provided CSV):
    1. best_cpu_comparison.png/svg   — best-thread time at each grid size
    2. thread_scaling.png/svg        — time vs threads at fixed grid size
    3. speedup_vs_laptop.png/svg     — speedup of each machine vs laptop 1-thread
    4. gpu_comparison.png/svg        — CUDA times comparison (if CUDA columns present)

Usage example:
    python3 scripts/plotting/plot_cluster_vs_laptop.py \\
        --laptop analysis/results/13th_Gen_Intel.../benchmark_*.csv \\
        --epyc   analysis/results/AMD_EPYC_9754.../benchmark_*.csv  \\
        --output analysis/results/cluster_vs_laptop/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────
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

MACHINE_STYLES: Dict[str, dict] = {
    'laptop':     {'color': '#2196F3', 'marker': 'o', 'ls': '-'},
    'epyc':       {'color': '#E65100', 'marker': 's', 'ls': '-'},
    'tr':         {'color': '#388E3C', 'marker': '^', 'ls': '-'},
    'laptop_gpu': {'color': '#9C27B0', 'marker': 'D', 'ls': '--'},
    'epyc_gpu':   {'color': '#F4511E', 'marker': 'P', 'ls': '--'},
    'tr_gpu':     {'color': '#00897B', 'marker': '*', 'ls': '--'},
}

MACHINE_LABELS: Dict[str, str] = {
    'laptop':     'Laptop (i7-13700H)',
    'epyc':       'Cluster EPYC 9754',
    'tr':         'Cluster Threadripper 5975WX',
    'laptop_gpu': 'Laptop RTX 4060',
    'epyc_gpu':   'Cluster RTX (EPYC node)',
    'tr_gpu':     'Cluster RTX 5090',
}


# ── Data loading ──────────────────────────────────────────────────────────────

BenchData = Dict[str, object]  # {'grid_sizes', 'cuda', 'cpu': {threads: {nx: time}}}


def load_csv(path: Path) -> BenchData:
    """Load a benchmark CSV into a structured dict."""
    grid_sizes: List[int] = []
    cuda: Dict[int, float] = {}          # nx -> time
    cpu: Dict[int, Dict[int, float]] = {}  # threads -> (nx -> time)

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            nx = int(row['grid_size'])
            grid_sizes.append(nx)

            if 'cuda' in row and row['cuda']:
                cuda[nx] = float(row['cuda'])

            for key, val in row.items():
                if key.startswith('cpu_') and val:
                    t = int(key.split('_')[1])
                    cpu.setdefault(t, {})[nx] = float(val)

    return {
        'grid_sizes': sorted(set(grid_sizes)),
        'cuda': cuda,
        'cpu': cpu,
    }


def best_cpu_time(data: BenchData, nx: int) -> Tuple[Optional[float], int]:
    """Return (best_time, best_thread_count) for grid size nx."""
    best_t: Optional[float] = None
    best_threads = 1
    for threads, times in data['cpu'].items():
        t = times.get(nx)
        if t is not None and (best_t is None or t < best_t):
            best_t = t
            best_threads = threads
    return best_t, best_threads


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        p = out_dir / f'{stem}.{ext}'
        fig.savefig(p, bbox_inches='tight')
        print(f'  Saved: {p}')


# ── helpers ───────────────────────────────────────────────────────────────────

def _thread_axis(ax: plt.Axes, threads: List[int]) -> None:
    """Set x-axis to show actual thread numbers (not powers of 2)."""
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])


def _grid_axis(ax: plt.Axes, sizes: List[int]) -> None:
    """Set x-axis ticks to the actual grid sizes used."""
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])


# ── Plot 1a: CPU absolute runtime ────────────────────────────────────────────

def plot_cpu_absolute(datasets: Dict[str, BenchData], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('CPU Runtime Across Machines')
    ax.set_xlabel('Grid size N (N×N)')
    ax.set_ylabel('Wall time (s)')

    all_xs: set = set()
    for name, data in datasets.items():
        st = MACHINE_STYLES[name]
        xs_all = sorted(data['grid_sizes'])
        pairs = [(nx, best_cpu_time(data, nx)[0]) for nx in xs_all]
        xs = [nx for nx, t in pairs if t is not None]
        ys = [t for _, t in pairs if t is not None]
        if xs:
            ax.plot(xs, ys, color=st['color'], marker=st['marker'],
                    ls=st['ls'], linewidth=2, markersize=7, label=MACHINE_LABELS[name])
            all_xs.update(xs)

    if all_xs:
        _grid_axis(ax, sorted(all_xs))
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, 'cpu_runtime_absolute')
    plt.close(fig)


# ── Plot 1b: CPU runtime log scale ───────────────────────────────────────────

def plot_cpu_log(datasets: Dict[str, BenchData], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('CPU Runtime Across Machines (log scale)')
    ax.set_xlabel('Grid size N (N×N)')
    ax.set_ylabel('Wall time (s)')
    ax.set_yscale('log')

    all_xs: set = set()
    for name, data in datasets.items():
        st = MACHINE_STYLES[name]
        xs_all = sorted(data['grid_sizes'])
        pairs = [(nx, best_cpu_time(data, nx)[0]) for nx in xs_all]
        xs = [nx for nx, t in pairs if t is not None]
        ys = [t for _, t in pairs if t is not None]
        if xs:
            ax.plot(xs, ys, color=st['color'], marker=st['marker'],
                    ls=st['ls'], linewidth=2, markersize=7, label=MACHINE_LABELS[name])
            all_xs.update(xs)

    if all_xs:
        _grid_axis(ax, sorted(all_xs))
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, 'cpu_runtime_log')
    plt.close(fig)


# ── Plot 2a: Thread scaling — runtime ────────────────────────────────────────

def _get_scaling_nx(datasets: Dict[str, BenchData], reference_nx: Optional[int]) -> int:
    all_sizes: set = set()
    for data in datasets.values():
        all_sizes.update(data['grid_sizes'])
    if reference_nx is not None:
        return reference_nx
    cpu_sizes: set = set()
    for data in datasets.values():
        for t_data in data['cpu'].values():
            cpu_sizes.update(t_data.keys())
    return max(cpu_sizes) if cpu_sizes else max(all_sizes)


def plot_thread_runtime(datasets: Dict[str, BenchData], out_dir: Path,
                        reference_nx: Optional[int] = None) -> None:
    nx = _get_scaling_nx(datasets, reference_nx)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f'Runtime vs Thread Count (grid {nx}×{nx})')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Wall time (s)')

    all_threads: List[int] = []
    for name, data in datasets.items():
        st = MACHINE_STYLES[name]
        threads_sorted = sorted(data['cpu'].keys())
        xs, ys = [], []
        for t in threads_sorted:
            v = data['cpu'][t].get(nx)
            if v is not None:
                xs.append(t)
                ys.append(v)
        if xs:
            ax.plot(xs, ys, color=st['color'], marker=st['marker'],
                    ls=st['ls'], linewidth=2, markersize=7, label=MACHINE_LABELS[name])
            all_threads.extend(xs)

    if all_threads:
        _thread_axis(ax, sorted(set(all_threads)))
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, 'thread_scaling_runtime')
    plt.close(fig)


# ── Plot 2b: Thread scaling — speedup ────────────────────────────────────────

def plot_thread_speedup(datasets: Dict[str, BenchData], out_dir: Path,
                        reference_nx: Optional[int] = None) -> None:
    nx = _get_scaling_nx(datasets, reference_nx)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f'Speedup vs 1 Thread (grid {nx}×{nx})')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Speedup')

    all_threads: List[int] = []
    for name, data in datasets.items():
        st = MACHINE_STYLES[name]
        threads_sorted = sorted(data['cpu'].keys())
        xs, ys = [], []
        for t in threads_sorted:
            v = data['cpu'][t].get(nx)
            if v is not None:
                xs.append(t)
                ys.append(v)
        if not xs:
            continue
        t1 = data['cpu'].get(1, {}).get(nx)
        if t1 is not None and t1 > 0:
            speedup = [t1 / y for y in ys]
            ax.plot(xs, speedup, color=st['color'], marker=st['marker'],
                    ls=st['ls'], linewidth=2, markersize=7, label=MACHINE_LABELS[name])
            all_threads.extend(xs)

    if all_threads:
        t_max = max(all_threads)
        ax.plot([1, t_max], [1, t_max], 'k--', linewidth=1.2, alpha=0.6, label='Ideal linear')
        _thread_axis(ax, sorted(set(all_threads)))

    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, 'thread_scaling_speedup')
    plt.close(fig)


# ── Plot 3: Speedup vs laptop single-thread baseline ─────────────────────────

def plot_speedup_vs_baseline(datasets: Dict[str, BenchData], out_dir: Path) -> None:
    if 'laptop' not in datasets:
        print('  [skip] speedup_vs_laptop: --laptop CSV not provided')
        return

    laptop = datasets['laptop']
    common_sizes = sorted(laptop['grid_sizes'])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('Speedup vs Laptop 1-Thread Baseline')
    ax.set_xlabel('Grid size N (N×N)')
    ax.set_ylabel('Speedup factor')

    all_xs: set = set()
    for name, data in datasets.items():
        st = MACHINE_STYLES[name]
        xs, ys = [], []
        for nx in common_sizes:
            laptop_1t = laptop['cpu'].get(1, {}).get(nx)
            t_best, _ = best_cpu_time(data, nx)
            if laptop_1t is not None and t_best is not None and t_best > 0:
                xs.append(nx)
                ys.append(laptop_1t / t_best)
        if xs:
            ax.plot(xs, ys, color=st['color'], marker=st['marker'],
                    ls=st['ls'], linewidth=2, markersize=7, label=MACHINE_LABELS[name])
            all_xs.update(xs)

    if all_xs:
        _grid_axis(ax, sorted(all_xs))
    ax.axhline(1.0, color='gray', linewidth=1, linestyle='--', alpha=0.7, label='Laptop 1-thread baseline')
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, 'speedup_vs_laptop')
    plt.close(fig)


# ── Plot 4a: GPU vs CPU runtime ───────────────────────────────────────────────

def plot_gpu_runtime(datasets: Dict[str, BenchData], out_dir: Path) -> None:
    gpu_machines = {name: data for name, data in datasets.items() if data['cuda']}
    if not gpu_machines:
        print('  [skip] gpu_runtime: no CUDA data found')
        return

    gpu_key_map = {'laptop': 'laptop_gpu', 'epyc': 'epyc_gpu', 'tr': 'tr_gpu'}

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('Runtime: GPU vs CPU')
    ax.set_xlabel('Grid size N (N×N)')
    ax.set_ylabel('Wall time (s)')
    ax.set_yscale('log')

    all_xs: set = set()
    for name, data in gpu_machines.items():
        gpu_style = MACHINE_STYLES.get(gpu_key_map.get(name, 'laptop_gpu'))
        cpu_style = MACHINE_STYLES[name]
        gpu_label = MACHINE_LABELS.get(gpu_key_map.get(name, ''), f'{name} GPU')
        cpu_label = MACHINE_LABELS[name]

        xs = sorted(data['cuda'].keys())
        gpu_times = [data['cuda'].get(nx, np.nan) for nx in xs]
        ax.plot(xs, gpu_times, color=gpu_style['color'], marker=gpu_style['marker'],
                ls=gpu_style['ls'], linewidth=2, markersize=7, label=gpu_label)
        all_xs.update(xs)

        cpu_xs = [nx for nx in xs if best_cpu_time(data, nx)[0] is not None]
        cpu_ys = [best_cpu_time(data, nx)[0] for nx in cpu_xs]
        if cpu_xs:
            ax.plot(cpu_xs, cpu_ys, color=cpu_style['color'], marker=cpu_style['marker'],
                    ls='--', linewidth=1.5, markersize=6, label=f'{cpu_label} (CPU)')
            all_xs.update(cpu_xs)

    # Add CPU-only machines (no GPU) as dashed CPU lines
    cpu_only = {name: data for name, data in datasets.items() if not data['cuda']}
    for name, data in cpu_only.items():
        st = MACHINE_STYLES[name]
        cpu_xs = sorted(nx for nx in data['grid_sizes'] if best_cpu_time(data, nx)[0] is not None)
        cpu_ys = [best_cpu_time(data, nx)[0] for nx in cpu_xs]
        if cpu_xs:
            ax.plot(cpu_xs, cpu_ys, color=st['color'], marker=st['marker'],
                    ls='--', linewidth=1.5, markersize=6, label=f'{MACHINE_LABELS[name]} (CPU)')
            all_xs.update(cpu_xs)

    _grid_axis(ax, sorted(all_xs))
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, 'gpu_runtime')
    plt.close(fig)


# ── Plot 4b: GPU speedup over CPU ────────────────────────────────────────────

def plot_gpu_speedup(datasets: Dict[str, BenchData], out_dir: Path) -> None:
    gpu_machines = {name: data for name, data in datasets.items() if data['cuda']}
    if not gpu_machines:
        print('  [skip] gpu_speedup: no CUDA data found')
        return

    gpu_key_map = {'laptop': 'laptop_gpu', 'epyc': 'epyc_gpu', 'tr': 'tr_gpu'}

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('GPU Speedup over CPU (same machine)')
    ax.set_xlabel('Grid size N (N×N)')
    ax.set_ylabel('Speedup factor')

    all_xs: set = set()
    for name, data in gpu_machines.items():
        gpu_style = MACHINE_STYLES.get(gpu_key_map.get(name, 'laptop_gpu'))
        gpu_label = MACHINE_LABELS.get(gpu_key_map.get(name, ''), f'{name} GPU')

        xs = sorted(data['cuda'].keys())
        speedups = []
        valid_xs = []
        for nx in xs:
            gpu_t = data['cuda'].get(nx)
            cpu_t, _ = best_cpu_time(data, nx)
            if gpu_t and cpu_t:
                speedups.append(cpu_t / gpu_t)
                valid_xs.append(nx)

        if valid_xs:
            ax.plot(valid_xs, speedups, color=gpu_style['color'], marker=gpu_style['marker'],
                    ls=gpu_style['ls'], linewidth=2, markersize=7, label=gpu_label)
            all_xs.update(valid_xs)

    if all_xs:
        _grid_axis(ax, sorted(all_xs))
    ax.axhline(1.0, color='gray', linewidth=1, ls='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, 'gpu_speedup')
    plt.close(fig)


# ── Summary text ──────────────────────────────────────────────────────────────

def write_summary(datasets: Dict[str, BenchData], out_dir: Path) -> None:
    lines = ['SALR DFT — Cluster vs Laptop Benchmark Summary', '=' * 60]

    # Best CPU at largest common grid
    all_sizes: set = set()
    for d in datasets.values():
        all_sizes.update(d['grid_sizes'])
    nx = max(all_sizes) if all_sizes else None

    if nx:
        lines.append(f'\nCPU time at grid {nx}×{nx}:')
        for name, data in datasets.items():
            t, th = best_cpu_time(data, nx)
            if t is not None:
                lines.append(f'  {MACHINE_LABELS[name]:<35} {t:.3f}s  ({th} threads)')

        lines.append(f'\nGPU (CUDA) time at grid {nx}×{nx}:')
        for name, data in datasets.items():
            t = data['cuda'].get(nx)
            if t is not None:
                key = {'laptop': 'laptop_gpu', 'epyc': 'epyc_gpu', 'tr': 'tr_gpu'}.get(name, name)
                lbl = MACHINE_LABELS.get(key, f'{name} GPU')
                lines.append(f'  {lbl:<35} {t:.3f}s')

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / 'comparison_summary.txt'
    summary_path.write_text('\n'.join(lines) + '\n')
    print(f'  Saved: {summary_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Compare benchmark CSVs from laptop and cluster nodes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--laptop', type=Path, metavar='CSV',
                        help='Benchmark CSV from laptop (run_benchmarks.py output)')
    parser.add_argument('--epyc', type=Path, metavar='CSV',
                        help='Benchmark CSV from cluster AMD EPYC node')
    parser.add_argument('--tr', type=Path, metavar='CSV',
                        help='Benchmark CSV from cluster Threadripper node (optional)')
    parser.add_argument('--output', '-o', type=Path,
                        default=Path('analysis/results/cluster_vs_laptop'),
                        metavar='DIR',
                        help='Output directory for plots (default: analysis/results/cluster_vs_laptop)')
    parser.add_argument('--scaling-grid', type=int, default=None, metavar='N',
                        help='Grid size used for thread-scaling plot (default: largest available)')
    args = parser.parse_args()

    datasets: Dict[str, BenchData] = {}
    for name, path in [('laptop', args.laptop), ('epyc', args.epyc), ('tr', args.tr)]:
        if path is None:
            continue
        if not path.exists():
            print(f'Error: file not found: {path}', file=sys.stderr)
            return 1
        print(f'Loading {name}: {path}')
        datasets[name] = load_csv(path)

    if not datasets:
        parser.error('Provide at least one CSV via --laptop, --epyc, or --tr.')

    out_dir = args.output
    print(f'\nGenerating plots → {out_dir}')

    plot_cpu_absolute(datasets, out_dir)
    plot_cpu_log(datasets, out_dir)
    plot_thread_runtime(datasets, out_dir, reference_nx=args.scaling_grid)
    plot_thread_speedup(datasets, out_dir, reference_nx=args.scaling_grid)
    plot_speedup_vs_baseline(datasets, out_dir)
    plot_gpu_runtime(datasets, out_dir)
    plot_gpu_speedup(datasets, out_dir)
    write_summary(datasets, out_dir)

    print('\nDone.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
