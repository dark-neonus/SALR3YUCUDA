#!/usr/bin/env python3
"""
plot_real_params_from_results.py - Build convergence and runtime figure
from an existing real_params_cuda_vs_cpu_* results directory.

Usage:
  python3 scripts/plotting/plot_real_params_from_results.py \
      --results analysis/results/real_params_cuda_vs_cpu_20260502_212517 \
      --output docs/report/SALR3YUCUDA/src/performance_real_params_laptop.png \
      --title-suffix "Laptop"
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_cfg_scalar(cfg_text: str, key: str, default: str = "N/A") -> str:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^#\n\r]+)", cfg_text, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def load_metrics(metrics_path: Path) -> Dict[str, Dict[str, str]]:
    data: Dict[str, Dict[str, str]] = {}
    with metrics_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            backend = row.get("backend", "unknown")
            data[backend] = row
    return data


def load_convergence(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.array([], dtype=int), np.array([], dtype=float)

    iters = []
    errs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) < 2:
                continue
            try:
                iters.append(int(parts[0]) + 1)
                errs.append(float(parts[1]))
            except ValueError:
                continue

    if not iters:
        return np.array([], dtype=int), np.array([], dtype=float)

    return np.array(iters, dtype=int), np.array(errs, dtype=float)


def parse_summary(summary_text: str) -> Tuple[str, str, str]:
    cpu_name = "Unknown CPU"
    gpu_name = "Unknown GPU"
    omp_threads = "N/A"

    cpu_match = re.search(r"^CPU:\s*(.+)$", summary_text, flags=re.MULTILINE)
    if cpu_match:
        cpu_name = cpu_match.group(1).strip()

    gpu_match = re.search(r"^GPU:\s*(.+)$", summary_text, flags=re.MULTILINE)
    if gpu_match:
        gpu_name = gpu_match.group(1).strip()

    omp_match = re.search(r"^CPU OpenMP threads:\s*(.+)$", summary_text, flags=re.MULTILINE)
    if omp_match:
        omp_threads = omp_match.group(1).strip()

    return cpu_name, gpu_name, omp_threads


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot real-params performance from saved results")
    parser.add_argument("--results", required=True, type=Path, help="results directory")
    parser.add_argument("--output", required=True, type=Path, help="output PNG path")
    parser.add_argument("--title-suffix", default="", help="suffix to append to figure title")
    args = parser.parse_args()

    results_dir = args.results
    metrics_path = results_dir / "metrics.csv"
    summary_path = results_dir / "summary.txt"

    if not metrics_path.exists():
        raise SystemExit(f"metrics.csv not found in {results_dir}")

    metrics = load_metrics(metrics_path)
    cpu = metrics.get("cpu", {})
    cuda = metrics.get("cuda", {})

    cpu_dir = results_dir / "cpu"
    cuda_dir = results_dir / "cuda"

    cpu_conv = cpu_dir / "convergence.dat"
    cuda_conv = cuda_dir / "convergence.dat"

    cpu_iters, cpu_errs = load_convergence(cpu_conv)
    cuda_iters, cuda_errs = load_convergence(cuda_conv)

    cfg_path = cpu_dir / "input_used.cfg"
    cfg_text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""

    summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    cpu_name, gpu_name, omp_threads = parse_summary(summary_text)

    cpu_time = float(cpu.get("wall_time_s", "nan"))
    cuda_time = float(cuda.get("wall_time_s", "nan"))
    cpu_iter = int(float(cpu.get("iterations", "0")))
    cuda_iter = int(float(cuda.get("iterations", "0")))
    cpu_err = float(cpu.get("final_error", "nan"))
    cuda_err = float(cuda.get("final_error", "nan"))
    tol = float(cpu.get("tolerance", cuda.get("tolerance", "nan")))

    speedup = cpu_time / cuda_time if cuda_time > 0 else float("nan")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
            "savefig.dpi": 300,
        }
    )

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    ax1, ax2, ax3, ax4 = axs.flatten()

    ax1.bar(["CPU (OpenMP)", "CUDA"], [cpu_time, cuda_time], color=["#2E8B57", "#005BBB"])
    ax1.set_ylabel("Wall time (s)")
    ax1.set_title("Runtime Comparison")
    if speedup == speedup:
        ax1.text(0.5, max(cpu_time, cuda_time) * 0.92, f"CUDA speedup vs CPU: {speedup:.2f}x",
                 ha="center", va="center", bbox={"boxstyle": "round", "fc": "#EAF3FF", "ec": "#AAC4E8"})

    ax2.bar(["CPU (OpenMP)", "CUDA"], [cpu_iter, cuda_iter], color=["#2E8B57", "#005BBB"])
    ax2.set_ylabel("Iterations")
    ax2.set_title("Iterations to Stop")

    ax3.bar(["CPU", "CUDA"], [cpu_err, cuda_err], color=["#2E8B57", "#005BBB"])
    ax3.axhline(tol, linestyle="--", color="#AA3A3A", linewidth=1.2, alpha=0.8, label=f"Tolerance = {tol:.1e}")
    ax3.set_yscale("log")
    ax3.set_ylabel("Final L2 Error")
    ax3.set_title("Final Error vs Tolerance")
    ax3.legend(loc="upper right")

    if cpu_iters.size:
        ax4.semilogy(cpu_iters, cpu_errs, color="#2E8B57", linewidth=2.0, label=f"CPU ({omp_threads} threads)")
    if cuda_iters.size:
        ax4.semilogy(cuda_iters, cuda_errs, color="#005BBB", linewidth=2.0, label="CUDA")
    if tol == tol:
        ax4.axhline(tol, linestyle="--", color="#AA3A3A", linewidth=1.2, alpha=0.8)
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("L2 Error")
    ax4.set_title("Convergence Trajectory")
    ax4.legend(loc="upper right")

    cfg_spec = (
        f"Grid: {parse_cfg_scalar(cfg_text, 'nx')} x {parse_cfg_scalar(cfg_text, 'ny')}"
        f"  (dx={parse_cfg_scalar(cfg_text, 'dx')}, dy={parse_cfg_scalar(cfg_text, 'dy')})\n"
        f"T={parse_cfg_scalar(cfg_text, 'temperature')}, "
        f"rho1={parse_cfg_scalar(cfg_text, 'rho1')}, rho2={parse_cfg_scalar(cfg_text, 'rho2')}, "
        f"rc={parse_cfg_scalar(cfg_text, 'cutoff_radius')}\n"
        f"boundary={parse_cfg_scalar(cfg_text, 'boundary_mode')}, init={parse_cfg_scalar(cfg_text, 'init_mode')}, "
        f"max_iter={parse_cfg_scalar(cfg_text, 'max_iterations')}, tol={parse_cfg_scalar(cfg_text, 'tolerance')}"
    )
    hw_spec = f"CPU: {cpu_name}\nGPU: {gpu_name}\nOpenMP threads (CPU run): {omp_threads}"

    title = "SALR DFT Performance Analysis (Default Physical Parameters)"
    if args.title_suffix:
        title += f" - {args.title_suffix}"

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.text(
        0.02,
        0.01,
        cfg_spec + "\n" + hw_spec,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "#F7F7F7", "edgecolor": "#CCCCCC"},
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
