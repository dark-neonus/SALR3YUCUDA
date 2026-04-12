#!/usr/bin/env python3
"""
performance_real_params.py - Convergence-focused CUDA vs CPU(OpenMP) analysis.

Runs SALR solvers with physical parameters from configs/default.cfg and compares:
- Wall-clock runtime
- Iterations to convergence (or max-iteration stop)
- Final error
- Convergence trajectory (L2 error vs iteration)

Outputs a presentation-ready figure and machine-readable metrics.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


@dataclass
class RunMetrics:
    backend: str
    executable: str
    converged: bool
    return_code: int
    wall_time_s: float
    gpu_solver_time_s: Optional[float]
    iterations: int
    final_error: float
    tolerance: float
    output_dir: Path
    convergence_file: Path
    stdout_log: Path


def run_cmd(cmd: List[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return ""
    return res.stdout.strip()


def get_cpu_name() -> str:
    out = run_cmd(["lscpu"])
    for line in out.splitlines():
        if line.lower().startswith("model name:"):
            return line.split(":", 1)[1].strip()
    return "Unknown CPU"


def get_gpu_name() -> str:
    out = run_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if not out:
        return "No NVIDIA GPU detected"
    return out.splitlines()[0].strip()


def parse_cfg_scalar(cfg_text: str, key: str, default: str = "N/A") -> str:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^#\n\r]+)", cfg_text, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def replace_output_dir(cfg_text: str, output_dir: Path) -> str:
    replacement = f"output_dir = {output_dir.as_posix()}"
    if re.search(r"^\s*output_dir\s*=", cfg_text, flags=re.MULTILINE):
        return re.sub(r"^\s*output_dir\s*=.*$", replacement, cfg_text, count=1, flags=re.MULTILINE)
    return cfg_text + f"\n[output]\n{replacement}\n"


def parse_stdout_metrics(stdout_text: str) -> Tuple[Optional[int], Optional[float], Optional[float], bool]:
    converged = "converged" in stdout_text.lower()

    iter_match = re.search(r"Converged at iteration\s+(\d+)", stdout_text)
    err_match = re.search(r"Converged at iteration\s+\d+\s*\(err\s*=?\s*([0-9.eE+\-]+)", stdout_text)
    gpu_time_match = re.search(r"Converged at iteration\s+\d+.*?,\s*([0-9.]+)\s*s\s*GPU\)", stdout_text)
    if gpu_time_match is None:
        gpu_time_match = re.search(r"did not converge in\s+\d+\s+iterations\s*\(([0-9.]+)\s*s\s*GPU\)", stdout_text)

    iters = int(iter_match.group(1)) if iter_match else None
    err = float(err_match.group(1)) if err_match else None
    gpu_time = float(gpu_time_match.group(1)) if gpu_time_match else None

    return iters, err, gpu_time, converged


def load_convergence(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.array([], dtype=int), np.array([], dtype=float)

    iters: List[int] = []
    errs: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) < 2:
                continue
            try:
                # Solver logs zero-based iteration index in convergence.dat.
                iters.append(int(parts[0]) + 1)
                errs.append(float(parts[1]))
            except ValueError:
                continue

    if not iters:
        return np.array([], dtype=int), np.array([], dtype=float)

    return np.array(iters, dtype=int), np.array(errs, dtype=float)


def run_solver(
    backend: str,
    executable: Path,
    base_cfg: Path,
    run_root: Path,
    omp_threads: int,
    timeout_s: int,
) -> RunMetrics:
    output_dir = run_root / backend
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_text = base_cfg.read_text(encoding="utf-8")
    temp_cfg_text = replace_output_dir(cfg_text, output_dir)

    with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{backend}.cfg", delete=False) as tf:
        tf.write(temp_cfg_text)
        temp_cfg_path = Path(tf.name)

    env = os.environ.copy()
    if backend == "cpu":
        env["OMP_NUM_THREADS"] = str(omp_threads)
        env["OMP_PROC_BIND"] = "true"
        env["OMP_PLACES"] = "cores"

    cmd = [str(executable), str(temp_cfg_path)]

    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        stdout_text = (exc.stdout or "") + "\n" + (exc.stderr or "")
        wall_time = time.perf_counter() - start
        ret_code = 124
    else:
        stdout_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        wall_time = time.perf_counter() - start
        ret_code = proc.returncode
    finally:
        try:
            temp_cfg_path.unlink(missing_ok=True)
        except OSError:
            pass

    stdout_log = output_dir / "run_stdout.log"
    stdout_log.write_text(stdout_text, encoding="utf-8")

    convergence_file = output_dir / "convergence.dat"
    conv_iters, conv_errs = load_convergence(convergence_file)

    out_iters, out_err, gpu_solver_time, out_converged = parse_stdout_metrics(stdout_text)

    iterations = out_iters if out_iters is not None else (int(conv_iters[-1]) if conv_iters.size else 0)
    final_error = out_err if out_err is not None else (float(conv_errs[-1]) if conv_errs.size else float("nan"))

    cfg_for_tol = base_cfg.read_text(encoding="utf-8")
    tol_str = parse_cfg_scalar(cfg_for_tol, "tolerance", "1e-8")
    tolerance = float(tol_str)

    converged = out_converged
    if not converged and np.isfinite(final_error):
        converged = final_error < tolerance

    return RunMetrics(
        backend=backend,
        executable=str(executable),
        converged=converged,
        return_code=ret_code,
        wall_time_s=wall_time,
        gpu_solver_time_s=gpu_solver_time,
        iterations=iterations,
        final_error=final_error,
        tolerance=tolerance,
        output_dir=output_dir,
        convergence_file=convergence_file,
        stdout_log=stdout_log,
    )


def plot_results(
    cpu: RunMetrics,
    cuda: RunMetrics,
    cfg_text: str,
    out_dir: Path,
    omp_threads: int,
    cpu_name: str,
    gpu_name: str,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    colors = {"cpu": "#2E8B57", "cuda": "#005BBB"}
    labels = ["CPU (OpenMP)", "CUDA"]
    x = np.arange(2)

    # Panel 1: Runtime
    ax1 = fig.add_subplot(gs[0, 0])
    time_vals = [cpu.wall_time_s, cuda.wall_time_s]
    bars = ax1.bar(x, time_vals, color=[colors["cpu"], colors["cuda"]], width=0.62)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Wall Time (s)")
    ax1.set_title("Runtime Comparison")
    for bar, value in zip(bars, time_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}s", ha="center", va="bottom")
    if cuda.wall_time_s > 0:
        speedup = cpu.wall_time_s / cuda.wall_time_s
        ax1.text(
            0.5,
            0.92,
            f"CUDA speedup vs CPU: {speedup:.2f}x",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "facecolor": "#EEF6FF", "edgecolor": "#A9CCE3"},
        )

    # Panel 2: Iterations and convergence status
    ax2 = fig.add_subplot(gs[0, 1])
    iter_vals = [cpu.iterations, cuda.iterations]
    bars2 = ax2.bar(x, iter_vals, color=[colors["cpu"], colors["cuda"]], width=0.62)
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Iterations")
    ax2.set_title("Iterations to Stop")
    for i, (bar, val, ok) in enumerate(zip(bars2, iter_vals, [cpu.converged, cuda.converged])):
        status = "converged" if ok else "not converged"
        ax2.text(bar.get_x() + bar.get_width() / 2, val, f"{val}\n({status})", ha="center", va="bottom")

    # Panel 3: Final error
    ax3 = fig.add_subplot(gs[1, 0])
    err_vals = [cpu.final_error, cuda.final_error]
    safe_errs = [v if np.isfinite(v) and v > 0 else np.nan for v in err_vals]
    bars3 = ax3.bar(x, safe_errs, color=[colors["cpu"], colors["cuda"]], width=0.62)
    ax3.set_xticks(x, labels)
    ax3.set_yscale("log")
    ax3.set_ylabel("Final L2 Error")
    ax3.set_title("Final Error vs Tolerance")
    ax3.axhline(cpu.tolerance, linestyle="--", color="#AA3A3A", linewidth=1.6, label=f"Tolerance = {cpu.tolerance:.1e}")
    for bar, value in zip(bars3, err_vals):
        if np.isfinite(value) and value > 0:
            ax3.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2e}", ha="center", va="bottom")
    ax3.legend(loc="upper right")

    # Panel 4: Convergence curves
    ax4 = fig.add_subplot(gs[1, 1])
    cpu_it, cpu_er = load_convergence(cpu.convergence_file)
    cuda_it, cuda_er = load_convergence(cuda.convergence_file)
    if cpu_it.size and cpu_er.size:
        ax4.semilogy(cpu_it, cpu_er, color=colors["cpu"], linewidth=2.0, label=f"CPU ({omp_threads} threads)")
    if cuda_it.size and cuda_er.size:
        ax4.semilogy(cuda_it, cuda_er, color=colors["cuda"], linewidth=2.0, label="CUDA")
    ax4.axhline(cpu.tolerance, linestyle="--", color="#AA3A3A", linewidth=1.2, alpha=0.8)
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

    fig.suptitle("SALR DFT Performance Analysis (Default Physical Parameters)", fontsize=16, fontweight="bold")
    fig.text(
        0.02,
        0.01,
        cfg_spec + "\n" + hw_spec,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "#F7F7F7", "edgecolor": "#CCCCCC"},
    )

    fig.savefig(out_dir / "performance_real_params.png", bbox_inches="tight")
    fig.savefig(out_dir / "performance_real_params.svg", bbox_inches="tight")
    plt.close(fig)


def write_metrics_csv(cpu: RunMetrics, cuda: RunMetrics, out_dir: Path, omp_threads: int) -> None:
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "backend",
                "wall_time_s",
                "gpu_solver_time_s",
                "iterations",
                "final_error",
                "tolerance",
                "converged",
                "return_code",
                "omp_threads",
                "output_dir",
                "stdout_log",
            ]
        )
        for m in [cpu, cuda]:
            writer.writerow(
                [
                    m.backend,
                    f"{m.wall_time_s:.6f}",
                    "" if m.gpu_solver_time_s is None else f"{m.gpu_solver_time_s:.6f}",
                    m.iterations,
                    "" if not np.isfinite(m.final_error) else f"{m.final_error:.12e}",
                    f"{m.tolerance:.12e}",
                    int(m.converged),
                    m.return_code,
                    omp_threads if m.backend == "cpu" else "",
                    m.output_dir.as_posix(),
                    m.stdout_log.as_posix(),
                ]
            )


def write_summary(cpu: RunMetrics, cuda: RunMetrics, out_dir: Path, cfg_path: Path, omp_threads: int) -> None:
    speedup = cpu.wall_time_s / cuda.wall_time_s if cuda.wall_time_s > 0 else float("nan")
    now = datetime.now().isoformat(timespec="seconds")

    lines = [
        "SALR DFT Performance Analysis (Default Physical Parameters)",
        "=" * 64,
        f"Generated: {now}",
        f"Config: {cfg_path}",
        f"CPU OpenMP threads: {omp_threads}",
        "",
        "Results:",
        f"  CPU  - time={cpu.wall_time_s:.3f}s, iterations={cpu.iterations}, final_error={cpu.final_error:.3e}, converged={cpu.converged}",
        f"  CUDA - time={cuda.wall_time_s:.3f}s, iterations={cuda.iterations}, final_error={cuda.final_error:.3e}, converged={cuda.converged}",
        f"  CUDA speedup vs CPU (wall-time): {speedup:.2f}x" if np.isfinite(speedup) else "  CUDA speedup vs CPU: N/A",
        "",
        "Artifacts:",
        "  - performance_real_params.png",
        "  - performance_real_params.svg",
        "  - metrics.csv",
        "  - cpu/run_stdout.log",
        "  - cuda/run_stdout.log",
    ]

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_executable(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} executable not found: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"{name} exists but is not executable: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run convergence-focused CUDA vs CPU analysis with default physical parameters.")
    parser.add_argument("--config", default="configs/default.cfg", help="Path to base config file (default: configs/default.cfg)")
    parser.add_argument("--cpu-exe", default="build/salr_dft", help="Path to CPU executable")
    parser.add_argument("--cuda-exe", default="build/salr_dft_cuda", help="Path to CUDA executable")
    parser.add_argument("--output-root", default="analysis/results", help="Root directory for generated analysis")
    parser.add_argument("--omp-threads", type=int, default=os.cpu_count() or 1, help="OpenMP thread count for CPU run")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout per solver run in seconds")

    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cpu_exe = Path(args.cpu_exe).resolve()
    cuda_exe = Path(args.cuda_exe).resolve()

    validate_executable(cpu_exe, "CPU")
    validate_executable(cuda_exe, "CUDA")

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root).resolve() / f"real_params_cuda_vs_cpu_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    cfg_text = cfg_path.read_text(encoding="utf-8")
    cpu_name = get_cpu_name()
    gpu_name = get_gpu_name()

    print("=" * 72)
    print("SALR DFT Real-Parameter Performance Analysis")
    print("=" * 72)
    print(f"Config           : {cfg_path}")
    print(f"CPU executable   : {cpu_exe}")
    print(f"CUDA executable  : {cuda_exe}")
    print(f"CPU threads      : {args.omp_threads}")
    print(f"CPU model        : {cpu_name}")
    print(f"GPU model        : {gpu_name}")
    print(f"Output directory : {run_root}")
    print("=" * 72)

    print("\n[1/2] Running CPU (OpenMP)...")
    cpu = run_solver("cpu", cpu_exe, cfg_path, run_root, args.omp_threads, args.timeout)
    print(
        f"  CPU done: time={cpu.wall_time_s:.2f}s, iters={cpu.iterations}, "
        f"final_error={cpu.final_error:.3e}, converged={cpu.converged}"
    )

    print("\n[2/2] Running CUDA...")
    cuda = run_solver("cuda", cuda_exe, cfg_path, run_root, args.omp_threads, args.timeout)
    print(
        f"  CUDA done: time={cuda.wall_time_s:.2f}s, iters={cuda.iterations}, "
        f"final_error={cuda.final_error:.3e}, converged={cuda.converged}"
    )

    print("\n[Post] Creating plots and reports...")
    plot_results(cpu, cuda, cfg_text, run_root, args.omp_threads, cpu_name, gpu_name)
    write_metrics_csv(cpu, cuda, run_root, args.omp_threads)
    write_summary(cpu, cuda, run_root, cfg_path, args.omp_threads)

    print("\nArtifacts:")
    print(f"  - {run_root / 'performance_real_params.png'}")
    print(f"  - {run_root / 'performance_real_params.svg'}")
    print(f"  - {run_root / 'metrics.csv'}")
    print(f"  - {run_root / 'summary.txt'}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
