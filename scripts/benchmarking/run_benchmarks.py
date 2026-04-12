#!/usr/bin/env python3
"""
run_benchmarks.py - Run SALR DFT solver benchmarks

Runs the solver with various configurations to collect performance data:
- CUDA GPU execution
- CPU with different OpenMP thread counts

Outputs benchmark data to CSV for analysis by benchmark_cuda_cpu.py
Results are saved in analysis/results/<cpu_model>_<gpu_model>/ folder
"""

import os
import sys
import subprocess
import argparse
import time
import re
from pathlib import Path
from datetime import datetime


def get_cpu_info():
    """Get CPU information for logging."""
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        info = {}
        for line in lines:
            if ':' in line:
                key, val = line.split(':', 1)
                info[key.strip()] = val.strip()
        
        model = info.get('Model name', 'Unknown_CPU')
        # Clean model name for folder name
        model = re.sub(r'[^\w\-]', '_', model)
        model = re.sub(r'_+', '_', model).strip('_')
        
        return {
            'model': model,
            'full_name': info.get('Model name', 'Unknown'),
            'cores': info.get('CPU(s)', 'Unknown'),
            'threads_per_core': info.get('Thread(s) per core', '1'),
        }
    except:
        return {'model': 'Unknown_CPU', 'full_name': 'Unknown', 'cores': 'Unknown'}


def get_gpu_info():
    """Get GPU information for logging."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                '--format=csv,noheader'], 
                               capture_output=True, text=True)
        gpu_info = result.stdout.strip()
        
        # Extract GPU model name for folder
        gpu_name = gpu_info.split(',')[0].strip() if gpu_info else 'No_GPU'
        gpu_name = re.sub(r'[^\w\-]', '_', gpu_name)
        gpu_name = re.sub(r'_+', '_', gpu_name).strip('_')
        
        return {
            'model': gpu_name,
            'full_name': gpu_info
        }
    except:
        return {'model': 'No_GPU', 'full_name': 'No NVIDIA GPU detected'}


def run_solver(executable, config_file, nx, ny, max_iter=1000, cuda=False, threads=None):
    """
    Run the solver and extract timing information.
    
    Returns:
        tuple: (total_time_seconds, iterations_completed, converged)
    """
    env = os.environ.copy()
    if threads and not cuda:
        env['OMP_NUM_THREADS'] = str(threads)
    
    cmd = [executable, config_file]
    
    temp_config = f'/tmp/benchmark_config_{nx}_{ny}.cfg'
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    config_content = re.sub(r'nx\s*=\s*\d+', f'nx = {nx}', config_content)
    config_content = re.sub(r'ny\s*=\s*\d+', f'ny = {ny}', config_content)
    config_content = re.sub(r'max_iterations\s*=\s*\d+', f'max_iterations = {max_iter}', config_content)
    config_content = re.sub(r'save_every\s*=\s*\d+', f'save_every = {max_iter + 1}', config_content)
    
    with open(temp_config, 'w') as f:
        f.write(config_content)
    
    cmd = [executable, temp_config]
    
    start_time = time.perf_counter()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                               timeout=1800, env=env)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        iter_match = re.search(r'Iteration\s+(\d+)', result.stdout)
        iterations = int(iter_match.group(1)) if iter_match else max_iter
        
        converged = 'converged' in result.stdout.lower()
        
        return elapsed, iterations, converged
        
    except subprocess.TimeoutExpired:
        return None, 0, False
    except Exception as e:
        print(f"    Error: {e}")
        return None, 0, False
    finally:
        if os.path.exists(temp_config):
            os.remove(temp_config)


def main():
    parser = argparse.ArgumentParser(description='Run SALR DFT solver benchmarks')
    parser.add_argument('--executable', '-e', default='./build/salr_dft',
                       help='Path to solver executable')
    parser.add_argument('--cuda-executable', default='./build/salr_dft_cuda',
                       help='Path to CUDA solver executable')
    parser.add_argument('--config', '-c', default='./configs/default.cfg',
                       help='Base config file')
    parser.add_argument('--output-dir', default='./analysis/results',
                       help='Output directory for results')
    parser.add_argument('--grid-sizes', '-g', nargs='+', type=int,
                       default=[16, 32, 64, 128, 150],
                       help='Grid sizes to benchmark (default: 32 64 128 192)')
    parser.add_argument('--threads', '-t', nargs='+', type=int,
                       default=[1, 2, 4, 8, 12, 16, 20],
                       help='Thread counts for CPU benchmarks (default: 1 2 4 8 12 16 20)')
    parser.add_argument('--iterations', '-i', type=int, default=1000,
                       help='Max iterations per run')
    parser.add_argument('--skip-cuda', action='store_true',
                       help='Skip CUDA benchmarks')
    parser.add_argument('--skip-cpu', action='store_true',
                       help='Skip CPU benchmarks')
    parser.add_argument('--repeats', '-r', type=int, default=3,
                       help='Number of repeats for averaging')
    
    args = parser.parse_args()
    
    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    
    # Create output folder with CPU and GPU model names
    folder_name = f"{cpu_info['model']}_{gpu_info['model']}"
    output_dir = Path(args.output_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'benchmark_{timestamp}.csv'
    
    print("=" * 60)
    print("SALR DFT Solver Benchmark Suite")
    print("=" * 60)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"CPU: {cpu_info['full_name']}")
    print(f"GPU: {gpu_info['full_name']}")
    print(f"Grid sizes: {args.grid_sizes}")
    print(f"Thread counts: {args.threads}")
    print(f"Iterations per run: {args.iterations}")
    print(f"Repeats: {args.repeats}")
    print(f"Output: {output_file}")
    print("=" * 60)
    
    results = {n: {} for n in args.grid_sizes}
    
    if not args.skip_cuda and os.path.exists(args.cuda_executable):
        print("\n[CUDA Benchmarks]")
        for n in args.grid_sizes:
            print(f"  Grid {n}x{n}...", end=' ', flush=True)
            times = []
            for r in range(args.repeats):
                t, iters, conv = run_solver(args.cuda_executable, args.config,
                                           n, n, args.iterations, cuda=True)
                if t is not None:
                    times.append(t)
            
            if times:
                avg_time = sum(times) / len(times)
                results[n]['cuda'] = avg_time
                print(f"{avg_time:.3f}s (avg of {len(times)})")
            else:
                print("FAILED")
    else:
        print("\n[CUDA] Skipped (no executable or --skip-cuda)")
    
    if not args.skip_cpu and os.path.exists(args.executable):
        print("\n[CPU Benchmarks]")
        for threads in args.threads:
            print(f"\n  {threads} thread(s):")
            for n in args.grid_sizes:
                print(f"    Grid {n}x{n}...", end=' ', flush=True)
                times = []
                for r in range(args.repeats):
                    t, iters, conv = run_solver(args.executable, args.config,
                                               n, n, args.iterations, 
                                               cuda=False, threads=threads)
                    if t is not None:
                        times.append(t)
                
                if times:
                    avg_time = sum(times) / len(times)
                    results[n][f'cpu_{threads}'] = avg_time
                    print(f"{avg_time:.3f}s (avg of {len(times)})")
                else:
                    print("FAILED")
    else:
        print("\n[CPU] Skipped (no executable or --skip-cpu)")
    
    print("\n[Saving Results]")
    
    columns = ['grid_size']
    if any('cuda' in results[n] for n in args.grid_sizes):
        columns.append('cuda')
    for t in args.threads:
        if any(f'cpu_{t}' in results[n] for n in args.grid_sizes):
            columns.append(f'cpu_{t}')
    
    with open(output_file, 'w') as f:
        f.write(','.join(columns) + '\n')
        for n in args.grid_sizes:
            row = [str(n)]
            for col in columns[1:]:
                val = results[n].get(col, '')
                row.append(f'{val:.6f}' if val else '')
            f.write(','.join(row) + '\n')
    
    print(f"  Saved to: {output_file}")
    
    # Also save metadata
    metadata_file = output_dir / f'benchmark_{timestamp}_info.txt'
    with open(metadata_file, 'w') as f:
        f.write(f"SALR DFT Benchmark Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"CPU: {cpu_info['full_name']}\n")
        f.write(f"  Cores: {cpu_info['cores']}\n")
        f.write(f"GPU: {gpu_info['full_name']}\n")
        f.write(f"Grid sizes: {args.grid_sizes}\n")
        f.write(f"Thread counts: {args.threads}\n")
        f.write(f"Max iterations: {args.iterations}\n")
        f.write(f"Repeats per test: {args.repeats}\n")
    
    print(f"  Metadata saved to: {metadata_file}")
    
    print("\n[Summary]")
    print("-" * 60)
    print(f"{'Grid':<10} | {'CUDA':<12} | {'CPU 1T':<12} | {'Speedup':<10}")
    print("-" * 60)
    
    for n in args.grid_sizes:
        cuda_t = results[n].get('cuda', 0)
        cpu1_t = results[n].get('cpu_1', 0)
        speedup = cpu1_t / cuda_t if cuda_t and cpu1_t else 0
        
        cuda_str = f"{cuda_t:.3f}s" if cuda_t else "N/A"
        cpu1_str = f"{cpu1_t:.3f}s" if cpu1_t else "N/A"
        speed_str = f"{speedup:.1f}x" if speedup else "N/A"
        
        print(f"{n}x{n:<7} | {cuda_str:<12} | {cpu1_str:<12} | {speed_str:<10}")
    
    print("-" * 60)
    print("\nDone! Run 'python scripts/benchmarking/benchmark_cuda_cpu.py' to generate plots.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
