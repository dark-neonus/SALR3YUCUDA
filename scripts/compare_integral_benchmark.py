#!/usr/bin/env python3
"""
compare_integral_benchmark.py - Compare CPU vs CUDA integral performance

Reads benchmark data from output/benchmark_cpu.dat and output/benchmark_cuda.dat,
generates comparison plots showing speedup and performance scaling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def load_benchmark(filepath):
    """Load benchmark data from file. Returns dict by method."""
    data = {}
    if not os.path.exists(filepath):
        return data
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                n = int(parts[0])
                method = parts[1]
                time_ms = float(parts[2])
                error = float(parts[3])
                
                if method not in data:
                    data[method] = {'n': [], 'time': [], 'error': []}
                data[method]['n'].append(n)
                data[method]['time'].append(time_ms)
                data[method]['error'].append(error)
    
    # Convert to numpy arrays
    for m in data:
        data[m]['n'] = np.array(data[m]['n'])
        data[m]['time'] = np.array(data[m]['time'])
        data[m]['error'] = np.array(data[m]['error'])
    
    return data

def main():
    # Load data
    cpu_data = load_benchmark('output/benchmark_cpu.dat')
    cuda_data = load_benchmark('output/benchmark_cuda.dat')
    
    if not cpu_data and not cuda_data:
        print("No benchmark data found. Run benchmarks first:")
        print("  ./build/benchmark_integral_cpu")
        print("  ./build/benchmark_integral_cuda")
        return
    
    # Create output directory
    os.makedirs('output/plots', exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_dpi = 150
    
    # Plot 1: 1D Integration Time Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trapezoidal comparison
    ax = axes[0]
    if 'trapezoidal' in cpu_data:
        ax.plot(cpu_data['trapezoidal']['n'], cpu_data['trapezoidal']['time'], 
                'o-', label='CPU OpenMP', linewidth=2, markersize=6)
    if 'trapezoidal' in cuda_data:
        ax.plot(cuda_data['trapezoidal']['n'], cuda_data['trapezoidal']['time'],
                's-', label='CUDA', linewidth=2, markersize=6)
    ax.set_xlabel('Array Size (N)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Trapezoidal Rule')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Simpson comparison
    ax = axes[1]
    if 'simpson' in cpu_data:
        ax.plot(cpu_data['simpson']['n'], cpu_data['simpson']['time'],
                'o-', label='CPU OpenMP', linewidth=2, markersize=6)
    if 'simpson' in cuda_data:
        ax.plot(cuda_data['simpson']['n'], cuda_data['simpson']['time'],
                's-', label='CUDA', linewidth=2, markersize=6)
    ax.set_xlabel('Array Size (N)')
    ax.set_ylabel('Time (ms)')
    ax.set_title("Simpson's Rule")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('1D Integration Performance: CPU vs CUDA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/plots/integral_1d_comparison.png', dpi=fig_dpi, bbox_inches='tight')
    plt.close()
    print("Saved: output/plots/integral_1d_comparison.png")
    
    # Plot 2: Speedup plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['trapezoidal', 'simpson']
    colors = ['#1f77b4', '#ff7f0e']
    
    for method, color in zip(methods, colors):
        if method in cpu_data and method in cuda_data:
            # Match sizes
            cpu_n = cpu_data[method]['n']
            cuda_n = cuda_data[method]['n']
            common_n = np.intersect1d(cpu_n, cuda_n)
            
            speedups = []
            for n in common_n:
                cpu_idx = np.where(cpu_n == n)[0][0]
                cuda_idx = np.where(cuda_n == n)[0][0]
                speedup = cpu_data[method]['time'][cpu_idx] / cuda_data[method]['time'][cuda_idx]
                speedups.append(speedup)
            
            ax.plot(common_n, speedups, 'o-', label=method.capitalize(), 
                    color=color, linewidth=2, markersize=6)
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline (1x)')
    ax.set_xlabel('Array Size (N)')
    ax.set_ylabel('Speedup (CPU time / CUDA time)')
    ax.set_title('CUDA Speedup over CPU OpenMP')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/plots/integral_speedup.png', dpi=fig_dpi, bbox_inches='tight')
    plt.close()
    print("Saved: output/plots/integral_speedup.png")
    
    # Plot 3: 2D Integration Comparison  
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time comparison
    ax = axes[0]
    if 'trap2d' in cpu_data:
        ax.plot(cpu_data['trap2d']['n'], cpu_data['trap2d']['time'],
                'o-', label='CPU OpenMP', linewidth=2, markersize=6)
    if 'trap2d' in cuda_data:
        ax.plot(cuda_data['trap2d']['n'], cuda_data['trap2d']['time'],
                's-', label='CUDA', linewidth=2, markersize=6)
    ax.set_xlabel('Grid Size (NxN)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('2D Trapezoidal Integration Time')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Speedup
    ax = axes[1]
    if 'trap2d' in cpu_data and 'trap2d' in cuda_data:
        cpu_n = cpu_data['trap2d']['n']
        cuda_n = cuda_data['trap2d']['n']
        common_n = np.intersect1d(cpu_n, cuda_n)
        
        speedups = []
        for n in common_n:
            cpu_idx = np.where(cpu_n == n)[0][0]
            cuda_idx = np.where(cuda_n == n)[0][0]
            speedup = cpu_data['trap2d']['time'][cpu_idx] / cuda_data['trap2d']['time'][cuda_idx]
            speedups.append(speedup)
        
        ax.bar(range(len(common_n)), speedups, tick_label=[str(int(n)) for n in common_n],
               color='#2ca02c', alpha=0.7)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Grid Size (NxN)')
        ax.set_ylabel('Speedup')
        ax.set_title('2D Integration CUDA Speedup')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('2D Integration Performance: CPU vs CUDA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/plots/integral_2d_comparison.png', dpi=fig_dpi, bbox_inches='tight')
    plt.close()
    print("Saved: output/plots/integral_2d_comparison.png")
    
    # Plot 4: Error comparison (accuracy)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for method, color in zip(methods, colors):
        if method in cpu_data:
            ax.semilogy(cpu_data[method]['n'], cpu_data[method]['error'],
                       'o-', label=f'CPU {method.capitalize()}', color=color, 
                       linewidth=2, markersize=6)
        if method in cuda_data:
            ax.semilogy(cuda_data[method]['n'], cuda_data[method]['error'],
                       's--', label=f'CUDA {method.capitalize()}', color=color,
                       linewidth=2, markersize=6, alpha=0.7)
    
    ax.set_xlabel('Array Size (N)')
    ax.set_ylabel('Absolute Error |result - exact|')
    ax.set_title('Integration Accuracy: CPU vs CUDA')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/plots/integral_accuracy.png', dpi=fig_dpi, bbox_inches='tight')
    plt.close()
    print("Saved: output/plots/integral_accuracy.png")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    
    for method in methods:
        if method in cpu_data and method in cuda_data:
            cpu_n = cpu_data[method]['n']
            cuda_n = cuda_data[method]['n']
            common_n = np.intersect1d(cpu_n, cuda_n)
            
            if len(common_n) > 0:
                speedups = []
                for n in common_n:
                    cpu_idx = np.where(cpu_n == n)[0][0]
                    cuda_idx = np.where(cuda_n == n)[0][0]
                    speedup = cpu_data[method]['time'][cpu_idx] / cuda_data[method]['time'][cuda_idx]
                    speedups.append(speedup)
                
                print(f"{method.capitalize()}:")
                print(f"  Mean speedup: {np.mean(speedups):.2f}x")
                print(f"  Max speedup:  {np.max(speedups):.2f}x (at N={common_n[np.argmax(speedups)]})")
    
    if 'trap2d' in cpu_data and 'trap2d' in cuda_data:
        cpu_n = cpu_data['trap2d']['n']
        cuda_n = cuda_data['trap2d']['n']
        common_n = np.intersect1d(cpu_n, cuda_n)
        
        if len(common_n) > 0:
            speedups = []
            for n in common_n:
                cpu_idx = np.where(cpu_n == n)[0][0]
                cuda_idx = np.where(cuda_n == n)[0][0]
                speedup = cpu_data['trap2d']['time'][cpu_idx] / cuda_data['trap2d']['time'][cuda_idx]
                speedups.append(speedup)
            
            print("2D Trapezoidal:")
            print(f"  Mean speedup: {np.mean(speedups):.2f}x")
            print(f"  Max speedup:  {np.max(speedups):.2f}x (at N={common_n[np.argmax(speedups)]})")

if __name__ == '__main__':
    main()
