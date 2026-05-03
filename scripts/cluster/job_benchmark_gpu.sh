#!/bin/bash
# =============================================================================
# job_benchmark_gpu.sh
# SLURM job — GPU benchmark on GeForce RTX 5090 (gnode-003)
#             + CPU comparison on same node (Threadripper PRO 5975WX)
#
# Runs both:
#   1. run_benchmarks.py     — grid-size sweep (CUDA + CPU)
#   2. performance_real_params.py — full physical-parameter convergence run
#
# Submit:  sbatch scripts/cluster/job_benchmark_gpu.sh
# =============================================================================

#SBATCH --job-name=salr_bench_gpu
#SBATCH --partition=gpu-gnode
#SBATCH --nodelist=gnode-003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_bench_gpu_%j.out
#SBATCH --error=logs/slurm_bench_gpu_%j.err

set -eu

# ── Environment ───────────────────────────────────────────────────────────────
# Adjust module names to those available on west3.
# Check with: module avail cuda  |  module avail gcc
module purge
module load gnu14/14.2.0
module load openmpi5/5.0.7
module load hwloc/2.12.0
module load cmake/4.0.0
module load cuda/12.8
module load python/3.11
_jlib=$(find /opt /usr/lib64 /usr/lib /usr/local/lib -name 'libjsoncpp.so*' -print -quit 2>/dev/null || true)
[ -n "$_jlib" ] && export LD_LIBRARY_PATH="$(dirname "$_jlib"):${LD_LIBRARY_PATH:-}"

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR="/scratch/users/nazarp/SALR3YUCUDA"
cd "$WORK_DIR"

mkdir -p logs

echo "============================================================"
echo "SALR GPU Benchmark — RTX 5090 + Threadripper (gnode-003)"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "CPU cores: $SLURM_CPUS_PER_TASK"
echo "Started : $(date)"
echo "============================================================"
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
           --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
echo ">>> Building project (CUDA + CPU) into build_cuda/"
# Use a separate build dir so it doesn't conflict with the CPU-only EPYC build
rm -rf build_cuda
cmake -S . -B build_cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_DB_ENGINE=OFF \
    -Wno-dev \
    2>&1 | tail -20
cmake --build build_cuda --parallel 4 2>&1 | tail -10
echo "Build complete."

pip install --quiet --user numpy matplotlib 2>/dev/null || true

# ── Part 1: grid-size sweep (CUDA + CPU) ─────────────────────────────────────
echo ""
echo ">>> [Part 1] Grid-size benchmark — CUDA + CPU (1 2 4 8 16 threads)"
python3 scripts/benchmarking/run_benchmarks.py \
    --executable        ./build_cuda/salr_dft \
    --cuda-executable   ./build_cuda/salr_dft_cuda \
    --config            ./configs/benchmark.cfg \
    --output-dir        ./analysis/results \
    --grid-sizes        16 32 64 128 160 \
    --threads           1 \
    --iterations        1000 \
    --repeats           3

# ── Part 2: real-parameter convergence run ───────────────────────────────────
echo ""
echo ">>> [Part 2] Full physical-parameter run (real convergence comparison)"
python3 scripts/benchmarking/performance_real_params.py \
    --cpu-exe       ./build_cuda/salr_dft \
    --cuda-exe      ./build_cuda/salr_dft_cuda \
    --config        ./configs/default.cfg \
    --output-root   ./analysis/results \
    --omp-threads   1

echo ""
echo "============================================================"
echo "GPU Benchmark finished: $(date)"
echo "Results in: analysis/results/"
echo "============================================================"
