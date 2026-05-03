#!/bin/bash
# =============================================================================
# job_benchmark_cpu_epyc.sh
# SLURM job — CPU OpenMP benchmark on AMD EPYC 9754 (node-003)
#
# Resources: up to 64 cores, node-003
# Benchmark: grid-size sweep, thread-count sweep (1,2,4,8,16,32,64)
#
# Submit:  sbatch scripts/cluster/job_benchmark_cpu_epyc.sh
# =============================================================================

#SBATCH --job-name=salr_bench_cpu_epyc
#SBATCH --partition=mpi-node
#SBATCH --nodelist=node-003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_bench_cpu_epyc_%j.out
#SBATCH --error=logs/slurm_bench_cpu_epyc_%j.err

set -eu

# ── Environment ───────────────────────────────────────────────────────────────
# Adjust module names to match what is available on west3.
# Run 'module avail' on the cluster to find the right names.
module purge
module load gnu14/14.2.0
module load cmake/4.0.0
module load python/3.11
_jlib=$(find /opt /usr/lib64 /usr/lib /usr/local/lib -name 'libjsoncpp.so*' -print -quit 2>/dev/null || true)
[ -n "$_jlib" ] && export LD_LIBRARY_PATH="$(dirname "$_jlib"):${LD_LIBRARY_PATH:-}"

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR="/scratch/users/nazarp/SALR3YUCUDA"
cd "$WORK_DIR"

# Create log directory if it does not exist
mkdir -p logs

echo "============================================================"
echo "SALR CPU Benchmark — AMD EPYC 9754 (node-003)"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Cores   : $SLURM_CPUS_PER_TASK"
echo "Started : $(date)"
echo "============================================================"

# ── Build ─────────────────────────────────────────────────────────────────────
echo ""
echo ">>> Building project (CPU only — no CUDA on EPYC node)"
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_DB_ENGINE=OFF \
    -Wno-dev \
    2>&1 | tail -20
cmake --build build --parallel "$SLURM_CPUS_PER_TASK" 2>&1 | tail -10
echo "Build complete."

# ── Install Python dependencies (user-level, cached after first run) ──────────
pip install --quiet --user numpy matplotlib 2>/dev/null || true

# ── Run benchmark ─────────────────────────────────────────────────────────────
echo ""
echo ">>> Running grid-size benchmark (CPU only, threads: 1 2 4 8 16 32 64)"
python3 scripts/benchmarking/run_benchmarks.py \
    --executable   ./build/salr_dft \
    --config       ./configs/benchmark.cfg \
    --output-dir   ./analysis/results \
    --grid-sizes   16 32 64 128 160 \
    --threads      1 2 4 8 16 32 64 \
    --iterations   1000 \
    --repeats      3 \
    --skip-cuda

echo ""
echo "============================================================"
echo "Benchmark finished: $(date)"
echo "Results in: analysis/results/"
echo "============================================================"
