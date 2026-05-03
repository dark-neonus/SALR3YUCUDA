#!/bin/bash
# =============================================================================
# job_benchmark_cuda_only.sh
# SLURM job — CUDA-only benchmark on GeForce RTX 5090 (gnode-003)
#
# Runs grid-size sweep with larger grids (up to 320x320) — CPU skipped.
# Longer GPU runtime makes it easy to observe with nvidia-smi.
#
# Submit:  sbatch scripts/cluster/job_benchmark_cuda_only.sh
# =============================================================================

#SBATCH --job-name=salr_cuda_only
#SBATCH --partition=gpu-gnode
#SBATCH --nodelist=gnode-003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_cuda_only_%j.out
#SBATCH --error=logs/slurm_cuda_only_%j.err

set -eu

# ── Environment ───────────────────────────────────────────────────────────────
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
echo "SALR CUDA-Only Benchmark — RTX 5090 (gnode-003)"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo "============================================================"
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
           --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
echo ">>> Building project (CUDA) into build_cuda/"
rm -rf build_cuda
cmake -S . -B build_cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_DB_ENGINE=OFF \
    -Wno-dev \
    2>&1 | tail -20
cmake --build build_cuda --parallel 4 2>&1 | tail -10
echo "Build complete."

pip install --quiet --user numpy matplotlib 2>/dev/null || true

# ── CUDA-only grid-size sweep (larger grids) ──────────────────────────────────
echo ""
echo ">>> Running CUDA-only benchmark (grids: 16 32 64 128 160 256 320)"
python3 scripts/benchmarking/run_benchmarks.py \
    --cuda-executable   ./build_cuda/salr_dft_cuda \
    --config            ./configs/benchmark.cfg \
    --output-dir        ./analysis/results \
    --grid-sizes        16 32 64 128 160 256 320 \
    --iterations        1000 \
    --repeats           3 \
    --skip-cpu

echo ""
echo "============================================================"
echo "CUDA benchmark finished: $(date)"
echo "Results in: analysis/results/"
echo "============================================================"
