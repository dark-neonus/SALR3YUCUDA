# scripts/

Role-based helper scripts for plotting, benchmarking, and maintenance.

## Layout

- `plotting/` contains heatmap and 3D visualisation helpers.
- `benchmarking/` contains CPU/CUDA timing and comparison scripts.
- `maintenance/` contains database integration utilities.

## Requirements

- Python 3 with `matplotlib` and `numpy` for the Python helpers
- gnuplot 5+ for the `.gp` visualisation helpers

## Plotting

```bash
python3 scripts/plotting/plot_joint_heatmap.py output/
python3 scripts/plotting/plot_density.py output/
python3 scripts/plotting/plot_density_3d.py output/
gnuplot scripts/plotting/density_browser.gp
gnuplot scripts/plotting/density_browser_merged.gp
```

## Benchmarking

```bash
python3 scripts/benchmarking/run_benchmarks.py --grid-sizes 32 64 128 256 --threads 1 2 4 8
python3 scripts/benchmarking/benchmark_cuda_cpu.py
python3 scripts/benchmarking/performance_real_params.py
```

## Maintenance

```bash
scripts/maintenance/rebuild_db_engine.sh
scripts/maintenance/verify_db_integration.sh
```
