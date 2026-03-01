# How to build

## Requirements

### CPU build
- **CMake ≥ 3.18** or **GNU Make**
- **C11 compiler** (gcc recommended)
- **OpenMP** support

### CUDA build (GPU)
- Everything above, plus:
- **CUDA Toolkit ≥ 11.0** (nvcc compiler)
- **GPU with compute capability ≥ 6.0** (Pascal or newer)

## Build with CMake

```bash
mkdir build && cd build
cmake ..
make
```

This builds:
- `salr_dft` — CPU version (OpenMP parallelised)
- `salr_dft_cuda` — GPU version (CUDA, built automatically if toolkit is found)
- `test_solver`, `test_potential` — CPU test programs

## Run

```bash
# CPU version
./build/salr_dft configs/default.cfg

# CUDA version (requires NVIDIA GPU)
./build/salr_dft_cuda configs/default.cfg

# Tests
./build/test_solver
./build/test_potential
```

## Clean rebuild

```bash
rm -rf build
mkdir build && cd build
cmake ..
make
```

## Notes

- The CUDA build produces the same output as the CPU build but runs
  significantly faster on the Φ convolution (O(N²) parallelised over N GPU threads).
- If no CUDA toolkit is found, CMake silently skips the GPU target.
- Set `CMAKE_CUDA_ARCHITECTURES` to target your specific GPU.
