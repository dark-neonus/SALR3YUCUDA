# Build Instructions

## Requirements

### CPU build
- CMake >= 3.18
- C11 compiler (GCC recommended)
- OpenMP support

### CUDA build (planned)
- All CPU requirements, plus:
- CUDA Toolkit >= 11.0 (nvcc compiler)
- GPU with compute capability >= 6.0 (Pascal or newer)

## Build with CMake

```bash
mkdir build && cd build
cmake ..
make
```

This produces the following binaries:
- `salr_dft` -- CPU solver (OpenMP parallelised)
- `test_solver`, `test_potential` -- CPU test programs

When the CUDA toolkit is available and the GPU implementation is complete, CMake will additionally produce `salr_dft_cuda`.

## Run

```bash
./build/salr_dft configs/default.cfg
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

- If no CUDA toolkit is found, CMake silently skips the GPU target.
- Set `CMAKE_CUDA_ARCHITECTURES` to target a specific GPU architecture.
