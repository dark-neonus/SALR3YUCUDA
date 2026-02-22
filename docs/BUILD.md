# How to build

## Requirements

- **CMake ≥ 3.12** or **GNU Make**
- **C11 compiler** (gcc recommended)
- No GPU or CUDA needed for current CPU-only version

## Build with CMake

```bash
mkdir build && cd build
cmake ..
make
```

## Build with Make

```bash
make          # builds everything
make salr_dft # main executable only
make tests    # test programs only
```

## Run

```bash
./build/salr_dft configs/default.cfg
./build/test_solver
./build/test_potential
```

## Clean rebuild

```bash
make clean
make
```

## Future: CUDA build

When CUDA kernels are added, the build system will be extended to
require CMake ≥ 3.18 and the CUDA Toolkit. The Makefile will use nvcc.
