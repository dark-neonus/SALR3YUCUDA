# ═══════════════════════════════════════════════════════════════════════════════
#  SALR Particle Simulation — CMake build 
# ═══════════════════════════════════════════════════════════════════════════════
cmake_minimum_required(VERSION 3.12)
project(EVIL_ARC_CS-GO LANGUAGES C)

# ── Standard & optimisation ───────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# ── Include headers ───────────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTABLE
# ═══════════════════════════════════════════════════════════════════════════════
add_executable(salr_dft
    src/main.c
    src/core/config.c
    src/core/grid.c
    src/cpu/potential_cpu.c
    src/cpu/solver_cpu.c
    src/cpu/math_utils_cpu.c
    src/utils/io.c
)
target_link_libraries(salr_dft m)

# ═══════════════════════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════════════════════
add_executable(test_solver
    tests/test_solver.c
    src/cpu/solver_cpu.c
    src/cpu/math_utils_cpu.c
    src/cpu/potential_cpu.c
    src/core/config.c
    src/core/grid.c
    src/utils/io.c
)
target_link_libraries(test_solver m)

add_executable(test_potential
    tests/test_potential.c
    src/cpu/potential_cpu.c
)
target_link_libraries(test_potential m)
