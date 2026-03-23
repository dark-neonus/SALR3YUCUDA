# SALR DFT solver build (CPU + CUDA)
cmake_minimum_required(VERSION 3.18)
project(SALR_DFT LANGUAGES C CXX)

# Standard settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# OpenMP (CPU build)
find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

# CUDA (GPU build)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Auto-detect GPU architecture; fall back to common ones
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
    endif()
    set(HAVE_CUDA TRUE)
    message(STATUS "CUDA found — will build salr_dft_cuda")
else()
    set(HAVE_CUDA FALSE)
    message(STATUS "CUDA not found - building CPU-only targets")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# CPU executable
add_executable(salr_dft
    src/main.c
    src/core/config.c
    src/core/grid.c
    src/cpu/potential_cpu.c
    src/cpu/solver_cpu.c
    src/cpu/math_utils_cpu.c
    src/utils/io.c
)
target_link_libraries(salr_dft m OpenMP::OpenMP_C)

# CUDA DFT solver
if(HAVE_CUDA)
    add_executable(salr_dft_cuda
        src/main.c
        src/core/config.c
        src/core/grid.c
        src/cuda/solver_cuda.cu
        src/cuda/math_utils_cuda.cu
        src/cuda/potential_cuda.cu
        src/utils/io.c
    )
    set_target_properties(salr_dft_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CUDA
    )
    target_link_libraries(salr_dft_cuda m)
endif()

# Tests (CPU)
add_executable(test_solver
    tests/test_solver.c
    src/cpu/solver_cpu.c
    src/cpu/math_utils_cpu.c
    src/cpu/potential_cpu.c
    src/core/config.c
    src/core/grid.c
    src/utils/io.c
)
target_link_libraries(test_solver m OpenMP::OpenMP_C)

add_executable(test_potential
    tests/test_potential.c
    src/cpu/potential_cpu.c
)
target_link_libraries(test_potential m OpenMP::OpenMP_C)

# CPU integral benchmark
add_executable(benchmark_integral_cpu
    tests/benchmark_integral.c
    src/cpu/math_utils_cpu.c
)
target_link_libraries(benchmark_integral_cpu m OpenMP::OpenMP_C)

# CUDA integral benchmark
if(HAVE_CUDA)
    add_executable(benchmark_integral_cuda
        tests/benchmark_integral_cuda.cu
        src/cuda/math_utils_cuda.cu
    )
    set_target_properties(benchmark_integral_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CUDA
    )
    target_link_libraries(benchmark_integral_cuda m)
endif()

# =============================================================================
# Database Engine (HDF5 + SQLite support)
# =============================================================================

option(ENABLE_DB_ENGINE "Build with HDF5/SQLite database engine" ON)

if(ENABLE_DB_ENGINE)
    find_package(HDF5 COMPONENTS C)
    find_package(SQLite3)

    if(HDF5_FOUND AND SQLite3_FOUND)
        message(STATUS "Database engine enabled (HDF5 + SQLite found)")
        set(HAVE_DB_ENGINE TRUE)

        # Build the database engine library
        add_subdirectory(database_engine)

        # Create executables with database support
        add_executable(salr_dft_db
            src/main.c
            src/core/config.c
            src/core/grid.c
            src/cpu/potential_cpu.c
            src/cpu/solver_cpu.c
            src/cpu/math_utils_cpu.c
            src/utils/io.c
        )
        target_compile_definitions(salr_dft_db PRIVATE USE_DB_ENGINE)
        target_include_directories(salr_dft_db PRIVATE
            ${CMAKE_SOURCE_DIR}/database_engine/include
            ${HDF5_INCLUDE_DIRS}
        )
        target_link_libraries(salr_dft_db
            m OpenMP::OpenMP_C
            salr_db_engine
        )

        if(HAVE_CUDA)
            add_executable(salr_dft_cuda_db
                src/main.c
                src/core/config.c
                src/core/grid.c
                src/cuda/solver_cuda.cu
                src/cuda/math_utils_cuda.cu
                src/cuda/potential_cuda.cu
                src/utils/io.c
            )
            target_compile_definitions(salr_dft_cuda_db PRIVATE USE_DB_ENGINE)
            target_include_directories(salr_dft_cuda_db PRIVATE
                ${CMAKE_SOURCE_DIR}/database_engine/include
                ${HDF5_INCLUDE_DIRS}
            )
            set_target_properties(salr_dft_cuda_db PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                LINKER_LANGUAGE CUDA
            )
            target_link_libraries(salr_dft_cuda_db
                m
                salr_db_engine
            )
        endif()
    else()
        message(STATUS "Database engine disabled (HDF5 or SQLite not found)")
        if(NOT HDF5_FOUND)
            message(STATUS "  - HDF5 not found. Install with: sudo apt install libhdf5-dev")
        endif()
        if(NOT SQLite3_FOUND)
            message(STATUS "  - SQLite3 not found. Install with: sudo apt install libsqlite3-dev")
        endif()
        set(HAVE_DB_ENGINE FALSE)
    endif()
else()
    message(STATUS "Database engine disabled by user")
    set(HAVE_DB_ENGINE FALSE)
endif()
