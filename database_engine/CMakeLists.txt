# database_engine/CMakeLists.txt
# Build configuration for the SALR database engine library

cmake_minimum_required(VERSION 3.18)

# Find dependencies
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(SQLite3 REQUIRED)

# Print found libraries info
message(STATUS "HDF5 include dirs: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 libraries: ${HDF5_C_LIBRARIES}")
message(STATUS "SQLite3 include dirs: ${SQLite3_INCLUDE_DIRS}")
message(STATUS "SQLite3 libraries: ${SQLite3_LIBRARIES}")

# Source files
set(DB_ENGINE_SOURCES
    src/db_engine.c
    src/db_utils.c
    src/registry.c
    src/hdf5_io.c
)

# Static library for linking with main solver
add_library(salr_db_engine STATIC ${DB_ENGINE_SOURCES})

target_include_directories(salr_db_engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${SQLite3_INCLUDE_DIRS}
)

target_link_libraries(salr_db_engine
    PUBLIC
        ${HDF5_C_LIBRARIES}
        ${SQLite3_LIBRARIES}
)

# Shared library for Python bindings (optional)
option(BUILD_SHARED_DB_ENGINE "Build shared library for Python bindings" OFF)

if(BUILD_SHARED_DB_ENGINE)
    add_library(salr_db_shared SHARED ${DB_ENGINE_SOURCES})

    target_include_directories(salr_db_shared
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/include
        PRIVATE
            ${HDF5_INCLUDE_DIRS}
            ${SQLite3_INCLUDE_DIRS}
    )

    target_link_libraries(salr_db_shared
        PUBLIC
            ${HDF5_C_LIBRARIES}
            ${SQLite3_LIBRARIES}
    )

    set_target_properties(salr_db_shared PROPERTIES
        OUTPUT_NAME "salr_db"
        VERSION 1.0.0
        SOVERSION 1
    )
endif()

# Unit tests
option(BUILD_DB_ENGINE_TESTS "Build database engine tests" ON)

if(BUILD_DB_ENGINE_TESTS)
    # Test HDF5 operations
    add_executable(test_hdf5
        tests/test_hdf5.c
        src/hdf5_io.c
        src/db_utils.c
        ${CMAKE_SOURCE_DIR}/src/core/config.c
        ${CMAKE_SOURCE_DIR}/src/core/grid.c
    )
    target_include_directories(test_hdf5 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${HDF5_INCLUDE_DIRS}
    )
    target_link_libraries(test_hdf5 ${HDF5_C_LIBRARIES} m)

    # Test registry operations
    add_executable(test_registry
        tests/test_registry.c
        src/registry.c
        src/db_utils.c
        ${CMAKE_SOURCE_DIR}/src/core/config.c
        ${CMAKE_SOURCE_DIR}/src/core/grid.c
    )
    target_include_directories(test_registry PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${SQLite3_INCLUDE_DIRS}
    )
    target_link_libraries(test_registry ${SQLite3_LIBRARIES} m)

    # Integration test
    add_executable(test_db_integration
        tests/test_integration.c
    )
    target_include_directories(test_db_integration PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${HDF5_INCLUDE_DIRS}
    )
    target_link_libraries(test_db_integration salr_db_engine m)
endif()
